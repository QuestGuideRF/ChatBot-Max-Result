import cv2
import torch
import torch.nn.functional as f
import torchvision.transforms as transforms
from backbone import Backbone
from PIL import Image
import os
import glob
from config import (
    camera_index,
    frame_width,
    frame_height,
    frame_skip,
    detection_frame_size,
    model_det_path,
    model_rec_path,
    data_root,
    cos_sim_threshold
)

from camera_notifier import notify_camera_blocked

class FaceDetector:
    def __init__(self, loop):
        self.loop = loop

        self.cap  = cv2.VideoCapture(camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        self.frame_skip = frame_skip
        self.frame_count = 0
        self.detection_frame_size = detection_frame_size

        self.last_detected_faces_with_similarities = []

        self.saved_faces_count = self.get_next_face_count()

        self.previous_unknown_faces_count = 0
        self.unknown_faces_current_count = 0
        self.unknown_faces_appear_time = None

        from config import foreign_faces_path
        self.foreignfaces_path = foreign_faces_path

        import time
        self.last_check_time = time.time()
        self.check_interval = 10

        self.camera_blocked = False

        self.detector = cv2.FaceDetectorYN.create(
            model=model_det_path,
            config="",
            input_size=detection_frame_size,
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=10
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.backbone = Backbone([112, 112])
        self.backbone.load_state_dict(torch.load(model_rec_path, map_location=torch.device("cpu")))
        self.backbone.to(self.device)
        self.backbone.eval()

        self.reference_embeddings = []
        reference_files = glob.glob(os.path.join(data_root, "*.jpg"))

        processed_files = set()
        for ref_file in reference_files:
            try:

                filename = os.path.basename(ref_file)

                if filename not in processed_files:
                    pil_face = self.get_ref_face(ref_file)
                    embedding = self.get_embedding(pil_face, backbone=self.backbone, device=self.device)
                    self.reference_embeddings.append(embedding)
                    processed_files.add(filename)
            except ValueError as e:
                print(f"Ошибка при обработке файла {ref_file}: {e}")
                continue

        if self.reference_embeddings:
            self.reference_embeddings = torch.stack(self.reference_embeddings)
            print(f"Загружено эталонных эмбеддингов при инициализации: {len(self.reference_embeddings)}")
        else:
            print("Не найдено эталонных изображений в директории data_root")
            self.reference_embeddings = torch.empty(0, 512)

        if not os.path.exists(data_root):
            os.makedirs(data_root)

    def is_camera_blocked(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        std_dev = cv2.meanStdDev(gray)[1][0][0]

        if std_dev < 25:
            return True

        mean_brightness = cv2.mean(gray)[0]
        if mean_brightness < 20:
            return True

        return False

    def run(self):

        while True:

            ret, frame = self.cap.read()

            if not ret:
                print("Не удалось получить кадр с камеры")
                break

            is_blocked = self.is_camera_blocked(frame)
            if is_blocked and not self.camera_blocked:
                print("УВЕДОМЛЕНИЕ: Камера загорожена!")
                notify_camera_blocked()
                self.camera_blocked = True
            elif not is_blocked and self.camera_blocked:

                self.camera_blocked = False

            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:

                faces = self.detect_faces(frame)
                face_similarities = self.calculate_face_similarities(frame, faces)
                self.draw_faces(frame, face_similarities)

                self.last_detected_faces_with_similarities = []
                for item in face_similarities:
                    self.last_detected_faces_with_similarities.append(item)

                unknown_faces_count = sum(1 for item in face_similarities if item['similarity'] < cos_sim_threshold)

                import time

                if unknown_faces_count != self.previous_unknown_faces_count:

                    if unknown_faces_count > self.previous_unknown_faces_count:
                        self.unknown_faces_appear_time = time.time()
                    self.previous_unknown_faces_count = unknown_faces_count
                elif unknown_faces_count > 0 and self.unknown_faces_appear_time is not None:

                    elapsed_time = time.time() - self.unknown_faces_appear_time

                    if elapsed_time >= 0.5:
                        self.save_unknown_faces_screenshot(frame)

                        self.unknown_faces_appear_time = None

                self.frame_count = 0
            else:

                self.draw_last_faces(frame)

            self.check_and_update_reference_embeddings()

            cv2.imshow('Face Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and faces:

                for face in faces:
                    self.save_face(frame, face)
            elif key == ord('q') or key == 27:
                break

            if cv2.getWindowProperty('Face Detection', cv2.WND_PROP_VISIBLE) < 1:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def detect_faces(self, frame):

        resized_frame = cv2.resize(frame, self.detection_frame_size)

        faces = self.detector.detect(resized_frame)

        detected_faces = []
        if faces[1] is not None:
            for detection in faces[1]:

                x, y, w, h = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])

                x = int(x * frame.shape[1] / self.detection_frame_size[0])
                y = int(y * frame.shape[0] / self.detection_frame_size[1])
                w = int(w * frame.shape[1] / self.detection_frame_size[0])
                h = int(h * frame.shape[0] / self.detection_frame_size[1])

                face_rect = type('FaceRect', (), {
                    'left': lambda self, x=x: x,
                    'top': lambda self, y=y: y,
                    'right': lambda self, x=x, w=w: x + w,
                    'bottom': lambda self, y=y, h=h: y + h,
                    'width': lambda self, w=w: w,
                    'height': lambda self, h=h: h
                })()

                detected_faces.append(face_rect)

        return detected_faces

    def calculate_face_similarities(self, frame, faces):

        if self.camera_blocked:

            return []

        result = []
        recognized_count = 0
        unrecognized_count = 0

        for face in faces:
            cosine_sim = self.recognize_face_with_similarity(frame, face)
            result.append({'face': face, 'similarity': cosine_sim})

            if cosine_sim >= cos_sim_threshold:
                recognized_count += 1
            else:
                unrecognized_count += 1

        print(f"Распознанные лица: {recognized_count}, Нераспознанные лица: {unrecognized_count}")

        return result

    def draw_faces(self, frame, face_similarities):

        for item in face_similarities:
            face = item['face']
            cosine_sim = item['similarity']

            x = face.left()
            y = face.top()
            w = face.width()
            h = face.height()

            is_match = cosine_sim >= cos_sim_threshold

            color = (0, 255, 0) if is_match else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            similarity_text = f"Sim: {cosine_sim:.2f}"
            cv2.putText(frame, similarity_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, "Press 's' to save face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_last_faces(self, frame):

        self.draw_faces(frame, self.last_detected_faces_with_similarities)

    def recognize_face_with_similarity(self, frame, face):

        x = face.left()
        y = face.top()
        x2 = x + face.width()
        y2 = y + face.height()

        face_image = frame[y:y2, x:x2]

        if face_image.size == 0:
            return 0.0

        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_image_rgb)

        embedding = self.get_embedding(face_pil, 512, [112, 112], backbone=self.backbone, device=self.device)

        if self.reference_embeddings.numel() == 0:
            return 0.0

        cosine_sims = torch.matmul(self.reference_embeddings, embedding.unsqueeze(1)).squeeze(1).clip(min=0, max=1)

        max_cosine_sim = torch.max(cosine_sims).item()

        return max_cosine_sim

    def get_embedding(self, pil_image, embedding_size=512, input_size=[112, 112], backbone=None, device=None):
        """Получение эмбеддинга из изображения лица"""

        transform = transforms.Compose(
            [
                transforms.Resize(
                    [int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)],
                ),
                transforms.CenterCrop([input_size[0], input_size[1]]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
        )

        image_tensor = transform(pil_image)

        image_tensor = image_tensor.unsqueeze(0)

        if backbone is None or device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            backbone = Backbone(input_size)
            backbone.load_state_dict(torch.load(model_rec_path, map_location=torch.device("cpu")))
            backbone.to(device)
            backbone.eval()

        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            embedding = torch.nn.functional.normalize(backbone(image_tensor)).cpu()

        embedding = torch.squeeze(embedding)

        embedding = embedding / torch.norm(embedding)

        return embedding

    def get_ref_face(self, file_name):

        image_pil = Image.open(file_name).convert('RGB')
        image_cv = cv2.imread(file_name)

        resized_frame = cv2.resize(image_cv, self.detection_frame_size)

        reference_faces = self.detector.detect(resized_frame)
        if reference_faces[1] is not None and len(reference_faces[1]) > 0:

            face_data = reference_faces[1][0]
            x, y, w, h = int(face_data[0]), int(face_data[1]), int(face_data[2]), int(face_data[3])

            x = int(x * image_cv.shape[1] / self.detection_frame_size[0])
            y = int(y * image_cv.shape[0] / self.detection_frame_size[1])
            w = int(w * image_cv.shape[1] / self.detection_frame_size[0])
            h = int(h * image_cv.shape[0] / self.detection_frame_size[1])

            pil_face = image_pil.crop((x, y, x + w, y + h))

            return pil_face

        else:
            raise ValueError("Не удалось обнаружить лицо на эталонном изображении")

    def save_face(self, frame, face):

        if not os.path.exists(data_root):
            os.makedirs(data_root)

        x = face.left()
        y = face.top()
        w = face.width()
        h = face.height()

        face_image = frame[y:y+h, x:x+w]

        if face_image.size == 0:
            print("Не удалось вырезать изображение лица")
            return False

        self.saved_faces_count += 1
        filename = os.path.join(data_root, f"saved_face_{self.saved_faces_count}.jpg")

        success = cv2.imwrite(filename, face_image)
        if success:
            print(f"Изображение лица сохранено: {filename}")
        else:
            print(f"Ошибка при сохранении изображения лица: {filename}")

        return success

    def save_unknown_faces_screenshot(self, frame):
        """Сохраняет скриншот с нераспознанными лицами в папку foreign"""
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.foreignfaces_path, f"unknown_faces_{timestamp}.jpg")

        success = cv2.imwrite(filename, frame)
        if success:
            print(f"Скриншот с нераспознанными лицами сохранен: {filename}")
            return filename
        else:
            print(f"Ошибка при сохранении скриншота: {filename}")
            return None

    def get_next_face_count(self):

        if not os.path.exists(data_root):
            return 0

        saved_faces = glob.glob(os.path.join(data_root, "saved_face_*.jpg"))

        max_count = 0
        for face_file in saved_faces:
            filename = os.path.basename(face_file)
            try:

                count = int(filename.replace("saved_face_", "").replace(".jpg", ""))
                max_count = max(max_count, count)
            except ValueError:

                continue
        return max_count

    def check_and_update_reference_embeddings(self):
        """Проверяет наличие новых файлов в папке data и обновляет эталонные эмбеддинги"""
        import time

        current_time = time.time()

        if current_time - self.last_check_time < self.check_interval:
            return False

        self.last_check_time = current_time

        reference_files = glob.glob(os.path.join(data_root, "*.jpg"))

        current_filenames = set(os.path.basename(f) for f in reference_files)

        loaded_embedding_count = self.reference_embeddings.size(0) if hasattr(self.reference_embeddings, 'size') else len(self.reference_embeddings)

        if len(current_filenames) != loaded_embedding_count:
            print(f"Обнаружено изменение в папке data: {len(current_filenames)} файлов (было {loaded_embedding_count})")

            all_embeddings = []
            processed_files = set()

            for ref_file in reference_files:
                try:

                    filename = os.path.basename(ref_file)

                    if filename not in processed_files:
                        pil_face = self.get_ref_face(ref_file)
                        embedding = self.get_embedding(pil_face, backbone=self.backbone, device=self.device)
                        all_embeddings.append(embedding)
                        processed_files.add(filename)

                except ValueError as e:
                    print(f"Ошибка при обработке файла {ref_file}: {e}")
                    continue

            if all_embeddings:
                self.reference_embeddings = torch.stack(all_embeddings)
                print(f"Обновлено эталонных эмбеддингов: {len(all_embeddings)}")
                return True
            else:
                print("Не удалось обработать ни один файл")

        return False