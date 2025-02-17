import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
from skimage.io import imread
from skimage import transform
from tqdm import tqdm
import glob2
import json
import time
import logging
from PIL import Image

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Giả sử bạn đã có module face_detector.py cung cấp class YoloV5FaceDetector
from face_detector import YoloV5FaceDetector

# ==========================
# Định nghĩa class Detect
# ==========================
class Detect:
    def __init__(self, model_file, known_user_path=None):
        """
        Khởi tạo detector với mô hình và dữ liệu người dùng
        
        :param model_file: Đường dẫn tới file mô hình face embedding
        :param known_user_path: Đường dẫn thư mục chứa ảnh các người dùng đã biết
        """
        try:
            # Khởi tạo detector khuôn mặt
            self.det = YoloV5FaceDetector()
            
            # Tải mô hình embedding
            self.face_model = tf.keras.models.load_model(model_file, compile=False) if model_file else None
            
            # Preload embeddings của người dùng đã biết
            if known_user_path:
                self.image_classes, self.embeddings, _ = self.embed_images(known_user_path)
                logger.info(f"Loaded {len(self.image_classes)} known user embeddings")
            else:
                self.image_classes, self.embeddings = None, None
        
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def preprocess_image(self, img, target_size=(112, 112)):
        """
        Tiền xử lý ảnh để cải thiện độ chính xác
        
        :param img: Ảnh đầu vào
        :param target_size: Kích thước ảnh mục tiêu
        :return: Ảnh đã được chuẩn hóa
        """
        try:
            # Chuyển sang không gian màu RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Chuẩn hóa độ sáng và tương phản
            img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            # Làm mờ để giảm nhiễu
            img = cv2.GaussianBlur(img, (3, 3), 0)
            # Resize và chuẩn hóa
            img = cv2.resize(img, target_size)
            img = img.astype(np.float32) / 127.5 - 1.0
            return img
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            raise

    def face_align_landmarks(self, img, landmarks, image_size=(112, 112), method="similar"):
        """
        Căn chỉnh khuôn mặt dựa trên landmark
        
        :param img: Ảnh đầu vào
        :param landmarks: Các điểm landmark của khuôn mặt
        :param image_size: Kích thước ảnh mục tiêu
        :param method: Phương pháp transform
        :return: Ảnh khuôn mặt đã được căn chỉnh
        """
        tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()
        src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], 
                        [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32)
        
        aligned_faces = []
        for landmark in landmarks:
            tform.estimate(landmark, src)
            aligned_faces.append(transform.warp(img, tform.inverse, output_shape=image_size))
        
        return (np.array(aligned_faces) * 255).astype(np.uint8)

    def detect_faces(self, image, image_format="BGR"):
        """
        Phát hiện khuôn mặt trong ảnh
        
        :param image: Ảnh đầu vào
        :param image_format: Định dạng màu của ảnh
        :return: Bounding boxes, confidence scores, ảnh khuôn mặt đã căn chỉnh
        """
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        imm_BGR = image if image_format == "BGR" else image[:, :, ::-1]
        imm_RGB = image[:, :, ::-1] if image_format == "BGR" else image
        
        bboxes, pps, ccs = self.det.__call__(imm_BGR)
        nimgs = self.face_align_landmarks(imm_RGB, pps)
        bbs, ccs = bboxes[:, :4].astype("int"), bboxes[:, -1]
        return bbs, ccs, nimgs

    def embed_images(self, known_user, batch_size=2, force_reload=False):
        """
        Tạo embedding cho tập ảnh người dùng đã biết
        
        :param known_user: Đường dẫn thư mục chứa ảnh
        :param batch_size: Kích thước batch
        :param force_reload: Bắt buộc tải lại embedding
        :return: Nhãn và embedding của ảnh
        """
        known_user = known_user.rstrip('/')
        dest_pickle = os.path.join(known_user, os.path.basename(known_user) + "_embedding.npz")

        if not force_reload and os.path.exists(dest_pickle):
            data = np.load(dest_pickle)
            return data["image_classes"], data["embeddings"], dest_pickle

        image_names = glob2.glob(os.path.join(known_user, "*/*.png"))
        nimgs, image_classes = [], []
        
        for image_name in tqdm(image_names, "Detecting faces"):
            img = imread(image_name)
            nimg = self.detect_faces(img, image_format="RGB")[-1]
            
            if nimg.shape[0] > 0:
                # Tiền xử lý ảnh trước khi embedding
                preprocessed_img = self.preprocess_image(nimg[0])
                nimgs.append(preprocessed_img)
                image_classes.append(os.path.basename(os.path.dirname(image_name)))

        # Nếu không tìm thấy ảnh nào hợp lệ
        if len(nimgs) == 0:
            raise ValueError("Không có ảnh nào được phát hiện khuôn mặt trong folder known_user")

        # Tạo embedding theo batch
        nimgs = np.array(nimgs)
        embeddings = [
            self.face_model(nimgs[ii * batch_size : (ii + 1) * batch_size]) 
            for ii in tqdm(range(len(image_classes) // batch_size), "Embedding")
        ]
        
        # Chuẩn hóa embedding
        embeddings = normalize(np.concatenate(embeddings, axis=0))
        
        # Lưu embedding
        np.savez_compressed(dest_pickle, 
                            embeddings=embeddings, 
                            image_classes=np.array(image_classes))
        
        return image_classes, embeddings, dest_pickle

    def recognize_faces(self, image, dist_thresh=0.45):
        """
        Nhận diện khuôn mặt trong ảnh
        
        :param image: Ảnh đầu vào
        :param dist_thresh: Ngưỡng khoảng cách để phân loại
        :return: Khoảng cách, nhãn, bounding boxes, confidence scores
        """
        if self.embeddings is None or self.image_classes is None:
            raise ValueError("No known user embeddings loaded")

        bbs, ccs, nimgs = self.detect_faces(image, image_format="RGB")
        
        if len(bbs) == 0:
            return None

        # Tiền xử lý ảnh trước khi tạo embedding
        processed_nimgs = np.array([self.preprocess_image(img) for img in nimgs])
        
        # Tạo embedding cho khuôn mặt chưa biết
        emb_unk = self.face_model(processed_nimgs).numpy()
        emb_unk = normalize(emb_unk)
        
        # So sánh embedding
        dists = np.dot(self.embeddings, emb_unk.T).T
        rec_idx = dists.argmax(-1)
        rec_dist = [dists[i, rec_idx[i]] for i in range(len(rec_idx))]
        
        # Gán nhãn "Unknown" nếu khoảng cách thấp hơn ngưỡng
        rec_class = []
        for dist, idx in zip(rec_dist, rec_idx):
            rec_class.append("Unknown" if dist < dist_thresh else self.image_classes[idx])
        return rec_dist, rec_class, bbs, ccs

    def draw_polyboxes(self, frame, rec_dist, rec_class, bbs, ccs, dist_thresh):
        """
        Vẽ bounding boxes và nhãn lên ảnh
        
        :param frame: Ảnh gốc
        :param rec_dist: Khoảng cách nhận diện
        :param rec_class: Nhãn nhận diện
        :param bbs: Bounding boxes
        :param ccs: Confidence scores
        :param dist_thresh: Ngưỡng khoảng cách
        :return: Ảnh có vẽ bounding boxes
        """
        for dist, label, bb, cc in zip(rec_dist, rec_class, bbs, ccs):
            color = (0, 0, 255) if dist < dist_thresh else (0, 255, 0)
            label = "Unknown" if dist < dist_thresh else label
            left, up, right, down = bb
            cv2.rectangle(frame, (left, up), (right, down), color, 2)
            cv2.putText(frame, f"Label: {label}, dist: {dist:.4f}", 
                        (left, up - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# ==========================
# Hàm nhận diện folder ảnh
# ==========================
def recognize_folder_images(detector, folder_path, output_json_path="results_folder.json", dist_thresh=0.45):
    """
    Duyệt qua tất cả ảnh trong folder và thực hiện nhận diện khuôn mặt.
    Kết quả của mỗi ảnh (bao gồm thời gian xử lý, kết quả nhận diện) sẽ được lưu lại vào một file JSON.
    
    :param detector: Đối tượng Detect đã được khởi tạo
    :param folder_path: Đường dẫn folder chứa ảnh cần nhận diện
    :param output_json_path: Đường dẫn file JSON lưu kết quả
    :param dist_thresh: Ngưỡng khoảng cách để phân loại
    """
    # Lấy danh sách các file ảnh (có đuôi jpg, jpeg, png)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [os.path.join(root, file)
                   for root, dirs, files in os.walk(folder_path)
                   for file in files if file.lower().endswith(valid_extensions)]
    
    logger.info(f"Tìm thấy {len(image_files)} ảnh trong folder {folder_path}")

    results = {}

    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Đọc ảnh bằng PIL và chuyển sang numpy array (RGB)
            img = Image.open(image_file).convert("RGB")
            img_np = np.array(img)
            
            # Đo thời gian nhận diện cho ảnh
            start_time = time.time()
            result = detector.recognize_faces(img_np, dist_thresh=dist_thresh)
            end_time = time.time()
            recognition_time = end_time - start_time

            if result:
                rec_dist, rec_class, bbs, ccs = result
                result_data = {
                    "recognized_labels": rec_class,
                    "confidence": [float(dist) for dist in rec_dist],
                    "bounding_boxes": bbs.tolist(),  # chuyển sang list nếu cần
                    "confidence_scores": ccs.tolist() if hasattr(ccs, "tolist") else ccs,
                    "processing_time": recognition_time
                }
            else:
                result_data = {
                    "message": "No face detected",
                    "processing_time": recognition_time
                }
            
            results[os.path.basename(image_file)] = result_data
        
        except Exception as e:
            logger.error(f"Lỗi xử lý ảnh {image_file}: {e}")
            results[os.path.basename(image_file)] = {"error": str(e)}

    # Lưu kết quả vào file JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info(f"Kết quả nhận diện đã được lưu vào file {output_json_path}")

# ==========================
# Main: Khởi tạo detector và chạy nhận diện folder
# ==========================
if __name__ == "__main__":
    # Chỉnh sửa đường dẫn dưới đây cho phù hợp với hệ thống của bạn:
    MODEL_FILE = "/app/GhostFaceNets/checkpoints/GhostFaceNet_W1.3_S1_ArcFace.h5"
    MODEL_FILE2= "/"
    KNOWN_USER_PATH = "data/test/complete_images/train"  # Folder chứa ảnh đã biết
    TEST_FOLDER = "data/test/folder_images"             # Folder chứa ảnh cần nhận diện
    OUTPUT_JSON = "results_folder.json"

    try:
        # Khởi tạo detector
        detector = Detect(model_file=MODEL_FILE, known_user_path=KNOWN_USER_PATH)
        # Thực hiện nhận diện trên folder ảnh
        recognize_folder_images(detector, TEST_FOLDER, output_json_path=OUTPUT_JSON, dist_thresh=0.45)
    except Exception as e:
        logger.error(f"Error in main: {e}")
