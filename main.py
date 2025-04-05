import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
from skimage.io import imread
from skimage import transform
from tqdm import tqdm
import time
import glob2
from face_detector import YoloV5FaceDetector
from fastapi import FastAPI, HTTPException, UploadFile, File, Query 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from PIL import Image
import io
import logging
import ssl
import unicodedata
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from voice2text import WhisperTranscriber
from typing import Optional
import time 
os.environ['HF_HOME']= '/app/GhostFaceNets/cache'

# Cấu hình logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
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
        :return: Bounding boxes, confidence scores, ảnh khuôn mặt
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

        # Chuẩn hóa ảnh
        nimgs = (np.array(nimgs) - 127.5) * 0.0078125
        
        # Tạo embedding theo batch
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
        :return: Khoảng cách, nhãn, bounding boxes
        """
        if self.embeddings is None or self.image_classes is None:
            raise ValueError("No known user embeddings loaded")

        bbs, ccs, nimgs = self.detect_faces(image, image_format="RGB")
        
        if len(bbs) == 0:
            return None

        # Tiền xử lý ảnh trước khi tạo embedding
        processed_nimgs = np.array([self.preprocess_image(img) for img in nimgs])
        
        # Tạo embedding
        emb_unk = self.face_model(processed_nimgs).numpy()
        emb_unk = normalize(emb_unk)
        
        # So sánh embedding
        dists = np.dot(self.embeddings, emb_unk.T).T
        rec_idx = dists.argmax(-1)
        rec_dist = [dists[id, ii] for id, ii in enumerate(rec_idx)]
        
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

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image: str

class StringData(BaseModel):
    message : str

def remove_accents(text):
    # Normalize the text to decompose accented characters
    normalized = unicodedata.normalize('NFD', text)
    # Filter out combining diacritical marks
    return ''.join(c for c in normalized if not unicodedata.combining(c))

# Khởi tạo detector với việc preload embeddings
detector = Detect(
    model_file="checkpoints/GhostFaceNet_W1.3_S1_ArcFace.h5", 
    known_user_path="data/distorted_faces_112_112/train"
)

UPLOAD_DIR = "/app/GhostFaceNets/testtt"  # Folder to store files
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure the folder exists

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
transcriber = WhisperTranscriber()

@app.post("/api_chatbot")
async def upload_audio(file: UploadFile = File(...)):
    # Read the audio file content from the uploaded file
    audio_data = await file.read()
    if len(audio_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size allowed: {MAX_FILE_SIZE/1024/1024}MB"
            )
    
    original_file_path = os.path.join(UPLOAD_DIR, 'testsomethinghere.temp')

    os.makedirs(os.path.dirname(original_file_path), exist_ok=True)
    with open(original_file_path, "wb") as temp_file:
        temp_file.write(audio_data)

    wav_file_path = os.path.join(UPLOAD_DIR, "audio.wav")
    # Convert to .wav format
    audio = AudioSegment.from_file(original_file_path)
    audio.export(wav_file_path, format="wav")
    # Transcribe audio với hoặc không có language specified
    result = transcriber.transcribe(
        wav_file_path,
        language=None, 
        return_timestamps=True
    )
    print(result["text"])   
    return {"message": result["text"]}
@app.post("/detect")
async def detect_face(data: ImageData):
    """
    Endpoint nhận diện khuôn mặt
    """
    # start_time = time.time()  # Bắt đầu đo thời gian
    print('get request!!!!')
    try:
        # Giải mã ảnh
        image_data = base64.b64decode(data.image)
        img = Image.open(io.BytesIO(image_data))
        img_np = np.array(img.convert("RGB"))

        # Nhận diện khuôn mặt
        print('detecting face')
        result = detector.recognize_faces(img_np, dist_thresh=0.55)
        rec_dist, rec_class, bbs, ccs = result
        print(f"Result: {rec_dist}")
        
        if result:
            rec_dist, rec_class, bbs, ccs = result
            result_image = detector.draw_polyboxes(img_np, rec_dist, rec_class, bbs, ccs, dist_thresh=0.6)
            end = time.time()  # Bắt đầu đo thời gian
            # print(end - start_time)
            return {
                "result_image": rec_class,
                "confidence": [float(dist) for dist in rec_dist]
            }
        else:
            end = time.time()  # Bắt đầu đo thời gian
            print(end - start_time)
            return {"result_image": "Xin chào"}
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the face detection API"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080, ssl_keyfile="/ohmni/privatekey.pem", ssl_certfile="/ohmni/certificate.pem")
