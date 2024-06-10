import cv2
import numpy as np
from skimage.feature import hog
from skimage import color
import os

# Hàm tính toán HOG
def compute_hog(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features, _ = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return hog_features

# Hàm so sánh HOG của hai frame liên tiếp
def compare_hog(hog1, hog2, threshold=0.3):
    distance = np.linalg.norm(hog1 - hog2)
    return distance < threshold

# Hàm xử lý video và lưu frame tương đồng
def process_video(video_path, output_folder, frame_skip=3, threshold=0.3):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    saved_frame_index = 0

    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error reading video: {video_path}")
        return

    prev_hog = compute_hog(prev_frame)

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        if frame_index % frame_skip == 0:
            current_hog = compute_hog(current_frame)
            if compare_hog(prev_hog, current_hog, threshold):
                frame_filename = f"{output_folder}/{video_name}_frame_{saved_frame_index:04d}.jpg"
                cv2.imwrite(frame_filename, current_frame)
                saved_frame_index += 1
                # Bỏ qua frame tiếp theo bằng cách đọc nó nhưng không xử lý
                for _ in range(frame_skip - 1):
                    ret, _ = cap.read()
                if not ret:
                    break
                prev_hog = compute_hog(current_frame)
            else:
                prev_hog = current_hog

        frame_index += 1

    cap.release()

# Ví dụ: xử lý và lưu frame tương đồng từ một video
video_path = r'D:/Wed_basic/Python/He_CSDL_DPT_Nhom_1/He_CSDL_DPT/process/video/video.mp4'
output_folder = r'D:/Wed_basic/Python/He_CSDL_DPT_Nhom_1/He_CSDL_DPT/process/imageOfVideo'

# Đảm bảo thư mục đầu ra tồn tại
os.makedirs(output_folder, exist_ok=True)

process_video(video_path, output_folder, frame_skip=5, threshold=0.3)
