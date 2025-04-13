import wget
import cv2


def download_url(urls, save_dir):
    for url in tqdm(urls):
        wget.download(url, out=save_dir)

def save_pose_from_mp4(file_path, out_path):
    fps = get_fps(file_path)
    frames = read_frames(file_path)
    kps_results = []
    for i, frame_pil in enumerate(frames):
        result, score = detector(frame_pil)
        score = np.mean(score, axis=-1)
        kps_results.append(result)
        
    save_videos_from_pil(kps_results, out_path, fps=fps)

def save_first_frame(file_path, out_path):
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    cv2.imwrite(out_path, frame)
    cap.release()