import cv2
import wget
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../Moore-AnimateAnyone")
from src.utils.util import get_fps, read_frames, save_videos_from_pil
import os
 
def download_url(urls, save_dir):
    """
    Скачиваем видео по ссылке

    Args:
        urls (list[str]): список ссылок
        save_dir (str): путь сохранения
    """
    for url in tqdm(urls):
        wget.download(url, out=save_dir)

def get_params_mp4(path, log=False):    
    """
    Получаем параметры видео

    Args:
        path (str): путь к видео
        log (bool): отображение параметров текстом
    Returns:
        width (int): ширина
        height (int): высота
        frame_count (int): общее  кол-во кадров
        fps (int): кадров в секунду
    """
    cap = cv2.VideoCapture(path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if log:
        print(f"Ширина кадров: {width}")
        print(f"Высота кадров: {height}")
        print(f"Количество кадров: {frame_count}")
        print(f"Кадров в секунду: {fps}")
    cap.release()
    return width, height, frame_count, fps

def show_video_frames(video_path, frame_numbers):
    """
    Загружает указанные кадры из видео и отображает их в виде горизонтальной ленты
    
    Args:
        video_path (str): путь к видеофайлу
        frame_numbers (list): список номеров кадров (начиная с 0)
    """

    cap = cv2.VideoCapture(video_path)    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_num in frame_numbers:
        if frame_num >= total_frames:
            raise ValueError(f"Кадр {frame_num} превышает общее количество кадров ({total_frames})")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    combined = np.hstack(frames)
    
    plt.figure(figsize=(15, 5))
    plt.imshow(combined)
    plt.axis('off')
    plt.show()

def generate_triplets(vid_path_list, save_dir_path, detector, sample_margin=70, triplet_per_vid=1, max_triplet=200):
    """
    Генерируем выборку для первой стадии модели

    Args:
        vid_path_list (list[str]): список путей к видео 
        save_dir_path (str): путь сохранения семплов
        detector (DWposeDetector): предобученный детектор позы
        sample_margin (int): минимальное расстояние от изначальго кадра к таргентному
        triplet_per_vid (int): кол-во триплетов для одного видел
        max_triplet (int): максимальное кол-во триплетов
    """

    for j, path in tqdm(enumerate(vid_path_list), total=len(vid_path_list)):
        for i in range(triplet_per_vid):
            n_triplet = (j) * triplet_per_vid + i
            if n_triplet < max_triplet:
                width, height, frame_count, fps = get_params_mp4(path, log=False)
        
                sample_path = save_dir_path + f"sample_{n_triplet}"
                os.makedirs(sample_path, exist_ok=True)
                
                ref_img_idx = np.random.randint(0, frame_count - sample_margin - 1)
                tgt_img_idx = np.random.randint(ref_img_idx + sample_margin, frame_count - 1)
                
                frames = read_frames(path)
                
                source_pil = frames[ref_img_idx]
                target_pil = frames[tgt_img_idx]
                target_pose, score = detector(target_pil)
                
                source_pil.save(sample_path + "/source.png","PNG")
                target_pil.save(sample_path + "/target.png","PNG")
                target_pose.save(sample_path + "/target_pose.png","PNG")
            else:
                break

def save_pose_from_mp4(file_path, out_path, detector):
    """
    Извлекает позу по видео

    Args:
        file_path (str): путь к исходному видео 
        out_path (str): путь для сохранения позы видео
        detector (DWposeDetector): предобученная модель
    """
    fps = get_fps(file_path)
    frames = read_frames(file_path)
    kps_results = []
    for i, frame_pil in enumerate(frames):
        result, score = detector(frame_pil)
        score = np.mean(score, axis=-1)
        kps_results.append(result)
    
    save_videos_from_pil(kps_results, out_path, fps=fps)

def reduce_frames(input_video_path, output_video_path, step=3):
    """
    Уменьшает количество кадров в видео, сохраняя только каждый N-й кадр.
    
    Args:
        input_video_path (str): Путь к исходному видео.
        output_video_path (str): Путь для сохранения результата.
        step (int): Шаг пропуска кадров (1 = без пропуска, 2 = каждый второй, 3 = каждый третий и т. д.).
    """
    cap = cv2.VideoCapture(input_video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps / step, (width, height))
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            out.write(frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    out.release()

def generate_vid_triplets(vid_path_list, save_dir_path, detector, n_examples=10, clip_length=100):
    """
    Генерирует видео для тестировния. На выходе сохраняем:
    * входной кадр
    * видео с позой
    * оригинальное видео

    Args:
        vid_path_list (list[str]): список путей до оригинальных видео 
        save_dir_path (str): путь для сохранения
        detector (DWposeDetector): предобученная модель
        n_examples (int): кол-во примеров для генерации
        clip_length (int): длина одной генерации
    """
    
    n_generated = 0
     
    for i, path in enumerate(vid_path_list):
        width, height, frame_count, fps = get_params_mp4(path, log=False)
        fps = get_fps(path)
        frames = read_frames(path)
        
        if (clip_length < frame_count) and (n_generated < n_examples):
            sample_path = save_dir_path + f"sample_{i}"
            os.makedirs(sample_path, exist_ok=True)
            orig_vid = []
            pose_vid = []

        
            start_idx = np.random.randint(0, frame_count - clip_length - 1)
            ref_img_idx = np.random.randint(0, frame_count - 1)
            ref_image = frames[ref_img_idx]

            for j in tqdm(range(clip_length), desc=f'generation: {n_generated}'):
                frame_pil = frames[start_idx+j]
                target_pose, score = detector(frame_pil)

                orig_vid.append(frame_pil)
                pose_vid.append(target_pose)

            ref_image.save(sample_path + "/ref_image.png","PNG")

            save_videos_from_pil(orig_vid, sample_path + "/orig_vid.mp4", fps=fps)
            save_videos_from_pil(pose_vid, sample_path + "/pose_vid.mp4", fps=fps)
            
            n_generated += 1
        else:
            break