import cv2
import wget
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../Moore-AnimateAnyone")
from src.utils.util import read_frames

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

def change_params_mp4(original_path, new_path, fps=30, height=768, width=576, num_frames=90):
    """
    Меняем параметры видео

    Args:
        original_path (str): путь к изначальномсу видео
        new_path (str): путь сохранения нового
        fps (int): новое кол-во кадров в секунду
        height (int): высота кадра
        width (int): ширина кадра
        num_frames (int): общее число кадров
    """
    cap = cv2.VideoCapture(original_path)    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(new_path, fourcc, fps, (width, height))
    
    for frame_id in range(num_frames):
        ret, frame = cap.read()
        resized_frame = cv2.resize(frame, (width, height))
        if not ret:
            break 
        out.write(resized_frame)  
    
    cap.release()
    out.release() 