from torch.nn import CosineSimilarity
import torch
import numpy as np
from src.utils.util import read_frames
from tqdm import tqdm
import glob
import pandas as pd
import matplotlib.pyplot as plt

def get_clip_embed(ref_image, clip_image_processor, image_encoder):
    """
    Получаем эмбединг CLIP для изображения

    Args:
        ref_image (PIL.Image): входное изображение
        clip_image_processor (CLIPImageProcessor): предобработка изображения
        image_encoder (CLIPVisionModelWithProjection): моделель CLIP

    Returns:
        clip_image_embeds (torch.Tensor): (1, 768) эмбединг
    """
    clip_image = clip_image_processor.preprocess(
    ref_image.resize((224, 224)), return_tensors="pt"
        ).pixel_values
    clip_image_embeds = image_encoder(
            clip_image.to("cuda", dtype=image_encoder.dtype)
        ).image_embeds
    return clip_image_embeds

def get_cosine_similarity(original_frame, generated_frame, clip_image_processor, image_encoder):
    """
    Получаем эмбединг CLIP для изображения

    Args:
        original_frame (PIL.Image): оригинальное изображение
        generated_frame (PIL.Image): сгенерированное изображение
        clip_image_processor (CLIPImageProcessor): предобработка изображения
        image_encoder (CLIPVisionModelWithProjection): моделель CLIP

    Returns:
        cos_sim (float): значение метрики
    """
    
    cos = CosineSimilarity(dim=1, eps=1e-8)
    original_embed = get_clip_embed(original_frame, clip_image_processor, image_encoder)
    generated_embed = get_clip_embed(generated_frame, clip_image_processor, image_encoder)
    cos_sim = cos(original_embed, generated_embed).item()

    return cos_sim

def get_pose_embed(ref_image, detector):
    """
    Получаем эмбединг (координаты ключевых точек) позы 

    Args:
        ref_image (PIL.Image): входное изображение
        detector (DWposeDetector): предобученная модель

    Return:
        pose_coord (np.array): (256, ) вектор позы
    """
    pose = detector(ref_image, output_type='pose')
    bodies = pose['bodies']['candidate'].reshape(1, -1)
    hands = pose['hands'].reshape(1, -1)
    faces = pose['faces'].reshape(1, -1)
    pose_coord = np.hstack([bodies, hands, faces])[0]
    return pose_coord

def get_pose_distance(original_frame, generated_frame, detector):
    """
    Получаем L2 расстояние между эмбедингами поз оригинального и сгенерированного изображения 

    Args:
        original_frame (PIL.Image): оригинальное изображение
        generated_frame (PIL.Image): сгенерированное изображение
        detector (DWposeDetector): предобученная модель

    Return:
        pose_coord (np.array): (256, ) вектор позы
    """
    orig_pose = get_pose_embed(original_frame, detector)
    gen_pose = get_pose_embed(generated_frame, detector)
    dist = np.linalg.norm(orig_pose-gen_pose)
    return dist

def stage_2_estimation(n, clip_image_processor, image_encoder, detector):
    """
    Оценка последовательности сгенерированных кадров относительно таргетных изображений

    Args:
        n (int): кол-во видео для обработки
        clip_image_processor (CLIPImageProcessor): предобработка изображения
        image_encoder (CLIPVisionModelWithProjection): моделель CLIP
        detector (DWposeDetector): предобученная модель
    
    Return:
        df (pd.DataFrame): покадровые метрики
    """
    data = []
    for i in tqdm(range(n)):
        target_path = glob.glob(f"../dataset/test/stage_2/sample_{i}/orig_vid.mp4")[0]
        target_frames = read_frames(target_path)
        generated_path = glob.glob(f"../dataset/test/stage_2/sample_{i}/generationstep*")
        sample_data = {}
        
        for gen in generated_path:
            epoch = int(gen.split('/')[-1].strip('.gif').split('_')[-1])
            gen_frames = read_frames(gen)
            
            sample_data[epoch] = {}
            for idx in range(len(gen_frames)):
    
                source_image = target_frames[idx]
                generated_image = gen_frames[idx].resize(source_image.size)
            
                pose_sim = get_pose_distance(source_image, generated_image, detector)
                clip_sim = get_cosine_similarity(source_image, generated_image, clip_image_processor, image_encoder)
    
                sample_data[epoch][idx] = {
                    "pose_sim" : pose_sim,
                    "clip_sim" : clip_sim
                }
        data.append(sample_data)

    records = []
    for i, run in enumerate(data):
        for epoch, frame_score in run.items():
            for frame, metrics, in frame_score.items():
                record = {
                    "run": i,
                    "iter": epoch,
                    "frame": frame,
                    **metrics
                }
                records.append(record)
    
    df = pd.DataFrame.from_records(records)
    df = df.set_index(["run", "iter", "frame"])
    return df

def show_frames(path, idx, size):
    """
    Отрисовка кадров видел в ряд

    Args:
        path (str): путь к видео
        idx (list[int]): индексы кадров для отрисовка
        size (tuple(int, int)): размеры кадров
    """
    frames = read_frames(path)
    selected_frames = [] 
    for i in idx:
        selected_frames.append(frames[i].resize(size))

    combined = np.hstack(selected_frames)
    plt.figure(figsize=(15, 5))
    plt.imshow(combined)
    plt.axis('off')