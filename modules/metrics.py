from torch.nn import CosineSimilarity
import torch
import numpy as np

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