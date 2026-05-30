# src/attention_viz.py
"""
Utilidades de visualización para mapas de atención de Attention U-Net 3D 
específicamente diseñadas para volúmenes DBT.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
import os
from typing import List, Tuple, Optional
import seaborn as sns

def create_attention_colormap():
    """Crear colormap personalizado para mapas de atención"""
    colors = ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('attention', colors, N=n_bins)
    return cmap

def visualize_attention_maps(
    model, 
    volume: torch.Tensor, 
    slice_indices: Optional[List[int]] = None,
    save_dir: str = "outputs/attention_maps",
    study_id: str = "unknown",
    device: torch.device = torch.device("cpu")
) -> List[str]:
    """
    Visualizar mapas de atención para un volumen DBT
    
    Args:
        model: Modelo AttentionUNet3D entrenado
        volume: Volumen de entrada (1, 1, Z, H, W)
        slice_indices: Índices de cortes a visualizar (si None, selecciona automáticamente)
        save_dir: Directorio donde guardar las visualizaciones
        study_id: ID del estudio para nombrar archivos
        device: Dispositivo donde ejecutar el modelo
    
    Returns:
        Lista de rutas de archivos guardados
    """
    model.eval()
    model.to(device)
    volume = volume.to(device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Obtener mapas de atención
    with torch.no_grad():
        attention_maps = model.get_attention_maps(volume)
    
    # Seleccionar cortes si no se especifican
    if slice_indices is None:
        Z = volume.shape[2]
        slice_indices = [Z//4, Z//2, 3*Z//4]  # Cuartiles
    
    saved_files = []
    cmap = create_attention_colormap()
    
    for level, attn_map in enumerate(attention_maps):
        # attn_map: (1, 1, Z', H', W')
        attn_np = attn_map.squeeze().cpu().numpy()
        
        # Redimensionar a tamaño original si es necesario
        if attn_np.shape != volume.shape[2:]:
            target_shape = volume.shape[2:]
            attn_resized = F.interpolate(
                attn_map, 
                size=target_shape, 
                mode='trilinear', 
                align_corners=False
            )
            attn_np = attn_resized.squeeze().cpu().numpy()
        
        # Crear visualización para este nivel
        fig, axes = plt.subplots(2, len(slice_indices), figsize=(15, 8))
        if len(slice_indices) == 1:
            axes = axes.reshape(2, 1)
        
        for i, slice_idx in enumerate(slice_indices):
            if slice_idx >= attn_np.shape[0]:
                slice_idx = attn_np.shape[0] - 1
            
            # Imagen original
            orig_slice = volume[0, 0, slice_idx].cpu().numpy()
            axes[0, i].imshow(orig_slice, cmap='gray', aspect='auto')
            axes[0, i].set_title(f'Original - Slice {slice_idx}')
            axes[0, i].axis('off')
            
            # Mapa de atención
            attn_slice = attn_np[slice_idx]
            im = axes[1, i].imshow(attn_slice, cmap=cmap, aspect='auto', vmin=0, vmax=1)
            axes[1, i].set_title(f'Attention Level {level+1} - Slice {slice_idx}')
            axes[1, i].axis('off')
        
        # Añadir colorbar
        plt.tight_layout()
        cbar = plt.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', 
                           fraction=0.046, pad=0.08)
        cbar.set_label('Attention Weight', fontsize=12)
        
        plt.suptitle(f'Attention Maps - {study_id} - Level {level+1}', 
                     fontsize=14, y=0.95)
        
        # Guardar
        filename = f"{study_id}_attention_level_{level+1}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(filepath)
    
    return saved_files

def create_attention_overlay(
    original_volume: torch.Tensor,
    attention_maps: List[torch.Tensor],
    slice_idx: int,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Crear overlay de atención sobre imagen original
    
    Args:
        original_volume: Volumen original (1, 1, Z, H, W)
        attention_maps: Lista de mapas de atención
        slice_idx: Índice del corte
        alpha: Transparencia del overlay
    
    Returns:
        Imagen RGB con overlay de atención
    """
    # Imagen base en escala de grises
    base_slice = original_volume[0, 0, slice_idx].cpu().numpy()
    
    # Normalizar a [0, 1]
    base_slice = (base_slice - base_slice.min()) / (base_slice.max() - base_slice.min())
    
    # Crear imagen RGB
    rgb_image = np.stack([base_slice, base_slice, base_slice], axis=-1)
    
    # Combinar mapas de atención
    combined_attention = torch.zeros_like(attention_maps[0])
    for attn_map in attention_maps:
        # Redimensionar si es necesario
        if attn_map.shape[2:] != original_volume.shape[2:]:
            attn_resized = F.interpolate(
                attn_map,
                size=original_volume.shape[2:],
                mode='trilinear',
                align_corners=False
            )
        else:
            attn_resized = attn_map
        
        combined_attention = torch.max(combined_attention, attn_resized)
    
    # Obtener corte de atención
    attn_slice = combined_attention[0, 0, slice_idx].cpu().numpy()
    
    # Crear overlay rojo para regiones de alta atención
    overlay = np.zeros_like(rgb_image)
    overlay[:, :, 0] = attn_slice  # Canal rojo
    
    # Combinar
    result = (1 - alpha) * rgb_image + alpha * overlay
    result = np.clip(result, 0, 1)
    
    return result

def plot_attention_statistics(
    attention_maps: List[torch.Tensor],
    save_path: str = None
) -> plt.Figure:
    """
    Generar estadísticas de los mapas de atención
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Estadísticas por nivel
    levels = []
    means = []
    stds = []
    maxs = []
    
    for i, attn_map in enumerate(attention_maps):
        attn_np = attn_map.cpu().numpy().flatten()
        levels.append(f'Level {i+1}')
        means.append(np.mean(attn_np))
        stds.append(np.std(attn_np))
        maxs.append(np.max(attn_np))
        
        # Histograma
        if i < 4:  # Solo primeros 4 niveles
            row, col = i // 2, i % 2
            axes[row, col].hist(attn_np, bins=50, alpha=0.7, density=True)
            axes[row, col].set_title(f'Attention Distribution - Level {i+1}')
            axes[row, col].set_xlabel('Attention Weight')
            axes[row, col].set_ylabel('Density')
            axes[row, col].grid(True, alpha=0.3)
    
    # Ocultar subplots no usados
    for i in range(len(attention_maps), 4):
        row, col = i // 2, i % 2
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def visualize_attention_evolution(
    model,
    volume: torch.Tensor,
    save_dir: str = "outputs/attention_evolution",
    study_id: str = "unknown",
    device: torch.device = torch.device("cpu")
) -> List[str]:
    """
    Visualizar la evolución de la atención a través de los niveles
    """
    model.eval()
    model.to(device)
    volume = volume.to(device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        attention_maps = model.get_attention_maps(volume)
    
    saved_files = []
    
    # Seleccionar corte central
    Z = volume.shape[2]
    central_slice = Z // 2
    
    # Crear figura con evolución
    n_levels = len(attention_maps)
    fig, axes = plt.subplots(2, n_levels, figsize=(4*n_levels, 8))
    
    if n_levels == 1:
        axes = axes.reshape(2, 1)
    
    # Imagen original
    orig_slice = volume[0, 0, central_slice].cpu().numpy()
    
    for level, attn_map in enumerate(attention_maps):
        # Redimensionar atención al tamaño original
        if attn_map.shape[2:] != volume.shape[2:]:
            attn_resized = F.interpolate(
                attn_map,
                size=volume.shape[2:],
                mode='trilinear',
                align_corners=False
            )
        else:
            attn_resized = attn_map
        
        attn_slice = attn_resized[0, 0, central_slice].cpu().numpy()
        
        # Plot original + atención
        axes[0, level].imshow(orig_slice, cmap='gray', aspect='auto')
        axes[0, level].set_title(f'Original - Level {level+1}')
        axes[0, level].axis('off')
        
        im = axes[1, level].imshow(attn_slice, cmap='hot', aspect='auto', vmin=0, vmax=1)
        axes[1, level].set_title(f'Attention Level {level+1}')
        axes[1, level].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Attention Evolution - {study_id} - Slice {central_slice}', 
                 fontsize=16, y=0.98)
    
    # Guardar
    filename = f"{study_id}_attention_evolution.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(filepath)
    
    return saved_files

def compare_predictions_with_attention(
    model,
    volume: torch.Tensor,
    ground_truth: torch.Tensor,
    save_path: str,
    study_id: str = "unknown",
    device: torch.device = torch.device("cpu")
):
    """
    Comparar predicciones con mapas de atención y ground truth
    """
    model.eval()
    model.to(device)
    volume = volume.to(device)
    
    with torch.no_grad():
        # Predicción
        prediction = model(volume)
        # Mapas de atención
        attention_maps = model.get_attention_maps(volume)
    
    # Seleccionar corte con más información
    gt_np = ground_truth.squeeze().cpu().numpy()
    slice_sums = np.sum(gt_np, axis=(1, 2))
    best_slice = np.argmax(slice_sums)
    
    # Crear visualización comparativa
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Imagen original
    orig_slice = volume[0, 0, best_slice].cpu().numpy()
    axes[0, 0].imshow(orig_slice, cmap='gray', aspect='auto')
    axes[0, 0].set_title('Original DBT Slice')
    axes[0, 0].axis('off')
    
    # Ground truth
    gt_slice = gt_np[best_slice]
    axes[0, 1].imshow(gt_slice, cmap='Reds', aspect='auto', alpha=0.8)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # Predicción
    pred_slice = prediction[0, 0, best_slice].cpu().numpy()
    axes[0, 2].imshow(pred_slice, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[0, 2].set_title('Prediction')
    axes[0, 2].axis('off')
    
    # Overlay: Original + GT
    overlay1 = create_attention_overlay(volume, [ground_truth.unsqueeze(0).unsqueeze(0)], best_slice, alpha=0.3)
    axes[1, 0].imshow(overlay1, aspect='auto')
    axes[1, 0].set_title('Original + Ground Truth')
    axes[1, 0].axis('off')
    
    # Overlay: Original + Predicción
    overlay2 = create_attention_overlay(volume, [prediction.unsqueeze(0)], best_slice, alpha=0.3)
    axes[1, 1].imshow(overlay2, aspect='auto')
    axes[1, 1].set_title('Original + Prediction')
    axes[1, 1].axis('off')
    
    # Mapa de atención principal
    if attention_maps:
        # Usar el último nivel de atención (más refinado)
        main_attention = attention_maps[-1]
        if main_attention.shape[2:] != volume.shape[2:]:
            main_attention = F.interpolate(
                main_attention,
                size=volume.shape[2:],
                mode='trilinear',
                align_corners=False
            )
        
        attn_slice = main_attention[0, 0, best_slice].cpu().numpy()
        im = axes[1, 2].imshow(attn_slice, cmap='hot', aspect='auto', vmin=0, vmax=1)
        axes[1, 2].set_title('Attention Map')
        axes[1, 2].axis('off')
        
        # Colorbar para atención
        plt.colorbar(im, ax=axes[1, 2], orientation='horizontal', fraction=0.046, pad=0.1)
    
    plt.tight_layout()
    plt.suptitle(f'Prediction Analysis with Attention - {study_id} - Slice {best_slice}', 
                 fontsize=16, y=0.98)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path 