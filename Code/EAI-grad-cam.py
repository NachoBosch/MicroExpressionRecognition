import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

def grad_cam(model, img_array, layer_name, class_idx):
    """
    Genera el heatmap de Grad-CAM para una imagen y clase específica.
    
    Args:
        model: Modelo de Keras
        img_array: Imagen preprocesada (batch_size, height, width, channels)
        layer_name: Nombre de la última capa convolucional
        class_idx: Índice de la clase a explicar
    
    Returns:
        heatmap: Mapa de calor normalizado (0-1)
    """
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]
    
    # Calcular gradientes
    grads = tape.gradient(loss, conv_outputs)
    
    # Ponderar cada feature map por su importancia (promedio global)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Combinación lineal ponderada
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Aplicar ReLU y normalizar
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()


def superimpose_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Superpone el heatmap sobre la imagen original.
    
    Args:
        img: Imagen original (numpy array)
        heatmap: Heatmap de Grad-CAM
        alpha: Transparencia del heatmap (0-1)
        colormap: Mapa de colores de OpenCV
    
    Returns:
        Imagen con heatmap superpuesto
    """
    # Redimensionar heatmap al tamaño de la imagen
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convertir heatmap a color
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
    
    # Convertir de BGR a RGB (OpenCV usa BGR)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Normalizar imagen original si está en [0, 1]
    if img.max() <= 1.0:
        img = np.uint8(255 * img)
    
    # Superponer
    superimposed = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)
    
    return superimposed


def visualize_gradcam(model, img_path, class_names, layer_name='conv5_block3_out', 
                      img_size=(224, 224), preprocess_input=None):
    """
    Visualiza Grad-CAM para una imagen específica.
    
    Args:
        model: Modelo entrenado
        img_path: Path a la imagen
        class_names: Lista con nombres de clases
        layer_name: Nombre de la última capa convolucional de ResNet50V2
        img_size: Tamaño de entrada del modelo
        preprocess_input: Función de preprocesamiento (opcional)
    """
    # Cargar y preprocesar imagen
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Guardar imagen original para visualización
    img_display = img_array.copy()
    
    # Preprocesar
    if preprocess_input:
        img_array = preprocess_input(img_array)
    else:
        img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    # Hacer predicción
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    print(f"Predicción: {class_names[predicted_class]} ({confidence*100:.2f}%)")
    print(f"Probabilidades por clase:")
    for i, prob in enumerate(predictions[0]):
        print(f"  {class_names[i]}: {prob*100:.2f}%")
    
    # Generar Grad-CAM
    heatmap = grad_cam(model, img_array, layer_name, predicted_class)
    
    # Superponer heatmap
    superimposed = superimpose_heatmap(img_display, heatmap, alpha=0.4)
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Imagen original
    axes[0].imshow(img_display.astype('uint8'))
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    # Heatmap solo
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Superposición
    axes[2].imshow(superimposed.astype('uint8'))
    axes[2].set_title(f'Pred: {class_names[predicted_class]} ({confidence*100:.1f}%)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return heatmap, superimposed, predictions


def batch_gradcam_visualization(model, val_generator, class_names, 
                                 layer_name='conv5_block3_out', num_samples=9):
    """
    Visualiza Grad-CAM para múltiples imágenes del validation set.
    
    Args:
        model: Modelo entrenado
        val_generator: Generador de validación
        class_names: Lista con nombres de clases
        layer_name: Nombre de la última capa convolucional
        num_samples: Número de imágenes a visualizar
    """
    # Obtener un batch de imágenes
    imgs, labels = next(val_generator)
    
    # Limitar a num_samples
    imgs = imgs[:num_samples]
    labels = labels[:num_samples]
    
    # Calcular número de filas y columnas
    cols = 3
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        img_array = np.expand_dims(imgs[i], axis=0)
        
        # Predicción
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        true_class = np.argmax(labels[i])
        confidence = predictions[0][predicted_class]
        
        # Generar Grad-CAM
        heatmap = grad_cam(model, img_array, layer_name, predicted_class)
        
        # Superponer (la imagen ya viene normalizada de 0-1)
        img_display = imgs[i] * 255.0  # Desnormalizar para visualización
        superimposed = superimpose_heatmap(img_display, heatmap, alpha=0.4)
        
        # Visualizar
        axes[i].imshow(superimposed.astype('uint8'))
        
        color = 'green' if predicted_class == true_class else 'red'
        title = f'True: {class_names[true_class]}\nPred: {class_names[predicted_class]} ({confidence*100:.1f}%)'
        axes[i].set_title(title, color=color, fontweight='bold')
        axes[i].axis('off')
    
    # Ocultar ejes sobrantes
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def compare_classes_gradcam(model, img_path, class_names, layer_name='conv5_block3_out', 
                            img_size=(224, 224)):
    """
    Compara los heatmaps de Grad-CAM para todas las clases en una misma imagen.
    Útil para ver qué regiones activan cada clase.
    
    Args:
        model: Modelo entrenado
        img_path: Path a la imagen
        class_names: Lista con nombres de clases
        layer_name: Nombre de la última capa convolucional
        img_size: Tamaño de entrada del modelo
    """
    # Cargar imagen
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_display = img_array.copy()
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predicción
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    
    # Generar heatmap para cada clase
    num_classes = len(class_names)
    cols = min(4, num_classes)
    rows = (num_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if num_classes > 1 else [axes]
    
    for class_idx in range(num_classes):
        heatmap = grad_cam(model, img_array, layer_name, class_idx)
        superimposed = superimpose_heatmap(img_display, heatmap, alpha=0.4)
        
        axes[class_idx].imshow(superimposed.astype('uint8'))
        
        prob = predictions[0][class_idx]
        title = f'{class_names[class_idx]}\n({prob*100:.1f}%)'
        
        # Destacar la clase predicha
        if class_idx == predicted_class:
            axes[class_idx].set_title(title, fontweight='bold', 
                                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        else:
            axes[class_idx].set_title(title)
        
        axes[class_idx].axis('off')
    
    # Ocultar ejes sobrantes
    for i in range(num_classes, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Grad-CAM para todas las clases', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    # Cargar modelo entrenado
    model = tf.keras.models.load_model('casme_multiclass_model.h5')
    
    # Nombres de clases (usa los de tu dataset)
    class_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']  # AJUSTA ESTO
    
    # Última capa convolucional de ResNet50V2
    # Opciones comunes: 'conv5_block3_out', 'conv5_block2_out', 'conv4_block6_out'
    layer_name = 'conv5_block3_out'
    
    # 1. Visualizar una imagen específica
    print("=== Visualización de imagen específica ===")
    img_path = 'path/to/test/image.jpg'  # CAMBIA ESTO
    visualize_gradcam(model, img_path, class_names, layer_name=layer_name)
    
    # 2. Visualizar batch del validation set
    print("\n=== Visualización de batch ===")
    # Recrear val_generator (ajusta paths)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        'data/val',  # AJUSTA PATH
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )
    
    batch_gradcam_visualization(model, val_generator, class_names, 
                                layer_name=layer_name, num_samples=9)
    
    # 3. Comparar todas las clases en una misma imagen
    print("\n=== Comparación de clases ===")
    compare_classes_gradcam(model, img_path, class_names, layer_name=layer_name)