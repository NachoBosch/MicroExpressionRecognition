import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as ms
import os


# data_dir = "C:/Doctorado/Neurociencia/CASME/CASME2_splitted"
data_dir = "../CASMEII/CASME-II-splitted"
img_size = (224, 224)
batch_size = 32
num_classes = 7
epochs = 200

for dir, dirname, files in os.walk(data_dir):
    print(f"Dir: {dir} | subdir: {dirname} | cant de imagenes: {len(files)} ")


# def add_noise(image):
#     if np.random.rand() < 0.5:
#         noise = np.random.normal(0, 0.002, image.shape) #ruido gaussiano con media 0 y std 0.02
#         noisy_image = image + noise
#     else:
#         noisy_image = image
#     return np.clip(noisy_image, 0., 1.)

def add_salt_pepper_noise(image, salt_prob:float, pepper_prob:float):
    """
    Agrega ruido Salt & Pepper a una imagen.
    - salt_prob: probabilidad de poner pixel blanco (255)
    - pepper_prob: probabilidad de poner pixel negro (0)
    """
    if np.random.rand() < 0.5:
        image = tf.cast(image, tf.float32)
        random_noise = tf.random.uniform(tf.shape(image), minval=0, maxval=1.0)
        salt_mask = random_noise <= salt_prob
        pepper_mask = (random_noise > salt_prob) & (random_noise <= salt_prob + pepper_prob)
        image = tf.where(salt_mask, 255.0, image)
        image = tf.where(pepper_mask, 0.0, image)

    return image

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=lambda x: add_salt_pepper_noise(x, salt_prob=0.005, pepper_prob=0.005),
                rescale=1./255,
                rotation_range=5,
                width_shift_range=0.05,
                height_shift_range=0.05)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(f"{data_dir}/train",
                                                    target_size=img_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

val_generator = val_datagen.flow_from_directory(f"{data_dir}/val",
                                                target_size=img_size,
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                shuffle=False)

class_names = list(train_generator.class_indices.keys())
print(class_names)
'''
imgs, labels = next(train_generator)
print(imgs[0].shape,imgs[0].max(),imgs[0].min())
labels = labels[0]
plt.imshow(imgs[0])
plt.show()
print(labels)
'''

base_model = tf.keras.applications.ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(*img_size, 3))
base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    # tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    # tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    # loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'casme_multiclass_model_noisy_dataaug_sinEStop_complex.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[model_checkpoint],#[early_stopping, model_checkpoint],
    verbose=1
)

val_loss, val_acc, val_precision, val_recall = model.evaluate(val_generator, verbose=1)

predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

print(ms.classification_report(y_true, y_pred, target_names=class_names))

fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].plot(history.history['loss'], label='Entrenamiento')
axes[0].plot(history.history['val_loss'], label='Validación')
axes[0].set_title('Pérdida del Modelo')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Pérdida')
axes[0].legend()
axes[0].grid(True)
axes[1].plot(history.history['accuracy'], label='Entrenamiento')
axes[1].plot(history.history['val_accuracy'], label='Validación')
axes[1].set_title('Precisión del Modelo')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Precisión')
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig('training_history_multiclass_noisy_dataaug_sinEStop_complex.png', dpi=300, bbox_inches='tight')


cm = ms.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=class_names,
           yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_multiclass_noisy_dataaug_sinEStop_complex.png', dpi=300, bbox_inches='tight')
plt.show()

print('¡Entrenamiento completado!')
