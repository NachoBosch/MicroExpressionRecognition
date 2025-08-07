import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as ms
import os


# data_dir = "C:/Doctorado/Neurociencia/CASME/CASME2_splitted"
data_dir = "C:/Doctorado/Neurociencia/CASME/CASME-II-Binary_splitted"
img_size = (224, 224)
batch_size = 4
num_classes = 2
epochs = 100

for dir, dirname, files in os.walk(data_dir):
    print(f"Dir: {dir} | subdir: {dirname} | cant de imagenes: {len(files)} ")



train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,horizontal_flip=True)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(f"{data_dir}/train",
                                                    target_size=img_size,
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    shuffle=True)

val_generator = val_datagen.flow_from_directory(f"{data_dir}/val",
                                                target_size=img_size,
                                                batch_size=batch_size,
                                                class_mode='binary',
                                                shuffle=False)

class_names = list(train_generator.class_indices.keys())
print(class_names)


base_model = tf.keras.applications.ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(*img_size, 3))
base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    # tf.keras.layers.Dropout(0.25),
    # tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(num_classes, activation='softmax')
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    # loss=tf.keras.losses.CategoricalCrossentropy(),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'casme_binary_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

val_loss, val_acc, val_precision, val_recall = model.evaluate(val_generator, verbose=1)

predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

print(ms.classification_report(y_true, y_pred, target_names=class_names))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes[0, 0].plot(history.history['loss'], label='Entrenamiento')
axes[0, 0].plot(history.history['val_loss'], label='Validación')
axes[0, 0].set_title('Pérdida del Modelo')
axes[0, 0].set_xlabel('Época')
axes[0, 0].set_ylabel('Pérdida')
axes[0, 0].legend()
axes[0, 0].grid(True)
axes[0, 1].plot(history.history['accuracy'], label='Entrenamiento')
axes[0, 1].plot(history.history['val_accuracy'], label='Validación')
axes[0, 1].set_title('Precisión del Modelo')
axes[0, 1].set_xlabel('Época')
axes[0, 1].set_ylabel('Precisión')
axes[0, 1].legend()
axes[0, 1].grid(True)
plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

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
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# model.save('casme_2_binary_model.h5')
print('¡Entrenamiento completado!')
