import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
import matplotlib.pyplot as plt

print("✅ Starting training script...")

# -------------------------------
# 1️⃣ Pfade
# -------------------------------
train_dir = 'data/train'
val_dir = 'data/val'
model_path = 'saved_resnet'  # vorhandenes Modell

print("Loading dataset from:", train_dir, "and", val_dir)

# -------------------------------
# 2️⃣ Daten vorbereiten
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print("✅ Data generators ready")

# -------------------------------
# 3️⃣ Modell laden
# -------------------------------
print("Loading model from", model_path)
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded")
print(model.summary())

# Optional: Kopf für 2 Klassen anpassen, falls nötig
if model.output_shape[-1] != 2:
    print("Adjusting model head for 2 classes")
    x = model.layers[-2].output
    new_output = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=model.input, outputs=new_output)
    print("✅ Model head adjusted")

# -------------------------------
# 4️⃣ Kompilieren
# -------------------------------
print("Compiling model...")
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("✅ Model compiled")

# -------------------------------
# 5️⃣ Training
# -------------------------------
num_epochs = 3
print(f"Starting training for {num_epochs} epochs...")
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator,
    verbose=1  # zeigt Batch-Fortschritt automatisch
)
print("✅ Training completed")

# -------------------------------
# 6️⃣ Trainingsverlauf plotten
# -------------------------------
print("Plotting training loss...")
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
plt.show()
print("✅ Training loss plot saved as loss_plot.png")

# -------------------------------
# 7️⃣ Modell speichern (überschreiben)
# -------------------------------
print("Saving updated model (overwriting existing)...")
model.save(model_path)
print(f"✅ Model saved and overwritten at {model_path}")
