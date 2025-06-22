import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

class F1ScoreCallback(Callback):
    def __init__(self, val_generator, save_path):
        super().__init__()
        self.val_generator = val_generator
        self.save_path = save_path
        self.best_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for x_batch, y_batch in self.val_generator:
            preds = self.model.predict(x_batch)
            y_pred.extend((preds >= 0.5).astype(int).flatten())
            y_true.extend(y_batch)
            if len(y_true) >= len(self.val_generator.labels):
                break

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        print(f"\nEpoch {epoch+1} - val_f1: {f1:.4f}, precision: {prec:.4f}, recall: {rec:.4f}")

        if f1 > self.best_f1:
            print(f"New best F1. Saving model to {self.save_path}")
            self.best_f1 = f1
            self.model.save(self.save_path)

        if logs is not None:
            logs['val_f1'], logs['val_precision'], logs['val_recall'] = f1, prec, rec

def build_cnn_model(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(256, 256, 12)))

    for i in range(hp.Int('conv_layers', 2, 4)):
        filters = hp.Int(f'filters_{i}', 32, 128, step=32)
        model.add(layers.Conv2D(filters, 3, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'))
    model.add(layers.Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1)))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
