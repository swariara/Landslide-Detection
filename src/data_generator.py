import os
import numpy as np
import tensorflow as tf

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image

class LandslideDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_ids, labels, folder_path, batch_size=32, shuffle=True, augment=False):
        self.image_ids = image_ids
        self.labels = np.array(labels)
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = [self.image_ids[i] for i in indexes]
        batch_labels = self.labels[indexes]

        batch_images = [self.load_and_normalize_npy_image(image_id) for image_id in batch_ids]
        batch_images = np.array(batch_images)

        if self.augment:
            batch_images = np.array([augment_image(tf.convert_to_tensor(img)).numpy() for img in batch_images])

        return batch_images, np.array(batch_labels)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_and_normalize_npy_image(self, image_id):
        path = os.path.join(self.folder_path, f"{image_id}.npy")
        img = np.load(path)
        img = (img - img.mean(axis=(0, 1))) / (img.std(axis=(0, 1)) + 1e-5)
        return img
