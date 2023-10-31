import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2


class SaliencyMap:
    def __init__(self, model, image_shape, image_path, class_index, num_classes):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = cv2.resize(img, image_shape) / 255
        self.image = np.expand_dims(self.img, axis=0)
        expected_output = tf.one_hot([class_index], num_classes)
        with tf.GradientTape() as tape:
            inputs = tf.cast(self.image, tf.float32)
            tape.watch(inputs)
            predictions = model(inputs)
            loss = tf.keras.losses.categorical_crossentropy(
                expected_output, predictions
            )
        gradients = tape.gradient(loss, inputs)
        grayscale_tensor = tf.reduce_sum(tf.abs(gradients), -1)

        # normalize_value = (value − min_value) / (max_value − min_value)
        normalized_gradients = (grayscale_tensor - tf.reduce_min(grayscale_tensor)) / (
                    tf.reduce_max(grayscale_tensor) - tf.reduce_min(grayscale_tensor))
        gradients_mask = tf.cast(255 * normalized_gradients, tf.uint8)
        self.gradients_mask = tf.squeeze(gradients_mask)

    def show_image(self):
        plt.imshow(self.img)
        plt.axis('off')
        plt.title('Input Image')
        plt.show()

    def show_gradients_mask(self):
        plt.axis('off')
        plt.imshow(self.gradients_mask, cmap='gray')
        plt.title('Gradients Mask')
        plt.show()

    def saliency_map(self):
        gradient_color = cv2.applyColorMap(self.gradients_mask.numpy(), cv2.COLORMAP_HOT)
        gradient_color = gradient_color / 255.0
        super_imposed = cv2.addWeighted(self.img, 0.5, gradient_color, 0.5, 0.0)
        plt.imshow(super_imposed)
        plt.axis('off')
        plt.title('Saliency Map')
        plt.show()
