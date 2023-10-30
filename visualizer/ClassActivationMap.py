import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np
import scipy as sp


class ClassActivationMap:
    def __init__(self, classification_model, final_conv_layer_index, input_images):
        self.classification_model = classification_model
        self.final_conv_layer_index = final_conv_layer_index
        self.input_images = input_images
        self.final_conv_layer = classification_model.layers[final_conv_layer_index]
        self.output_layer = classification_model.layers[-1]
        self.cam_model = Model(inputs=self.classification_model.input,
                               outputs=(self.final_conv_layer.output,
                                        self.output_layer.output))
        self.features, self.final_output = self.cam_model.predict(input_images)

    def show_image(self, image_index, image_title):
        plt.imshow(self.input_images[image_index])
        plt.title(image_title)
        plt.axis('off')
        plt.show()

    def cam_view(self, image_index, scale_factor):
        gap_params = self.output_layer.get_weights()
        gap_weights = gap_params[0]
        predicted_class = np.argmax(self.final_output[image_index])
        upscaled_image = sp.ndimage.zoom(self.features[image_index, :, :, :], scale_factor, order=2)
        cam_image = np.dot(upscaled_image, gap_weights[:, predicted_class])
        if self.final_output[image_index][predicted_class] >= 0.95:
            cmap_color = 'Greens'
        else:
            cmap_color = 'Reds'
        plt.imshow(self.input_images[image_index])
        plt.imshow(cam_image, cmap=cmap_color, alpha=0.5)
        plt.title('Class Activation Map')
        plt.axis('off')
        plt.show()