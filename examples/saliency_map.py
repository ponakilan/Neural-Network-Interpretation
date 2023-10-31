import tensorflow_hub as hub
import tensorflow as tf
from visualizer.saliencymap import SaliencyMap

model = tf.keras.Sequential([
    hub.KerasLayer('https://tfhub.dev/google/tf2-preview/inception_v3/classification/4'),
    tf.keras.layers.Activation('softmax')
])
model.build([None, 300, 300, 3])

visualizer = SaliencyMap(model=model,
                         image_shape=(300, 300),
                         image_path="C:\\Users\\ponak\\Downloads\\the-pacific-ocean-3185553_960_720.jpg",
                         class_index=251,
                         num_classes=1001)
visualizer.show_image()
visualizer.show_gradients_mask()
visualizer.saliency_map()
