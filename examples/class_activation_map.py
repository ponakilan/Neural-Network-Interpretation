from visualizer.cam import ClassActivationMap
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train/255
X_test = X_test/255
X_train = X_train.astype('float')
X_test = X_test.astype('float')

model = Sequential([
    Conv2D(16, input_shape=(28, 28, 1), kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    GlobalAveragePooling2D(),
    Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
model.fit(X_train,y_train,batch_size=32, epochs=5, validation_split=0.1, shuffle=True)

cam_visualizer = ClassActivationMap(model, -3, X_test)
cam_visualizer.show_image(2, 'Test Image')
cam_visualizer.cam_view(2, (28/3, 28/3, 1))
