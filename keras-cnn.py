
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 入力データはmnistデータセットです
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()

# 6つの畳み込みカーネル
# サイズ5×5
# 活性化関数relu
# 入力画像サイズ28×28
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))

# プール層2*2
model.add(MaxPooling2D(pool_size=(2, 2)))

# 16つの畳み込みカーネル
# サイズ5×5
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))

# プール層2*2
model.add(MaxPooling2D(pool_size=(2, 2)))

# 完全接続層の出力サイズ120 活性化関数relu
model.add(Flatten())
model.add(Dense(120, activation='relu'))

# 出力サイズ84 活性化関数relu
model.add(Dense(84, activation='relu'))

# 出力サイズ10 活性化関数softmax
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1, validation_data=(x_test, y_test))

# テスト
score = model.evaluate(x_test, y_test)
pre = model.predict_classes(x_test).astype('int')
prediction = model.predict_classes(x_test).astype('int')
y_test = np.argmax(y_test.astype('int'),axis=1)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

# 間違えた画像
diff = np.where(prediction != y_test)[0]
fig = plt.figure(figsize = (18, 7))
x_test = x_test.reshape((10000, 28, 28))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[np.where(prediction != y_test)[0]][i], cmap = "gray")
    plt.title("y_test : {}, prediction : {}".format(y_test[diff][i], prediction[diff][i]), fontsize = 15)
plt.show()