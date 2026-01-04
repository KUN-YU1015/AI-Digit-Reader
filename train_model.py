import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 載入 MNIST 資料集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 資料預處理
# 將資料正規化 (0~1)，並調整形狀以符合卷積層 (CNN) 輸入
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 3. 建立數據增強產生器 (核心改進！)
# 這會隨機旋轉、縮放、平移圖片，模擬現實中拍照不準或原子筆較細的情況
datagen = ImageDataGenerator(
    rotation_range=15,      # 隨機旋轉
    zoom_range=0.15,        # 隨機縮放
    width_shift_range=0.1,  # 隨機水平平移
    height_shift_range=0.1  # 隨機垂直平移
)
datagen.fit(x_train)

# 4. 建立 CNN 模型架構
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2), # 防止過擬合
    layers.Dense(10, activation='softmax')
])

# 5. 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. 開始訓練 (使用數據增強)
print("開始強化訓練模型，請稍候...")
model.fit(datagen.flow(x_train, y_train, batch_size=32), 
          epochs=10, # 增加訓練次數
          validation_data=(x_test, y_test))

# 7. 儲存模型
model.save('mnist_model.h5')
print("✅ 模型訓練完成並已更新為 mnist_model.h5")