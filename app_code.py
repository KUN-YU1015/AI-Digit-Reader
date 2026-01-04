import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# 1. 頁面設定
st.set_page_config(page_title="AI手寫辨識APP", layout="wide")
# 修改標題
st.title("🔢 AI手寫辨識APP")
# 增加使用提示
st.markdown("##### 💡 **提示：使用較粗的筆書寫（如馬克筆）並在光源充足的情況下拍攝照片，以提高辨識成功率。**")

# 2. 載入模型
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('mnist_model.h5')

try:
    model = load_my_model()
    st.success("✅ AI 辨識模型已成功啟動！")
except Exception as e:
    st.error(f"❌ 找不到模型檔，請確保 'mnist_model.h5' 在同一資料夾。")

# 3. 側邊欄：互動式性能優化
st.sidebar.header("🛠️ 辨識性能優化")
sensitivity = st.sidebar.slider(
    "1. 捕捉靈敏度", 1, 25, 12, 
    help="針對不同光線。若筆跡太淡，請『降低』數值；若雜訊過多，請『提高』數值。"
)
thickness = st.sidebar.slider(
    "2. 字體加粗程度", 1, 5, 2,
    help="針對細筆跡補強。若數字斷裂，請『提高』數值以連接筆劃。"
)
min_area = st.sidebar.slider(
    "3. 雜訊過濾強度", 100, 1500, 300,
    help="剔除微小雜點。若畫面出現非數字的小碎框，請『提高』數值。"
)

st.sidebar.divider()
option = st.sidebar.radio("📸 選擇輸入來源：", ("上傳圖片檔", "使用相機拍照"))

# 4. 影像輸入與處理
img_file = st.file_uploader("請上傳圖片", type=["jpg", "png", "jpeg"]) if option == "上傳圖片檔" else st.camera_input("拍照")

if img_file is not None:
    image = Image.open(img_file)
    img_array = np.array(image.convert('RGB'))
    st.image(image, caption="原始圖片", width=400)

    # 影像增強處理
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.convertScaleAbs(gray, alpha=3.0, beta=-150)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, sensitivity)
    
    # 形態學優化
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=thickness)
    
    # 尋找與過濾輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if valid_contours:
        # 由左至右排序輪廓
        valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])
        
        st.write("### 🤖 AI 辨識細節 (28x28 特徵提取)：")
        cols = st.columns(min(len(valid_contours), 10))
        results = []

        for i, cnt in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(cnt)
            pad = 25
            sq_size = max(w, h) + pad * 2
            roi = thresh[y:y+h, x:x+w]
            digit_canvas = np.zeros((sq_size, sq_size), dtype="uint8")
            digit_canvas[pad:pad+h, pad:pad+w] = roi
            
            final_img = cv2.resize(digit_canvas, (28, 28), interpolation=cv2.INTER_AREA)
            input_data = final_img.astype('float32') / 255.0
            input_data = np.expand_dims(input_data, axis=(0, -1))
            
            prediction = model.predict(input_data, verbose=0)
            digit = np.argmax(prediction)
            results.append(str(digit))
            
            if i < 10:
                with cols[i]:
                    st.image(final_img, caption=f"預測: {digit}")

        st.divider()
        st.success(f"## 🔢 最終辨識結果： {''.join(results)}")
    else:
        st.warning("偵測不到明顯數字。")