import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 1. 頁面設定 (保持你原本的 layout="wide")
st.set_page_config(page_title="AI手寫辨識APP", layout="wide")

# --- [精確插入] 針對行動端 APP 的 CSS 補強，解決下拉刷新問題 ---
st.markdown(
    """
    <style>
    /* 核心修復：強制禁止 Webview 的下拉刷新手勢 */
    html, body, [data-testid="stAppViewContainer"] {
        overscroll-behavior-y: contain !important;
        overflow: hidden !important;
    }
    
    /* 確保畫板區域不會因為滑動而帶動整個頁面捲動 */
    canvas {
        touch-action: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 保持你原本的標題與三點提示內容 ---
st.title("🔢 AI手寫辨識APP")
st.markdown("""
##### 💡 **使用說明：**
1. **多元輸入模式**：支援畫板手寫、拍照、圖片檔案上傳。
2. **最佳辨識建議**：建議使用較粗的筆書寫（如馬克筆），並在光源充足的情況下拍攝，以提高辨識成功率。
3. **性能優化手腕**：若辨識不佳，可使用側面板微調參數提高辨識成功率。
""")
st.divider()

# 2. 載入模型 (保持原本邏輯)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('mnist_model.h5')

try:
    model = load_my_model()
    st.sidebar.success("✅ AI 模型已就緒")
except Exception as e:
    st.sidebar.error("❌ 模型載入失敗")

# 3. 側邊欄 (保持原本參數)
st.sidebar.header("🛠️ 系統功能設定")
option = st.sidebar.radio("📸 選擇輸入來源：", ("手寫畫板模式", "使用相機拍照", "上傳圖片檔"))

st.sidebar.divider()
st.sidebar.write("🔍 辨識參數微調 (拍照/上傳專用)")
min_area = st.sidebar.slider("1. 雜訊過濾強度", 100, 1500, 300)
sensitivity = st.sidebar.slider("2. 捕捉靈敏度", 1, 25, 12)
thickness = st.sidebar.slider("3. 字體加粗程度", 1, 5, 2)

# 4. 影像處理核心函數 (完全不動原本邏輯)
def process_and_predict(img_gray, is_canvas=False):
    if is_canvas:
        _, thresh = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    else:
        enhanced = cv2.convertScaleAbs(img_gray, alpha=1.5, beta=0)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, sensitivity)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=thickness)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = sorted([c for c in contours if cv2.contourArea(c) > min_area], 
                            key=lambda c: cv2.boundingRect(c)[0])
    
    if not contours or not valid_contours:
        return None, None

    results = []
    roi_images = []
    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        pad = 20
        sq_size = max(w, h) + pad * 2
        roi = thresh[y:y+h, x:x+w]
        digit_canvas = np.zeros((sq_size, sq_size), dtype="uint8")
        digit_canvas[pad:pad+h, pad:pad+w] = roi
        final_img = cv2.resize(digit_canvas, (28, 28), interpolation=cv2.INTER_AREA)
        
        input_data = final_img.astype('float32') / 255.0
        input_data = np.expand_dims(input_data, axis=(0, -1))
        prediction = model.predict(input_data, verbose=0)
        results.append(str(np.argmax(prediction)))
        roi_images.append(final_img)
        
    return results, roi_images

# 5. 模式切換邏輯 (保持原本介面)
if option == "手寫畫板模式":
    st.write("### ✍️ 請在黑色畫板內寫入數字：")
    
    # [精確插入] 在畫板上方加入一個小空白，讓手指不要太靠近頂端
    st.write("") 
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",
        stroke_width=15, # 保持原本寬度
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=700,
        height=300,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if canvas_result.image_data is not None:
        img_raw = canvas_result.image_data.astype('uint8')
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_RGBA2GRAY)
        
        if st.button("🚀 進行 AI 辨識"):
            res, imgs = process_and_predict(img_gray, is_canvas=True)
            if res:
                st.success(f"## 最終辨識結果： {''.join(res)}")
                cols = st.columns(len(imgs))
                for i, im in enumerate(imgs):
                    cols[i].image(im, caption=f"預測: {res[i]}")
            else:
                st.warning("請在畫板上書寫數字後再進行辨識。")

elif option == "使用相機拍照" or option == "上傳圖片檔":
    img_file = st.camera_input("📸 立即拍攝數字") if option == "使用相機拍照" else st.file_uploader("📁 上傳圖片檔案", type=["jpg", "png", "jpeg"])
    
    if img_file:
        image = Image.open(img_file)
        img_array = np.array(image.convert('RGB'))
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        st.write("### 🖼️ 處理細節：")
        st.image(image, width=400)
        
        res, imgs = process_and_predict(img_gray)
        if res:
            st.divider()
            st.success(f"## 🔢 最終辨識結果： {''.join(res)}")
            cols = st.columns(min(len(imgs), 10))
            for i, im in enumerate(imgs):
                with cols[i]:
                    st.image(im, caption=f"預測: {res[i]}")
        else:
            st.warning("偵測不到數字，請試著調整側面板參數。")