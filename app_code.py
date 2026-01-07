import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 1. 頁面設定
st.set_page_config(page_title="AI手寫辨識APP", layout="wide")

# 初始化統計數據 (Session State)
if 'total_count' not in st.session_state:
    st.session_state.total_count = 0
if 'correct_count' not in st.session_state:
    st.session_state.correct_count = 0

# --- CSS 補強 (防止行動端下拉刷新) ---
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        overscroll-behavior-y: contain !important;
        overflow: hidden !important;
    }
    canvas { touch-action: none !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔢 AI手寫辨識APP")
st.markdown("""
##### 💡 **使用說明：**
1. **多元輸入模式**：支援畫板手寫、拍照、圖片檔案上傳。
2. **最佳辨識建議**：建議使用較粗的筆書寫，並在光源充足的情況下拍攝。
3. **性能優化手腕**：若辨識不佳，可使用側面板微調參數。
4. **手寫注意事項**：手寫部分勿太靠近邊框，以免辨識錯誤。
""")
st.divider()

# 2. 載入模型
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('mnist_model.h5')

try:
    model = load_my_model()
    st.sidebar.success("✅ AI 模型已就緒")
except Exception as e:
    st.sidebar.error(f"❌ 模型載入失敗")

# 3. 側邊欄
st.sidebar.header("🛠️ 系統功能設定")
option = st.sidebar.radio("📸 選擇輸入來源：", ("手寫畫板模式", "使用相機拍照", "上傳圖片檔"))

# 歷史統計顯示
st.sidebar.divider()
st.sidebar.subheader("📊 歷史辨識統計")
if st.session_state.total_count > 0:
    acc = (st.session_state.correct_count / st.session_state.total_count) * 100
    st.sidebar.write(f"總辨識次數: {st.session_state.total_count}")
    st.sidebar.write(f"正確次數: {st.session_state.correct_count}")
    st.sidebar.metric("歷史正確率", f"{acc:.2f}%")
    if st.sidebar.button("🗑️ 刪除統計紀錄"):
        st.session_state.total_count = 0
        st.session_state.correct_count = 0
        st.rerun()
else:
    st.sidebar.write("尚無統計資料")

st.sidebar.divider()
st.sidebar.write("🔍 辨識參數微調 (拍照/上傳專用)")
min_area = st.sidebar.slider("1. 雜訊過濾強度", 100, 1500, 300)
sensitivity = st.sidebar.slider("2. 捕捉靈敏度", 1, 25, 12)
thickness = st.sidebar.slider("3. 字體加粗程度", 1, 5, 2)

# 4. 影像處理函數 (包含 Padding 優化)
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
    
    if not valid_contours:
        return None, None, None

    results, confidences, roi_images = [], [], []
    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = thresh[y:y+h, x:x+w]
        
        # 加大 Padding 確保 1 不會變形
        pad = 30
        digit_canvas = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        final_img = cv2.resize(digit_canvas, (28, 28), interpolation=cv2.INTER_AREA)
        
        input_data = final_img.astype('float32') / 255.0
        input_data = np.expand_dims(input_data, axis=(0, -1))
        prediction = model.predict(input_data, verbose=0)
        
        results.append(np.argmax(prediction))
        confidences.append(np.max(prediction))
        roi_images.append(final_img)
        
    return results, confidences, roi_images

# 5. 模式切換邏輯
if option == "手寫畫板模式":
    st.write("### ✍️ 請在黑色畫板內寫入數字：")
    
    # 移除橡皮擦切換，回歸單一穩定畫板
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=700, height=500,
        drawing_mode="freedraw",
        key="canvas_fixed",
    )
    
    if canvas_result.image_data is not None:
        if st.button("🚀 進行 AI 辨識"):
            img_raw = canvas_result.image_data.astype('uint8')
            img_gray = cv2.cvtColor(img_raw, cv2.COLOR_RGBA2GRAY)
            res, confs, imgs = process_and_predict(img_gray, is_canvas=True)
            if res:
                final_str = ''.join(map(str, res))
                st.session_state['last_res'] = final_str # 暫存結果給回饋區用
                st.success(f"## 最終辨識結果： {final_str}")
                
                cols = st.columns(len(imgs))
                for i, im in enumerate(imgs):
                    with cols[i]:
                        st.image(im, caption=f"預測: {res[i]} ({confs[i]*100:.1f}%)")
            else:
                st.warning("請在畫板上書寫數字。")

    # --- 修正後的報錯回饋區 (放在辨識按鈕外，確保反應靈敏) ---
    if 'last_res' in st.session_state:
        st.divider()
        st.subheader("🚩 辨識回饋與統計")
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            user_val = st.text_input("如果辨識錯誤，請在此輸入正確數值：", value=st.session_state['last_res'])
        with col_btn:
            st.write(" ") # 對齊用
            if st.button("提交回饋紀錄"):
                st.session_state.total_count += 1
                if user_val == st.session_state['last_res']:
                    st.session_state.correct_count += 1
                    st.success("紀錄成功！")
                else:
                    st.warning("已紀錄錯誤。")
                del st.session_state['last_res'] # 清除暫存強制重整
                st.rerun()

elif option == "使用相機拍照" or option == "上傳圖片檔":
    img_file = st.camera_input("📸 立即拍攝數字") if option == "使用相機拍照" else st.file_uploader("📁 上傳圖片檔案", type=["jpg", "png", "jpeg"])
    
    if img_file:
        image = Image.open(img_file)
        img_array = np.array(image.convert('RGB'))
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        st.write("### 🖼️ 處理細節：")
        st.image(image, width=400)
        
        res, confs, imgs = process_and_predict(img_gray)
        if res:
            st.divider()
            st.success(f"## 🔢 最終辨識結果： {''.join(map(str, res))}")
            cols = st.columns(min(len(imgs), 10))
            for i, im in enumerate(imgs):
                with cols[i]:
                    st.image(im, caption=f"預測: {res[i]} ({confs[i]*100:.1f}%)")
        else:
            st.warning("偵測不到數字，請試著調整側面板參數。")