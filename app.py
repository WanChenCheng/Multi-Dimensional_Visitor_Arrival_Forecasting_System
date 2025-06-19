import streamlit as st

# Important: st.set_page_config must come before other Streamlit commands
st.set_page_config(
    page_title="國泰人壽 - 用戶行為預測工具", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)


import pandas as pd
import numpy as np
import base64
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Function to load image as Base64
@st.cache_data
def get_base64_image(path: str) -> str:
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load and embed logo
img_base64 = get_base64_image("logo.png")

# Custom CSS
theme_css = """
<style>
  body { background-color: #e9f7f5; }
  .stApp { background-color: #e9f7f5; color: #333333; }

  /* Hide default Streamlit header/menu */
  header, footer { visibility: hidden; height: 0; margin: 0; padding: 0; }

  /* Multiselect tag pills */
  div[data-baseweb="tag"] {
      background-color: #4fc08d !important;
  }
  div[data-baseweb="tag"] > span:first-child,
  div[data-baseweb="tag"] svg {
      color: #ffffff !important;
  }

  /* Dropdown selected options */
  div[data-baseweb="option"][aria-selected="true"] {
      background-color: #4fc08d !important;
      color: #ffffff !important;
  }

  /* File uploader hover */
  input[type="file"]::file-selector-button:hover {
      background-color: #ffffff !important;
      color: #4fc08d !important;
      border: 1px solid #4fc08d !important;
  }

  /* Button hover & active */
  div[data-testid="stButton"] > button:hover {
      background-color: #ffffff !important;
      color: #4fc08d !important;
      border: 1px solid #4fc08d !important;
  }
  div[data-testid="stButton"] > button:active {
      background-color: #4fc08d !important;
      color: #ffffff !important;
  }

  /* Form and expander */
  div[data-testid="stForm"],
  div[data-testid="stExpander"] > div:first-child {
      background-color: #f8fffc !important;
      border-radius: 12px;
      border: 1px solid #cceee4;
      padding: 16px;
  }
  div[data-testid="stExpander"] > summary {
      font-size: 1.15rem !important;
      font-weight: 600;
      color: #2a6154;
  }
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# Single consolidated header with larger logo
st.markdown(
    f"""
    <div style="display:flex;align-items:center;margin-bottom:20px;">
        <img src="data:image/png;base64,{img_base64}" alt="logo" style="width:80px;height:auto;margin-right:12px;" />
        <h1 style="margin:0;color:#1f3f3e;font-size:2.5rem;">國泰人壽 - 多元訪客進站行為預測工具</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# 嘗試載入 TensorFlow 和相關套件
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from joblib import load
    import joblib
    TF_AVAILABLE = True
except ImportError as e:
    st.error(f"載入依賴套件時發生錯誤: {e}")
    TF_AVAILABLE = False

# 初始化 session state
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "all_columns" not in st.session_state:
    st.session_state.all_columns = []
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "raw_uploaded_data" not in st.session_state:
    st.session_state.raw_uploaded_data = None
if "start_date" not in st.session_state:
    st.session_state.start_date = None
if "end_date" not in st.session_state:
    st.session_state.end_date = None


# 模型與編碼器參數
SEQ_LEN = 10
cat_features = ['action_group', 'source', 'medium', 'platform']
num_features = ['staytime', 'has_shared', 'revisit_count']

# === 模型與編碼器載入封裝 ===
@st.cache_resource
def load_model_with_log():
    import os
    import joblib
    from tensorflow.keras.models import load_model

    log_lines = []
    log_lines.append("開始載入模型與編碼器...")

    # 模型
    model_file = "lstm_multi_output_model_v2.h5"
    model = load_model(model_file)
    model_size = os.path.getsize(model_file) / (1024 * 1024)
    log_lines.append(f"模型成功載入（{model_size:.2f} MB）")

    # 類別編碼器
    encoders = {}
    for col in cat_features:
        path = f"encoder_{col}.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 不存在")
        encoder = joblib.load(path)
        encoders[col] = encoder
        log_lines.append(f"encoder_{col} 載入成功（類別數: {len(encoder.classes_)}）")

    # 數值 scaler
    scalers = {}
    for col in num_features:
        path = f"scaler_feature_{col}.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 不存在")
        scaler = joblib.load(path)
        scalers[col] = scaler
        log_lines.append(f"scaler_{col} 載入成功")

    log_lines.append("所有模型與編碼器皆成功載入")
    return model, encoders, scalers, log_lines

# === 載入模型與初始化 ===
if TF_AVAILABLE:
    with st.spinner("正在初始化模型與編碼器..."):
        model, encoders, scalers, load_log = load_model_with_log()

    st.success("模型與特徵編碼器已成功載入")

    with st.expander("查看詳細載入記錄", expanded=False):
        for line in load_log:
            st.markdown(f"- {line}")
else:
    st.error("無法載入模型，請確認 TensorFlow 安裝狀態")
        

# 1. 定義等級映射函式
def get_level(action_group: str) -> float:
    # 檢查輸入是否為有效值
    if action_group is None or pd.isna(action_group) or action_group == '':
        return -1
        
    # 轉換為字符串以確保一致性
    action_group = str(action_group).strip()
    
    # 等級 0
    if action_group in {"其他", "保險視圖、保單明細、資產總覽、保險明細"}:
        return 0
    # 等級 1
    if action_group == "找服務（尋求服務與客服）":
        return 1
    # 等級 2
    if "商品資訊頁" in action_group or action_group == "好康優惠":
        return 2
    # 等級 3
    combo3 = {
        "自由配－投資規劃", "自由配", "訂製保險組合", "自由配－套餐", "自由配－保障規劃"
    }
    if action_group in combo3 or action_group.startswith("試算"):
        return 3
    # 等級 4
    result4 = {
        "自由配－保障規劃試算結果",
        "訂製保險組合－人身規劃試算結果",
        "我的保險試算結果",
        "訂製保險組合－投資規劃試算結果",
        "自由配－配置我的基金試算結果"
    }
    if action_group in result4:
        return 4
    # 等級 5
    if action_group in {
        "保存與分享試算結果",
        "保存與分享自由配、訂製組合結果",
        "Line分享轉傳"
    }:
        return 5
    # 等級 6
    if action_group in {
        "挑選預約顧問",
        "預約顧問與商品諮詢",
        "立即投保"
    }:
        return 6
    # 等級 7.x
    if action_group in {"方案確認", "填寫預約資料"}:
        return 7.1
    if action_group in {"資料填寫與確認", "投保資格確認", "手機驗證碼"}:
        return 7.2
    if action_group == "線上繳費":
        return 7.3
    # 等級 8
    if action_group in {"完成網路投保", "完成O2O"}:
        return 8
    # 未知
    return -1

# 2. 推薦策略函式（使用 next_action_group 參數）
def recommend_strategy(
        next_action_group: str,
        history: list[str],
        last_event_time: datetime
    ) -> str | None:

    # 優先條件：若過去10步已完成轉換，則暫時不推薦行銷建議
    if any("完成網路投保" in (ag or "") for ag in history[:10]):
        return "此用戶於期間內已完成網路投保"
    if any("完成O2O" in (ag or "") for ag in history[:10]):
        return "此用戶於期間內已完成預約O2O"
    
    # 檢查 next_action_group 是否為 None 或空值
    if next_action_group is None or pd.isna(next_action_group) or next_action_group == '':
        return None
        
    # 處理時區問題：統一轉換為 naive datetime
    now = datetime.now()
    if hasattr(last_event_time, 'tz_localize'):
        # 如果是 pandas Timestamp 且有時區信息，轉換為 naive
        if last_event_time.tz is not None:
            last_event_time = last_event_time.tz_localize(None)
    elif hasattr(last_event_time, 'tzinfo'):
        # 如果是 datetime 且有時區信息，轉換為 naive
        if last_event_time.tzinfo is not None:
            last_event_time = last_event_time.replace(tzinfo=None)

    # 等級 0
    if next_action_group in {"其他", "保險視圖、保單明細、資產總覽、保險明細"}:
        return "暫無建議，持續觀察"

    # 等級 1：找服務
    if next_action_group == "找服務（尋求服務與客服）":
        count = history[:10].count("找服務（尋求服務與客服）")
        if count == 0:
            return "畫面上方箭頭提示客服選單位置"
        elif count > 3:
            return "讓右下角的阿發的「Hi 需要幫忙嗎」unhide"
        else:
            return None

    # 等級 2：商品資訊頁
    if "商品資訊頁" in next_action_group:
        last_action = history[0]  # 最後一步（last_action_group）
        # 如果最後一步不包含特定關鍵字，才考慮推動試算
        if all(kw not in (last_action or "") for kw in ["找服務", "試算", "自由配", "訂製保險組合"]):
            cnt = sum(1 for ag in history[:10] if ag and "商品資訊頁" in ag)
            if cnt >= 3:
                return "顯示試算入口，推動試算"
    
        return None

    # 等級 2：好康優惠
    if next_action_group == "好康優惠":
        last_action = history[0]  # 最後一個 action_group
        if last_action != "好康優惠":
            return "彈窗顯示好康優惠相關資訊"

    # 等級 3：挑選組合
    combo3 = {"自由配－投資規劃", "自由配", "訂製保險組合", "自由配－套餐", "自由配－保障規劃"}
    if next_action_group in combo3:
        # 取出最近三個行為
        recent_actions = history[:3]
        # 如果都沒有包含「自由配」或「訂製保險組合」，才顯示彈窗
        if all(
            not any(kw in (ag or "") for kw in ["自由配", "訂製保險組合"])
            for ag in recent_actions
        ):
            prompt = (
                "不知道如何開始嗎？簡單三個問題，讓系統提供最適合你的商品！"
                if next_action_group == "訂製保險組合"
                else "一鍵搭配個人化的商品組合，只要 2 分鐘！"
            )
            return f"彈窗：「{prompt}」"
            
    # 等級 3：試算
    if next_action_group.startswith("試算"):
        last_ag = history[0] or ""
        if not any(keyword in last_ag for keyword in ["找服務", "試算", "自由配", "訂製保險組合"]):
            return "彈窗：「一鍵帶你完成試算，只要 2 步驟，取得商品費用估算」"

    # 等級 4：試算結果
    result4 = {
        "自由配－保障規劃試算結果",
        "訂製保險組合－人身規劃試算結果",
        "我的保險試算結果",
        "訂製保險組合－投資規劃試算結果",
        "自由配－配置我的基金試算結果"
    }
    if next_action_group in result4:
        last_ag = history[0] or ""
        if any(keyword in last_ag for keyword in ["試算", "自由配", "訂製保險組合"]):
            return "提供進度提醒：就快完成了！試算結果即將產生"

    # 等級 5：保存／分享結果
    if next_action_group in {
        "保存與分享試算結果",
        "保存與分享自由配、訂製組合結果"
    }:
        if any("結果" in (ag or "") for ag in history[:3]):
            return "諮詢按鈕提示：「對試算結果有疑問嗎？預約免費專人解讀！」"
    
    if next_action_group == "Line分享轉傳":
        if any("結果" in (ag or "") for ag in history[:3]):
            return "彈窗鼓勵推薦、推播推薦獎勵機制或分享回饋活動"

    # 等級 6：挑顧問 / 諮詢需求
    if next_action_group in {"挑選預約顧問", "預約顧問與商品諮詢"}:
        return "彈窗推薦顧問：「這位專家最擅長XX險，3 分鐘內確認預約」"
    if next_action_group == "立即投保":
        return "立即投保按鈕或橫幅CTA：「立即投保享優惠！」"

    # 等級 7.1：方案確認 / 填寫預約資料
    if next_action_group in {"方案確認", "填寫預約資料"}:
        if now - last_event_time > timedelta(minutes=30):
            return "發EDM提醒完成投保或預約：「上次還沒填完？點此一鍵回到流程」"
        if now - last_event_time <= timedelta(minutes=5):
            return (
                "進度提醒：「再三步即可完成投保」"
                if next_action_group == "方案確認"
                else "進度提醒：「再兩步即可完成預約」"
            )

    # 等級 7.2：資料填寫與確認 / 投保資格確認 / 手機驗證碼
    if next_action_group in {"資料填寫與確認", "投保資格確認", "手機驗證碼"}:
        if now - last_event_time > timedelta(minutes=30):
            return "發EDM提醒完成投保或預約：「上次還沒填完？點此一鍵回到流程」"
        if now - last_event_time <= timedelta(minutes=5):
            return (
                "進度提醒：「還差最後一步就OK，即將完成預約」"
                if next_action_group == "手機驗證碼"
                else "進度提醒：「還差最後兩步就OK，即將完成投保」"
            )

    # 等級 7.3：線上繳費
    if next_action_group == "線上繳費":
        if now - last_event_time > timedelta(minutes=30):
            return "寄發EDM提醒完成投保或預約：「上次還沒填完？點此一鍵回到流程」"
        if now - last_event_time <= timedelta(minutes=5):
            return "進度提醒：「還差最後一步就OK，即將完成投保」"

    # 等級 8：完成網投 / 完成O2O
    if next_action_group == "完成網路投保":
        if now - last_event_time > timedelta(minutes=30):
            return "寄發EDM提醒即將完成投保，並附上連結「還差最後一步就OK，點我回到流程」"
    if now - last_event_time <= timedelta(minutes=5):
            return "進度提醒：「還差最後一步就OK，即將完成投保」"
        
    if next_action_group == "完成O2O":
        if now - last_event_time > timedelta(minutes=30):
            return "寄發EDM提醒即將完成預約，並附上連結「還差最後一步就OK，點我回到流程」"
        if now - last_event_time <= timedelta(minutes=5):
            return "進度提醒：「還差最後一步就OK，即將完成投保」"

    return None

# 3. 根據 Top1~Top5 選出 next_action_group
def pick_next_action_group(row) -> str:
    candidates = []
    for i in range(1, 6):
        ag = row[f'Top{i}_next_action_group']
        conf = row[f'Top{i}_confidence']
        
        # 檢查 action_group 是否為有效值
        if ag is None or pd.isna(ag) or ag == '':
            lvl = -1  # 給無效值最低等級
        else:
            lvl = get_level(ag)
            
        candidates.append((ag, conf, lvl))
    
    # 先比等級，再比機率，過濾掉無效值
    valid_candidates = [(ag, conf, lvl) for ag, conf, lvl in candidates if lvl >= 0]
    
    if valid_candidates:
        best = max(valid_candidates, key=lambda x: (x[2], x[1]))
        return best[0]
    else:
        # 如果沒有有效候選，返回第一個非空值或默認值
        for ag, conf, lvl in candidates:
            if ag is not None and not pd.isna(ag) and ag != '':
                return ag
        return "其他"  # 默認值

def safe_label_transform(encoder, value, default=0):
    """
    安全地進行 label encoding，確保不會產生負值
    """
    try:
        # 處理 NaN, None, 空字符串
        if pd.isna(value) or value is None or value == '' or value == 'nan':
            return default
        
        # 轉換為字符串（保持與訓練時一致）
        str_value = str(value).strip()
        
        # 如果是空字符串，返回默認值
        if str_value == '' or str_value == 'nan':
            return default
        
        # 檢查是否在已知類別中
        if str_value in encoder.classes_:
            result = encoder.transform([str_value])[0]
            # 確保結果不是負數
            if result < 0:
                print(f"警告：編碼結果為負數 {result}，使用默認值 {default}")
                return default
            return result
        else:
            # 未知類別，返回默認值
            print(f"警告：未知類別 '{str_value}' (原值: {value})，使用默認值 {default}")
            return default
            
    except Exception as e:
        print(f"Label encoding 錯誤 (值: {value}): {e}，使用默認值 {default}")
        return default

def safe_scale(scaler, value, default=0.0):
    """
    安全地進行數值標準化
    """
    try:
        # 處理 NaN, None
        if pd.isna(value) or value is None:
            return default
        
        # 轉換為浮點數
        float_value = float(value)
        
        # 檢查是否為有限數
        if not np.isfinite(float_value):
            return default
        
        # 進行標準化
        scaled_value = scaler.transform([[float_value]])[0][0]
        
        # 檢查結果是否有效
        if not np.isfinite(scaled_value):
            return default
        
        return scaled_value
        
    except Exception as e:
        print(f"Scaling 錯誤 (值: {value}): {e}，使用默認值 {default}")
        return default

def validate_and_fix_sequence(seq, seq_name):
    """
    驗證並修復序列中的無效值
    """
    seq = np.array(seq)
    
    # 檢查並修復負值
    negative_mask = seq < 0
    if np.any(negative_mask):
        print(f"警告：{seq_name} 中發現 {np.sum(negative_mask)} 個負值，已修復為 0")
        seq[negative_mask] = 0
    
    # 檢查並修復 NaN 值
    nan_mask = ~np.isfinite(seq)
    if np.any(nan_mask):
        print(f"警告：{seq_name} 中發現 {np.sum(nan_mask)} 個無效值，已修復為 0")
        seq[nan_mask] = 0
    
    return seq

def predict_from_uploaded_csv(df):
    if model is None or encoders is None or scalers is None:
        st.error("模型未正確載入，無法進行預測")
        return pd.DataFrame()
    
    # 檢查必要欄位
    required_columns = ['user_pseudo_id', 'event_time', 'platform', 'action', 'action_group'] \
                       + cat_features + num_features
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"缺少必要欄位: {missing_columns}")
        return pd.DataFrame()
    
    # 資料預處理
    df = df.copy()
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # 統一處理時區：如果有時區信息，轉換為 naive datetime
    if df['event_time'].dt.tz is not None:
        df['event_time'] = df['event_time'].dt.tz_localize(None)
    
    df = df.sort_values(by=["user_pseudo_id", "event_time"])
    
    results = []
    
    for user_id, group in df.groupby("user_pseudo_id"):
        try:
            # 取最後 SEQ_LEN 步
            last_steps = group.tail(SEQ_LEN)
            pad_len = SEQ_LEN - len(last_steps)
            
            # --- 擷取 raw features ---
            last = last_steps.iloc[-1]
            raw_last_event_time   = last['event_time']
            raw_last_platform     = last['platform']
            raw_last_action       = last['action']
            raw_last_action_group = last['action_group']
            
            # 取前 2~9 步的 raw action & action_group
            prev_records = {}
            for i in range(2, 11):
                if len(last_steps) >= i:
                    rec = last_steps.iloc[-i]
                    prev_records[f"-{i}_action"]        = rec['action']
                    prev_records[f"-{i}_action_group"]  = rec['action_group']
                else:
                    prev_records[f"-{i}_action"]        = None
                    prev_records[f"-{i}_action_group"]  = None
            
            # --- 類別特徵編碼 & 填充 ---
            cat_inputs = []
            for col in cat_features:
                raw_vals = last_steps[col].tolist()
                encoded = [ safe_label_transform(encoders[col], v, default=0)
                            for v in raw_vals ]
                padded = [0]*pad_len + encoded
                seq = validate_and_fix_sequence(padded, f"{col}_sequence")
                cat_inputs.append(np.array(seq, dtype=np.int32).reshape(1, SEQ_LEN))
            
            # --- 數值特徵標準化 & 填充 ---
            num_inputs = []
            for col in num_features:
                raw_vals = last_steps[col].tolist()
                scaled = [ safe_scale(scalers[col], v, default=0.0)
                           for v in raw_vals ]
                padded = [0.0]*pad_len + scaled
                seq = validate_and_fix_sequence(padded, f"{col}_sequence")
                num_inputs.append(np.array(seq, dtype=np.float32).reshape(1, SEQ_LEN))
            
            all_inputs = cat_inputs + num_inputs

            # 進行預測並取 Top5
            y_pred_action_group, y_pred_online, y_pred_o2o = model.predict(all_inputs, verbose=0)
            top5_idx   = y_pred_action_group[0].argsort()[-5:][::-1]
            top5_confs = y_pred_action_group[0][top5_idx]
            inv_ag     = {i: v for i, v in enumerate(encoders['action_group'].classes_)}
            top5_actions = [inv_ag.get(idx, "未知") for idx in top5_idx]
            
            # 正確地 build 中介 dict，注意 for 迴圈後要有冒號
            temp = {}
            for i in range(5):
                temp[f"Top{i+1}_next_action_group"] = top5_actions[i]
                temp[f"Top{i+1}_confidence"]       = round(float(top5_confs[i]), 4)
                
            # 再把這個 dict 傳給 pick_next_action_group
            next_ag = pick_next_action_group(temp)
            
            # 計算行銷策略
            strategy = recommend_strategy(
                next_ag,
                [last['action_group']] + [ prev_records.get(f"-{i}_action_group") for i in range(1, 11) ],
                raw_last_event_time
            )
            
            # --- 組裝結果 ---
            result = {
                # 基本身份信息
                "user_pseudo_id": user_id,
                
                # 完整Top5預測 (最重要，放在前面)
                "Top1_next_action_group": top5_actions[0],
                "Top1_confidence": round(float(top5_confs[0]), 4),
                "Top2_next_action_group": top5_actions[1],
                "Top2_confidence": round(float(top5_confs[1]), 4),
                "Top3_next_action_group": top5_actions[2],
                "Top3_confidence": round(float(top5_confs[2]), 4),
                "Top4_next_action_group": top5_actions[3],
                "Top4_confidence": round(float(top5_confs[3]), 4),
                "Top5_next_action_group": top5_actions[4],
                "Top5_confidence": round(float(top5_confs[4]), 4),
                
                # 轉換機率和策略
                "Online_conversion_prob": round(float(y_pred_online[0][0]), 4),
                "O2O_reservation_prob": round(float(y_pred_o2o[0][0]), 4),
                "Marketing_Strategy": strategy,
                
                # 當前行為信息
                "last_event_time": raw_last_event_time,
                "last_platform": raw_last_platform,
                "last_action": raw_last_action,
                "last_action_group": raw_last_action_group,
                
                # 歷史行為記錄
                **prev_records
            }
            
            results.append(result)
        
        except Exception as e:
            st.error(f"處理用戶 {user_id} 時發生錯誤: {e}")
            continue
    
    return pd.DataFrame(results)

# UI 介面
# streamlit run app.py

# =========================
# 頁面定義與初始化
# =========================
pages = [
    "1. 檔案上傳與分析期間篩選",
    "2. 開始預測",
    "3. 預測結果篩選與下載",
    "4. 統計圖表分析"
]

# 初始化頁面 index（只跑一次）
if "current_page_index" not in st.session_state:
    st.session_state.current_page_index = 0

# 初始化資料狀態（只跑一次）
if "raw_uploaded_data" not in st.session_state:
    st.session_state.raw_uploaded_data = None
if "filtered_input_data" not in st.session_state:
    st.session_state.filtered_input_data = None
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "all_columns" not in st.session_state:
    st.session_state.all_columns = []

# 取得當前頁面名稱與 index
current_index = st.session_state.current_page_index
page = pages[current_index]

# =========================
# 側邊欄樣式與功能選單
# =========================
st.markdown("""<style>
    section[data-testid='stSidebar'] {
        background-color: #d8f3e8;
        padding: 1.5rem 1rem;
    }
    section[data-testid='stSidebar'] h2 {
        font-size: 20px;
        margin-bottom: 1rem;
        color: #2a6154;
    }
    section[data-testid='stSidebar'] div[role='radiogroup'] label {
        margin-bottom: 0.75rem;
        font-size: 16px;
        font-weight: normal;
        color: #333;
        padding-left: 0.2rem;
    }
    section[data-testid='stSidebar'] div[role='radiogroup'] input[type="radio"] {
        display: none;
    }
    section[data-testid='stSidebar'] div[role='radiogroup'] label[data-selected="true"] {
        font-size: 17px;
        font-weight: 700;
        color: #2a6154;
    }
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 功能選單")
    selected = st.radio("功能頁面", pages, index=current_index, label_visibility="collapsed")
    if selected != page:
        st.session_state.current_page_index = pages.index(selected)
        st.rerun()

# =========================
# 導覽按鈕函式（放在頁面底部呼叫）
# =========================
def render_next_page_button():
    col1, col2, col3, col4 = st.columns([1, 3, 3, 1])

    # 上一頁（只有當不是第一頁時才顯示）
    if st.session_state.current_page_index > 0:
        with col1:
            if st.button("← 上一頁", key="prev_page_btn"):
                st.session_state.current_page_index -= 1
                st.rerun()

    # 下一頁（只有當不是最後一頁時才顯示）
    if st.session_state.current_page_index < len(pages) - 1:
        with col4:
            if st.button("下一頁 →", key="next_page_btn"):
                st.session_state.current_page_index += 1
                st.rerun()


# =========================
# 頁面 1: 上傳與篩選
# =========================
if page == "1. 檔案上傳與分析期間篩選":
    st.markdown("### 步驟 1: 上傳檔案")

    uploaded_file = st.file_uploader(
    "上傳包含用戶行為歷程資料的 CSV 文件",
    type=["csv"],
    help="需包含欄位：user_pseudo_id, event_time, action_group, source, medium, platform, staytime, has_shared, revisit_count"
        )

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.session_state.raw_uploaded_data = user_df
            st.success(f"上傳成功，共 {len(user_df)} 筆資料")
    
            required_cols = [
                "user_pseudo_id", "event_time", "action_group", "source",
                "medium", "platform", "staytime", "has_shared", "revisit_count"
            ]
            missing_cols = [col for col in required_cols if col not in user_df.columns]
            if missing_cols:
                st.error(f"缺少欄位：{', '.join(missing_cols)}")
                st.stop()
    
            with st.expander("預覽資料"):
                st.dataframe(user_df.head(10), use_container_width=True)
    
            user_df['event_time'] = pd.to_datetime(user_df['event_time'])
            st.session_state["date_range"] = {
                "min": user_df['event_time'].min().date(),
                "max": user_df['event_time'].max().date()
            }
    
        except Exception as e:
            st.error(f"上傳錯誤：{e}")
            st.session_state.raw_uploaded_data = None
            st.session_state["date_range"] = None
    else:
        st.session_state["date_range"] = None
    
    
    # === 步驟2: 篩選分析期間 ===
    st.markdown("### 步驟 2: 篩選分析期間")
    
    if st.session_state.get("date_range"):
        min_date = st.session_state["date_range"]["min"]
        max_date = st.session_state["date_range"]["max"]
    
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("起始日期", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("截止日期", value=max_date, min_value=min_date, max_value=max_date)
    
        if start_date > end_date:
            st.error("起始日期不能大於截止日期")
            st.session_state.filtered_input_data = None
        else:
            filtered_df = user_df[(user_df['event_time'].dt.date >= start_date) & (user_df['event_time'].dt.date <= end_date)]
            st.session_state.filtered_input_data = filtered_df
            st.info(f"選定期間內資料: {len(filtered_df)} 筆 ({start_date} ~ {end_date})")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.date_input("起始日期", disabled=True)
        with col2:
            st.date_input("截止日期", disabled=True)
        st.info("請先上傳資料後才可選擇分析期間")


    render_next_page_button()
    
    
# =========================
# 頁面 2: 預測與結果總覽
# =========================

elif page == "2. 開始預測":
    st.markdown("### 步驟 3: 執行預測")

    prediction_ready = (
        st.session_state.get("raw_uploaded_data") is not None and
        st.session_state.get("filtered_input_data") is not None
    )

    if prediction_ready:
        if st.session_state.get("prediction_data") is not None:
            st.success("預測結果已生成")
            
        else:
            if st.button("開始預測", use_container_width=True):
                with st.spinner("正在進行模型預測，請稍候..."):
                    try:
                        df_result = predict_from_uploaded_csv(st.session_state.filtered_input_data)
                    except Exception as e:
                        st.error(f"模型預測失敗：{e}")
                        df_result = pd.DataFrame()

                if not df_result.empty:
                    df_result["Marketing_Strategy"] = df_result["Marketing_Strategy"].fillna("暫無建議，持續觀察")
                    st.session_state.prediction_data = df_result
                    st.session_state.all_columns = df_result.columns.tolist()
                    st.success("預測完成")
                    with st.expander("查看預測結果", expanded=False):
                        st.dataframe(df_result.head(), use_container_width=True)
                else:
                    st.error("預測結果為空，請檢查資料格式")
                    st.session_state.prediction_data = None
    else:
        st.button("開始預測", disabled=True, help="請先完成前面步驟")
        st.info("請先完成資料上傳以進行預測")

    # === 預測結果總覽 ===
    st.markdown("### 步驟 4: 預測結果總覽")

    if st.session_state.get("prediction_data") is not None:
        df = st.session_state.prediction_data

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("總用戶數", len(df))
        with col2:
            avg_conf = df['Top1_confidence'].mean()
            avg_conf = 0 if pd.isna(avg_conf) else avg_conf
            st.metric("平均 Top1 信心", f"{avg_conf:.3f}")
        with col3:
            rate_online = (df['Online_conversion_prob'] >= 0.3).mean() * 100
            st.metric("網投機率 ≥0.3 的用戶", f"{int((rate_online / 100) * len(df))} ({rate_online:.1f}%)")
        with col4:
            rate_o2o = (df['O2O_reservation_prob'] >= 0.3).mean() * 100
            st.metric("O2O 機率 ≥0.3 的用戶", f"{int((rate_o2o / 100) * len(df))} ({rate_o2o:.1f}%)")

        with st.expander("查看完整預測結果", expanded=False):
            st.dataframe(df, use_container_width=True)
    else:
        st.info("完成預測後將在此顯示結果總覽")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("總用戶數", "---")
        with col2:
            st.metric("平均 Top1 信心", "---")
        with col3:
            st.metric("網投機率 ≥0.3 的用戶", "---")
        with col4:
            st.metric("O2O 機率 ≥0.3 的用戶", "---")

    render_next_page_button()

# =========================
# 頁面 3: 篩選與下載
# =========================

elif page == "3. 預測結果篩選與下載":
    st.markdown("### 步驟 5: 篩選預測結果")

    if st.session_state.get("prediction_data") is None:
        st.warning("請先完成預測後再執行篩選與下載")
    else:
        df = st.session_state.prediction_data.copy()
        df["Marketing_Strategy"].fillna("暫無建議，持續觀察", inplace=True)

        with st.expander("查看完整預測結果", expanded=False):
            st.dataframe(df, use_container_width=True)

        # 1️⃣ 歷史行為篩選
        st.markdown("**歷史行為篩選**")
        history_steps = st.selectbox("最近Ｎ步內", options=list(range(1, 11)), index=6)
        history_columns = [f"-{i}_action_group" for i in range(2, history_steps + 2)]
        all_history_actions = set()
        for col in history_columns:
            if col in df.columns:
                all_history_actions.update(df[col].dropna().unique())
        all_history_actions = sorted([x for x in all_history_actions if pd.notna(x)])

        selected_history_actions = st.multiselect(
            "曾執行以下行為",
            options=all_history_actions,
            help="選擇用戶在歷史中曾經執行過的行為"
        )
        if selected_history_actions:
            mask = pd.Series(False, index=df.index)
            for col in history_columns:
                if col in df.columns:
                    mask |= df[col].isin(selected_history_actions)
            df = df[mask]

        # 2️⃣ 預測行為篩選
        st.markdown("**預測行為篩選**")
        top_n = st.selectbox("預測下一步的TopＮ中", options=[1, 2, 3, 4, 5], index=0)
        prediction_columns = [f"Top{i}_next_action_group" for i in range(1, 6)]
        all_prediction_actions = set()
        for col in prediction_columns:
            if col in df.columns:
                all_prediction_actions.update(df[col].dropna().unique())
        all_prediction_actions = sorted([x for x in all_prediction_actions if pd.notna(x)])

        selected_prediction_actions = st.multiselect(
            "包含以下行為",
            options=all_prediction_actions,
            help="選擇預測的下一步行為"
        )
        if selected_prediction_actions:
            mask = pd.Series(False, index=df.index)
            for i in range(1, top_n + 1):
                col = f"Top{i}_next_action_group"
                if col in df.columns:
                    mask |= df[col].isin(selected_prediction_actions)
            df = df[mask]

        # 3️⃣ 預測信心門檻
        st.markdown("**預測信心門檻**")
        confidence_option = st.radio(
            "選擇信心門檻策略",
            ["自定義", "保守策略(Top1≥0.4)", "平衡策略(Top1≥0.3)", "積極策略(Top1≥0.2)"],
            help="根據模型準確度：Top1≈70%, Top3≈85%, Top5≈93%"
        )

        if confidence_option == "保守策略(Top1≥0.4)":
            min_confidence = 0.4
            st.info("保守策略：優先選擇高信心預測，降低誤判風險")
        elif confidence_option == "平衡策略(Top1≥0.3)":
            min_confidence = 0.3
            st.info("平衡策略：在準確度和覆蓋率間取得平衡")
        elif confidence_option == "積極策略(Top1≥0.2)":
            min_confidence = 0.2
            st.info("積極策略：最大化觸及用戶數，適合探索性營銷")
        else:
            min_confidence = st.number_input(
                "Top1 最低機率",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.01,
                help="建議值：0.2-0.4 之間，考慮模型準確度平衡"
            )
        df = df[df["Top1_confidence"] >= min_confidence]

        # 4️⃣ 轉換機率篩選
        st.markdown("**轉換機率篩選（任一條件符合即可）**")
        enable_conversion_filter = st.checkbox("啟用轉換機率篩選條件（任一符合）", value=False)

        min_online_conv = 0.0
        min_o2o_conv = 0.0
        if enable_conversion_filter:
            col1, col2 = st.columns(2)
            with col1:
                min_online_conv = st.slider("網路投保機率 ≥", 0.0, 1.0, 0.50, 0.01, format="%.2f")
            with col2:
                min_o2o_conv = st.slider("O2O 預約機率 ≥", 0.0, 1.0, 0.50, 0.01, format="%.2f")

            df = df[
                (df["Online_conversion_prob"] >= min_online_conv) |
                (df["O2O_reservation_prob"] >= min_o2o_conv)
            ]

        # 5️⃣ 行銷策略篩選
        st.markdown("**行銷策略篩選**")
        strategy_options = sorted(df['Marketing_Strategy'].dropna().unique().tolist())
        selected_strategies = st.multiselect(
            "行銷策略篩選",
            options=strategy_options,
            help="選擇想要匯出的行銷策略對象（可複選）"
        )
        if selected_strategies:
            df = df[df["Marketing_Strategy"].isin(selected_strategies)]

        # 6️⃣ 欄位選擇
        st.markdown("**選擇輸出欄位**")
        
        # 提供快速選項
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("全選", key="select_all"):
                st.session_state.selected_columns = st.session_state.all_columns
        with col2:
            if st.button("核心欄位", key="select_core"):
                # 按業務重要性排序的核心欄位
                core_columns = [
                    'user_pseudo_id', 'last_action','last_action_group','last_event_time',
                    'Top1_next_action_group', 'Top1_confidence',
                    'Top2_next_action_group', 'Top2_confidence',
                    'Top3_next_action_group', 'Top3_confidence',
                    'Online_conversion_prob', 'O2O_reservation_prob', 
                    'Marketing_Strategy'
                ]
                st.session_state.selected_columns = [col for col in core_columns if col in st.session_state.all_columns]
        with col3:
            if st.button("預測欄位", key="select_prediction"):
                prediction_cols = [
                    'user_pseudo_id',
                    'Top1_next_action_group', 'Top1_confidence',
                    'Top2_next_action_group', 'Top2_confidence', 
                    'Top3_next_action_group', 'Top3_confidence',
                    'Top4_next_action_group', 'Top4_confidence',
                    'Top5_next_action_group', 'Top5_confidence',
                    'Online_conversion_prob', 'O2O_reservation_prob'
                ]
                st.session_state.selected_columns = [col for col in prediction_cols if col in st.session_state.all_columns]

        # 欄位多選器
        if 'selected_columns' not in st.session_state:
            st.session_state.selected_columns = st.session_state.all_columns
        
        selected_columns = st.multiselect(
            "選擇要輸出的欄位",
            options=st.session_state.all_columns,
            default=st.session_state.selected_columns,
            key="column_selector"
        )
        all_columns = st.session_state.get("all_columns", [])
        
        # 自動更新選中的欄位
        if selected_columns != st.session_state.selected_columns:
            st.session_state.selected_columns = selected_columns
            st.rerun()

        st.session_state["filtered_prediction_data"] = df

        # ==== 步驟 6: 確認條件並下載 ====
        st.markdown("### 步驟 6: 確認條件並下載")

        df = st.session_state.prediction_data.copy()
        filtered_df = st.session_state.get("filtered_prediction_data", df.copy())
        filter_conditions = []
        max_history_steps = 10

        # 條件摘要
        if selected_history_actions:
            filter_conditions.append(f"最近 {history_steps} 步內包含： {'、'.join(selected_history_actions)}")
        if selected_prediction_actions:
            filter_conditions.append(f"Top{top_n} 中包含： {'、'.join(selected_prediction_actions)}")
        if min_confidence > 0:
            filter_conditions.append(f"Top1 信心 ≥ {min_confidence:.2f}")
        if enable_conversion_filter:
            filter_conditions.append(
                f"網投機率 ≥ {min_online_conv:.2f} 或 O2O預約機率 ≥ {min_o2o_conv:.2f}"
            )
        if selected_strategies:
            filter_conditions.append(f"行銷策略為： {'、'.join(selected_strategies)}")
        if selected_columns and len(selected_columns) < len(all_columns):
            filter_conditions.append(f"輸出欄位數量： {len(selected_columns)} / {len(all_columns)} ")

        st.markdown("#### 篩選條件摘要")
        if filter_conditions:
            for condition in filter_conditions:
                st.markdown(f"- {condition}")
        else:
            st.markdown("_未設定任何篩選條件_")

        st.markdown(f"---\n **目前符合條件的用戶數量**：{len(filtered_df)} 人")

        today_str = datetime.now().strftime("%Y%m%d")
        default_filename = f"prediction_result_{len(filtered_df)}users_{today_str}"
        custom_filename = st.text_input(
            " 自訂檔名（選填，系統會自動加上 .csv）",
            value=default_filename,
            placeholder="ex: 旅平險_Top3_信心0.3"
        )

        if len(filtered_df) > 0:
            if st.button(" 確認條件並準備下載"):
                filename = f"{custom_filename}.csv"
                export_cols = st.session_state.get("selected_columns", filtered_df.columns.tolist())
                csv = filtered_df[export_cols].to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label=" 下載結果 CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    key="download_filtered_csv",
                    use_container_width=True
                )
                with st.expander(" 下載內容預覽", expanded=False):
                    st.dataframe(filtered_df[export_cols], use_container_width=True)
        else:
            st.warning("⚠️ 目前條件下沒有符合的用戶，請調整條件後再試")

    
    render_next_page_button()


# =========================
# 頁面 4: 統計圖表分析
# =========================

elif page == "4. 統計圖表分析":
    st.markdown("### 統計圖表分析")

    # 取得原始與篩選後資料
    original_df = st.session_state.get("prediction_data")
    filtered_df = st.session_state.get("filtered_prediction_data")

    # 加入資料來源切換按鈕
    data_source = st.radio(
        "資料來源選擇",
        ["全部預測結果", "頁面三篩選結果"],
        index=1 if filtered_df is not None else 0,
        horizontal=True
    )

    # 根據選擇決定要用哪份資料
    df = filtered_df if data_source == "頁面三篩選結果" and filtered_df is not None else original_df

    if df is not None and not df.empty:
        tab1, tab2, tab3, tab4 = st.tabs(["行為分佈", "信心分數", "轉換分析", "策略分佈"])

        with tab1:
            chart_df = df["Top1_next_action_group"].value_counts().reset_index()
            chart_df.columns = ["action_group", "count"]
            fig1 = px.bar(chart_df, x="action_group", y="count", title="Top1 預測行為分佈")
            fig1.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            fig2 = px.histogram(
                df,
                x="Top1_confidence",
                nbins=20,
                title="Top1 預測信心分數分佈（人數）",
                labels={"Top1_confidence": "Top1 信心分數", "count": "人數"}
            )
            fig2.update_layout(bargap=0)
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.markdown("#### 網路投保轉換機率分佈")
            fig3_online = px.histogram(
                df,
                x="Online_conversion_prob",
                nbins=20,
                title="網路投保轉換機率分佈（人數）",
                labels={"Online_conversion_prob": "網路投保轉換機率", "count": "人數"}
            )
            fig3_online.update_layout(bargap=0)
            st.plotly_chart(fig3_online, use_container_width=True)

            st.markdown("#### O2O 預約轉換機率分佈")
            fig3_o2o = px.histogram(
                df,
                x="O2O_reservation_prob",
                nbins=20,
                title="O2O 預約轉換機率分佈（人數）",
                labels={"O2O_reservation_prob": "O2O 預約機率", "count": "人數"}
            )
            fig3_o2o.update_layout(bargap=0)
            st.plotly_chart(fig3_o2o, use_container_width=True)

        with tab4:
            strategy_df = df["Marketing_Strategy"].fillna("暫無建議，持續觀察").value_counts().reset_index()
            strategy_df.columns = ["strategy", "count"]
            fig4 = px.pie(
                strategy_df,
                names="strategy",
                values="count",
                title="建議行銷策略分佈（含持續觀察）"
            )
            st.plotly_chart(fig4, use_container_width=True)

    else:
        st.info("完成預測後，這裡將顯示詳細的數據分析圖表")
        tab1, tab2, tab3, tab4 = st.tabs(["行為分佈", "信心分數", "轉換分析", "策略分佈"])

        with tab1:
            st.markdown("**Top1 預測行為分佈圖**")
            st.info("將顯示各種預測行為的用戶數量分佈")

        with tab2:
            st.markdown("**Top1 預測信心分數分佈（人數）**")
            st.info("將顯示模型信心分數在各區間的人數分佈")

        with tab3:
            st.markdown("**轉換機率分佈圖（人數）**")
            st.info("將顯示網路投保 與 預約 O2O 的轉換機率與人數分佈情況")

        with tab4:
            st.markdown("**建議行銷策略分佈**")
            st.info("將顯示系統建議的行銷策略類型比例")

    render_next_page_button()

# py -3.10 --version 
# Remove-Item -Recurse -Force .venv
# py -3.10 -m venv .venv
# cd CathayLifeProject
# .venv\Scripts\Activate
# pip install -r requirements.txt
# streamlit run app.py
