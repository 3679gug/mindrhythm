import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import random
from datetime import datetime

# --- [1. Page Configuration] ---
st.set_page_config(page_title="MindRhythm", page_icon="🌞", layout="wide")

# --- [2. Backend Functions] ---
@st.cache_resource
def load_model():
 if os.path.exists("model_artifacts.pkl"):
  return joblib.load("model_artifacts.pkl")
 return None

# --- [3. Custom CSS] ---
st.markdown("""
<style>
 .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: 'Noto Sans KR'; }
 .block-container { max-width: 900px; padding: 2rem; margin: auto; background-color: #f8f9fa; border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
 .css-card { background: white; border-radius: 20px; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 20px; }
 .header-area { background: linear-gradient(180deg, #FF6B9D 0%, #FFA07A 100%); padding: 25px; border-radius: 20px; color: white; text-align: center; }
 .garden-plot { background: #e8f5e9; border: 2px dashed #a5d6a7; border-radius: 15px; aspect-ratio: 1; display: flex; align-items: center; justify-content: center; font-size: 2.5rem; }
 .mission-card { background: #fff; padding: 12px; border-radius: 12px; margin-bottom: 8px; border-left: 5px solid #FF6B9D; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
 .status-label { font-weight: bold; padding: 4px 12px; border-radius: 20px; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# --- [4. State Initialization] ---
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'completed_missions' not in st.session_state: st.session_state.completed_missions = [False, False, False]

# --- [5. Sidebar: 사용자 입력 및 피처 매핑] ---
with st.sidebar:
 st.header("👤 내 정보 입력")
 user_age = st.number_input("나이", 1, 100, 25)
 user_gender = st.selectbox("성별", ["여성", "남성"])
 gender_val = 0 if user_gender == "여성" else 1

 st.markdown("---")
 st.subheader("📝 오늘의 활동 설문")
 
 q_lethargy = st.slider("1. 오늘 얼마나 무기력하게 느껴졌나요?", 0, 10, 3)
 q_bed = st.slider("2. 오늘 깨어있는 동안 침대에 누워있던 시간은?", 0, 12, 2, help="낮잠이나 가만히 누워 휴식한 총 시간")
 
 q_energy = st.radio("3. 오늘 나의 활동 패턴은 어떠했나요?", 
  ["매우 정적임 (거의 움직이지 않음)", "평범함 (일상적인 움직임)", "매우 역동적임 (운동이나 활동적 업무)"])
 q_routine = st.select_slider("4. 오늘 생활 리듬이 규칙적이었나요?", options=["불규칙", "보통", "매우 규칙적"])

 base_mean = 380; base_std = 130; base_zero = 0.08; base_auto = 0.6; base_psd = 55000; base_cv = 0.9

 f_mean = base_mean * (1 - (q_lethargy * 0.08))
 f_zero = base_zero + (q_bed * 0.06) 
 
 if q_energy == "매우 정적임 (거의 움직이지 않음)":
  f_std = base_std * 0.4; f_psd = base_psd * 0.2; f_cv = 0.4
 elif q_energy == "매우 역동적임 (운동이나 활동적 업무)":
  f_std = base_std * 1.7; f_psd = base_psd * 2.5; f_cv = 1.8
 else:
  f_std = base_std; f_psd = base_psd; f_cv = base_cv

 f_auto = 0.25 if q_routine == "불규칙" else (0.6 if q_routine == "보통" else 0.85)

 user_features = {
  'mean_act': f_mean, 'std_act': f_std, 'skew_act': 1.6, 'kurt_act': 3.2, 
  'autocorr': f_auto, 'psd_mean': f_psd, 'zero_ratio': f_zero, 'cv_act': f_cv, 
  'age': user_age, 'gender': gender_val
 }
 feat_df = pd.DataFrame([user_features])

# --- [6. Model Inference & MADRS Scoring] ---
artifacts = load_model()
if artifacts:
 model, scaler, feature_names = artifacts['model'], artifacts['scaler'], artifacts['feature_names']
 X_scaled = scaler.transform(feat_df[feature_names])
 prob_depressed = model.predict_proba(X_scaled)[0][1]
 estimated_madrs = prob_depressed * 60
 rhythm_score = int((estimated_madrs / 60) * 100)
 
 if rhythm_score <= 10:
  status, color, desc = "안정 (Normal)", "#4CAF50", "생체 리듬이 매우 안정적입니다. 우울증 징후가 발견되지 않습니다."
 elif rhythm_score <= 32:
  status, color, desc = "경도 의심 (Mild)", "#FBC02D", "활동 리듬이 다소 불규칙합니다. 가벼운 우울감이 의심되는 단계입니다."
 elif rhythm_score <= 57:
  status, color, desc = "중등도 의심 (Moderate)", "#FF9800", "활동 저하가 뚜렷하고 리듬 기복이 큽니다. 우울증이 의심됩니다."
 else:
  status, color, desc = "강력 의심 (Severe)", "#FF5252", "활동 패턴이 무너져 있습니다. 전문가와의 상담이 시급합니다."
else:
 rhythm_score, status, color, desc = 30, "준비중", "#999", "모델 로딩 중..."

# --- [7. Mission Logic] ---
def get_missions(score):
 if score > 57:
  return [("🧘", "심호흡 명상 5분"), ("✍️", "감정 단어 하나 적기"), ("🍵", "따뜻한 물 한 잔 마시기")]
 elif score > 10:
  return [("🚶", "15분 가벼운 산책"), ("🧹", "책상 한 구석 정리"), ("🍎", "비타민/과일 챙겨먹기")]
 else:
  return [("🏃", "30분 활기차게 달리기"), ("📸", "오늘의 예쁜 사진 찍기"), ("💪", "새로운 목표 세우기")]

missions = get_missions(rhythm_score)

# --- [8. UI Rendering] ---
def render_home():
 st.markdown(f'<div class="header-area"><h1>🌞 마인드리듬</h1><p>{user_age}세 {user_gender} 분석 리포트</p></div>', unsafe_allow_html=True)
 
 st.markdown(f"""
 <div class="css-card">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
   <span style="font-size: 1.1rem; font-weight: bold;">마음 리듬 분석 점수 (우울 위험도)</span>
   <span class="status-label" style="background: {color}22; color: {color}; border: 1px solid {color};">{status}</span>
  </div>
  <div style="display: flex; align-items: baseline;">
   <span style="font-size: 2.5rem; font-weight: bold; color: {color};">{rhythm_score}</span>
   <span style="font-size: 1.2rem; color: #888; margin-left: 5px;">/ 100점</span>
  </div>
  <p style="margin-top: 15px; color: #444; line-height: 1.6;"><b>진단 결과:</b> {desc}</p>
  <div style="background: #eee; height: 12px; border-radius: 6px; margin-top: 10px; overflow: hidden;">
   <div style="background: {color}; width: {rhythm_score}%; height: 100%; transition: width 1s;"></div>
  </div>
  <p style="font-size: 0.8rem; color: #999; margin-top: 8px;">*본 점수는 높을수록 우울 위험이 높음을 의미하며, 실제 MADRS 점수(0~60)와 비례합니다.</p>
 </div>
 """, unsafe_allow_html=True)

 col1, col2 = st.columns([1.2, 1])
 
 with col1:
  st.subheader("✨ 오늘의 리듬 미션")
  st.caption("미션을 완료하여 정원에 꽃을 피워주세요.")
  for i, (icon, text) in enumerate(missions):
   c_m1, c_m2 = st.columns([4, 1])
   with c_m1:
    st.markdown(f'<div class="mission-card">{icon} {text}</div>', unsafe_allow_html=True)
   with c_m2:
    if st.checkbox("완료", key=f"mission_{i}", value=st.session_state.completed_missions[i]):
     st.session_state.completed_missions[i] = True
    else:
     st.session_state.completed_missions[i] = False

 with col2:
  st.subheader("🌿 나의 마음 정원")
  complete_count = sum(st.session_state.completed_missions)
  garden_cols = st.columns(3)
  for i in range(3):
   with garden_cols[i % 3]:
    if i < complete_count:
     flower = random.choice(["🌻", "🌷", "🌹", "🌸", "🌺"])
     st.markdown(f'<div class="garden-plot" style="background:#fff9c4; border-color: #ffd54f;">{flower}</div>', unsafe_allow_html=True)
    else:
     st.markdown(f'<div class="garden-plot">🌱</div>', unsafe_allow_html=True)
 
 if complete_count == 3:
  st.balloons()
  st.success("오늘의 모든 리듬 미션을 달성했습니다! 정원이 활짝 피었습니다.")

def render_analysis():
 st.markdown("## 📊 데이터 분석 상세")
 st.write("사용자의 답변이 모델 피처로 변환된 결과입니다.")
 
 # 변수명 한글 매핑
 rename_dict = {
  'mean_act': '평균 활동량',
  'std_act': '활동 변동성',
  'skew_act': '활동 비대칭성',
  'kurt_act': '활동 첨도',
  'autocorr': '자기상관성(규칙성)',
  'psd_mean': '주파수 전력 밀도',
  'zero_ratio': '무활동 비율',
  'cv_act': '전환 변동성',
  'age': '나이',
  'gender': '성별 (0:여, 1:남)'
 }
 kor_feat_df = feat_df.rename(columns=rename_dict)
 st.table(kor_feat_df)
 
 st.info(f"추정 MADRS 지수: {int(estimated_madrs)}점 (리듬 점수와 비례)")
 
 # 점수 영향 요인 설명 추가
 st.markdown("### 🔍 주요 영향 요인 분석")
 factors = []
 if q_bed > 4:
  factors.append("📍 **무활동 비율 상승**: 침대에 누워있는 시간이 길어지면서 생체 리듬의 활력이 크게 저하되었습니다.")
 if q_lethargy > 5:
  factors.append("📍 **평균 활동량 감소**: 주관적인 무기력감이 활동량 수치를 낮추어 우울 위험도를 높이는 주요 원인이 되었습니다.")
 if q_routine == "불규칙":
  factors.append("📍 **규칙성(자기상관성) 저하**: 생활 패턴의 불규칙함이 모델에서 리듬 붕괴의 신호로 감지되었습니다.")
 if q_energy == "매우 정적임 (거의 움직이지 않음)":
  factors.append("📍 **활동 에너지 세기 부족**: 움직임의 강도가 매우 낮아 뇌의 활성 리듬이 둔화된 상태입니다.")
 
 if factors:
  for f in factors:
   st.write(f)
 else:
  st.write("✅ 모든 수치가 안정적입니다. 규칙적인 활동이 유지되고 있습니다.")

# --- [9. Navigation] ---
st.markdown("---")
nav = st.columns(2)
if nav[0].button("🏠 홈 화면", use_container_width=True): st.session_state.page = 'home'; st.rerun()
if nav[1].button("📊 상세 분석", use_container_width=True): st.session_state.page = 'analysis'; st.rerun()

if st.session_state.page == 'home': render_home()
else: render_analysis()