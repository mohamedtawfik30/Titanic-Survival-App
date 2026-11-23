import streamlit as st
import pickle
import numpy as np

# 1. تحميل المودل وأداة التحجيم
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# 2. إعداد واجهة التطبيق
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter the passenger's details to predict whether or not they would survive the Titanic disaster!")

# 3. تصميم المدخلات للمستخدم
st.sidebar.header("بيانات الراكب")

pclass = st.sidebar.selectbox("درجة التذكرة (Pclass)", [1, 2, 3])
sex = st.sidebar.selectbox("الجنس (Sex)", ["ذكر (Male)", "أنثى (Female)"])
age = st.sidebar.slider("العمر (Age)", 0, 80, 25)
sibsp = st.sidebar.number_input("عدد الإخوة / الأزواج على السفينة (SibSp)", 0, 10, 0)
parch = st.sidebar.number_input("عدد الآباء / الأبناء على السفينة (Parch)", 0, 10, 0)
fare = st.sidebar.number_input("سعر التذكرة (Fare)", 0.0, 500.0, 32.2)

# تحويل الجنس لرقم
sex_encoded = 0 if sex == "ذكر (Male)" else 1

# 4. زر التنبؤ
if st.button("توقع النتيجة (Predict)"):
    # بما أن الـ Scaler كان متوقع عمودين فقط (غالباً Age و Fare)، هنقوم بتحجيمهم لوحدهم
    scaled_features = scaler.transform([[age, fare]])
    
    age_scaled = scaled_features[0][0]
    fare_scaled = scaled_features[0][1]
    
    # تجميع الـ 6 أعمدة بالترتيب الصح اللي المودل اتدرب عليه
    # (تأكد إن الترتيب هنا مطابق تماماً لترتيب الأعمدة في X_train عندك)
    user_input = np.array([[pclass, sex_encoded, age_scaled, sibsp, parch, fare_scaled]])
    
    # التنبؤ
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)
    
    # عرض النتيجة
    st.subheader("Result's Predict: ")
    if prediction[0] == 1:
        st.success(f"🎉 Very optimistic! This passenger **survived** the accident (probability of survival:) {prediction_proba[0][1]*100:.2f}%)")
    else:
        st.error(f"❌ Unfortunately, this passenger **did not survive** the accident (probability of death): {prediction_proba[0][0]*100:.2f}%)")