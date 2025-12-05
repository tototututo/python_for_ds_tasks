import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Завантаження моделі та компонентів
try:
    aussie_rain = joblib.load('models/aussie_rain.joblib')
except FileNotFoundError:
    st.error("Файл 'aussie_rain.joblib' не знайдено.")
    st.stop()

model = aussie_rain['model']
imputer = aussie_rain['imputer']
scaler = aussie_rain['scaler']
encoder = aussie_rain['encoder']
numeric_cols = aussie_rain['numeric_cols']
categorical_cols = aussie_rain['categorical_cols']
encoded_cols = aussie_rain['encoded_cols']

# Препроцесинг та прогнозування
def predict_rain(single_input: dict):
    """Виконує препроцесинг та прогнозування для однієї вхідної точки."""
    
    # 1. Створення DataFrame
    input_df = pd.DataFrame([single_input])
    
    if 'Date' in input_df.columns:
        input_df = input_df.drop(columns=['Date'])
    
    # 2. Імпутація числових значень
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    
    # 3. Масштабування числових значень (MinMaxScaler)
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # 4. Кодування категоріальних ознак (OneHotEncoder)
    encoded_features = encoder.transform(input_df[categorical_cols])
    input_df[encoded_cols] = encoded_features
    
    # 5. Створення фінального вхідного вектору
    X_input = input_df[numeric_cols + encoded_cols]
    
    # 6. Прогнозування
    prediction = model.predict(X_input)[0]
    
    # 7. Ймовірність прогнозу
    prob_index = list(model.classes_).index(prediction)
    prob_all = model.predict_proba(X_input)[0]
    prob_value = prob_all[prob_index]
    
    return prediction, prob_value, prob_all

# Інтерфейс Streamlit

st.set_page_config(page_title="🌧️ Прогнозування дощу в Австралії 🌧️", layout="wide")

st.image('images/landscape.jpg')

st.title("Прогнозування дощу в Австралії")
st.markdown("Введіть поточні погодні умови для прогнозування, чи піде дощ завтра.")

# Константи
TEMP_RANGE = (-10.0, 50.0)
RAIN_RANGE = (0.0, 370.0)
WIND_RANGE = (0.0, 150.0)
PRESSURE_RANGE = (950.0, 1050.0)
HUMIDITY_RANGE = (0.0, 100.0)
CLOUD_RANGE = (0.0, 9.0)

tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Температура/Опади", "💧 Вологість/Тиск", "🌬️ Вітер/Хмарність", "📋 Інше"])

with tab1:
    st.header("Температура та Опади")
    
    locations = sorted([col.split('_')[1] for col in encoded_cols if col.startswith('Location_')])
    location_default = 'Sydney' if 'Sydney' in locations else locations[0]

    input_location = st.selectbox(
        "Місцезнаходження",
        options=locations,
        index=locations.index(location_default)
    )
    
    input_min_temp = st.slider("Мінімальна температура (°C)", min_value=TEMP_RANGE[0], max_value=TEMP_RANGE[1], value=15.0)
    input_max_temp = st.slider("Максимальна температура (°C)", min_value=TEMP_RANGE[0], max_value=TEMP_RANGE[1], value=25.0)
    input_rainfall = st.slider("Кількість опадів (мм)", min_value=RAIN_RANGE[0], max_value=RAIN_RANGE[1], value=0.0)
    input_rain_today = st.selectbox("Сьогодні був дощ?", options=['No', 'Yes'], index=0)

with tab2:
    st.header("Вологість та Тиск")
    
    input_humidity9am = st.slider("Вологість о 9 ранку (%)", min_value=HUMIDITY_RANGE[0], max_value=HUMIDITY_RANGE[1], value=60.0)
    input_humidity3pm = st.slider("Вологість о 3 дня (%)", min_value=HUMIDITY_RANGE[0], max_value=HUMIDITY_RANGE[1], value=40.0)
    input_pressure9am = st.slider("Тиск о 9 ранку (hPa)", min_value=PRESSURE_RANGE[0], max_value=PRESSURE_RANGE[1], value=1015.0)
    input_pressure3pm = st.slider("Тиск о 3 дня (hPa)", min_value=PRESSURE_RANGE[0], max_value=PRESSURE_RANGE[1], value=1012.0)
    
with tab3:
    st.header("Вітер та Хмарність")

    wind_dirs = sorted([col.split('_')[1] for col in encoded_cols if col.startswith('WindGustDir_')])
    wind_dir_default = 'W' if 'W' in wind_dirs else wind_dirs[0]

    input_wind_gust_dir = st.selectbox("Напрямок пориву вітру", options=wind_dirs, index=wind_dirs.index(wind_dir_default))
    input_wind_gust_speed = st.slider("Швидкість пориву вітру (км/год)", min_value=WIND_RANGE[0], max_value=WIND_RANGE[1], value=40.0)

    input_cloud9am = st.slider("Хмарність о 9 ранку", min_value=CLOUD_RANGE[0], max_value=CLOUD_RANGE[1], value=4.0)
    input_cloud3pm = st.slider("Хмарність о 3 дня", min_value=CLOUD_RANGE[0], max_value=CLOUD_RANGE[1], value=4.0)

with tab4:
    input_evaporation = st.number_input("Випаровування (мм)", value=5.0)
    input_sunshine = st.number_input("Сонячне сяйво (години)", value=7.0)
    
    input_wind_speed9am = st.number_input("Швидкість вітру о 9 ранку (км/год)", value=10.0)
    input_wind_speed3pm = st.number_input("Швидкість вітру о 3 дня (км/год)", value=15.0)
    input_wind_dir9am = st.selectbox("Напрямок вітру о 9 ранку", options=wind_dirs, index=wind_dirs.index(wind_dir_default))
    input_wind_dir3pm = st.selectbox("Напрямок вітру о 3 дня", options=wind_dirs, index=wind_dirs.index(wind_dir_default))
    input_temp9am = st.slider("Температура о 9 ранку (°C)", min_value=TEMP_RANGE[0], max_value=TEMP_RANGE[1], value=input_min_temp + 5)
    input_temp3pm = st.slider("Температура о 3 дня (°C)", min_value=TEMP_RANGE[0], max_value=TEMP_RANGE[1], value=input_max_temp - 5)
    
# Кнопка прогнозування
st.markdown("---")
if st.button("Чи буде дощ завтра?"):
    
    # Збір всіх вхідних даних у словник
    user_input = {
        'Date': '2025-01-01',
        'Location': input_location,
        'MinTemp': input_min_temp,
        'MaxTemp': input_max_temp,
        'Rainfall': input_rainfall,
        'Evaporation': input_evaporation if input_evaporation is not None else np.nan,
        'Sunshine': input_sunshine if input_sunshine is not None else np.nan,
        'WindGustDir': input_wind_gust_dir,
        'WindGustSpeed': input_wind_gust_speed,
        'WindDir9am': input_wind_dir9am,
        'WindDir3pm': input_wind_dir3pm,
        'WindSpeed9am': input_wind_speed9am,
        'WindSpeed3pm': input_wind_speed3pm,
        'Humidity9am': input_humidity9am,
        'Humidity3pm': input_humidity3pm,
        'Pressure9am': input_pressure9am,
        'Pressure3pm': input_pressure3pm,
        'Cloud9am': input_cloud9am if input_cloud9am is not None else np.nan,
        'Cloud3pm': input_cloud3pm if input_cloud3pm is not None else np.nan,
        'Temp9am': input_temp9am,
        'Temp3pm': input_temp3pm,
        'RainToday': input_rain_today
    }
    
    # Виконання прогнозування
    prediction, probability, all_probs = predict_rain(user_input)
    
    st.markdown("## ✅ Результат прогнозування")
    
    if prediction == 'Yes':
        status_emoji = "🌧️"
        status_text = "Ймовірно, піде дощ! Парасолька знадобиться!"
        st.balloons() # Невеликий візуальний ефект для "Yes"
        st.success(f"**{status_emoji} Прогноз:** {status_text}")
    
    else:
        status_emoji = "☀️"
        status_text = "Ймовірно, дощу не буде."
        st.info(f"**{status_emoji} Прогноз:** {status_text}")
    
    # Виведення ймовірності
    st.metric(
        label=f"Ймовірність прогнозу ('{prediction}')", 
        value=f"{probability:.2f}",
        delta=f"{(probability * 100):.0f}%"
    )