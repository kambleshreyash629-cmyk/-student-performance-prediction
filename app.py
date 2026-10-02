import streamlit as st
import pandas as pd
import joblib


# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load("models/student_performance_model.pkl")


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================
st.title("🎓 Student Performance Predictor")

st.write(
    "Enter the student's information below to estimate "
    "the final mathematics grade (G3)."
)

st.divider()


# =========================================================
# 1. PERSONAL INFORMATION
# =========================================================
st.header("👤 Personal Information")

col1, col2, col3 = st.columns(3)

with col1:
    school = st.selectbox(
        "School",
        ["GP", "MS"]
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["F", "M"]
    )

with col3:
    age = st.number_input(
        "Age",
        min_value=15,
        max_value=22,
        value=17
    )


col1, col2, col3 = st.columns(3)

with col1:
    address = st.selectbox(
        "Address",
        ["U", "R"]
    )

with col2:
    famsize = st.selectbox(
        "Family Size",
        ["GT3", "LE3"]
    )

with col3:
    Pstatus = st.selectbox(
        "Parent Status",
        ["A", "T"]
    )


# =========================================================
# 2. FAMILY & EDUCATION
# =========================================================
st.header("👨‍👩‍👦 Family & Education")

col1, col2, col3 = st.columns(3)

with col1:
    Medu = st.slider(
        "Mother Education",
        0, 4, 2
    )

with col2:
    Fedu = st.slider(
        "Father Education",
        0, 4, 2
    )

with col3:
    guardian = st.selectbox(
        "Guardian",
        ["mother", "father", "other"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    Mjob = st.selectbox(
        "Mother Job",
        ["teacher", "health", "services", "at_home", "other"]
    )

with col2:
    Fjob = st.selectbox(
        "Father Job",
        ["teacher", "health", "services", "at_home", "other"]
    )

with col3:
    famrel = st.slider(
        "Family Relationship Quality",
        1, 5, 4
    )


col1, col2, col3 = st.columns(3)

with col1:
    famsup = st.selectbox(
        "Family Support",
        ["yes", "no"]
    )

with col2:
    schoolsup = st.selectbox(
        "School Support",
        ["yes", "no"]
    )

with col3:
    nursery = st.selectbox(
        "Attended Nursery",
        ["yes", "no"]
    )


# =========================================================
# 3. ACADEMIC & STUDY INFORMATION
# =========================================================
st.header("📚 Academic & Study Information")

col1, col2, col3 = st.columns(3)

with col1:
    studytime = st.slider(
        "Study Time",
        1, 4, 2
    )

with col2:
    failures = st.slider(
        "Past Class Failures",
        0, 4, 0
    )

with col3:
    traveltime = st.slider(
        "Travel Time",
        1, 4, 2
    )


col1, col2, col3 = st.columns(3)

with col1:
    paid = st.selectbox(
        "Extra Paid Classes",
        ["yes", "no"]
    )

with col2:
    higher = st.selectbox(
        "Wants Higher Education",
        ["yes", "no"]
    )

with col3:
    reason = st.selectbox(
        "Reason for Choosing School",
        ["home", "reputation", "course", "other"]
    )


# =========================================================
# 4. ACTIVITIES & LIFESTYLE
# =========================================================
st.header("❤️ Lifestyle & Activities")

col1, col2, col3 = st.columns(3)

with col1:
    activities = st.selectbox(
        "Extra Activities",
        ["yes", "no"]
    )

with col2:
    internet = st.selectbox(
        "Internet Access",
        ["yes", "no"]
    )

with col3:
    romantic = st.selectbox(
        "Romantic Relationship",
        ["yes", "no"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    freetime = st.slider(
        "Free Time",
        1, 5, 3
    )

with col2:
    goout = st.slider(
        "Going Out",
        1, 5, 3
    )

with col3:
    health = st.slider(
        "Health",
        1, 5, 4
    )


col1, col2, col3 = st.columns(3)

with col1:
    Dalc = st.slider(
        "Workday Alcohol Consumption",
        1, 5, 1
    )

with col2:
    Walc = st.slider(
        "Weekend Alcohol Consumption",
        1, 5, 1
    )

with col3:
    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=4
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================
st.divider()

predict_button = st.button(
    "🔮 Predict Final Grade",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================
if predict_button:

    input_data = pd.DataFrame([{
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences
    }])

    prediction = model.predict(input_data)[0]

    # Limit display to the model's target range
    prediction = max(0, min(20, prediction))

    st.subheader("📊 Prediction Result")

    st.success(
        f"### Predicted Final Grade: {prediction:.2f} / 20"
    )

    # Simple interpretation
    if prediction < 10:
        st.warning("The predicted score is below 10/20.")
    elif prediction < 15:
        st.info("The predicted score is in the middle range.")
    else:
        st.success("The predicted score is in the higher range.")

    st.caption(
        "This is a machine-learning estimate based on the training "
        "data and is not a guaranteed result."
    )


# =========================================================
# ABOUT THE MODEL
# =========================================================
st.divider()

with st.expander("ℹ️ About This Model"):

    st.write("**Dataset:** UCI Student Performance")
    st.write("**Students:** 395")
    st.write("**Input Features:** 30")
    st.write("**Target:** G3 (Final Mathematics Grade)")
    st.write("**Model:** Random Forest Regressor")
    st.write("**Task:** Regression")

    st.write(
        "The model was trained using student background, "
        "education, study habits, activities and lifestyle-related features."
    )