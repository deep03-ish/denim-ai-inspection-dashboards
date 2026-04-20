import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Denim AI Inspection", layout="wide")

st.title("Denim AI Inspection Dashboard")
st.caption("Real-time garment defect tracking with ML prediction")

# -------------------------------
# DATA
# -------------------------------
data = [
    {"Garment": "DEN-001", "Zone": "A", "Stage": "Wash", "Part": "Front Panel", "Delta_E": 0.8},
    {"Garment": "DEN-002", "Zone": "A", "Stage": "Wash", "Part": "Back Panel", "Delta_E": 1.9},
    {"Garment": "DEN-003", "Zone": "B", "Stage": "Stitch", "Part": "Pocket", "Delta_E": 2.2},
    {"Garment": "DEN-004", "Zone": "C", "Stage": "Finish", "Part": "Hem", "Delta_E": 0.5},
    {"Garment": "DEN-005", "Zone": "B", "Stage": "Stitch", "Part": "Side Seam", "Delta_E": 1.7},
    {"Garment": "DEN-006", "Zone": "A", "Stage": "Wash", "Part": "Waistband", "Delta_E": 1.1},
    {"Garment": "DEN-007", "Zone": "C", "Stage": "Finish", "Part": "Hem", "Delta_E": 2.8},
    {"Garment": "DEN-008", "Zone": "B", "Stage": "Stitch", "Part": "Pocket", "Delta_E": 0.9},
    {"Garment": "DEN-009", "Zone": "A", "Stage": "Wash", "Part": "Front Panel", "Delta_E": 2.5},
    {"Garment": "DEN-010", "Zone": "C", "Stage": "Finish", "Part": "Label Area", "Delta_E": 1.3},
]

df = pd.DataFrame(data)

# -------------------------------
# DEFECT RULE
# -------------------------------
df["Defect"] = df["Delta_E"].apply(lambda x: 1 if x > 1.5 else 0)

# -------------------------------
# ML MODEL
# -------------------------------
df_encoded = pd.get_dummies(df[["Zone", "Stage", "Part"]])
df_encoded["Delta_E"] = df["Delta_E"]

X = df_encoded
y = df["Defect"]

model = LogisticRegression()
model.fit(X, y)

# -------------------------------
# SIDEBAR (SIMULATION)
# -------------------------------
st.sidebar.header("Live Inspection Input")

zone_input = st.sidebar.selectbox("Zone", ["A", "B", "C"])
stage_input = st.sidebar.selectbox("Stage", ["Wash", "Stitch", "Finish"])
part_input = st.sidebar.selectbox("Garment Part",
    ["Front Panel","Back Panel","Pocket","Hem","Side Seam","Waistband","Label Area"])
delta_input = st.sidebar.slider("Delta E", 0.0, 3.0, 1.0)

# -------------------------------
# INPUT FOR ML
# -------------------------------
input_df = pd.DataFrame({
    "Zone_A":[zone_input=="A"],
    "Zone_B":[zone_input=="B"],
    "Zone_C":[zone_input=="C"],
    "Stage_Wash":[stage_input=="Wash"],
    "Stage_Stitch":[stage_input=="Stitch"],
    "Stage_Finish":[stage_input=="Finish"],
    "Part_Front Panel":[part_input=="Front Panel"],
    "Part_Back Panel":[part_input=="Back Panel"],
    "Part_Pocket":[part_input=="Pocket"],
    "Part_Hem":[part_input=="Hem"],
    "Part_Side Seam":[part_input=="Side Seam"],
    "Part_Waistband":[part_input=="Waistband"],
    "Part_Label Area":[part_input=="Label Area"],
    "Delta_E":[delta_input]
})

input_df = input_df.reindex(columns=X.columns, fill_value=0)

prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

# -------------------------------
# LIVE RESULT
# -------------------------------
st.subheader("Live AI Inspection Result")

c1, c2 = st.columns(2)
c1.metric("Prediction", "DEFECT" if prediction==1 else "OK")
c2.metric("Probability", f"{round(prob*100,2)}%")

# -------------------------------
# ADD SIMULATED DATA (DYNAMIC)
# -------------------------------
simulated_row = {
    "Garment": "LIVE_INPUT",
    "Zone": zone_input,
    "Stage": stage_input,
    "Part": part_input,
    "Delta_E": delta_input,
    "Defect": 1 if delta_input > 1.5 else 0
}

df_dynamic = pd.concat([df, pd.DataFrame([simulated_row])], ignore_index=True)

# -------------------------------
# DYNAMIC PIE CHARTS
# -------------------------------
col1, col2 = st.columns(2)

# Defect %
with col1:
    st.subheader("Defect Percentage (Live)")
    counts = df_dynamic["Defect"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(counts, labels=["OK","Defect"], autopct="%1.1f%%")
    st.pyplot(fig)

# Part Defects
with col2:
    st.subheader("Defects by Garment Part (Live)")
    part_defects = df_dynamic[df_dynamic["Defect"]==1]["Part"].value_counts()

    if not part_defects.empty:
        fig2, ax2 = plt.subplots()
        ax2.pie(part_defects, labels=part_defects.index, autopct="%1.1f%%")
        st.pyplot(fig2)
    else:
        st.info("No defects detected")

# -------------------------------
# ZONE RISK
# -------------------------------
st.subheader("Zone Risk Analysis")
zone_risk = df_dynamic.groupby("Zone")["Defect"].sum()
st.bar_chart(zone_risk)

# -------------------------------
# TABLE
# -------------------------------
st.subheader("Inspection Data")
st.dataframe(df_dynamic, use_container_width=True)

# -------------------------------
# ALERTS
# -------------------------------
st.subheader("Live Alerts")

alerts = df_dynamic[df_dynamic["Defect"]==1]

for _, row in alerts.iterrows():
    st.warning(f"{row['Garment']} → Defect in {row['Part']} (Zone {row['Zone']})")

# -------------------------------
# FOOTER
# -------------------------------
st.write("---")
st.caption("AI-based garment inspection with real-time adaptive visualization")