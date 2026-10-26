"""
Campus Facility Reservation — Data Mining Midterm (Streamlit version)

Run with:
    pip install streamlit pandas numpy matplotlib scikit-learn
    streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch
from io import BytesIO
from xml.sax.saxutils import escape

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

st.set_page_config(page_title="Campus Facility Reservation – Data Mining", layout="wide")

BAD_STATUSES = ["cancelled", "no-show", "no show", "noshow"]
REQUIRED_COLUMNS = [
    "Reservation_ID", "Facility_Type", "User_Category", "Reservation_Date",
    "Submission_Date", "Lead_Time_Days", "Participants", "Actual_Attendance",
    "Usage_Status", "Cancellation_Days_Before", "Purpose", "Time_Slot",
]


# ----------------------------------------------------------------------------
# Data generation / cleaning / modeling (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    facility_types = ["Gymnasium", "Auditorium", "Computer Laboratory", "Meeting Room", "Lecture Hall"]
    user_categories = ["Student", "Student Organization", "Faculty/Staff", "External Guest"]
    purposes = ["Meeting", "Practice", "Workshop", "Seminar", "Competition", "Event"]
    time_slots = ["Morning", "Afternoon", "Evening"]

    facility = rng.choice(facility_types, n, p=[0.22, 0.20, 0.23, 0.17, 0.18])
    user = rng.choice(user_categories, n, p=[0.38, 0.27, 0.25, 0.10])
    purpose = rng.choice(purposes, n)
    slot = rng.choice(time_slots, n, p=[0.32, 0.43, 0.25])
    lead = np.clip(rng.gamma(shape=2.2, scale=6.0, size=n).round().astype(int), 0, 60)
    participants = np.clip(rng.gamma(shape=2.3, scale=15, size=n).round().astype(int), 1, 180)

    res_dates = pd.to_datetime("2026-04-01") + pd.to_timedelta(rng.integers(0, 150, n), unit="D")
    sub_dates = res_dates - pd.to_timedelta(lead, unit="D")

    risk = (0.13 + 0.18 * (lead <= 2) + 0.10 * (user == "External Guest")
            + 0.08 * (slot == "Evening") + 0.06 * (participants > 80))
    risk = np.clip(risk, 0.08, 0.68)
    draw = rng.random(n)
    status = np.where(draw < risk * 0.58, "Cancelled", np.where(draw < risk, "No-Show", "Used"))

    attendance, cancel_days = [], []
    for i, s in enumerate(status):
        if s == "Used":
            attendance.append(int(rng.binomial(int(participants[i]), 0.78)))
            cancel_days.append(np.nan)
        elif s == "Cancelled":
            attendance.append(0)
            cancel_days.append(int(rng.integers(0, max(1, min(int(lead[i]) + 1, 15)))))
        else:
            attendance.append(0)
            cancel_days.append(np.nan)

    df = pd.DataFrame({
        "Reservation_ID": [f"RES-{i:04d}" for i in range(1, n + 1)],
        "Facility_Type": facility,
        "User_Category": user,
        "Reservation_Date": res_dates.strftime("%Y-%m-%d"),
        "Submission_Date": sub_dates.strftime("%Y-%m-%d"),
        "Lead_Time_Days": lead,
        "Participants": participants,
        "Actual_Attendance": attendance,
        "Usage_Status": status,
        "Cancellation_Days_Before": cancel_days,
        "Purpose": purpose,
        "Time_Slot": slot,
    })

    # Small amount of realistic missingness for the cleaning tasks
    df.loc[rng.choice(n, size=5, replace=False), "Lead_Time_Days"] = np.nan
    df.loc[rng.choice(n, size=4, replace=False), "Purpose"] = np.nan
    df.loc[rng.choice(n, size=3, replace=False), "Usage_Status"] = np.nan
    return df


@st.cache_data(show_spinner=False)
def clean_data(df_original: pd.DataFrame):
    """Conservative cleaning + derived variables. Returns (df, log dict)."""
    df = df_original.copy(deep=True)
    log = {}

    for col in ["Facility_Type", "User_Category", "Purpose", "Time_Slot", "Usage_Status"]:
        df[col] = df[col].astype("string").str.strip()

    status_map = {"used": "Used", "cancelled": "Cancelled", "canceled": "Cancelled",
                  "no-show": "No-Show", "no show": "No-Show", "noshow": "No-Show"}
    df["Usage_Status"] = df["Usage_Status"].map(
        lambda x: status_map.get(str(x).strip().lower(), x) if pd.notna(x) else pd.NA)

    for col in ["Facility_Type", "User_Category", "Purpose", "Time_Slot"]:
        df[col] = df[col].map(lambda x: str(x).title().strip() if pd.notna(x) else pd.NA).astype("string")

    log["rows_before"] = len(df)
    log["duplicates"] = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    for col in ["Reservation_Date", "Submission_Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["Lead_Time_Days", "Participants", "Actual_Attendance", "Cancellation_Days_Before"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df["Lead_Time_Days"] < 0, "Lead_Time_Days"] = np.nan
    df.loc[df["Participants"] <= 0, "Participants"] = np.nan
    invalid_att = df["Actual_Attendance"].lt(0) | (
        df["Actual_Attendance"].notna() & df["Participants"].notna()
        & df["Actual_Attendance"].gt(df["Participants"]))
    df.loc[invalid_att, "Actual_Attendance"] = np.nan
    df.loc[df["Cancellation_Days_Before"] < 0, "Cancellation_Days_Before"] = np.nan

    date_lead = (df["Reservation_Date"] - df["Submission_Date"]).dt.days
    mismatch = date_lead.notna() & df["Lead_Time_Days"].notna() & date_lead.ne(df["Lead_Time_Days"])
    log["mismatches"] = int(mismatch.sum())
    log["rows_after"] = len(df)

    # Task 5 — derived variables
    df["Lead_Time_Category"] = pd.cut(
        df["Lead_Time_Days"], bins=[-np.inf, 2, 7, 14, np.inf],
        labels=["Last Minute", "Short Lead", "Medium Lead", "Early Booking"])
    df["Reservation_Outcome"] = df["Usage_Status"].map(
        {"Used": "Completed", "Cancelled": "Not Completed", "No-Show": "Not Completed"})
    denominator = df["Participants"].where(df["Participants"] > 0)
    df["Attendance_Ratio"] = df["Actual_Attendance"].div(denominator)
    df.loc[~np.isfinite(df["Attendance_Ratio"]), "Attendance_Ratio"] = np.nan
    return df, log


def is_bad(status: pd.Series) -> pd.Series:
    return status.astype("string").str.lower().isin(BAD_STATUSES)


@st.cache_resource(show_spinner="Training decision tree…")
def train_model(df: pd.DataFrame, seed: int, max_depth: int, min_leaf: int):
    model_df = df.copy()
    for date_col, prefix in [("Reservation_Date", "Reservation"), ("Submission_Date", "Submission")]:
        parsed = pd.to_datetime(model_df[date_col], errors="coerce")
        model_df[f"{prefix}_Month"] = parsed.dt.month
        model_df[f"{prefix}_DayOfWeek"] = parsed.dt.dayofweek

    predictors = [c for c in [
        "Facility_Type", "User_Category", "Lead_Time_Days", "Participants", "Purpose",
        "Time_Slot", "Reservation_Month", "Reservation_DayOfWeek",
        "Submission_Month", "Submission_DayOfWeek"] if c in model_df.columns]

    model_df = model_df[model_df["Reservation_Outcome"].notna()].copy()
    X = model_df[predictors].copy()
    for col in predictors:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        else:
            X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    y = model_df["Reservation_Outcome"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y)

    cat_feats = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_feats = [c for c in X_train.columns if c not in cat_feats]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_feats),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
    ])
    tree = DecisionTreeClassifier(random_state=seed, class_weight="balanced",
                                  max_depth=max_depth, min_samples_leaf=min_leaf)
    clf = Pipeline([("preprocessor", pre), ("model", tree)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Aggregate one-hot importances back to original predictors
    encoded = clf.named_steps["preprocessor"].get_feature_names_out()
    imp = pd.DataFrame({"Encoded feature": encoded,
                        "Importance": clf.named_steps["model"].feature_importances_})

    def original_predictor(name):
        name = name.split("__", 1)[-1]
        for f in sorted(predictors, key=len, reverse=True):
            if name == f or name.startswith(f + "_"):
                return f
        return name

    imp["Original predictor"] = imp["Encoded feature"].apply(original_predictor)
    grouped = (imp.groupby("Original predictor", as_index=False)["Importance"]
               .sum().sort_values("Importance", ascending=False))

    return {"clf": clf, "predictors": predictors, "y": y, "X_train": X_train, "X_test": X_test,
            "y_test": y_test, "y_pred": y_pred, "grouped_importance": grouped}


# ----------------------------------------------------------------------------
# UML diagram (matplotlib PNG + draw.io XML), built from one shared layout
# ----------------------------------------------------------------------------
USE_CASES = {  # id: (label, cx, cy, w, h, fill, edge)
    "uc1": ("Submit Reservation\nRequest", 330, 105, 210, 70, "#dbeafe", "#2563eb"),
    "uc2": ("Cancel Reservation", 330, 205, 210, 70, "#dbeafe", "#2563eb"),
    "uc3": ("View Reservation\nStatus", 330, 305, 210, 70, "#dbeafe", "#2563eb"),
    "uc9": ("Validate & Clean\nReservation Data", 330, 575, 210, 70, "#fef3c7", "#d97706"),
    "uc10": ("Train & Evaluate\nDecision Tree Model", 330, 695, 210, 70, "#fef3c7", "#d97706"),
    "uc4": ("View Reservation Statistics\nby Facility Type", 730, 105, 230, 70, "#dcfce7", "#16a34a"),
    "uc5": ("Analyze Cancellation /\nNo-Show Patterns", 730, 205, 230, 70, "#dcfce7", "#16a34a"),
    "uc8": ("Review & Decide on\nReservation (Manual)", 730, 325, 230, 70, "#dcfce7", "#16a34a"),
    "uc6": ("View Predicted\nNon-Completion Risk", 730, 455, 230, 70, "#fce7f3", "#db2777"),
    "uc7": ("Review Prediction\nExplanation", 730, 575, 230, 70, "#fce7f3", "#db2777"),
}
ACTORS = {  # id: (label, x, y_top)
    "req": ("Reservation Requester\n(Student / Org /\nFaculty / Guest)", 75, 150),
    "admin": ("Facility\nAdministrator", 965, 290),
    "analyst": ("Data Analyst", 75, 590),
}
ASSOC = [("req", "uc1"), ("req", "uc2"), ("req", "uc3"),
         ("admin", "uc4"), ("admin", "uc5"), ("admin", "uc8"), ("admin", "uc6"), ("admin", "uc7"),
         ("analyst", "uc9"), ("analyst", "uc10")]
RELATIONS = [("uc8", "uc6", "«include»"), ("uc7", "uc6", "«extend»"), ("uc10", "uc9", "«include»")]
NOTE = ("Note: Risk prediction is advisory only. The system never automatically rejects a "
        "reservation; an administrator makes the final decision.")


def draw_uml_figure():
    def actor_anchor(a):
        _, x, y = ACTORS[a]
        return np.array([x, y + 28.0])

    def edge_point(center, a, b, toward):
        d = np.array(toward) - np.array(center)
        k = 1 / np.sqrt((d[0] / a) ** 2 + (d[1] / b) ** 2)
        return np.array(center) + d * k

    fig, ax = plt.subplots(figsize=(13, 9.5))
    ax.set_xlim(0, 1040); ax.set_ylim(780, 0); ax.axis("off")
    ax.add_patch(Rectangle((190, 20), 680, 740, fc="#f8fafc", ec="#334155", lw=2, zorder=0))
    ax.text(530, 42, "Campus Facility Reservation and Monitoring System",
            ha="center", va="center", fontsize=12, fontweight="bold")

    for a_id, u_id in ASSOC:
        _, cx, cy, w, h, _, _ = USE_CASES[u_id]
        p0 = actor_anchor(a_id)
        p1 = edge_point((cx, cy), w / 2, h / 2, p0)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#111827", lw=1.2, zorder=1)

    for src, dst, label in RELATIONS:
        _, sx, sy, sw, sh, _, _ = USE_CASES[src]
        _, dx, dy, dw, dh, _, _ = USE_CASES[dst]
        p0 = edge_point((sx, sy), sw / 2, sh / 2, (dx, dy))
        p1 = edge_point((dx, dy), dw / 2, dh / 2, (sx, sy))
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="->", mutation_scale=14,
                                     linestyle="--", color="#111827", lw=1.2, zorder=3))
        mid = (p0 + p1) / 2
        ax.text(mid[0] + 8, mid[1], label, fontsize=9, style="italic", va="center",
                bbox=dict(fc="#f8fafc", ec="none", pad=1), zorder=4)

    for _, (label, cx, cy, w, h, fc, ec) in USE_CASES.items():
        ax.add_patch(Ellipse((cx, cy), w, h, fc=fc, ec=ec, lw=1.5, zorder=2))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=9, zorder=5)

    for _, (label, x, y) in ACTORS.items():
        ax.add_patch(Ellipse((x, y + 8), 16, 16, fc="white", ec="black", lw=1.5, zorder=2))
        ax.plot([x, x], [y + 16, y + 38], "k", lw=1.5)
        ax.plot([x - 14, x + 14], [y + 24, y + 24], "k", lw=1.5)
        ax.plot([x, x - 12], [y + 38, y + 58], "k", lw=1.5)
        ax.plot([x, x + 12], [y + 38, y + 58], "k", lw=1.5)
        ax.text(x, y + 66, label, ha="center", va="top", fontsize=9)

    ax.add_patch(Rectangle((615, 650), 230, 80, fc="#fff9c4", ec="#a16207", lw=1, zorder=2))
    ax.text(623, 690, "Note: Risk prediction is advisory only.\nThe system never automatically rejects a\n"
            "reservation; an administrator makes the\nfinal decision.", fontsize=8, va="center", zorder=5)
    ax.set_title("Figure — UML Use Case Diagram: Campus Facility Reservation and Monitoring System",
                 fontsize=11, pad=8)
    plt.tight_layout()
    return fig


def build_drawio_xml() -> str:
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']

    def geom(x, y, w, h):
        return f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'

    cells.append(
        '<mxCell id="boundary" value="Campus Facility Reservation and Monitoring System" '
        'style="rounded=0;whiteSpace=wrap;html=1;verticalAlign=top;fontStyle=1;fontSize=14;'
        'fillColor=#f8fafc;strokeColor=#334155;strokeWidth=2;" vertex="1" parent="1">'
        + geom(190, 20, 680, 740) + "</mxCell>")

    for a_id, (label, x, y) in ACTORS.items():
        val = escape(label.replace("\n", " "))
        cells.append(
            f'<mxCell id="actor_{a_id}" value="{val}" style="shape=umlActor;verticalLabelPosition=bottom;'
            f'verticalAlign=top;html=1;outlineConnect=0;" vertex="1" parent="1">'
            + geom(x - 15, y, 30, 60) + "</mxCell>")

    for u_id, (label, cx, cy, w, h, fc, ec) in USE_CASES.items():
        val = escape(label.replace("\n", " "))
        cells.append(
            f'<mxCell id="{u_id}" value="{val}" style="ellipse;whiteSpace=wrap;html=1;'
            f'fillColor={fc};strokeColor={ec};" vertex="1" parent="1">'
            + geom(cx - w // 2, cy - h // 2, w, h) + "</mxCell>")

    cells.append(
        f'<mxCell id="note" value="{escape(NOTE)}" style="shape=note;whiteSpace=wrap;html=1;size=14;'
        f'fillColor=#fff9c4;strokeColor=#a16207;fontSize=11;align=left;spacingLeft=6;" vertex="1" parent="1">'
        + geom(615, 650, 230, 80) + "</mxCell>")

    for i, (a_id, u_id) in enumerate(ASSOC, 1):
        cells.append(f'<mxCell id="a{i}" style="endArrow=none;html=1;" edge="1" parent="1" '
                     f'source="actor_{a_id}" target="{u_id}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    for i, (s, t, lab) in enumerate(RELATIONS, 1):
        cells.append(f'<mxCell id="r{i}" value="{escape(lab)}" style="endArrow=open;dashed=1;html=1;endFill=0;'
                     f'labelBackgroundColor=#f8fafc;fontSize=11;" edge="1" parent="1" source="{s}" target="{t}">'
                     f'<mxGeometry relative="1" as="geometry"/></mxCell>')

    return ('<mxfile host="app.diagrams.net"><diagram id="uc1" name="Use Case Diagram">'
            '<mxGraphModel dx="1000" dy="780" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
            'arrows="1" fold="1" page="1" pageScale="1" pageWidth="1100" pageHeight="800" math="0" shadow="0">'
            "<root>" + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
seed = st.sidebar.number_input("Random seed", value=42, step=1)
n_rows = st.sidebar.number_input("Synthetic rows", min_value=100, max_value=5000, value=402, step=1)
st.sidebar.markdown("**Decision tree**")
max_depth = st.sidebar.slider("max_depth", 2, 12, 5)
min_leaf = st.sidebar.slider("min_samples_leaf", 1, 30, 5)
uploaded = st.sidebar.file_uploader("…or upload your own CSV (optional)", type="csv")

if uploaded is not None:
    df_original = pd.read_csv(uploaded)
    data_source = f"Uploaded file: {uploaded.name}"
    missing_req = [c for c in REQUIRED_COLUMNS if c not in df_original.columns]
    if missing_req:
        st.error(f"Uploaded file is missing required columns: {missing_req}")
        st.stop()
else:
    df_original = generate_data(int(n_rows), int(seed))
    data_source = f"Reproducibly generated synthetic sample ({int(n_rows)} rows; seed={int(seed)})"

df, clean_log = clean_data(df_original)

st.title("🏫 Campus Facility Reservation — Data Mining Midterm")
st.caption(f"Dataset: {df_original.shape[0]} rows × {df_original.shape[1]} columns · Source: {data_source}")

tabs = st.tabs([
    "I. Data Understanding", "II. Cleaning & Features", "III. EDA",
    "IV. Modeling", "V. Requirements & UML", "VI. Decision Support", "Export",
])

# ----------------------------------------------------------------------------
# Part I
# ----------------------------------------------------------------------------
with tabs[0]:
    st.header("Part I — Dataset Acquisition and Understanding")
    st.subheader("Task 1 — Preview, dimensions, types, summary statistics")
    st.dataframe(df_original.head(10), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Rows", df_original.shape[0])
    c2.metric("Columns", df_original.shape[1])
    st.markdown("**Column data types**")
    st.dataframe(df_original.dtypes.astype(str).rename("Data type").to_frame())
    st.markdown("**Summary statistics**")
    st.dataframe(df_original.describe(include="all").T, use_container_width=True)

    st.subheader("Task 2 — Predictors, target, variable types")
    st.markdown(
        "`Reservation_ID` is an identifier, not a useful behavioral predictor. `Usage_Status` is used to "
        "construct the target. `Actual_Attendance` and `Cancellation_Days_Before` are post-outcome fields "
        "and must not be predictors.")
    candidates = [c for c in ["Facility_Type", "User_Category", "Lead_Time_Days", "Participants",
                              "Purpose", "Time_Slot", "Reservation_Date", "Submission_Date"]
                  if c in df_original.columns]
    cat_vars = df_original.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_vars = df_original.select_dtypes(include=np.number).columns.tolist()
    st.write("**Predictor candidates (verify available at submission time):**", ", ".join(candidates))
    st.write("**Target source:** Usage_Status → Reservation_Outcome")
    st.write("**Categorical variables:**", cat_vars)
    st.write("**Numerical variables:**", num_vars)

    st.subheader("Task 3 — Data acquisition and validation")
    st.markdown(
        "A production system could retrieve authorized records from an authenticated API or parameterized "
        "SQL query, with access controls and only the necessary fields. Validate required columns, date "
        "formats, permitted status values, and identifier uniqueness before analysis.")
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df_original.columns]
    st.write("**Required columns missing:**", missing_required or "None")
    st.write("**Duplicate Reservation_ID values:**", int(df_original["Reservation_ID"].duplicated().sum()))
    for col in ["Reservation_Date", "Submission_Date"]:
        st.write(f"**Unparseable {col} values:**",
                 int(pd.to_datetime(df_original[col], errors="coerce").isna().sum()))

# ----------------------------------------------------------------------------
# Part II
# ----------------------------------------------------------------------------
with tabs[1]:
    st.header("Part II — Data Cleaning and Feature Engineering")
    st.subheader("Task 4 — Data quality checks and cleaning decisions")
    st.markdown(
        "The original data is preserved. Cleaning is conservative: standardize labels, remove exact duplicate "
        "rows, parse dates and numerics, and mark invalid numeric values as missing rather than inventing "
        "replacements. Missing cancellation lead time is retained as missing because it may indicate no "
        "cancellation occurred.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows before", clean_log["rows_before"])
    c2.metric("Duplicates removed", clean_log["duplicates"])
    c3.metric("Rows after", clean_log["rows_after"])
    c4.metric("Lead-time/date mismatches", clean_log["mismatches"], help="Retained for audit")
    st.markdown("**Missing values after cleaning**")
    st.dataframe(df.isna().sum().rename("Missing count").to_frame())

    st.subheader("Task 5 — Derived variables")
    st.markdown(
        "- `Lead_Time_Category`: 0–2 = Last Minute; 3–7 = Short Lead; 8–14 = Medium Lead; >14 = Early Booking.\n"
        "- `Reservation_Outcome`: Completed for Used; Not Completed for Cancelled or No-Show; other statuses remain missing.\n"
        "- `Attendance_Ratio`: Actual_Attendance / Participants; missing if numerator/denominator is missing "
        "or denominator is not positive.")
    preview_cols = ["Lead_Time_Days", "Lead_Time_Category", "Usage_Status", "Reservation_Outcome",
                    "Participants", "Actual_Attendance", "Attendance_Ratio"]
    st.dataframe(df[preview_cols].head(10), use_container_width=True)

# ----------------------------------------------------------------------------
# Part III
# ----------------------------------------------------------------------------
with tabs[2]:
    st.header("Part III — Exploratory Data Analysis")
    st.caption("Percentages use reservations with known `Usage_Status` as the denominator. "
               "Charts describe historical association, not causation.")

    st.subheader("Reservations by facility type")
    facility_counts = df["Facility_Type"].value_counts(dropna=False)
    st.dataframe(facility_counts.rename("Reservation count").to_frame())
    st.write(f"**Highest-count facility type:** {facility_counts.index[0]} — "
             f"{int(facility_counts.iloc[0])} reservations")
    fig, ax = plt.subplots(figsize=(9, 4))
    facility_counts.sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_title("Reservation Count by Facility Type")
    ax.set_xlabel("Facility Type"); ax.set_ylabel("Number of Reservations")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    show_fig(fig)

    st.subheader("Cancellation / no-show rate by user category")
    tmp = df.assign(Is_Cancelled_NoShow=is_bad(df["Usage_Status"]))
    valid = tmp[tmp["Usage_Status"].notna()]
    user_rates = valid.groupby("User_Category", dropna=False).agg(
        Total_with_known_status=("Is_Cancelled_NoShow", "size"),
        Cancelled_or_NoShow=("Is_Cancelled_NoShow", "sum"))
    user_rates["Cancellation_NoShow_Percent"] = (
        100 * user_rates["Cancelled_or_NoShow"] / user_rates["Total_with_known_status"])
    user_rates = user_rates.sort_values("Cancellation_NoShow_Percent", ascending=False)
    st.dataframe(user_rates, use_container_width=True)
    if len(user_rates):
        top = user_rates.iloc[0]
        st.write(f"**Highest rate:** {user_rates.index[0]} — {int(top['Cancelled_or_NoShow'])}/"
                 f"{int(top['Total_with_known_status'])} known-status reservations "
                 f"({top['Cancellation_NoShow_Percent']:.1f}%).")
    fig, ax = plt.subplots(figsize=(9, 4))
    user_rates["Cancellation_NoShow_Percent"].plot(kind="bar", ax=ax)
    ax.set_title("Cancellation/No-Show Rate by User Category")
    ax.set_xlabel("User Category"); ax.set_ylabel("Cancelled or No-Show (%)"); ax.set_ylim(0, 100)
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    show_fig(fig)

    st.subheader("Outcome by lead-time category")
    cross = pd.crosstab(df["Lead_Time_Category"], df["Reservation_Outcome"])
    row_pct = pd.crosstab(df["Lead_Time_Category"], df["Reservation_Outcome"], normalize="index") * 100
    c1, c2 = st.columns(2)
    c1.dataframe(cross, use_container_width=True)
    c2.dataframe(row_pct.round(1).add_suffix(" (%)"), use_container_width=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    row_pct.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Reservation Outcome by Lead-Time Category")
    ax.set_xlabel("Lead-Time Category"); ax.set_ylabel("Percent within Lead-Time Category")
    ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    show_fig(fig)

    st.info("**Interpretation:** Record the highest facility count, the highest user-category "
            "cancellation/no-show rate (with numerator/denominator), and the lead-time category with the "
            "highest *Not Completed* share. Use the displayed values, not assumptions, and do not claim causation.")

# ----------------------------------------------------------------------------
# Part IV
# ----------------------------------------------------------------------------
with tabs[3]:
    st.header("Part IV — Data Mining and Predictive Modeling")
    st.markdown(
        "**Leakage prevention:** `Usage_Status` creates the target but is not a predictor. `Actual_Attendance`, "
        "`Cancellation_Days_Before`, `Attendance_Ratio`, and outcome-derived fields are excluded. Date fields "
        "are represented by month and day of week.")

    res = train_model(df, int(seed), max_depth, min_leaf)
    predictors, y = res["predictors"], res["y"]
    y_test, y_pred = res["y_test"], res["y_pred"]

    if len(predictors) < 5:
        st.error("At least five submission-time predictors are required.")
        st.stop()
    if y.nunique() < 2:
        st.error("Target contains fewer than two classes.")
        st.stop()

    st.write(f"**Predictors ({len(predictors)}):**", ", ".join(predictors))
    st.dataframe(y.value_counts().rename("Count").to_frame())
    st.write(f"Training rows: {len(res['X_train'])} | Test rows: {len(res['X_test'])}")

    labels = ["Completed", "Not Completed"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    m2.metric("Precision (macro)", f"{precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    m3.metric("Recall (macro)", f"{recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    m4.metric("F1 (macro)", f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax, values_format="d")
        ax.set_title("Decision Tree — Confusion Matrix"); plt.tight_layout()
        show_fig(fig)
    with c2:
        st.markdown("**Class-specific report**")
        report = classification_report(y_test, y_pred, labels=labels, zero_division=0, output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.subheader("Feature importance (aggregated by original predictor)")
    grouped = res["grouped_importance"]
    st.write("**Three most influential predictors:**")
    st.dataframe(grouped.head(3), use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    grouped.head(10).sort_values("Importance").plot(
        x="Original predictor", y="Importance", kind="barh", legend=False, ax=ax)
    ax.set_title("Feature Importance by Original Predictor")
    ax.set_xlabel("Aggregated impurity-based importance"); ax.set_ylabel("Predictor")
    plt.tight_layout()
    show_fig(fig)

    st.subheader("Task 8 — Reliability assessment")
    st.markdown(
        "Do not judge reliability by accuracy alone. Inspect the confusion matrix and class-specific "
        "recall/F1 for `Not Completed`. The dataset is synthetic and may be imbalanced; one train/test split "
        "may not generalize. Improvements include stratified cross-validation, comparison with a simple "
        "baseline and alternative models, and testing on newer validated data. Predictions should support "
        "human review, not automatic rejection.")

# ----------------------------------------------------------------------------
# Part V
# ----------------------------------------------------------------------------
with tabs[4]:
    st.header("Part V — Requirements Analysis and Use Case Modeling")
    st.subheader("Task 9 — Functional requirements tied to actual findings")
    st.caption("Evidence statements are generated from this run's descriptive analysis.")

    evidence = []
    if df["Facility_Type"].notna().any():
        counts = df["Facility_Type"].value_counts()
        evidence.append(f"{counts.index[0]} has the highest observed count ({int(counts.iloc[0])} reservations).")
    known = df.assign(_bad=is_bad(df["Usage_Status"]))
    known = known[known["Usage_Status"].notna()]
    rates = known.groupby("User_Category").agg(total=("_bad", "size"), bad=("_bad", "sum"))
    if len(rates):
        rates["pct"] = 100 * rates["bad"] / rates["total"]
        rates = rates.sort_values("pct", ascending=False)
        r = rates.iloc[0]
        evidence.append(f"{rates.index[0]}: {int(r['bad'])}/{int(r['total'])} known-status "
                        f"reservations ({r['pct']:.1f}%).")
    tab = pd.crosstab(df["Lead_Time_Category"], df["Reservation_Outcome"], normalize="index") * 100
    if "Not Completed" in tab and len(tab):
        cat = tab["Not Completed"].idxmax()
        evidence.append(f"{cat} has the highest observed Not Completed share "
                        f"({tab.loc[cat, 'Not Completed']:.1f}% within that category).")

    requirements = [
        ("FR-DM-01", "The system shall display reservation counts by facility type to help administrators monitor demand."),
        ("FR-DM-02", "The system shall display cancellation/no-show rates by user category with numerator and denominator."),
        ("FR-DM-03", "The system shall show estimated non-completion risk, the main contributing predictors, and a human-review notice."),
    ]
    requirements_df = pd.DataFrame([
        {"Finding / Evidence": evidence[i] if i < len(evidence)
         else "Confirm a supporting finding from current EDA/model output before submission.",
         "Requirement ID": rid, "Functional Requirement": req}
        for i, (rid, req) in enumerate(requirements)])
    st.dataframe(requirements_df, use_container_width=True, hide_index=True)

    st.subheader("Task 10 — UML use case diagram")
    st.markdown(
        "Three actors, ten use cases, associations, and include/extend relationships. Risk prediction is "
        "advisory; the diagram does not automate rejection. Open the `.drawio` file in "
        "[diagrams.net](https://app.diagrams.net/) to edit.")
    uml_fig = draw_uml_figure()
    buf = BytesIO()
    uml_fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    show_fig(uml_fig)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download PNG", buf.getvalue(), "Campus_Facility_Reservation_UML.png", "image/png")
    d2.download_button("⬇️ Download .drawio", build_drawio_xml(),
                       "Campus_Facility_Reservation_UML.drawio", "application/xml")

# ----------------------------------------------------------------------------
# Part VI
# ----------------------------------------------------------------------------
with tabs[5]:
    st.header("Part VI — Decision Support and Critical Analysis")
    st.subheader("Task 11 — Dashboard wireframe")
    st.caption("Separates **observed historical measures** from **model predictions**. Predicted risk is an "
               "estimate and should show model version/date and explanation.")

    total_reservations = len(df)
    known_status = int(df["Usage_Status"].notna().sum())
    bad_count = int(is_bad(df["Usage_Status"]).sum())
    bad_rate = 100 * bad_count / known_status if known_status else np.nan

    st.markdown("##### 📊 Observed facts")
    o1, o2, o3 = st.columns(3)
    o1.metric("Total reservations", total_reservations)
    o2.metric("Known outcome status", known_status)
    o3.metric("Cancelled / no-show", f"{bad_count} ({bad_rate:.1f}%)" if known_status else "n/a")
    st.markdown("**Observed trends:** facility counts; cancellation/no-show by user category; "
                "outcomes by lead-time category.")

    st.markdown("##### 🔮 Predicted risk — estimate, not observed fact")
    st.warning("Show probability/risk band, model version and timestamp, contributing factors, and "
               "administrator review. Do not automatically reject a reservation based only on a model score.")

    st.subheader("Task 12 — Short answers")
    st.markdown(
        "**(a) Why should predicted high risk not automatically reject a reservation?**  \n"
        "A prediction is uncertain and can be wrong. Automatic rejection may unfairly deny legitimate access. "
        "Use risk as a review aid, explain the factors, and let an authorized administrator apply transparent policy.\n\n"
        "**(b) How does poor data quality affect the model and information system?**  \n"
        "Missing, duplicated, inconsistent, or invalid records can distort counts and rates, bias training, reduce "
        "predictive quality, and lead administrators to make decisions from unreliable information. Validation and "
        "documented cleaning are essential.\n\n"
        "**(c) What information should explain a high-risk prediction?**  \n"
        "Display the estimated probability and risk band, model version/date, contributing factors, relevant "
        "historical context, uncertainty and limitations, and a clear statement that the prediction is advisory "
        "and requires human review.")

# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------
with tabs[6]:
    st.header("Export cleaned data and data-quality summary")
    quality_summary = pd.DataFrame({
        "Missing_Count": df.isna().sum(),
        "Missing_Percent": (100 * df.isna().mean()).round(2),
        "Data_Type": df.dtypes.astype(str),
    })
    st.dataframe(quality_summary, use_container_width=True)

    res = train_model(df, int(seed), max_depth, min_leaf)
    y_test, y_pred = res["y_test"], res["y_pred"]
    report_lines = [
        "DATA MINING LABORATORY — RUN SUMMARY",
        f"Source: {data_source}",
        f"Original shape: {df_original.shape}",
        f"Cleaned shape: {df.shape}",
        f"Exact duplicates removed: {clean_log['duplicates']}",
        "", "Data-quality summary:", quality_summary.to_string(),
        "", "Reservation counts by facility type:", df["Facility_Type"].value_counts(dropna=False).to_string(),
        "", "Top model predictors:", res["grouped_importance"].head(3).to_string(index=False),
        "", "Model evaluation:",
        f"Accuracy: {accuracy_score(y_test, y_pred):.4f}",
        f"Macro precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}",
        f"Macro recall: {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}",
        f"Macro F1: {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}",
        "Inspect class-specific results; accuracy alone is insufficient.",
    ]

    e1, e2, e3 = st.columns(3)
    e1.download_button("⬇️ facility_reservations_cleaned.csv", df.to_csv(index=False),
                       "facility_reservations_cleaned.csv", "text/csv")
    e2.download_button("⬇️ data_quality_summary.csv", quality_summary.to_csv(index=True),
                       "data_quality_summary.csv", "text/csv")
    e3.download_button("⬇️ findings_report.txt", "\n".join(report_lines),
                       "findings_report.txt", "text/plain")
