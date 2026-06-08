import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import pipeline

# =====================================================
# CREATE OUTPUT FOLDER
# =====================================================

os.makedirs("output", exist_ok=True)

# =====================================================
# FILE PATHS
# =====================================================

HR_FILE = r"data/synthetic_hr_performance_data.xlsx"

COMMENTS_FILE = r"data/synthetic_comments_data.xlsx"

OUTPUT_SENTIMENT = r"output/Sentiment_Analysis_Output.xlsx"

OUTPUT_MERGED = r"output/Final_HR_Analytics.xlsx"

OUTPUT_HEATMAP = r"output/Correlation_Heatmap.png"

OUTPUT_CORR = r"output/Correlation_Matrix.xlsx"

# =====================================================
# LOAD FILES
# ======================================================

print("\nLoading files...")

try:

    df_hr = pd.read_excel(HR_FILE)

    df_comments = pd.read_excel(COMMENTS_FILE)

except Exception as e:

    print(f"\nFILE LOADING ERROR: {e}")

    exit()

print("\nHR FILE COLUMNS:")

print(df_hr.columns.tolist())

print("\nCOMMENTS FILE COLUMNS:")

print(df_comments.columns.tolist())

# =====================================================
# CLEAN COLUMN NAMES
# =====================================================

df_hr.columns = df_hr.columns.str.strip()

df_comments.columns = df_comments.columns.str.strip()

# =====================================================
# FIX TYPO IN MANAGER COMMENT COLUMN
# =====================================================

df_comments.rename(

    columns={

        "MangerComment": "ManagerComment"

    },

    inplace=True

)

# =====================================================
# VERIFY REQUIRED COLUMNS
# =====================================================

required_hr_columns = [

    "Employee Name",
    "Employee Overall Score",
    "Manager Overall Score",
]

required_comment_columns = [

    "EmployeeName",
    "EmployeeComment",
    "ManagerComment"
]

for col in required_hr_columns:

    if col not in df_hr.columns:

        print(f"\nMissing HR Column: {col}")

        exit()

for col in required_comment_columns:

    if col not in df_comments.columns:

        print(f"\nMissing Comment Column: {col}")

        exit()

# =====================================================
# CLEAN EMPLOYEE NAMES
# =====================================================

df_hr["Employee Name"] = (

    df_hr["Employee Name"]

    .astype(str)

    .str.lower()

    .str.strip()

)

df_comments["EmployeeName"] = (

    df_comments["EmployeeName"]

    .astype(str)

    .str.lower()

    .str.strip()

)

# =====================================================
# REMOVE DUPLICATES
# =====================================================

df_comments = df_comments.drop_duplicates(

    subset=["EmployeeName"]

)

# =====================================================
# CLEAN COMMENTS
# =====================================================

df_comments["EmployeeComment"] = (

    df_comments["EmployeeComment"]

    .fillna("")

    .astype(str)

)

df_comments["ManagerComment"] = (

    df_comments["ManagerComment"]

    .fillna("")

    .astype(str)

)

print("\nComments cleaned")

# =====================================================
# LOAD SENTIMENT MODEL
# =====================================================

print("\nLoading sentiment model...")

try:

    sentiment_pipeline = pipeline(

        "sentiment-analysis",

        model="distilbert-base-uncased-finetuned-sst-2-english"

    )

    print("Model loaded")

except Exception as e:

    print(f"\nMODEL ERROR: {e}")

    exit()

# =====================================================
# SENTIMENT FUNCTION
# =====================================================

def get_sentiment(text):

    text = str(text).strip()

    if text == "":

        return "NEUTRAL", 0.0

    try:

        result = sentiment_pipeline(text[:512])[0]

        label = result["label"].upper()

        score = float(result["score"])

        if label == "POSITIVE":

            return "POSITIVE", round(score, 4)

        elif label == "NEGATIVE":

            return "NEGATIVE", round(-score, 4)

        else:

            return "NEUTRAL", 0.0

    except Exception as e:

        print(f"Sentiment Error: {e}")

        return "NEUTRAL", 0.0

# =====================================================
# EMPLOYEE SENTIMENT
# =====================================================

print("\nAnalyzing Employee Comments...")

employee_results = (

    df_comments["EmployeeComment"]

    .apply(get_sentiment)

)

df_comments[

    [

        "employee_sentiment_label",

        "employee_sentiment_score"

    ]

] = pd.DataFrame(

    employee_results.tolist(),

    index=df_comments.index

)

# =====================================================
# MANAGER SENTIMENT
# =====================================================

print("\nAnalyzing Manager Comments...")

manager_results = (

    df_comments["ManagerComment"]

    .apply(get_sentiment)

)

df_comments[

    [

        "manager_sentiment_label",

        "manager_sentiment_score"

    ]

] = pd.DataFrame(

    manager_results.tolist(),

    index=df_comments.index

)

# =====================================================
# SAVE SENTIMENT OUTPUT
# =====================================================

sentiment_output = df_comments[

    [

        "EmployeeId",

        "EmployeeCustId",

        "EmployeeName",

        "EmployeeComment",

        "employee_sentiment_label",

        "employee_sentiment_score",

        "ManagerComment",

        "manager_sentiment_label",

        "manager_sentiment_score"

    ]

]

try:

    sentiment_output.to_excel(

        OUTPUT_SENTIMENT,

        index=False

    )

    print("\nSentiment output saved")

except PermissionError:

    print(

        "\nClose Sentiment_Analysis_Output.xlsx and rerun"

    )

    exit()

# =====================================================
# MERGE DATASETS
# =====================================================

print("\nMerging datasets using Employee Name...")

merged_df = pd.merge(

    df_hr,

    sentiment_output,

    left_on="Employee Name",

    right_on="EmployeeName",

    how="left"

)

print(f"\nMerged rows: {len(merged_df)}")

# =====================================================
# FILL NULL SENTIMENTS
# =====================================================

merged_df["employee_sentiment_score"] = (

    merged_df["employee_sentiment_score"]

    .fillna(0)

)

merged_df["manager_sentiment_score"] = (

    merged_df["manager_sentiment_score"]

    .fillna(0)

)

# =====================================================
# PERFORMANCE RATING MAPPING
# =====================================================

rating_mapping = {

    "A+": 5.0,
    "A": 4.5,

    "B+": 4.0,
    "B": 3.5,

    "C+": 3.0,
    "C": 2.5,

    "D": 2.0,

    "E": 1.0
}

# =====================================================
#
# =====================================================
# CONVERT RATINGS TO NUMERIC

# =====================================================
# CONVERT NUMERIC COLUMNS
# =====================================================

numeric_cols = [

    "Employee Overall Score",

    "Manager Overall Score",

    "employee_sentiment_score",

    "manager_sentiment_score"

]

for col in numeric_cols:

    merged_df[col] = pd.to_numeric(

        merged_df[col],

        errors="coerce"

    )

# =====================================================
# ADVANCED HR METRICS
# =====================================================

merged_df["score_gap"] = (

    merged_df["Employee Overall Score"]

    - merged_df["Manager Overall Score"]

)


merged_df["sentiment_gap"] = (

    merged_df["employee_sentiment_score"]

    - merged_df["manager_sentiment_score"]

)

merged_df["employee_positive_flag"] = np.where(

    merged_df["employee_sentiment_score"] > 0,

    1,

    0

)

merged_df["manager_positive_flag"] = np.where(

    merged_df["manager_sentiment_score"] > 0,

    1,

    0

)

# =====================================================
# SAVE MERGED FILE
# =====================================================

try:

    merged_df.to_excel(

        OUTPUT_MERGED,

        index=False

    )

    print("\nFinal merged file saved")

except PermissionError:

    print(

        "\nClose Final_HR_Analytics.xlsx and rerun"

    )

    exit()

# =====================================================
# CORRELATION MATRIX
# =====================================================

print("\nGenerating enterprise correlation matrix...")

corr_columns = [

    "Employee Overall Score",

    "Manager Overall Score",


    "employee_sentiment_score",

    "manager_sentiment_score",

    "score_gap",

    "sentiment_gap",

    "employee_positive_flag",

    "manager_positive_flag"

]

corr_df = merged_df[corr_columns].corr()

print("\nCorrelation Matrix:\n")

print(corr_df)

# =====================================================
# SAVE CORRELATION MATRIX
# =====================================================

try:

    corr_df.to_excel(OUTPUT_CORR)

    print("\nCorrelation matrix saved")

except PermissionError:

    print(

        "\nClose Correlation_Matrix.xlsx and rerun"

    )

# =====================================================
# HEATMAP
# =====================================================

plt.figure(figsize=(15, 11))

sns.heatmap(

    corr_df,

    annot=True,

    cmap="coolwarm",

    fmt=".2f",

    linewidths=0.5,

    center=0

)

plt.title(

    "Enterprise HR Analytics Correlation Matrix",

    fontsize=18,

    fontweight="bold"

)

plt.xticks(rotation=45, ha="right")

plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(

    OUTPUT_HEATMAP,

    dpi=300

)

print("\nHeatmap saved")

# =====================================================
# SUMMARY ANALYTICS
# =====================================================

print("\n===================================")

print("HR ANALYTICS SUMMARY")

print("===================================")

print(

    "\nAverage Employee Sentiment:",

    round(

        merged_df["employee_sentiment_score"]

        .mean(),

        3

    )

)

print(

    "Average Manager Sentiment:",

    round(

        merged_df["manager_sentiment_score"]

        .mean(),

        3

    )

)

print(

    "Average Employee Score:",

    round(

        merged_df["Employee Overall Score"]

        .mean(),

        3

    )

)

print(

    "Average Manager Score:",

    round(

        merged_df["Manager Overall Score"]

        .mean(),

        3

    )

)



# =====================================================
# TOP INSIGHTS
# =====================================================

print("\n===================================")

print("KEY INSIGHTS")

print("===================================")

highest_positive = merged_df.loc[
    merged_df["employee_sentiment_score"].idxmax()
]

highest_negative = merged_df.loc[
    merged_df["employee_sentiment_score"].idxmin()
]

print(
    f"\nMost Positive Employee: "
    f"{highest_positive['Employee Name']}"
)

print(
    f"Sentiment Score: "
    f"{highest_positive['employee_sentiment_score']}"
)

print(
    f"\nMost Negative Employee: "
    f"{highest_negative['Employee Name']}"
)

print(
    f"Sentiment Score: "
    f"{highest_negative['employee_sentiment_score']}"
)


# =====================================================
# DONE
# =====================================================

print("\n====================================")

print("ENTERPRISE HR ANALYTICS COMPLETE")

print("====================================")

print(f"\nSentiment File:\n{OUTPUT_SENTIMENT}")

print(f"\nMerged File:\n{OUTPUT_MERGED}")

print(f"\nCorrelation Matrix:\n{OUTPUT_CORR}")

print(f"\nHeatmap:\n{OUTPUT_HEATMAP}")
# Change this:
FILE_NAME = r"data/Employee_Sheet.xlsx"