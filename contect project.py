# Let's get started and get the full grade as A PROFESSIONAL ENGINEER :)
# Student Name, ID( Hassan Al-Zahrani,451401862 __ Yasser Alsaidlani,451401188
# Hashema Alsaidalani,451400324 __Mohammed Alshehri,451400125 )
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve
from google.colab import drive #For filr sharing

def prep_numeric(df_input):
    """Prepares the DataFrame for numerical analysis by handling NaNs and converting types."""
    df_local = df_input.copy()
    df_local["Age"] = df_local["Age"].fillna(df_local["Age"].mean())
    df_local["Sex"] = df_local["Sex"].astype(str).str.strip().str.lower()
    df_local["SexNum"] = df_local["Sex"].map({"female": 1, "male": 0})
    for c in ["Survived", "Pclass", "Age", "SexNum"]:
        df_local[c] = pd.to_numeric(df_local[c], errors="coerce")
    df_local = df_local.dropna(subset=["Survived", "Pclass", "Age", "SexNum"])
    return df_local

#-------------------------------------------------
# Read the file + DataFrame
#-------------------------------------------------

df = pd.read_csv("titanic.csv", sep=',')

#Preparing Ages
df["Age"] = df["Age"].fillna(df["Age"].mean())

#Prepare SexNum for numeric representation (Moved here for consistency)
df['Sex'] = df['Sex'].astype(str).str.strip().str.lower()
sex_map = {'female': 1, 'male': 0}
df['SexNum'] = df['Sex'].map(sex_map)

#Preparing the classification for the heatmap
df["Category"] = df.apply(
    lambda r: "Child" if r["Age"] <= 18 else ("Female" if r["Sex"] == "female" else "Male"),
    axis=1
)

#Ensure 'Survived', 'Pclass', 'Age', 'SexNum' are numeric and handle NaNs after initial setup
for c in ["Survived", "Pclass", "Age", "SexNum"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["Survived", "Pclass", "Age", "SexNum"]) # Drop rows with NaNs in key columns
#-------------------------------------------------
# Program interfaces
#-------------------------------------------------
def main():
    print("\n" * 3)
    print("************* LIST SYSTEM PROJECT **********")
    print("************************************************")
    print("\n================ MAIN MENU ================")
    print("1. Content List Menu")
    print("2. Exit")
    print("========================================")
def menu():
    print("\n" * 2)
    print("\n============= LIST MENU ================")
    print("1. Descriptive Statistics (describe)")
    print("2. Hypothesis Testing (H0 / H1)")
    print("3. Linear Regression Model (All)")
    print("4. Logistic Regression (All)")
    print("5. Logistic Regression Separation")
    print("6. Survival Heatmap (Titanic)")
    print("7. Correlation Matrix (numeric)")
    print("8. Exit")
    print("========================================")
#-------------------------------------------------
# Program start
#-------------------------------------------------
main()
try:
    option = int(input("Please select your option(1-2): "))
except:
    option = 1
while option != 2:
    if option == 1:
        menu()
        try:
            sub_option = int(input("Please select your option(1-8): "))
        except:
            sub_option = 1
        while sub_option != 8:




            #===========================================================
            # 1) Descriptive Statistics
            #===========================================================
            if sub_option == 1:
                print("\n====== Descriptive Statistics (describe) ======\n")
                cf = pd.read_csv('titanic.csv', sep=',')
                print("\n====== Descriptive Statistics ======\n")
                print(cf.describe())
                print("\n------ Data Info ------")
                print(cf.info())
                print("\n------ Missing Values per Column ------")
                print(cf.isnull().sum())
            #===========================================================
            # 2) Hypothesis Testing (H₀ / H₁))
            #===========================================================
            elif sub_option == 2:
                print("\n====== Hypothesis Testing (H₀ / H₁) ======\n")
                print(" Null Hypothesis (H₀):")
                print("There is no significant relationship between passenger characteristics and survival.")
                print()
                print(" Alternative Hypothesis (H₁):")
                print("Passenger characteristics including gender, class, and age significantly affect survival chances.")
                print()
            #===========================================================
            # 3) Linear Regression Model
            #===========================================================
            elif sub_option == 3:
                print("\n====== Linear Regression Model (Titanic) ======\n")
                def classify_reg(row):
                    if row["Age"] < 18:
                        return 0
                    else:
                        return 1 if str(row.get("Sex","None")).lower() == "female" else 2
                df["RegCat"] = df.apply(classify_reg, axis=1)
                print("\nEnter your custom settings (press Enter to use default):\n")
                user_test_size = input("Enter test_size (default = 0.40): ")
                test_size = 0.40 if user_test_size.strip() == "" else float(user_test_size)
                user_random_state = input("Enter random_state (default = 42): ")
                random_state = 42 if user_random_state.strip() == "" else int(user_random_state)
                print(f"\nUsing: test_size = {test_size}, random_state = {random_state}\n")
                X = df[["Pclass", "RegCat"]]
                y = pd.to_numeric(df["Survived"], errors="coerce")
                mask = X.notnull().all(axis=1) & y.notnull()
                X = X[mask]
                y = y[mask]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                model = LinearRegression()
                model.fit(X_train, y_train)
                a = model.intercept_
                b1, b2 = model.coef_
                print("Regression Equation:")
                print(f"Y = {a:.4f} + {b1:.4f} * Pclass + {b2:.4f} * Category(RegCat)")
                score = model.score(X_test, y_test)
                print("\nModel Accuracy (R^2):", round(score, 4))
                print("\n[Note] Linear Regression used here for exploratory purposes on a binary target.")
            #===========================================================
            # 4) Logistic Regression (All) + ROC-AUC
            #===========================================================
            # ===== 4) Logistic Regression (ALL FEATURES TOGETHER) =====
            elif sub_option == 4:
                print("\n====== Logistic Regression (All Features Together) ======\n")
                # Ensure numeric + clean
                df_local = df.copy()
                df_local["Age"] = df_local["Age"].fillna(df_local["Age"].mean())
                df_local["Sex"] = df_local["Sex"].astype(str).str.strip().str.lower()
                df_local["SexNum"] = df_local["Sex"].map({"female": 1, "male": 0})
                for c in ["Survived", "Pclass", "Age", "SexNum"]:
                    df_local[c] = pd.to_numeric(df_local[c], errors="coerce")
                df_local = df_local.dropna(subset=["Survived", "Pclass", "Age", "SexNum"])
                # Features & target (ALL together)
                X = df_local[["Pclass", "Age", "SexNum"]].values
                y = df_local["Survived"].astype(int).values
                # Fixed split (no tuning)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.30, random_state=42, stratify=y
                )
                # Train Logistic Regression
                log_clf = LogisticRegression(max_iter=1000)
                log_clf.fit(X_train, y_train)
                # Predictions
                y_pred = log_clf.predict(X_test)


                y_proba = log_clf.predict_proba(X_test)[:, 1]
                # Metrics
                print("[Logistic Regression — All Features]")
                print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
                print("Precision:", round(precision_score(y_test, y_pred), 4))
                print("Recall :", round(recall_score(y_test, y_pred), 4))
                print("F1-score :", round(f1_score(y_test, y_pred), 4))
                print("ROC-AUC :", round(roc_auc_score(y_test, y_proba), 4))
                print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
                print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
                # ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                plt.figure()
                plt.plot(fpr, tpr, label=f"Logistic (AUC={roc_auc_score(y_test, y_proba):.3f})")
                plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve — Logistic (All Features)")
                plt.legend()
                plt.show()
            #===========================================================
            # 5) Logistic Regression Separation + ROC-AUC
            #===========================================================
            elif sub_option == 5:
                print("\n====== Logistic Regression Separation (Titanic) ======\n")
                df_sep = prep_numeric(df)
                features_sep = [
                    ('Age', 'Age'),
                    ('SexNum', 'Sex (0=Male, 1=Female)'),
                    ('Pclass', 'Pclass')
                ]
                for col, nice_name in features_sep:
                    data = df_sep[[col, 'Survived']].dropna().copy()
                    if data.empty:
                        print(f"Feature = {nice_name}: No valid data after removing NaN.\n")
                        continue
                    X = data[[col]].values
                    y = data['Survived'].astype(int).values
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.35, random_state=42, stratify=y
                    )
                    print("="*70)
                    print(f"Feature = {nice_name} (n={len(data)})")
                    print("="*70)
                    # Logistic Regression
                    log_model = LogisticRegression(max_iter=1000)
                    log_model.fit(X_train, y_train)
                    y_pred_log = log_model.predict(X_test)
                    print("[Logistic Regression]")
                    print("Accuracy:", round(accuracy_score(y_test, y_pred_log), 4))
                    print(classification_report(y_test, y_pred_log, digits=4))
                    # ROC–AUC + ROC curve
                    y_proba = log_model.predict_proba(X_test)[:, 1]
                    print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 4))
                    RocCurveDisplay.from_estimator(log_model, X_test, y_test)
                    plt.title(f'ROC Curve – {nice_name}')
                    plt.show()
                    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
                    print("-"*70)
                    # Decision Tree
                    tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
                    tree_model.fit(X_train, y_train)
                    y_pred_tree = tree_model.predict(X_test)
                    print("[Decision Tree]")
                    print("Accuracy:", round(accuracy_score(y_test, y_pred_tree), 4))
                    print(classification_report(y_test, y_pred_tree, digits=4))
                    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
                    print("="*70, "\n")
            #===========================================================
            # 6) Survival Heatmap
            #===========================================================
            elif sub_option == 6:
                print("\n====== Survival Heatmap (Titanic) ======\n")
                heatmap_data = df.pivot_table(
                    values="Survived",
                    index="Pclass",
                    columns="Category",
                    aggfunc="mean"
                ) * 100
                sns.heatmap(heatmap_data, annot=True, cmap="coolwarm_r", fmt=".1f")
                plt.title("Survival Heatmap (Pclass × Category)")
                plt.xlabel("Category")
                plt.ylabel("Pclass")
                plt.show()
            #===========================================================
            # 7) Correlation Matrix (numeric)
            #===========================================================
            elif sub_option == 7:

                print("\n====== Correlation Matrix (numeric) ======\n")

                # Use SexNum (already prepared globally) for consistency
                num_cols = ['Survived', 'Pclass', 'Age', 'SexNum']
                corr = df[num_cols].corr()

                #Print correlation values
                print(corr.round(3))
                print()

                #Plot correlation heatmap
                plt.figure(figsize=(6, 5))
                sns.heatmap(corr, annot=True, cmap="coolwarm_r", vmin=-1, vmax=1)
                plt.title("Correlation Matrix")
                plt.show()

                # Features / Target - using the globally prepared df which now has SexNum
                X = df[["Pclass", "Age", "SexNum"]].values
                y = df["Survived"].astype(int).values

                # Fixed split (no tuning)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.30, random_state=42, stratify=y )
                #Add model training and probability prediction hereLogistic Regression
                log_model = LogisticRegression(max_iter=1000)
                log_model.fit(X_train, y_train)
                y_proba_log = log_model.predict_proba(X_test)[:, 1]

                # Decision Tree
                tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
                tree_model.fit(X_train, y_train)
                y_proba_tree = tree_model.predict_proba(X_test)[:, 1]
                print("\n\n\n\n")
                #================== ROC Curves (both models) ==================
                fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
                fpr_tree, tpr_tree, _ = roc_curve(y_test, y_proba_tree)

                plt.figure()
                plt.plot(fpr_log,  tpr_log,  label=f"Logistic (AUC={roc_auc_score(y_test, y_proba_log):.3f})")
                plt.plot(fpr_tree, tpr_tree, label=f"Decision Tree (AUC={roc_auc_score(y_test, y_proba_tree):.3f})")
                plt.plot([0,1],[0,1], linestyle="--", label="Chance")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curves")
                plt.legend()
                plt.show()

            else:
                print("Invalid input!")
            menu()
            try:
                sub_option = int(input("Please select your option(1-8): "))
            except:
                sub_option = 8
            if sub_option == 8:
                print("\nThanks for using our program.")
                option = 2
                break
    else:
        print("Invalid input!")
    main()
    try:
        option = int(input("Please select your option(1-2): "))
    except:
        option = 2
print("\nThanks for using our program.")
