# Let's get started and get the full grade as A PROFESSIONAL ENGINEER :)

# Student Name, ID(  Hassan Al-Zahrani,451401862  __ Yasser Abdullah Alsaidlani,451401188 __ Hashem Jamal Alsaidalani,451400324 __Mohammed Abdulla Alshehri,451400125 )

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_score, recall_score, f1_score, r2_score)

#-------------------------------------------------
# Read the file + DataFrame
#-------------------------------------------------

df = pd.read_csv("titanic.csv", sep=';')

# Preparing Ages
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Prepare SexNum for numeric representation (Moved here for consistency)
df['Sex'] = df['Sex'].astype(str).str.strip().str.lower()
sex_map = {'female': 1, 'male': 0}
df['SexNum'] = df['Sex'].map(sex_map)

# Preparing the classification for the heatmap
df["Category"] = df.apply(
    lambda r: "Child" if r["Age"] <= 18 else ("Female" if r["Sex"] == "female" else "Male"),
    axis=1
)

# Ensure 'Survived', 'Pclass', 'Age', 'SexNum' are numeric and handle NaNs after initial setup
for c in ["Survived", "Pclass", "Age", "SexNum"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["Survived", "Pclass", "Age", "SexNum"]) # Drop rows with NaNs in key columns

#-------------------------------------------------
# Program interfaces
#-------------------------------------------------
def main():
    print("\n" * 5)
    print("*************LIST*******SYSTEM*******PROJECT**********")
    print("**************************************************************")
    print("\n================MAIN MENU================")
    print("1. List Menu")
    print("2. Exit")
    print("=========================================")


def menu():
    print("\n" * 3)
    print("\n============= LIST MENU ================")
    print("1. Descriptive Statistics (describe)")
    print("2. Hypothesis Testing (H0 / H1)")
    print("3. Linear Regression Model (All)")
    print("4. Logistic Regression (All)")
    print("5. Logistic Regression Separation")
    print("6. Survival Heatmap (Titanic)")
    print("7. Correlation Matrix (numeric)")
    print("8. Back To The Main Menu")
    print("=========================================")


#-------------------------------------------------
# Program start
#-------------------------------------------------

main()
option = int(input("Please select your option(1-2): "))

while option != 2:     # Exit
    if option == 1:
        menu()
        print("\n*** PLEASE SCROLL DOWN TO ENTER YOUR OPTION ***\n") # Added explicit message
        sub_option = int(input("Please select your option(1-8): "))

        # 8 = Back to main menu
        while sub_option != 8:

            #===========================================================
            # 1) Descriptive Statistics
            #===========================================================
            if sub_option == 1:
                cf = pd.read_csv("titanic(AutoRecovered).csv", sep=';') 
                print("\n====== Descriptive Statistics ======\n")
                print(cf.describe())
                print("\n------ Data Info ------")
                cf.info()
                print("\n------ Missing Values per Column ------")
                print(cf.isnull().sum())


            #===========================================================
            # 2) Hypothesis Testing (H₀ / H₁)
            #===========================================================
            elif sub_option == 2:
                print("\n====== Hypothesis Testing (H₀ / H₁) ======\n")

                print("    Null Hypothesis (H₀):")
                print("There is no significant relationship between passenger characteristics and survival.")
                print()
                print("    Alternative Hypothesis (H₁):")
                print("Passenger characteristics including gender, class, and age significantly affect survival chances.")
                print()

            #===========================================================
            # 3) Linear Regression Model
            #===========================================================
            elif sub_option == 3:
                print("\n====== Linear Regression Model (Titanic) ======\n")

                # Classify function for regression category
                def classify_reg(row):
                    if row["Age"] < 18:
                        return 0
                    else:
                        return 1 if row["Sex"] == "female" else 2

                df["RegCat"] = df.apply(classify_reg, axis=1)

                # Enter user settings
                print("\nEnter your custom settings (press Enter to use default):\n")

                user_test_size = input("Enter test_size (example = 0.40): ")
                if user_test_size.strip() == "":
                    test_size = 0.40
                else:
                    test_size = float(user_test_size)

                user_random_state = input("Enter random_state (example = 42): ")
                if user_random_state.strip() == "":
                    random_state = 42
                else:
                    random_state = int(user_random_state)

                print(f"\nUsing: test_size = {test_size}, random_state = {random_state}\n")

                # Selecting variables
                X = df[["Pclass", "RegCat"]]
                y = df["Survived"]

                # Data segmentation
                X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size, random_state=random_state)

                # Model training
                model = LinearRegression()
                model.fit(X_train, y_train)

                # The equation
                a = model.intercept_
                b1, b2 = model.coef_

                print("Regression Equation:")
                print(f"Y = {a:.4f} + {b1:.4f} * Pclass + {b2:.4f} * Category")

                # Accuracy
                score = model.score(X_test, y_test)
                print("\nModel Accuracy (R^2):", round(score, 4))

            #===========================================================
            # 4) Logistic Regression (All)
            #===========================================================
            elif sub_option == 4:
                print("\n====== Logistic Regression (All) (Titanic) ======\n")

                # Use a copy of df to ensure local modifications
                df_local = df.copy()

                features = [
                    ('Age',    'Age'),
                    ('SexNum', 'Sex (0=Male, 1=Female)'),
                    ('Pclass', 'Pclass')
                ]

                for col, nice_name in features:

                    data = df_local[[col, 'Survived']].dropna().copy()
                    if data.empty:
                        print(f"Feature = {nice_name}: No valid data after removing NaN.\n")
                        continue

                    X = data[[col]].values
                    y = data['Survived'].astype(int).values

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.35, random_state=42, stratify=y
                    )

                    # Logistic Regression
                    log_model = LogisticRegression(max_iter=1000)
                    log_model.fit(X_train, y_train)
                    y_pred_log = log_model.predict(X_test)
                    y_proba_log = log_model.predict_proba(X_test)[:, 1]

                    print("="*70)
                    print(f"Feature = {nice_name}  (n={len(data)})")
                    print("-"*70)

                    print("[Logistic Regression]")
                    print("Accuracy:", round(accuracy_score(y_test, y_pred_log), 4))
                    print(classification_report(y_test, y_pred_log, digits=4))
                    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
                    print("ROC-AUC (Logistic):", round(roc_auc_score(y_test, y_proba_log), 4))
                    print("-"*70)

                    # Decision Tree
                    tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
                    tree_model.fit(X_train, y_train)
                    y_pred_tree = tree_model.predict(X_test)
                    y_proba_tree = tree_model.predict_proba(X_test)[:, 1]

                    print("[Decision Tree]")
                    print("Accuracy:", round(accuracy_score(y_test, y_pred_tree), 4))
                    print(classification_report(y_test, y_pred_tree, digits=4))
                    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
                    print("ROC-AUC (Decision Tree):", round(roc_auc_score(y_test, y_proba_tree), 4))
                    print("="*70, "\n")

            #===========================================================
            # 5) Logistic Regression Separation
            #===========================================================
            elif sub_option == 5:
                print("\n====== Logistic Regression Separation (Titanic) ======\n")

                # Use a copy of df to ensure local modifications
                df_local = df.copy()

                features_sep = [
                    ('Age',    'Age'),
                    ('SexNum', 'Sex (0=Male, 1=Female)'),
                    ('Pclass', 'Pclass')
                ]

                for col, nice_name in features_sep:

                    data = df_local[[col, 'Survived']].dropna().copy()
                    if data.empty:
                        print(f"Feature = {nice_name}: No valid data after removing NaN.\n")
                        continue

                    X = data[[col]].values
                    y = data['Survived'].astype(int).values

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.35, random_state=42, stratify=y
                    )

                    # Logistic Regression
                    log_model = LogisticRegression(max_iter=1000)
                    log_model.fit(X_train, y_train)
                    y_pred_log = log_model.predict(X_test)
                    y_proba_log = log_model.predict_proba(X_test)[:, 1]

                    print("="*70)
                    print(f"Feature = {nice_name}  (n={len(data)})")
                    print("="*70)

                    print("[Logistic Regression]")
                    print("Accuracy:", round(accuracy_score(y_test, y_pred_log), 4))
                    print(classification_report(y_test, y_pred_log, digits=4))
                    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
                    print("ROC-AUC (Logistic):", round(roc_auc_score(y_test, y_proba_log), 4))
                    print("-"*70)

                    # Decision Tree
                    tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
                    tree_model.fit(X_train, y_train)
                    y_pred_tree = tree_model.predict(X_test)
                    y_proba_tree = tree_model.predict_proba(X_test)[:, 1]

                    print("[Decision Tree]")
                    print("Accuracy:", round(accuracy_score(y_test, y_pred_tree), 4))
                    print(classification_report(y_test, y_pred_tree, digits=4))
                    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
                    print("ROC-AUC (Decision Tree):", round(roc_auc_score(y_test, y_proba_tree), 4))
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
            #7) Correlation Matrix (numeric)
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

                # Add model training and probability prediction here
                # Logistic Regression
                log_model = LogisticRegression(max_iter=1000)
                log_model.fit(X_train, y_train)
                y_proba_log = log_model.predict_proba(X_test)[:, 1]

                # Decision Tree
                tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
                tree_model.fit(X_train, y_train)
                y_proba_tree = tree_model.predict_proba(X_test)[:, 1]

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
            print("\n*** PLEASE SCROLL DOWN TO ENTER YOUR OPTION ***\n") # Added explicit message
            sub_option = int(input("Please select your option(1-8): "))


    else:
        print("Invalid input!")

    main()
    option = int(input("Please select your option(1-2): "))

print("\nThanks for using our program.")
