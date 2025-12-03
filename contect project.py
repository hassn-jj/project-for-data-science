# Let's get started and get the full grade as A PROFESSIONAL ENGINEER :)

# Student Name, ID(  Hassan Al-Zahrani,451401862  __ Yasser Abdullah Alsaidlani,451401188 __ Hashem Jamal Alsaidalani,451400324 __Mohammed Abdulla Alshehri,451400125 )

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#-------------------------------------------------
# Read the file + DataFrame
#-------------------------------------------------

df = pd.read_csv("titanic.csv", sep=';')

# Preparing Ages
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Preparing the classification for the heatmap
df["Category"] = df.apply(
    lambda r: "Child" if r["Age"] <= 18 else ("Female" if r["Sex"] == "female" else "Male"),
    axis=1
)

#-------------------------------------------------
# Program interfaces
#-------------------------------------------------
def main():
    print("\n" * 5)
    print("******CONTENT*******LIST*******SYSTEM*******PROJECT**********")
    print("**************************************************************")
    print("\n================MAIN MENU================")
    print("1. Content List Menu")
    print("2. Exit")
    print("=========================================")


def menu():
    print("\n" * 3)
    print("\n================CONTENT LIST MENU================")
    print("1. Descriptive Statistics (describe)")
    print("2. Survival Heatmap (Titanic)")
    print("3. Linear Regression Model (All)")
    print("4. Logistic Regression (All)")
    print("5. Logistic Regression Separation")
    print("6. Hypothesis Testing (H0 / H1)")
    print("7. Back To The Main Menu")
    print("=========================================")


#-------------------------------------------------
# Program start
#-------------------------------------------------

main()
option = int(input("Please select your option(1-2): "))

while option != 2:     # Exit
    if option == 1:
        menu()
        sub_option = int(input("Please select your option(1-7): "))

        # 7 = Back to main menu
        while sub_option != 7:

            #===========================================================
            # 1) Descriptive Statistics
            #===========================================================
            if sub_option == 1:
                print("\n====== Descriptive Statistics ======\n")
                print(df.describe())
                print("\n------ Data Info ------")
                df.info()
                print("\n------ Missing Values per Column ------")
                print(df.isnull().sum())

            #===========================================================
            # 2) Survival Heatmap
            #===========================================================
            elif sub_option == 2:
                print("\n====== Survival Heatmap (Titanic) ======\n")

                heatmap_data = df.pivot_table(
                    values="Survived",
                    index="Pclass",
                    columns="Category",
                    aggfunc="mean"
                ) * 100

                sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".1f")
                plt.title("Survival Heatmap (Pclass × Category)")
                plt.xlabel("Category")
                plt.ylabel("Pclass")
                plt.show()

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

                user_test_size = input("Enter test_size (default = 0.40): ")
                if user_test_size.strip() == "":
                    test_size = 0.40
                else:
                    test_size = float(user_test_size)

                user_random_state = input("Enter random_state (default = 42): ")
                if user_random_state.strip() == "":
                    random_state = 42
                else:
                    random_state = int(user_random_state)

                print(f"\nUsing: test_size = {test_size}, random_state = {random_state}\n")

                # Selecting variables
                X = df[["Pclass", "RegCat"]]
                y = df["Survived"]

                # Data segmentation
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

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

                # Working version
                df = df.copy()

                # Converting variables to numbers
                df['Survived'] = pd.to_numeric(df['Survived'], errors='coerce')
                df['Pclass']   = pd.to_numeric(df['Pclass'],   errors='coerce')
                df['Age']      = pd.to_numeric(df['Age'],      errors='coerce')

                df['Sex'] = df['Sex'].astype(str).str.strip().str.lower()
                sex_map = {'female': 1, 'male': 0}
                df['SexNum'] = df['Sex'].map(sex_map)

                features = [
                    ('Age',    'Age'),
                    ('SexNum', 'Sex (0=Male, 1=Female)'),
                    ('Pclass', 'Pclass')
                ]

                for col, nice_name in features:

                    data = df[[col, 'Survived']].dropna().copy()
                    if data.empty:
                        print(f"Feature = {nice_name}: No valid data after removing NaN.\n")
                        continue

                    X = data[[col]].values
                    y = data['Survived'].astype(int).values

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.35, random_state=42, stratify=y
                    )

                    print("="*70)
                    print(f"Feature = {nice_name}  (n={len(data)})")
                    print("-"*70)

                    # Logistic Regression
                    log_model = LogisticRegression(max_iter=1000)
                    log_model.fit(X_train, y_train)
                    y_pred_log = log_model.predict(X_test)

                    print("[Logistic Regression]")
                    print("Accuracy:", round(accuracy_score(y_test, y_pred_log), 4))
                    print(classification_report(y_test, y_pred_log, digits=4))
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
            # 5) Logistic Regression Separation
            #===========================================================
            elif sub_option == 5:
                print("\n====== Logistic Regression Separation (Titanic) ======\n")

                df_sep = df.copy()

                df_sep['Survived'] = pd.to_numeric(df_sep['Survived'], errors='coerce')
                df_sep['Pclass']   = pd.to_numeric(df_sep['Pclass'],   errors='coerce')
                df_sep['Age']      = pd.to_numeric(df_sep['Age'],      errors='coerce')

                df_sep['Sex'] = df_sep['Sex'].astype(str).str.strip().str.lower()
                sex_map = {'female': 1, 'male': 0}
                df_sep['SexNum'] = df_sep['Sex'].map(sex_map)

                features_sep = [
                    ('Age',    'Age'),
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
                    print(f"Feature = {nice_name}  (n={len(data)})")
                    print("="*70)

                    # Logistic Regression
                    log_model = LogisticRegression(max_iter=1000)
                    log_model.fit(X_train, y_train)
                    y_pred_log = log_model.predict(X_test)

                    print("[Logistic Regression]")
                    print("Accuracy:", round(accuracy_score(y_test, y_pred_log), 4))
                    print(classification_report(y_test, y_pred_log, digits=4))
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
            # 6) Hypothesis Testing (H₀ / H₁)
            #===========================================================
            elif sub_option == 6:
                print("\n====== Hypothesis Testing (H₀ / H₁) ======\n")

                print("    Null Hypothesis (H₀):")
                print("There is no significant relationship between passenger characteristics and survival.")
                print()
                print("    Alternative Hypothesis (H₁):")
                print("Passenger characteristics including gender, class, and age significantly affect survival chances.")
                print()

            else:
                print("Invalid input!")

            menu()
            sub_option = int(input("Please select your option(1-7): "))

    else:
        print("Invalid input!")

    main()
    option = int(input("Please select your option(1-2): "))

print("\nThanks for using our program.")
