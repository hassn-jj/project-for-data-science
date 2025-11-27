# Let's get started and get the full grade as A PROFESSIONAL ENGINEER :) 

# Student Name, ID(  Hassan Al-Zahrani,451401862  __ Yasser Abdullah Alsaidlani,451401188 __ Hashem Jamal Alsaidalani,451400324 __Mohammed Abdulla Alshehri,451400125 )

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#-------------------------------------------------
# قراءة الملف كسطور + DataFrame
#-------------------------------------------------

df = pd.read_csv("titanic.csv", sep=';')

# تجهيز الأعمار
df["Age"] = df["Age"].fillna(df["Age"].mean())

# تجهيز التصنيف للهيت ماب
df["Category"] = df.apply(
    lambda r: "Child" if r["Age"] <= 18 else ("Female" if r["Sex"] == "female" else "Male"),
    axis=1
)

#-------------------------------------------------
# واجهات البرنامج
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
    print("3. Linear Regression Model (Prediction)")
    print("4. Back To The Main Menu")
    print("=========================================")


#-------------------------------------------------
# بداية البرنامج
#-------------------------------------------------

main()
option = int(input("Please select your option(1-2): "))

while option != 2:     # Exit
    if option == 1:
        menu()
        sub_option = int(input("Please select your option(1-4): "))

        while sub_option != 4:   # Back to main menu

            #===========================================================
            # 1) Descriptive Statistics
            #===========================================================
            if sub_option == 1:
                print("\n====== Descriptive Statistics ======\n")
                print(df.describe())

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

                # Classify function
                def classify(row):
                    if row["Age"] < 18:
                        return 0
                    else:
                        return 1 if row["Sex"] == "female" else 2

                df["RegCat"] = df.apply(classify, axis=1)

                # اختيار المتغيرات
                X = df[["Pclass", "RegCat"]]
                y = df["Survived"]

                # تقسيم البيانات
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.40, random_state=42
                )

                # تدريب النموذج
                model = LinearRegression()
                model.fit(X_train, y_train)

                # المعادلة
                a = model.intercept_
                b1, b2 = model.coef_

                print("Regression Equation:")
                print(f"Y = {a:.4f} + {b1:.4f} * Pclass + {b2:.4f} * Category")

                # الدقة
                score = model.score(X_test, y_test)
                print("\nModel Accuracy (R^2):", round(score, 4))

            else:
                print("Invalid input!")

            menu()
            sub_option = int(input("Please select your option(1-4): "))

    else:
        print("Invalid input!")

    main()
    option = int(input("Please select your option(1-2): "))

print("\nThanks for using our program.")
