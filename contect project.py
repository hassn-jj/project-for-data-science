#print("hi")
# Let's get started and get the full grade as A PROFESSIONAL ENGINEER :) 

# Student Name, ID(  Hassan Al-Zahrani,451401862  __ Yasser Abdullah Alsaidlani,451401188 __  )

import csv

# --- Load CSV file ---
infile = open("project for data science.csv", "r", encoding="utf-8-sig")
reader = csv.DictReader(infile)
records = list(reader)

def main():
    # Clear screen
    print("\n" * 10)

    # Display header
    print("*******CONTENT*******LIST*******SYSTEM*******PROJECT*******")
    print("************************************************************")
    print("\n==================MAIN MENU==================")
    
    # Menu options
    print("1. Content List Menu")
    print("2. Exit")
    print("============================================")


def menu():
    print("\n==================CONTENT LIST MENU==================")
    print("1. Show all content")
    print("2. Add New Contact")
    print("3. Search Contact")
    print("4. Delete Contact")
    print("5. Back to Main Menu")
    print("=====================================================")


main()
option = int(input("Please select your option(1-2): "))

while option != 2:
    if option == 1:
        menu()
        option = int(input("Please select your option(1-5): "))
        while option != 5:
            if option == 1:  
                for i in range(0,6):                           
                    
                    record = records[i].strip().split(";")
                    #for x in range(2, 30, 3):


                    if i != 0 and len(record)==3:
                        print("Name:", record[0])
                        print("Phone:", record[1])
                        print("Email:", record[2], "\n")
    

            elif option == 2:  
                print("\n====== Add New Contact ======")     
                Name = input("Enter Name: ")
    else:
        print("Oh be cearful!")
        break
        

