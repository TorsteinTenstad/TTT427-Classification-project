from lib import *

import pandas as pd

def task1():

    #Load data
    data_class1 = pd.read_csv("Data/class_1")
    data_class2 = pd.read_csv("Data/class_2")
    data_class3 = pd.read_csv("Data/class_3")

    #Define set for training and testing
    train_class1 = data_class1.iloc[:30, :]
    test_class1 = data_class1.iloc[30:50, :]

    train_class2 = data_class2.iloc[:30, :]
    test_class2 = data_class2.iloc[30:50, :]

    train_class3 = data_class3.iloc[:30, :]
    test_class3 = data_class3.iloc[30:50, :]


def main():
    print("======  task1  ======")
    task1()

main()