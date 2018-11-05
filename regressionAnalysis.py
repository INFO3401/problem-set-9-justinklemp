#Justin Klemperer
#INFO-3401
#Problem Set 9

#Collaborators: Steven, Lucas, Harold, Zach, Marissa


                                        #Monday:
#Imports
import csv
import pandas as pd
import numpy as np
import parser
import matplotlib
import matplotlib.pyplot as plt

#sklearn imports:
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score


#####PART A#####

class AnalysisData:
    def __init__(self):
        
        self.dataset = []
        self.variables = []     
    def parserFile(self, candy_file):
        
        self.dataset = pd.read_csv(candy_file)
        for column in self.dataset.columns.values:
            if column != "competitorname":
                
                self.variables.append(column)
            
            
#####PART B#####

class LinearAnalysis:
    def __init__(self, data_targetY):
        
        self.bestX = ""
        self.targetY = data_targetY
        self.fit = ""
        
    def runSimpleAnalysis(self, data):
        
        highest_sugar = ""
        highest_r2 = -1
        
        for column in data.variables:
            if column != self.targetY:
                
                data_variable = data.dataset[column].values
                data_variable = data_variable.reshape(len(data_variable),1)
                
                regr = LinearRegression()
                regr.fit(data_variable, data.dataset[self.targetY])
                variable_prediction = regr.predict(data_variable)
                r_score = r2_score(data.dataset[self.targetY],variable_prediction)
                
                if r_score > highest_r2:
                    
                    highest_r2 = r_score
                    highest_sugar = column
        self.bestX = highest_sugar
        print(highest_sugar, highest_r2)
        

#####PART C#####

class LogisticAnalysis:
    def __init__(self, data_targetY):
        self.bestX = ""
        self.targetY = data_targetY
        self.fit = ""
        

#####PROBLEM 1#####

candy_data_curation = AnalysisData()
candy_data_curation.parserFile('candy-data.csv')

#####PROBLEM 2#####
    #Attached in B & C
    
    
#----------
    

                                        #Wednesday//Friday:

#####PROBLEM 3#####

candy_line_analysis = LinearAnalysis('sugarpercent')
candy_line_analysis.runSimpleAnalysis(candy_data_curation)


#----------

#Link to additional source(s) we used to help:
#http://scikitlearn.org/stable/auto_examples/linear_model/plot_ols.html/https://dziganto.github.io/classes/data%20science/linear%20regression/machine%20learning/object-oriented%20programming/python/Understanding-Object-Oriented-Programming-Through-Machine-Learning/
