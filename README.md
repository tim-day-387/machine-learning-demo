# machine-learning-demo
This is a demonstration of Python's machine learning tools available in a Graphical User Interface. 

# Overview
This program is intended to demonstrate a simple form of machine learning. The user is able to utilize a GUI in order to load data from a file, visualize/summarize the data, compare various machine learning models, and finally select a model in order to train it to categorize data. 

# Testing
A file "TestData.csv" was included for the purposes of testing. The data represents 30,000 randomly generated quadratic equations of the form Ax^2 + Bx + C = 0. The labels indicate which interval the positive root of this equation falls, found by the equation: (-B + sqrt((B^2) - 4AC) / 2A).

The file must first be loaded into the software. Insert the follow text into the appropriate text boxes and click Load File:

TestData.csv -----> File Path
['A', 'B', 'C', 'Positive_Root'] -----> Header

Now the data has been loaded. Now, you may display various charts and data summarizations using the dropdown menus and the Display and Graph buttons. 

Next, the different models may be evaluated by pressing the Model Evaluation button. This will display five different machine learning models and their performance on the given data. A plot of this data can be seen by using Evaluation Plot. The best model may be selected and train using the dropdown and Train Model button. Then, the performance of the model may be validated (with a portion of the data that has been reserved) using the Validate Model button. 

Finally, the model may be used to make a prediction. Simply input a data point of the form:

[A, B, C]

and press the Make Prediction button. Try the following data points:

[33, 40, -23]
[-9, 19, -22]
[28, -43, -23]

Feel free to try your own data. Format your .csv and Header similarly to the ones provided. 
