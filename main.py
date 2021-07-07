#!/user/bin/env python3

# main.py
# Timothy G. Day

## Load Libraries ##

# GUI libraries
from tkinter import *
from tkinter import ttk

# ML/Data libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# General libraries
import sys
import time

## Variables ##

# Variables
dataset = None                                                                       # Stores the dataset from the .csv
labelName = None                                                                     # Name for the label (y) column
models = []                                                                          # Array to store the various models
results = []                                                                         # Stores the results of the models
names = []                                                                           # Stores the names of the models
X_train = None                                                                       # X training data
X_validation = None                                                                  # X validation data
y_train = None                                                                       # y training data
y_validation = None                                                                  # y validation data
model = None                                                                         # The selected model to be trained
val_pred = None                                                                      # Validation predictions                    

# Flags
fileLoaded = False                                                                   # Is the file loaded?
modelsEval = False                                                                   # Have the models been evaluated?
modelTrain = False                                                                   # Has the selected model been trained?

# Define what to perform on program exit
def onExit():
    print("----- Quitting! -----")                                                   # Notify the user that the program is ending
    root.update_idletasks()                                                          
    root.update()                                                                    
    time.sleep(1)                                                                    
    root.destroy()                                                                   
    sys.exit("Terminated")                                                           # End the program execution

# Load file and assign column headers
def loadFile():
    global dataset                                                                    
    global labelName                                                                 
    global fileLoaded                                                                
    global X_train                                                                   
    global X_validation                                                              
    global y_train                                                                   
    global y_validation                                                             

    # Save the header input as a list into the names variable    
    try:
        names = eval(gui.header.get())
        labelName = names[-1]
    except:
        print("Could not read CSV headers!")
        return
    
    # Get data from CSV
    try:
        filePath = gui.filePath.get()
        dataset = read_csv(filePath, names=names)
    except:
        print("Could not read filepath!")
        return

    # Create Validation Dataset
    array = dataset.values                                  
    X = array[:,0:-1]                                       
    y = array[:,-1]                                         
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.20, random_state = 1)  

    # Success!
    fileLoaded = True                                                                
    print("The file was loaded successfully!")                                       

# Display the selected view
def display():
    global fileLoaded                                                                
    selection = gui.dispOpt.get()                                          # Save the display option into a local variable

    if(fileLoaded == False):                                               # Check if file is loaded
        print("Please load a file first.")                                           
        return                                                                       
    
    if(selection == "Data Shape"):
        print("Data Shape")                                                          
        print(dataset.shape)                                                         
        print("\n")                                                                  
    if(selection == "Data Sample"):
        print("Sample of Data")                                                      
        print(dataset.head(20))                                                      
        print("\n")                                                        
    if(selection == "Data Description"):
        print("Description")                                               
        print(dataset.describe())                                          
        print("\n")                                                        
    if(selection == "Data Grouping"):
        print("Data Grouping")                                             
        print(dataset.groupby(labelName).size())                           
        print("\n")                                                        

# Graph the selected view
def graph():
    global fileLoaded                                                      
    global dataset                                                         
    selection = gui.graphOpt.get()                                                   # Save the graph option into a local variable
    
    if(fileLoaded == False):                                                         # Check if file is loaded
        print("Please load a file first.")                                 
        return                                                             

    if(selection == "Box/Whisker"):
        dataset.plot(kind = 'box', subplots = True, sharex = False, sharey = False)  # Create Box/Whisker plot
        pyplot.show()                                                                
    if(selection == "Histograms"):
        dataset.hist()                                                               # Create the histograms 
        pyplot.show()                                                               
    if(selection == "Scatter Plot Matrix"):
        scatter_matrix(dataset)                                                      # Create a scatter matrix
        pyplot.show()                                                               
        
# Complete the model evaluation 
def modelEval():
    global modelsEval                                                               
    global fileLoaded                                                               
    global models                                                                   
    global results                                                                  
    global names                                                                    
    global X_train                                                                  
    global X_validation                                                             
    global y_train                                                                  
    global y_validation                                                             
    
    if(fileLoaded == False):                                                         # Check if file is loaded
        print("Please load a file first.")                                           
        return                                                                       

    # Select the  Different Models
    models.append(('Log. Regression', LogisticRegression(solver = 'liblinear', multi_class = 'ovr'))) 
    models.append(('Lin. Disc. Analysis', LinearDiscriminantAnalysis()))                              
    models.append(('K-Nearest Neighbors', KNeighborsClassifier()))                                    
    models.append(('Class. Regress. Trees', DecisionTreeClassifier()))                                
    models.append(('Gauss. Naive Bayes', GaussianNB()))                                               

    # Evaluate the different models
    for name, model in models:
        kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)    
        cv_results = cross_val_score(model, X_train, y_train, cv = kfold, scoring = 'accuracy') 
        results.append(cv_results)                                                                             
        names.append(name[0:3])                                                                                
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))                                     

    modelsEval = True                                                                # The models have been evaluated

# Train the selected model on the data
def trainMod():
    global modelTrain                                                                
    global fileLoaded                                                                
    global model                                                                     
    selection = gui.modelOpt.get()                                                   # Save the model option into a local variable
    
    if(fileLoaded == False):                                                         # Check if file is loaded
        print("Please load a file first.")                                           
        return                                                                       

    if(selection == "Log. Regression"):
        model = LogisticRegression(solver = 'liblinear', multi_class = 'ovr')        
    if(selection == "Lin. Disc. Analysis"):
        model = LinearDiscriminantAnalysis()                                         
    if(selection == "K-Nearest Neighbors"):
        model = KNeighborsClassifier()                                               
    if(selection == "Class. Regress. Trees"):
        model = DecisionTreeClassifier()                                             
    if(selection == "Gauss. Naive Bayes"):
        model = GaussianNB()                                                         

    model.fit(X_train, y_train)                                                      # Train the model on the given data
    print("The training was successful!")                                            

    modelTrain = True                                                                # The model has been trained

# Displays the plots for the model evaluation
def evalPlots():
    global modelsEval                                                                
    global results                                                                   
    global names                                                                     

    if(modelsEval == False):                                                         # Check if models have been evaluated
        print("Please evaluate the models first.")                                   
        return                                                                       
    
    pyplot.boxplot(results, labels=names)                                            # Create a boxplot
    pyplot.title('Algorithm Comparison')                                             
    pyplot.show()                                                                    

# Validates that the model is working correctly
def modelValid():
    global modelTrain                                                                
    global val_pred                                                                  
    global model                                                                     
    global y_validation                                                              
    global X_validation                                                              

    if(modelTrain == False):                                                         # Check if a model has been trained
        print("Please select a model and train it first.")                           
        return                                                                       
    
    val_pred = model.predict(X_validation)                                           # Make the predicts using the model
    
    print("Accuracy Score")                                                          
    print(accuracy_score(y_validation, val_pred))                                    
    print("\n")                                                                      
    print("Confusion Matrix")                                                        
    print(confusion_matrix(y_validation, val_pred))                                  
    print("\n")                                                                      
    print("Classification Report")                                                   
    print(classification_report(y_validation, val_pred, zero_division=0))            
    print("\n")                                                                      

# Make a prediction on a single peice of data
def predSingle():
    global modelTrain                                                                
    global model                                                                     

    if(modelTrain == False):                                                         # Check if a model has been trained
        print("Please select a model and train it first.")                           
        return                                                                       

    # Save the data box as a local
    data = [eval(gui.data.get())]                                                   
    
    print("Prediction: " + str(model.predict(data)))                                 # Make the prediction

# Used to redirect stdout and stderr 
class StdRedirector():                                                        
    # Constructor
    def __init__(self, text_widget):
        self.text_space = text_widget                                                # text_space attribute

    # Alternative Write Method (inserts text in text_widget)
    def write(self, string):
        self.text_space.config(state = NORMAL)                                       # config state NORMAL
        self.text_space.insert("end", string)                                        # insert string
        self.text_space.see("end")                                                   # end
        self.text_space.config(state = DISABLED)                                     # config state DISABLE

# Creates the GUI for the program        
class GUI:
    # Constructor
    def __init__(self, master):
        #Variables
        self.master = master                                                         # tk window attribute
        self.filePath = StringVar()                                                  # for user input filePath
        self.header = StringVar()                                                    # for user input header
        self.data = StringVar()                                                      # for user input data
        self.dispOpt = StringVar()                                                   # for display option drop down
        self.graphOpt = StringVar()                                                  # for graph option drop down
        self.modelOpt = StringVar()                                                  # for model option drop down

        dispCho = {"Data Shape", "Data Sample", "Data Description", "Data Grouping"} # Eligable display choices
        self.dispOpt.set("Data Shape")                                               # Set the default option

        graphCho = {"Box/Whisker", "Histograms", "Scatter Plot Matrix"}              # Eligable graph choices
        self.graphOpt.set("Box/Whisker")                                             # Set the default option

        modelCho = {"Log. Regression", "Lin. Disc. Analysis", "K-Nearest Neighbors", "Class. Regress. Trees", "Gauss. Naive Bayes"} # Eligable model choices
        self.modelOpt.set("Log. Regression")                                                                                        # Set the default option

        # Give window title   
        master.title ("Machine Learning Applet")                                     # Give window title
        master.resizable(False, False)                                               # Make window not resizable

        # Create window
        self.mainframe = Frame(master, height = 200, width = 500)                    # Set window size
        self.mainframe.pack_propagate(0)                                             # Set propogation

# Terminal Heading

        # Terminal title
        self.term_label = Label(master, text = "Terminal", font = ("Helvetica", 16))             # Create a Terminal title
        self.term_label.grid(row = 0, column = 0, columnspan = 5)                                # Position that title

        # Terminal screen 
        self.text_box = Text(master, state = DISABLED)                                           # Create text box
        self.text_box.grid(row = 1, column = 0, columnspan = 5, padx = 10)                       # Pad the new text box

        sys.stdout = StdRedirector(self.text_box)                                                # Redirect terminal output
        sys.stderr = StdRedirector(self.text_box)                                                # Redirect terminal output

# Data Selection Heading

        # Selection title
        self.cont_label = Label(master, text = "Data Selection", font = ("Helvetica", 16))       # Create a selection title
        self.cont_label.grid(row = 2, column = 0, columnspan = 5)                                # Position that title

        # Data summary dropdown 
        self.disp_menu = OptionMenu(master, self.dispOpt, *dispCho)                              # Creates the display menu
        self.disp_menu.config(width = 25)                                                        # Maintain a consistent width
        self.disp_menu.grid(row = 3, column = 2, sticky = E, padx = 10, pady = 10)               # Position display menu

        # Data summary button
        self.disp_button = Button(master, width = 25, text = "Display", command = display)       # Creates a display button
        self.disp_button.grid(row = 3, column = 1, sticky = W, padx = 10, pady = 10)             # Position display button

        # Graph selection dropdown 
        self.graph_menu = OptionMenu(master, self.graphOpt, *graphCho)                           # Creates the graph menu
        self.graph_menu.config(width = 25)                                                       # Maintain a consistent width
        self.graph_menu.grid(row = 4, column = 2, sticky = E, padx = 10, pady = 10)              # Position graph menu

        # Graph selection button
        self.graph_button = Button(master, width = 25, text = "Graph", command = graph)          # Creates a graph button
        self.graph_button.grid(row = 4, column = 1, sticky = W, padx = 10, pady = 10)            # Position graph button

        # File Path input box
        self.fp_box = ttk.Entry(master, width = 25, textvariable = self.filePath)                # Create input box
        self.fp_box.grid(row = 3, column = 4, sticky = W, padx = 10, pady = 10)                  # Position the input box
        self.fp_label = Label(master, text = "File Path:")                                       # Create a label for input box
        self.fp_label.grid(row = 3, column = 3, sticky = E)                                      # Position that label 

        # Header input box
        self.hd_box = ttk.Entry(master, width = 25, textvariable = self.header)                  # Create input box
        self.hd_box.grid(row = 4, column = 4, sticky = W, padx = 10, pady = 10)                  # Position the input box
        self.hd_label = Label(master, text = "Header:")                                          # Create a label for input box
        self.hd_label.grid(row = 4, column = 3, sticky = E)                                      # Position that label 

        # Load File button 
        self.load_button = Button(master, width = 25, text = "Load File", command = loadFile)    # Loads the csv file from the filePath
        self.load_button.grid(row = 3, column = 0, sticky = W, padx = 10, pady = 10)             # Position load button

        # Exit button 
        self.exit_button = Button(master, width = 25, text = "Quit", command = onExit)           # Create exit button 
        self.exit_button.grid(row = 4, column = 0, sticky = W, padx = 10, pady = 10)             # Position exit button

# Model Training Heading

        # Model training title
        self.model_label = Label(master, text = "Model Training", font = ("Helvetica", 16))      # Create a model training title
        self.model_label.grid(row = 5, column = 0, columnspan = 5)                               # Position that title

        # Model evaluation button 
        self.model_button = Button(master, width = 25, text = "Model Evaluation", command = modelEval)    # Create exit button 
        self.model_button.grid(row = 6, column = 0, sticky = W, padx = 10, pady = 10)                     # Position exit button

        # Model selection dropdown 
        self.model_menu = OptionMenu(master, self.modelOpt, *modelCho)                                    # Creates the model menu
        self.model_menu.config(width = 25)                                                                # Maintain a consistent width
        self.model_menu.grid(row = 6, column = 4, sticky = E, padx = 10, pady = 10)                       # Position model menu

        # Model selection button
        self.model_button = Button(master, width = 25, text = "Train Model", command = trainMod)          # Creates a model button
        self.model_button.grid(row = 6, column = 3, sticky = W, padx = 10, pady = 10)                     # Position model button

        # Evaluation plot button
        self.evalPlot_button = Button(master, width = 25, text = "Evaluation Plot", command = evalPlots)  # Creates a eval plot button
        self.evalPlot_button.grid(row = 6, column = 1, sticky = W, padx = 10, pady = 10)                  # Position button
        
        # Validation model button
        self.modelVal_button = Button(master, width = 25, text = "Validate Model", command = modelValid)  # Creates a model validate button
        self.modelVal_button.grid(row = 6, column = 2, sticky = W, padx = 10, pady = 10)                  # Position button

# Prediction Heading

        # Prediction title
        self.pred_label = Label(master, text = "Prediction", font = ("Helvetica", 16))                    # Create a prediction title
        self.pred_label.grid(row = 7, column = 0, columnspan = 5)                                         # Position that title

        # Validation model button
        self.pred_button = Button(master, width = 25, text = "Make Prediction", command = predSingle)     # Creates a prediction button
        self.pred_button.grid(row = 8, column = 2, sticky = W, padx = 10, pady = 10)                      # Position button

        # Data input box
        self.data_box = ttk.Entry(master, width = 25, textvariable = self.data)                  # Create input box
        self.data_box.grid(row = 8, column = 4, sticky = W, padx = 10, pady = 10)                # Position the input box
        self.data_label = Label(master, text = "Data:")                                          # Create a label for input box
        self.data_label.grid(row = 8, column = 3, sticky = E)                                    # Position that label

## GUI Mainloop ##
root = Tk()                                                                          # Make the root window
gui = GUI(root)                                                                      # Intialize the root window
root.mainloop()                                                                      # Start loop


