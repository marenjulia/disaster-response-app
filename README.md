# Disaster Response Pipeline Project

## Table of Contents
1. [Installations](#installations)
2. [Project Motivation](#motivation)
3. [File Overview](#overview)
4. [Instructions](#instructions)
5. [Acknowledgements](#acknowledgements)

## <a id="installations"/> Installations
This project uses Python 3, all used libraries are available in the Anaconda distribution of Python. The used libraries are: 
- pandas
- numpy
- re
- sys
- json
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3

## <a id="motivation"/> Project Motivation
The goal of the project is to provide a machine learning pipeline that can categorize message data from disaster events, so that they can be forwarded to an appropriate disaster relief agency. 

The project has three components:
1. ETL Pipeline: a script for data cleaning that: 
- Loads the messages an categories datasets
- Merges the two datasets 
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline: a script containing the machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App: a web app where an emergency worker can input a new message and get classification results in several categories, as well as see visualizations of the data.

![Screenshot of the disaster response app](/img/classification_visualization.png?raw=true "Output of the classification")

![Screenshot of a visualization](/img/overview_visualization.png?raw=true "Distribution of message genres")

![Screenshot of a visualization](/img/top10_visualization.png?raw=true "Top 10 categories")

## <a id="overview"/> File Overview
The file structure is as listed below:
README.md: read me file
- \app
	- run.py: flask file to run the app
    -  \templates
	    - master.html: main page of the web application 
	    - go.html: result web page
- \data
	- disaster_categories.csv: categories dataset
	- disaster_messages.csv: messages dataset
	- DisasterResponse.db: disaster response database
	- process_data.py: ETL process
- \models
	- train_classifier.py: classification code
    - classifier.pkl: exported model
- \img: screenshots of the web app

## <a id="instructions"/> Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## <a id="acknowledgements"/> Acknowledgements
This project uses disaster data from [Figure Eight](https://www.figure-eight.com/).
