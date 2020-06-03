# Disaster Response Pipeline

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)


This Project is a part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The initial dataset contains pre-labelled tweet and messages from real-life disaster. The aim of the project is to build a Natural Language Processing tool that categorize messages.

### Execution:

- Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
	
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
		
    - To run ML pipeline that trains classifier and saves
	
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

- Run the following command in the app's directory to run the web app.
    `python run.py`

- Go to http://localhost:3001 or http://0.0.0.0:3001


### Project Description:

- Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
- Machine Learning Pipeline to train a model able to classify text message in categories
- Web App to show model results in real time.


### Web App Screenshots:

![](https://github.com/s0umitra/Disaster-Response-Pipeline/blob/master/screenshots/intro.png)

### License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](https://github.com/s0umitra/dash-101-wdb/blob/master/LICENSE)

This software is licenced under [MIT](https://github.com/s0umitra/dash-101-wdb/blob/master/LICENSE)
