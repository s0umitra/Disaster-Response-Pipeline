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

**The Main Page**

![](https://github.com/s0umitra/Disaster-Response-Pipeline/blob/master/screenshots/main_page.png)

**Sample Message you can type to test the system**

![](https://github.com/s0umitra/Disaster-Response-Pipeline/blob/master/screenshots/sample_input.png)

**After clicking Classify Message, the categories which the message belongs to are shown highlighted in green**

![](https://github.com/s0umitra/Disaster-Response-Pipeline/blob/master/screenshots/sample_output.png)


### NoteBooks:

In the data and models folder you can find two jupyter notebook

- ETL Preparation Notebook: learn everything about the implemented ETL pipeline
- ML Pipeline Preparation Notebook: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn

You can use these Notebooks to re-train the model or tune it through a dedicated Grid Search section

### License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](https://github.com/s0umitra/Disaster-Response-Pipeline/blob/master/LICENSE)

This software is licenced under [MIT](https://github.com/s0umitra/Disaster-Response-Pipeline/blob/master/LICENSE)

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)
