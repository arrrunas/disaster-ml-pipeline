# Disaster Response Pipeline Project

### Intro:
This app takes Tweets data, processes it, and classifies messages according to 32 disaster response categories using a multi output K-Neighbors classifier. Using flask, the data is visualised and new messages can be entered to be classified with the pre-trained model.

Note: model pickle file not included due to filesize constraints.

### Contents:
    - app
    | - template
    | |- master.html  # main page of web app
    | |- go.html  # classification result page of web app
    |- run.py  # Flask file that runs app

    - data
    |- disaster_categories.csv  # data to process 
    |- disaster_messages.csv  # data to process
    |- process_data.py # data processing script
    |- DisasterResponse.db   # database containing processed data

    - models
    |- train_classifier.py # ML script 

    - README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
