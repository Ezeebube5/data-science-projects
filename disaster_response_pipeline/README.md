<<<<<<< HEAD
# Disaster Response Pipeline Project
### Table of Contents

1. [Project Motivation](#motivation)
2. [Instructions](#instructions)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)


## Project Motivation<a name="motivation"></a>
I set out to create an ETL and machine learning pipeline to categorize messages sent during a disaster event. This project includes a web app to display visualizations of the data


## Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

Example message to classify: "Really Thirsty"

## File Descriptions <a name="files"></a>
* app/templates/~.html: HTML pages for the web app.
* app/run.py: Starts the Python server for the web app to input message to classify and visualize results.
* data/process_data.py: The ETL pipeline used to process data before building model.
* models/train_classifier.py: The Machine Learning pipeline used to accept database file path and model file path as input. It trains a classifier, tests, and exports the model to a Python pickle.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The data for this project was gotten from the Udacity project workspace and code samples from udacity practice notebooks 
=======
# udacity-data-scientist-project
Repo contains projects for the Udacity Data Scientist Nanodegree


## Projects include;
1. Write a data science blog post (In Progress)
2. Build Disaster Response Pipelines with Figure Eight (Coming Soon)
3. Design a Recommendation Engine with IBM (Coming Soon)
4. Data Science Capstone Project (Coming Soon)

>>>>>>> 98c70a893c6eef4063aec71fc35f3a00f2a3a854
