# Nanodegree Capstone - StarBucks Challenge

My results, observations and takeways are contained in a medium post [here](https://medium.com/@ebube_eze/surprising-takeaways-from-the-stack-overflow-survey-results-485fdfc6fc3d?sk=fc876c5cd48896928e6a3f21e48bca36).

### Table of Contents

1. [Project Motivation](#motivation)
2. [Instructions](#instructions)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)


## Project Motivation<a name="motivation"></a>
I set out to analyze to combine transaction, demographic and offer data to determine which demographic groups respond best to different StarBucks offers. I built a model to predict offer success or failure based on customer demographics and offer details 

Results, observations and conclusions are contained in the Notebook - **Starbucks_Capstone_notebook.ipynb**

## Instructions <a name="instructions"></a>
1. Download the Notebook and run with Jupyter Notebooks. Anaconda (with python 3) is a good route to get up and running with the Jupyter Notebooks.




## File Descriptions <a name="files"></a>
* Starbucks_Capstone_notebook.ipynb: Notebook containing the code for this project
* Pictures: Contains Pictures of Data visualizations
* Data: Folder containing Porfolio, Profile and Transcript files which provided as raw data for the project.

### Data Sets

The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The data for this project was gotten from the Udacity project workspace and code samples from udacity practice notebooks 

