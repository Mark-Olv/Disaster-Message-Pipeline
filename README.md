# Disaster-Message-Pipeline

This project is sponsored by Figure Eight and Udacity. It is centered around designing a disaster relief category classifier. A user would input some training data in the form of messages received during a disaster and the corresponding categories these messages fall under. Then the user could train a machine learning model to categorize future data into the provided categories. This model can be operated with the provided flask application.

### How to run the python scripts.

This project contains three python scripts. The first is process_data.py, in order to run this a user will need a message file, category file, and a name for the database that is created. The message file contains an id for each message, and the message itself. The category file contains an id for the message and the various categories that each message falls under. The process_data file also cleans the data. The database is where the post-processed data ends up so that it can be called by the model later on. To run the process_data.py file, the user can use the command line with the message file, category file, and database name as inputs.

The second python script used is train_classifier.py. This file called on the database that was previously made using sqlalchemy and pandas. This file makes a random_forest_classifer and trains the model with the data pulled from the database. The model is then dumped into a pickle file so that it can be utilized later.
To run the train_classifier.py file, the user can use the command line with the database name and model name and file path as inputs. 

The third and last python script used is run.py. This file utilizes the flask library in combination with the model we previously created. The end user can input messages they wish to classify inside the flask application. To run the run.py file, the user needs to edit where to find the database and where to find the pickled model. 

### Libraries used in this project.

The libaries that are used in the project are pandas, sklearn, pickle, sqlalchemy and nltk (natural language toolkit).

Pandas' main object is the dataframe. It is compatible with numpy, a popular data analysis tool. Pandas is a versatile library that allows you to read data from various file types, write to different file types, and even upload and pull data from databases. The dataframe are essentially tables that allow easy manipulated and allows for different types to be used, unlike in numpy which doesn't particular like different types in its arrays.

Sklearn (Scikit-Learn) is a machine learning library. It contains an incredible number of different objects and tools to be used. This library has machine learning models such as neural networks, linear models, random forests, and SVM to name a few.

Pickle is a library that allows users to save sklearn models. These models can then be unpacked and used in various situations such as further training or deployment into an application. 

Sqlalchemy is a database interaction libary. This library allows users to make connections to various different sql databases and pull or push data. This libary can be used alongside Pandas to upload data or pull data using SQL. 

Nltk is a natural language processing library. It enables users to parse through large amount of text data and simpifiy text.

 
The motivation of this analysis is that I am apart of a Udacity course that requires an analysis of dataset that can have business relevance.
Achknowledgements: Udacity and Figure Eight.
