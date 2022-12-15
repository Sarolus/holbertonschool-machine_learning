# Data Preprocessing

## Sourcing

> The source of our data will come from forum on mental health, we will retrieve words given by anonymous people that will be used for our model. In order to obtain a maximum amount of data, we will scrap on different forum that correspond to what we need.

## Data format

> The data will first be scraped text, we will create bags of word in order to apply our natural language processing model and it will have features on different level to organize them and help the model prediction.

## Features

> In our case, we will organize words by symptoms that will be assigned to several psychology speciality that manage this type of symptoms, and on the next level there will be the practitioners registered with their proper speciality, in order to precognise the orientation that could be the best for the patient.

## Splitting

> Our data will be split into training, validation and testing sets. In order we will have 60 % of the data in the training dataset, 20 % in the validation dataset and 20 % in the testing dataset.

## Biases

>In order to avoid biases, we will first corretly defines the symptoms and the specialities, and it will be review by a professional that will be the only one able to validate the features. Other than that, we will try to define a maximum amount of confounding variables in order to manage them in our project. And to improve the performance of the NLP model, the data will be normalized using a Z-score normalization, we will adapt our model depending of the amount of data and fine tune our hyperparameters to avoid underfitting and overfitting.

## Training

> The features mentioned above will be included into the training, as we need them in order for our model to give a pertinent precognisation to the practitioner that will valide or invalidate the result, and will decide if it's necessary or not to redirect the patient to another practitioner that could be fit for there problem.

## Data type

> The type of data will be words and they will be handle on a categorical level. We will create a dataset composed of the words, and labels of each symptoms and each specialities. And we will trained the model to link the symtpoms and specialities first, using a valid dataset already labeled, and after the same model will be trained to link words to specific symptoms and create coherent precognisation based on the verbose given by the patient.

## Data storage

> The data we will retrieve will be stored in a database.
