# Cat-or-Dog-Image-Classification
I have trained two different CNN for a Binary Classification task. Their job is to classify and tell if it's a image of a Cat or a Dog. The dataset was developed as a partnership between Petfinder.com and Microsoft. The first CNN  has 4 Convolution layers with BatchNormalization and Max Pooling layers used on every Convolutional  layer, L2 kernel regularization used on 2nd(L2 lambda =0.001), 3rd(L2 lambda=0.001) and 4th(L2 lambda=0.00005) layer, HeUniform used on all the Convolutional  layer as a kernel_initializer after flattening dropout layer of 30% is used, a layer of 32 neurons and then the output layer of one neuron with sigmoid activation as this is a binary task. He weight initialization methods works well with Relu activation function. It's an upgrade to the Xavier initialization, it's very stable and creates smooth training progression, fast to converge to a desirable validation accuracy and multiple runs are similar. This was trained this on a GPU for 60 epochs this took about 5 hour 14 mins and achieved an accuracy of 96% on the test set.

The second CNN  has 5 Convolution layers with Max Pooling layers used on every Convolutional layer and BatchNormalization used on 2nd, 3rd, 4th and 5th Convolutional layer followed by flattening the layers and then a dense layer consisting of 512 neurons, a dropout layer of 40% another dense layer with 64 neurons then the output layer of one neuron. ReLU activation function is used on every layer except the output layer which has sigmoid activation function. L2 regularization and weight initialization has not been used in this layer. This was trained on a GPU for 100 epochs this took about 6 hours 55 minutes almost 7 hours and achieved an accuracy of 95.5% on the test set.

# Software requirements
To be able to run this you need to have installed
* [Tensorflow](https://www.tensorflow.org/tutorials)
* [Numpy](https://numpy.org/)
* [Scikit learn](https://scikit-learn.org/stable/)
* [Keras](https://keras.io/)
* [Visual Keras](https://github.com/paulgavrikov/visualkeras/)
* [Matplotlib](https://matplotlib.org/)
* [PIL](https://pypi.org/project/Pillow/)
* [Shutil](https://docs.python.org/3/library/shutil.html#:~:text=The%20shutil%20module%20offers%20a,level%20file%20copying%20functions%20(%20shutil. )

# Key Files
* [cat_or_dog.ipynb](https://github.com/Moddy2024/Cat-or-Dog-Image-Classification/blob/main/cat_or_dog.ipynb) - This file contains the first CNN it shows how the data has been checked and cleaned, the splitting of data in different sets, training of the model for 60 epochs, evaluation of the model. In the last cell you can see how the data looks like after passing through the different layers of the Convolutional layers.
* [trained_models](https://github.com/Moddy2024/Cat-or-Dog-Image-Classification/tree/main/trained_models) - This directory consists best of the trained models of the CNN. The trained models from the first CNN are model00000055.h5 and model00000060.h5 . The trained models from the second CNN are model00000060differentarchitecture.h5, model00000090.h5 and model00000100.h5 .
* [prediction.ipynb](https://github.com/Moddy2024/Cat-or-Dog-Image-Classification/blob/main/prediction.ipynb) -  Using any one of the trained model files from the above folder and images, this predicts if it's a picture of a cat or a dog.
* [cat_or_dog_evaluation.ipynb](https://github.com/Moddy2024/Cat-or-Dog-Image-Classification/blob/main/cat_or_dog_evaluation.ipynb) - Here I have used a dataset of 999 images of cats and dogs created labels in a dataframe for evaluation of all the models. Confusion matrix and classification report(precision, recall and f-1 score) is used for the evaluation of the models.
* [cat_vs_dog100epoch.ipynb](https://github.com/Moddy2024/Cat-or-Dog-Image-Classification/blob/main/cat_vs_dog100epoch.ipynb) - This file contains the second CNN  it shows how the data has been checked and cleaned, the splitting of data in different sets, training of the model for 100 epochs, evaluation of the model. In the last cell you can see how the data looks like after passing through the different layers of the Convolutional layers.
* [test_data](https://github.com/Moddy2024/Cat-or-Dog-Image-Classification/tree/main/test_data) - Here I have added some of the images that I have randomly saved from the internet which I have used to see the predictions of the models.

# Training and Validation Image Statistics
The images were collected by Microsoft from a mix of different angles and colors of cats and dogs, it even consists of pictures with humans holding the cats and dogs. The dataset consists a total of 25,002 files (%50 cats, %50 dogs). After cleaning the data 90% of the total dataset has been used for training which is a total of 22,498 images (%50 cats, %50 dogs). 2300 images (%50 cats, %50 dogs) are used for validation set. 200 images (%50 cats, %50 dogs) are used for test set. The dataset is normalized by dividing by 255 before training or inference. For data augmentation random horizontal flipping, rotation, width shifting, height shifting and shearing has been used.

The dataset used in this notebook [cat_or_dog_evaluation.ipynb](https://github.com/Moddy2024/Cat-or-Dog-Image-Classification/blob/main/cat_or_dog_evaluation.ipynb) contains 999 images (499 cats, 500 dogs) contains images from many different angles and colors and all of the dataset is used for evaluating the different models.

# Dataset
The dataset can be downloaded from [here](https://www.microsoft.com/en-us/download/details.aspx?id=54765) or by running the 3rd cell of this notebook [cat_or_dog.ipynb](https://github.com/Moddy2024/Cat-or-Dog-Image-Classification/blob/main/cat_or_dog.ipynb). The dataset used in this notebook [cat_or_dog_evaluation.ipynb](https://github.com/Moddy2024/Cat-or-Dog-Image-Classification/blob/main/cat_or_dog_evaluation.ipynb) can be downloaded from [here](https://www.kaggle.com/datasets/erkamk/cat-and-dog-images-dataset).

# Conclusion
After observing the above results we can see that the first CNN architecture is better than the second one not only did it takes less time to train but also has better accuracy and metrics in the classification report. Adding L2 regularization with a low lambda value reduces the training error, using less layers reduced the parameters which decreases the chance of overfitting and using weight initialization helped to achieve better accuracy in less epochs. Tuning hyperparamters to the right value helps Machine Learning models train faster and achieve better accuracy. 

