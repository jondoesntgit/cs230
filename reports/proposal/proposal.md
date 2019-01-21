# Deep Hearing - Identifying Audio Underwater 

Authors:

- Behrad Afshar (bhafshar)
- Jeremy Witmer (jwitmer)
- Jonathan Wheeler (jamwheel)

--- 
Hydrophones are underwater sensors that measure acoustic pressure in audio frequencies, ranging from a few Hertz to tens of kHz. 
These sensors are often deployed on marine/submarine vessels and in harbors to detect nearby objects or activity.
In this project, we seek to train a neural network to classify audio clips according to the source that produced them.
In particular, we are interested in being able to demonstrate that fiber-optic hydrophones developed by the Digonnet Group can reliably classify audio clips. \cite{}
The ability to classify audio signals historically has been of interest both in maritime navigation and defense applications as well as in the study of marine biology. \cite{} \cite{} \cite{} 
In these applications, tens or hundreds of sensors may be deployed in a large-area array, and classification by a human agent may be unreasonable.

There exist several audio datasets that have a short sample of audio and a corresponding label that indicates what type of source produced the audio (e.g. boats, speech, wind). \cite{} \cite{} \cite{}
In the first phase of our study, we will divide our labeled dataset into training and testing sets. We will use a [TECHNIQUE] to train the neural net on sound cilps from the dataset, and evaluate the model on the testing set. In the second phase, we will apply an finite-impulse reponse (FIR) filter in software to the datasets to simulate the effect of the audio propagating underwater. In this phase, we will evaluate how robust the neural net from the first phase is to our simulated underwater channel's frequency response, and will retrain the model as necessary. Finally, in the third phase, we will play selected audio clips through a speaker into a submerged hydrophone, and allow the hydrophone to attempt to classify the audio clip's source with the neural network to validate the fit. To evaluate our performance, we will use a confusion matrix. We will also assess the number of training samples required to achieve a certain performance in order to give an estimate of how many examples would be needed in a real-world defense or science application.
        
Algorithms that have traditionally been used to classify audio and other time series are recurrent neural networks (RNNs) and Convolutional Neural Networks, where the data is split into spectrograms and then passed through a neural networks (NN) [^cs230-winter-project]. Other algorithms involve Mel-frequency cepstral coefficients (MFCCs), and Convolutional Deep Belief Networks (CDBNs)[^cs230-winter-project].

One challenge we expect is that hydrophones often detect ambient noise in addition to signals of interest.
For example, a hydrophone affixed to the rear of a large vessel may detect the superposition of the vessel it is mounted on in addition to whales in addition to another submarine vessel nearby. 
We have selected datasets that associate more than one label to each audio clip in order to anticipate this end-application performance specification.

[^cs230-winter-project]: [Matthew Meza, Job Naliaya. *Music Genre Classification Using Deep Learning*. March 2018. CS 230](http://cs230.stanford.edu/files_winter_2018/projects/6936608.pdf)
