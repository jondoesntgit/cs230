# cs230

# 

This is school project for CS230 at Stanford University (Deep Learning). This project has to do with classifying audio clips according to their Mel Spectrograms. Input audio is downloaded, and optionally trimmed to 10 seconds, and saved as a .wav file. The audio is resampled to 44100 Hz, and then converted into a spectrogram. The spectrogram is then fed into a pre-trained model produced by Google called VGGIsh. The output of the VGGish neural net is then passed through a convolutional layer, and a few fully-connected layers. The final layer is a few softmax nodes each corresponding to an audio category (e.g. "Male Speech", "Siren"). 

There's a fair amount of modularized code inside of this repository. There is a Tornado webapp for showing a live demo, where audio can be uploaded and classified.

There are some remnants from some earlier work that we did to duplicate the results of Piczak with a classifier on the ESC50 dataset.

There are some files stored under /src/utils for scraping data from youtube, and storing it into a PostgreSQL, S3 database to be accessed later.

If you are interested in using or re-using any of this code, it's probably better for you to email jamwheel@stanford.edu, and have me walk through how to use this code, as it would take some time to fully document all of this code.

# Installation 

(At least, how to install some of the stuff when we first started)

Somewhere on your local machine (perhaps even inside of Dropbox), run the following in a terminal.

    git clone https://github.com/jondoesntgit/cs230

Next, create an environment so we can all work using the same version of python.

    # Add these channels so conda knows where to look for packages
    conda config --append channels conda-forge 
    conda config --append channels pytorch
    
    # Create the environment
    conda create --name cs230 python=3.6 --yes --file requirements.txt
    
    # Apply the changes
    source activate cs230

If your environment has changed (perhaps due to the requirements.txt file being updated), you can upgrade your environment through

    conda install --yes --file requirements.txt

To dump your environment into the `requirements.txt`, you can run

    conda list -e > requirements.txt

If for some reason, conda doesn't install the requirements, you can install the dependencies using pip, using sudo as required. Ensure that you're using python==3.6.* however.

    pip install -r requirements.txt

Set up your `.env` file to determine where on your computer, the large amounts of raw and processed data will live (ideally, not on your Dropbox folder, unless you have lots of synchronized storage space).
A default `.env_example` file is provided for your reference. In general, personal `.env` files should not be checked into version control for security purposes.

Then, you should be able to gather all of the data and run a feature extractor on it by typing this command in the project root:

    $ make

Currently, this command only downloads the dataset and runs a feature extractor from an existing Github repository.
As we build this repo more, make will automate more tasks.

## Organization

Driven Data has a nice [template](https://drivendata.github.io/cookiecutter-data-science/) for data science projects. That template is more or less what we use in this repository

## VGGish features

`make vggish_params` downloads the parameters needed to do VGGish feature extraction
