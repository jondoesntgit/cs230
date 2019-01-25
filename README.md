# cs230

# Installation

Somewhere on your local machine (perhaps even inside of Dropbox), run the following in a terminal.

    git clone https://github.com/jondoesntgit/cs230

Set up your `.env` file to determine where on your computer, the large amounts of raw and processed data will live (ideally, not on your Dropbox folder, unless you have lots of synchronized storage space).
A default `.env_example` file is provided for your reference. In general, personal `.env` files should not be checked into version control for security purposes.

Then, you should be able to gather all of the data by typing this command in the project root:

    $ make

Currently, this command only downloads the dataset, but as we build out more features, it will automate more tasks.


## Organization

Driven Data has a nice [template](https://drivendata.github.io/cookiecutter-data-science/) for data science projects. That template is more or less what we use in this repository
