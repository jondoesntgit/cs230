Some notes on setting up docker
===============================

Visit the [ECR segment of AWS](https://console.aws.amazon.com/ecr/repositories?region=us-east-1) 
Create a new repository. Keep track of the url.

Log into the the repository using. Note that you may need to install `pip install awscli` first, and ensure that it's in your shell's PATH variable (mine wasn't on a mac, and I had to enter it into ~/.zshrc)

    $ aws ecr get-login --region us-east-1

This will give you the comand that you're wanting to use to authenticate against ECR. However, you're going to have to strip off the `-e none https://` bit at the end. The rest should be fine.

Next, you want to see all of your docker images with `docker images`, and you'll want to tag the one you want to push

    $ docker tag cs230 YOUR-AMAZON-USER-ID.dkr.ecr.us-east-1.amazonaws.com/cs230
    $ docker push YOUR-AMAZON-USER-ID.dkr.ecr.us-east-1.amazonaws.com/cs230


    
