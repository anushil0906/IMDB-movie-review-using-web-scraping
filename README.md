**CODE EXECUTION DEMO**

https://github.com/user-attachments/assets/aacc0fa0-041d-4d8b-933a-c9247d692b4e


# IMDB-movie-review-using-web-scraping
use dataset https://ai.stanford.edu/~amaas/data/sentiment/
to train the model

**for running in virtual env**

step 1:- created a virtual env in virtualrum folder through this command in CMD (py -3.9 -m venv venv)

step2:- activate the virtual env by running this command in CMD (venv\Scripts\activate)

step3:- (venv) C:\Users\sharm\Downloads\virtualrum> pip install -r req.txt


**EXPLAINATION**

I structured the code using classes such as Scraper, Sentimental Models, Preprocessor, and Review Dataset, each encapsulating a distinct responsibility. Libraries such as BeautifulSoup and Selenium implement a robust web scraper that collects movie reviews, followed by a preprocessing and sentiment classification pipeline.

**Processing**

-> The code first scrapes the movie reviews from the IMDB top 100 movies list using Selenium and BeautifulSoup, and we can select any movie and extract its 500 reviews in a text format for further analysis. 

-> Then a sentiment analysis model has been implemented by me to decide whether a review is positive or negative, which uses the en_core_web_sm model from a Python library known as spaCy for natural language for  processing the extracted reviews for a text categorizer, which is a convolutional neural network that classifies text as positive or negative.

-> The text categorizer got trained for at least 20 iterations with the IMDB dataset (acllmdb) stored locally, which is already labeled with positive and negative reviews.

-> It uses minibatches and compounding to train effectively and track loss, precision, recall, and f-score in each round. Learns different word patterns that indicate negative and positive sentiments.

-> Then it saves the model into a folder called model-artifacts. 

->For testing the model on extracted IMDB reviews, it uses spaCy.load("model-artifacts") and prints the predicted positive or negative sentiment and confidence score.
