import selenium
#webdriver is our tool to interact with the webpage
from selenium import webdriver 
import requests #needed to load the page for BS4
from bs4 import BeautifulSoup
import pandas as pd #Using panda to create our dataframe
import time 
#import re

from cleantext import clean  
import sys
import os
import random
import spacy
from spacy.util import minibatch, compounding

total_positive=0

TEST_REVIEW =  ""






def train_model(
    training_data: list, test_data: list, iterations: int = 20
) -> None:
    # Build pipeline
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    # Train only textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Beginning training")
        print("Loss\tPrecision\tRecall\tF-score")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # A generator that yields infinite series of input numbers
        #print(batch_sizes)
        for i in range(iterations):
            print(f"Training iteration {i}")
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            #print(batches)
            for batch in batches:
                text, labels = zip(*batch)
                #print(text)
                #print(labels)
                nlp.update(text, labels, drop=0.2, sgd=optimizer, losses=loss)
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data,
                )
                eval_list_loss = loss['textcat']
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )

    # Save model
    with nlp.use_params(optimizer.averages):
        nlp.to_disk("model_artifacts")


def evaluate_model(tokenizer, textcat, test_data: list) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    print(reviews)
    true_positives = 0
    false_positives = 1e-8  # Can't be 0 because of presence in denominator
    true_negatives = 0
    false_negatives = 1e-8
    
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]["cats"]
        for predicted_label, score in review.cats.items():
            # Every cats dictionary includes both labels, you can get all
            # the info you need with just the pos label
            if predicted_label == "neg":
                continue
            if score >= 0.5 and true_label["pos"]:
                true_positives += 1
            elif score >= 0.5 and true_label["neg"]:
                false_positives += 1
            elif score < 0.5 and true_label["neg"]:
                true_negatives += 1
            elif score < 0.5 and true_label["pos"]:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}


def test_model(TEST_REVIEW):
    #  Load saved trained model
    loaded_model = spacy.load("model_artifacts")
    # Generate prediction
    parsed_text = loaded_model(TEST_REVIEW)
    
    # Determine prediction to return
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"Review text: {TEST_REVIEW}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )
    return {"Review text": TEST_REVIEW, "Predicted sentiment": prediction, "Score": score}


def load_training_data(
    data_directory: str = "aclImdb/train", split: float = 0.8, limit: int = 0
) -> tuple:
    # Load from files
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label,
                            }
                        }
                        reviews.append((text, spacy_label))
    random.shuffle(reviews)
    #print(reviews)

    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split)
    return reviews[:split], reviews[split:]



    


#path to the webdriver file
PATH = r"/home/qwerty/.ZAP/webdriver/linux/64/chromedriver"
#tell selenium to use Chrome and find the webdriver file in this location
driver = webdriver.Chrome(PATH)
#Set the url link and load the webpage
url = 'https://www.imdb.com/search/title/?groups=top_100'
driver.get(url)

#Set initial empty list for each element:
title = []
link = []
year = []

block = driver.find_elements_by_class_name('lister-item')
#Set up for loop to run through all 50 movies
for i in range(0,50):
    #Extracting title
    ftitle = block[i].find_element_by_class_name('lister-item-header').text
    print(ftitle)

    

     #The extracted title has extra elements, so we will have to do some cleaning
    
    #Remove the order in front of the title
    forder = block[i].find_element_by_class_name('lister-item-index').text
    

    fyear = ftitle[-6:]
    #Drop the order and year and only keep the movie's name
    ftitle = ftitle.replace(forder+' ', '')[:-7 ]

    #print(ftitle)

    #Then extract the link with cleaned title
    if ftitle == "Joker (I)":
    	ftitle = "Joker"
    flink = block[i].find_element_by_link_text(ftitle).get_attribute('href')
    #Add item to the respective lists
    title.append(ftitle)
    year.append(fyear)
    link.append(flink)



x = int(input("enter movie no here: "))

if x > 50:
	sys.exit('should be less Then 50')




url = link[x]
new_url = url.split('?')
user_agent = {'User-agent': 'Mozilla/5.0'}
response = requests.get(url, headers = user_agent)
soup = BeautifulSoup(response.text, 'html.parser')
review_link = new_url[0]+soup.find('a', text = 'User reviews').get('href')
print(review_link)
driver = webdriver.Chrome(PATH)
driver.get(review_link)
driver.implicitly_wait(1)

page = 1
while page<6:  
	try:
		load_more = driver.find_element_by_id('load-more-trigger')
		load_more.click()
		page+=1
		print(page) 
		time.sleep(5)
	except:
		break
review = driver.find_elements_by_class_name('review-container')		

#train, test = load_training_data(limit=2500)
#train_model(train, test)

movie = str(title[x])+'.txt'
			
with open(movie, 'w') as fp:
	for n in range(0,100):
		try:
			fcontent = review[n].find_element_by_class_name('content').get_attribute("textContent").strip()
			fcontent = fcontent.replace('Was this review helpful?  Sign in to vote.','')
			fcontent = fcontent.replace('Permalink','')
			train, test = load_training_data(limit=2500)
			fp.write("%s\n" % test_model(fcontent))
		except:
			continue

print("done")			



