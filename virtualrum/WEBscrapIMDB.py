import os
import random
import time
import requests
import spacy
from spacy.util import minibatch, compounding
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from cleantext import clean
import sys


# === SENTIMENT MODEL ==


def train_model(training_data: list, test_data: list, iterations: int = 20) -> None:
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe("textcat", config={"architecture": "simple_cnn"})
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    training_excluded_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]

    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        print("Beginning training")
        print("Loss\tPrecision\tRecall\tF-score")

        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(iterations):
            print(f"Training iteration {i}")
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(text, labels, drop=0.2, sgd=optimizer, losses=loss)

            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer, textcat=textcat, test_data=test_data
                )
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )

    with nlp.use_params(optimizer.averages):
        nlp.to_disk("model_artifacts")


def evaluate_model(tokenizer, textcat, test_data: list) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8
    true_negatives = 0
    false_negatives = 1e-8

    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]["cats"]
        for predicted_label, score in review.cats.items():
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
    f_score = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}


def test_model(TEST_REVIEW):
    loaded_model = spacy.load("model_artifacts")
    parsed_text = loaded_model(TEST_REVIEW)
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(f"Predicted: {prediction} (Score: {score:.3f})")
    return {"Predicted sentiment": prediction, "Score": score}


def load_training_data(data_directory: str = "aclImdb/train", split: float = 0.8, limit: int = 0) -> tuple:
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}", encoding="utf-8") as f:
                    text = f.read().replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {"cats": {"pos": label == "pos", "neg": label == "neg"}}
                        reviews.append((text, spacy_label))
    random.shuffle(reviews)
    if limit:
        reviews = reviews[:limit]
    split_index = int(len(reviews) * split)
    return reviews[:split_index], reviews[split_index:]



# === SCRAPE IMDb TOP 100 (WORKING SELECTORS) =====


chrome_options = Options()
chrome_options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

url = "https://www.imdb.com/search/title/?groups=top_100&sort=user_rating,desc"
driver.get(url)
time.sleep(3)

blocks = driver.find_elements(By.CSS_SELECTOR, ".ipc-metadata-list-summary-item")
print(f"Found {len(blocks)} movies")

titles, years, links = [], [], []

for i, block in enumerate(blocks[:100]):
    try:
        ftitle = block.find_element(By.CSS_SELECTOR, "h3.ipc-title__text").text
        ftitle = ftitle.split(". ", 1)[1] if ". " in ftitle else ftitle
        fyear = block.find_element(By.CSS_SELECTOR, "span.dli-title-metadata-item").text
        flink = block.find_element(By.CSS_SELECTOR, "a.ipc-title-link-wrapper").get_attribute("href")

        titles.append(ftitle)
        years.append(fyear)
        links.append(flink)
        print(f"{i+1}. {ftitle} ({fyear})")
    except Exception as e:
        print(f"Skipping movie {i}: {e}")

x = int(input("\nEnter movie number (1–100): ")) - 1
if not (0 <= x < len(links)):
    sys.exit("Invalid input! Must be between 1 and 100.")

movie_title = titles[x]
movie_url = links[x]

# ================= FETCH REVIEWS =================
response = requests.get(movie_url, headers={'User-Agent': 'Mozilla/5.0'})
soup = BeautifulSoup(response.text, "html.parser")
#review_anchor = soup.find("a", string="User reviews")
review_anchor = soup.find("a", string=lambda s: s and "user reviews" in s.lower())

if not review_anchor:
    sys.exit("No reviews found.")

review_link = "https://www.imdb.com" + review_anchor["href"]
print(f"📝 Review page: {review_link}")

driver.get(review_link)
time.sleep(5)

# Try switching into iframe (IMDb sometimes loads reviews inside one)
try:
    iframe = driver.find_element(By.CSS_SELECTOR, "iframe#sis_frame")
    driver.switch_to.frame(iframe)
    print("Switched into iframe containing reviews.")
    time.sleep(2)
except:
    print("No iframe detected — continuing normally.")

# Try to click "load more" button if available
for _ in range(5):
    try:
        load_more = driver.find_element(By.CSS_SELECTOR, "#load-more-trigger, .ipl-load-more__button")
        driver.execute_script("arguments[0].click();", load_more)
        time.sleep(3)
        print("Loaded more reviews...")
    except:
        break

# Grab reviews (handle multiple IMDb layouts)
reviews = driver.find_elements(By.CSS_SELECTOR, ".review-container, div.lister-item-content, .ipc-html-content")
print(f"Found {len(reviews)} reviews on page.")

filename = f"{movie_title}.txt"
print(f"\nSaving reviews for '{movie_title}'...")

# Train model if not already trained
if not os.path.exists("model_artifacts"):
    print("Training SpaCy sentiment model...")
    train, test = load_training_data(limit=2000)
    train_model(train, test)

# Analyze and write reviews
with open(filename, "w", encoding="utf-8") as fp:
    for r in reviews[:100]:
        try:
            try:
                content = r.find_element(By.CSS_SELECTOR, ".text.show-more__control").text
            except:
                content = r.text  # fallback for alternate layouts

            cleaned = clean(content)
            result = test_model(cleaned)
            fp.write(f"{cleaned}\nSentiment: {result['Predicted sentiment']} (Score: {result['Score']:.3f})\n\n")
        except Exception as e:
            continue

driver.quit()
print(f" Done! Saved {len(reviews[:100])} reviews and sentiment predictions to '{filename}'.")
