"""
Generates a small synthetic news dataset (news.csv) with columns
[text, label] where label=1 -> FAKE, label=0 -> REAL.

IMPORTANT: This is only a stand-in so the pipeline runs end-to-end out of
the box. For your actual submission, replace data/news.csv with the real
Kaggle "Fake News Detection Dataset" or the UCI Fake News Dataset (see
README.md for download links/instructions) -- just make sure the CSV has
a 'text' column and a 'label' column (1=fake, 0=real).
"""

import random
import pandas as pd

random.seed(42)

REAL_TEMPLATES = [
    "The {org} reported today that {topic} has shown steady growth according to officials.",
    "Local authorities in {place} confirmed the new {topic} policy will take effect next month.",
    "According to a study published by {org}, {topic} rates have changed modestly this year.",
    "{org} released its quarterly report on {topic}, noting a stable outlook for the region.",
    "Officials at {org} held a press conference in {place} to discuss the {topic} initiative.",
    "Researchers at {org} say more data is needed before drawing conclusions about {topic}.",
    "The city council in {place} voted to fund additional resources for {topic} this year.",
    "A spokesperson for {org} clarified the recent changes to {topic} regulations.",
]

FAKE_TEMPLATES = [
    "You won't BELIEVE what {org} is hiding about {topic} -- shocking truth revealed!!!",
    "Doctors HATE this one weird trick that instantly cures {topic}, {org} refuses to comment.",
    "BREAKING: secret {org} documents PROVE {topic} conspiracy, share before it's deleted!!!",
    "Miracle cure for {topic} discovered in {place}, big pharma and {org} don't want you to know.",
    "Anonymous insider at {org} leaks SHOCKING truth about {topic} that will change everything.",
    "{place} residents in PANIC after {org} allegedly covers up {topic} scandal, click to see proof.",
    "Scientists baffled: {topic} linked to {org} in bizarre secret experiment, media silent!!!",
    "URGENT WARNING: {org} insiders admit {topic} crisis is far worse than reported, share NOW.",
]

ORGS = ["the Ministry of Health", "the World Bank", "NASA", "the CDC", "the local university",
        "the national weather service", "the Department of Education", "a leading tech company",
        "the World Health Organization", "the central bank"]
TOPICS = ["inflation", "vaccine safety", "climate change", "unemployment", "a new vaccine",
          "election results", "5G towers", "a asteroid impact", "crop yields", "public transit",
          "cybersecurity", "housing prices", "a rare disease outbreak", "renewable energy"]
PLACES = ["Springfield", "Mumbai", "Austin", "Berlin", "Nairobi", "Manila", "Toronto",
          "Cairo", "Jakarta", "Lagos"]


def make_sentence(templates):
    t = random.choice(templates)
    return t.format(org=random.choice(ORGS), topic=random.choice(TOPICS), place=random.choice(PLACES))


def make_article(templates, n_sentences=3):
    return " ".join(make_sentence(templates) for _ in range(n_sentences))


def generate(n_per_class: int = 300):
    rows = []
    for _ in range(n_per_class):
        rows.append({"text": make_article(REAL_TEMPLATES), "label": 0})
        rows.append({"text": make_article(FAKE_TEMPLATES), "label": 1})
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate(300)
    df.to_csv("data/news.csv", index=False)
    print(f"Wrote {len(df)} rows to data/news.csv")
    print(df.head())
