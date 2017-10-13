import pandas as pd

from datasets.text_utils import clean_text


def get_data():
    df = pd.read_csv("data/news-summary-keggle/news_summary.csv", encoding="latin1")
    df.drop_duplicates(subset=["ctext"], inplace=True)
    df.dropna(inplace=True)
    df.drop(['author', 'date', 'headlines', 'read_more'], 1, inplace=True)
    df.reset_index(drop=True, inplace=True)

    clean_summaries = []
    for summary in df.text:
        clean_summaries.append(clean_text(summary, remove_stopwords=False))

    clean_texts = []
    for text in df.ctext:
        clean_texts.append(clean_text(text))

    return clean_texts, clean_summaries
