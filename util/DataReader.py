import pandas as pd

from util.TextPreprocessor import text_to_wordlist


def load_user_stories(file_path):
    stories = pd.read_csv(file_path, encoding="utf-16")

    stories_df = text_to_wordlist(stories)

    print(stories_df.head())
