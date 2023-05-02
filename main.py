import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from textblob import TextBlob


def main():
    file = 'test_reviews.csv'
    df = pd.read_csv(file, quotechar='"', skipinitialspace=True, encoding='utf8', engine='python')
    df['Sentiment'] = df['Response'].apply(lambda x: TextBlob(x).sentiment.polarity)
    plot_data_basic(df)
    overall_sentiment_message(overall_sentiment=df['Sentiment'].mean())


def plot_data_basic(df):
    sns.histplot(df['Sentiment'], bins=20, kde=True)
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.show()


def overall_sentiment_message(overall_sentiment):
    if (overall_sentiment > 0.2):
        adj = "positive"
    elif (overall_sentiment < -0.2):
        adj = "negative"
    else:
        adj = "neutral"
    print(f"Overall sentiment is {adj} with a score of {overall_sentiment}")


if __name__ == '__main__':
    main()

