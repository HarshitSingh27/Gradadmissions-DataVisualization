from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
# import plotly
import plotly.express as px
from plotly.offline import plot
from wordcloud import WordCloud
import plotly.io as pio

import matplotlib.pyplot as plt
from io import BytesIO
import base64


app = Flask(__name__)


def show_bargraph():
    df_replies = pd.read_csv("word_count.csv" , index_col=0)
    fig = px.histogram(df_replies, x='word_count', nbins=20, title='Word Count Per Post Distribution')
    fig.update_layout(
        bargap=0.2,     
        xaxis_title='Word Count',
        yaxis_title='Frequency',
        hovermode='x',
        
    )
    fig.update_traces(
        hoverinfo='y+x', 
        marker=dict(line=dict(width=1, color='white'))
    )
    bar_graph = pio.to_html(fig, full_html=False)
    return bar_graph



def show_cloud_unigram():
    df_unigram = pd.read_csv("unigram.csv")
    unigram_word_frequencies = df_unigram.set_index('ngram')['count'].to_dict()
    # unigram_wordcloud = WordCloud(width=600, height=300, background_color='white').generate_from_frequencies(unigram_word_frequencies)
    unigram_wordcloud = WordCloud(background_color='white').generate_from_frequencies(unigram_word_frequencies)
    img = BytesIO()
    unigram_wordcloud.to_image().save(img, 'PNG')
    img.seek(0)
    unigram_img = base64.b64encode(img.getvalue()).decode('utf-8')
    return unigram_img

    
def show_cloud_bigram():
    df_unigram = pd.read_csv("bigram.csv")
    bigram_word_frequencies = df_unigram.set_index('ngram')['count'].to_dict()
    # bigram_wordcloud= WordCloud(width=600, height=300, background_color='white').generate_from_frequencies(bigram_word_frequencies)
    bigram_wordcloud= WordCloud(background_color='white').generate_from_frequencies(bigram_word_frequencies)
    img = BytesIO()
    bigram_wordcloud.to_image().save(img, 'PNG')
    img.seek(0)
    bigram_img = base64.b64encode(img.getvalue()).decode('utf-8')
    return bigram_img

def show_cloud_trigram():
    df_unigram = pd.read_csv("trigram.csv")
    trigram_word_frequencies = df_unigram.set_index('ngram')['count'].to_dict()
    # trigram_wordcloud= WordCloud(width=600, height=300, background_color='white').generate_from_frequencies(trigram_word_frequencies)
    trigram_wordcloud= WordCloud(background_color='white').generate_from_frequencies(trigram_word_frequencies)
    img = BytesIO()
    trigram_wordcloud.to_image().save(img, 'PNG')
    img.seek(0)
    trigram_img = base64.b64encode(img.getvalue()).decode('utf-8')
    return trigram_img


def show_graph():
    df_words = pd.read_csv("word_frequency.csv")
    df_words['category'] = df_words['category'].astype('category')
    df_words['jittered_x'] = np.random.uniform(-0.3, 0.3, size=len(df_words)) + df_words['category'].cat.codes
    fig = px.scatter(df_words,
                    x='jittered_x',
                    y='frequency',
                    size='frequency',
                    color='category',
                    hover_name='word',
                    title='Word Frequencies by Support Category')

    fig.update_xaxes(title_text='', showticklabels=False)
    fig.update_yaxes(title_text='Word Frequency')
    graph_html = plot(fig, output_type='div', include_plotlyjs=False)
    return graph_html

@app.route('/',methods = ["GET","POST"])
def index():
    bar_graph = show_bargraph()
    unigram_img=show_cloud_unigram()
    bigram_img = show_cloud_bigram()
    trigram_img = show_cloud_trigram()
    graph_html = show_graph()
    return render_template('index.html',bar_graph = bar_graph, unigram_img=unigram_img, bigram_img = bigram_img, trigram_img = trigram_img, graph_html=graph_html)

if __name__ == '__main__':
    app.run(debug=True)
