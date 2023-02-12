import re
import numpy as np
import pandas as pd
from pprint import pprint
from collections import Counter
import os
from wordcloud import WordCloud

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import math

from nltk.util import trigrams
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.palettes import d3
from bokeh.transform import factor_cmap
from bokeh.layouts import layout
from bokeh.embed import components
from bokeh.palettes import Spectral3

static_dir = os.path.join(os.getcwd(), 'static')
images_dir = os.path.join(static_dir, 'images')
instagram_dir = os.path.join(os.getcwd(), 'instagram')

output_notebook()
class TopicModelling():
  def __init__(self, target):
    self.target = target
    self.target_dir = os.path.join(instagram_dir, target)
    self.target_images_dir = os.path.join(images_dir, target)
    try:
      self.arg = pd.read_csv(os.path.join(self.target_dir, f"{target}_final.csv"))
    except:
      self.arg = pd.read_csv(os.path.join(self.target_dir, f"{target}.csv"))

  def input_check(self):
    if len(list(set("".join(self.arg.final.values).split(' ')))) < 50:
      return False
    return True

  def sent_to_words(self):
    data = self.arg.final.values.tolist()
    for sentence in data:
      yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

  def make_trigrams(self):
    data_words = list(self.sent_to_words())

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    def make_trigram(texts):
      return [trigram_mod[bigram_mod[doc]] for doc in texts]

    data_words_trigrams = make_trigram(data_words)

    return data_words_trigrams

  def asign(self):
    self.final = self.arg.reset_index(drop=True)
    self.data = self.arg.final.values.tolist()
    
    data_lemmatized = self.make_trigrams()
    # Create Dictionary
    self.id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    self.texts = data_lemmatized

    # Term Document Frequency
    self.corpus = [self.id2word.doc2bow(text) for text in self.texts]

    return self.id2word, self.texts, self.corpus

  def compute_coherence_values(self, limit=10, start=2, step=1):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    id2word, texts, corpus = self.asign()

    dictionary = id2word
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
      model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                              id2word=id2word,
                                              num_topics=num_topics,
                                              random_state=100,
                                              update_every=1,
                                              chunksize=100,
                                              passes=10,
                                              alpha='auto',
                                              per_word_topics=True)
      model_list.append(model)
      coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary)
      coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

  def get_coherence_graph(self, limit=10, start=2, step=1):
    self.model_list, self.coherence_values = self.compute_coherence_values()
    
    x= range(start, limit, step)
    plt.plot(x, self.coherence_values)
    plt.xlabel('Num Topics')
    plt.ylabel('Coherence score')
    plt.legend(('coherence_values'), loc='best')
    plt.savefig(os.path.join(self.target_images_dir, f"coherence_{self.target}.png"), dpi=300, bbox_inches='tight')

  def pick_optimal_model(self, topic):
    self.model_list, self.coherence_values = self.compute_coherence_values()
    topic = topic-2
    self.optimal_model = self.model_list[topic]
    self.model_topics = self.optimal_model.show_topics(formatted=False)
    return self.optimal_model, self.model_topics

  def format_topics_sentences(self, topic):
    ldamodel, _ = self.pick_optimal_model(topic)
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[self.corpus]):
      row = row_list[0] if ldamodel.per_word_topics else row_list
      row = sorted(row, key=lambda x: (x[1]), reverse=True)
      # Get the Dominant topic, Perc Contribution and Keywords for each document
      for j, (topic_num, prop_topic) in enumerate(row):
        if j == 0: # => dominant topic
          wp = ldamodel.show_topic(topic_num)
          topic_keywords = ', '.join([word for word, prop in wp])
          sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,10), topic_keywords]), ignore_index=True)
        else:
          break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(self.texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df

  def get_df_dominant_topic(self, topic):
    df_topic_sents_keywords = self.format_topics_sentences(topic)
    # Formatting
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic['Dominant_Topic'] = df_dominant_topic['Dominant_Topic'].astype(str)
    df_dominant_topic['Dominant_Topic'] = df_dominant_topic['Dominant_Topic'].str.replace('.0', '')
    df_dominant_topic[['Date', 'likes', 'comments', 'url']] = self.final[['Date','likes','comments','url']]
    df_dominant_topic['engagement'] = df_dominant_topic['comments'] + df_dominant_topic['likes']
    df_dominant_topic['Id'] = df_dominant_topic['Date']
    df_dominant_topic['Dominant_Topic'] = df_dominant_topic['Dominant_Topic'].astype('string')
    df_dominant_topic['Date'] = df_dominant_topic['Date'].apply(pd.to_datetime)
    df_dominant_topic['Date'] = df_dominant_topic['Date'] + pd.DateOffset(hours=7)
    df_dominant_topic['Hour'] = df_dominant_topic['Date'].dt.hour
    df_dominant_topic['Day Name'] = df_dominant_topic['Date'].dt.strftime('%A')
    df_dominant_topic['Day Number'] = df_dominant_topic['Date'].dt.weekday + 1
    df_dominant_topic = df_dominant_topic.reset_index()
    return df_dominant_topic

  def engagement_check(self, df):
    df = df.groupby(['Dominant_Topic', 'Keywords'])['engagement'].agg(Engagement='mean', Count='count')
    df = df.reset_index()
    return df

  def viz(self, topic, limit=10):
    from bokeh.palettes import d3
    df = self.get_df_dominant_topic(topic)
    self.df = df
    df[['likes','comments']] = df[['likes','comments']].astype('int')
    cat = self.engagement_check(df)
    # Preparation
    d3 = d3
    topic_count = df['Dominant_Topic'].nunique()
    if topic_count > 2:
      color = d3['Category20'][topic_count]
    else:  
      color = ('#1f77b4', '#aec7e8')

    # Engagement Per Topic
    cat['Dominant_Topic'] = cat['Dominant_Topic'].map(lambda x: 'Topic No {}'.format(x))
    keywords = cat['Dominant_Topic']
    engagement = cat.Engagement
    print(keywords)
    print(type(keywords))

    source = cat

    ept = figure(x_range=keywords,width=750, height=600, toolbar_location=None, title="Engagement Per Topic", sizing_mode='scale_width')
    ept.vbar(x='Dominant_Topic', top='Engagement', width=0.9, source=source, legend_field="Dominant_Topic",
          line_color='white', fill_color=factor_cmap('Keywords', palette=color,
                            factors=sorted(cat.Keywords.unique())))

    ept.xgrid.grid_line_color = None
    ept.y_range.start = 0
    ept.legend.visible=False 

    # Count Per Topic
    #cat['Dominant_Topic'] = 
    #keywords = cat['Dominant_Topic']
    engagement = cat.Count

    #source = cat

    cpt = figure(x_range=keywords,width=750, height=600, toolbar_location=None, title="Count Per Topic", sizing_mode='scale_width')
    cpt.vbar(x='Dominant_Topic', top='Count', width=0.9, source=source, legend_field="Dominant_Topic",
          line_color='white', fill_color=factor_cmap('Keywords', palette=color,
                            factors=sorted(cat.Keywords.unique())))

    cpt.xgrid.grid_line_color = None
    cpt.y_range.start = 0
    cpt.legend.visible=False 
    # Scatter Timeline
    p = figure(plot_width=1500, plot_height=600, title = "Persebaran Konten",x_axis_type="datetime", sizing_mode='scale_width')
    p.scatter('Date','engagement',source=df,fill_alpha=1, fill_color=factor_cmap('Keywords', palette=color,
                            factors=sorted(cat.Keywords.unique())),size=10,legend='Dominant_Topic')
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'engagement'
    p.legend.location = "top_left"
    p.legend.visible=False


    # Content Count Per Day
    group_cpd = df.groupby(["Keywords","Day Name","Day Number"])['Keywords'].agg(Count='count')
    group_cpd = group_cpd.reset_index()
    group_cpd = group_cpd.pivot(index=['Day Number','Day Name'], columns='Keywords', values='Count').reset_index()
    group_cpd = group_cpd.rename_axis(None, axis=1)
    group_cpd = group_cpd.fillna(0)

    df['Title'] = "Topic No " + df['Dominant_Topic'] + " " + df['Keywords']
    Topics = sorted((df.Title.unique()))
    Topic = sorted((df.Keywords.unique()))
    Day = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    cpd = figure(x_range=Day, width=750, height=600, title="Count per Day",
              toolbar_location=None, tools="", sizing_mode='scale_width')

    cpd.vbar_stack(Topic, x='Day Name', width=0.9, color=color, source=group_cpd,
                legend_label=Topics)

    cpd.xaxis.axis_label = 'Day'
    cpd.yaxis.axis_label = 'Count'
    cpd.x_range.range_padding = 0.1
    cpd.legend.visible=False

    # Content Count Per Hour
    group_cph = df.groupby(["Keywords","Hour"])['Keywords'].agg(Count='count')
    group_cph = group_cph.reset_index()
    group_cph = group_cph.pivot(index='Hour', columns='Keywords', values='Count').reset_index()
    group_cph = group_cph.rename_axis(None, axis=1)
    group_cph = group_cph.fillna(0)

    Hour = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']

    cph = figure(x_range=Hour, width=750, height=600, title="Count per hour",
              toolbar_location=None, tools="", sizing_mode='scale_width')

    cph.vbar_stack(Topic, x='Hour', width=0.9, color=color, source=group_cph,
                legend_label=Topics)

    cph.xaxis.axis_label = 'Hour'
    cph.yaxis.axis_label = 'Count'
    cph.x_range.range_padding = 0.1
    cph.legend.visible=False

    layouts = layout([
        [ept, cpt],
        [p],
        [cpd, cph]
    ])
    #show(layouts)

    plots = {'ept': ept, 'cpt': cpt, 'p':p, 'cpd':cpd, 'cph': cph}
    scripts, div = components(plots)

    # topics and cloudword matplotlib
    j = math.ceil(cat.Dominant_Topic.count())
    width = 20
    height = j*10
    fig, ax = plt.subplots(j,2,figsize = (width,height))
    i=0
    for t in cat.Keywords:
      long_string = ','.join(list(df['Text'].astype('string')[df['Keywords'] == t].values))

      # Create a WordCloud object
      wordcloud = WordCloud(width=1600, height=800, background_color="white", max_words=200, contour_width=3, contour_color='steelblue')

      # Generate a word cloud
      wordcloud = wordcloud.generate(long_string)

      # Visualize the word cloud
      ax[i,0].imshow(wordcloud, interpolation="bilinear")
      ax[i,0].title.set_text("Word Cloud for Topic Topic #" +str(i))

      cnt = Counter()
      for text_all in df['Text'][df['Keywords'] == t].values:
          for word in str(text_all).split():
              cnt[word] += 1
      nilai = cnt.most_common(10)
      word = pd.DataFrame(nilai, columns = ['Words', 'Count'])
      word.plot.barh(ax=ax[i,1],x="Words", y="Count")
      ax[i,0].title.set_text("Frequent Word for Topic #" +str(i))


      i+=1

    fig.savefig(os.path.join(self.target_images_dir, f"topic_cloud_{self.target}.png"), dpi=300, bbox_inches='tight')
    return scripts, div

