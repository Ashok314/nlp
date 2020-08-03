def plotwords(words):    
    from wordcloud import WordCloud,ImageColorGenerator, STOPWORDS
    from PIL import Image
    import urllib
    import requests
    import numpy as np 
    import matplotlib.pyplot as plt


    fpath = "/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf"


    # (3) indicate words I don't want to include in the word cloud
    stopwords = set(STOPWORDS)
    stopwords.add("です")
    stopwords.add("する")
    stopwords.add("はい")
    stopwords.add("いい")
    stopwords.add("ない")
    stopwords.add("ない")
    stopwords.add("ある")
    stopwords.add("いる")


   # combining the image with the dataset
    Mask = np.array(Image.open("sentiment_analysis/visualize_data/cloud.png"))

    # We use the ImageColorGenerator library from Wordcloud 
    # Here we take the color of the image and impose it over our wordcloud
    image_colors = ImageColorGenerator(Mask)

    # Now we use the WordCloud function from the wordcloud library 
    wc = WordCloud(background_color='black',font_path=fpath, stopwords=stopwords, height=1500, width=4000,mask=Mask).generate(words)

    # Size of the image generated 
    plt.figure(figsize=(10,20))

    # Here we recolor the words from the dataset to the image's color
    # recolor just recolors the default colors to the image's blue color
    # interpolation is used to smooth the image generated 
    plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")

    plt.axis('off')
    plt.show()
   

from os import path
import MeCab

d = path.dirname("sentiment_analysis/visualize_data/") 
with open(path.join(d, 'all_positive_words.txt')) as f:
    positive = f.readlines()
# (2) tokenize the text
tagger = MeCab.Tagger('-r /etc/mecabrc -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
all_words_positive = ''
for sentence in positive:
    jparse = tagger.parseToNode(sentence)
    while jparse:
        tagger_split = jparse.feature.split(',')
        all_words_positive = all_words_positive + tagger_split[6] + ' '  # keep dictionary form
        jparse = jparse.next


with open(path.join(d, 'all_negative_words.txt')) as f:
    negative = f.readlines()

tagger = MeCab.Tagger('-r /etc/mecabrc -O chasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
all_words_negative = ''
for sentence in negative:
    jparse = tagger.parseToNode(sentence)
    while jparse:
        tagger_split = jparse.feature.split(',')
        all_words_negative = all_words_negative + tagger_split[6] + ' '  # keep dictionary form
        jparse = jparse.next


with open(path.join(d, 'all_profane_words.txt')) as f:
    profane = f.readlines()

tagger = MeCab.Tagger('-r /etc/mecabrc -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
all_words_profane = ''
for sentence in profane:
    jparse = tagger.parseToNode(sentence)
    while jparse:
        tagger_split = jparse.feature.split(',')
        all_words_profane = all_words_profane + tagger_split[6] + ' '  # keep dictionary form
        jparse = jparse.next

#Word cloud of data
#plotwords(all_words_positive)
#plotwords(all_words_negative)
plotwords(all_words_profane)    