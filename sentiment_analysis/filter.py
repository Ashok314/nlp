import fasttext
import os
model = fasttext.load_model("model.bin")

#print(model.predict("your word "))


# filter

input ="ナオン"
profane_word=""
profane_flag=False
file=open("mecab_tagger/profane.csv","r").read()

if (model.predict(input)[0][0] == "__label__profane" and model.predict(input)[1][0]>0.4):
    print("Predicted as Pofane,")

    #tokanize input 
    import MeCab
    tagger = MeCab.Tagger('-r /etc/mecabrc -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    node = tagger.parseToNode(input)
    
    while node:
        if (node.surface in file):
            profane_flag=True
            profane_word+=node.surface     
        node = node.next    

    if (profane_flag):
        print("and It is profane.")
        print("word is ->",profane_word)
    else:
        print("But it is not actually profane!") 
else:
    print("clean word! with polarity",model.predict(input)[0][0][9:17])        