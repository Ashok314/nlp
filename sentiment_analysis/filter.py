def profanity_filter(input1):   
    import fasttext
    import os
    import preprocess

    model = fasttext.load_model("/home/ashok/Desktop/nlp/model.bin")

    #print(model.predict("your word "))
    # filter



    #print("input is  being processed ... ")

    #to check only meanaing full words with profane words, li bit of cleaning 
    input2=preprocess.get_sentences(input1) # ??

    # print("input after processing",input)

    profane_word=[]
    profane_flag=False
    file=open("/home/ashok/Desktop/nlp/mecab_tagger/profane.csv","r").read()

   
    prediction= str(model.predict(input2)[0][0])



    if (prediction == "__label__profane"):
        print("Predicted as Pofane,")

        #tokanize input 
        import MeCab
        tagger = MeCab.Tagger('-r /etc/mecabrc -Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        text = tagger.parse(input2)
    
        text=text.strip('\n').split(" ")

        for item in text:
            if item and item in file:
                profane_flag=True
                profane_word.append(item)     
            

        if (profane_flag):
            print("and It is profane.")

            #print("word is ->",profane_word)

            print("Before profane words : \t ",input1 )

            
            for i in profane_word:
                input1=input1.replace(i,"****")
                
            print("After censoring : \t",input1)

        else:
            print("But it is not actually profane!") 
    else:
        polarity=model.predict(input2)[0][0][9:17]
        print("clean word! with polarity",polarity)        


print(" Please Provide input  word / sentence")
input1=input()
profanity_filter(input1)


#issue, proper encoding of the text input !!