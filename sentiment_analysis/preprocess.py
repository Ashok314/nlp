import MeCab
import re
import dask.dataframe as dd
import unicodedata
import os


def unicode_normalize(cls, s):
        pt = re.compile('([{}]+)'.format(cls))
        
        def norm(c):
            return unicodedata.normalize('NFKC', c) if pt.match(c) else c #unicodedata.normalizeのNFKC（Normalization Form Compatibility Composition）
        s = ''.join(norm(x) for x in re.split(pt, s))                     # 半角カタカナ、全角英数、ローマ数字・丸数字、異体字などなどを正規化。
        s = re.sub('－', '-', s)
        return s

def normalize(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))
    
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s

def remove_spaces(s):
            s = re.sub('[  ]+', ' ', s)
            blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                              '\u3040-\u309F',  # HIRAGANA
                              '\u30A0-\u30FF',  # KATAKANA
                              '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                              '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                              ))
            basic_latin = '\u0000-\u007F'

            def remove_space_between(cls1, cls2, s):
                p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
                while p.search(s):
                    s = p.sub(r'\1\2', s)
                return s
            s = remove_space_between(blocks, blocks, s)
            s = remove_space_between(blocks, basic_latin, s)
            s = remove_space_between(basic_latin, blocks, s)
            return s

def remove_newline(s):
	s=s.replace("\n", " ")
	s=s.rstrip()
	return s
    
def remove_emoji(s):  #only specific type!!  #can be extended for all 
    
    m= re.search('^#x[a-zA-Z_0-9]*;$', s) 
    n=re.search('^&#x[a-zA-Z_0-9]*;*', s)  #  &#x1F614;
   

    if m :
        s=s.replace(m.string, "\n")
    if n :
        s=s.replace(n.string,"\n")

    return s

def get_sentences(text):
    tagger = MeCab.Tagger("/home/ashok/Desktop/nlp/mecab_tagger/ja_profane_words.csv")
    tagger.parse("")
    node = tagger.parseToNode(normalize(text))
    words = [] 

    match=["名詞","動詞","形容詞","感動詞"]
    donotmatch=["代名詞","係助詞","連体化","接尾"]

    while node:
        #match first features from list of features     #ignore bad character and katakana(with no meaning ) too ! ignored special char and everything with no entry in the dictionary !?, 
        if node.feature.split(",")[0] in match:          #description of matched token eg. 私  名詞,代名詞,一般,*,*,*,私,ワタシ,ワタシ
            if node.feature.split(",")[1] not in donotmatch:
                if node.surface not in words and not re.match('^[ぁ-ん]$',node.surface):  #ignore repeat words and single character 
                    words.append(node.surface)                    #matched text eg 私 
        node = node.next
    return ' '.join(words)

def noise_remove(s):
    s = get_sentences(s) 
    s = remove_emoji(s)
    s = remove_spaces(s)
    s = remove_newline(s)
    s=re.sub(r'\W+', ' ', s)  #remove special character 
    s = re.sub('[a-zA-Z_0-9]*', '', s) #remove number and english alphabets  !?
    s=normalize(s)
    s+='\n' #necessary !new line in csv !
    return s

def preprocess(cls_name):
    f_data= open("temp.csv", 'w')    
    if cls_name=="positive":
        df = dd.read_csv('sentiment_analysis/data/positive/*.csv', names=['head']) 
        print( "length of positive files  ",len(df))
    elif cls_name=="profane" :
        df = dd.read_csv('sentiment_analysis/data/profane/*.csv', names=['head']) 
        print( "length of profane files  ",len(df))
    else:
        df = dd.read_csv('sentiment_analysis/data/negative/*.csv', names=['head'])  
        print( "length of negative files  ",len(df))  

    count=0 #no need just to check data loss
    print("\nStart Processing data ...\n")
    for index,row in df.iterrows():
        if type(row['head']) == str and row['head']: #if str and not empty
            input ="__label__"+cls_name + "," +noise_remove(row['head'])
            f_data.write(input)

            count = count+1

    print("Input file denoised, normalized and labeled -->> ",f_data.name)
    f_data.close()

    print("\nWarning !!  Clean data.train and data.test before datapreprocessing!\n")
    print("now splitting into train and test.... ")
    f_train = open("data.train", 'a+')
    f_test = open("data.test", 'a+')


    df =dd.read_csv('temp.csv',names=['__label__'+cls_name,''])  #description as second row name instead of empty if english is not filtered !
        

    c=0
    for index,row in df.iterrows():
        if type(row['']) == str and row['']: #if not empty after normalization
            input = row['__label__'+cls_name] + ' ' +remove_spaces(row[''])+ "\n"
            
            #split into training set and test set 
            if index < count/1.3:
                f_train.write(input)
            else:
                f_test.write(input)
        else:
            #skip empty 
            c +=1;
    f_train.close();
    f_test.close();
    print("\nempty data skipped ", c ,"\n\n------------------------------\n")
    
    #remove duplicate via cmd line !? 
    #sort  data.train >temp 
    #uniq temp>train
    #same for data.test
    os.system("sort  data.train >temp && uniq temp>train  && sort  data.test >temp && uniq temp>test")

if __name__ == '__main__' :
    
    preprocess("positive")
    preprocess("negative")
    preprocess("profane")
    os.system("rm data.test data.train temp temp.csv")
    print("Finished !  Ready for Training!\n\n")
    