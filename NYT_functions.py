#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:19:15 2021

@author: apple
"""

def clean_text(txt_in):
    import re
    clean = re.sub('[^A-Za-z0-9]+', " ", txt_in).lower().strip()
    return clean

def file_seeker(path_in, sw):
    import os
    import pandas as pd
    import re
    import nltk
    dictionary = nltk.corpus.words.words("en")
    tmp_pd = pd.DataFrame()
    for dirName, subdirList, fileList in os.walk(path_in):
        tmp_dir_name = dirName.split("/")[-1::][0]
        try:
            for word in fileList:
                #print(word)
                tmp = open(dirName+"/"+word, "r", encoding="ISO-8859-1")
                tmp_f = tmp.read()
                tmp.close()
                tmp_f = re.sub('[^A-Za-z0-9]+', " ", tmp_f).strip().lower() #\n extra spacex
                if sw == "check_words":
                    tmp_f = [word for word in tmp_f.split() if word in dictionary]
                    tmp_f = ' '.join(tmp_f)
                tmp_pd = tmp_pd.append({'label': tmp_dir_name, 
                                        #'path': dirName+"/"+word}, 
                                        'body': tmp_f},
                                        ignore_index = True)
        except Exception as e:
            print (e)
            pass
    return tmp_pd

def rem_sw(var):
    from nltk.corpus import stopwords
    sw = set(stopwords.words("English"))
    my_test = [word for word in var.split() if word not in sw]
    my_test = ' '.join(my_test)
    return my_test

def check_dictionary(var):
    import nltk
    dictionary = nltk.corpus.words.words("en")
    fin_txt = [word for word in var.split() if word in dictionary]
    fin_txt = ' '.join(fin_txt)
    return fin_txt



def check_additional_sw(var):
    additional_sw = ["sonya","au","permission","proquest","new", "work", "time", "times","permiss", "one", "would", 
                     "two", "men", "man", "three", "year", "work", "span",
                     "r", "e", "p", "copyright", "index", "n", "a", "b", "c",
                     "d", "f", "g", "h", "i", "j","k","l","m","n","o","q",
                     "s","t","u","v","w",'x','y','z', "said", "could", "upon",
                     "may", "without", "made", "make", "through", "saying",
                     "even","New", "York", "Times", "copyright", "said", "would", "one" ,"two", "year", "mr", "added", "including", "years",
                      "monday", "see", "make", "three", "since", "say", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                     "made", "want", "many", "without", "need", "get", "way", "through", "whether", "next", "used", "saying", "going",
                     "even", "month", "week", "president", "trump", "led", "also", "Trump", "go", "man", "men", "still", "told", "may",
                  "back", "like", "news", "people", "take", "never", "often", "every", "though", "later", "recent", "use", "could", "much",
                 "last", "around", "put", "away", "began", "left", "four", "five", "six", "seven", "eight", "nine", "ten", "good", "months",
                 "accross", "others", "first", "second", "third", "ms", "already", "long", "known", "wrote", "among", "asked", "accross",
                 "according", "might", "high", "Obama", "obama", "Obama", "yesterday", "today", "tomorrow", "american", "government", "state",
                 "along", "must", "went", "little", "nan", "published", "oe", "te", "mrs",
                 "e", "ay", "te", "ta", "et", "ad", "however", "other", "another", "en", "os",
                 "ar", "shall", "less", "more", "great", "well", "large", "old", "yet", "nothing",
                 "work", "upon", "several", "whose", "matter", "es", "ever", "almost",
                 "know",  "ago", "cannot", "thing", "cause", "no", "yes", "er", 
                 "span", "index", "newspapers", "1923", "current", "reproduction", "reproduced", "1923current", "newyork", "prohibited",
                 "city", "country", "county", "day", "time", "either", "pg", "owner", "states", "united", "us", "reproduc", "reproduct", "word",
                   "office", "official", "name", "unit", "file","000","1","2","3","4","5","6","7","8","9","00","10","13","11","30","50","95","12","31","movie","age"]
    
    fin_txt = [word for word in var.split() if word not in additional_sw]
    fin_txt = ' '.join(fin_txt)
    return fin_txt

def word_cloud_save(table, pic_name, numbers_of_words):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    d = {}
    for a, x in zip(table.head(numbers_of_words)["token"],
                    table.head(numbers_of_words)["raw_frequency"]):
        d[a] = x
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(pic_name+".png")
    plt.show()

def stem_fun(var):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    tmp_txt = [stemmer.stem(word) for word in var.split()]
    tmp_txt = ' '.join(tmp_txt)
    return tmp_txt

def give_year_labels(df):
    import numpy as np
    conditions = [
        ((df["year"] >= 1851) & (df["year"] <= 1900)),
        ((df["year"] >= 1901) & (df["year"] <= 1930)),
        ((df["year"] >= 1931) & (df["year"] <= 1950)),
        ((df["year"] >= 1951) & (df["year"] <= 1960)),
        ((df["year"] >= 1961) & (df["year"] <= 1980))]
    values = ['1851-1900', '1901-1930', '1931-1950', '1951-1960', '1961-1980']
    df['label'] = np.select(conditions, values)
    df["num_tokens"] = df["clean_sw_dict_stem"].apply(lambda x: len(x.split(" ")))
    
    return df

def get_token_in_years(df, token_column, label_in):
    tokenlist = []
    for i in df[df.label ==  label_in][token_column]:
        tokenlist.extend(i.split(' '))
    return(tokenlist)


def token_to_probability_table(token_list):
    import pandas as pd
    from nltk.probability import FreqDist
    total_counts = sum(FreqDist(token_list).values()) #calculate the frequencies of tokens
    temp_pd = pd.DataFrame() 
    temp_pd = temp_pd.append({"probability(%)":"", "raw_frequency":"", "token":"", }, ignore_index = True) #create an empty dataframe
    temp_pd = temp_pd[["token", "raw_frequency", "probability(%)"]]
    for k, l in FreqDist(token_list).items(): #get the probability, frequencies, and tokens into dataframe
        dict1 = {"probability(%)":l*100/total_counts, "raw_frequency":l, "token":k}
        temp_pd = pd.concat([temp_pd, pd.DataFrame.from_dict(dict1, orient = "index").T])
    temp_pd = temp_pd[1::] #drop the first empty row
    temp_pd = temp_pd.sort_values(by=['probability(%)'], ascending=False) #sort by probability(%)
    return(temp_pd)

def jaccard_fun(var_a, var_b):
    #DOCUMENT SIMILARITY
    doc_a_set = set(var_a.split())
    doc_b_set = set(var_b.split())
    j_d = float(len(doc_a_set.intersection(
        doc_b_set)))/ float(len(doc_a_set.union(doc_b_set)))
    return j_d

def extract_embeddings_pre(df_in, num_vec_in, path_in, filename):
    from gensim.models import Word2Vec
    import pandas as pd
    from gensim.models import KeyedVectors
    import pickle
    my_model = KeyedVectors.load_word2vec_format(filename, binary=True) 
    my_model = Word2Vec(df_in.str.split(),
                        min_count=1, vector_size=num_vec_in)#size=300)
    #word_dict = my_model.wv.key_to_index
    #my_model.similarity("trout", "fish")
    def get_score(var):
        try:
            import numpy as np
            tmp_arr = list()
            for word in var:
                tmp_arr.append(list(my_model.wv[word]))
        except:
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    pickle.dump(my_model, open(path_in + "embeddings.pkl", "wb"))
    pickle.dump(tmp_data, open(path_in + "embeddings_df.pkl", "wb" ))
    return tmp_data

def cosine_fun(df_a, df_b, label_in): 
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    cos_matrix_t = pd.DataFrame(cosine_similarity(df_a, df_b))
    cos_matrix_t.index = label_in
    cos_matrix_t.columns = label_in
    return cos_matrix_t

def extract_embeddings_domain(df_in, num_vec_in, path_in, filename):
    #domain specific, train out own model specific to our domains
    from gensim.models import Word2Vec
    #import pandas as pd
    #import numpy as np
    #import pickle

    model = Word2Vec(
        df_in.str.split(), min_count=1,
        vector_size=num_vec_in, workers=3, window=5, sg=0)
    #wrd_dict = model.wv.key_to_index
    #model.wv.most_similar('machine', topn=10)
    # def get_score(var):
    #     try:
    #         tmp_arr = list()
    #         for word in var:
    #             tmp_arr.append(list(model.wv[word]))
    #     except:
    #         pass
    #     return np.mean(np.array(tmp_arr), axis=0)
    # tmp_out = df_in.str.split().apply(get_score)
    # tmp_data = tmp_out.apply(pd.Series).fillna(0)
    #model.wv.save_word2vec_format(path_in + "embeddings_domain.pkl")
    
    #pickle.dump(model, open(path_in + filename, "wb"))
    #pickle.dump(tmp_data, open(path_in + filename, "wb" ))
    return model #tmp_data, wrd_dict

def my_vec_fun(df_in, m, n, path_o):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    import pickle
    vectorizer = CountVectorizer(ngram_range=(m, n))
    my_vec_t = pd.DataFrame(vectorizer.fit_transform(df_in).toarray())
    my_vec_t.columns = vectorizer.vocabulary_
    pickle.dump(vectorizer, open(path_o + "vectorizer.pkl", "wb" ))
    pickle.dump(my_vec_t, open(path_o + "my_vec_data.pkl", "wb" ))
    return my_vec_t

def my_tf_idf_fun(df_in, m, n, path_o):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    import pickle
    vectorizer = TfidfVectorizer(ngram_range=(m, n))
    my_tf_idf_t = pd.DataFrame(vectorizer.fit_transform(df_in).toarray())
    my_tf_idf_t.columns = vectorizer.vocabulary_
    pickle.dump(vectorizer, open(path_o + "tf_idf.pkl", "wb" ))
    pickle.dump(my_tf_idf_t, open(path_o + "my_tf_idf_data.pkl", "wb" ))
    return my_tf_idf_t

def my_model_fun(df_in, label_in, path_o):
    #function 1
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support
    import pickle
    my_model = RandomForestClassifier(max_depth=10, random_state=123)
    #my_model = GaussianNB()
    #80/20 train,test,split
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, label_in, test_size=0.20, random_state=123)
    my_model.fit(X_train, y_train)
    pickle.dump(my_model, open(path_o + "my_model.pkl", "wb"))
    y_pred = my_model.predict(X_test)
    model_metrics = pd.DataFrame(precision_recall_fscore_support(
    y_test, y_pred, average='weighted'))
    model_metrics.index = ["precision", "recall", "fscore", "none"]
    #function 2 prediction
    the_preds = pd.DataFrame(my_model.predict_proba(X_test))
    the_preds.columns = my_model.classes_

    return my_model, model_metrics, the_preds

# def file_seeker(path_in):
#     import os
#     import pandas as pd
#     tmp_pd = pd.DataFrame()
#     for dirName, subdirList, fileList in os.walk(path_in):
#         tmp_dir_name = dirName.split("/")[-1::][0]
#         for word in fileList:
#             try:
#                 tmp_pd = tmp_pd.append(
#                     {"label": tmp_dir_name,"path": dirName+"/"+word},
#                     ignore_index=True)
#             except Exception as e:
#                 print (e)
#                 pass
#     return tmp_pd



def display_pca_scatterplot(model, words=None, sample=0):
    import numpy as np
    # Get the interactive Tools for Matplotlib
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    from sklearn.decomposition import PCA
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]
        
    word_vectors = np.array([model.wv[w] for w in words])

    twodim = PCA().fit_transform(word_vectors)[:,:2]
    
    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)


def word2vec_visual_hamilton(keyword, topn_number, filename, stem = True, google_background = False):
    the_path = "C:/Users/tosea/NYT_API/"
    import pickle
    import numpy as np
    topn_number = topn_number
    if stem == True:
        model_1851_1900  = pickle.load(open(
            the_path + "clean_sw_dict_stem_1851_1900_embeddings_domain_model.pkl", 'rb'))
        model_1901_1930  = pickle.load(open(
            the_path + "clean_sw_dict_stem_1901_1930_embeddings_domain_model.pkl", 'rb'))
        model_1931_1950  = pickle.load(open(
            the_path + "clean_sw_dict_stem_1931_1950_embeddings_domain_model.pkl", 'rb'))
        model_1951_1960  = pickle.load(open(
            the_path + "clean_sw_dict_stem_1951_1960_embeddings_domain_model.pkl", 'rb'))
        model_1961_1980  = pickle.load(open(
            the_path + "clean_sw_dict_stem_1961_1980_embeddings_domain_model.pkl", 'rb'))
    else:
        model_1851_1900  = pickle.load(open(
            the_path + "clean_sw_dict_1851_1900_embeddings_domain_model.pkl", 'rb'))
        model_1901_1930  = pickle.load(open(
            the_path + "clean_sw_dict_1901_1930_embeddings_domain_model.pkl", 'rb'))
        model_1931_1950  = pickle.load(open(
            the_path + "clean_sw_dict_1931_1950_embeddings_domain_model.pkl", 'rb'))
        model_1951_1960  = pickle.load(open(
            the_path + "clean_sw_dict_1951_1960_embeddings_domain_model.pkl", 'rb'))
        model_1961_1980  = pickle.load(open(
            the_path + "clean_sw_dict_1961_1980_embeddings_domain_model.pkl", 'rb'))
    
    models = [model_1851_1900, model_1901_1930, 
              model_1931_1950, model_1951_1960,
              model_1961_1980]

    total_similar_words = []
    embeddings = []
    similarity = []

    for model in models:
        for similar_word, sim in model.wv.most_similar(keyword, topn=topn_number):
            total_similar_words.append(similar_word)
            embeddings.append(model.wv[similar_word])
            similarity.append(sim)

    if google_background == True:
        from gensim.models import KeyedVectors
        the_path_in = "C:/Users/tosea/NLP@Columbia/GOOGLEVECTORS/"
        google_model = KeyedVectors.load_word2vec_format(the_path_in + "GoogleNews-vectors-negative300.bin.gz", binary=True) 
        word_vectors = []
        for w in total_similar_words:
            try:
                word_vectors.append(google_model[w])
            except KeyError:
                try:
                    word_vectors.append(google_model[w.capitalize()])
                except KeyError:
                    total_similar_words.remove(w)
                    print(w, "has been removed from similar words")
    else:
        word_vectors = [model_1961_1980.wv[w] for w in total_similar_words]


        
    
    import plotly
    import plotly.graph_objs as go
    from sklearn.manifold import TSNE

    
    topn = topn_number
    ##############  
    
    
    word_vectors.append(model_1851_1900.wv[keyword])
    word_vectors.append(model_1901_1930.wv[keyword])
    word_vectors.append(model_1931_1950.wv[keyword])
    word_vectors.append(model_1951_1960.wv[keyword])
    word_vectors.append(model_1961_1980.wv[keyword])
    
    word_vectors = np.array(word_vectors)
    
    two_dim = TSNE(n_components = 2, random_state=0, perplexity = 5, learning_rate = 500, n_iter = 10000).fit_transform(word_vectors)[:,:2]
    
    
    # For 2D, change the three_dim variable into something like two_dim like the following:
        # two_dim = TSNE(n_components = 2, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:2]

    data = []
    
    
    count = 0
    user_input = [f"{keyword}(1851-1900)", f"{keyword}(1901-1930)", 
              f"{keyword}(1931-1950)", f"{keyword}(1951-1960)",
              f"{keyword}(1961-1980)"]
    
    year = ["1851-1900", "1901-1930", 
              "1931-1950", "1951-1960",
              "1961-1980"]
    
    for i in range (5):
    
                trace = go.Scatter(
                    x = two_dim[count:count+topn,0], 
                    y = two_dim[count:count+topn,1],  
    
                    text = total_similar_words[count:count+topn],
                    name = user_input[i],
                    textposition = "top center",
                    textfont_size = 12,
                    mode = 'markers+text',
                    marker = {
                        'size': 8,
                        'opacity': 0.8,
                        'color': "gray"
                    }
       
                )
                
                # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
                # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
                data.append(trace)
                count = count+topn
    
    trace_input = go.Scatter(
                    x = two_dim[count:,0], 
                    y = two_dim[count:,1],  
    
                    text = year,
                    name = f"{keyword}-Years",
                    textposition = "top center",
                    textfont_size = 18,
                    mode = 'markers+text',
                    marker = {
                        'size': 18,
                        'opacity': 1,
                        'color': 'red'
                    }
                    )
    
    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
    data.append(trace_input)
    
    # Configure the layout
    
    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=20,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 12),
        autosize = False,
        width = 1200,
        height = 1200
        )
    
    
    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.write_image(filename)
    plot_figure.show()

def word2vec_visual_cluster(keyword, topn_number, filename, stem = True, google_background = False):
    import numpy as np
    import pickle
    the_path = "C:/Users/tosea/NYT_API/"
    topn_number = topn_number
    topn = topn_number
    if stem == True:
        model_1851_1900  = pickle.load(open(
            the_path + "clean_sw_dict_stem_1851_1900_embeddings_domain_model.pkl", 'rb'))
        model_1901_1930  = pickle.load(open(
            the_path + "clean_sw_dict_stem_1901_1930_embeddings_domain_model.pkl", 'rb'))
        model_1931_1950  = pickle.load(open(
            the_path + "clean_sw_dict_stem_1931_1950_embeddings_domain_model.pkl", 'rb'))
        model_1951_1960  = pickle.load(open(
            the_path + "clean_sw_dict_stem_1951_1960_embeddings_domain_model.pkl", 'rb'))
        model_1961_1980  = pickle.load(open(
            the_path + "clean_sw_dict_stem_1961_1980_embeddings_domain_model.pkl", 'rb'))
    
    else:
        model_1851_1900  = pickle.load(open(
            the_path + "clean_sw_dict_1851_1900_embeddings_domain_model.pkl", 'rb'))
        model_1901_1930  = pickle.load(open(
            the_path + "clean_sw_dict_1901_1930_embeddings_domain_model.pkl", 'rb'))
        model_1931_1950  = pickle.load(open(
            the_path + "clean_sw_dict_1931_1950_embeddings_domain_model.pkl", 'rb'))
        model_1951_1960  = pickle.load(open(
            the_path + "clean_sw_dict_1951_1960_embeddings_domain_model.pkl", 'rb'))
        model_1961_1980  = pickle.load(open(
            the_path + "clean_sw_dict_1961_1980_embeddings_domain_model.pkl", 'rb'))
    
    
    models = [model_1851_1900, model_1901_1930, 
              model_1931_1950, model_1951_1960,
              model_1961_1980]
    
    total_similar_words = []
    embeddings = []
    similarity = []
    
    for model in models:
        for similar_word, sim in model.wv.most_similar(keyword, topn=topn_number):
            total_similar_words.append(similar_word)
            embeddings.append(model.wv[similar_word])
            similarity.append(sim)
    
    
    
    #####
    import plotly
    import plotly.graph_objs as go
    from sklearn.manifold import TSNE
    
    embeddings.append(model_1851_1900.wv[keyword])
    embeddings.append(model_1901_1930.wv[keyword])
    embeddings.append(model_1931_1950.wv[keyword])
    embeddings.append(model_1951_1960.wv[keyword])
    embeddings.append(model_1961_1980.wv[keyword])

    embeddings = np.array(embeddings)
    n, m = embeddings.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    two_dim  = np.array(tsne_model_en_2d.fit_transform(embeddings.reshape(n, m)))
    

    
    
    
        ##############  
        
    
    
    #two_dim = TSNE(n_components = 2, random_state=0, perplexity = 5, learning_rate = 500, n_iter = 10000).fit_transform(embeddings)[:,:2]
    
    
    # For 2D, change the three_dim variable into something like two_dim like the following:
        # two_dim = TSNE(n_components = 2, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:2]
    
    data = []
    
    
    count = 0
    user_input = [f"{keyword}(1851-1900)", f"{keyword}(1901-1930)", 
              f"{keyword}(1931-1950)", f"{keyword}(1951-1960)",
              f"{keyword}(1961-1980)"]
    
    year = ["1851-1900", "1901-1930", 
              "1931-1950", "1951-1960",
              "1961-1980"]
    
    for i in range (5):
    
                trace = go.Scatter(
                    x = two_dim[count:count+topn,0], 
                    y = two_dim[count:count+topn,1],  
    
                    text = total_similar_words[count:count+topn],
                    name = user_input[i],
                    textposition = "top center",
                    textfont_size = 16,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 2
                    }
       
                )
                
                # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
                # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
                data.append(trace)
                count = count+topn
    
    trace_input = go.Scatter(
                    x = two_dim[count:,0], 
                    y = two_dim[count:,1],  
    
                    text = year,
                    name = f"{keyword}-Years",
                    textposition = "top center",
                    textfont_size = 18,
                    mode = 'markers+text',
                    marker = {
                        'size': 18,
                        'opacity': 1,
                        'color': 'red'
                    }
                    )
    
    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
    data.append(trace_input)
    
    # Configure the layout
    
    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=20,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 12),
        autosize = False,
        width = 1200,
        height = 1200
        )
    
    
    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.write_image(filename)
    plot_figure.show()
    
    
    
def score_text(sample_text_i, the_path_i):
    #Classify arbitrary text, need to run ALL same pre-processing and 
    #transformation steps as what was applied to the training set
    #Step 1 call; up clean_text function
    import numpy as np
    from my_functions_NYT import clean_text, stem_fun
    import pickle 
    clean_text = clean_text(sample_text_i)
    #step 2 call up stem_fun
    clean_text = stem_fun(clean_text)
    #call up the appropriate transformation algo and transform our new data
    vec_model = pickle.load(open(the_path_i + "vectorizer.pkl", "rb"))
    clean_text_vec = vec_model.transform([clean_text]).toarray()
    my_model = pickle.load(open(the_path_i + "my_model.pkl", "rb"))
    the_pred = my_model.predict(clean_text_vec)
    the_pred_proba = np.max(np.array(my_model.predict_proba(clean_text_vec)))
    print ("sample_text predicted as", the_pred[0],
           "with likelihood of", the_pred_proba)
    return the_pred, the_pred_proba