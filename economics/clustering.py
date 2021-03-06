# =============================================================================
# NLP Clustering
# =============================================================================
# Original code by Paul Huynh 
# Modified by John Kiley
# =============================================================================
# packages required to run code.  Make sure to install all required packages.
# =============================================================================

import re,string
import multiprocessing
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.manifold import MDS
from sklearn.manifold import TSNE

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import matplotlib.pyplot as plt

import pandas as pd
import os

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import numpy as np

import json_lines

os.chdir(r'C:\Users\johnk\Desktop\Grad School\6. Spring 2019\1. MSDS_453_NLP\6. Homework\week7\economics\econ\files')

RANDOM_SEED=9999

print('Import packages complete ---------------------------------------------')

#%%
# =============================================================================
# read train and test dataframe.  Headers are ['category', 'corpus', 'dataset', 'filename', 'text']
# =============================================================================

labels={'labels':[]}
text={'text':[]}
dataset={'dataset':[]}
title={'title':[]}


with open('all.jsonl', 'rb') as f:
    for item in json_lines.reader(f):
        labels['labels'].append(item['labels'])
        text['text'].append(item['text'])
        title['title'].append(item['title'])

data=pd.concat([pd.DataFrame(labels),pd.DataFrame(text),pd.DataFrame(dataset),
                pd.DataFrame(title)], axis=1)

# Sample set is small enough to leverage all of the documents 211
data=data.sample(n=211, replace=False, random_state =RANDOM_SEED)
data=data.reset_index()

print('Generate dataframes complete -----------------------------------------')

#%%
# =============================================================================
# Function to process documents
# =============================================================================

def clean_doc(doc): 
    #split document into individual words
    tokens=doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 4]
    tokens = [word for word in tokens if len(word) < 21]
    #lowercase all words
    tokens = [word.lower() for word in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]         
    # word stemming    
    # ps=PorterStemmer()
    # tokens=[ps.stem(word) for word in tokens]
    return tokens

print('Define preprocssing function complete --------------------------------')

#%%
# =============================================================================
# Processing text into lists
# =============================================================================

#create empty list to store text documents titles
titles=[]

#for loop which appends the DSI title to the titles list
for i in range(0,len(data)):
    temp_text=data['title'].iloc[i]
    titles.append(temp_text)

print('Extract titles complete-----------------------------------------------')


#%%

# =============================================================================
# Processing text into lists
# =============================================================================

#create empty list to store text documents
text_body=[]

#for loop which appends the text to the text_body list
for i in range(0,len(data)):
    temp_text=data['text'].iloc[i]
    text_body.append(temp_text)
    
print('Extract text complete ------------------------------------------------')
    

#%%    
# =============================================================================
# Processing text into lists
# =============================================================================

#empty list to store processed documents
processed_text=[]
#for loop to process the text to the processed_text list
for i in text_body:
    text=clean_doc(i)
    processed_text.append(text)

print('Process & store text complete ----------------------------------------')


#Note: the processed_text is the PROCESSED list of documents read directly form 
#the csv.  Note the list of words is separated by commas.
#%%
# =============================================================================
# stitch back together individual words to reform body of text
# =============================================================================

final_processed_text=[]

for i in processed_text:
    temp_DSI=' '.join(i)
    final_processed_text.append(temp_DSI)

   
#Note: We stitched the processed text together so the TFIDF vectorizer can work.
#Final section of code has 3 lists used.  2 of which are used for further processing.
#(1) text_body - unused, (2) processed_text (used in W2V), 
#(3) final_processed_text (used in TFIDF), and (4) DSI titles (used in TFIDF Matrix)

print('Finalize preprocessing complete --------------------------------------')

#%%
# =============================================================================
# Sklearn TFIDF 
# =============================================================================

#note the ngram_range will allow you to include multiple words within the TFIDF matrix
#Call Tfidf Vectorizer
Tfidf=TfidfVectorizer(ngram_range=(1,1))

#fit the vectorizer using final processed documents.  The vectorizer requires the 
#stiched back together document.

TFIDF_matrix=Tfidf.fit_transform(final_processed_text)     

#creating datafram from TFIDF Matrix
matrix=pd.DataFrame(TFIDF_matrix.toarray(), columns=Tfidf.get_feature_names(), index=titles)

print('Sklearn TFIDF complete -----------------------------------------------')


#%%
# =============================================================================
# K Means Clustering - TFIDF
# =============================================================================

k=2
km = KMeans(n_clusters=k, random_state =RANDOM_SEED)
km.fit(TFIDF_matrix)
clusters = km.labels_.tolist()


terms = Tfidf.get_feature_names()
Dictionary={'Doc Name':titles, 'Cluster':clusters,  'Text': final_processed_text}
frame=pd.DataFrame(Dictionary, columns=['Cluster', 'Doc Name','Text'])


frame=pd.concat([frame,data['labels']], axis=1)

frame['record']=1

print('TF-IDF Kmeans Clustering Complete -------------------------------------------')


#%%
# =============================================================================
# Pivot table to see see how clusters compare to categories
# =============================================================================

pivot=pd.pivot_table(frame, values='record', index='labels',
                     columns='Cluster', aggfunc=np.sum)

print(pivot)

print('TF-IDF Pivot Table complete -------------------------------------------------')


#%%
# =============================================================================
# Top Terms per cluster
# =============================================================================

print("Top terms per cluster:")
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

terms_dict=[]


#save the terms for each cluster and document to dictionaries.  To be used later
#for plotting output.

#dictionary to store terms and titles
cluster_terms={}
cluster_title={}


for i in range(k):
    print("Cluster %d:" % i),
    temp_terms=[]
    temp_titles=[]
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
        terms_dict.append(terms[ind])
        temp_terms.append(terms[ind])
    cluster_terms[i]=temp_terms
    
    print("Cluster %d titles:" % i, end='')
    temp=frame[frame['Cluster']==i]
    for title in temp['Doc Name']:
        print(' %s,' % title, end='')
        temp_titles.append(title)
    cluster_title[i]=temp_titles

print('Identify top terms per cluster complete ------------------------------')

#%%
# =============================================================================
# TF-IDF Plotting - mds algorithm
# =============================================================================
 
# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.


mds = MDS(n_components=2, dissimilarity="precomputed", random_state=RANDOM_SEED)
#mds = TSNE(n_components=2, metric="euclidean", random_state=RANDOM_SEED)

dist = 1 - cosine_similarity(TFIDF_matrix)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]


#set up colors per clusters using a dict.  number of colors must correspond to K
cluster_colors = {0: 'black', 1: 'orange', 2: 'blue', 3: 'rosybrown', 4: 'firebrick', 
                  5:'red', 6:'darksalmon', 7:'sienna'}


#set up cluster names using a dict.  
cluster_dict=cluster_title

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=range(0,len(clusters)))) 

#group by cluster
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(12, 12)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y,
            marker='o', linestyle='', ms=12,
            label=cluster_dict[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True)
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=True)

plt.title('TF-IDF Clustering | MDS Algorithm')
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))      #show legend with only 1 point

#The following section of code is to run the k-means algorithm on the doc2vec outputs.
#note the differences in document clusters compared to the TFIDF matrix.

print('TF-IDF | MDS | Plot clusters complete -----------------------------------------------')

#%%
# =============================================================================
# TF-IDF Plotting - TSNE algorithm
# =============================================================================
 
# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.


#mds = MDS(n_components=2, dissimilarity="precomputed", random_state=RANDOM_SEED)
mds = TSNE(n_components=2, metric="euclidean", random_state=RANDOM_SEED)

dist = 1 - cosine_similarity(TFIDF_matrix)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]


#set up colors per clusters using a dict.  number of colors must correspond to K
cluster_colors = {0: 'black', 1: 'orange', 2: 'blue', 3: 'rosybrown', 4: 'firebrick', 
                  5:'red', 6:'darksalmon', 7:'sienna'}


#set up cluster names using a dict.  
cluster_dict=cluster_title

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=range(0,len(clusters)))) 

#group by cluster
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(12, 12)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y,
            marker='o', linestyle='', ms=12,
            label=cluster_dict[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True)
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=True)
    
plt.title('TF-IDF Clustering | TSNE Algorithm')
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))      #show legend with only 1 point

print('TF-IDF | TSNE | Plot clusters complete -----------------------------------------------')
#%%
# =============================================================================
# K Means Clustering - Terms - TFIDF 
# =============================================================================
# =============================================================================
# Sklearn TFIDF 
# =============================================================================

#note the ngram_range will allow you to include multiple words within the TFIDF matrix
#Call Tfidf Vectorizer
Tfidf=TfidfVectorizer(ngram_range=(1,1))

#fit the vectorizer using final processed documents.  The vectorizer requires the 
#stiched back together document.

TFIDF_matrix=Tfidf.fit_transform(final_processed_text)     

#creating datafram from TFIDF Matrix
matrix=pd.DataFrame(TFIDF_matrix.toarray(), columns=Tfidf.get_feature_names(), index=titles)

print('Sklearn TFIDF complete -----------------------------------------------')

matrix=matrix.transpose()

k=2
km = KMeans(n_clusters=k, random_state =RANDOM_SEED)
km.fit(matrix)
clusters = km.labels_.tolist()

terms = Tfidf.get_feature_names()
Dictionary={'Doc Name':terms, 'Cluster':clusters}
frame=pd.DataFrame(Dictionary, columns=['Cluster', 'Doc Name'])

#matrix=matrix.sample(n=1000, replace=False, random_state =RANDOM_SEED)


print('Terms - TF-IDF Kmeans Clustering Complete -------------------------------------------')

# =============================================================================
# #%%
# # =============================================================================
# # Terms - TF-IDF Plotting - mds algorithm
# # =============================================================================
#  
# # convert two components as we're plotting points in a two-dimensional plane
# # "precomputed" because we provide a distance matrix
# # we will also specify `random_state` so the plot is reproducible.
# 
# 
# mds = MDS(n_components=2, dissimilarity="precomputed", random_state=RANDOM_SEED)
# #mds = TSNE(n_components=2, metric="euclidean", random_state=RANDOM_SEED)
# 
# dist = 1 - cosine_similarity(matrix)
# 
# pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
# 
# xs, ys = pos[:, 0], pos[:, 1]
# 
# 
# #set up colors per clusters using a dict.  number of colors must correspond to K
# cluster_colors = {0: 'black', 1: 'orange', 2: 'blue', 3: 'rosybrown', 4: 'firebrick', 
#                   5:'red', 6:'darksalmon', 7:'sienna'}
# 
# 
# #set up cluster names using a dict.  
# cluster_dict=cluster_title
# 
# #create data frame that has the result of the MDS plus the cluster numbers and titles
# df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=range(0,len(clusters)))) 
# 
# #%%
# df=df.sample(n=211, replace=False, random_state =RANDOM_SEED)
# 
# #group by cluster
# groups = df.groupby('label')
# 
# fig, ax = plt.subplots(figsize=(12, 12)) # set size
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# 
# #iterate through groups to layer the plot
# #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
# for name, group in groups:
#     ax.plot(group.x, group.y,
#             marker='o', linestyle='', ms=12,
#             label=cluster_dict[name], color=cluster_colors[name], 
#             mec='none')
#     ax.set_aspect('auto')
#     ax.tick_params(\
#         axis= 'x',          # changes apply to the x-axis
#         which='both',      # both major and minor ticks are affected
#         bottom=False,      # ticks along the bottom edge are off
#         top=False,         # ticks along the top edge are off
#         labelbottom=True)
#     ax.tick_params(\
#         axis= 'y',         # changes apply to the y-axis
#         which='both',      # both major and minor ticks are affected
#         left=False,      # ticks along the bottom edge are off
#         top=False,         # ticks along the top edge are off
#         labelleft=True)
# 
# plt.title('TF-IDF Clustering | MDS Algorithm')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))      #show legend with only 1 point
# 
# #The following section of code is to run the k-means algorithm on the doc2vec outputs.
# #note the differences in document clusters compared to the TFIDF matrix.
# 
# print('TF-IDF | MDS | Terms | Plot clusters complete -----------------------------------------------')
# 
# #%%
# # =============================================================================
# # Terms - TF-IDF Plotting - TSNE algorithm
# # =============================================================================
#  
# # convert two components as we're plotting points in a two-dimensional plane
# # "precomputed" because we provide a distance matrix
# # we will also specify `random_state` so the plot is reproducible.
# 
# 
# #mds = MDS(n_components=2, dissimilarity="precomputed", random_state=RANDOM_SEED)
# mds = TSNE(n_components=2, metric="euclidean", random_state=RANDOM_SEED)
# 
# dist = 1 - cosine_similarity(matrix)
# 
# pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
# 
# xs, ys = pos[:, 0], pos[:, 1]
# 
# 
# #set up colors per clusters using a dict.  number of colors must correspond to K
# cluster_colors = {0: 'black', 1: 'orange', 2: 'blue', 3: 'rosybrown', 4: 'firebrick', 
#                   5:'red', 6:'darksalmon', 7:'sienna'}
# 
# 
# #set up cluster names using a dict.  
# cluster_dict=cluster_title
# 
# #create data frame that has the result of the MDS plus the cluster numbers and titles
# df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=range(0,len(clusters)))) 
# 
# #group by cluster
# groups = df.groupby('label')
# 
# fig, ax = plt.subplots(figsize=(12, 12)) # set size
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# 
# #iterate through groups to layer the plot
# #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
# for name, group in groups:
#     ax.plot(group.x, group.y,
#             marker='o', linestyle='', ms=12,
#             label=cluster_dict[name], color=cluster_colors[name], 
#             mec='none')
#     ax.set_aspect('auto')
#     ax.tick_params(\
#         axis= 'x',          # changes apply to the x-axis
#         which='both',      # both major and minor ticks are affected
#         bottom=False,      # ticks along the bottom edge are off
#         top=False,         # ticks along the top edge are off
#         labelbottom=True)
#     ax.tick_params(\
#         axis= 'y',         # changes apply to the y-axis
#         which='both',      # both major and minor ticks are affected
#         left=False,      # ticks along the bottom edge are off
#         top=False,         # ticks along the top edge are off
#         labelleft=True)
#     
# plt.title('TF-IDF Terms Clustering | TSNE Algorithm')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))      #show legend with only 1 point
# 
# print('TF-IDF | TSNE | Terms | Plot clusters complete -----------------------------------------------')
# =============================================================================

#%%
# =============================================================================
# Doc2Vec
# =============================================================================

print('\nBegin Doc2Vec Work')
cores = multiprocessing.cpu_count()
print("\nNumber of processor cores:", cores)

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(final_processed_text)]
model = Doc2Vec(documents, vector_size = 500, window = 2, 
	min_count = 2, workers = cores, epochs = 40)

doc2vec_df=pd.DataFrame()
for i in range(0,len(processed_text)):
    vector=pd.DataFrame(model.infer_vector(processed_text[i])).transpose()
    doc2vec_df=pd.concat([doc2vec_df,vector], axis=0)

#doc2vec_df=doc2vec_df.reset_index()

#doc_titles={'title': titles}
#t=pd.DataFrame(doc_titles)

#doc2vec_df=pd.concat([doc2vec_df,t], axis=1)

#doc2vec_df=doc2vec_df.drop('index', axis=1)
#doc2vec_df = doc2vec_df.set_index('title')

print('Doc2Vec Complete -----------------------------------------------------')

#%%
# =============================================================================
# K Means Clustering - Doc2Vec
# =============================================================================

k=2
km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init = 100, max_iter = 500)
km.fit(doc2vec_df)
clusters = km.labels_.tolist()

Dictionary={'Doc Name':titles, 'Cluster':clusters,  'Text': final_processed_text}
frame=pd.DataFrame(Dictionary, columns=['Cluster', 'Doc Name','Text'])
frame=pd.concat([frame,data['labels']], axis=1)

frame['record']=1

print('Doc2Vec Kmeans Clustering Complete -------------------------------------------')
#%%
# =============================================================================
# Doc2Vec Pivot Table
# =============================================================================

pivot=pd.pivot_table(frame, values='record', index='labels',
                     columns='Cluster', aggfunc=np.sum)

print(pivot)

print('Doc2Vec Pivot Table complete -------------------------------------------------')

#%%
# =============================================================================
# Doc2Vec Plotting - mds algorithm
# =============================================================================
 
# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.


mds = MDS(n_components=2, dissimilarity="precomputed", random_state=RANDOM_SEED)
#mds = TSNE(n_components=2, metric="euclidean", random_state=RANDOM_SEED)

dist = 1 - cosine_similarity(doc2vec_df)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]


#set up colors per clusters using a dict.  number of colors must correspond to K
cluster_colors = {0: 'black', 1: 'orange', 2: 'blue', 3: 'rosybrown', 4: 'firebrick', 
                  5:'red', 6:'darksalmon', 7:'sienna'}


#set up cluster names using a dict.  
cluster_dict=cluster_title

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=range(0,len(clusters)))) 

#group by cluster
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(12, 12)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y,
            marker='o', linestyle='', ms=12,
            label=cluster_dict[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True)
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=True)
    
plt.title('Doc2Vec Clustering | MDS Algorithm')
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))      #show legend with only 1 point

#The following section of code is to run the k-means algorithm on the doc2vec outputs.
#note the differences in document clusters compared to the TFIDF matrix.

print('doc2vec | MDS | Plot clusters complete -----------------------------------------------')

#%%
# =============================================================================
# doc2Vec Plotting - TSNE algorithm
# =============================================================================
 
# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.


#mds = MDS(n_components=2, dissimilarity="precomputed", random_state=RANDOM_SEED)
mds = TSNE(n_components=2, metric="euclidean", random_state=RANDOM_SEED)

dist = 1 - cosine_similarity(doc2vec_df)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]


#set up colors per clusters using a dict.  number of colors must correspond to K
cluster_colors = {0: 'black', 1: 'orange', 2: 'blue', 3: 'rosybrown', 4: 'firebrick', 
                  5:'red', 6:'darksalmon', 7:'sienna'}


#set up cluster names using a dict.  
cluster_dict=cluster_title

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=range(0,len(clusters)))) 

#group by cluster
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(12, 12)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y,
            marker='o', linestyle='', ms=12,
            label=cluster_dict[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True)
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=True)
    
plt.title('Doc2Vec Clustering | TSNE Algorithm')
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))      #show legend with only 1 point

print('Doc2Vec | TSNE | Plot clusters complete -----------------------------------------------')
