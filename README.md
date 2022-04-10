# 2022-Datahacks Intermediate Report - Prediction of a potential deal

### Team Members: Ziyu Wang, Zhaoyu Zhang

## Table of Contents:
 1. [Introduction](#i-introduction)
 2. [Data Cleaning &amp; Pre-processing](#ii-data-cleaning--pre-processing)
 3. [Data Visualization](#iii-data-visualization)
 4. [Modeling &amp; Analysis](#iv-modeling--analysis)
 5. [Conclusion](#v-conclusion)

---

# I. Introduction

In this project, we are trying to use Natural Language Processing to predict the pitch result of businesses in front of seasoned investors (aka sharks) who decide whether or not to invest in the businesses in Shark Tank. We exploited 4 ways to build our model:  
- CountVector and XGBoost
- TF-IDF and Decision Tree
- Random Forest
- LSTM

And we used permutation testing to generate tests on the accuracy of our models and conclude the most effective model in this scenario.


# II. Data Cleaning & Pre-processing:

* **NLP Visualizations**: To visualize the questions in this dataset, we cleaned the data by only keeping each valid english letter. Since this section of our analysis was done on the natural language in the dataset we only needed these 2 fields for all companies in our dataset to build a corpus.

* **LSTM**: Similar to the NLP section above, we built a corpus to analyze by using only the descriptions in the dataset and their corresponding outcome(status). After this, we looked at the distribution of description, which is relatively small and random. Deep learning models usually require a lot of data to learn intuitions about the structure of natural language and for this reason, we think there was little benefit in attempting to classify them when we are out of time embedding appropriate word vecotrs. If we had kept all these descriptions then our model would most likely have overfit on some of them. 

* **Cleaning and Pro-processing**: The dataset is relatvely clean and tidy. There does not exist missingness or abnormals. We decided to replace invalid characters and use regular tense of the word to keep track of the frequency. Furthermore, we did not discard or focus on any of the description since we found the ```Business_Identifier``` are chose randomly, which has little impact on classification. After understanding the data, we noticed that data comes from the TV show Shark Tank which started broadcasting in 2009, so it's appropriate to ignore geographic/spatial/time bias.

# III. Data Visualization:
* **Tableau Dashboard Links**:
  * [Word Cloud](https://user-images.githubusercontent.com/57332517/162641839-645f481a-273a-4a13-8db1-00d479cf2bb8.png)

  * [Description Length](https://user-images.githubusercontent.com/57332517/162641828-d731fec9-8caa-4409-bf25-f631c1d0c38b.png)

  * [Word Frequency Distibution](https://user-images.githubusercontent.com/57332517/162641802-f0d650e4-d798-49aa-bf31-3abd8a09027b.png)


* **Methodology**: The first step towards achieving these visualizations was to extract meaningful features from the text. We use Term Frequency-Inverse Document Frequency(TF-IDF). We used the sklearn module to create a sparse TF-IDF matrix for our corpus.

We tried the reduction algorithms in the sklearn module: Principal Component Analysis(PCA), Singular Value Decomposition(SVD). It was difficult to see any logical divisions or clusterings when visualizing these methods.

# IV. Modeling & Analysis:

* **LSTM**: At first we thought it will be effective since we know LSTM is very good at filtering spam emails which is a similar classification problem. But after the implementation, we found that it's hard for us to embed an appropriate word vector in a shout period of time. This potentially caused our training set accuracy was over 70% while our test set accuracy was < 55%. We tried to change parameters since we applied skleanr.keras.sequential package.We tried to limit the word length and the random state, the improvement was little. We also changed dropout on our LSTM cell so that our model learns not to rely too heavily on certain features of the cell states from previous timesteps and the word embeddings. 

* **CountVector and XGBoost**: This is a straightforward model since we applied the built-in package of CountVectorizer which quantify every word into a matrix that counts the appearance. We found that in our permutation test, the performance will be worse when we include stop-words in English to exclude useless meanings. Unsure about whether it will help to improve accuracy in test data.

* **TF-IDF, Decision Tree**: We create TF-IDF model which compare 1-gram and 2-gram as follows:
```
def KNN_TFIDF():
    # x, y
    x_train, x_test, y_train, y_test = train_test_split(dff['Des'], dff['Status'],\
         test_size = 0.2, random_state = 50)
    # TF-IDF
    vect      = TfidfVectorizer(min_df = 5, max_df = 0.7, \
        sublinear_tf = True, use_idf = True)
    train_vect= vect.fit_transform(x_train)
    test_vect = vect.transform(x_test)
    
    for k in [1,2]: # 1-gram, 2-gram
        model = KNeighborsClassifier(n_neighbors = k)
        model.fit(train_vect, y_train)

        predicted = model.predict(test_vect)
        accuracy  = model.score(train_vect, y_train)
        report = classification_report(y_test, predicted, output_dict=True)

        print("Classification for k = {} is:".format(k))
        print('Test-set Accuracy:', accuracy_score(y_test, predicted))
        print('\n')
```

* **Random Forest**: We applied random forest to vectorize the words and use a step-increasing threshold to check the highest classification line, which turns out to be around 0.6-0.63. We do believe random forest classification would perform well in a bigger size of data, and this counts as the reason for our choice of TF-IDF in the end.
```
Threshold 0.2 -- score 0.77
Threshold 0.25 -- score 0.87
Threshold 0.3 -- score 0.9
Threshold 0.35 -- score 0.93
Threshold 0.4 -- score 0.94
Threshold 0.45 -- score 0.94
Threshold 0.5 -- score 0.95
Threshold 0.55 -- score 0.95
Threshold 0.6 -- score 0.95
Threshold 0.65 -- score 0.95
Threshold 0.7 -- score 0.95
Threshold 0.75 -- score 0.91
---Optimum Threshold --- 0.6 --ROC-- 0.95
```

Overall, three of our models generate similar performance which is not clear from our permutation test. We tried to optimize the outcomes but the difference is still tiny.

# V. Conclusion:

Among all methods, TF-IDF and decision tree has the best performance, with accuracy approximately 65% in our own permutation testing.  That might be due to the logic of decision if best suitable in determination of pitch, investors will consider the paramters based on the previous parameters exposed by the business. We are unsure about the performance for test data since three models have a similar performance in our permutation test.

Future Improvements

Our model is pretty inaccurate because we meaured all the businesses in the same way, based on the tokens in their descriptions. However, in real life, investors  are likely to measure businesses with respect to the genres of the products. It might be helpful that we can identify certain keywords in the description to catagorize businesses and then use different datasets to train different models.
