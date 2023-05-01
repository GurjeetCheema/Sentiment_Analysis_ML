# Sentiment_Analysis_ML

## Introduction
Twitter, created in 2006, has been one of the most frequently used sites on the internet.(Arrington, 2006)(Rayome, 2019) With its short and frequent, yet public, posting methods by governments, celebrities (Kanalley, 2013) and people alike, it has proven to be an effective way of conveying thoughts and feelings on various topics.
This report aims to find the most effective model for analysing, then sorting, the tweets in three different sentiment groups: positive, neutral, and negative. The models we have selected are Decision Trees, K nearest neighbours and logistic regression due to the 1 dimensional nature of the data.

## Analysis
To identify aspects of the features the entire test dataset was graphed and compared to several visible aspects of tweets; presence of urls, hashtags and mentions. 

![unnamed](https://user-images.githubusercontent.com/124246311/235434944-a5170744-4886-4ca2-b1d3-610fe45dfd6b.png)
![unnamed2](https://user-images.githubusercontent.com/124246311/235434948-1cf4681b-38ae-49d3-b1e5-f5bd6bd6aebd.png)
![unnamed3](https://user-images.githubusercontent.com/124246311/235434953-f7e3ac72-c67b-44d0-b6e9-1e0f68b3ec3f.png)


As displayed in these graphs, the frequency of urls,hashtags and mentions for each sentiment are negligible as it seems to scale according to the size of the dataset. This proves that tokenizing the text and focusing on the words used as features would be the most helpful thing in the model to convey sentiment. So for the Vectorizer tokenizing the text would help in creating a more cohesive and useful featured set.

After tokenizing the values were then vectorised using Chi Squared filtering method as per research(mentioned in critical analysis). However deciding on the K-values being used required an analysis of accuracy as the values increased. As displayed in the graph below.


K-values between 4000 and 4500 were chosen as it can be seen that the accuracy is at its highest around that value for all models. The accuracy first increases until after k = 5000 and starts to decrease around there. This was in line with the way the models work as more features are added to the models the more noise it produces. Models such as zero-r have a major impact from this due to them not having any smoothing or prediction rules. For Logistic Regression it does improve the model however the runtime becomes longer, several minutes (and depending on the size of the values) or hours, would be spent training the model. This display diminishing returns to increasing the amount of features for the models.

![unnamed4](https://user-images.githubusercontent.com/124246311/235435024-aabb7eb4-5682-411e-b658-a39a13e1e2b2.png)

Following finding suitable K-values, a test set was derived from the training set, split by 90/10, to train the model and see how accurate the model is in relation to the split data set. As can be seen (in the results section) the results indicate that all algorithms run decently. This is run several times to identify the most suitable algorithm as the split is randomised. After observing these scores, the algorithms with the best accuracy were used to further be combined in an ensemble stacking classifier in which a logistic classifier was used as the base as it was the most reliable out of all. The results of that classifier were then also tested against the train test dataset (also present in results).

Then the same prediction models were trained against the entire test dataset and were made to predict the train dataset.

## Results
========= zero-r

          	precision	recall  f1-score   support

	negative   	0.44  	0.14  	0.21   	362
 	neutral   	0.66  	0.80  	0.72  	1311
	positive   	0.49  	0.46  	0.47   	508

	accuracy                       	0.61  	2181
   macro avg   	0.53  	0.47  	0.47  	2181
weighted avg   	0.58  	0.61  	0.58  	2181
========= GNB

          	precision	recall  f1-score   support

	negative   	0.00  	0.00  	0.00   	362
 	neutral   	0.60  	1.00  	0.75  	1311
	positive   	0.00  	0.00  	0.00   	508

	accuracy                       	0.60  	2181
   macro avg   	0.20  	0.33  	0.25  	2181
weighted avg   	0.36  	0.60  	0.45  	2181
========= Decision Tree

          	precision	recall  f1-score   support

	negative   	0.00  	0.00  	0.00   	362
 	neutral   	0.62  	0.98  	0.76  	1311
	positive   	0.70  	0.14  	0.23   	508

	accuracy                       	0.62  	2181
   macro avg   	0.44  	0.37  	0.33  	2181
weighted avg   	0.53  	0.62  	0.51  	2181
========= KNeighbors

          	precision	recall  f1-score   support

	negative   	0.43  	0.03  	0.06   	362
 	neutral   	0.61  	0.94  	0.74  	1311
	positive   	0.50  	0.15  	0.23   	508

	accuracy                       	0.60  	2181
   macro avg   	0.51  	0.37  	0.34  	2181
weighted avg   	0.55  	0.60  	0.51  	2181
========= LogisticRegression

          	precision	recall  f1-score   support

	negative   	0.60  	0.28  	0.38   	362
 	neutral   	0.69  	0.87  	0.77  	1311
	positive   	0.63  	0.45  	0.53   	508

	accuracy                       	0.67  	2181
   macro avg   	0.64  	0.53  	0.56  	2181
weighted avg   	0.66  	0.67  	0.65  	2181
========= ensemble Stacking

          	precision	recall  f1-score   support

	negative   	0.48  	0.33  	0.39   	362
 	neutral   	0.67  	0.78  	0.72  	1311
	positive   	0.54  	0.45  	0.49   	508

	accuracy                       	0.62  	2181
   macro avg   	0.56  	0.52  	0.53  	2181
weighted avg   	0.61  	0.62  	0.61  	2181




## Exaplanation

The vectoriser used was TFIDF. This vectoriser compared with a bag of words did not simply count frequency of words in the document, rather than using the algorithm to take into account the relevance of the words within the document. Allowing for a more useful vector to be used and removes the worry of over repeating words showing relevance to the machine learning algorithm when it does not.

The hypothesis that Chi Squared would be a superior filtering method in comparison to mutual information was confirmed as research was conducted as to what method would be more suitable. In a paper evaluating mutual information and chi square performance at Universidad Tecnológica de la Habana Jose Antonio Echeverria, Havana, Cuba, showed that their, “experiments show that concerning feature selection Chi Squared overcomes mutual information in all the tests we performed” (Párraga-Valle et al., 2020) This was further supported by previous practicals showing in feature selection mutual information was inferior as to supplying relevant features to the machine learning algorithm.

For the features, identifying the data within the text that will skew the prediction model was necessary to get a better result. When analysing this as mentioned in the method we noticed that certain features that were hypothesised to be useful in predicting sentiment were not and had to be removed, the only conclusion that can be drawn from this analysis is that the words used within tweets are the best predictors for sentiment.

A simple method was chosen as the baseline in which we could compare metrics between other models easier. At first, the 0R was the prefered, as there is no predictor present it makes it more useful to compare and decide which models seem to be more viable than others, and 1R and above have a rule at least to predict the model and do not always give neutral values.However, Naive Bayes was also a viable option and was implemented as to compare with 0R. The 0R started to have problems in comparison to Naive Bayes as it did not possess the smoothing properties within the algorithm of Naive Bayes.

For more advanced models, the ordinal logistic regression is chosen as it was a good binary classification model, which learned the difference between the three sentiment labels. It assumes that the dependent variable is categorical in an ordered way while the words in the tweets are correlated with each other. (Logistic Regression in Machine Learning - Javatpoint, 2021) It was also one of the most reliable models as it linearly describes the relationship between the independent and dependent variable, and is fast at doing so. 

The Support Vector Machine (SVM for short) has been considered; however, due to its long running time, it is not viable as a model for this case. It is a hyperplane-based classifier, which would place a plane in between sets of data to differentiate the train data. Then it would make a prediction based on where the data is compared with the plane. But in spite of that, the run time would be O(n^3) with N being the words (Abdiansah and Wardoyo, 2015), hence it would take a very long run time to run the data set as there are many tweets. Moreover, with the amount of words as features, the SVM would underperform as there are too many features to consider (Support vector machine in Machine Learning, 2020)

Decision Tree classifier was seen as a good choice as it is a fast algorithm which is highly regarded as a basic learning algorithm. Though it is known to overfit on occasion, it is still suitable for this task as it is very suitable for nominal attributes thus on a text analysis machine learning program it would fit perfectly. The way in which we decide to counteract this algorithm's propensity to be utilising irrelevant features is to lower the amount of irrelevant features being put through the algorithm thus allowing it to have a higher accuracy. This was already achieved through the use of a TFIDF vectorizer, Chi Squared filter method and tokenizing the words as mentioned in the method above. Further research had suggested that “the proposed system enhanced the accuracy in sentiment analysis up to 6-20% related to the existing systems.” (Kasthuri & NishaJebaseeli, 2020) 

K Nearest Neighbour is a model that did well for this dataset as it is a simple model that is able to handle extra data that is added later if required, hence being able to handle larger data sets, which is crucial for this project. This particular model and dataset tend to do better with a K-values of 4000 to 4500. With a different K-value, the model is also unfortunately more prone to noise if the data set is dense and would result in having high dimensional results, which could provide in an over analysis when there are only three sentiment labels to be classified between. In addition, another weakness can include the fact that a K-value has to be assigned to the data, where to increase or decrease the accuracy, the code containing the value must be changed and tweaked. Nevertheless, there was no need to weigh nor invert the linear distance between the dataset and its neighbours as it.

For the esembler model; Bagging, Cross Evaluation and Stacking were considered. Bagging mainly was used in conjunction with a Decision Tree, which was no longer considered. Though implementation of a Decision Tree was present, the model wasn’t as reliable ,especially in comparison to logistic regression. As there 4 models, 3 of which were reliable, that were already made, the best choice would be to user stacking and use a base of logistic regression as it was the most efficient.

There were a few things that could have been improved for the algorithm. One obvious problem was in the training dataset. As seen in the graph, the number of neutral tweets were far more present than the number of positive or negative sentiment tweets. This heavily impacts datas accuracy and could have been resolved in a number of ways. Two easy resolutions were to either lower the number of neutral tweets or create synthetic positive and negative tweets to make up for the low volume of those categories. This can be seen negatively impacting the results through the f1 scores of certain naive algorithms such as the Naive Bayes and R0.

![unnamed5](https://user-images.githubusercontent.com/124246311/235435339-09a5b0d0-af1f-43ad-b8ba-78ede98c5a4b.png)

## Conclusion

Using Naive Bayes as the baseline model, it can be concluded that using the logistic regression would be the most appropriate model to setimnets with this dataset of tweets, using a TFIDF vectoriser and Chi Squared classifier with K-values between 4000 to 4500. The other models were alright however each had their pitfalls. The biggest drawback to these models and algorithms is the lack of implementation to equalise the among of setments within the 3 categories within the dataset.


###Bibliography

Abdiansah, A., & Wardoyo, R. (2015). Time Complexity Analysis of Support Vector 
	Machines (SVM) in LibSVM. International Journal Of Computer Applications, 
	128(3), 28-34. https://doi.org/10.5120/ijca2015906480

Arrington, M. (2006). Odeo Releases Twttr. Techcrunch.com. Retrieved 13 May 2022, 
	from https://techcrunch.com/2006/07/15/is-twttr-interesting/.

Kanalley, C. (2013). Why Twitter Verifies Users: The History Behind the Blue 
	Checkmark. HuffPost. Retrieved 13 May 2022, from 
	https://www.huffpost.com/entry/twitter-verified-accounts_b_2863282.

Kasthuri, S., & NishaJebaseeli, A. (2020). An efficient Decision Tree Algorithm for 
	analyzing the Twitter Sentiment Analysis. Journal Of Critical Reviews, 7(4), 
	1010-1018. Retrieved 13 May 2022, from 
	http://www.jcreview.com/admin/Uploads/Files/61adfe235a6725.71379760.pdf.

Logistic Regression in Machine Learning - Javatpoint. Java T Point. (2021). Retrieved 13 
	May 2022, from 
  https://www.javatpoint.com/logistic-regression-in-machine-learning#:~:text=Logistic%20regression%20is%20one%20of,of%20a%20categorical%20dependent%20variable.

Párraga-Valle, J., García-Bermúdez, R., Rojas, F., Torres-Morán, C., & Simón-Cuevas, 
	A. (2020). Evaluating Mutual Information and Chi-Square Metrics in Text 
	Features Selection Process: A Study Case Applied to the Text Classification in 
	PubMed. Bioinformatics And Biomedical Engineering, 636-646. 
	https://doi.org/10.1007/978-3-030-45385-5_57

Rayome, A. (2019). 10 most-downloaded apps of the 2010s: Facebook was the 
	most-downloaded app of the decade. CNET. Retrieved 13 May 2022, from 
  https://www.cnet.com/tech/mobile/10-most-downloaded-apps-of-the-decade-facebook-dominated-2010-2019/.

Rosenthal, S. (2016). Developing a successful SemEval task in sentiment analysis of 
	Twitter and other social media texts. Language Resources And Evaluation, 50(1), 
	502-518. https://doi.org/10.1007/s10579-015-9328-1

Support vector machine in Machine Learning. GeeksforGeeks. (2020). Retrieved 13 May 
	2022, from 
	https://www.geeksforgeeks.org/support-vector-machine-in-machine-learning/.

