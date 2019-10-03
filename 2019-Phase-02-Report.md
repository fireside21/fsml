
# Fireside Machine Learning

*Phase II, Bryan Wilkinson (Miner & Kasch), Scott Sadlo, Isabella Seeman*

# Intro

Note: This document has been edited to remove all identifiers to the underlying constituent client data used to produce the report

# 1. Platform Architecture

The work described in this report was performed on AWS infrastructure, in the standard cloud, although this can be moved into GovCloud with little issue. To protect the security of the instances, the Bastion pattern was used, shown in the figure below.

In this set up, a hardened bastion instance is allowed to communicate with the internet, but is limited to specified IPs that can access it. The instances where the work is actually completed are in a private subnet with the bastion, and are only accessible from the bastion. Furthermore, only limited communication is allowed to leave the internal instances, on specified ports.

To further secure the instances, amazon machine images produced by the Center for Internet Security were used, which follow the benchmarks set up by CIS. These set strong defaults for the firewall, allowed programs, and other common issues with security. These instances were then modified in a minimum way to allow software like Jupyter through the firewall. 

Because of the bastion set-up, Git use is not as trivial as on a normal installation. Since the bastion server is the only system to access the outside word, any repositories must be pulled from the outside host to the bastion server. Then, on the internal machines, the repository on the bastion server is used as the origin repository. To get data to an external Git repository, it must first be pushed to the bastion repo, and then to the external repo.

Some of the models for this engagement require GPU machines to train them efficiently. We selected p2.8xlarge, but in practice any GPU instance can be used, as long as the time for training is acceptable. The GPU instances are set up similarly to other instances, but require NVIDIA drivers to be installed. Instructions for installing the NVIDIA drivers can be found at https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html. Note that they should not be installed from the system package manager.

For managing python packages, conda was used. The needed packages can be found in fsml/environment.yaml

# 2. Data

Data for this engagement was exported from Fireside internal databases to delimited text files. This data was uploaded directly to S3, from where it was downloaded onto the internal machines behind the bastion server. 

The data covers 4 districts, 3 of which are represented by Republicans, and one which is represented by a Democrat. The oldest data goes back to 1997, but was not used for most work. For this engagement we focused on email communication only. Future work could allow this to be expanded to letters and other forms of communication. For letters in particular, it may be necessary to run them through OCR again, as OCR performance has greatly improved in the past few years. 

The following data cleaning methods were used on the data before performing any of the processes described below. 

1. Remove all rows with no body and no subject and no notes
2. Remove rows whose body and subject are equal to the string "no data"
3. If the InboundType is letter, remove any rows with no body and notes less than 500 characters, as this tends to indicate a summary of the letter rather than the contents of the letter
4. If the InboundType is phone call, only keep it if it is after 2000
5. Remove phone numbers when they are the last line of the body
6. Remove the string "=== Organization Statement ===" when it is the first line in the body
7. Remove strings matching “Date Received: \d\d?/\d\d?/\d\d \d\d?:\d\d:\d\d [AP]M\n*”
8. Remove strings matching “\nRSP: Yes.\n\nTopic/Subject Desc: .*\n"
9. Remove strings at the end of the body matching "------ additional security info --------\n.*$"
10. Replace “Incoming Mail” in the subject field with the empty string
11. Remove any addresses in the message that match the address in the database of the user who sent the mail in
12. Remove any names in the message that match the name in the database of the user who sent the mail in

In addition, emails from the same user, with the same body on the same date were considered duplicates and were removed prior to further processing. The subject was not considered because it was found that a small group of users send multiple messages on the same day with the same content but different subject lines. 

# 3. Overview of Areas Covered

Guided by the SOW, we investigated extracting many different types of information from email messages. These included the detection of keywords, topics, entities, and bills, as well as the labeling of sentiment and document types. In addition we looked at representing constituent messages as vectors, for use in either clustering or similarity calculations.

All code to perform these tasks was written in Python 3. The list of files and short descriptions of them can be found in Appendix A. 

The rest of this document describes the methods and results for working on each of these problems. In addition, future work is suggested, including when human labeling of data would be helpful.

# 5. Keyword Extraction

To extract keywords from messages, we use the Term Frequency - Inverse Document Frequency (TF-IDF) scoring method. This method is capable of identifying terms that are common in a given document, but at the same time discounts terms that are frequently encountered in the whole collection of documents. We use this measure to estimate the importance of words within a document compared to expectation of finding a given word a given document.

The document frequencies of each word are calculated over a set of documents once, and then held constant until the model is updated again. 

Prior to calculating document and term frequencies, several types of words were removed. Words were removed if they were stop word, punctuation, spaces, URLS, emails, or numbers.

Words were tokenized and stripped to their base form (lemmatization) using the spacy python package.

Following tokenization and filtering, any word that occurred in less than 5 documents or more than 50% of documents was classified as ineligible to be a keyword, and therefore discarded. 

Following this process, a TFIDF model is trained using the implementation in the gensim library. The model can be persisted and new messages can have their keywords extracted using the statistics stored by the model. The model can also be updated with new data, to calibrate the document frequencies.

The keyword extraction algorithm and code to use a saved models is available in fsml/KeywordExtractor.py

## 4.1 Results

The results of keyword extraction trained and run on office with SiteID [Office 2] are shown below.
Body keywords

This same model was run on the office with SiteID of [Office 1], the results of which are shown below:
Body keywords

It does not appear that training on one office and running on another has any major drawbacks, but training on multiple offices will produce the most robust model.

## 4.2 Considerations for Operationalization

Manual review of extracted keywords by Fireside personal suggests that the keyword extraction model produces relevant keywords for a given communication item. As such, the model is a prime candidate for production implementation. Either a lambda function or API would be implemented to run on messages as they come into the system to generate keywords in real time. The optimal amount of data to train is still to be determined, but starting out with the last few years of emails from several offices would be a reasonable starting point, primarily to ensure the largest possible vocabulary.

As part of operationalization, there are two major situations to monitor as indications for when model retraining is necessary:

* The first is that a word that is not in the models vocabulary, either because it was not seen at training time or was not frequent enough. To monitor this situation, we suggest implementing functionality to count the number of times each word that is not in the vocabulary is seen, either across all offices or in a single office, and once a threshold is hit, retrain the model by updating on all recent messages.

* The second situation is when a word becomes so common that offering it as a keyword is no longer useful. With a large enough training set, most words relative document frequency will remain stable. When a words’ document frequency starts to change quickly, it is a good indicator that the model needs to be updated.

Retraining and immediately deploying a new model may have potential adverse effects that need to be considered. For example, as opposed to large systems with thousands of users, it is likely that the same users will be accessing the data daily, and may notice if new messages for a campaign come in and have different keywords, or if the keywords associated with a campaign seem to be different. Another issue is that after retraining, keywords may disappear from the system entirely, causing confusion.

One potential solution is to include logic to not allow keywords to change on campaigns, or to ensure keywords are not deleted if they have been presented in the past few days. Solutions to this would most likely require user-based testing to find what makes the users most comfortable.

## 4.3 Future Work

In addition to operationalization efforts for deployment and continuous retraining, there are a few other areas for improvement. These include:

* The stop words removed are based on a generic corpus, and could be improved based on common, yet still rare enough to avoid previous filtering, words, such as “115th”, etc.

* Introducing keyphrases into keywords will make them more useful in many situations. Because longer phrases are more rare, they cannot be compared directly to single words in many cases. Additional development would be needed to determine the best methodology to surface both the relevant keywords and keyphrases.

# 5. Topic Modeling

Topic modeling is an unsupervised technique that aims to find the salient topics in a group of documents. Topic modeling also learns a topic mixture for each document, that indicate the proportion of each topic represented in a document. In methods such as LDA, the number of topics must be specified a priori. Because this requires a lot of human intervention to determine the number of topics, it is not sustainable for application in a domain like constituent messages, where new topics may emerge and current topics may go dormant.

We instead investigated two methods of topic modeling that not only model the documents and topics, but also the number of topics as well. This ensures that the number of topics does not need to be specified apriori, providing greater system flexibility.

* The first method is an extension of the traditional LDA model, known as Hierarchical Dericlect Process. This method takes the standard LDA model and adds an additional dirichlet parameter to determine the number of topics to induce.

* The second model, TopSBM, is a newer modeling technique and stands counter to statistical modeling. The model represents documents and words as nodes in a graph with edges between them. The model uses a graph processing technique called Stochastic Block Modeling to find communities of nodes in the graphs. Under these conditions, identified communities in the graph are synonymous with topics.

TopSBM implements a specific case of the Stochastic Block Model, the Hierarchical Stochastic Block Model, allowing topics to be discovered at varying depths. The number of levels is chosen as part of the model, with lower levels producing smaller groups. This allows topSBM to have a finer resolution than is seen with traditional topic modeling approaches, which struggle when the number of topics approaches the square root of the number of words in a corpus. 

In practice we have found that the lower, finer levels provide more information than the coarser upper levels, but it may still be useful to have the ability to explore them. Furthermore, the upper levels of the model tend to be less stable between runs.

Both HDP and TopSBM are algorithms to recover partitions based on an assumed generative process. Some of the foundations of these two algorithms are mathematically equivalent. The major advantage TopSBM has is that it does not assume that the data comes from a Dirichlet distribution, allowing a wider variety of distributions to be recovered. For a graphical representation of this, see the image below, taken from the paper “A network approach to topic models” (Gerlach, et al. 2018)

Before using either of these methods, the documents were cleaned in a similar manner to as was done in keyword extraction. As a reminder, words were removed if they were stop word, punctuation,spaces, URLS, emails, or numbers. Words were tokenized and stripped to their base form (lemmatization) using the spacy python package. In addition to removing these words, a manually edited list of the most common words was removed. This list should be reviewed and amended prior to future work. The list can be viewed in appendix B.

The topics that emerge from both of these methods are not labels such as “Immigration” but a list of words, like “border, immigration, wall, asylum” that require a human for interpretation.

HDP was used as implemented in the gensim library. fsml/hdp_trial.py contains an example of how to train an HDP model and save it for future use. To train a topSBM model, the python script fsml/train_topSBM.py can be run. This script accepts command line arguments which can be seen by using the --help flag. 

In addition, the python class TopSBM.py provides easy interface to a trained model, including assigning topics to unseen messages. For an example of how to use this class, see Jupyter notebook fsml/example_notebooks/Topic Modeling.

## 5.1 Results

Evaluating topic models can be a difficult process requiring human interpretation of the induced topics. Fireside’s ML Staff is working with experts to evaluate their impressions of the different topic models. Example outputs for each of the two methods are shown below:

HDP finds 58 topics in the messages from January 1, 2019 to May 30th, 2019 in the office with ID [Office 1]. 

Subject Message Body Most Probable Topic

The results from using topSBM on the same messages, looking at the lowest level, are below.
Subject Message Body Most Probable Topic

As shown above, topSBM produces finer grained topics with an apparent stronger semantic cohesion. For this reason, the majority of our work focuses on topSBM. 

Note that there are a few messages above that have the topic “[worth, trouble, sales, valid, super, confident, anger, load, mere, spell]”. This seems to be a catch-all topic that gets bubbled up to the surface when the messages contain rare words. In message 27101044, the reason is most likely the misspellings of TARRIFF and illigals. In message 27101049, the most likely reason are rare words like “unamerican”, “ laissez” and  “faire”. A possible solution to this issue is to filter rare and overly common words. 

The table below illustrates examples of topics at higher levels of the hierarchy. We use the same messages as in the table above to facilitate easier comparison:

Subject Message Body Most Probable Topic

The topSBM model found 4 level, having 582 topics at the lowest level, 159 topics at level 2, 23 topics at level 3, and 3 topics at level 4. 

In our experiments, we measure and assess topic model stability. Topic model stability measures how stable the topic assignments are for each document between differing runs of the topic model. Because topics are comprised of words, a best effort matching must be done between the topics before the stability can be calculated. This best effort matching is done by determining the number of words the topics have in common, and then aligning the topics so that the maximum number of words between all topics can be matched.

The stability of documents was evaluated:

* across two time periods

* using 3 different lengths of training day (30,60,90 days), and

* simulating model runs every 1,7 and 30 days.

In addition, the configuration options of including keyphrases, only using one message per campaign, and combinations of them were tried with all combinations of the different time periods. We can first look at the stability averaged over the offices to identify some trends in training length, step size, key phrase use, and if campaigns were collapsed to one message or not.

While there are several trends shown here, the most immediate is that using key phrases and removing campaign duplicates causes the worst performance. The most likely reason for this is that the repeated use of words in campaigns helps the model group topics together run after run, while using keyphrases adds several rare occurrences to the model, which causes them to be grouped in different topics each run. 

For the rest of this analysis, we will focus on the configuration of not using key phrases and including all data from campaigns. When we look at this data, we see that all offices all have about the same stability, although the standard deviation is a bit different between offices.

Given this information, and the fact that training over both 30,90, and 60 days leads to good stability when running every 1 or 7 days, the most efficient training schedule would be to use 30 days of data, running every 7 days. This will use the least memory and least compute time. Unlike other work in this report, topic modeling works best when done on a per-office basis. This is not necessarily for performance reasons, but because the algorithm is extremely memory intensive and cannot handle multiple offices at once. 

## 5.2 Considerations for Operationalization

Topic modeling is an excellent way to explore the message datasets, but does not produce outputs that can be immediately given to end-users in a useful format. Operationalization of topic modeling should focus on functionality that is most useful for developers and non-end-users. 

One consideration for topic modeling is that model training can take up to 1 hour depending on data volume and available computational power. Under current assumption, we suggest at most daily training runs. This amount of time needed to learn a topic model is proportional to the number of messages and will increase as more offices are added. Performance speedups are possible in the script to train the model, train_topSBM, to directly access the data from the database rather than export files. 

Because of topic modeling stochastic nature, it is best to retrain the models on a set schedule. Because new words are frequently being added to the vocabulary of messages, it is quite possible for up to 10% of messages to be unassignable to a topic just days after training. This might be alleviated with a large enough training set, but it is computationally infeasible to be run on a frequent basis without an extremely high-memory machine. 

## 5.3 Future Work

Future work on topic modeling should aim to make the results more repeatable. Because of the graph-based nature of TopSBM, further edges could be added to the graph to promote this. Work like this is untested in topic modeling, but has been done in general community detection. Possible links to be added could be documents from the same author, or between words that are listed as synonyms in a thesaurus. 

In addition, the inference algorithms traditionally used in topSBM is quite simple, counting the number of words in each document in a topic, and then using these counts to calculate the most common topic in the document. This fails to consider the importance of a word in a topic, allowing many words from a single topic that are very weak in that topic to dominate. Potential solutions include rather than weighting all words the same, use their importance in the topic or other statistics to weigh them before determining the most probable topic for a document. Alternatively, TF-IDF could be used to weight the links prior to topic modeling, which may achieve a similar effect. 

A third area of future work does not involve topic modeling itself, but rather using the output of topic modeling as a guide to derive the set of labels for use in a supervised classification setting. This work would focus primarily on interface and process design, but there would need to be some work on ensuring that new labels are not created excessively, and that old labels were almost never deleted.

Additional exploratory work should be performed on determining which words to filter out from ever being considered part of a topic, which would be a pre-processing step prior to topic modeling.

# 6. Document Clustering

One of the uses of topic modeling is to group similar messages together. Rather than use an intermediate step, we can directly cluster documents based on their similarity. The downside of this is that it does not expose any information about how the documents are related.

To cluster documents, a vector representation of them is needed. To produce this vector representation we use a Universal Sentence Encoder. The Universal Sentence Encoder is a transformer neural network explicitly trained to map similar sequences of text to similar vectors. This network was downloaded pre-trained from Google (implemented using TensorFlow) and was not trained specifically for these messages. Despite the name, Universal Sentence Encoder was designed to be used on any sequence of words. We therefore feed the entire document to the encoder. This returns a 512-dimension real-valued vector for each document.

After encoding each document and calculating the cosine similarity between the vectors, we cluster the documents using Density Based Clustering (DBSCAN). DBSCASN was chosen because it has the ability to cluster documents not based on a preset desired number of clusters, but based on the density of discovered clusters. We set the minimum number of documents to constitute a cluster to be 2, and the maximum distance between docs in a cluster to be 0.15. This distance is something that should be experimented with and possibly further refined.
Another benefit of DBSCAN is that not all documents must belong to clusters. Those with no other documents in the immediate area (determined by the distance discussed above) are said to be singletons and are not assigned any cluster

This work can be found in the Jupyter notebook fsml/dev_notebooks/USE.

## 6.1 Results

The results of applying clustering to documents from office with id [Office 2] can be seen below. 

This produces 158 clusters, with an average size of 11.67 documents per cluster. 40% of the documents are labeled as singletons.

Example messages from different clusters are shown in the table below. Note that cluster 2 is omitted as it identifies messages containing all non-printable characters, while cluster 4 is omitted for containing only requests for flags flown over the Capitol.

Cluster Subject Body

Messages are grouped on topic, but are also being grouped on the general length of the message. This is due to what the authors of USE refer to as “dilution” of the representation in longer messages. Shorter messages allow the meaning of individual words to matter more. In general matching documents of differing lengths to each other is a difficult problem, and some post-processing would most likely be needed to avoid this issue, so that cluster 6 and 5 would end up together. 

As an example of some messages labeled as singletons, see the table below

Subject Body

As an investigation into why these particular messages were identified as singleton’s, the table below shows the document closest to these documents, along with the closest documents assigned cluster.

Body Nearest Neighbor Nearest Cluster Distance

Some of these clusters appear to be true singletons, but others seem like they should be grouped together, for example those mentioning the shutdown. Fine-tuning the embedding model (discussed below), may improve results, but some of the issue may be due to the length of the documents, as the vectors for each word are added together to form the embedding for the document, thereby allowing each word to contribute less in longer documents. 

## 6.2 Considerations for Operationalization

This exploratory work has focused on processing messages for clustering in batch mode. For operationalization, a decision would need to be made to either encode messages on the fly as they come in, or in a batch every so often. While a neural network is used for encoding the messages, it is lightweight enough to run on a CPU machine, especially in small batches. To save on computation time, each document should only be encoded one time. A suitable storage mechanism would be needed to store the 512-element vectors so that they could be quickly retrieved into a matrix form for similarity calculation. The similarity calculation itself is fast, taking only 92.8 ms for 3089 messages

On a pre-trained encoder, there should be no general reason for retraining. Future work based on customizing the encoder may call for retraining, but this would be established after development.

Messages are encoded without reference to other messages, but clustering relies on the similarities between messages, so this should only be run between messages in a single congressional office.

## 6.3 Future Work

An advantage to USE is that each article can be embedded one time, and saved off for future comparisons. It achieves near state of the art performance on semantic similarity tasks, but more recent methods that run two documents through a neural network and directly computes their similarity score perform better. The disadvantage of these new methods is that for each new message that comes in, it must be run through the network paired with each existing method. There also exists a comparable sentence embedding method, InferSent, which should be compared to the USE results.

A potential compromise solution would be to use the USE embeddings to produce large coarse general clusters, and then use the more expensive methods to compare the messages in that cluster, producing smaller sub clusters.

Another possible avenue of work would be to either finetune the USE embeddings based on pairs of messages, or train it from scratch. Fine-tuning will most likely be the more efficient method, as to achieve the current performance over 570,000 sentences were used in training. There are several ways to fine-tune the model, but one method would be to present two messages, and ask a human annotator to either mark them as related or not in a binary fashion, or on a scale from 1-5. 

Whether using USE or another embedding method, having vector representations allows for quick similarity comparisons, allowing not only clustering, but also functionality such as “see similar messages.” This will require some creative filtering, as it is not helpful to present only campaign messages. Vector representations can also be used to derive images of the embedding space to allow for either interactive exploration, or simply for a summary view. 

# 7. Entity Extraction

Entity extraction identifies and labels noun phrases as certain categories, such as PERSON, PLACE, ARTWORK, etc. The entity extraction algorithm we use is implemented as part of the spacy library. It was trained on the ONTONOTES corpus, and therefore contains the following classes:

* PERSON People, including fictional 

* NORP Nationalities or religious or political groups 

* FACILITY Buildings, airports, highways, bridges, etc.

* ORGANIZATION Companies, agencies, institutions, etc.
 
* GPE Countries, cities, states 

* LOCATION Non-GPE locations, mountain ranges, bodies of water 

* PRODUCT Vehicles, weapons, foods, etc. (Not services) 

* EVENT Named hurricanes, battles, wars, sports events, etc. 

* WORK OF ART Titles of books, songs, etc. 

* LAW Named documents made into laws

* LANGUAGE Any named language 

* DATE Absolute or relative dates or periods 

* TIME Times smaller than a day

* PERCENT Percentage (including “%”) 

* MONEY Monetary values, including unit 

* QUANTITY Measurements, as of weight or distance 

* ORDINAL “first”, “second” 

* CARDINAL Numerals that do not fall under another type

These models can be fine-tuned to either add new entity types, or to further train any of the existing categories. 

As a prototype, we used the spacy library to perform entity extraction, and then retrained the same model on those messages. We also manually edited 343 messages with entities identified as LAWs. This is quite a laborious process that could be improved with a GUI. 

The entity tagger can be fine-tuned by calling the python script ner_training.py. The code to generate training examples can be found in fsml/get_ner_training_examples.py.

In addition to entity extraction, we made a prototype entity linking engine, so that if the entities that were extracted were “US” and “United States”, they will both be standardized to “United States”. This simple prototype is based on finding all links in Wikipedia and counting the number of times each text is linked to a certain page, for example, “US” is linked to United States 6912

Times, compared to being linked to American English 44 times. The most common link for each text is considered is canonical form. This has many flaws, and steps to fix this will be discussed below.

## 7.1 Results

The entity tagger output is to be evaluated by the Fireside team, in terms of its accuracy and usefulness. For comparison, below are the same sentences annotated using the original entity tagger, the bootstrapped entity tagger, and the one trained on a small amount of manually corrected data.

LAW PERSON Body

Two major issues are apparent in the original entity tagger. The first is that the model is trained to find laws starting with the word “the”, which incorrectly cuts off part of the bill title in “Fix the Immigration Loopholes Act”. As a general rule, it would be preferable if the word “the” was not tagged as part of the title, as it is not used in official Congressional data.

The second issue is that many bill numbers are often tagged as PERSON. This could be fixed either by a rule based method, or by fine-tuning the model to not do this by giving examples of messages containing HR numbers, with no labels.

By taking this exact output and using it to retrain the model, we can allow the model to pick up domain specific vocabulary. The same messages run through this model are shown below.

LAW PERSON Body

By just using the bootstrapped labels, the PERSON detection remains almost the same, while the LAW detection degrades greatly. After manually editing some of the same data that was used to train the model in the bootstrapped version, the following is the result.

LAW PERSON Body

The model has now learned to not include the word “the”, but still struggles to detect complete act names, now seeming to prefer just a word or two before the word “Act”. It is quite likely that more training data will improve performance.

Another way to view the results is the look at all the entities found. The built in tagger finds 178 distinct entities tagged as LAW, while the bootstrapped tagger finds 358, and the manually edited tagger find 353.

The table below shows some entities found only in each tagger

Only found by original model
Only found by bootstrapped model
Only found by manually edited model
child and dependent care tax credit,
woof act,
lift america act,
equity act,
corridors act
outdoors for all act,
'telephone usage charge tax\nutility tax\nvehicle license registration tax,
each woman) act,
honey hill lane,
coastal protection act]
holderrieth rd & boudreaux,
pfas action act,
better way,
american health care act,
rural television content protection act

As each run is picking up new information, it is likely that better training data will converge to the best possible list. It should also be noted that the manually edited data only provided corrected LAW entities, and left all other entity types as is. Correcting all entities types in the training data will lead to a more robust tagger over all.

## 7.2 Considerations for Operalization

The basic entity tagger itself is ready to be used, but acceptance testing should be done for each type of entity. Most likely some types, such as person, will be ready to be presented to the end user, while others will require additional training. The processing for a single message is quick, and could either be performed online, or in daily batches. Because entity tagging is fairly easy to tag, a simple feedback button in the interface to indicate if an entity is wrong will allow for further training. 

Once the entity tagger reaches acceptable performance, the need to retrain should be minimal. Because the way entities are used should not change too much, it should maintain its prediction ability. What is more likely to happen is that a new version of the tagger from Spacy will be released, which could then be fine-tuned against using the best data, and compared with the old performance to see if an update is warranted.

The other situation where retraining would be warranted is in an active learning scenario. This could be done on a regular schedule, or when the number of erroneous entities labeled reaches a certain number. In either case, a test set should be developed to make sure the performance on the messages does not degrade after training on the user submitted errors.

## 7.3 Future Work

The performance of the entity tagger would be improved not through model architecture improvements, but through better training data, which would be achieved through hand labeling of instances. The tagger should be trained across all offices to provide greater variation in training data.There are many available tools to help promote this, such as Prodigy or BRAT, both shown below, which allow quick graphical selection of instance types. 

BRAT (open source)

There is a large amount of work to be done for the entity linking. The next step would be to integrate context of entities, so that not only the names must match, but the entities used around it are found on the Wikipedia page as well. This improvement should fix most cases where the entity in question has a reasonably detailed Wikipedia page. The primary challenge would be how to store this information in a quickly accessible format, and how long it might take to parse all occurrences of entities in Wikipedia. This information would need to be periodically updated, to take advantage of entities that did not previously have Wikipedia pages.

# 8. Bill Extraction

A follow-on to entity extraction is bill extraction from messages. Bills are represented in messages either by their number or a name. For bills represented by a number, a regular expression is used to extract the number. For bills mentioned by name, first entity extraction is done to identify all noun phrases tagged as legal documents. The entities identified are then matched to existing bills in the session corresponding to when the message was sent. The text extracted must match over 50% of the full bill title, with exact matches preferred. Once the bill is matched, it is standardized to its bill number

The bill extraction can be done by calling the method extract_bill_numbers_and_policy_areas on an object of the class OfficeData

## 8.1 Results

In total, 10022 messages of 28921 are identified as having a bill, with 3637 messages containing only a number, and 1595 messages containing only a bill title. The results of running the bill extraction are shown below. 

Body bill_numbers bill_names

## 8.2 Considerations for Operalization

The bill extraction mechanism itself is production ready, but the entity extraction mechanism it relies upon needs improvement. In its current form it expects data from an export file. It would need to be updated to work on the database. Like many other processes, it could either be run as messages come in, or once daily, depending on need. This process does not depend on being from any particular office.

## 8.3 Future Work

The success of bill extraction when it comes to names is highly dependent on the entity tagger. Improvements on the entity tagger are likely to improve the bill extraction mechanism.

Other improvements can be done on how bill names are mapped to bill numbers. It is common at the beginning of sessions to refer to bill from the last Congress, which may not have been introduced yet in this congress, and so potentially should be mapped to older bills. In addition, if a form letter is not updated, an old HR number can appear. The HR numbers could be verified by using a classifier to look at either the text of the bill or articles about it, and the text of the message to ensure that the right bill is chosen. 

# 9. Policy Area Prediction

There are many possible ways to classify messages, but to save on annotation cost and use a consistent label set across offices, we use the policy of terms as defined by the Congressional Research Service. This is a set of 32 labels that are applied to every piece of legislation introduced in Congress. Each legislation is only given one label, making it perfect for use as a classification label.

To label a message with a policy area, we considered only messages containing bill numbers for the training set. To ensure that the message was talking about mostly one thing, we only considered messages that mention a single specific bill, although it can be mentioned multiple times. The training was done in two forms:

-   once using all messages from the 115th Congress as a training set, and
    
-   using random segmentation by bill and policy area to construct a training and test set from messages in the 116th Congress.

The classifier is a neural network classifier built on top of the pretrained BERT architecture, the state of the art transformer-based classifier. The implementation used for this work is found in the python library pytorch-pretrained-bert, although the newest version released has changed the name to pytorch-transformers. This library, as the name suggests, is written using pyTorch. Because of memory limitations, we use the smaller BERT model, which contains 12 transformer layers. Additionally, we experiment with differing lengths of input, providing at most the first 256 words to the classifier from each document, again for memory performance reasons. The BERT classifier can take at most 512 words from each document, if given enough memory.

The two generation scripts are get_training_examples_for_classification.py and get_training_examples_for_classification_115th.py. To train a classifier, run the script train_policy_classifier.py. This script has command line options, which can be viewed by calling it with the --help flag. The file policy_prediction.py contains a helper method,

predict_policy_area which will read in a trained model and use it to make predictions on a DataFrame of messages.

Training was done on an AWS p2.8xlarge, which has 8 GPUs on it. To take advantage of the GPUs, cuda must be installed. See section 1for instructions on how to install cuda. Inference can be done on a standard machine without a GPU.

## 9.1 Results

The first models were trained on the offices with IDs [Office 1] and [Office 2] on messages from 1/1/2019 to 6/5/2019. For each policy area, some bills were chosen for the test set, while others were chosen for the training set. This was done psuedo-randomly to get as close to a 75/25 split of training and test data as possible. Not every topic was represented in this sample, and those that only had one message were not used. Some examples from the test set are shown below, along with their correct label and the predicted label.

Subject Body Actual Prediction

At least one of the errors above, the message with 27080761, refers to HR 21, which is an appropriations bill, so it is quite possible that the constituent wrote in using the wrong bill number. Further improvements to the bill extraction mechanism as described above will alleviate this issue. 

The primary purpose of this experiment was to determine if partisan bias affected the performance. It is not completely conclusive because of the small amount of data, but the table below indicates that the party affiliation of the training and test sets does not negatively impact the model. We tested training the model training for 3, 5, and 10 iterations, using input lengths of 64, 128, and 256 unites. When averaged over these configurations, the precision, recall, and f1 values for training and testing on different offices is shown below. 

training_set
test_set
f1
precision
recall
[Office 2]
[Office 2]
0.414625
0.427912
0.464947
[Office 1]
[Office 2]
0.459082
0.467943
0.504630
all
[Office 2]
0.471277
0.476450
0.520503
[Office 2]
[Office 1]
0.502808
0.542996
0.530864
[Office 1]
[Office 1]
0.595129
0.606710
0.629630
all
[Office 1]
0.601365
0.603339
0.637125

As shown above, while [Office 2] performs worse than [Office 1] irregardless of the training set, the best results for both test sets is when training on both offices. 

The second experiment was to achieve better accuracy using more data. For this we used all the messages sent during the 115th Congress, and tested on the 116th congress. The precision, recall, and F1 scores are shown below.

training_set
test_set
f1
precision
recall
[Office 1]
[Office 1]
0.500381
0.515459
0.511978
[Office 3]
0.506748
0.518719
0.519165
[Office 4]
0.513561
0.521344
0.531017
[Office 3]
[Office 1]
0.555789
0.577347
0.573927
[Office 4]
0.562905
0.577053
0.583414
[Office 3]
0.563572
0.579664
0.582476
[Office 4]
[Office 1]
0.713452
0.745296
0.710464
[Office 3]
0.724366
0.752063
0.719687
[Office 4]
0.741384
0.761038
0.742792
all
[Office 1]
0.780165
0.803165
0.775910
[Office 3]
0.784950
0.805036
0.780971
[Office 4]
0.796548
0.812431
0.796433
	
With enough data, training on one Congress and testing on another increases the scores greatly, from a max F1 of .6 to .79, almost a 20-point swing. Once again combining multiple offices performs the best. 

Here are some example messages along with their predicted and correct categories.

Subject Body Expected Predicted

To evaluate this on messages that don’t contain bills, we used the labels given by some of the offices that overlap with the CRS Policy areas. Some of this data contained clear mistakes, so we used our best judgment and removed those that did not make sense.

Some examples of the messages and their office assigned labels that were removed are below.

Subject Body Categories

After removing these, models perform as follows:

F1
Precision
Recall
Office 1
0.451173
0.538845
0.441400
Office 2
0.376133
0.543804
0.350263
	
It appears that it is performing less well, but it is helpful to look in more detail at the types of mistakes the model is making. The table below shows for each category of messages, what the most commonly predicted category was, outside of itself.

Correct Class
Most Common Prediction
Number of Documents
Immigration
Economics and Public Finance
53
Government Operations and Politics
Economics and Public Finance
45
Congress
Public Lands and Natural Resources
36
Environmental Protection
Public Lands and Natural Resources
23
Health
Crime and Law Enforcement
20
Law
Crime and Law Enforcement
13
Animals
Environmental Protection
12
Armed Forces and National Security
Immigration
9
Education
Taxation
9
Crime and Law Enforcement
Public Lands and Natural Resources
8
Science, Technology, Communications
Health
8
Civil Rights and Liberties, Minority Issues
Crime and Law Enforcement
7
Energy
Taxation
7
Families
Crime and Law Enforcement
7
Labor and Employment
Government Operations and Politics
5
Public Lands and Natural Resources
Environmental Protection
5
Social Welfare
Armed Forces and National Security
5
Transportation and Public Works
Health
5
Agriculture and Food
Health
4
Economics and Public Finance
Armed Forces and National Security
3
International Affairs
Government Operations and Politics
3
Commerce
Crime and Law Enforcement
2
Finance and Financial Sector
Economics and Public Finance
2
Foreign Trade and International Finance
Health
2
Taxation
Commerce
1
Emergency Management
Economics and Public Finance
1
Native Americans
Crime and Law Enforcement
1
Water Resources Development
Environmental Protection
1
	
There are two main reasons that can be identified for these mistakes. We will look at two categories now to illustrate them. 

The first table below shows messages that should be categorized as Immigration, but are labeled as Economics and Public Finance.

Subject Body Expected Prediction

These messages all mention the government shutdown, which triggers the label Economics and Public Finance because in the training set, any mention of government shutdown as tagged as a LAW and standardized to the bill End Government Shutdowns Act, which was labeled as CRS as Economics and Public Finance. This type of issue will be fixed by improving the entity detection algorithm, preventing government shutdown from being labeled as an entity, as well as diversifying the training set, so that government shutdowns are associated with more than one bill type. Finally, if this issue was to persist, active learning would allow users to flag this issue and provide correct labels, so the model can be updated.

The second table shows messages that should be categorized as Government Operation and Politics but are labeled as Economics and Public Finance.

Subject Body Expected Prediction

In these examples, offices tend to label anything mentioning the shutdown but not explicitly mentioning immigration as Government Operations and Politics, while for the reasons stated about, the model prefers Economics and Public Finance. It is possible in this situation that acceptance testing will reveal that the label given by the model is allowable, but most likely this will be fixed by more diverse training data. 

## 9.2 Considerations for Operalization

Acceptance testing is needed to determine if the model is ready to be deployed. While it appears that the model makes a lot of mistakes, it could be that when shown a messages and the predicted category, the users find the category assigned acceptable, even though an office member may have given a different category if asked. Once it is rolled out, an active learning set up will allow it to continually improve. Messages must be preprocessed in a specific way to be ready for input to the neural network. This code can be seen in the file fsml/policy_prediction.         

Our recommendation is to gather more data and further refine the model before deploying. This can either be done in the bootstrapping way above, or by assigning custom labels to the messages by hand. One issue is that offices are not consistent in their internal labeling of messages, so a third-party labeling, such as done by Fireside employees, may create a more robust model. The other issue to be aware of is that constituents write in about more than bills, and not every message will map to a bill category nicely. It may be best to augment or replace the CRS Policy Area label set. One way of doing this would be through explorations of topic models as mentioned above.

Further work is needed to determine the best retraining schedule, but the model should be retrained periodically. If the model is deployed in a manner that allows for user feedback, the amount of feedback given would serve as an indicator of when retraining was needed. Depending on the users uptake of the feedback mechanism, further hand-labeling of messages may also be necessary.

## 9.3 Future Work

The current model as implemented should provide a good baseline once given more training data. There are several potential improvements to be made to this model. One would be to determine if a different set of labels better matches the dataset, and to train a new model on that. Similarly, offices may want to add their own custom labels. In that case, each office would start with the same baseline classifier, but then be modified to allow for the prediction of new classifier types. In order to support this, the office would have to provide some examples of each new category to help the custom model be trained. The exact amount would need to be determined experimentally.

Additional work should be carried out to make the model more robust to variations in data. One such area that is relatively easy to account for is spelling robustness. Misspelled documents can cause errors in classification, so producing intentionally misspelled version of documents during training may strengthen the model.

Another way to strengthen the model would be through data augmentation. This has the potential to improve the classifier for all labels, but is especially important for labels that are rarely seen. For example, very few constituents are likely to write about “Water Resources Development”, but performance should not suffer because of that. There are several tools that can modify sentences to be about a different topic that can then be used in training. It will also be worthwhile to look into using text besides constituent messages to augment the data, such as news articles, opinion pieces, etc.

When it comes to the model itself, there are two immediate things to try, although it is likely that models will keep evolving. The model we used was the basic BERT model, which has 12 layers of transformers, but a larger model using 24 layers exists, and can boost performance slightly. Work would be needed to determine if the increase in resources needed to process the large model is worth it.

Recently a new model has emerged, with similar architecture to BERT, but trained on different tasks, that appears to do better at transfer learning, which is what we are interested in. This model, XLNET, is also available in 12 and 24 layer versions, both of which would need to be tested. In general, a system should be built such that dropping in a new pretrained transformer network is low-effort, as new models are likely to be produced that all are interchangeable from a higher level perspective with respect to classification. 

# 10. Sentiment Analysis

Generally speaking, sentiment analysis is any extraction of opinion from a piece of text. Usually it is used to refer to the labeling of a sequence of text as being positive or negative. This is similar to our goal of identifying messages in support or against something, but as we show below, they don’t line up completely. To perform sentiment analysis, we use a network very similar to the one used for classification above, using the 12 layer BERT pretrained network as a base, and adding on a 2 or 3 way classifier network to the output, depending on the training data. 

There are many available datasets for training sentiment analysis. We chose two datasets, based on their text content, to look at. The Stanford Sentiment Treebank is one of the most commonly used datasets for research into sentiment analysis, and consists of labeled reviews from the website Rotten Tomatoes. The other resource is EmoBank, which classifies newspaper articles, fiction, and other varied genre of text into 3 scores, valence, arousal, and dominance, which together can capture many emotions. Because we are interested in sentiment, we focus on only the valence score. Both of these allow the sentiment to be “positive”, “negative”, or “neutral”. 

To compare this to opposition and support of bills, we also use keywords to bootstrap labeled data from the constituent messages. The keywords contains things like “support”, “opposition”, etc. The full list can be seen in APPENDIX D. These examples were generated by the script fsml/get_sentiment_training_examples.py. The sentiment model is trained using the script train_sentiment_classifier.py which takes command line arguments. A full list of arguments can be seen by supplying the `--help` flag. To make predictions, the method predict_sentiment in the file fsml/predict_sentiment.py is used.

## 10.1 Results

The precision, recall, and f1 scores for running sentiment prediction on the test set of messages are shown below. 

training_set
test_set
f1
precision
recall
Office 1
Office 1
0.948207
0.949351
0.950130
Office 2
0.955647
0.955860
0.956487
Office 2
Office 1
0.949044
0.950425
0.950628
Office 2
0.929625
0.930732
0.932563
all
Office 1
0.952318
0.953270
0.952124
Office 2
0.951688
0.952617
0.951509
both
Office 1
0.960687
0.961265
0.961301
Office 2
0.958108
0.958357
0.958735
emo
Office 1
0.385395
0.791085
0.332535
Office 2
0.442047
0.815223
0.[Office 1]556
sst
Office 1
0.547801
0.770249
0.466986
Office 2
0.599253
0.776447
0.520873
	
As can be seen, the performance when using traditional training sets is very poor. While this could be due to different domains, it is most likely due to the notions of positive and negative not lining up exactly with supporting and opposing bills. On the other hand, the extremely high scores for the model trained on bootstrapped constituent messages is highly suspicious. Most likely the model learned to memorize the keywords that were used to find training examples.

To get a better idea of how well the model trained on bootstrap-labeled messages works, we selected 153 messages that did not contain any of the keywords, and labeled them by hand.

The results of running the model on these messages is shown below.

SiteID
F1
Precision
Recall
Office 1
0.4309
0.4137
0.3870
Office 2
0.5025
0.5238
0.5454

Some example sentences and their expected and predicted labels are below 

Subject Body Actual Prediction

These results illustrate several issues. The first is that the model overwhelmingly prefers to predict positive sentiment, but negative sentiment is quite common in constituent messages. The other issue is that messages that mention bills tend to have much more explicit mention of the sentiment. This can be confirmed by looking at messages which mention bills but do not have any of the keywords used to bootstrap the labels. 

SiteID
F1
Precision
Recall
Office 1
0.736
0.734
0.758
Office 2
0.873
0.873
0.859
	
These results show that the sentiment model needs to be trained on hand-labeled data to improve performance and be more generalizable. 

## 10.2 Considerations for Operationalization

These models should not be used in production as messages may often contain both positive and negative sentiment about different topics. In addition, these models require much more diverse set of input to generalize better. This could be achieved through human labeling of messages, or using data augmentation techniques. 

## 10.3 Future Work

Through discussions with the FS ML team it was decided that rather than gauging sentiment of an entire message, it is more useful to know how constituents feel about certain topics. This aligns well with an active research problem known as target or aspect based sentiment analysis. In this problem, a classifier is given a sentence and then specific words in that sentence to determine the sentiment of. An example is shown below
  
Training a classifier to do this will require ample training examples, which will need to be hand labeled. While the inference aspect of this model will require automatic identification of the targets, either through entity detection or some other means, the hand labeled data should rely on annotators to also label the targets of sentiment. It is important that some of the training data contain neutral targets, as not every entity will have a sentiment associated with it.  The training output might look like this:

Given the sentence “H.R. 586, Rep. Doug Collins' Fix the Immigration Loopholes Act, would help to streamline asylum claims by closing the credible fear loophole for asylum seekers and by modifying the Flores settlement to allow to hold minors for more than 20 days.”

The following output would be expected

Target
Sentiment
Sam Smith
Neutral
Fix the Immigration Loopholes Act
Pro
Smith settlement
Con
	
To make target based sentiment analysis ready for production, not only will annotation need to be done, but the entity detection system or other target identification model will need to be improved. By implementing target based sentiment analysis, creating something like is shown in the image below (taken from Fireside material) would be possible.
  
In addition, some of the data augmentation techniques discussed in the previous section will allow for a more robust model to be trained.

# APPENDIX

## Appendix A - List of Python Files

Experimental/Deprecated
clean_data.py - Early script used to do cleaning, functionality moved to OfficeData.py
read_db_data.py - Early script used to read in data from csv files, functionality moved to OfficeData.py
sbm_trial.py- Early script used to train topSBM, replaced by train_topSBM.py
Deliverables
combine_sentiment_data.py - Script to combine sentiment training data from SST dataset, emobank, and bootstrapped examples from constituent messages
get_ner_training_examples.py - Script to produce file for updating NER tagger in spacy. Produced a file that needs to be manually inspected and corrected for best results, but can be used right away
get_sentiment_training_examples.py - Script to produce data to train a sentiment analysis classifier by bootstrapping labels using key words such  "support", "oppose", etc.
get_sst.py - Script to take the Stanford Sentiment Treebank as it is given by Stanford, and convert to a csv to be used in the sentiment training script
get_training_examples_for_classification_115th.py - Generate training for policy area document classification training by using only messages from the 115th congress
get_training_examples_for_classification.py - Generate training for policy area document classification by selected random samples for each policy area from each district for training
get_unseen_policy_area_examples.py - Gather messages that don't refer to any known bills for use in testing trained classifier
get_unseen_sentiment_test_examples.py - Gather messages that don't contain any of the key words used for bootstrapping the sentiment classifier
get_unseen_test_examples_bills_only.py - Gather messages that don't contain any of the key words used for bootstrapping the sentiment classifier that contain at least one reference to a known bill
hdp_trial.py - Run HDP over messages and save the trained model as a pkl. This code needs manual editing to change settings
KeywordExtractor.py - Python class to be used to train and predict key words on messages
ner_training.py - Python script to update spacy NER tagger. Training data location must be set manually in the file
OfficeData.py - Python class to read in exported data as well as clean messages, extract bill numbers, and filter campaigns
policy_prediction.py - contains a single function, to apply the trained policy area prediction model to all messages in a DataFrame that is passed in, could potentially be moved to Office Data
predict_sentiment.py - contains a single function, to apply the trained sentiment prediction model to all messages in a DataFrame that is passed in, could potentially be moved to Office Data
process_emo_bank.py - Simple script to reformat the emobank data as downloaded into a format useable by the script train_sentiment_classifier.py
prune_ents.py - Script to take the entity mapping learned from Wikipedia, and remove all but the most common for each potential entity
simulate_doc_frequencies.py - Helper script designed to simulate the changed document frequencies that would be observed over time for keyword analysis
topic_model_metrics.py - Contains methods to compare two topic models and calculate the stability between them
TopSBM.py - Python Class to explore trained topic model, as well as apply it to new messages. Potentially could be combined with train_topSBM
train_policy_classifier.py - Script to train a neural network to predict the policy area of a message. Takes command line arguments that can be seen by calling script with --help
train_sbm_over_time.py - Script to run train_topSBM.py with different arguments to train many different topic models for analysis
train_sentiment_classifier.py - Script to train a neural network to predict sentiment of a message. Takes command line arguments that can be seen by calling script with --help
train_topSBM.py - Script to train topic model using topSBM. Takes command line arguments that can be seen by calling script with --help

## APPENDIX B - List of words removed during topic modeling

"support","american","house","act","brady","bill","urge","congress","vote",
                       "need","government","representative","people","year","law","legislation",'state',
                       "federal", "dear", "u.s.", "office","sincerely","consituent","tx","kevin","country",
                       "help","like","thank","national","=","representatives","include","nation","america",                    "oppose","write","ask","plan","pass","know","use","want","dc","honorable","cut","allow",
                       "important","united","states","concern","issue","member","end","rep.","sponsor",
                       "cosponsor", "h.r.", "cannon","longworth","ford","rayburn"

## APPENDIX C - Transformer Architecture

Self-Attention Mechanism in Transformer Layer

Transformer Layer 

## Appendix D - List of keywords used to bootstrap sentiment analysis

## Negative

Oppose
Not support
Vote against
Vote no
Reject
Object
Improve

## Positive

Support
In favor of
Vote yes
Pass
Vote for

## Appendix E - List of Jupyter Notebooks

Jupyter Notebooks used in this engagement are split into two directories:

* Dev_notebooks - which were used in exploration of ideas 
* Example_notebooks - which are used in final analysis and are generally more complete

The notebooks in dev_notebooks are:

* Cleaning Data Development.ipynb - Work on how to best clean messages, expanded in OfficeData.py
* Ella.ipynb - Isabella’s workspace
* Entity Detection.ipynb - Initial steps towards entity extraction
* HR Detection.ipynb - Initial steps towards detecting HR numbers, further developed in OfficeData.py
* Languages.ipynb - Experiments about which languages are used in messages
* Most Common Words.ipynb - Basic statistics about most common words in messages
* Phrase Extraction.ipynb - Basic work on extracting phrases from messages
* Reading Congress XMLs.ipynb - Initial work on reading XML dumps from congress, later used in OfficeData.py
* Scott.ipynb - Scott’s workspace
* Scratch.ipynb - Catch all notebook for random code execution, not well organized
* TextRank Keyword Extraction.ipynb - Brief look at alternative method for keyword extraction, didn’t pan out
* TFIDF Keyword Extraction.ipynb - Initial work for extracting keywords, used as basis for KeywordExtractor.py
* Topic Modeling.ipynb - Initial steps on topic modeling using HDP, continued in hdp_trial.py
* TopSBM Exploration.ipynb' - Initial steps on topic modeling using TopSBM,  continued in train_topSBM.py
* USE.ipynb - Exploration of encoded messages as vectors and clustering them
* Wikipedia Processing for Entities.ipynb - Initial steps to extract Wikipedia links and the pages they link to

The notebooks in example_notebooks are:

* Comparing TopSBM Models.ipynb - Basic comparisons of topSBM models over time, a much more robust and in depth comparison is done int “Stability of TopSBM over Time.ipynb”
* Entity Detection.ipynb - More complete investigation in entity detection, including comparison of different models
* Extracting Bills and Policy Areas.ipynb - Shows the results of bill extraction, as well as a brief exploration into different types of possible failures in bill extraction
* Policy Area Prediction.ipynb - Analysis of policy prediction NN
* Sentiment Analysis.ipynb - Analysis of sentiment prediction NN
* Stability of TopSBM over Time.ipynb - In depth analysis and plot creation of model stability
* TFIDF Keyword Extraction.ipynb - Example use of the KeywordExtraction class 
* Topic Modeling.ipynb - Example use of the TopSBM class
* Topic Prediction of Unseen Documents with TopSBM.ipynb - Analysis of using topSBM for predicting unseen messages, focusing on number of documents that are not assigned a topic

## Appendix F - List of data files generated

During the engagement, numerous data files were generated. Some are used in training and testing, while others were exported for review by Fireside. 

The files that are in the main fsml directory are 

   * Classification_test_115th.csv - Policy area labeled messages from the 116th congress (for use when trained on the 115th)
   * Classification_test.csv - Random selection of messages from 116th congress used to test policy classifier
   * classification_training_115th.csv - Policy area labeled messages from the 115th Congress used for training
   * Classification_training.csv -  Random selection of messages from 116th congress used to train policy classifier
   * comparison_of_topSBM_164.csv - Comparison of all topSBM models trained on Office 164
   * comparison_of_topSBM_[Office 4].csv - Comparison of all topSBM models trained on Office [Office 4]
   * comparison_of_topSBM.csv  - Comparison of all topSBM models trained on [ Office 1 ]
   * Ner_edited_training_examples.csv - Partially hand corrected training examples for entity detection
   * Office_labeled_messages.csv - Messages labeled by offices that contained only one labeled
   * Sentiment_test.csv - Messages used for testing sentiment analysis, that are labeled by using bootstrapped keywords
   * Sentiment_training.csv - Messages used for training sentiment analysis, that are labeled by using bootstrapped keywords
   * Unseen_policy_area_test.csv - 20 messages from each office that do not contain any bill reference, hand labeled
   * Unseen_sentiment_test_bills_only.csv - 100 messages sampled from offices [Office 1] and [Office 2] that mention bills that are hand labeled for sentiment (some examples dropped due to being empty, not expressing sentiment, etc.)
   * Unseen_sentiment_test.csv - 100 messages sampled from offices [Office 1] and [Office 2] that are hand labeled for sentiment (some examples dropped due to being empty, not expressing sentiment, etc.)
The files found in the directory fsml/dev_notebooks are 
   * [Office 1]_doc_groups_and_topics_no_campaign_filtering.csv - Result of running topic modeling on [ Office 1 ] without campaign filtering
   * [Office 1]_doc_groups_and_topics_one_message_per_campaign.csv - Results of running topic modeling (topSBM) on [ Office 1 ] while limiting each campaign to one message 
   * [Office 1]_emails_with_topics_v3.csv - Results of running topic modeling (HDP) on [ Office 1 ]
   * [Office 1]_entities_found_v1.csv - Initial results of extracting entities from [ Office 1 ]
   * [Office 1]_first_top_sbm_results.csv - Output of topics and associated words for [ Office 1 ]
   * [Office 1]_no_law_found.csv - Messages with no legal entities identified by entity extraction
   * [Office 1]_topic_hierarchy_no_campaign_filter.csv - The topic hierarchy generated for [ Office 1 ] using topSBM
   * [Office 1]_topic_hierarchy_one_message_per_campaign.csv - The topic hierarchy generated for [ Office 1 ] using topSBM, limiting candidates to one message
   * [Office 1]_topics_for_emails_v3.csv - The list of topics found using HDP
   * [Office 1]_topics_with_only_one_message_per_campaign.csv - List of the topics found using HDP using one message per campaign
   * [Office 2]_messages_with_clusters.csv - All messages in [Office 2], labeled with cluster ID from running DBSCAN over vectors from USE
   * Linked_entities.csv - Results of running entity linking prototype over messages
   * Ner_potential_training_examples.csv - Entity tagged messages in format needed for training ner in spacy, as output from spacy, with no manual correction
   * Scott_entities_with_ids.csv - messages with entities extracted, labeled with site and message ID
   * topics_noCampaignFiltering.csv - File generated by Isabella from Ella.ipynb
   * topics_oneMsg_perCampaign_[Office 1].csv  - File generated by Isabella from Ella.ipynb


The files in fsml/example_notebooks 

   * 116th_policy_predictions_from_115th_training.csv - Results of running policy prediction trained on 115th Congress on messages from 116th Congress
   * 2016_new_topics_count_by_day.csv - The number of new topics per day in 2016 when using frozen topSBM model 
   * 2016_new_topics_percentage_by_day.csv - The percentage of new topics per day in 2016 when using frozen topSBM model 
   * 2018_new_topics_count_by_day.csv - The number of new topics per day in 2018 when using frozen topSBM model 
   * 2018_new_topics_percentage_by_day.csv - The number of new topics per day in 2018 when using frozen topSBM model 
   * [Office 1]_first_document_classification_results.csv - Results of first attempt of running policy area prediction on [ Office 1 ]
   * [Office 2]_bill_titles_resovled.csv - Messages from [Office 2] annotated with bill names and numbers
   * [Office 2]_first_document_classification_results.csv -  Results of first attempt of running policy area prediction on [ Office 1 ]
   * Classification_training_115th.csv - Copy of file from fsml main directory of ease of use in notebook
   * Classification_training.csv  - Copy of file from fsml main directory of ease of use in notebook
   * Deleted_messages_for_policy_area.csv - Messages which were deleted from test set for having policy areas given by congressional offices that did not appear to match the content of the email
   * Office_labeled_messages.csv - Messages with one policy label as given by congressional offices
   * Scott_keywords.csv - Results of apply keyword extraction on [ Office 1 ]
   * Sentiment_test.csv - Copy of file from main fsml director for ease of use in notebook
   * Sentiment_training.csv - Copy of file from main fsml director for ease of use in notebook


