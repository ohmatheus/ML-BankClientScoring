# ML-BankClientScoring
Machine Learning Projet from Kaggle - application of several machine learning techniques to predict the capacity of clients to repay a loan at time.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-this-project">About This Project</a>
      <ul>
        <li><a href="#using">Using</a></li>
      </ul>
	  <ul>
        <li><a href="#dataset-used">Dataset Used</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
	<li><a href="#unet">UNet</a></li>
		<li>
			<a href="#unet-presentation">Unet Presentation</a>
			<ul>
			<li><a href="#unet-trainning">Unet Trainning</a></li>
			</ul>
			<ul>
			<li><a href="#unet-results-&-inference">Unet Results & Inference</a></li>
			</ul>
		</li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#references">references</a></li>
  </ol>
</details>

<!-- ABOUT THIS PROJECT -->
## About This Project
This project is from a competition in Kaggle called 'Home Credit Default Risk' where the goal is to predict how capable each applicant is of repaying a loan:
[Link to Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk/overview)

The database used for this project is quite big, and uses several SQL table. Which can be represented by this image :
![database](./Images/DB.png)

Our goal is here to identify and/or create variables that will permit us to build and train appropriate machine mearning models so we can predict as much as possible if a client is able to repay a load in time, or not.

Throughout this entiere process, we will use the [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) metrics to compare models, as it is the same metric used by Kaggle for this competition. As a disclaimer, we reached 0.706 on the ROC Auc for this project, best results are arount 0.850 last time i checked, wich leave us some room for many improvements.

<!-- EDA -->
## EDA
The database is built upon raw data, meaning we have to clean it. If you want more details you can see the first notebook, but as part of the cleaning we :
- Took care of missing values by removing them or imputation
- Encoded (oneHot or Label encoding) some of the categorical strings
- Took care of anomalies

After that we end up with a total ~350K clients (labelised) and 239 features usable for trainning. We split test/train to something arount 0.2/0.8

Reguarding the proportion of clients who repaid in time and those who didn't, we have :
![proportion](./Images/proportion.png)

Meaning we will need to adapt our algorithm to fit those unbalanced data.

<!-- Basic correlations -->
### Basic correlations
A simple correlation table (check 1st notebook for more infos) reveal that the 5 first features positively correlated to our TARGET are :
- `DAYS_BIRTH` : "Client's age in days at the time of application" (Pearson 0.078)
- `DAYS_EMPLOYED` : "How many days before the application the person started current employment" (Pearson 0.075)
- `REGION_RATING_CLIENT_W_CITY` : "Our rating of the region where client lives with taking city into account (1,2,3)" (Pearson 0.061)
- `REGION_RATING_CLIENT` : Our rating of the region where client lives (1,2,3) (Pearson 0.059)
- `NAME_EDUCATION_TYPE` (encoded) : "Level of highest education the client achieved" (Pearson 0.054)

And the 5 most negatively correlated features are :
- `EXT_SOURCE_3` : "Normalized score from external data source"
- `EXT_SOURCE_2` : "Normalized score from external data source"
- `EXT_SOURCE_1` : "Normalized score from external data source"
- `AMT_GOODS_PRICE` : For consumer loans it is the price of the goods for which the loan is given
- `NAME_EDUCATION_TYPE` (encoded) : "Level of highest education the client achieved"

`EXT_SOURCE_X' 1 2 and 3 are data provided by the Home Credit, and we have no intel on their meaning.

<!-- DAYS_EMPLOYED -->
#### DAYS_EMPLOYED
Distribution:
![days_employed_prop](./Images/days_employed_prop.png)

Distribution compared to target:
![days_empoyed_traget](./Images/days_empoyed_traget.png)

![faillure_Daysemployed](./Images/faillure_Daysemployed.png)

We clearly have a pattern here. People employed since less time seems to have a lower probability to repay the loan at time.

<!-- DAYS_BIRTH -->
#### DAYS_BIRTH
![daysbirth_distrib](./Images/daysbirth_distrib.png)

![daysbirth_target](./Images/daysbirth_target.png)

![daysbirth_failure](./Images/daysbirth_failure.png)

Younger people seems to have a lower chance to repay the loan at time.

<!-- NAME_EDUCATION_TYPE -->
#### NAME_EDUCATION_TYPE
![nameeducation_target](./Images/nameeducation_target.png)

![nameeducation_prop](./Images/nameeducation_prop.png)

In the same way, people with a higher degree seems to have more chance to repay their loan.

<!-- EXT_SOURCE (1, 2, 3) -->
#### EXT_SOURCE (1, 2, 3)

Let's do a correlation heatmap for those variable:

![ext_corrheat](./Images/ext_corrheat.png)

There is a negative correlation between our Target and all 3 EXT_SOURCE.
Plus high correlation between DAYS_BIRTH and EXT_SOURCE_1

Here is the distribution of all EXT_Source_X compared to our Target:
![exttarget](./Images/exttarget.png)

We can see some sort of correlation between those features and the Target.

<!-- Feature Engineering -->
## Feature Engineering
Now we need to create mores variables/features to see if we can get more correlated variable to our Target.

We can :
- Use PolynomialFeatures from scikit-learn on all Ext_Source_X features -> created some interesting features added to our dataset
- Create some other features using common sense :
	- `CREDIT_INCOME_PERCENT` : % of credit cost in comparision of revenus
	- `ANNUITY_INCOME_PERCENT` : % of yearly credit cost in comparision of revenus
	- `CREDIT_TERM` : credit time (in month)
	- `DAYS_EMPLOYED_PERCENT` : ratio between DAYS_EMPLOYED and DAYS_BIRTH

![handlyfeature](./Images/handlyfeature.png)

Nothing particularly special, but we still add them into our dataset.
 
<!-- Feature Aggregation -->
### Feature Aggregation, numeric and categorical
We aggregate the features present in all the tables relative to our Target to get the maximum values out of our database.
After numerical and categorical aggregation (see 2nd notebook for info), 

> [!NOTE]  
> There is some automatic feature engineering librairies like featuretools that we could use, but to stay simple for now we wont.
> We also can check different operation on some variables, like derivation (acceleration), to see if it could gives us more correlated features.

<!-- Feature Selection -->
### Feature Selection
After this engineering we end up with more than 1760 features, we need to reduce that number:
- Cleaning again missing values, eventual duplicates, and ID feaures
- removeing colinear variables (>850)

And we are going to select only the features that represents the first 95% of the cumulative importance relative to Target with LightGBM (lightGradientBoosting) :

![featureimportance1](./Images/featureimportance1.png)

It's interesting to see that in those feature, we can see the 3 EXT_SOURCE_X features that we already worked with, but also some feature that has been created with aggregation like 'burea u_DAYS_CREDIT_max' wich can be translated as 'the maximum number of days between each credit'

![cumulativeimportance1](./Images/cumulativeimportance1.png)

After selecting this 95% of cumulative importance, we end up with ~350 featuers, wich is what we need to train some models.

> [!NOTE]
> ACP could have been used for dimentionnality reduction, but not optimal for features explanation
> ICA
> Manifold Learning ?

<!-- Modeling -->
## Modeling
That we get to the interesting stuff. We have ~350 features to predict a probability, between 0 and 1. We need to find the model that fit our needs.

<!-- Model Selection -->
### Model Selection

We create a custom scoring method (unbalanced)  based on accuracy to compare models.
Then we do a first grid search on 
using SMOTE

![Baseline.png](./Images/Baseline.png)

![randomsearch](./Images/randomsearch.png)

![testdistrib](./Images/testdistrib.png)

![featureimportance_reel](./Images/featureimportance_reel.png)

![cumimportance_reel](./Images/cumimportance_reel.png)

![goodclient](./Images/goodclient.png)

![badclient](./Images/badclient.png)

