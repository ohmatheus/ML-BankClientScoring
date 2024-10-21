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
### EDA
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

- Remove colinears features

Feature Engineering

Modeling
