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

Throughout this entiere process, we will use the [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) metrics to compare models, as it is the same metric used by Kaggle for this competition.

EDA

Feature Engineering

Modeling
