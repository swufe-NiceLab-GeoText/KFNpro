# Few-Shot Text Classification Meets Black-Box Attacks: Ready or Not?
 KFNpro is a Variational Prototype Learning model. It designed to retrieve, calibrate class prototypes resilient to adversarial inference.


## Requirement
	matplotlib==3.6.2
    nltk==3.8.1
    numpy==1.20.3
    pandas==2.0.3
    seaborn==0.12.2
    torch==1.13.0
    tqdm==4.61.1
    transformers==4.44.2
##  Datasets
For all the datasets used in our experiments, you can download them from [here](https://github.com/tttyyyzzz-zty/SELP/tree/master).
For the attack dataset, you can do the following for these dataset.
The textual adversarial attack algorithm BERT-Attack (Li et al. 2020) is used to perform synonym substitution and generate adversarial examples. The codes of BERT-Attack can be found at [https://github.com/LinyangLee/BERT-Attack](https://github.com/LinyangLee/BERT-Attack).

After processing the dataset put it in the same directory as the original dataset and add attack to the file name to differentiate it, for example `attack_{dataset}`.
##  Training and Evaluation
We provide examples of training and evaluating models in the `attack` and `clean` folder. For example, to train and evaluate a model on the News dataset in a 1-shot setup, simply run the following command:

	bash attack/attack_Domain1_1shot.sh
    bash clean/Domain1_1shot.sh
 
