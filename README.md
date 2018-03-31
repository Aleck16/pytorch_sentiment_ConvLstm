# ConvLstm Models for Sentence Analysis in PyTorch
This repo is aiming for reproducing the sentence classifcation experiments in Hassan et al. (IEEE 2017).
http://ieeexplore.ieee.org/document/7942788/

## Datasets  
### MR Sentiment Analysis  

https://www.cs.cornell.edu/people/pabo/movie-review-data/  
Train+dev+test = rt-polarity.neg + rt-polarity.pos  
all = 5331*2 = 10662 = 8500(train) + 1100(dev) + 1062(test) 

## Dependencies  
* torch  
* python3.6

## Usage  
To train: 
 
    python ConvLstm_sa_all2.py  

Print result graph:  

    python view_result.py
      
  
## Performances  
The results of this experiment did not reach the accuracy of the paper. If the optimization parameters are estimated several times, a good result can be obtained. Here only a code reference, a basic implementation. The following is the experimental result.  

Loss:
  
![avatar](https://github.com/Aleck16/pytorch_sentiment_ConvLstm/blob/master/sa_loss.png)

Accuracy:  
  
![avatar](https://github.com/Aleck16/pytorch_sentiment_ConvLstm/blob/master/sa_acc.png)  
  
## References:  
https://github.com/yuchenlin/lstm_sentence_classifier  
  
## Notes  
The best_models folder is stored in my experimental model.  
The logs folder contains the training output log.  
If you have any questions, welcome to ask questions.  
 

 
 