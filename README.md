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
    
## References:  
https://github.com/automan000/Convolution_LSTM_PyTorch

## Notes  
none
 

 
 