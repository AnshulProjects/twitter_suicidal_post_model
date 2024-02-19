# twitter_suicide_detection_model

BERT model fine-tuned to determine whether Twitter posts are potentially suicidal.


I used Huggingface libraries to perform the fine-tuning.


## Format of prompt  

```python
prompt = "Is this post potentially suicidal: " + tweet
```

## Before Use
Use the above format to generate the most accurate completions.

## Completion
0 = " Not a suicidal post ", 
1 = " Potentially Suicidal "

## Training/Test split and other information

80:20 %
Epochs = 1

I want to be able to run this on a GPU and aquire more training data as well.

This model is in the hugging face repository:

https://huggingface.co/Anshul2000s/twitter_suicide_detection_model