# twitter_suicide_detection_model

This is a BERT model fine tuned to determine whether twitter posts are potentialy suicidal in nature.


I used Hugging Face libraries to preform the fine tuning.


## Format of prompt  

```python
prompt = "Is this post potentially suicidal: " + tweet
```

## Before Use
Use the above format to generate most accurate completions.

## Completion
0 = " Not a suicidal post ", 
1 = " Potentially Suicidal "

## Training/Test split and other information

80:20 %
Epochs = 1

I want to be able to run this on a gpu sand more traiing data as well.

This model is in the hugging face repository:

https://huggingface.co/Anshul2000s/twitter_suicide_detection_model

