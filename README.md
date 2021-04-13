# Taylor_Swift_Text_Generator
As a fun personal project, I used a Recursive Neural Network Model trained on all of Taylor Swift's song lyrics (through the 2020 Evermore album) to predict the next word in a given dialogue.

# Web Scrape Data
First scraped all of Talor Swift's lyrics from Music Genius along with a few other attributes I might find interesting. Saved this as an outfile.

# Building Neural Network
Used keras and tensorflow to build a Recursive Neural Network with Embedding and LSTM hidden layers to predict the next word in a given sequence. I trained the model on 25 word sequences and 6 word sequences but liked the 6 word model was better.

# Prediction Function
A function was needed that takes in a user defined string of text (can be a taylor swift lyric or your own text). If the user inputs less than 6 words, the model will pad the vectorized sequence with 0's to a length of 6 before predicting. If the user inputs more than 6 words, the model will make its prediction on the first 6 words in the sequence. This function requests the user pass a desired length of words to pass back. So if I pass the model a 6 worded sequence and want to know the next 10 words Taylor Swift would say, I ask the model to generate the next 10 words. Within this function, the model will be called 10 times with each iteration predicting on the previous 6 words. (So by the time we get to the prediction of the 7th word, the model will be predicting on a sequence of the 6 previous words it predicted.) The model uses a beam search with k=5 and then returns the top 5 best sequences.

# Conclusions
A pretty accurate model when fed Taylor Swift Lyrics! When feeding the model pieces of my own poetry, the output sounded deep and mystical, like maybe the start of Taylor Swift's next new pop hit!?
