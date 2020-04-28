from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import warnings


warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

article = Article('https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521')
article.download()
article.parse()
article.nlp()
corpus = article.text

#print(corpus)

text = corpus
sent_tokens = nltk.sent_tokenize(text)

#print(sent_tokens)

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

#Print the punctuations
#print(string.punctuation)

#print the dictionary
#print(remove_punct_dict)

def LemNormalize(text):
  return nltk.word_tokenize(text.lower().translate(remove_punct_dict))

#Print the tokenization text
#print(LemNormalize(text))


#Keyword Matching

#Greeting Inputs
GREETING_INPUTS = ["hi", "hello", "hola", "greetings", "wassup", "hey", "hi there", "good morning", "good afternoon"]

#Greeting responses
GREETING_RESPONSES = ["howdy", "Hi", "Hey", "What's good", "Hello", "Hey there"]

#Function to return a random greeting response
def greeting(sentence):
  #if user input's a greeting, then return a randomly choosen greeting response
  for word in sentence.split():
    if word.lower() in GREETING_INPUTS:
      return random.choice(GREETING_RESPONSES)

#Generate the response
def response(user_response):

  #The user's query
  #user_response = 'what is chronic kidney disease'
  user_response = user_response.lower()
  #print(user_response)

  #Set the chatbot response to an empty string
  robo_response = ''

  #Append the users response to the sentence list
  sent_tokens.append(user_response)
  #print(sent_tokens)
  #Create a TfidfVectorizer object
  TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words='english')

  #Convert the text to a matrix of Tf-Idf features
  tfidf = TfidfVec.fit_transform(sent_tokens)
  #print(tfidf)

  #Get the measure of similarity (similarity score)
  vals = cosine_similarity(tfidf[-1], tfidf)
  #print(vals)
  

  #Get the most similar text/sentence to the users response
  idx = vals.argsort()[0][-2] 

  #Reduce the dimensionality of vals
  flat = vals.flatten()

  #sort the list in ascending order
  flat.sort()

  #Get the most similar score to the users response
  score = flat[-2]
  #print(score)

  #If the variable 'score' is 0, then there is no text similar to users response
  if(score == 0):
    robo_response = robo_response + " I apologise, I don't understand."
  else:
    robo_response = robo_response + sent_tokens[idx]

  #Print the chatbot response
  #print(robo_response)

  #Remove the user response from sentence token list
  sent_tokens.remove(user_response)
   
  return robo_response


flag = True
print("VirDoc: Hey, I am your Virtual Doctor. I will answer your queries about Chronic Kidney Diseases. If you want to exit type Bye")
bye = ['bye', 'talk to you later', 'ttyl', 'it was nice talking to you', 'seeya soon', 'i dont need your help', 'enough for now']
bye_response = ['Talk to you later, bye!', 'It was nice talking to you.', 'Have a nice day :)', 'Anytime you need me, I am always here. Bye:)', 'Get well soon, bye.', 'Okay, bye!!']
thanks = ['thanks','thankx','thnkx','thank you', 'thenkx','it was helpful']
while(flag == True):
  print("You: ")
  user_response = input()
  user_response = user_response.lower()
  remove_punctuation_dict = dict((ord(punct), None) for punct in string.punctuation)
  user_response.translate(remove_punctuation_dict)
  if(user_response not in bye):
    if(user_response in thanks):
      flag = False
      print("VirDoc: You are welcome! ")
    else:
      if(greeting(user_response) != None):
        print("VirDoc: " + greeting(user_response))  
      else:
        print("VirDoc: "+ response(user_response))    
  else:
    flag = False
    print("VirDoc: " + random.choice(bye_response))
    


