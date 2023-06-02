import telebot
import json
import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# I intentionally left it here
token = "6129286702:AAHj2Jt4G782LZOO_yCIpY-tOjprJ5gL5JE"
bot = telebot.TeleBot(token)


json_file_with_questions_and_answers = open('questions_to_answers.json', 'r', errors = 'ignore')
questions_to_answers = json.load(json_file_with_questions_and_answers)
questions_to_answers = { k.lower() : questions_to_answers[k] for k in questions_to_answers }

questions = '\n'.join(questions_to_answers.keys())

nltk.download('punkt') #tokenizer for english
nltk.download('wordnet')

sentence_tokens = nltk.sent_tokenize(questions) #convert to list of sentences
word_tokens = nltk.word_tokenize(questions) #convert to list of words

print(f'len(sentence_tokens): {len(sentence_tokens)} len(questions_to_answers.keys()): {len(questions_to_answers.keys())}')



lemmer = nltk.stem.WordNetLemmatizer()

#Wordnet is a semantically-oriented dictionary of English included in NLTK

def remove_puctuation(text):
   return text.translate(
        dict((ord(punct), None) for punct in string.punctuation)
    )

def lemNormalize(text):
    return [
        lemmer.lemmatize(token)
        for token in nltk.word_tokenize(remove_puctuation(text.lower()))
    ]




def get_answer_to(message_of_user):
    sentence_tokens.append(message_of_user)
    TfidfVec = TfidfVectorizer(tokenizer=lemNormalize, stop_words = 'english')

    # Get the TF IDF weighted Document-Term Matrix
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    sentence_tokens.remove(message_of_user)

    # Get the cosine similarity between user query and all the sentences in corpora. This will be a vector of similarities
    vals = cosine_similarity(tfidf[-1], tfidf) # type: ignore
    print(vals)

    # Get the sentence (document) which matches the most with the query
    idx = vals.argsort()[0][-2]

    # Get the td-idf value of the index which matched the most
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    return (
       None if req_tfidf == 0
       else random.choice(questions_to_answers[sentence_tokens[idx]])
    )


@bot.message_handler(content_types=['text'])
def check_message(message_of_user):

    bot.send_message(
        message_of_user.chat.id,
        'Conversation started!' if message_of_user.text == '\\start'
        else (get_answer_to(message_of_user.text) or "I am sorry! I don't understand you")
    )

if __name__ == "__main__":
    bot.polling()
