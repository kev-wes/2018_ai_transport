# coding: utf-8

# In[80]:


# -*- coding: utf-8 -*-
# Needed project interpreters:
# keras
# tensorflow
import re
from keras.preprocessing.text import hashing_trick
import csv, os
import pynlpir
from langdetect import detect
# Needed project interpreters:
# keras
# tensorflow
from keras.preprocessing.text import hashing_trick
import spacy
from spacy.tokenizer import Tokenizer
import csv, os
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentAn2 = SentimentIntensityAnalyzer()
import pandas, json
from pandas.io.json import json_normalize
import hashlib
from nltk.stem import PorterStemmer
from nltk.stem.snowball import GermanStemmer
from nltk.stem.snowball import FrenchStemmer

#tokenize english sentences using NLTK library
def token(column):
    list_of_words = [i.lower() for i in wordpunct_tokenize(column) if i.lower()]  # not in stop]
    return list_of_words


######Init####
#define stopwords
#load english stopwords from nltk
stop = set(stopwords.words('english'))
#load additional stopwords representing punctuation
strsplit = [i.split('\t')[0] for i in string.punctuation]
#remove words from stopword set
stop.remove('not')
stop.remove('same')
#add project-specific stop-words that were encountered during programming
stop.update(
    ['.,', '.?', 'œ', 'donâ', '¤', 'Â', 'â', '€', '.', ',', '"', "'", '°', '?', '!', '^', '“', '°', '–', '\n', '\r',
     ':', ';', '`', '´', '(', ')', '[', ']', '{', '}', '-', '--', '----', '/', "'\'",
     '...'] + strsplit)
#stopwords for each language
stopGen = set(
    ['.,', '.?', 'œ', 'donâ', '¤', 'Â', 'â', '€', '.', ',', '"', "'", '°', '?', '!', '^', '“', '°', '–', '\n', '\r',
     ':', ';', '`', '´', '(', ')', '[', ']', '{', '}', '/', "'\'", 'dear', 'best'
                                                                           '...'] + strsplit)
# stopGen.update(['mit', 'freundlichen', 'grüßen', 'grüssen'])
# stopGen.update(['best', 'regards', 'dear'])

stop2 = [i.lower() for i in stop]
listR = []
#load german stop-words
stopDe = stopwords.words('german')
#load german training terms for tokenization and POS tagging
nlpDE = spacy.load('de_core_news_sm')
#load german stop-words
stopFr = stopwords.words('french')
#load french training terms for tokenization and POS tagging
nlpFR = spacy.load('fr_core_news_sm')
#load chinese stop-words
stopwordsCHN = [line.rstrip() for line in
                open(os.path.join(path, 'Chinese\\chinese_stopwords.txt'), "r", encoding="utf-8")]
stopwordsCHN = stopwordsCHN + [" ", "  "]
#define contractions which should be replaced
contractions = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have",
    "couldnt": "could not",
    "couldn": "could not",
    "wasnt": "was not",
    "wasn": "was not",
    "isnt": "is not",
    "isn": "is not",
    "doesnt": "does not",
    "doesn": "does not",
    "arent": "are not",
    "aren": "are not",
    "shouldnt": "should not",
    "shouldn": "should not",
    "wont": "could not",
    "mightnt": "might not",
    "mightn": "might not",
    "aint": "should not"
}
######Init End####
#Do language-specific pre-processing, col = email text, lang = language string ( e.g. 'en')
def cleanAll(col, lang):
    #English language pre-processing
    if (lang == 'en'):
        c = list(wordpunct_tokenize(col))
        #replace contractions with full term
        for key, value in contractions.items():
            for x in c:
                if key == x:
                    index = c.index(x)
                    c.remove(c[index])
                    c.extend(wordpunct_tokenize(contractions[key]))
                    # c = c.replace(x, contractions[key])
        #remove stop-words
        for i in c:
            for j in i:
                if j.lower() in stop2:
                    i.replace(j, '')
        #stem words
        port = PorterStemmer(mode='NLTK_EXTENSIONS')
        list2 = [port.stem(i) for i in c if i.lower() not in stop2 or i not in stop2]
        #extract sentiment (positive, negative, neutral) from each email text
        sentPol = sentAn2.polarity_scores(col)
        #POS tagging
        P = nltk.pos_tag(list2)
        #P consists of set of tupels: first value original word, second one POS tag
        POS = [i[1] for i in P]
        return col, POS, list2, sentPol['neg'], sentPol['neu'], sentPol['pos'], sentPol['compound']
    # Chinese language pre-processing
    if(lang == 'chn'):
        pynlpir.open()
        #chinese tokenization
        tokenPOS = pynlpir.segment(col)
        #remove stop-words
        list_body = [i[0] for i in tokenPOS if i[0] not in stopwordsCHN]
        #POS tagging for email text (with stop-words removed)
        pos_body = [i[1] for i in tokenPOS if i[0] not in stopwordsCHN]
        return col, pos_body, list_body, 0, 0, 0, 0
    # French language pre-processing
    if (lang == "fr"):
        #tokenization only defined in if-block, called later in function
        tokenPOS = nlpFR(col)
        #define stemmer language
        Stem = FrenchStemmer()
        #set stop-word pointer to french stop-word set
        stopW = stopFr
    #German language pre-processing
    if (lang == "ger"):
        #tokenization only defined in if-block, called later in function
        tokenPOS = nlpDE(col)
        # define stemmer language
        Stem = GermanStemmer()
        #set stop-word pointer to french stop-word set
        stopW = stopDe
    pos = []
    tokens = []
    #Tokenization for all languages but English and Chinese, tokenizer points to previously trained, language specific one
    for t in tokenPOS:
        pos.append(t.pos_)
    #get POS tokens
        if t.text not in stopW:
            tokens.append(t.text)
    #Stemming with pointer reference to relevant language
    list2 = [Stem.stem(i) for i in tokens if i.lower() not in stopW]
    #return original text, part of speech tags, stemmed text and sentiment scores set to 0
    return col, pos, list2, 0, 0, 0, 0

#Function which calls function for language-specific pre-processing, c1 = e-mail subject, c2 = e-mail body
def cleanL(c1, c2):
    lang = 'other'
    #if c1 and c2 two separate sets (e-mail body and e-mail subject), then length of CL = 2
    #if only e-mail body exists, then length of CL = 1
    cL = [c1, c2]
    #initialize values
    pos_body, list2_body, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj, neg_body, neu_body, posi_body, compound_body = "", c1, "", c2, 0, 0, 0, 0, 0, 0, 0, 0
    #detect language, if language english, then pass 'en' as parameter
    if (detect(c2) == 'en'):
        print('EN')
        lang = 'en'
        #if both e-mail body and subject exist, pass both on to pre-processing, else only e-mail body
        if (len(cL) > 1):
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj = cleanAll(c1, 'en')
            col_body, pos_body, list2_body, neg_body, neu_body, posi_body, compound_body = cleanAll(c2, 'en')
        else:
            col_body, pos_body, list2_body, neg_body, neu_body, posi_body, compound_body = cleanAll(c2, 'en')
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj = "", "", "", 0, 0, 0, 0
    #detect language, if language french, then pass 'fr' as parameter
    elif (detect(c2) == 'fr'):
        lang = 'fr'
        print('FR')
        # if both e-mail body and subject exist, pass both on to pre-processing, else only e-mail body
        if (len(cL) > 1):
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj = cleanAll(c1, 'fr')
            col_body, pos_body, list2_body, neg_body, neu_body, posi_body, compound_body = cleanAll(c2, 'fr')
        else:
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj = cleanAll(c2, 'fr')
            col_body, pos_body, list2_body, neg_body, neu_body, posi_body, compound_body = "", "", "", 0, 0, 0, 0
    # detect language, if language german, then pass 'de' as parameter
    elif (detect(c2) == 'de'):
        lang = 'de'
        print('DE')
        # if both e-mail body and subject exist, pass both on to pre-processing, else only e-mail body
        if (len(cL) > 1):
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj = cleanAll(c1, 'ger')
            col_body, pos_body, list2_body, neg_body, neu_body, posi_body, compound_body = cleanAll(c2, 'ger')
        else:
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj = cleanAll(c2, 'ger')
            col_body, pos_body, list2_body, neg_body, neu_body, posi_body, compound_body = "", "", "", 0, 0, 0, 0
    # detect language, if language chinese, then pass 'chn' as parameter
    elif (detect(c2) == 'zh-cn'):
        lang = 'chn'
        print("CHN")
        # if both e-mail body and subject exist, pass both on to pre-processing, else only e-mail body
        if (len(cL) > 1):
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj = cleanAll(c1, 'en')
            col_body, pos_body, list2_body, neg_body, neu_body, posi_body, compound_body = cleanAll(c2, 'chn')
        else:
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj = cleanAll(c2, 'chn')
            col_body, pos_body, list2_body, neg_body, neu_body, posi_body, compound_body = "", "", "", 0, 0, 0, 0
        # return pos_body, list2_body, pos_subj, list2_subj,neg_subj, neu_subj, posi_subj,compound_subj, neg_body, neu_body, posi_body,compound_body
    else:
        # if no known language is detected, apply english language-specific pre-processing
        lang = "Other"
        print("Other", c2)
        # if both e-mail body and subject exist, pass both on to pre-processing, else only e-mail body
        if (len(cL) > 1):
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj = cleanAll(c1, 'en')
            col_body, pos_body, list2_body, neg_body, neu_body, posi_body, compound_body = cleanAll(c2, 'en')
        else:
            col_body, pos_body, list2_body, neg_body, neu_body, posi_body, compound_body = cleanAll(c2, 'en')
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj = "", "", "", 0, 0, 0, 0
    return pos_body, list2_body, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj, neg_body, neu_body, posi_body, compound_body, lang


# In[177]:
def clean(col):
    #retrieve e-mail address of sender of original e-mail using regex pattern search
    #regex string match consists of: "mailto: " + any string + "@" + any string
    fromEmail = re.search("mailto:" + r'[\w\.-]+@[\w\.-]+', str(col).lower())
    #if e-mail address is found (non-empty string) then transform it to lowercase string object
    if fromEmail:
        fromEmail = str(fromEmail.group(0)).lower()
    else:
        fromEmail = ""
    #retrieve e-mail address of related person (cc of previous e-mail) using regex pattern search
    #regex string match consists of: "cc: " + any string + "@" + any string
    ccEmail = re.search("cc: " + r'[\w\.-]+@[\w\.-]+', str(col).lower())
    # if e-mail address is found (non-empty string) then transform it to lowercase string object
    if ccEmail:
        ccEmail = str(ccEmail.group(0)).lower()
    else:
        ccEmail = ""
    #retrieve e-mail address of sender of current e-mail using regex pattern search
    #regex string match consists of: "to: " + any string + "@" + any string
    toEmail = re.search("to: " + r'([\w\.-]*)' + "; " + r'[\w\.-]+@[\w\.-]+', str(col).lower())
    # if e-mail address is found (non-empty string) then transform it to lowercase string object
    if toEmail:
        toEmail = str(toEmail.group(0)).lower()
        ES = toEmail.split(" ")
        toEmail = ES[len(ES) - 1]
    else:
        toEmail = ""
    # get TO number --> TO+' '+ 8 digiit number  --> find regexpression
    # get sender email --> to:...@ --> find regexpression -->  use set & dict combination

    sign = ""
    #split e-mail into original message (e-mail to current sender) and relevant message (reply from sender), if possible
    colR = col.lower().split('-----original message-----')
    if (len(colR) > 1):
        col = colR[0]
        #encode existence of original message as 1, non-existence as 0
        #value is later stored in resulting csv file
        response = 1
        #original message is stored in 'response_txt' and stored in separate column
        response_txt = colR[1]
    else:
        response = 0
        response_txt = ""
    #filter out general (all language stop-words
    c = wordpunct_tokenize(col)
    c = [i for i in c if i.lower() not in list(stopGen)]
    col = " ".join(c)

    #split e-mail body by polite closing expression for different languages into two texts
    #first text contains main email body with maximum amount of content while second one contains the sender's signature
    colS = col.lower().split('regards')
    if (len(colS) > 1):
        sign = col.lower().split('regards')[1]
        col = str(col.lower().split('regards')[0])
    else:
        colS = col.lower().split('mit freundlichen grüßen')
        if (len(colS) > 1):
            sign = col.lower().split('mit freundlichen grüßen')[1]
            col = col.lower().split('mit freundlichen grüßen')[0]
        else:
            colS = col.lower().split('mit freundlichen grüssen')
            if (len(colS) > 1):
                sign = col.lower().split('mit freundlichen grüssen')[1]
                col = col.lower().split('mit freundlichen grüßen')[0]
            else:
                colS = col.lower().split('cordialement')
                if (len(colS) > 1):
                    sign = col.lower().split('cordialement')[1]
                    col = col.lower().split('cordialement')[0]

    # search for TO ID in e-mail using regex and extract it if possible
    m = re.search("to " + r"\b[\d]{8}\b", str(col).lower())
    if (m is None):
        m = re.search("to" + r"\b[\d]{8}\b", str(col).lower())
    if m:
        m = str(m.group(0)).lower().replace('to', '').replace(' ', '')
    #split e-mail into two texts: subject line and email body
    c2 = str(col).split('----')
    #call language specific pre-processing, depending on if subject exists or not
    if (type(c2) is list and len(c2) > 1 and c2[1] is not ''):
        pos_body, list2_body, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj, neg_body, neu_body, posi_body, compound_body, lang = cleanL(
            c2[0], c2[1])
    elif (type(c2) is list and len(c2) > 1 and c2[1] is ''):
        pos_body, list2_body, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj, neg_body, neu_body, posi_body, compound_body, lang = cleanL(
            "", c2[0])
    elif (type(c2) is list and len(c2) == 1):
        pos_body, list2_body, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj, neg_body, neu_body, posi_body, compound_body, lang = cleanL(
            "", c2[0])
    elif (type(c2) is str):
        pos_body, list2_body, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj, neg_body, neu_body, posi_body, compound_body, lang = cleanL(
            "", c2)
    return pos_body, list2_body, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj, neg_body, neu_body, posi_body, compound_body, lang, response, response_txt, sign, m, fromEmail, ccEmail, toEmail

    print('---------')


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    # Path of the source and destination file
    raw_file = os.path.join(path, 'email_lear_start1499000000000end1502000000000.json')
    src_file = os.path.join(path, 'email_src.csv')
    dst_file = os.path.join(path, 'email_train_lear_start1499000000000end1502000000000_new.csv')
    #    dst_file_orig = os.path.join(path,'email_train_orig.csv')

    with open(raw_file, 'r', encoding='UTF8') as f:
        data = json.load(f)
    df = pandas.read_json(raw_file, encoding='utf-8')
    # transform json to csv file and write it to src_file file path
    df.to_csv(src_file, sep=';', quoting=csv.QUOTE_ALL, encoding='utf-8')
    # fill all the cols here that have to be anonymized

    mail_body_column = {'description': {}}
    #define which columns need to be hashed
    hash_only_columns = {'assignedGroup': {},
                         'assigneeUsername': {},
                         'projectKey': {},
                         'reporterUsername': {},
                         'responsibleCompanyKey': {},
                         'responsibleCompanyKeyManual': {},
                         'responsibleParty': {}}
    # Insert salt value here, please keep salt secure and confidential
    salt = "insert_salt_here"
    #read transformed csv-file
    with open(src_file, 'r', encoding='utf-8') as csvfile:
        # The csv file gets assigned to reader
        reader = csv.DictReader(csvfile, delimiter=';', quotechar='"')
        # Open a new csv file in which we will write the update content
        with open(dst_file, 'w', encoding="utf-8") as destination:
            # reset reader to first line
            csvfile.seek(0)
            # Define output columns
            reader.fieldnames = "", "actionCodes", "assignedGroup", "assigneeUsername", "countComments", "countIncomingMails", "countOutgoingMails", "created", "creationType", "description", "issueId", "issueKey", "issueTypeCode", "issueTypeId", "issueTypeName", "lastStatusTransistion", "loadId", "loggedWork", "priorityName", "projectKey", "reporterUsername", "responsibleCompanyKey", "responsibleCompanyKeyManual", "responsibleParty", "slas", "statusName", "transportOrderId", "updated", "slaGoal", "slaMet", "slaName", "slaStatus", "slaTimeLeft", "txt_subj", "txt_main", "language", "pos_main", "pos_subj", "neg_subj", "neu_subj", "posi_subj", "compound_subj", "neg_body", "neu_body", "posi_body", "compound_body", "ID", "lang", "response", "response_txt", "sign", "to", "fromEmail", "ccEmail", "toEmail"
            #create writer object to save each row to the dest_csv file
            writer = csv.DictWriter(destination, delimiter=';', fieldnames=reader.fieldnames, lineterminator='\n',
                                    quotechar='"', quoting=csv.QUOTE_ALL)
            # Write fieldnames of reader to header of new csv
            writer.writeheader()
            print("Successfully finished row 1 (header)")
            # Skip header line and read second line of source csv
            next(reader)
            # Index points to the current row of the iterator
            index = 2
            # Loop over every row from line 2 to end of file
            for row in reader:
                # temp stores the text of column description for the current row in a map
                temp = row
                # skip emails without text
                if (temp['description'] is ''):
                    continue
                # Main routine for the pre-processing of the mail body
                # Generates
                try:
                    #example for one row
                    # 'Description': 'TO 10237544 \n----\nHello 4flow, DHL team,\n\ncould you please... \n',
                    # temp[col] is the mapped value of a specific column (e.g. a string)
                    pos_body, list2_body, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj, neg_body, neu_body, posi_body, compound_body, lang, response, response_txt, sign, to, fromEmail, ccEmail, toEmail = clean(
                        temp["description"])
                    #store additionally extracted information like TO ID, sentiment, language and original text
                    temp['to'] = to
                    temp['lang'] = lang
                    temp["posi_subj"] = posi_subj
                    temp["neu_subj"] = neu_subj
                    temp["neg_subj"] = neg_subj
                    temp["compound_subj"] = compound_subj
                    temp["posi_body"] = posi_body
                    temp["neu_body"] = neu_body
                    temp["neg_body"] = neg_body
                    temp["compound_body"] = compound_body
                    temp["response"] = response
                    temp["response_txt"] = response_txt
                except:
                    # Error handling
                    # skip rows with random errors --> e.g. formatting, escape characters ...
                    print("Temp Error in row " + str(index))
                    continue
                try:
                    #####Hash all personal information in email (on a column basis)
                    # Hash subject POS
                    pos_subj_hash = ""
                    for i in pos_subj:
                        pos_subj_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["pos_subj"] = hashing_trick(pos_subj_hash, 1000000, hash_function='md5')
                    # Hash Mail subject
                    txt_subj_hash = ""
                    if (type(list2_subj) is str):
                        list2_subj = wordpunct_tokenize(list2_subj)
                    for i in list2_subj:
                        txt_subj_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["txt_subj"] = hashing_trick(txt_subj_hash, 1000000, hash_function='md5')
                    # Hash main POS
                    pos_main_hash = ""
                    for i in pos_body:
                        pos_main_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["pos_main"] = hashing_trick(pos_main_hash, 1000000, hash_function='md5')
                    # Hash Mail body
                    txt_main_hash = ""
                    if (type(list2_body) is str):
                        list2_body = wordpunct_tokenize(list2_body)
                    for i in list2_body:
                        txt_main_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["txt_main"] = hashing_trick(txt_main_hash, 1000000, hash_function='md5')
                    # Hash original email if current one is a response
                    txt_resp_hash = ""
                    if (type(response_txt) is str):
                        response_txt = wordpunct_tokenize(response_txt)
                    for i in response_txt:
                        if i not in stopGen:
                            txt_resp_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["response_txt"] = hashing_trick(txt_resp_hash, 1000000, hash_function='md5')
                    # Hash signature of e-mail if it exists
                    sign_hash = ""
                    if (type(sign) is str):
                        sign = wordpunct_tokenize(sign)
                    for i in sign:
                        sign_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["sign"] = hashing_trick(sign_hash, 1000000, hash_function='md5')
                    #Hash extracted email addresses: original sender, cc and sender of current e-mail
                    temp['fromEmail'] = hashing_trick(
                        hashlib.sha512((salt + fromEmail).encode('utf-8')).hexdigest() + " ", 1000000,
                        hash_function='md5')
                    temp['ccEmail'] = hashing_trick(hashlib.sha512((salt + ccEmail).encode('utf-8')).hexdigest() + " ",
                                                    1000000, hash_function='md5')
                    temp['toEmail'] = hashing_trick(hashlib.sha512((salt + toEmail).encode('utf-8')).hexdigest() + " ",
                                                    1000000, hash_function='md5')
                except:
                    # Error handling
                    print("Error in row " + str(index))
                    raise
                # For every row, iterate over columns defined in hash_only_columns
                # Columns not assigned in hash_only_columns will simply be transported
                # from source to destination w/o changing anything
                for col in hash_only_columns:
                    try:
                        col_hash = ""
                        for i in temp[col].split():
                            col_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                        temp[col] = hashing_trick(col_hash, 1000000, hash_function='md5')
                        # temp[col] = ' '.join(map(str, hashing_trick(temp[col], 1000000, hash_function='md5')))
                    except:
                        # Error handling
                        print("Error in row " + str(index), temp)
                        raise
                # remove the column description, this column has been transformed into
                # "sub_subj", "pol_subj", "pos_subj", "txt_subj", "sub_main", "pol_main","pos_main", "txt_main"

                # Assign a incremental ID to the email to later return the class and the ID
                temp['ID'] = str(index - 1)
                #set all unimportant/later available columns to None to reduce file size
                temp['actionCodes'] = None
                temp['assigneeUsername'] = None
                temp['countComments'] = None
                temp['countIncomingMails'] = None
                temp['description'] = None
                temp['countOutgoingMails'] = None
                temp['lastStatusTransistion'] = None
                temp['loggedWork'] = None
                temp['priorityName'] = None
                temp['reporterUsername'] = None
                temp['responsibleCompanyKey'] = None
                temp['responsibleCompanyKeyManual'] = None
                temp['responsibleParty'] = None
                temp['statusName'] = None
                temp['transportOrderId'] = None
                temp['updated'] = None
                temp['slaGoal'] = None
                temp['slaMet'] = None
                temp['slaName'] = None
                temp['slas'] = None
                temp['slaStatus'] = None
                temp['slaTimeLeft'] = None

                # print("Successfully finished row " + str(index))
                index += 1
                # "", "actionCodes", "assignedGroup","assigneeUsername","countComments", "countIncomingMails","countOutgoingMails","created","creationType","description","issueId","issueKey","issueTypeCode","issueTypeId","issueTypeName","lastStatusTransistion","loadId","loggedWork","priorityName","projectKey","reporterUsername","responsibleCompanyKey","responsibleCompanyKeyManual","responsibleParty","slas","statusName","transportOrderId","updated","slaGoal","slaMet","slaName","slaStatus","slaTimeLeft", "txt_subj","txt_main", "language", "pos_main", "pos_subj", "neg_subj", "neu_subj", "posi_subj", "compound_subj",                 "neg_body", "neu_body", "posi_body", "compound_body", "ID", "lang", "response", "response_txt", "sign", "to"

                # Row is written including the changes made
                writer.writerow(temp)

