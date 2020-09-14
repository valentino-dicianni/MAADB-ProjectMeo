import csv
import os
from config import config
import psycopg2
import re
import nltk

from wordcloud import WordCloud
import math
from tqdm import tqdm
from pymongo import MongoClient, errors
import time
from bson.code import Code

from utils.slang import slang_words
from utils.emoji import emojiNeg
from utils.emoji import emojiPos
from utils.emoji import othersEmoji
from utils.emoji import negemoticons
from utils.emoji import posemoticons
from utils.punctuation import punctuation

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import TweetTokenizer
import demoji

demoji.download_codes()
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

res_base_path = "Materiale/Risorse_lessicali/"
tweets_path = "Materiale/Twitter_messaggi/"
#feeling_list = ['Trust']
feeling_list = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']

tags = {}
emoji = {}
tweets = {}
words = {}
stats = {}
resources = {}
afinnScore = {}
anewScore = {}
dalScore = {}


# final structure: {Emotion: {word: {count, NRC, EmoSN, sentisensem, afinn, anew, del},...}
def create_resources_dictionary(feeling):
    list_words = {}
    create_afinn_anew_dal()
    for file_feeling in os.listdir(res_base_path + feeling):
        if not file_feeling.startswith('.'):
            with open(res_base_path + feeling + "/" + file_feeling, 'r') as file:
                t = file_feeling.split('_')[0]
                lines = file.readlines()
                for line in lines:
                    if '_' not in line:
                        key = line.replace('\n', "")

                        if key not in list_words:
                            list_words[key] = {'afinn': afinnScore.get(key, 0),
                                               'anew': anewScore.get(key, 0),
                                               'dal': dalScore.get(key, 0), 'count': 1, t: 1}
                        else:
                            list_words[key].update({t: 1})
                            list_words[key]['count'] += 1
    return list_words


def create_afinn_anew_dal():
    # Affin
    tsv_file = open(res_base_path + "ConScore" + "/afinn.tsv", 'r')
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
        afinnScore[row[0]] = row[1]

    # ANEW
    tsv_file = open(res_base_path + "ConScore" + "/anewAro_tab.tsv", 'r')
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
        anewScore[row[0]] = row[1]

    # DAL
    tsv_file = open(res_base_path + "ConScore" + "/Dal_Activ.csv", 'r')
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
        dalScore[row[0]] = row[1]


def create_resources_sql():
    # Connect to the PostgreSQL database server
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()
        if resources:
            for feeling, w_list in resources.items():
                cur.execute(f'DROP TABLE IF EXISTS resources_{feeling} CASCADE')
                cur.execute(
                    f'CREATE TABLE resources_{feeling} ('
                    f'id SERIAL PRIMARY KEY,'
                    f'word varchar(255), '
                    f'w_count integer , '
                    f'nrc integer , '
                    f'emosn integer, '
                    f'sentisense integer, '
                    f'afinn real , '
                    f'anew real, '
                    f'del real)'

                )
                for key, value in w_list.items():
                    cur.execute(
                        f"INSERT INTO resources_{feeling}(word, w_count, nrc, EmoSN, sentisense, afinn, anew, del)"
                        f"VALUES('{key}', {value['count']}, {value.get('NRC', 0)}, {value.get('EmoSN', 0)},"
                        f"{value.get('sentisense', 0)}, {value.get('afinn', 0)},"
                        f"{value.get('anew', 0)}, {value.get('del', 0)})"
                    )

        # close the communication with the PostgreSQL
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Database ERROR: ", error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


def create_twitter_sql():
    # Connect to the PostgreSQL database server
    conn = None
    print('Creating tweets tables...')
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()
        if len(tweets) > 0:
            for feeling, w_list in tweets.items():
                cur.execute(f"DROP TABLE IF EXISTS tweet_{feeling} CASCADE")
                cur.execute(
                    f'CREATE TABLE tweet_{feeling} ('
                    f'id SERIAL PRIMARY KEY,'
                    f'word varchar(255) UNIQUE NOT NULL, '
                    f'w_count integer ,'
                    f'resources_id integer,'
                    f'CONSTRAINT fk_resources FOREIGN KEY(resources_id) REFERENCES resources_{feeling}(id));'
                )
                for key, value in w_list.items():
                    key = key.replace("\'", "")
                    cur.execute(
                        f'INSERT INTO tweet_{feeling}(word, w_count, resources_id)'
                        f'VALUES(\'{key}\', 1,  (SELECT id FROM resources_{feeling} WHERE word = \'{key}\') )'
                        f'ON  CONFLICT (word) '
                        f'DO UPDATE  SET w_count = tweet_{feeling}.w_count + 1'
                    )

        # close the communication with the PostgreSQL
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Database ERROR: ", error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


def create_twitter_mongo(feeling):
    """
    Popola la collezione indicata da feeling con un oggetto:
    {
        feeling: <feeling>
        name: <word>
    }
    """
    try:
        client = MongoClient(host='localhost', port=27017,
                             serverSelectionTimeoutMS=3000)

        db = client.maadbProject_id
        collection = db["words"]
        listBulk = []
        for w in words[feeling]:
            entry = {"feeling": feeling, "name": w}
            listBulk.append(entry)
        collection.insert_many(listBulk)
        print("Sharding collection crated...")

    except errors.ServerSelectionTimeoutError as err:
        print("pymongo ERROR:", err)



def get_resources_sql():
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        for feeling in feeling_list:
            cur.execute(f"SELECT * FROM {feeling}")
            print(cur.fetchall())

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Database ERROR: ", error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


def create_resources_mongo():
    try:
        client = MongoClient(host='localhost', port=27017,
                             serverSelectionTimeoutMS=3000)
        if resources:
            db = client.maadbProject_id

            db.drop_collection("resources_mongo")
            collection = db["resources_mongo"]
            collection.insert_one(resources)

    except errors.ServerSelectionTimeoutError as err:
        # catch pymongo.errors.ServerSelectionTimeoutError
        print("pymongo ERROR:", err)


def analyze_tweets(feeling):
    tag_list = {}
    emoji_list = {}
    words[feeling] = []
    lemmatized_tweets = {}
    tk = TweetTokenizer()
    lemmatizer = WordNetLemmatizer()

    # with open(tweets_path + "dataset_dt_" + feeling.lower() + "_test_60k.txt", 'r', encoding="utf8") as file:
    with open(tweets_path + "dataset_dt_" + feeling.lower() + "_60k.txt", 'r', encoding="utf8") as file:
        lines = file.readlines()
        print("Start Analyzing tweet. Feeling: ", feeling)
        for line in tqdm(lines):

            # build map for hashtag and remove from line
            if '#' in line:
                hashtags = re.findall(r"#(\w+)", line)
                for htag in hashtags:
                    tag_list[htag] = tag_list.get(htag, 0) + 1
                    line = line.replace('#' + htag, '').replace('#', '')
                    words[feeling].append(htag)

            # find, store and replace emoji from line
            ejs = [demoji.replace_with_desc(em, ":") for em in emojiNeg + emojiPos + othersEmoji + negemoticons +
                   posemoticons if (em in line)]

            for e in ejs:
                emoji_list[e] = emoji_list.get(e, 0) + 1
                line = line.replace(e, '')
                words[feeling].append(e)

            # replace slang from sentences
            slang_list = [s for s in slang_words.keys() if (s in line.split())]
            for s in slang_list:
                line = line.replace(s, slang_words[s])

            # remove punctuation
            punct_list = [p for p in punctuation if (p in line)]
            for p in punct_list:
                line = line.replace(p, '')

            # remove USERNAME and URL
            line = line.replace('USERNAME', '').replace('URL', '').lower()

            # remove citations
            citations = re.findall(r"@(\w+)", line)
            for cit in citations:
                line = line.replace('@' + cit, '').replace('@', '')

            # tokenize sentence
            word_tokens = tk.tokenize(line)
            pos_line = pos_tagging(word_tokens)

            # lemmatize nouns, adjective, verbs
            for pos in pos_line:
                if pos[1] in ['j', 'n', 'v']:
                    lemm_w = lemmatizer.lemmatize(pos[0], pos[1])
                    words[feeling].append(lemm_w)
                    lemmatized_tweets[lemm_w] = lemmatized_tweets.get(lemm_w, 0) + 1

        # display word cloud
        wordcloud_words = WordCloud(max_font_size=50, background_color="white", width=800,
                                    height=400).generate_from_frequencies(
            lemmatized_tweets)

        wordcloud_emoji = WordCloud(max_font_size=50, background_color="white", width=800,
                                    height=400).generate_from_frequencies(
            emoji_list)
        wordcloud_tag = WordCloud(max_font_size=50, background_color="white", width=800,
                                  height=400).generate_from_frequencies(
            tag_list)
        wordcloud_words.to_file("img/cloud_words_" + feeling + ".png")
        wordcloud_emoji.to_file("img/cloud_emoji_" + feeling + ".png")
        wordcloud_tag.to_file("img/cloud_tag_" + feeling + ".png")

    # Store emoji, tags and tweets for feeling
    emoji[feeling] = emoji_list
    tweets[feeling] = lemmatized_tweets
    tags[feeling] = tag_list


def pos_tagging(word_tokens):
    # remove stop words
    sw = set(stopwords.words('english'))
    stop_in_line = [w for w in word_tokens if (w in sw)]

    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    # Map POS tag to first character lemmatize() accepts
    res = []
    ts = nltk.pos_tag(word_tokens)
    for t in ts:
        if t[0] not in stop_in_line:
            # return q if is not ADJ NOUN VERB ADV
            res.append((t[0], tag_dict.get(str(t[1][0]).upper(), 'q')))
    return res


def create_lexical_res():
    for feeling in tqdm(feeling_list):
        word_list = create_resources_dictionary(feeling)
        resources[feeling] = word_list
    # create_resources_sql()
    # create_resources_mongo()
    print("Resources dictionary built...")


def normalize_double(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


def get_stats():
    out_handle = open('out/stats.txt', 'a')
    if resources and tweets:
        for feeling, w_list in tweets.items():
            stat_list = {'found': [], 'not_found': []}
            for word, count in w_list.items():
                if word in resources[feeling]:
                    stat_list['found'].append(word)
                else:
                    stat_list['not_found'].append(word)

            stats[feeling] = {'count_found': len(stat_list['found']), 'count_not_found': len(stat_list['not_found']),
                              'perc_pr_lex': normalize_double(len(stat_list['found']) / len(resources[feeling]), 2),
                              'perc_pr_tw': normalize_double(len(stat_list['found']) / len(tweets[feeling]), 2)}
            # Write output file
            out_handle.write(f"\n############# {feeling} #############\n")
            out_handle.write(f"Words Found: {stats[feeling]['count_found']}\n")
            out_handle.write(f"Words Not Found: {stats[feeling]['count_not_found']}\n")
            out_handle.write(f"% Presence lexical resources : {stats[feeling]['perc_pr_lex']}\n")
            out_handle.write(f"% Presence Twitter : {stats[feeling]['perc_pr_tw']}\n")
            out_handle.write(f"#####################################\n")

        return stats


def execute_map_reduce(feeling):
    try:
        client = MongoClient(host='localhost', port=27017,
                             serverSelectionTimeoutMS=3000)

        mongo_map = Code("function() { "
                         "  emit(this.name, 1); "
                         "}")
        mongo_reduce = Code("function (key, values) { "
                            "   return Array.sum(values); "
                            "}")

        db = client.maadbProject_id
        db.drop_collection(f"{feeling}_result")
        start_t = time.time()

        result = db["words"].map_reduce(mongo_map,
                                        mongo_reduce,
                                        f"{feeling}_result",
                                        query={"feeling": feeling},
                                        full_response=True)
        print(
            f"Risultato {feeling} con tabella sharding \'ok\': {result.get('ok')} in {(time.time() - start_t)} seconds")

    except errors.ServerSelectionTimeoutError as err:
        print("pymongo ERROR:", err)


# --------------------------------- #
# create_lexical_res()
# start_time = time.time()
# for f in feeling_list:
#     print("#######################")
#     print(f'Executing feeling: {f}')
#     analyze_tweets(f)
# print("--- Analyzing tweets took: %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# for f in feeling_list:
#     create_twitter_mongo(f)
# print("--- Creating twitter mongo: %s seconds ---" % (time.time() - start_time))


start_time = time.time()
for f in feeling_list:
    execute_map_reduce(f)
print("--- Executing Map-Reduce: %s seconds ---" % (time.time() - start_time))

# Analisi tempi SQL Postgres
# create_lexical_res()
# start_time = time.time()
# for f in feeling_list:
#     print("#######################")
#     print(f'Executing feeling: {f}')
#     analyze_tweets(f)
# print("--- Analyzing tweets took: %s seconds ---" % (time.time() - start_time))
#
# start_time = time.time()
# create_twitter_sql()
# print("--- Inserting tweets took: %s seconds ---" % (time.time() - start_time))
#
# get_stats()
# print(stats)
