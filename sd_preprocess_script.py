import pandas as pd
import gensim, spacy
import gensim.corpora as corpora
import nltk
import sys


def make_bigrams(bigram_model, texts):
    return [bigram_model[doc] for doc in texts]

def make_trigrams(trigram_model, bigram_model, texts):
    return [trigram_model[bigram_model[doc]] for doc in texts]

def lemmatization(nlp_model, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    allowed_postags_set = set(allowed_postags)
    for sent in texts:
        doc = nlp_model(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags_set])
    return texts_out


def main(FILENAME, OUTPUTPATH):
  df_train = pd.read_csv(FILENAME)
  #nltk.download()
  df_train['tokens'] = df_train.apply(lambda row: nltk.word_tokenize(row['title_article_cleaned']), axis = 1)
  # MAKE BIGRAMS
  token_words = df_train['tokens']
  bigram = gensim.models.Phrases(token_words, min_count=5, threshold=100)
  bigram_mod = gensim.models.phrases.Phraser(bigram)
  token_words_bigrams = make_bigrams(bigram_mod, token_words)

  # MAKE TRIGRAMS
  trigram = gensim.models.Phrases(bigram[token_words], threshold=100)
  trigram_mod = gensim.models.phrases.Phraser(trigram)
  token_words_trigrams = make_trigrams(trigram_mod, bigram_mod, token_words)

  # lemmatization
  nlp = spacy.load("en_core_web_sm", disable=['parser','ner'])
  token_words_trigrams_lemm = lemmatization(nlp, token_words_trigrams,
    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
  id2word = gensim.corpora.Dictionary(token_words_trigrams_lemm)
  corpus = [id2word.doc2bow(text) for text in token_words_trigrams_lemm]

  id2word.save(OUTPUTPATH+'/id2word')
  corpora.MmCorpus.serialize(OUTPUTPATH+'/corpus', corpus)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
