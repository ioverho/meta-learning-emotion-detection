import re

import emoji
import spacy
from transformers import AutoTokenizer

nlp = spacy.load('en_core_web_sm')
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def manual_tokenizer(text, bert_vocab=None, OOV_cutoff=0.5):

    # Ensure proper encoding
    try:
        txt = text.encode('latin1').decode('utf8')
    except UnicodeDecodeError:
        print('Removed sentence for bad encoding.')
        return None

    # Remove bad character references
    txt = re.sub(r'\&\w+;', '', txt)

    norm_sent = ""
    OOV_count, total = 0, 0
    for i, t in enumerate(nlp.tokenizer(txt)):

        norm_token = t.norm_

        if norm_token[0] == "'":
            # Just knock off the apostrophe for contractions
            norm_token = norm_token[1:]
        elif t.like_url:
            # Special token for URLs
            norm_token = "HTTPURL"
        elif t.like_email:
            # Remove emails
            continue
        elif "@" in norm_token:
            # Special token for handles
            # Also works for retweets
            norm_token = re.sub(r"\@\w+\:*", "@USER", norm_token)
        elif (not t.is_alpha) and (not t.is_punct):
            # Properly space all punctuation
            # When attached to a word
            norm_token = re.sub(r"(\w)([^a-zA-Z0-9\s]+)", r"\1 \2 ", norm_token)

        if bert_vocab != None:
            # Check if a lot of tokens are OOV for BERT
            # Hopefully avoids non-English tweets
            if (not (norm_token in bert_vocab)):
                OOV_count += 1
            total += 1

        norm_sent += norm_token + " "

    if bert_vocab != None:
        if OOV_count / total >= OOV_cutoff:
            print('Removed sentence for too many OOV terms.')
            return None
    # Remove double whitespaces and trailing
    norm_sent = re.sub('\s{2,}', ' ', norm_sent)[:-1]

    # Convert emojis to tokens
    norm_sent = emoji.demojize(norm_sent)

    return norm_sent
