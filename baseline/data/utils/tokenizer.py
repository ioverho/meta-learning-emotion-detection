import re

import emoji
import spacy
from spacy.symbols import ORTH

nlp = spacy.load('en_core_web_sm')

nlp.tokenizer.add_special_case("<NAME>", [{ORTH: "<NAME>"}])
nlp.tokenizer.add_special_case("<URL>", [{ORTH: "<URL>"}])
nlp.tokenizer.add_special_case("<USER>", [{ORTH: "<USER>"}])

def specials():
    return ["<NAME>", "<URL>", "<USER>"]

def manual_tokenizer(batch, bert_vocab=None, OOV_cutoff=0.5, verbose=False, norm=False, remove_emojis=False, lower=False):
    """Takes a string and converts it to less noisy input.

    Args:
        batch: batch of sentences
        bert_vocab (encoder vocab, [optional): vocab to check. Defaults to None.
        OOV_cutoff (float, optional): cutoff point for OOV terms. Defaults to 0.5.
        verbose (bool, optional): whether or not to print when removing sentences. Defaults to False.
        norm (bool, optional): whether or not to normalize tokens according to Spacy. Defaults to False.
        remove_emojis (bool, optional): whether or not to keep emojis. Defaults to False.

    Returns:
        [type]: [description]
    """

    text = batch['text']

    # Remove bad character references
    txt = re.sub(r"\&[a-zA-Z0-9]+;*", "", text)

    # Remove #SemST from SSEC dataset
    txt = re.sub(r"#SemST", "", txt)

    # Remove all URLs
    txt = re.sub(r"(http|www)\S+", "", txt)

    # Convert reddit usernames to <USER> as well
    txt = re.sub(r"u\/[a-zA-Z0-9]+", "<USER>", txt)

    # Convert goemotion names to <NAME>
    txt = re.sub(r"\[NAME\]", "<NAME>", txt)

    norm_sent = ""
    OOV_count, total = 0, 0
    for t in nlp.tokenizer(txt):

        if norm:
            norm_token = t.norm_
        elif lower:
            norm_token = t.lower_
        else:
            norm_token = t.text

        if norm_token[0] == "'":
            # Just knock off the apostrophe for contractions
            norm_token = norm_token[1:]
        elif t.like_url:
            # Special token for URLs
            norm_token = "<URL>"
        elif t.like_email:
            # Remove emails
            continue
        elif "@" in norm_token:
            # Special token for handles
            # Also works for retweets
            norm_token = re.sub(r"\@\w+\:*", "<USER>", norm_token)
        #elif (not t.is_alpha) and (not t.is_punct):
        #    # Properly space all punctuation
        #    # When attached to a word
        #    norm_token = re.sub(r"(\w)([^a-zA-Z0-9\s]+)", r"\1 \2 ", norm_token)

        if bert_vocab != None:
            # Check if a lot of tokens are OOV for BERT
            # Hopefully avoids non-English tweets
            if (not (norm_token in bert_vocab)):
                OOV_count += 1
        total += 1

        norm_sent += norm_token + " "

    norm_sent = re.sub(r"n't", "not", norm_sent)

    if total > 0 and bert_vocab != None:
        if OOV_count / total >= OOV_cutoff:
            if verbose:
                print('Removed sentence for too many OOV terms.')
                print(f'{norm_sent}')
            return None
    # Remove double whitespaces and trailing
    norm_sent = re.sub('\s{2,}', ' ', norm_sent)[:-1]

    # Convert emojis to tokens
    norm_sent = emoji.demojize(norm_sent)
    if remove_emojis:
        norm_sent = re.sub(r":[^:]+:", "", norm_sent)

    batch['text'] = norm_sent
    return batch
