import re
from nltk.corpus import stopwords
from nltk.stem.porter import *

LONGEST_WORD = len("pneumonoultramicroscopicsilicovolcanoconiosis")
REDDIT_URL_REGEX = r"""\[([^\]]*?)](\([^\s]*?\))"""
BLOCKQUOTE_REGEX = r"""^&gt;(.*?)$"""
QUOTE = r"""&gt;"""
URL_REGEX = r"""(http://[^\s)]*)"""
URL_REGEX_S = r"""(https://[^\s)]*)"""
PUNCT_LIST = r"""!#$%&()*+,-./:;<=>?@[\]^_`{|}~"""
DELTA_REGEX = r"^delta(.*?)$"
PIPE_PUNCT = r"|"
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()
STEMMED_STOPWORDS = {stemmer.stem(sw) for sw in STOPWORDS}

def process_comment_inst(inp: str) -> str:
    """
    :return: str comment without special characters and delta
    """
    inp = inp.strip()
    if 'Hello, users of CMV' in inp or 'This is a footnote' in inp:
        return ''
    #out = re.sub(r'Hello, users of CMV.*', '', out)
    #out = re.sub(r'This is a footnote.*','',out)
    inp = re.sub(REDDIT_URL_REGEX, r"\1", inp, flags=re.MULTILINE)
    inp = re.sub(URL_REGEX, " @URL@ ", inp)
    inp = re.sub(URL_REGEX_S, " @URL@ ", inp)
    inp = re.sub(BLOCKQUOTE_REGEX, r'“\1”', inp, flags=re.MULTILINE)
    inp = inp.replace("&.amp;#8710;", "∆")
    inp = inp.replace("&amp;#8710;", "∆")
    inp = inp.replace("&amp", "&")
    inp = inp.replace("\’", "\'")
    inp = inp.replace("\“", "\"")
    inp = inp.replace("δ", "∆")
    inp = inp.replace("Δ", "∆")
    inp = inp.replace("&;#8710;", "∆")
    inp = inp.replace("∆", " delta ")
    inp = inp.replace("/u/", "")
    inp = inp.replace("/r/", "")
    inp = inp.replace("r/", "")
    inp = inp.replace("!delta", " delta ")
    inp = re.sub(r'.*EDIT\*\*:.*', '', inp)
    inp = re.sub(r'Edit(.*?):.*', '', inp)
    inp = re.sub(r'EDIT(.*?):.*', '', inp)
    inp = re.sub(r"^EDIT(.*?)$",'',inp)
    patterns =[ r'\t', r'\r', r'\s\s+', r'__+', r'\*\*+', r'——+', r'--+']
    for pat in patterns:
        inp = replace_pat(pat,inp)
    return inp


def process_comment(inp: str) -> str:
    out = ''
    for curr in inp.split('\n'):
        if len(curr) > 0:
            out += ' ' + process_comment_inst(curr)
    out = out.strip()
    out = re.sub(DELTA_REGEX, r"\1", out)
    out = re.sub(r"\s\s+", " ",out)
    out = out.strip()
    return out


def replace_pat(pat: str, x: str) -> str:
    return re.sub(pat, ' ', x)


def is_stop(tok, stem: bool = True) -> bool:
    stopwords = STEMMED_STOPWORDS if stem else STOPWORDS
    return tok.lower() in stopwords
