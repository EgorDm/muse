import re
import urllib.error
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup


def fix_string_path(s):
    return re.sub('[^\w\-_\. ]', '_', s)


def get_html(url):
    response = urllib.request.urlopen(url)
    html_lyrics = response.read()
    return BeautifulSoup(html_lyrics, 'html5lib')
