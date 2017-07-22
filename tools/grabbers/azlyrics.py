import time
import re
from tools.grabbers.models import Song

import utils.text
from utils.text import get_html

BASE_URL = 'http://azlyrics.com'
ANTI_THROTTLE_DELAY = 20


def get_artists_songs(artist):
    ret = []
    for url in get_artist_song_urls(artist):
        ret.append(get_song_lyrics(url, True))
    return ret


def get_artist_song_urls(artist):
    url = '{0}/{1}/{2}.html'.format(BASE_URL, artist[0], artist)
    html = get_html(url)
    ret = []
    for song in html.findAll('a', {'target': '_blank'}):
        if 'lyrics/' in song['href']:
            ret.append([song.string, BASE_URL + song['href'][1:]])
    return ret


def get_song_lyrics_by_name(artist, song):
    url = '{0}/lyrics/{1}/{2}.html'.format(BASE_URL, artist, song)
    return get_song_lyrics(url)


def get_song_lyrics(url, throttle=False):
    if (throttle): time.sleep(ANTI_THROTTLE_DELAY)
    html = get_html(url)
    title = html.find_all("b", attrs={"class": None, "id": None})[1].getText().replace('"', '')
    print('Title: ' + title)
    lyrics = html.find_all("div", attrs={"class": None, "id": None})
    lyrics = [re.sub("[\(\[].*?[\)\]]", '', x.getText()) for x in lyrics]
    # for x in lyrics:
    #     print(x, end="\n\n")
    return Song(title, lyrics)
