from tools.grabbers.models import *
from utils.text import get_html, fix_string_path
from bs4 import NavigableString
import time

BASE_URL = 'http://www.darklyrics.com'
ANTI_THROTTLE_DELAY = 20


def get_album_links(artist):
    url = '{}/{}/{}.html'.format(BASE_URL, artist[0], artist)
    html = get_html(url)
    albums = html.select('div.album')
    links = []
    for album in albums:
        for line in album.contents:
            if line.name == 'a' and 'lyrics/' in line['href']:
                links.append(BASE_URL + line['href'][2:])
                break
    return links


def get_lyrics_by_album_name(artist, album):
    url = '{}/lyrics/{}/{}.html'.format(BASE_URL, artist, album)
    return get_lyrics_by_album(url)


def get_lyrics_by_album(url, throttle=True):
    if throttle: time.sleep(ANTI_THROTTLE_DELAY)
    html = get_html(url)
    lyrics_div = html.select('div.lyrics')[0]
    songs = []
    for line in lyrics_div.contents:
        if line.name == 'h3':
            title = line.text.split(' ', 1)[1]
            songs.append(Song(title, []))
            continue
        if len(songs) == 0: continue

        if isinstance(line, NavigableString):
            text = line.string
        else:
            text = line.text

        songs[-1].lyrics.append(text)
    return songs
