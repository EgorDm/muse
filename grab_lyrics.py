import os

from tools import clean
from tools.grabbers.models import *
from tools.grabbers import azlyrics, darklyrics
from utils.text import *

artist = input('Artist: ')
source = eval(input('Source: (azlyrics=0, darklyrics=1)'))

songs_dir = 'data/lyrics/' + fix_string_path(artist)

if not os.path.exists(songs_dir):
    os.makedirs(songs_dir)

existing = [file[:(len(file) - 4)] for file in os.listdir(songs_dir)]


def process_song(song):
    song_path = '{}/{}.txt'.format(songs_dir, song.title)
    lyrics = song.get_lyrics()
    lyrics = clean.clean_lines(lyrics)
    print('Writing song {} to {}'.format(song.title, song_path))
    with open(song_path, 'w', encoding='utf8') as sf:
        sf.write(''.join(lyrics))


def process_existing(title):
    print('Song {} already exists'.format(title))
    song_path = '{}/{}.txt'.format(songs_dir, title)
    with open(song_path, encoding='utf8') as f:
        lyrics = f.read()
    process_song(Song(title, [lyrics]))


def grab_from_azlyrics(artist):
    for url in azlyrics.get_artist_song_urls(artist):
        song_title = fix_string_path(url[0])
        if song_title in existing:
            process_existing(song_title)
        else:
            print('Downloading song {}'.format(song_title))
            song = azlyrics.get_song_lyrics(url[1], True)
            process_song(song)


def grab_from_darklyrics(artist):
    for url in darklyrics.get_album_links(artist):
        print('Downloading album {}'.format(url))
        songs = darklyrics.get_lyrics_by_album(url)
        for song in songs:
            song.title = fix_string_path(song.title)
            process_song(song)


if source == 0:
    for a in artist.split(','):
        grab_from_azlyrics(a)
else:
    for a in artist.split(','):
        grab_from_darklyrics(a)