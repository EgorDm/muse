import os

from tools import clean
from tools.grabbers.models import *
from tools.grabbers import azlyrics, darklyrics
from utils.text import *

artists = input('Artist: ').split(',')
source = eval(input('Source: (azlyrics=0, darklyrics=1)'))

SONGS_DIR = 'data/lyrics/'


class Grabber():
    def __init__(self, artist):
        self.artist = artist
        self.dir = SONGS_DIR + artist
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.existing = [file[:(len(file) - 4)] for file in os.listdir(self.dir)]

    def grab(self, source):
        if source == 0:
            self._grab_from_azlyrics()
        else:
            self._grab_from_darklyrics()

    def _process_existing(self, title):
        print('Song {} already exists'.format(title))
        song_path = '{}/{}.txt'.format(self.dir, title)
        with open(song_path, encoding='utf8') as f:
            lyrics = f.read()
        self._process_song(Song(title, [lyrics]))

    def _process_song(self, song):
        song_path = '{}/{}.txt'.format(self.dir, song.title)
        lyrics = song.get_lyrics()
        lyrics = clean.clean_lines(lyrics)
        print('Writing song {} to {}'.format(song.title, song_path))
        with open(song_path, 'w', encoding='utf8') as sf:
            sf.write(''.join(lyrics))

    def _grab_from_azlyrics(self):
        for url in azlyrics.get_artist_song_urls(self.artist):
            song_title = fix_string_path(url[0])
            if song_title in self.existing:
                self._process_existing(song_title)
            else:
                print('Downloading song {}'.format(song_title))
                song = azlyrics.get_song_lyrics(url[1], True)
                self._process_song(song)

    def _grab_from_darklyrics(self):
        for url in darklyrics.get_album_links(self.artist):
            print('Downloading album {}'.format(url))
            songs = darklyrics.get_lyrics_by_album(url)
            for song in songs:
                song.title = fix_string_path(song.title)
                self._process_song(song)


for artist in artists:
    grabber = Grabber(artist)
    grabber.grab(source)

