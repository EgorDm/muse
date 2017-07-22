class Song():
    def __init__(self, title, lyrics):
        super().__init__()
        self.title = title
        self.lyrics = lyrics

    def get_lyrics(self): return '\n'.join(self.lyrics)
