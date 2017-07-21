import os
import string


def clean_lines(text):
    pos = 0
    while pos < len(text):
        if text[pos] == ',' and pos + 1 < len(text) and text[pos + 1] in string.ascii_letters:
            text = text[:pos + 1] + ' ' + text[pos + 1:]
        if text[pos] == '\n':
            if pos - 1 < 0 or pos + 1 >= len(text) or text[pos - 1] == '\n' == text[pos + 1]:
                text = text[:pos] + text[pos + 1:]
                continue
        pos += 1
        continue
    return text


def clean_song(song_path):
    with open(song_path, 'r', encoding='utf8') as sf:
        text = clean_lines(sf.read())
    with open(song_path, 'w', encoding='utf8') as sf:
        sf.write(''.join(text))
