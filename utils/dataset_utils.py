import os


def read_songs(data_path: str) -> str:
    """
    :rtype: str
    :param data_path:
    :return:
    """
    files = os.listdir(data_path)
    data = []
    for file in files:
        with open(data_path + '/' + file, 'r', encoding='utf8') as fs:
            data.append(fs.read())
    return '\n\n'.join(data)
