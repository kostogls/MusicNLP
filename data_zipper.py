import pandas as pd
import os
import re

path_in = 'C:/Users/User/Desktop/genres/'
genres = os.listdir(path_in)

artists = []
for i in genres:
    artists.append(os.listdir(path_in + i))


def zipper(path_songs, path_lyrics, artist, genre):
    f = open(path_songs, "r", encoding='utf-8')
    a = f.readline()
    song_names = a.replace("[", '').replace("]", '').replace('β€™', "")  # .split(", '")
    song_names = re.split(r", '|, \"", song_names)
    song_names = [song.replace("'", '') for song in song_names]
    # print(song_names)
    g = open(str(path_lyrics), "r", encoding='utf-8')
    b = g.readlines()
    b = [re.sub(r"\d*EmbedShare URLCopyEmbedCopy", '', s) for s in b]
    zipped = " ".join(b)
    lyrics = zipped.split("***\n")
    lyrics = [l.replace("\n \n", ".").replace("\n", '.') for l in lyrics]
    artist = [artist for w in song_names]
    genre = [genre for w in song_names]
    # below i transform every list to same length
    if len(lyrics) > len(artist) or len(lyrics) > len(song_names) or len(lyrics) > len(genre):
        counter = len(lyrics)-min(len(artist), len(song_names), len(genre))
        while counter > 0:
            artist.append('NaN')
            song_names.append('NaN')
            genre.append('NaN')
            counter -= 1
    # print(len(lyrics), len(artist), len(genre), len(song_names))

    # print(song_names)

    d = {'genre': genre, 'artist': artist, 'song names': song_names, 'lyrics': lyrics}

    # df = pd.DataFrame.from_dict(d, orient='index')

    df = pd.DataFrame({'genre': genre, 'artist': artist, 'song names': song_names, 'lyrics': lyrics})

    ##df.to_csv((artist+".csv"),header = True,index = False)
    return df


def to_dataframe():
    final = pd.DataFrame()
    for i in range(len(genres)):
        print(i)
        path = path_in + str(genres[i]) + "/"
        for j in range(0, len(artists[i]), 2):
            print(artists[i][j])
            final = pd.concat([final, zipper(path + artists[i][j + 1], path + artists[i][j],
                                             artists[i][j + 1].replace('_songs.txt', ''), genres[i])],
                              ignore_index=True)
    final.to_csv("newlyrics_dataset.csv", header=True, index=False)
    final4genres = final[
        (final.genre == "country") | (final.genre == "rock") | (final.genre == "punk") | (final.genre == "rap")]
    final4genres.to_csv("4genres_lyrics_dataset.csv", header=True, index=False)


to_dataframe()
