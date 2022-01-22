import lyricsgenius
import csv
import re
import time
import os, os.path

# here change your api key
api = lyricsgenius.Genius("Wf4EZbfBaUGc9tEYXKtKniHVF_qPwo4bVjpGmoP4S7-9im5VoMi_I5bkwP7lR4NB", timeout=80, retries=30)
genre = ["country", "metal", "punk", "rock", "pop", "rap", "reggae"]
api.remove_section_headers = True
api.skip_non_songs = True


def artists_to_lists():
    data_ = []
    for g in genre:
        with open(f"Exports/{g}_bs4.csv", newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            flat_list = [item for sublist in data for item in sublist]
            #print(flat_list)
        data_.append(flat_list)

    # keep same number of artists per genre (20, except pop that is 18)
    new_data = []
    for i in data_:
        new_data.append(i)
    print(new_data)
    return new_data


def get_lyrics(genre, artists_per_genre):
    c = 0
    genre_counter = 0
    for g in artists_per_genre:
        os_name = f"{genre[genre_counter]}"
        os.mkdir(os_name)
        for artist in g:

            try:
                time.sleep(10)
                songs = (api.search_artist(artist, max_songs=20, sort='popularity')).songs
                song_titles_list = []
                file1 = f"{artist}_songs.txt"
                complete1 = os.path.join(os_name, file1)

                for song in songs:
                    if api.search_song(song.title, artist) is not None:
                        song_titles_list.append(song.title)
                    else:
                        continue
                # song_titles_list = [song.title for song in songs]
                songs_of_artist = open(complete1, "w", encoding='utf-8')
                songs_of_artist.write(str(song_titles_list))
                songs_of_artist.close()
                song_lyrics_list = [song.lyrics for song in songs]
                file2 = f"{artist}_lyrics_of_{len(song_lyrics_list)}_songs.txt"
                complete2 = os.path.join(os_name, file2)
                file = open(complete2, "w", encoding='utf-8')
                file.write("\n***\n".join(song_lyrics_list))
                c += 1
                print(f"Songs taken:{len(song_lyrics_list)}")
                file.close()
                time.sleep(6)
            except Exception as e:
                print(e, e.args)
        genre_counter += 1



s = artists_to_lists()
#search_songs(s)
#test = ['Led Zeppelin', 'Queen', 'The Beatles']
get_lyrics(genre, s)


