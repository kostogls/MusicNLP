# MusicNLP
Generation of dataset containing 3154 songs with their artist, title, lyrics and genre.

Exports were extracted using lastfm-scraper https://github.com/dbeley/lastfm-scraper. The genres that were chosen are Country, Metal, Punk, Rock, Pop, Rap and Reggae.
Also lyricgenius API https://github.com/johnwmillr/LyricsGenius#usage was very useful.

You can use whatever genre you like, and run extract_lyrics.py (you should change api key in line 8 with your valid api key from Genius.com, just make an account and you can get one).
After, create a folder with the extracted files/genre that is called genres (change in data_zipper.py the path_in variable pointing to this folder) and just run
data_zipepr.py.
