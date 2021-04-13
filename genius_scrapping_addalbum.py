# app website url = https://karis_genius_scapping.com/
# app redirect url = https://karis_genius_scrapping.com

#client id = WJ2EdOsgBmbolihHEsju5yVh9vZi5e0SAbSyi6iqDw0sNa4PHRllmibJTm7Vt0xj
#client secret = bjUrs5uJNRrlgbyR7z5cNJ1xWZhck01ikebnb8XihEXEWYoyR2ONraCasFGDLoiLoAekpiKk5ErX05_qE372-w
CLIENT_ACCESS_TOKEN = 'FeDgPQlbtvtGEyr6eEd2atCqT5nBo4o04qrVIHjhiRnktKcBUgt3NXpqT9yTh9T2'


import lyricsgenius as genius
from openpyxl import Workbook
import datetime
import pandas as pd
import numpy as np

#calling api and scrapping data

api = genius.Genius(CLIENT_ACCESS_TOKEN,  skip_non_songs = True, remove_section_headers = True, replace_default_terms= True)
#creates artist object with properties: name, image_url, songs, and num_songs
taylorswift = api.search_artist( 'Taylor Swift',max_songs=None, sort='popularity', get_full_info=False)

#preparing data for dataframe
ranking_list = np.arange(start=1, stop=taylorswift.num_songs+1)
title_list = []
artist_list = []
lyrics_list = []
album_list = []
year_list = []
for song in taylorswift.songs:
    title = song.title 
    title_list.append(title)   
    artist = song.artist
    artist_list.append(artist)
    lyrics = song.lyrics
    lyrics_list.append(lyrics)
    try:   
        testsong = api.search_song(title, taylorswift.name)
    except:
        print('unable to call api to search song', song)
        continue
    try:
        album = testsong.album
        album_list.append(album)
    except:
        album = 'NaN'
        album_list.append(album)
    try:   
        year = testsong.year
        year_list.append(year) 
    except:
        year = 'NaN'
        year_list.append(year) 


datadict = {'rank': ranking_list, 'title': title_list, 'artist': artist_list,\
     'lyrics': lyrics_list, 'album': album_list, 'year': year_list}

pd.set_option('max_colwidth',5000)
tswiftdf= pd.DataFrame(datadict)
'''
************************************************************************************************
DATA CLEANSING
************************************************************************************************
'''   

#good idea just to get an overview of the dataframe, number of columns, rows, datatypes, number of null values
tswiftdf.info()
print()
print()
tswiftdf.drop_duplicates(subset='lyrics', inplace=True)

#excluded_termskj = ['(Original)', '[Liner Notes]', '[Discography List]','(Intro)','(Live)', '(Acoustic)', '(Remix)',\
    # '(Live/2011)', '(Voice Memmo)', '(Grammys 2016)', '(Piano Version)', '(Piano/Vocal)',\
    # '(Solo Version)', '(Alternate Demo)', '(Alternate Version)','(Demo)', '(Vevo Version)', '(For Daddy Gene)',\
    # '(Radio Disney Version)', '(Happy Voting)', '(Pop Mix)', '(Digital Dog Radio Mix)']

#drop rows in place that consist of words in remove_titles variable
remove_titles = 'Live|Acoustic|Remix|version|Disney|Demo|Radoio|Memo|Mix|Grammys|Dog|Linear'
tswiftdf.drop(tswiftdf[tswiftdf['title'].str.contains(remove_titles, case=False)].index, inplace = True)

tswiftdf['lyrics'].replace('\n', '', inplace=True)
#test that desired \n was removed
print(tswiftdf.iloc[0].loc['lyrics'])

# Check if any text was truncated
pd_width    = pd.get_option('max_colwidth')
maxsize     = tswiftdf['lyrics'].map(len).max()
n_truncated = (tswiftdf['lyrics'].map(len) > pd_width).sum()
print("\nTEXT LENGTH:")
print("{:<17s}{:>6d}".format("   Max. Accepted", pd_width))
print("{:<17s}{:>6d}".format("   Max. Observed", maxsize))
print("{:<17s}{:>6d}".format("   Truncated", n_truncated))
print()
print()

#actual date was imported into year column. i still want this data but I also just want to know the year
tswiftdf['year'] = pd.to_datetime(tswiftdf['year'], format = '%Y/%m/%d')
tswiftdf['date'] = tswiftdf['year']
#extract just the year 
tswiftdf['year'] = pd.DatetimeIndex(tswiftdf['year']).year
#get rid of the time component and just keeps the date
tswiftdf['date'] = tswiftdf['date'].dt.date
#make sure the datatypes are correct
#not sure why they revert back to int and string object but the cleanup worked in the process
tswiftdf.info()
print()
print()


#reindex rank column after all data cleansing
tswiftdf['rank']= np.arange(start=1, stop=tswiftdf.shape[0]+1)

#write cleaned up df to excel file
tswiftdf.to_excel('TaylorSwift_Songs.xlsx',sheet_name='song_data')

print('TSWIZZLE cleanup complete')
print()
print('LETS GET ANNALYZING')

