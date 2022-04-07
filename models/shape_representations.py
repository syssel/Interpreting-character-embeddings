from PIL import Image,ImageDraw,ImageFont
import sys
from builtins import FileExistsError
import os
from os import path

from skimage import io, color

import numpy as np
import pandas as pd


#BEFORE RUNNING THIS SCRIPT, CREATE AN EMPTY DIRECTORY CALLED data_shape_{LANG}
#ALSO, THIS LOADS THE Arial Unicode FONT FROM THE OSX LIBRARY.
#IT MAY HAVE TO BE CHANGED IN DIFFERENT COMPUTERS

'''
THIS SCRIPT GENERATES IMAGES OF LETTERS OF A SPECIFIC FONT
IT THEN SAVES THEM ALL IN A DIRECTORY CALLED data_shape
THE NAMES OF FILES ARE [a-z].png and [a-z]-up.png
'''

#Based on https://stackoverflow.com/questions/24085996/how-i-can-load-a-font-file-with-pil-imagefont-truetype-without-specifying-the-ab

#font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 28, encoding="unic")
font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 28, encoding="unic")


lang = sys.argv[1]
emb_output_file = "shape-embeddings-"+lang+".csv"

if path.exists(emb_output_file):
    print ("Please be advised that the file "+emb_output_file+" will be overwritten.")
    rsp = input("Are you sure you want to continue? [y]/n)")
    if (rsp == "n"):
        exit()

    


#ENGLISH ALPHABET
if lang=="en" or lang =="nl":
    seq="abcdefghijklmnopqrstuvwxyz"

#SPANISH ALPHABET
elif lang=="es":
    seq="abcdefghijklmnñopqrstuvwxyz"

#KOREAN SYLLABLE ALPHABET
elif lang=="krsyll":
    f=open("korean_syllables_list.txt")
    seq = [line.rstrip() for line in f]
    f.close()

#KOREAN HANGUL
elif lang =="kr":
    seq="ㅂㅈㄷㄱㅅㅁㄴㅇㄹㅎㅋㅌㅊㅍㅛㅕㅑㅐㅔㅗㅓㅏㅣㅠㅜㅡ"

#JAPANESE HIRAGANA
elif lang == "jp":
    seq = [chr(i) for i in range(12353, 12436)]

letters = seq
#letters = list(seq+seq.upper())
print (letters)

max_text_height = 0
max_text_width  = 0
for letter in letters:
    # get the line size
    text_width, text_height = font.getsize(letter)
    if text_width > max_text_width:
        max_text_width = text_width
    if text_height > max_text_height:
        max_text_height = text_height



        
data_path = "data_shape_"+lang
try:
    os.mkdir(data_path)
except FileExistsError:
    print ("The directory already exists.\nPlease delete the <"+data_path+"> directory.")
    raise



for letter in letters:
    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', (max_text_width + 10, max_text_height + 10), "white")

    # draw the text onto the text canvas, and use black as the text color
    draw = ImageDraw.Draw(canvas)
    draw.text((5,5), letter, 'black', font)

    # save the blank canvas to a file
    if letter.isupper():
        canvas.save(data_path+"/"+letter+"-up.png", "PNG")
    else:
        canvas.save(data_path+"/"+letter+".png", "PNG")
    #canvas.show()









list_files = os.listdir(data_path)
image_dict  = {}
letter_dict = {}
for filename in sorted(list_files):
    print (filename)
    #This should be uncommented when getting data from more fonts
#    actual_letter = "".join(filename.split("-")[1:])[:-4]
    #This should be uncommented when getting data from only one font
    actual_letter = filename[:-4]
    
    print (filename,actual_letter)
    actual_letter = actual_letter[0] #I do this to remove the "up" from, e.g., "Qup"
    fontname = filename.split("-")[0]
    print (filename,actual_letter,fontname)
    if actual_letter not in image_dict.keys():
        image_dict[actual_letter]=[]
    if actual_letter not in letter_dict.keys():
        letter_dict[actual_letter]=[]
    image_dict[actual_letter].append(io.imread(data_path + "/" + filename, as_gray=True))
    letter_dict[actual_letter].append(actual_letter+"fontname")

shape_representations = {}
for letter in image_dict.keys():
    #Create average image, and reshape everything to flatten the image itself to a long vector
    shape_representations[letter] = np.mean(image_dict[letter],axis=0).reshape(-1)

pd.DataFrame(shape_representations).T.to_csv(emb_output_file)

