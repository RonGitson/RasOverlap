#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:17:18 2020

@author: eric
"""

import os
import time

import rasterio as rio
import numpy as np
import logging.config



#
np.set_printoptions(suppress=True)

"""
Distance entre deux centrïdes, en x ET en y 
inférieure à la somme des moitiés des steps.
"""

#---------time function
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap



class Raster:
    """
    Raster object for the raster overlap checking.
    """

    def __init__(self, path):

        if os.path.exists(path):
            logging.info('Initializing from HDF5 file')
            self.path = path
        else:
            logging.error("Bad path.")
            raise RuntimeError("Bad Path.")
        with rio.open(path) as src:
            self.profile = src.profile
            self.bounds =  [src.bounds.left, src.bounds.right, 
                            src.bounds.bottom, src.bounds.top]
        self.transform = self.profile['transform']
        self.xstep = self.transform[0]
        self.ystep = self.transform[4]
        self.rotone = self.transform[1]
        self.rottwo = self.transform[3]
        # when referecing data, width or height MINUS ONE.
        self.width = self.profile['width'] #== first axis [axis, nope]
        self.height = self.profile['height'] #== second axis [nope, axis]
        self.affine = self.profile['transform']
        return
    def read(self):
        with rio.open(self.path) as src:
            a = np.squeeze(src.read())
        return a



def check_rotation(raster_one, raster_two):
#If I understand correctly : 
#https://www.esri.com/about/newsroom/arcuser/understanding-raster-georeferencing/
# affine [1] et affine[3] sont des "rotation terms"S

#Tout ce que j'ai à faire, c'est vérifier que dans les deux rasters, ce sont
#les mêmes.
    
    test = (raster_one.rotone == raster_two.rotone) and \
            (raster_one.rottwo == raster_two.rottwo)
    return test    

def check_crs(raster_one, raster_two):
#If I understand correctly : 
#https://www.esri.com/about/newsroom/arcuser/understanding-raster-georeferencing/
# affine [1] et affine[3] sont des "rotation terms"S

    
    test = raster_one.profile['crs'] == raster_two.profile['crs']
    return test    


def get_overlap_coords(raster_one, raster_two, xmargin, ymargin):
    """This gets me the coordinates of the top left, and bottom right pixels
    of raster_one that do overlap rastertwo
    """
    
    """
    ((17*4)//5)%4
    ((cellnumber*smallstep)//bigstep)%smallstep
    full = 
    
    #This checks wether a cell is full or not.
    (cellnumber*smallstep)//bigstep==((cellnumber+1)*smallstem)//bigstep
    with offset : 
    ((offset + (cellnumber*smallstep))//bigstep==((cellnumber+1)*smallstem)//bigstep
        
     if overlap : 
         left = bigstep * ((cellnumber*smallstep)//bigste)+1 - cellnumber*smallstep
         #right = smallstep-left
         right = (cellnumber+1)*smallstep - (bigstep * ((cellnumber+1)*smallstep//bigstep))
         
    a generator may be the way to go to be quick for the percentage generation.
    
    """
    
    #TODO : Two things to test, top left, bottom right.
    # Begin at 0,0, and increment each one until all conditions work.
    # Same for end, end.
    
    #Check top
    inbound = False
    x = y = 0
    while not inbound and (x<raster_one.width and y<raster_one.height):
        long, lat = rio.transform.xy(raster_one.transform, [x],[y])
        long, lat = long[0], lat[0]
        testx = (long>=raster_two.bounds[0]-xmargin and 
                     long<=raster_two.bounds[1]+xmargin)
        testy = (lat>=raster_two.bounds[2]-ymargin and 
                     lat<=raster_two.bounds[3]+ymargin)
        inbound = testx and testy
        if not testx:
            x+=1
        if not testy:
            y+=1
    topleft=[x,y]
    if not inbound:
        logging.error("Failure finding overlap")
        topleft = 0
    #Check bottom
    x = raster_one.width - 1
    y = raster_one.height - 1
    while not inbound and (x>=0 and y>=0) :
        long, lat = rio.transform.xy(raster_one.transform, [x],[y])
        long, lat = long[0], lat[0]

        testx = (long>=raster_two.bounds[0]-xmargin and 
                     long<=raster_two.bounds[1]+xmargin)
        testy = (lat>=raster_two.bounds[2]-ymargin and 
                     lat<=raster_two.bounds[3]+ymargin)
        inbound = testx and testy
        if not testx: x-=1
        if not testy: y-=1
    if not inbound:
        logging.error("Failure finding overlap")    
    bottomright = [x,y]
    return topleft, bottomright

def percents_gent(rang, smallstep, bigstep, offset):
    """
    Generateur les pourcentages d'overlap pour chaque cellule.
    """
    for cellnumber in range(rang):
        inside = ((cellnumber*smallstep)-offset)//bigstep==(((cellnumber+1)*smallstep)-offset)//bigstep
        if inside:
            yield [100,0]
        else:
            nbigsteps = ((cellnumber+1)*smallstep)//bigstep
            left = 100*( ( (bigstep * nbigsteps)  \
                         - ((cellnumber*smallstep)-offset)) / smallstep)
#            left = 100*(bigstep*((((cellnumber*smallstep)-offset)//bigstep)+1)
#                        - ((cellnumber*smallstep) / smallstep))
            right = 100 - left
            yield [left, right]

def get_offset(raster_one, raster_two, top):
    """
    raster_one[top], - raster_two[0,0]
    offset values
       if offset == 'center':
        coff, roff = (0.5, 0.5)
    elif offset == 'ul':
        coff, roff = (0, 0)
    elif offset == 'ur':
        coff, roff = (1, 0)
    elif offset == 'll':
        coff, roff = (0, 1)
    elif offset == 'lr':
        coff, roff = (1, 1)
    """
    # TODO : not finished
    coords_one = rio.transform.xy(raster_one.transform, top[0], top[1], offset='ul')
    coords_two = rio.transform.xy(raster_two.transform, 0, 0, offset='ul')
    
    xoffset = coords_two[0] - coords_one[0]
    yoffset = coords_two[1] - coords_one[1]

    return [xoffset, yoffset]


def dilate_array(raster, weights):
    """ based on the weights array, duplicates raster values.
    """
    
    boold_duplicates = [x[0]==100 for x in weights]
    
    
    return

def compute_stuff():
    """
    Here is the challenge. Getting through to the computing in a smart way.
    How to use the proper weights, for the proper pixels?
    # TODO : popping the "0" off the weights may be a solution. Each weight Used
    only once. The I just need to use the proper code for combining the x and y weights    
    # TODO : If i am going for this repetition idea, I need to pop out th "right" 
    off the last one of each column/line
    
    also have the problem of combining the weights. one is 100, doesn't mea
    The problem, even having the weights and corresponding values,
    is that then I need to get them for the right cells.problem for tomorrow.
    
    Nan, le plus simple ça reste une fonction :
        Compte tenu de l'offset, et des steps, quelles cellules sont intersectées?
        Ou alors.
        Au moment où jecalcule mes poids, je garde en mémoire. et je fais un masque...
        Bah, ça revient à faire un slice /pixel...c'est beaucoup.
    Autre solution. plus facile.
    Je fais une matrice en remplissant avec des NA : pour avoir le maximum nombre de pixels
    intersectés.
    Les valeurs pour mon raster deux seront dans les fenêtres de la taille donnée.
    et puis faire du fancy indexing
    """
    
    
    return

"""
Pour être très clair.
mes overlap ont forcément déjà la bonne forme.
ce qui n'a pas la bonne forme, c'est le raster.
"""
def extend_array(array, boolarray):
    """Duplicates values where pixel weights are shared between several bigger.
    array is a line from the raster.
    """
    where = np.where(boolarray)
    where = where[0]
    res = np.ones(array.size+np.sum(boolarray))
    res[0:] = array
    for i in where:
        res[i:] = array[i:]
        return
"""
NB : Raster One, c'est le petit, raster two c'est le gros.
(One= hanse27m, Two=Age100m)
"""

startTime = time.time()

hansen = "/home/eric/DATA/project_r2intersect/DATA/tile_rep_Hansen_GFC-2018-v1.6_lossyear_BINARY.tif"
age = "/home/eric/DATA/project_r2intersect/DATA/V45_10pix_1supports_50badpixls_mean.tif"

outputh_folder = "/home/eric/DATA/project_r2intersect/RESULTS/test_intersects/"

rastertwo = Raster(age)
rasterone = Raster(hansen)

pof = check_rotation(rasterone, rastertwo)
if pof==False:
    logging.error('Not same rotation on rasters')
else:
    print("continuing")

# TODO : get this parameter globalized:
xmargin = 1/2*(rasterone.xstep + rastertwo.xstep)
ymargin = 1/2*(np.abs(rasterone.ystep) + np.abs(rastertwo.ystep))


top, bot = get_overlap_coords(rasterone, rastertwo, xmargin, ymargin)


npixels = [bot[0]+1 - top[0], bot[1]-top[1]+1]
xoffset, yoffset = get_offset(rasterone, rastertwo, top)
xoverlap = percents_gent(npixels[0], rasterone.xstep, rastertwo.xstep, xoffset)
yoverlap = percents_gent(npixels[1], rasterone.ystep, rastertwo.ystep, yoffset)

overlap_x = np.around(np.array([c for c in xoverlap]), decimals=3)
overlap_y = np.around(np.array([c for c in yoverlap]), decimals=3)
# When the first part of the pixel is notinside, need to erase it.
if xoffset > 0:
    overlap_x[0,0] = 0
if yoffset > 0:
    overlap_y[0,0] = 0

#Other checking that needs toFR
# be done : the last values of the overlap series
#may not be contained. To check.
    
    
    
#Here I'm producing the overlap matrix.

#remove 0
weights_x = overlap_x.ravel()
weights_x = overlap_x[overlap_x>0]
weights_y = overlap_y.ravel()
weights_y = overlap_y[overlap_y>0]

# different cases:
 #Here I have to check wether the left of the rasters is PERFECTLY aligned;
"""
if aligned:
    print("do serious stuff)
else:
if weights_x[0] =!0:
    weights_x = np.delete(weights_x[0])
if weights_y[0] =!0:
    weights_y = np.delete(weights_x[0])
"""
#x and y_toduplicate are the positions where pixels are cut(!=100) so
# the values of rasters need to be duplicated to correspond.
# Gets good number of values to duplicate
x_toduplicate = np.array([x[0]!=100 for x in overlap_x])
if xoffset > 0:
    x_toduplicate[0] = False 
y_toduplicate = np.array([y[0]!=100 for y in overlap_y])
if yoffset > 0:
    y_toduplicate[0] = False

#---------------------illus overlap
#This is just an illustration : print out the raster showing if a pixel if
# inside raster 2, overlaps two, or four pixels.
overlap_bool = np.repeat(x_toduplicate[np.newaxis,...].astype(int), y_toduplicate.size, axis=0)
overlap_bool = overlap_bool + y_toduplicate[:,np .newaxis].astype(int)
# TODO : output this matrix, and check how it fits with my rasters.
#plt.imshow(overlap_bool[0:30,0:30])
output = os.path.join(outputh_folder, "test_one")
profile = rasterone.profile
profile.update(dtype=rio.int32,
               nodata=-1)

overlap_bool = overlap_bool.astype(rio.int32)
with rio.open(output, mode='w', **profile) as dst:
    dst.write(np.expand_dims(overlap_bool, axis=0))
#-----------------------Continuing

# Here I compute the weights : combination of x and y axis weights
result = np.repeat(weights_x[np.newaxis,...], weights_y.size, axis=0)
result_total = result.T * weights_y 
weights_total = result_total /100

#Whats left now is to use the rep matrix to generate an array bigger than
#raster one that would contain its values. repeated for intersecting pixels :
#possible to do twofold : first I get the reps for the lines.
#Then I do the same for the columns.

x_indices = np.where(x_toduplicate==1)[0]
y_indices = np.where(y_toduplicate==1)[0]

data = rasterone.read()

# ici truc intelligent. Je duplique les indexs d'overlap, et je les utilise 
# pour récupérer les valeurs des cellules 

"""
test = np.arange(0,25)
test = test.reshape(5,5)
x_indices_test  = np.where(test[0] //2 ==0)[0]
y_indices_test  = np.where(test[3] % 3 ==0)[0]

a = test[0:3].ravel()
b = [0,3,7,11]
values = np.ones([test.shape[0]+x_indices_test.size,
                 test.shape[1]+y_indices_test.size ])
for i, value_line in enumerate(test):
    values[i] = np.insert(value_line, y_indices_test, value_line[y_indices_test])
#    print(np.insert(data[i], y_indices_test, data[i, y_indices_test]).shape)
    
for i, value_column in enumerate(values.T[0:test.shape[0]+x_indices_test.size,:]):
    print(i)
    values.T[i] = np.insert(value_column, x_indices_test, value_column[x_indices_test]
                                            )[0:values.shape[0]]


#nouveau. Possible que ça marche
#Pour l'instant non. Mais je suis pas loin.
values = np.ones(result_total.shape)

for i, value_line in enumerate(data):
    values[i] = np.insert(value_line, y_indices, value_line[y_indices])
#    print(np.insert(data[i], y_indices, data[i, y_indices]).shape)
    
for i, value_column in enumerate(values.T[0:data.shape[0],:]):
    print(i)
    values.T[i] = np.insert(value_column, x_indices, value_column[x_indices]
                                            )[0:values.shape[0]]"""
values_extended = np.ones(result_total.shape)
for i, value_line in enumerate(data):
    values_extended [i] = np.insert(value_line, y_indices, value_line[y_indices])
    
for i, value_column in enumerate(values_extended .T[0:data.shape[0]+x_indices.size,:]):
    values_extended .T[i] = np.insert(value_column, x_indices, value_column[x_indices]
                                            )[0:values_extended.shape[0]]



#x_indices = np.repeat(x_indices, 2)
#y_indices = np.repeat(y_indices, 2)
#add extreme limits to x/y indices
"""
#This should work. But for now it does not.
if x_indices[0] != 0:
    x_indices = np.insert(x_indices, 0,0)
if y_indices[0] != 0:
    y_indices = np.insert(y_indices, 0,0)

if x_indices[-1] != data.shape[1]:
    x_indices = np.append(x_indices, data.shape[1])
if y_indices[-1] != data.shape[0]:
    y_indices = np.append(x_indices, data.shape[0])
"""
result = np.ones_like(rastertwo.read())

# NO c'est pas bon : j'utilise les indices à partir du raster, et pas du raster etendu
# Après, en théorie j'ai tout ce qu'il me faut : les valeurs, 
elapsedTime = time.time() - startTime
print('function [{}] finished in {} ms, or {} s'.format(
    'before big loop', int(elapsedTime * 1000), int(elapsedTime)))

"""
x_extended_index = x_indices + np.arange(x_indices.size)+1
y_extended_index = y_indices + np.arange(y_indices.size)+1
weighted = weights_total * values
for i, x_index in enumerate(x_indices[0:-1]): 
    for j, y_index in enumerate(y_indices[0:-1]): 
        values = data[x_index:x_index+1, y_index:y_index+1]
        weights = weights_total[x_extended_index[i]:x_extended_index[i]+1,
                                y_extended_index[j]:y_extended_index[j]+1]
        result[i,j] = np.sum(values * weights)
#Ici problème posé par les répétitions : en gardant les valeurs 100
        # elles sont utilisées plusieurs fois
"""
#big loop v2

if x_indices[0] != 0:
    x_indices = np.insert(x_indices, 0,0)
if y_indices[0] != 0:
    y_indices = np.insert(y_indices, 0,0)

if x_indices[-1] != data.shape[1]:
    x_indices = np.append(x_indices, data.shape[1])
if y_indices[-1] != data.shape[0]:
    y_indices = np.append(y_indices, data.shape[0])
    
x_extended_index = x_indices + np.arange(x_indices.size)#+1
y_extended_index = y_indices + np.arange(y_indices.size)+1
#weighted = weights_total * values_extended  
for i in range(len(x_indices)-1):
    for j in range(len(y_indices)-1): 
        values = values_extended[x_extended_index[i]:x_extended_index[i+1],
                                y_extended_index[j]:y_extended_index[j+1]]
        weights = weights_total[x_extended_index[i]:x_extended_index[i+1],
                                y_extended_index[j]:y_extended_index[j+1]]/100
        try : 
            result[i,j] = np.sum(values * weights)
        except IndexError:
#            print(i,j)
            pass


"""
#Vals to remember
weights_x[0:5]
Out[182]: array([ 74.867, 100.   , 100.   ,  91.554,   8.446])

weights_y[0:5]
Out[183]: array([ 40.625,  59.375, 100.   , 100.   , 100.   ])
"""

area = False
if area:
    result = np.round(result * np.abs(rasterone.xstep*rasterone.ystep)  
                      / (np.abs(rastertwo.ystep*rastertwo.xstep)),2)*100


output = os.path.join(outputh_folder, "test_result2")
profile = rastertwo.profile
profile.update(dtype=rio.int32,
               nodata=-1)
with rio.open(output, mode='w', **profile) as dst:
    dst.write(np.expand_dims(result.astype('int32'), axis=0))    
    
"""
for i, x_index in enumerate(x_indices ): 
    for j, y_index in enumerate(y_indices ): 
        # values c'est le problème...en fait, il faut juste répéter les
        #valeurs à un emplacement donné.
        values = raster_one[x_index:x_index+1, y_index:y_index+1]
        weights = result_total[x_index:x_index+1, y_index:y_index+1]
        result[i,j] = np.sum(values * weights)/100
"""        
#Expanded me donne les données de base, weights total donne
# l'ensemble des poids, franction de cellule par fraction de cellule




#------------------Bordel


#xone = profone['transform'][0]
#yone = profone['transform'][3]

#--------------test the difference in coordinates between the centroids of 
# the two first pixels
#pixels = rio.transform.xy(rasterone.transform, [0,0],[0,1], offset='center')
#pixels = np.array(pixels).T
#a = percents_gent(100000, 27, 100, -14)

"""
startTime = time.time()
b = np.array([c for c in a])

elapsedTime = time.time() - startTime
print('function [{}] finished in {} ms'.format(
    'lol', int(elapsedTime * 1000)))
"""
elapsedTime = time.time() - startTime
print('function [{}] finished in {} ms, or {} s'.format(
    'lol', int(elapsedTime * 1000), int(elapsedTime)))


#xoffset, yoffset = getoffset(rasterone, rastertwo)

"""

overlap_x = np.around(np.array([c for c in percents_gent(rasterone.width, rasterone.xstep,
                             np.abs(rastertwo.xstep), 15)]), decimals=3)
overlap_y = np.around(np.array([c for c in percents_gent(rasterone.height, rasterone.ystep,
                             rastertwo.ystep, -20)]), decimals=3)
vecoverlap_x = np.array([c for c in percents_gent(rasterone.width, rasterone.xstep,
                             np.abs(rastertwo.xstep), 15)])
vecoverlap_y = np.array([c for c in percents_gent(rasterone.height, rasterone.ystep,
                             rastertwo.ystep, -20)])
"""






    



