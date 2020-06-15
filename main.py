#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:17:18 2020

@author: eric
"""
import os

import rasterio as rio
import numpy as np
import logging.config
import time

import matplotlib.pyplot as plt

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
    Mainly useful to parse and keep track of the geotransform parameters

    Parameters
    ----------
    path : str
        path to raster, in a format readable by rasterio.
    
    Attributes
    ----------
    transform : obj, Affine
        Contains the geotransform info for the raster in rasterio format
    xstep : float
        
    ystep : int
        Numeric error code.
    bounds : int
        Numeric error code.
    profile : dict
        Numeric error code.
    code : int
        Numeric error code.

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
        """
        """
        with rio.open(self.path) as src:
            data = np.squeeze(src.read())
        return data



def check_rotation(raster_one, raster_two):
    """compares rotation terms of two rasters.
    """
    test = (raster_one.rotone == raster_two.rotone) and \
            (raster_one.rottwo == raster_two.rottwo)
    return test    

def check_crs(raster_one, raster_two):
    """compares crs of two rasters.
    """
    test = raster_one.profile['crs'] == raster_two.profile['crs']
    return test    


def get_overlap_coords(raster_one, raster_two):
    """This gets the coordinates of the top left, and bottom right pixels
    of raster_one that do overlap raster_two
    
    Parameters
    ----------
    raster_one, raster_two : obj, Raster
        Number of pixels for which to compute the overlaps
 
    Returns
    -------
    [left, right] : list
        Contains the perentage of overlap between the given pixel with his 
        closest right and left(or upper/under) pixels
    """
    
    xmargin = 1/2*(raster_one.xstep + raster_two.xstep)
    ymargin = 1/2*(np.abs(raster_one.ystep) + np.abs(raster_two.ystep))
    
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
        if not testx: x+=1
        if not testy: y+=1
            
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

def percents_gent(npixels, smallstep, bigstep, offset):
    """ Generate the percentage of overlap for each pixel. 
    
    Parameters
    ----------
    npixels : int
        Number of pixels for which to compute the overlaps
    smallstep : float
        measure in projection units of the width for the smaller raster
    bigstep : float
        measure in projection units of the width for the bigger raster
    offset: float
        offset in given dimension between rasters

    Returns
    -------
    [left, right] : list
        Contains the perentage of overlap between the given pixel with his 
        closest right and left(or upper/under) pixels
    """
    for cellnumber in range(npixels):
        inside = ((cellnumber*smallstep)-offset)//bigstep==(((cellnumber+1)*smallstep)-offset)//bigstep
        if inside:
            yield [100,0]
        else:
            nbigsteps = ((cellnumber+1)*smallstep)//bigstep
            left = 100*( ( (bigstep * nbigsteps)  \
                         - ((cellnumber*smallstep)-offset)) / smallstep)
            right = 100 - left
            yield [left, right]

def get_offset(raster_one, raster_two, top):
    """Computes the x and y offfset between rasters in projection units.
    Parameters
    ----------
    raster_one, raster_two: obj, Raster
        
    top : list, [int, int]
        coordinates of the top left overlapping pixel from raster one.
    """
    # TODO : not completely clean : raster_two also should have a top pixel.
    coords_one = rio.transform.xy(raster_one.transform, top[0], top[1], offset='ul')
    coords_two = rio.transform.xy(raster_two.transform, 0, 0, offset='ul')
    
    xoffset = coords_two[0] - coords_one[0]
    yoffset = coords_two[1] - coords_one[1]
    return [xoffset, yoffset]

def illustrate_overlap(x_toduplicate, y_toduplicate, outpath=None):
    """creates and shows the overlapping pattern between rasters. For 
    verification and lulz.
    
    Parameters
    ----------
    input_one, input_two : str
        paths to the rasters in a format readable by rasterio
    
    """
    overlap_bool = np.repeat(x_toduplicate[np.newaxis,...].astype(int), y_toduplicate.size, axis=0)
    overlap_bool = overlap_bool + y_toduplicate[:,np .newaxis].astype(int)
    plt.imshow(overlap_bool[0:30,0:30])
    plt.show()
    plt.clf()
    if outpath is not None:
        output = os.path.join(outpath, "overlap_illustration")
        profile = rasterone.profile
        profile.update(dtype=rio.int32,
                   nodata=-1)
        overlap_bool = overlap_bool.astype(rio.int32)
        with rio.open(output, mode='w', **profile) as dst:
            dst.write(np.expand_dims(overlap_bool, axis=0))
    return        

def set_rasters(input_one, input_two):
    """Sets the order of the rasters and initiates the Raster objects
    
    Parameters
    ----------
    input_one, input_two : str
        paths to the rasters in a format readable by rasterio
    Returns
    -------
    obj, Raster:
        Raster objects based on the input paths.
    """
    with rio.open(input_one) as src:
        step_one = src.profile['transform'][0]
    with rio.open(input_two) as src:
        step_two = src.profile['transform'][0]
    if np.abs(step_one) > np.abs(step_two): 
        rastertwo = input_one
        rasterone = input_two
    else: 
        rasterone = input_one
        rastertwo = input_two

    return Raster(rasterone), Raster(rastertwo)

def extend_values(rasterone, shape, x_indices, y_indices):
    """ Duplicates values of raster at given indices, effectively expanding it.
    Notes
    -----
        The shape could be computed here from raster one, x and y indices..
    Parameters
    ----------
        rasterone
        shape : list
            shape of the expected raster
        x_indices, y_indices : 
        
    """
    data = rasterone.read()
    values_extended = np.ones(shape)
    logging.debug("Begin x extension ")

    for i, value_line in enumerate(data):
        values_extended[i] = np.insert(value_line, y_indices, 
                                                value_line[y_indices])
    logging.debug("x extension done")
    for i, value_column in enumerate(
                        values_extended .T[0:data.shape[0]+x_indices.size,:]):
        values_extended .T[i] = np.insert(value_column, x_indices, value_column[x_indices]
                                                )[0:values_extended.shape[0]]
    return values_extended


def main(rasterone, rastertwo, output=None, area=None, ilus=True):
    """from two rasters with same projection and rotation parameters, but
    with different resolution, computes the % of overlap or 
    weighted mean in each pixel of the raster with less resolution.
    # TODO : Could also go the other way : and just use the rasters orders.
    But this would have to be tested.
    """    
    startTime = time.time()
    
    rasterone, rastertwo = set_rasters(rasterone, rastertwo)
    
    pof = check_rotation(rasterone, rastertwo)
    if pof==False:
        logging.error('Not same rotation on rasters')
        return
    else:
        pass
    # Get the top and bottom pixels from rasterone overlapping rastertwo
    top, bot = get_overlap_coords(rasterone, rastertwo)
    
    npixels = [bot[0]+1 - top[0], bot[1]-top[1]+1]
    xoffset, yoffset = get_offset(rasterone, rastertwo, top)
    xoverlap = percents_gent(npixels[0], rasterone.xstep, rastertwo.xstep, xoffset)
    yoverlap = percents_gent(npixels[1], rasterone.ystep, rastertwo.ystep, yoffset)
    overlap_x = np.around(np.array([c for c in xoverlap]), decimals=5)
    overlap_y = np.around(np.array([c for c in yoverlap]), decimals=5)
    # When the first pixel overlaps, its left overlap does not exist.
    # TODO : Check how valid this condition is. Should probably be !=0
    if xoffset > 0:
        overlap_x[0,0] = 0
    if yoffset > 0:
        overlap_y[0,0] = 0
     
    #remove 0
    weights_x = overlap_x.ravel()
    weights_x = overlap_x[overlap_x>0]
    weights_y = overlap_y.ravel()
    weights_y = overlap_y[overlap_y>0]
    
    #x and y_toduplicate are the positions where pixels are cut(!=100) so
    # the values of rasters need to be duplicated to correspond.
    x_toduplicate = np.array([x[0]!=100 for x in overlap_x])
    if xoffset > 0:
        x_toduplicate[0] = False 
    y_toduplicate = np.array([y[0]!=100 for y in overlap_y])
    if yoffset > 0:
        y_toduplicate[0] = False
    
    #---------------------illus overlap
    #This is just an illustration : print out the raster showing if a pixel if
    # inside or overlaps two/four pixels.
    if ilus : illustrate_overlap(x_toduplicate, y_toduplicate)
    #-----------------------Continuing
    
    # weights computation: combination of x and y axis weights
    result = np.repeat(weights_x[np.newaxis,...], weights_y.size, axis=0)
    result_total = result.T * weights_y 
    weights_total = result_total /100
    
    x_indices = np.where(x_toduplicate==1)[0]
    y_indices = np.where(y_toduplicate==1)[0]
    
    # Extend values for them to have the shape as the weights
    datashape = rasterone.read().shape

    values_extended = extend_values(rasterone, result_total.shape,
                                    x_indices, y_indices)
        
    elapsedTime = time.time() - startTime
    print('function [{}] finished in {} ms, or {} s'.format(
        'before big loop', int(elapsedTime * 1000), int(elapsedTime)))
    
    # add min and max limits to indices if needed.
    if x_indices[0] != 0:
        x_indices = np.insert(x_indices, 0,0)
    if y_indices[0] != 0:
        y_indices = np.insert(y_indices, 0,0)
    
    if x_indices[-1] != datashape[1]:
        x_indices = np.append(x_indices, datashape[1])
    if y_indices[-1] != datashape[0]:
        y_indices = np.append(y_indices, datashape[0])

    # TODO : for now, all pixels from raster two are considered overlapped.
    #Offset should have an impact here.
    result = np.ones_like(rastertwo.read())      
    # produce the indices for the extented matrices
    # TODO : check why +1 needed in y and not in x...
    x_extended_index = x_indices + np.arange(x_indices.size)
    y_extended_index = y_indices + np.arange(y_indices.size)+1
    for i in range(len(x_indices)-1):
        for j in range(len(y_indices)-1): 
            values = values_extended[x_extended_index[i]:x_extended_index[i+1],
                                    y_extended_index[j]:y_extended_index[j+1]]
            weights = weights_total[x_extended_index[i]:x_extended_index[i+1],
                                    y_extended_index[j]:y_extended_index[j+1]]/100
            try : 
                result[i,j] = np.sum(values * weights)
            except IndexError: # end of raster 2
                pass

    if area:
        result = np.round(result * np.abs(rasterone.xstep*rasterone.ystep)  
                          / (np.abs(rastertwo.ystep*rastertwo.xstep)),2)*100
    
    
    output = os.path.join(output, "test_result2")
    profile = rastertwo.profile
    profile.update(dtype=rio.int32,
                   nodata=-1)
    with rio.open(output, mode='w', **profile) as dst:
        dst.write(np.expand_dims(result.astype('int32'), axis=0))    
        
    
    elapsedTime = time.time() - startTime
    print('function [{}] finished in {} ms, or {} s'.format(
        'The End', int(elapsedTime * 1000), int(elapsedTime)))
    
    return

if __name__ =="__main__":
    print("Hello")
    rasterone = "/home/eric/DATA/project_r2intersect/DATA/tile_rep_Hansen_GFC-2018-v1.6_lossyear_BINARY.tif"
    rastertwo = "/home/eric/DATA/project_r2intersect/DATA/V45_10pix_1supports_50badpixls_mean.tif"
    out = "/home/eric/DATA/project_r2intersect/RESULTS/test_intersects/main/"
    main(rasterone, rastertwo, output=out)