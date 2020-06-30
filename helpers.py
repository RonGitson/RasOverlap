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
        
    #TODO : add top and bot interesect as properties of Raster class

    """

    def __init__(self, path):

        if os.path.exists(path):
            self.path = path
        else:
            logging.error("Bad path.")
            raise RuntimeError("Bad Path.")
        with rio.open(path) as src:
            self.profile = src.profile
            self.bounds =  [src.bounds.left, src.bounds.right, 
                            src.bounds.bottom, src.bounds.top]
            self.shape = np.squeeze(src.read()).shape
        
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
    raster_one_coords, raster_one_coords : list [[top, left], [bottom, right]]
        Contains the perentage of overlap between the given pixel with his 
        closest right and left(or upper/under) pixels
    """
    raster_one_coords = _small_raster_coords(raster_one, raster_two)
    raster_two_coords = _big_raster_coords(raster_one, 
                                          raster_two, raster_one_coords)
    
    return raster_one_coords, raster_two_coords 

def _small_raster_coords(raster_one, raster_two):
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
    xmargin = 1/2*(np.abs(raster_one.xstep) + np.abs(raster_two.xstep))
    ymargin = 1/2*(np.abs(raster_one.ystep) + np.abs(raster_two.ystep))
    
    #Check top
    xinbound = False
    x = y = 0
    while not xinbound and (x<raster_one.width and y<raster_one.height):
        long, lat = rio.transform.xy(raster_one.transform, [x],[y])
        long, lat = long[0], lat[0]
        testx = (long>=raster_two.bounds[0]-xmargin and 
                     long<=raster_two.bounds[1]+xmargin)
        testy = (lat>=raster_two.bounds[2]-ymargin and 
                     lat<=raster_two.bounds[3]+ymargin)
        xinbound = testx and testy
        if not testx: y+=1
        if not testy: x+=1
            
    topleft=[x,y]
    if not xinbound:
        logging.error("Failure finding overlap")
        topleft = 0
    #Check bottom
    x = raster_one.width - 1
    y = raster_one.height - 1
    yinbound = False
    while not yinbound and (x>=0 and y>=0) :
        long, lat = rio.transform.xy(raster_one.transform, [x],[y])
        long, lat = long[0], lat[0]
        testx = (long>=raster_two.bounds[0]-xmargin and 
                     long<=raster_two.bounds[1]+xmargin)
        testy = (lat>=raster_two.bounds[2]-ymargin and 
                     lat<=raster_two.bounds[3]+ymargin)
        yinbound = testx and testy
        if not testx: y -= 1
        if not testy: x -= 1
    
    bottomright = [x,y]     
    if not yinbound:
        logging.error("Failure finding overlap")    
  
    return [topleft, bottomright]

def getinbound(raster_one, raster_two):
    """
    x = raster_one.width - 1
    y = raster_one.height - 1
    yinbound = False
    while not yinbound and x>=0 :
        long, lat = rio.transform.xy(raster_one.transform, [x],[y])
        long, lat = long[0], lat[0]

        testx = (long>=raster_two.bounds[0]-xmargin and 
                     long<=raster_two.bounds[1]+xmargin)
        yinbound = testx and testy
        if not testx: x-=1
        if not testy: y-=1
    if not yinbound:
        logging.error("Failure finding overlap")    
    bottomright = [x,y]
    """

    return



def _big_raster_coords(raster_one, raster_two, raster_one_coords):
    """This returns the coordinates of raster two(big raster) the overlap small
    raster.
    TODO: I need to get both the x,y and the transform in order to update the 
    result's profile ---- NOT
    # I just need the coords (row,col) of raster result that need to be updated
    # Inbound coords ensures raster_one_coords overlap rastertwo. However, they
    may overlap at -1 first. In this case +=1
    """
    top, bot = raster_one_coords
    
    unpack = lambda x : [item for sublist in x for item in sublist]
    addone = lambda x : [item+1 if item==-1 else item for item in x]
    long, lat = rio.transform.xy(raster_one.transform, top[0],top[1])
    raster_two_top = unpack(rio.transform.rowcol(raster_two.transform, 
                                                             [long],[lat]))
    
    long, lat = rio.transform.xy(raster_one.transform, bot[0], bot[1])
    raster_two_bot = unpack(rio.transform.rowcol(raster_two.transform, 
                                                             [long],[lat]))
    raster_two_top = addone(raster_two_top)
    raster_two_bot = addone(raster_two_bot)
    return [raster_two_top, raster_two_bot]

def gen_overlap(npixels, smallstep, bigstep, offset):
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
        
        
    Note:
        Xone - offset
    """
    for cellnumber in range(npixels):
        inside = ((cellnumber*smallstep)-offset)//bigstep== \
                    (((cellnumber+1)*smallstep)-offset)//bigstep
        if inside:
            yield [100,0]
        else:
            nbigsteps = (((cellnumber+1)*smallstep)- offset )//bigstep
            temp =  (bigstep * nbigsteps) - ((cellnumber*smallstep)-offset)
            left = 100*temp /smallstep
            right = 100 - left
            
            yield [left, right]

def get_offset(raster_one, raster_two, topone, toptwo):
    """Computes the x and y offfset between rasters in projection units.
    Parameters
    ----------
    raster_one, raster_two: obj, Raster
        
    top : list, [int, int]
        coordinates of the top left overlapping pixel from raster one.
    """
    # TODO : not completely clean : raster_two also should have a top pixel.
    coords_one = rio.transform.xy(raster_one.transform, topone[0], 
                                                      topone[1], offset='ul')
    coords_two = rio.transform.xy(raster_two.transform, toptwo[0], 
                                                      toptwo[1], offset='ul')
    
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

def extend_values(rasterone, shape, x_indices, y_indices, topone, botone):
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
    data = rasterone.read()[topone[0]:botone[0], topone[1]:botone[1]]
    values_extended = np.ones(shape)-1
    logging.debug("Begin x extension ")
    logging.debug("Shape values_extended : ", values_extended.shape)
    
    for i, value_line in enumerate(data):
        values_extended[i] = np.insert(value_line, y_indices, 
                                                value_line[y_indices])
    logging.debug("x extension done")
    for i, value_column in enumerate(
                        values_extended.T[0:data.shape[0]+x_indices.size,:]):
        values_extended.T[i] = np.insert(value_column, x_indices, 
                         value_column[x_indices])[0:values_extended.shape[0]]
    return values_extended



if __name__ =="__main__":
    print("Hello")
    #my test
#    rasterone = "/home/eric/DATA/Trinidad/DATA/HANSEN/reproj/tile/tile_rep_Hansen_GFC-2018-v1.6_lossyear_BINARY_2015-2018.tif"
    rasterone = "/home/eric/DATA/project_r2intersect/DATA/tile_rep_Hansen_GFC-2018-v1.6_lossyear_BINARY.tif"
    rastertwo = "/home/eric/DATA/project_r2intersect/DATA/V45_10pix_1supports_50badpixls_mean.tif"
    out = "/home/eric/DATA/Trinidad/DATA/ROVERLAP/"
    name= "overlap_loss_2015-2018_v2"


    #test marc
#    rasterone = "/home/eric/DATA/project_r2intersect/DATA/marctest/high_res.tif"
#    rastertwo = "/home/eric/DATA/project_r2intersect/DATA/marctest/low_res.tif"
#    out = "/home/eric/DATA/Trinidad/DATA/ROVERLAP/"
#    name= "overlap_marc"
    main(rasterone, rastertwo, output=out, name=name)
    """
    import sys
    sys.path.append("/home/eric/DATA/scripts/NOTMINE/RasterIntersect/raster-intersection/raster_intersection")
    import raster_intersection
    
    hrs = rasterone
    lrs = rastertwo
    dst = out+name
    nomen = "/home/eric/DATA/project_r2intersect/DATA/marctest/nomenclature.txt"
    marc(hrs, lrs, dst, nomen=nomen)
    """
    
    
    
    
    