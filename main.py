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
import helpers

def main(rasterone, rastertwo, output=None, area=True, ilus=True, name=None):
    """from two rasters with same projection and rotation parameters, but
    with different resolution, computes the % of overlap or 
    weighted mean in each pixel of the raster with less resolution.
    # TODO : IF rasters don't completely overlap : use [row,col] of overlap to
    fill result
    """    
    startTime = time.time()
    
    rasterone, rastertwo = helpers.set_rasters(rasterone, rastertwo)
    
    pof = helpers.check_rotation(rasterone, rastertwo)
    if pof==False:
        logging.error('Not same rotation on rasters')
        return
    else:
        pass
    # Get the top and bottom pixels for the overlapping region
    rasterone_coords, rastertwo_coords = helpers.get_overlap_coords(rasterone, rastertwo)
    topone, botone = rasterone_coords
    toptwo, bottwo = rastertwo_coords
    
    #n pixels has just the shape of the overlapping area
    npixels = [botone[0]+1 - topone[0], botone[1]-topone[1]+1]
    xoffset, yoffset = helpers.get_offset(rasterone, rastertwo, topone, toptwo)
    xoverlap = helpers.gen_overlap(npixels[0], rasterone.xstep, rastertwo.xstep, xoffset)
    yoverlap = helpers.gen_overlap(npixels[1], rasterone.ystep, rastertwo.ystep, yoffset)
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
    y_toduplicate = np.array([y[0]!=100 for y in overlap_y])
    
    #---------------------illus overlap
    #This is just an illustration : print out the raster showing if a pixel if
    # inside or overlaps two/four pixels.
    if ilus : helpers.illustrate_overlap(x_toduplicate, y_toduplicate)
    #-----------------------Continuing
    if xoffset > 0:
        x_toduplicate[0] = False 
    if yoffset > 0:
        y_toduplicate[0] = False
    
    # weights computation: combination of x and y axis weights
    result = np.repeat(weights_x[np.newaxis,...], weights_y.size, axis=0)
    result_total = result.T * weights_y 
    weights_total = result_total /100
    
    x_indices = np.where(x_toduplicate==1)[0]
    y_indices = np.where(y_toduplicate==1)[0]
    
    # Extend values for them to have the shape as the weights
    datashape = rasterone.read().shape

    values_extended = helpers.extend_values(rasterone, result_total.shape,
                                    x_indices, y_indices)
        
    elapsedTime = time.time() - startTime
    print('function [{}] finished in {} ms'.format(
        'before big loop', int(elapsedTime * 1000)))
    
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
    
    top, bot = rastertwo_coords
    
    
    result = np.ones(npixels) - 1      
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
                # TODO : here, i, j have to be modified so that only overlapping
                # part is filled.
                result[i,j] = np.sum(values * weights)
            except IndexError: # end of raster 2
                pass

    if area:
        result = np.round(result * np.abs(rasterone.xstep*rasterone.ystep)  
                          / (np.abs(rastertwo.ystep*rastertwo.xstep)),2)*100
    
    # TODO: what I need here is to assign the results to the correct shape:
    endres = np.ones_like(rastertwo.read())
    endres[toptwo[0]:bottwo[0], toptwo[1]:bottwo[1]] = result
    
    if name is None:
        name = 'roverlap'
    if output is None:
        output = os.path.dirname(rasterone)
    output = os.path.join(output, name)
    profile = rastertwo.profile
    profile.update(dtype=rio.int32,
                   nodata=-1)
    with rio.open(output, mode='w', **profile) as dst:
        dst.write(np.expand_dims(endres.astype('int32'), axis=0))    
        
    
    elapsedTime = time.time() - startTime
    print('function [{}] finished in {} ms, or {} s'.format(
        'The End', int(elapsedTime * 1000), int(elapsedTime)))
    
    return

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
    
    
    
    
    