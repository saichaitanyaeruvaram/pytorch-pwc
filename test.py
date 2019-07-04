#!/usr/bin/env python

import torch
import cv2
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
from flownwarp import estimate, warp_flow
import f2i

print_frequency = 10
str_flofolder = 'flow'
str_f2ifolder = 'f2i'
str_warpfolder = 'warp'
str_outfileprefix = 'flow_'
arguments_str_fileprefix = 'frame_'
arguments_str_infolder = './'
arguments_str_outfolder = './'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--infolder' and strArgument != '': arguments_str_infolder = strArgument # input folder which has image sequences
	if strOption == '--outfolder' and strArgument != '': arguments_str_outfolder = strArgument # output folder
	if strOption == '--fileprefix' and strArgument != '': arguments_str_fileprefix = strArgument # file prefix
# end

def sortfilenames(val):
    return int(val.split(arguments_str_fileprefix)[1].split('.')[0])

if __name__ == '__main__':

    img_seq_arr = [f for f in os.listdir(arguments_str_infolder) if os.path.isfile(os.path.join(arguments_str_infolder, f)) and f.find(arguments_str_fileprefix) != -1]
    img_seq_arr.sort(key=sortfilenames)
    print('<', len(img_seq_arr), '> match found')
    
    if len(img_seq_arr) < 2:
        raise('Atleast 2 images must be present')
    # end
    
    if not os.path.exists(arguments_str_outfolder):
        os.mkdir(arguments_str_outfolder)
    # end
    
    str_out_flofolder = os.path.join(arguments_str_outfolder, str_flofolder)
    if not os.path.exists(str_out_flofolder):
        os.mkdir(str_out_flofolder)
    # end
    
    str_out_f2ifolder = os.path.join(arguments_str_outfolder, str_f2ifolder)
    if not os.path.exists(str_out_f2ifolder):
        os.mkdir(str_out_f2ifolder)
    # end
    
    str_out_warpfolder = os.path.join(arguments_str_outfolder, str_warpfolder)
    if not os.path.exists(str_out_warpfolder):
        os.mkdir(str_out_warpfolder)
    # end    
    
    flow = f2i.Flow()
        
    for index, filename in enumerate(img_seq_arr):
        cur_framenumber = filename.split(arguments_str_fileprefix)[1].split('.')[0]
        img_np = numpy.array(PIL.Image.open(os.path.join(arguments_str_infolder, filename)))
        cur_tensor = torch.FloatTensor(img_np[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
        
        if index == 0:
            prev_framenumber = cur_framenumber
            prev_tensor = cur_tensor
            continue
        # end        
        
        tensorOutput = estimate(prev_tensor, cur_tensor)
        flow_np = numpy.array(tensorOutput.numpy().transpose(1, 2, 0), numpy.float32)
        
        outfilepath = os.path.join(str_out_flofolder, str_outfileprefix+prev_framenumber+'_'+cur_framenumber+'.flo')
        objectOutput = open(outfilepath, 'wb')

        numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objectOutput)
        numpy.array([ tensorOutput.size(2), tensorOutput.size(1) ], numpy.int32).tofile(objectOutput)
        numpy.array(flow_np).tofile(objectOutput)
        objectOutput.close()
                              
        # flow to image
        outfilepath = os.path.join(str_out_f2ifolder, str_outfileprefix+prev_framenumber+'_'+cur_framenumber+'.jpg')
        img = flow._flowToColor(flow_np)
        cv2.imwrite(outfilepath, img)
        
        # warping        
        outfilepath = os.path.join(str_out_warpfolder, str_outfileprefix+prev_framenumber+'_'+cur_framenumber+'.jpg')
        im2w = warp_flow(img_np, flow_np)
        cv2.imwrite(outfilepath, im2w)
                                       
        prev_tensor = cur_tensor
        prev_framenumber = cur_framenumber
        
        if index % print_frequency == 0:
            print('processed<', index, '>')
        # end
        
    # end
    
# end