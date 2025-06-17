import numpy as np
from pyhdf.SD import SD
from pyhdf.VS import *
from pyhdf.HDF import *

def readMod06(file):
    '''
    Desc: read mod1km
    Param: 
    Return: 
    '''    
    ds = SD(file)

    attributes = ds.select('Cloud_Optical_Thickness').attributes() 
    scales = attributes['scale_factor']       
    offset = attributes['add_offset']     
    cot = ds.select('Cloud_Optical_Thickness')[:]   # 1353
    cotMask = cot==-9999
    cot = scales * (cot - offset)
    cot[cotMask] = -999

    attributes2 = ds.select('Cloud_Effective_Radius').attributes()
    scales2 = attributes2['scale_factor']       
    offset2 = attributes2['add_offset']     
    cer = ds.select('Cloud_Effective_Radius')[:]
    cerMask = cer==-9999
    cer = scales2 * (cer - offset2)
    cer[cerMask] = -999
    
    attributes3 = ds.select('cloud_top_temperature_1km').attributes() 
    scales3 = attributes3['scale_factor']      
    offset3 = attributes3['add_offset']     
    ctt = ds.select('cloud_top_temperature_1km')[:]
    cttMask = (ctt==-32767)
    ctt = scales3 * (ctt - offset3)
    ctt[cttMask] = -999
    

    attributes4 = ds.select('cloud_top_height_1km').attributes() 
    scales4 = attributes4['scale_factor']      
    offset4 = attributes4['add_offset']     
    cth = ds.select('cloud_top_height_1km')[:]
    cthMask = (cth==-32767)
    cth = scales4 * (cth - offset4)
    cth = cth/1000
    cth[cthMask] = -999

    phase = ds.select('Cloud_Phase_Infrared_1km')[:]
    phase = phase.astype(np.float32)
    phase[phase==127] = -999
    # phase[phase==25] = -999

    cer = cer.astype(np.float32)
    cot = cot.astype(np.float32)
    cth = cth.astype(np.float32)

    ds.end()

    return cth, cer, cot, phase, ctt