from tesspoint import TESSPoint, footprint
import pytest
import numpy as  np

RowOffset=1
ColOffset=45
Accuracy = 0.01

def test_pix2radec():
    ''' Test that pix2radec runis in general
    To DO: We need to add type checks to output 
    (and maybe standardize pix2radec and radec2pix output)'''
    tp = TESSPoint(1, 1, 1)
    tp.pix2radec(footprint())

def conversion(Sector, Camera, CCD):
    '''General funtion that takes a pixel footprint, converts to radec for a given s/c/c,
    then converts ra, dec back to pixels.  
    This then checks the difference between the input pixels and the output pixels.  
    This will be used in many future tests.  '''
    tp = TESSPoint(Sector, Camera, CCD)
    radec_footprint=tp.pix2radec(footprint())
    pix_footprint=tp.radec2pix(np.array([radec_footprint[0],radec_footprint[1]]).T)
    deviation = footprint()-pix_footprint[0]
    maxdev=max(max(deviation[:,0] + ColOffset), max(deviation[:,1] + RowOffset))
    return maxdev

def assert_str(Sector, Camera, CCD, maxdev, Accuracy):
    return "Accuracy less than requirement: Sector:{} Camera:{} CCD{} Maximum Deviation: {} Accuracy Requirement{}".format(Sector,Camera,CCD,maxdev,Accuracy)

def test_radec2pix():
    '''Test a single sector/camera/ccd without an accuracy requirement to make sure we can 
    convert in both directions'''
    conversion(1,1,1)
    
def test_conversion():
    '''Test a single sector /camera / ccd with an accuracy requirement'''
    maxdev=conversion(1,1,1)
    assert maxdev < Accuracy, assert_str(Sector, Camera, CCD, maxdev, Accuracy)

def test_converstion_AllSectorCameraCCD():
    '''Test all sector / camera / ccd's with an accuracy requirement'''
    sector_list=range(1,69)
    camera_list=range(1,4)
    ccd_list=range(1,4)
    for sector in sector_list:
        for camera in camera_list:
            for ccd in ccd_list:
                maxdev=conversion(sector, camera, ccd)
                assert maxdev < Accuracy, assert_str(sector, camera, ccd, maxdev, Accuracy)