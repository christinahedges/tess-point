from tesspoint import TESSPoint, footprint
import pytest
import numpy as np
import pandas as pd

RowOffset=1
ColOffset=45
PixAccuracy = 1
WCSAccuracy = PixAccuracy * 0.015000 # Pixel Accuracy * pixel scale

def assert_str(type, Sector, Camera, CCD, maxdev, Accuracy):
    return type+" Accuracy vs Benchmark less than requirement: Sector:{} Camera:{} CCD{} Maximum Deviation: {} Accuracy Requirement{}".format(Sector,Camera,CCD,maxdev,Accuracy)

def compare_pix2radec(Sector, Camera, CCD):
    '''This will compare our pixel -> WCS calculation for a single Sector, Camera, CCD against a benchmark 
    csv that can be found in a separate repository'''

    footprint_url="https://raw.githubusercontent.com/tylerapritchard/TESSPoint_CreateTestFiles/main/footprint_input.dat"
    benchmark_url="https://raw.githubusercontent.com/tylerapritchard/TESSPoint_CreateTestFiles/main/testfiles/TEST_pix2radec_Sec{:02d}_Cam{}_CCD{}_stars2px.dat".format(Sector,Camera,CCD)

    footprint_pix=pd.read_csv(footprint_url, delimiter=' ',names=['tic','col','row'], index_col=False)
    benchmark_radec=pd.read_csv(benchmark_url,delimiter=' ',names=['tic','ra','dec'], skiprows=1, index_col=False)


    benchmark_radec['index']=benchmark_radec['tic']
    tp=TESSPoint(Sector,Camera,CCD)
    footprint_pix_arr=np.array([footprint_pix.col.to_numpy()-ColOffset,footprint_pix.row.to_numpy()-RowOffset]).T
    footprint_radec_arr=tp.pix2radec(footprint_pix_arr)

    footprint_radec = pd.DataFrame({'tic':footprint_pix.tic.to_numpy(),
                            'ra':footprint_radec_arr[0],
                            'dec':footprint_radec_arr[1]})
    footprint_radec['index'] = footprint_radec['tic']

    idx = footprint_radec.index.intersection(benchmark_radec.index)

    d_ra = abs(footprint_radec.loc[idx, 'ra']  - benchmark_radec.loc[idx, 'ra'])
    d_dec= abs(footprint_radec.loc[idx, 'dec'] - benchmark_radec.loc[idx, 'dec'])

    return max([max(d_ra), max(d_dec)])

def compare_radec2pix(Sector, Camera, CCD):
    '''This will compare our WCS -> Pixel calculation for a single Sector, Camera, CCD against a benchmark 
    csv that can be found in a separate repository'''

    footprint_url="https://raw.githubusercontent.com/tylerapritchard/TESSPoint_CreateTestFiles/main/testfiles/TEST_pix2radec_Sec{:02d}_Cam{}_CCD{}_stars2px.dat".format(Sector,Camera,CCD)
    benchmark_url="https://raw.githubusercontent.com/tylerapritchard/TESSPoint_CreateTestFiles/main/testfiles/TEST_radec2pix_Sec{:02d}_Cam{}_CCD{}_stars2px.dat".format(Sector,Camera,CCD)

    footprint_radec=pd.read_csv(footprint_url, delimiter=' ', names=['tic','ra','dec'], skiprows=1,index_col=False)
    footprint_radec['index']=footprint_radec['tic']

    tp=TESSPoint(Sector,Camera,CCD)

    footprint_radec_arr=np.array([footprint_radec.ra.to_numpy(),
                                  footprint_radec.dec.to_numpy()]).T

    footprint_pix_arr=tp.radec2pix(footprint_radec_arr)

    footprint_pix = pd.DataFrame({'tic':footprint_radec.tic.to_numpy(),
                                  'row':footprint_pix_arr[0][:,0],
                                  'col':footprint_pix_arr[0][:,1]})
    footprint_pix['index'] = footprint_pix['tic']

    benchmark_pix = pd.read_csv(benchmark_url,names=['tic','ra','dec','el','ela','s','c','ccd','row','col','edge'],
                         index_col=0, skiprows=16,delimiter="|" )
    idx = footprint_pix.index.intersection(benchmark_pix.index)

    d_col = abs(footprint_pix.loc[idx, 'col'] - benchmark_pix.loc[idx, 'col'])
    d_row = abs(footprint_pix.loc[idx, 'row'] - benchmark_pix.loc[idx, 'row'])

    return np.median([d_col, d_row])

def test_pix2radec_AllCameraSectorCCD():
    '''this will test the pixel->WCS calculation for all Sector / Camera / CCD's 
    against a benchmark repository that was made from the previous version of tesspoint'''
    sector_list=range(1,69)
    camera_list=range(1,4)
    ccd_list=range(1,4)
    for sector in sector_list:
        for camera in camera_list:
            for ccd in ccd_list:
                maxdev=compare_pix2radec(sector, camera, ccd)
                assert maxdev < WCSAccuracy, assert_str('Pix->WCS',sector, camera, ccd, maxdev, WCSAccuracy)

def test_radec2pix_AllCameraSectorCCD():
    '''this will test the pixel->WCS calculation for all Sector / Camera / CCD's 
    against a benchmark repository that was made from the previous version of tesspoint'''
    sector_list=range(1,69)
    camera_list=range(1,4)
    ccd_list=range(1,4)
    for sector in sector_list:
        for camera in camera_list:
            for ccd in ccd_list:
                maxdev=compare_radec2pix(sector, camera, ccd)
                assert maxdev < WCSAccuracy, assert_str('WCS->PIX Median',sector, camera, ccd, maxdev, WCSAccuracy)