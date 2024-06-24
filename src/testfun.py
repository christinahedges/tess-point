import pandas as pd
import numpy as np
from tesspoint import TESSPoint

def test_pix2radec_SectorCameraCCD(SectorCameraCCD):
    Sector, Camera, CCD = SectorCameraCCD
    
    fprfile="/Users/tapritc2/tessgi/tesspoint/TESSPoint_CreateTestFiles/footprint_input.dat"
    footprint_df=pd.read_csv(fprfile,delimiter=' ',names=['tic','x','y'],index_col=False)

    pointing=TESSPoint(Sector,Camera,CCD)
    
    farr=np.array([footprint_df.x.to_numpy(),footprint_df.y.to_numpy()]).T
    footprint_radec=pointing.pix2radec(farr)
    
    test_df = pd.DataFrame({'tic':footprint_df.tic.to_numpy(),
                            'ra':footprint_radec[0],
                            'dec':footprint_radec[1]})
    test_df['index']=test_df['tic']
    return test_df


def test_pix2radec_benchmark_SectorCameraCCD(SectorCameraCCD):
    Sector, Camera, CCD = SectorCameraCCD

    dir_testfiles='/Users/tapritc2/tessgi/tesspoint/TESSPoint_CreateTestFiles/testfiles'
    wcsfile=dir_testfiles+"/TEST_pix2radec_Sec{:02d}_Cam{}_CCD{}_stars2px.dat".format(Sector,Camera,CCD)

    test_df = test_pix2radec_SectorCameraCCD(SectorCameraCCD)
    benchmark_df = pd.read_csv(wcsfile,delimiter=' ',names=['tic','ra','dec'],skiprows=1,index_col=False)

    idx = test_df.index.intersection(benchmark_df.index)
    dra = test_df.loc[idx, 'ra']  - benchmark_df.loc[idx, 'ra']
    ddec= test_df.loc[idx, 'dec'] - benchmark_df.loc[idx, 'dec']
    
    return idx, dra, ddec, Sector, Camera, CCD

def test_radec2pix_SectorCameraCCD(SectorCameraCCD):
    Sector, Camera, CCD = SectorCameraCCD
    
    dir_testfiles='/Users/tapritc2/tessgi/tesspoint/TESSPoint_CreateTestFiles/testfiles'
    wcsfile=dir_testfiles+"/TEST_pix2radec_Sec{:02d}_Cam{}_CCD{}_stars2px.dat".format(Sector,Camera,CCD)
    
    input_df=pd.read_csv(wcsfile,delimiter=' ',names=['tic','x','y'],skiprows=1,index_col=False)
    input_arr=np.array([input_df.x.to_numpy(),input_df.y.to_numpy()]).T

    pointing=TESSPoint(Sector,Camera,CCD)
        
    output_pix=pointing.radec2pix(input_arr)
    
    output_df = pd.DataFrame({'tic':input_df.tic.to_numpy(),
                            'row':output_pix[0][:,0],
                            'col':output_pix[0][:,1]})
    output_df['index']=output_df['tic']
    
    return output_df

def test_radec2pix_benchmark_SectorCameraCCD(SectorCameraCCD):
    Sector, Camera, CCD = SectorCameraCCD
    
    dir_testfiles='/Users/tapritc2/tessgi/tesspoint/TESSPoint_CreateTestFiles/testfiles'
    
    pixfile=dir_testfiles+"/TEST_radec2pix_Sec{:02d}_Cam{}_CCD{}_stars2px.dat".format(Sector,Camera,CCD)

    output_pix_df = test_radec2pix_SectorCameraCCD(SectorCameraCCD)
    benchmark_pix_df =  pd.read_csv(pixfile,names=['tic','ra','dec','el','ela','s','c','ccd','row','col','edge'],
                         index_col=0, skiprows=16,delimiter="|" )

    idx = output_pix_df.index.intersection(benchmark_pix_df.index)
    
    dr = output_pix_df.loc[idx, 'row'] - benchmark_pix_df.loc[idx, 'row']
    dc = output_pix_df.loc[idx, 'col'] - benchmark_pix_df.loc[idx, 'col']
    
    return idx, dr, dc, Sector, Camera, CCD

def test_radec2pix_initial_SectorCameraCCD(SectorCameraCCD):
    Sector, Camera, CCD = SectorCameraCCD
    
    dir_testfiles='/Users/tapritc2/tessgi/tesspoint/TESSPoint_CreateTestFiles/testfiles'
    fprfile="/Users/tapritc2/tessgi/tesspoint/TESSPoint_CreateTestFiles/footprint_input.dat"    
    pixfile=dir_testfiles+"/TEST_radec2pix_Sec{:02d}_Cam{}_CCD{}_stars2px.dat".format(Sector,Camera,CCD)

    output_pix_df = test_radec2pix_SectorCameraCCD(SectorCameraCCD)
    init_pix_df=pd.read_csv(fprfile,delimiter=' ',names=['tic','row','col'],index_col=False)
    init_pix_df['index']=init_pix_df['tic']

    idx = output_pix_df.index.intersection(init_pix_df.index)
    
    dr = output_pix_df.loc[idx, 'row'] - init_pix_df.loc[idx, 'row']
    dc = output_pix_df.loc[idx, 'col'] - init_pix_df.loc[idx, 'col']
    
    return idx, dr, dc, Sector, Camera, CCD
