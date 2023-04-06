from dataclasses import dataclass
from typing import Optional, List

from tesspoint import TESSPointSCC
from . import PACKAGEDIR

import numpy as np
__all__ = ["TESSPoint"]

@dataclass
class TESSPoint(object):    
    id : Optional[list] = None # ID assosciated with each pixel/coordinate - could be TIC or arbitrary for bookeeping
    coord : Optional[list] = None # SkyCoord?
    ra : Optional[list] = None
    dec : Optional[list] = None
    row : Optional[list] = None
    column : Optional[list] = None
    targetname : Optional[list] = None
    filename : Optional[str] = None
    
    coord_type : Optional[str] = None # keep track of if we're going from radec-> pix or pix->radec 
    obsSector : Optional[list] = None # keep track of Sectors assosciated with transformations
    obsCamera : Optional[list] = None # keep track of Cameras assosciated with transformations
    obsCCD : Optional[list] = None # keep track of CCDs assosciated with transformations
    obsId : Optional[list] = None # Keep track of Ids assosciated with transformations
    _sector_max: Optional[int] = -1 # Will read from pointings.csv to grab the default max from the last line
    _sector_min: Optional[int] = 1
    _camera_max: Optional[int] = 4
    _camera_min: Optional[int] = 1
    _ccd_max: Optional[int] = 4
    _ccd_min: Optional[int] = 1
    
    def __post_init__(self):
        if self.targetname is not None:
            self._read_name()
        elif self.filename is not None:
            self._read_csv()
        elif self.coord is not None:
            self._read_skycoord()
        if ((not self.column) and (not self.row)):
            # Assuming that we're working from Sky Coordinates and going to pixels now since we did not pass a pixel column, row 
            self.coord_type="radec"
            if(isinstance(self.ra,list)):
                self.ra=np.array(self.ra)
            if(isinstance(self.dec,list)):
                self.dec=np.array(self.dec)
        elif(len(self.column) == len(self.row)):
            # Assuming that we're working from Pixel and goint to Sky Coordinates since we did pass a pixel column, row 
            self.coord_type="pixels"
        else:
            raise ValueError("Input mismatch")# is this the flag we want?
        if(self._sector_max == -1):
            # Will read from pointings.csv to grab the default maximum sector from the last line
            with open(f"{PACKAGEDIR}/data/pointings.csv") as pointings_file:
                lines=pointings_file.readlines()
            self._sector_max == lines[-1].split(",")[0]
        self.validate()
    
    def _read_skycoord(self):
        if isinstance(self.coord, list):
            if isinstance(self.coord[0], SkyCoord):
                self.coord = SkyCoord(self.coord)
            else:
                raise ValueError("Must pass either a `astropy.coordinate.SkyCoord` object or a list of `astropy.coordinate.SkyCoord` objects to `coord`")
        elif not isinstance(self.coord, SkyCoord):
            raise ValueError("Must pass either a `astropy.coordinate.SkyCoord` object or a list of `astropy.coordinate.SkyCoord` objects to `coord`")
        if len(self.coord.shape) == 0:
            self.coord = SkyCoord([self.coord])
        self.ra, self.dec =  self.coord.ra.deg, self.coord.dec.deg
    
    def _read_name(self):
        if isinstance(self.targetname, str):
            c = SkyCoord.from_name(self.targetname)
            self.ra, self.dec = c.ra.deg, c.dec.deg
        elif isinstance(self.targetname, (list, np.ndarray)):
            self.ra, self.dec = [], []
            for name in self.targetname:
                c = SkyCoord.from_name(name)
                self.ra.append(c.ra.deg)
                self.dec.append(c.dec.deg)
    
    def _read_csv(self):
        df = pd.read_csv(self.filename)
        cols = np.asarray([c.lower().strip() for c in df.columns])
        if np.any(cols == 'col'):
            cols[cols == 'col'] = 'column'
        
        if not np.in1d(['ra', 'dec', 'row', 'column'], cols).any():
            raise ValueError('Must pass a dataframe with column headers of "ra", "dec", "column", or "row".')
        
        [setattr(self, attr, np.asarray(df[attr])) for attr in ['ra', 'dec', 'row', 'column'] if attr in cols]
        
        
    def validate(self):
        attrs = np.asarray(['ra', 'dec', 'row', 'column'])
        
        isnone = np.asarray([getattr(self, attr) is None for attr in attrs])
        # Passed in something
        if isnone.all():
            raise ValueError(f"Must pass either RA and Dec, Column and Row, a target name, or a filename.")

        if np.atleast_1d(np.where(~isnone)[0] == [0, 1]).all():
            self.coord_type = 'radec'
        elif np.atleast_1d(np.where(~isnone)[0] == [2, 3]).all():
            self.coord_type = 'pixels'
        else:
            raise ValueError("Must pass either RA and Dec, or Column and Row.")

        # Correct length
        valid_lengths = len(np.unique([len(np.atleast_1d(getattr(self, attr))) for attr in attrs[np.where(~isnone)]])) == 1
        if not valid_lengths:
            raise ValueError("Must pass arrays of the same length.")
        [setattr(self, attr, np.atleast_1d(getattr(self, attr))) for attr in attrs[np.where(~isnone)]]
        self.nvals = len(np.atleast_1d(getattr(self, attrs[np.where(~isnone)[0][0]])))
             
    def __len__(self):
        return self.nvals
    
    def __repr__(self):
        if self.coord_type == 'pixels':
            return f'TESSpoint ({self.nvals} Row/Column pairs)'
        elif self.coord_type == 'radec':
            return f'TESSpoint ({self.nvals} RA/Dec pairs)'
        else:
            return 'TESSpoint'
        
    def __getitem__(self, idx):
        if self.coord_type == 'radec':
            return TESSpoint(ra=self.ra[idx], dec=self.dec[idx])
        elif self.coord_type == 'pixels':
            return TESSpoint(row=self.row[idx], column=self.column[idx])
        else:
            raise ValueError('No `coord_type` set.')
        
            
    def to_RADec(self, sector:Optional[List] = None, camera:Optional[List] = None, ccd:Optional[List] = None):
        if sector == None:
            #By Default All Sectors
            sector = range(self._sector_max)+1
        if camera == None:
            # By Default All Cameras
            camera = range(self._camera_max)+1
        if ccd == None:
            # By Default All Cameras
            ccd = range(self._ccd_max)+1
        # Testing notes - test one happens if any single one of these is an int, wierd lists
        # What are we returning? sector, camera, ccd, 

        for sector_iter in sector:
            for camera_iter in camera:
                for ccd_iter in ccd:
                    SectorCameraCCD=TESSPointSCC(sector_iter,camera_iter,ccd_iter)
                    iter_targets=SectorCameraCCD.pix2radec([self.column, self.row].T)
                    fov_mask=iter_targets[1]
                    #if fov_mask.any():         
        raise NotImplementedError        
        #return df # with dates?
        
    def to_Pixel(self, sector:Optional[List] = None, camera:Optional[List] = None, ccd:Optional[List] = None):
        if sector == None:
            #By Default All Sectors
            sector = range(1,self._sector_max)
        if camera == None:
            # By Default All Cameras
            camera = range(1,self._camera_max)
        if ccd == None:
            # By Default All Cameras
            ccd = range(1,self._ccd_max)
        # Clear any prior calculations, many to many mapping issues, rethink, ugh
        self.obsId=[]
        self.obsSector=[]
        self.obsCamera=[]
        self.obsCCD=[]

        # Testing notes - test one happens if any single one of these is an int, wierd lists
        for sector_iter in sector:
            for camera_iter in camera:
                for ccd_iter in ccd:
                    SectorCameraCCD=TESSPointSCC(sector_iter,camera_iter,ccd_iter)
                    iter_targets=SectorCameraCCD.radec2pix(np.array([self.ra, self.dec]).T)
                    fov_mask=iter_targets[0]
                    if(fov_mask.any()):
                        n_mask=self.ra[fov_mask].count_nonzero()
                        #tess_stars2px.py has ecliptic long, lat, figure that out
                        #iter_output=self.id[mask],self.ra[mask],self.dec[mask],iter_targets[0][0][mask],iter_targets[0][1][mask]
                        self.obsId.append(self.id[fov_mask])
                        self.obsSector.append([sector_iter]*n_mask)
                        self.obsCamera.append([camera_iter]*n_mask)
                        self.obsCCD.append([ccd_iter]*n_mask)
                        self.column.append(iter_targets[1][:,0][fov_mask])
                        self.row.append(iter_targets[1][:,1][fov_mask])
        return self.obsId,self.obsSector,self.obsCamera,self.obsCCD,self.column,self.row
        #return df # with dates?
        
    def ObservabilityMask(self, sectors:Optional[List] = None, cycle:Optional[List] = None):  
        raise NotImplementedError
        #return np.ndarray of bools
        
    def NumberOfObservations(self, sectors:Optional[List] = None, cycle:Optional[List] = None):  
        raise NotImplementedError
        #return np.ndarray of ints