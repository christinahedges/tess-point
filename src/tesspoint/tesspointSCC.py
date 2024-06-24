"""Python vectorized version of tess-point"""
import numpy as np
import logging
from dataclasses import dataclass
import scipy.optimize as opt
from . import PACKAGEDIR
__all__ = ["TESSPointSCC", "footprint"]

# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.warning)

pointings = {
    key: col
    for col, key in zip(
        np.loadtxt(f"{PACKAGEDIR}/data/pointings.csv", delimiter=",", skiprows=1).T,
        np.loadtxt(
            f"{PACKAGEDIR}/data/pointings.csv", delimiter=",", max_rows=1, dtype=str
        ),
    )
}

tess_params = {
    1: {
        "ang1": 0.101588,
        "ang2": -36.022035,
        "ang3": 90.048315,
        "fl": 145.948116,
        "opt_coef1": 1.00000140,
        "opt_coef2": 0.24779006,
        "opt_coef3": -0.22681254,
        "opt_coef4": 10.78243356,
        "opt_coef5": -34.97817276,
        "x0_ccd1": 31.573417,
        "y0_ccd1": 31.551637,
        "ang_ccd1": 179.980833,
        "x0_ccd2": -0.906060,
        "y0_ccd2": 31.536148,
        "ang_ccd2": 180.000000,
        "x0_ccd3": -31.652818,
        "y0_ccd3": -31.438350,
        "ang_ccd3": -0.024851,
        "x0_ccd4": 0.833161,
        "y0_ccd4": -31.458180,
        "ang_ccd4": 0.001488,
    },
    2: {
        "ang1": -0.179412,
        "ang2": -12.017260,
        "ang3": 90.046500,
        "fl": 145.989933,
        "opt_coef1": 1.00000140,
        "opt_coef2": 0.24069345,
        "opt_coef3": 0.15391120,
        "opt_coef4": 4.05433503,
        "opt_coef5": 3.43136895,
        "x0_ccd1": 31.653635,
        "y0_ccd1": 31.470291,
        "ang_ccd1": 180.010890,
        "x0_ccd2": -0.827405,
        "y0_ccd2": 31.491388,
        "ang_ccd2": 180.000000,
        "x0_ccd3": -31.543794,
        "y0_ccd3": -31.550699,
        "ang_ccd3": -0.006624,
        "x0_ccd4": 0.922834,
        "y0_ccd4": -31.557268,
        "ang_ccd4": -0.015464,
    },
    3: {
        "ang1": 0.066596,
        "ang2": 12.007750,
        "ang3": -89.889085,
        "fl": 146.006602,
        "opt_coef1": 1.00000140,
        "opt_coef2": 0.23452229,
        "opt_coef3": 0.33552009,
        "opt_coef4": 1.92009863,
        "opt_coef5": 12.48880182,
        "x0_ccd1": 31.615486,
        "y0_ccd1": 31.413644,
        "ang_ccd1": 179.993948,
        "x0_ccd2": -0.832993,
        "y0_ccd2": 31.426621,
        "ang_ccd2": 180.000000,
        "x0_ccd3": -31.548296,
        "y0_ccd3": -31.606976,
        "ang_ccd3": 0.000298,
        "x0_ccd4": 0.896018,
        "y0_ccd4": -31.569542,
        "ang_ccd4": -0.006464,
    },
    4: {
        "ang1": 0.030756,
        "ang2": 35.978116,
        "ang3": -89.976802,
        "fl": 146.039793,
        "opt_coef1": 1.00000140,
        "opt_coef2": 0.23920416,
        "opt_coef3": 0.13349450,
        "opt_coef4": 4.77768896,
        "opt_coef5": -1.75114744,
        "x0_ccd1": 31.575820,
        "y0_ccd1": 31.316510,
        "ang_ccd1": 179.968217,
        "x0_ccd2": -0.890877,
        "y0_ccd2": 31.363511,
        "ang_ccd2": 180.000000,
        "x0_ccd3": -31.630470,
        "y0_ccd3": -31.716942,
        "ang_ccd3": -0.024359,
        "x0_ccd4": 0.824159,
        "y0_ccd4": -31.728751,
        "ang_ccd4": -0.024280,
    },
}


@dataclass
class TESSPointSCC:
    sector: int
    camera: int
    ccd: int

    def __post_init__(self):
        xeul = np.hstack(
            [
                (np.pi / 180.0) * pointings["ra"][pointings["sector"] == self.sector],
                np.pi / 2.0
                - (np.pi / 180.0)
                * pointings["dec"][pointings["sector"] == self.sector],
                (np.pi / 180.0) * pointings["roll"][pointings["sector"] == self.sector]
                + np.pi,
            ]
        )

        self.rmat1 = eulerm323(xeul)
        eulcam = np.asarray(
            [tess_params[self.camera][f"ang{idx}"] for idx in np.arange(1, 4)]
        )
        self.rmat2 = eulerm323(eulcam * (np.pi / 180.0))
        self.rmat4 = np.matmul(self.rmat2, self.rmat1)

    @property
    def opt_coeffs(self):
        return np.asarray(
            [
                tess_params[self.camera][key]
                for key in np.hstack(
                    [["fl"], [f"opt_coef{idx}" for idx in np.arange(1, 6)]]
                )
            ]
        )

    def pix_to_mm(self, coords):
        """convert pixel to mm focal plane position"""
        pixsz = 0.015000
        angle = -tess_params[self.camera][f"ang_ccd{self.ccd}"]
        xyb = xyrotate(angle, (coords + 0.5) * pixsz)
        return np.vstack(
            [
                xyb[:, 0] + tess_params[self.camera][f"x0_ccd{self.ccd}"],
                xyb[:, 1] + tess_params[self.camera][f"y0_ccd{self.ccd}"],
            ]
        ).T

    def pix2radec(self, coords):
        
        # Following tess_stars2pix.py legacy version, we're assuming that (1,1) is our corner science pixel
        RowOffset=1    # 1-to-0 indexing
        ColOffset=45   ## 44 collateral pixels + 1-to-0 indexing
        coords = coords - np.array([[ColOffset],[RowOffset]]).T

        xyfp = self.pix_to_mm(coords)
        lng_deg, lat_deg = fp_optics(xyfp, self.opt_coeffs)
        vcam = np.asarray(sphereToCart(lng_deg, lat_deg)).T
        curVec = np.matmul(self.rmat4.T, vcam.T).T
        ra, dec = cartToSphere(curVec)
        return ra / (np.pi / 180.0), dec / (np.pi / 180.0)

    def mm_to_pix(self, xy):
        """Convert focal plane to pixel location also need to add in the
            auxillary pixels added into FFIs """
        #created xy_to_ccdpix function to minimize repeated math
        #re-indiced ccd for 1-4 vs 03 given the the change in tess_param
        CCDWD_T = 2048
        CCDHT_T = 2058
        ROWA = 44
        ROWB = 44
        COLDK_T = 20
        fitpx = np.zeros_like(xy)
        if xy[0] >= 0.0:
            if xy[1] >= 0.0:
                self.ccd = 1
                ccdpx = xy_to_ccdpx(self, xy,)
                fitpx[0] = (CCDWD_T - ccdpx[0]) + CCDWD_T + 2 * ROWA + ROWB - 1.0
                fitpx[1] = (CCDHT_T - ccdpx[1]) + CCDHT_T + 2 * COLDK_T - 1.0
            else:
                self.ccd = 4
                ccdpx = xy_to_ccdpx(self, xy)
                fitpx[0] = ccdpx[0] + CCDWD_T + 2 * ROWA + ROWB
                fitpx[1] = ccdpx[1]
        else:
            if xy[1] >= 0.0:
                self.ccd = 2
                ccdpx = xy_to_ccdpx(self, xy)
                fitpx[0] = (CCDWD_T - ccdpx[0]) + ROWA - 1.0
                fitpx[1] = (CCDHT_T - ccdpx[1]) + CCDHT_T + 2 * COLDK_T - 1.0
            else:
                self.ccd = 3
                ccdpx = xy_to_ccdpx(self, xy)
                fitpx[0] = ccdpx[0] + ROWA
                fitpx[1] = ccdpx[1]
        return ccdpx, fitpx

    def radec2pix(self, coords):
        """ After the rotation matrices are defined to the actual
            ra and dec to pixel coords mapping
        """
        # removed pointing check since its now in the package
        # vectorized
        # like previous, doesn't return anything for things not in fov
        # Old version - iterates each camera, checks FoV, assigns a ccd to them
        # not sure how this works with the current OO-version,
        # we're sending an array of ras,decs
        # decs but have camera, ccd as an int property
        #preserve vectorization - check each array against Fov, iterate per camera?
        deg2rad = np.pi / 180.0
        curVec = np.asarray(sphereToCart(coords[:,0],coords[:,1]),dtype=np.double)
        logging.debug("radec2pix: curVec: {0}".format(curVec.T))
        logging.debug("radec2pix: curVec Shape: {0}".format(curVec.T.shape))

        camVec = np.matmul(self.rmat4, curVec)
        logging.debug("radec2pix: camVec: {0}".format(camVec))
        logging.debug("radec2pix: camVec Shape: {0}".format(camVec.shape))

        lng, lat = cartToSphere(camVec.T)
        lng = lng / deg2rad
        lat = lat / deg2rad
        logging.debug("radec2pix: lng: {0}".format(lng))
        logging.debug("radec2pix: lat: {0}".format(lat))

        g=np.vectorize(star_in_fov) # this is lazy, look at re-writing star_in_fov
        # Actually, in our current spec where we know sector, camera, ccd
        # is this a nescessary check? move to parent class?

        cut = g(lng,lat)

        if(cut.any()):
            lng=lng[cut]
            lat=lat[cut]
            xyfp = optics_fp(lng, lat, self.opt_coeffs)
            logging.debug("radec2pix: xyfp: {0}".format(xyfp))
            logging.debug("radec2pix: xyfp Shape: {0}".format(xyfp.shape))
            ccdpx, fitpix = mm_to_pix(self,xyfp)
            logging.debug("radec2pix: xyfp: {0}".format(xyfp))
            logging.debug("radec2pix: ccdpx: {0}".format(ccdpx))
            logging.debug("radec2pix: fitpx: {0}".format(fitpix))
            #return inCamera, ccdNum, fitsxpos, fitsypos, ccdxpos, ccdypos
            return fitpix2pix(fitpix,ccdpx)
        else:
            print('No specified targets in Field of View')
            return False
        
def footprint(npoints=50):
    """Gets the column and row points for CCD edges"""
    column = np.hstack(
        [
            np.zeros(npoints),
            np.linspace(0, 2048, npoints),
            np.linspace(0, 2048, npoints),
            np.ones(npoints) * 2048,
        ]
    )
    row = np.hstack(
        [
            np.linspace(0, 2048, npoints),
            np.zeros(npoints),
            np.ones(npoints) * 2048,
            np.linspace(0, 2048, npoints),
        ]
    )
    return np.vstack([column, row]).T


def xyrotate(angle, coords):
    ca = np.cos((np.pi / 180.0) * angle)
    sa = np.sin((np.pi / 180.0) * angle)
    return np.vstack(
        [ca * coords[:, 0] + sa * coords[:, 1], -sa * coords[:, 0] + ca * coords[:, 1]]
    ).T.astype(coords.dtype)


def rev_az_asym(coords):
    asymang = 0.0
    asymfac = 1.0
    xypa = xyrotate(asymang, coords) * np.asarray([1 / asymfac, 1])
    return xyrotate(-asymang, xypa)


def r_of_tanth(z, opt_coeffs):
    tanth = np.tan(z)
    rfp0 = tanth * opt_coeffs[0]
    rfp = np.sum(opt_coeffs[1:] * (tanth ** (2 * np.arange(5))[:, None]).T, axis=1)
    return rfp0 * rfp


def tanth_of_r_noscipy(rfp_times_rfp0, opt_coeffs):
    zi = np.arctan(rfp_times_rfp0 ** 0.5 / opt_coeffs[0])
    # Minimize...
    # This is a way to minimize that
    # 1) let's us minimize the whole vector and
    # 2) doesn't use scipy, so we could do something similar in other scripting languates
    # But it's not even close to optimal.
    # If you pass in a lot of points this might fill up your memory though...
    # ----
    bounds = (0, 0.55)
    resolution = 0.001
    x = np.arange(*bounds, resolution)[:, None] * np.ones((1, len(zi)))
    for count in range(3):
        minimize = np.asarray(
            [
                (r_of_tanth(zi + x[idx], opt_coeffs) - rfp_times_rfp0) ** 2
                for idx in range(x.shape[0])
            ]
        )
        argmin = np.argmin(minimize, axis=0)
        xmin = np.asarray([x[am, idx] for idx, am in enumerate(argmin)])
        # Every iteration, scale down the offset to be narrower around the minimum
        x = (x - xmin) * 0.25 + xmin
    # ----
    return xmin + zi

def tanth_of_r(rfp_times_rfp0, opt_coeffs):
    logging.debug("tanth_of_r: opt_coeffs: {0}".format(opt_coeffs))
    if np.abs(rfp_times_rfp0) > 1.0e-10:
        c0 = opt_coeffs[0]
        zi = np.arctan(np.sqrt(rfp_times_rfp0) / c0)
        def minFunc(z, opt_coeffs, rfp_times_rfp0):
            rtmp = r_of_tanth(z, opt_coeffs)
            return (rtmp - rfp_times_rfp0) ** 2
        
        optResult = opt.minimize(minFunc, [zi], \
                                 args=(opt_coeffs, rfp_times_rfp0), method='Nelder-Mead', \
                                 tol=1.0e-10, \
                                 options={'maxiter':500})
        #print(optResult)
        return optResult.x[0]
    else:
        return 0.0

def fp_optics(xyfp, opt_coeffs):
    logging.debug("fp_optics: opt_coeffs: {0}".format(opt_coeffs))

    tanth_of_r_vectorize=np.vectorize(tanth_of_r, excluded=[1])

    xy = rev_az_asym(xyfp)
    rfp_times_rfp0 = np.sum(xy ** 2, axis=1) ** 0.5
    phirad = np.arctan2(-xy[:, 1], -xy[:, 0])
    phideg = phirad / (np.pi / 180.0) % 360
    thetarad = tanth_of_r_vectorize(rfp_times_rfp0, opt_coeffs)
    thetadeg = thetarad / (np.pi / 180.0)
    return phideg, 90.0 - thetadeg


def sphereToCart(ras, decs):
    """Convert 3d spherical coordinates to cartesian"""
    rarads = (np.pi / 180.0) * ras
    decrads = (np.pi / 180.0) * decs
    sinras = np.sin(rarads)
    cosras = np.cos(rarads)
    sindecs = np.sin(decrads)
    cosdecs = np.cos(decrads)
    vec0s = cosras * cosdecs
    vec1s = sinras * cosdecs
    vec2s = sindecs
    return vec0s, vec1s, vec2s


def eulerm323(eul):
    mat1 = rotm1(2, eul[0])
    mat2 = rotm1(1, eul[1])
    mata = np.matmul(mat2, mat1)
    mat1 = rotm1(2, eul[2])
    rmat = np.matmul(mat1, mata)
    return rmat


def rotm1(ax, angle):
    mat = np.zeros((3, 3), dtype=np.double)
    n1 = ax
    n2 = np.mod((n1 + 1), 3)
    n3 = np.mod((n2 + 1), 3)
    sinang = np.sin(angle)
    cosang = np.cos(angle)
    mat[n1][n1] = 1.0
    mat[n2][n2] = cosang
    mat[n3][n3] = cosang
    mat[n2][n3] = sinang
    mat[n3][n2] = -sinang
    return mat


def cartToSphere(vec):
#   print("cartToSphere: vec: {0}".format(vec)) )# axis=1 breaking for 1 ra dec only
#   norm = np.sqrt(np.sum(vec ** 2))
#   dec = np.arcsin(vec[ 2] / norm)
#   ra = np.arctan2(vec[ 1], vec[0])
#   ra = np.mod(ra, 2.0 * np.pi)
#breaking for 1 coord only, fix later
   logging.debug("cartToSphere: vec: {0}".format(vec))
   norm = np.sqrt(np.sum(vec ** 2, axis=1))
   dec = np.arcsin(vec[:, 2] / norm)
   ra = np.arctan2(vec[:, 1], vec[:, 0])
   ra = np.mod(ra, 2.0 * np.pi)
   return ra, dec

def star_in_fov(lng, lat):
    deg2rad = np.pi / 180.0
    inView = False
    if lat > 70.0:
        vec = np.asarray(sphereToCart(lng, lat))
        norm = np.sqrt(np.sum(vec ** 2))
        if norm > 0.0:
            vec = vec / norm
            xlen = np.abs(np.arctan(vec[0] / vec[2]))
            ylen = np.abs(np.arctan(vec[1] / vec[2]))
            if (xlen <= (12.5 * deg2rad)) and (ylen <= (12.5 * deg2rad)):
                inView = True
    return inView


def make_az_asym(coords):
    # I dont think we need this function at all in the specific tess case where
    # asymang=0 asymfac=1
    # We're rotating a matrix by 0, multiplying its x coord by 1
    # and re-rotating by -0

    asymang = 0.0
    asymfac = 1.0
    xyp = xyrotate(asymang, coords)
    logging.debug("make_az_asym: xyp: {0}".format(xyp))
    logging.debug("make_az_asym: xyp: shape:  {0}".format(xyp.shape))

    xypa = np.zeros_like(xyp)
    xypa[:,0] = asymfac * xyp[:,0]
    xypa[:,1] = xyp[:,1]
    xyout = xyrotate(-asymang, xypa)
    return xyout


def optics_fp(lng_deg, lat_deg, opt_coeffs):
    # Check Back Later For more Optimization
    # angle to focal plane location, I think
    # Had lines that recreated r_tanth, subbed the function in
    # had np.power not **, speed v precision?  check r_tanth
    deg2rad = np.pi / 180.0
    thetar = np.pi / 2.0 - (lat_deg * deg2rad)

    rtanth = r_of_tanth(thetar, opt_coeffs)

    logging.debug("optics_fp: rtanth: {0}".format(rtanth))

    cphi = np.cos(deg2rad * lng_deg)
    sphi = np.sin(deg2rad * lng_deg)
    logging.debug("optics_fp: cphi: {0}".format(cphi))
    logging.debug("optics_fp: sphi: {0}".format(sphi))
    xyfp = np.zeros((len(lng_deg),2), dtype=np.double)
    xyfp[:,0] = -cphi * rtanth
    xyfp[:,1] = -sphi * rtanth
    logging.debug("optics_fp: xyfp: {0}".format(xyfp))
    logging.debug("optics_fp: xyfp shape: {0}".format(xyfp.shape))

    return make_az_asym(xyfp)


def xy_to_ccdpx(self, xy):
    xyb = np.zeros_like(xy)
    ccdpx = np.zeros_like(xy)
    ccdx0 = tess_params[self.camera][f"x0_ccd{self.ccd}"]
    ccdy0 = tess_params[self.camera][f"y0_ccd{self.ccd}"]
    ccdang = tess_params[self.camera][f"ang_ccd{self.ccd}"]
    pixsz = 0.015000

    xyb[:,0] = xy[:,0] - ccdx0
    xyb[:,1] = xy[:,1] - ccdy0
    xyccd = xyrotate(ccdang, xyb)
    ccdpx[:,0] = (xyccd[:,0] / pixsz) - 0.5
    ccdpx[:,1] = (xyccd[:,1] / pixsz) - 0.5
    return ccdpx

def mm_to_pix(self, xy):
    # Convert focal plane to pixel location also need to add in the
    #    auxillary pixels added into FFIs
    #
    #created xy_to_ccdpix function to minimize repeated math
    #re-indiced ccd for 1-4 vs 03 given the the change in tess_param
    CCDWD_T = 2048
    CCDHT_T = 2058
    ROWA = 44
    ROWB = 44
    COLDK_T = 20
    fitpx = np.zeros_like(xy)
    logging.debug("mm_to_pix: fitpx: {0}".format(fitpx))
    logging.debug("mm_to_pix: fitpx.shape: {0}".format(fitpx.shape))


    if self.ccd == 1:
        ccdpx = xy_to_ccdpx(self, xy,)
        fitpx[:,0] = (CCDWD_T - ccdpx[:,0]) + CCDWD_T + 2 * ROWA + ROWB - 1.0
        fitpx[:,1] = (CCDHT_T - ccdpx[:,1]) + CCDHT_T + 2 * COLDK_T - 1.0
    if self.ccd == 4:
        ccdpx = xy_to_ccdpx(self, xy)
        fitpx[:,0] = ccdpx[:,0] + CCDWD_T + 2 * ROWA + ROWB
        fitpx[:,1] = ccdpx[:,1]
    if self.ccd == 2:
        ccdpx = xy_to_ccdpx(self, xy)
        fitpx[:,0] = (CCDWD_T - ccdpx[:,0]) + ROWA - 1.0
        fitpx[:,1] = (CCDHT_T - ccdpx[:,1]) + CCDHT_T + 2 * COLDK_T - 1.0
    if self.ccd == 3:
        ccdpx = xy_to_ccdpx(self, xy)
        fitpx[:,0] = ccdpx[:,0] + ROWA
        fitpx[:,1] = ccdpx[:,1]

    return ccdpx, fitpx

def fitpix2pix(fitpix,ccdpx):
    # SPOC calibrated FFIs have 44 collateral pixels in x and are 1 based  
    # Assuming combinedFits = False & args.noCollateral = False (for Now)
    # In OG stars2px, xyUse choses between a few options depending on flags
    # passed to stars2px, here we're making assumptions and will generalize after
    xyUse = np.zeros(ccdpx.shape)
    logging.debug("fitpix2pix: ccdpx: shape: {0}".format(ccdpx.shape))

    xyUse[:,0] = ccdpx[:,0] + 45.0
    xyUse[:,1] = ccdpx[:,1] + 1.0
    xMin = 44.0
    ymaxCoord = 2049
    xmaxCoord = 2093
    # visCut = xyUse[:,0]>xMin & xyUse[:,1]>0 & xyUse[:,0]<xmaxCoord & xyUse[:,1]<ymaxCoord
    visCut=True
    logging.debug("fitpix2pix: visCut: {0}".format(visCut))

    if visCut: # only reun if everything is valid.  should this be any and just run on what passes?
        findAny=True
        edgeWarn = np.zeros(len(xyUse[:,0]))
        edgePix = 6
        edgeWarn = (xyUse[:,0]<=(xMin+edgePix)) | (xyUse[:,0]>=(xmaxCoord-edgePix)) | (xyUse[:,1]<=edgePix) | (xyUse[:,1]==(ymaxCoord-edgePix))
    return xyUse, edgeWarn