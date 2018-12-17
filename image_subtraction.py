from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from reproject import reproject_interp
import os
from astropy.wcs import WCS
from astropy.visualization import PercentileInterval, ImageNormalize
import requests
from astropy.table import Table
from astropy.nddata import CCDData, NDData
from photutils import psf, EPSFBuilder
from io import BytesIO
from PyZOGY.subtract import calculate_difference_image
from PyZOGY.image_class import ImageClass
import scipy


def read_with_datasec(filename):
    ccddata = CCDData.read(filename, unit='adu')
    if 'datasec' in ccddata.meta:
        jmin, jmax, imin, imax = eval(ccddata.meta['datasec'].replace(':', ','))
        ccddata = ccddata[imin-1:imax, jmin-1:jmax]
    return ccddata


def get_ccd_bbox(ccddata):
    corners = [[0.], [0.5], [1.]] * np.array(ccddata.shape)[::-1]
    (ra_min, dec_min), (ra_ctr, dec_ctr), (ra_max, dec_max) = ccddata.wcs.all_pix2world(corners, 0.)
    max_size_dec = 0.199
    if dec_max - dec_min > max_size_dec:
        dec_min = dec_ctr - max_size_dec / 2.
        dec_max = dec_ctr + max_size_dec / 2.
    return (ra_min, dec_min, ra_max, dec_max), (ra_ctr, dec_ctr)


def get_ps1_catalog(ra_min, dec_min, ra_max, dec_max, mag_max=21., mag_min=16., mag_filter='r'):
    res = requests.get('http://gsss.stsci.edu/webservices/vo/CatalogSearch.aspx',
                       params={'cat': 'PS1V3OBJECTS', 'format': 'csv', 'mindet': 25,
                               'bbox': '{},{},{},{}'.format(ra_min, dec_min, ra_max, dec_max)})
    t = Table.read(res.text, format='csv', header_start=1, data_start=2)
    psfmag_key = mag_filter + 'MeanPSFMag'
    is_point_source = t[psfmag_key] - t[mag_filter + 'MeanKronMag'] < 0.05
    mag_cut = (t[psfmag_key] < mag_max) & (t[psfmag_key] > mag_min)
    t_stars = t[is_point_source & mag_cut]
    return t_stars


def make_psf(data, catalog, show=False):
    catalog = catalog.copy()
    catalog['x'], catalog['y'] = data.wcs.all_world2pix(catalog['raMean'], catalog['decMean'], 0)
    bkg = np.nanmedian(data)
    nddata = NDData(data - bkg)

    stars = psf.extract_stars(nddata, catalog, size=25.)
    epsf_builder = EPSFBuilder(oversampling=1.)
    epsf, fitted_stars = epsf_builder(stars)
    
    if show:
        plt.figure()
        plt.imshow(epsf.data)
        
        nrows = int(np.ceil(len(stars)**0.5))
        fig, axarr = plt.subplots(nrows, nrows, figsize=(20, 20), squeeze=True)
        for ax, star in zip(axarr.ravel(), fitted_stars):
            ax.imshow(star)
            ax.plot(star.cutout_center[0], star.cutout_center[1], 'r+')
    
    return epsf, fitted_stars


def update_wcs(wcs, p):
    wcs.wcs.crval += p[:2]
    c, s = np.cos(p[2]), np.sin(p[2])
    if wcs.wcs.has_cd():
        wcs.wcs.cd = wcs.wcs.cd @ np.array([[c, -s], [s, c]])
    else:
        wcs.wcs.pc = wcs.wcs.pc @ np.array([[c, -s], [s, c]])


def wcs_offset(p, radec, xy, origwcs):
    wcs = origwcs.deepcopy()
    update_wcs(wcs, p)
    test_xy = wcs.all_world2pix(radec, 0)
    rms = (np.sum((test_xy - xy)**2) / len(radec))**0.5
    return rms


def refine_wcs(wcs, stars, catalog):
    xy = np.array([star.center for star in stars])
    t_match = catalog[[star.id_label - 1 for star in stars]]
    radec = np.array([t_match['raMean'], t_match['decMean']]).T

    res = scipy.optimize.minimize(wcs_offset, [0., 0., 0.], args=(radec, xy, wcs),
                                  bounds=[(-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)])

    orig_rms = wcs_offset([0., 0., 0.], radec, xy, wcs)
    print(' orig_fun: {}'.format(orig_rms))
    print(res)
    update_wcs(wcs, res.x)


def download_ps1_image(ra, dec, filt):
    res = requests.get('http://ps1images.stsci.edu/cgi-bin/ps1filenames.py',
                 params={'ra': ra, 'dec': dec, 'filters': filt})
    t = Table.read(res.text, format='ascii')
    res = requests.get('http://ps1images.stsci.edu' + t['filename'][0])
    hdulist = fits.open(BytesIO(res.content))
    
    # https://outerspace.stsci.edu/display/PANSTARRS/PS1+DR1+caveats#PS1DR1caveats-FITSimageformatquirks
    # https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service#PS1ImageCutoutService-ImportantFITSimageformat,WCS,andflux-scalingnotes
    boffset = hdulist[1].header['boffset']
    bsoften = hdulist[1].header['bsoften']
    data_linear = boffset + bsoften * 2 * np.sinh(hdulist[1].data * np.log(10.) / 2.5)
    ccddata = CCDData(data_linear, wcs=WCS(hdulist[1].header), unit='adu')
    return ccddata


if __name__ == '__main':
    # # Read the science image

    show = False
    workdir = '/Users/griffin/imgsub/'
    filename = os.path.join(workdir, 'PS17dbf.Science.5615_proc.fits')
    scidata0 = read_with_datasec(filename)

    # # Download the PS1 catalog

    ccd_bbox, (ra, dec) = get_ccd_bbox(scidata0)
    catalog = get_ps1_catalog(*ccd_bbox)

    # # Pretend to make the PSF for the science image
    # This is just to find the centroids of the stars so we can update the WCS in the next step

    _, sci_stars = make_psf(scidata0, catalog, show=show)

    # # Update the WCS for the science image

    scidata = scidata0.copy()
    refine_wcs(scidata.wcs, sci_stars, catalog)

    science_filename = os.path.join(workdir, 'science.fits')
    scidata.write(science_filename, overwrite=True)

    # # Actually make the PSF for the science image

    sci_psf, _ = make_psf(scidata, catalog, show=show)

    # # Download the reference image

    refdata0 = download_ps1_image(ra, dec, scidata.meta['filter'][0])

    # # Update the WCS for the reference image

    _, ref_stars = make_psf(refdata0, catalog, show=show)
    refine_wcs(refdata0.wcs, ref_stars, catalog)

    # # Reproject the reference image to match the science image

    refdata_reproj, refdata_foot = reproject_interp((refdata0.data, refdata0.wcs), scidata.wcs, scidata.shape)

    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
        x, y = scidata.wcs.all_world2pix(catalog['raMean'], catalog['decMean'], 0.)

        vmin, vmax = np.percentile(scidata.data, (15., 99.5))
        ax1.imshow(scidata.data, vmin=vmin, vmax=vmax)
        ax1.plot(x, y, marker='o', mec='r', mfc='none', ls='none')

        norm = ImageNormalize(refdata_reproj, PercentileInterval(99.))
        ax2.imshow(refdata_reproj, norm=norm)
        ax2.plot(x, y, marker='o', mec='r', mfc='none', ls='none')

    template_filename = os.path.join(workdir, 'template.fits')
    refdata_reproj[np.isnan(refdata_reproj)] = 0.
    refdata = CCDData(refdata_reproj, wcs=scidata.wcs, mask=1-refdata_foot, unit='adu')
    refdata.write(template_filename, overwrite=True)

    # # Make the PSF for the reference image

    ref_psf, _ = make_psf(refdata, catalog, show=show)

    # # Subtract the images and view the result

    output_filename = os.path.join(workdir, 'diff.fits')
    science = ImageClass(scidata.data, sci_psf.data, saturation=65565)
    reference = ImageClass(refdata.data, ref_psf.data, refdata.mask)
    difference = calculate_difference_image(science, reference, show=show, max_iterations=1)
    fits.writeto(output_filename, difference, overwrite=True)

    if show:
        vmin, vmax = np.percentile(difference, (15., 99.5))
        plt.figure(figsize=(7., 15.))
        plt.imshow(difference, vmin=vmin, vmax=vmax)
