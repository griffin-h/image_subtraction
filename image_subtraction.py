from astropy.stats import sigma_clipped_stats
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
from PyZOGY import subtract
from PyZOGY.image_class import ImageClass
import scipy
from photutils import Background2D, MedianBackground
from astropy.stats import SigmaClip
import matplotlib.patheffects as PathEffects
import sys
from astropy.coordinates import SkyCoord
from astropy import units as u
import warnings


def read_with_datasec(filename, hdu=0):
    ccddata = CCDData.read(filename, format='fits', unit='adu', hdu=hdu)
    if 'datasec' in ccddata.meta:
        jmin, jmax, imin, imax = eval(ccddata.meta['datasec'].replace(':', ','))
        ccddata = ccddata[imin-1:imax, jmin-1:jmax]
    return ccddata


def get_ccd_bbox_and_data(filename, max_size=0.25):
    """
    Get the data from a local .fits file, but crop it according to the DATASEC
    parameter if found in the header. Then determine the coordinates at the edges
    of the image.

    Parameters
    --------------------
    filename : .fits file with good WCS
    max_size : Maximum size of the image in deg

    Output
    ---------------------
    ccdata           : np.array with image data
    ra_min, ra_max   : Minimum and Maximum RA and DEC
    dec_min, dec_max   in units of degrees

    """

    # For the 48" KeplerCam data, the Epoch label needs 
    # to be overwritten by the equinox label
    try:
        equinox = fits.getval(filename, 'EQUINOX')
        fits.setval(filename, 'EPOCH', value=equinox)
    except KeyError:
        pass

    # Read in image data into a 2D array
    # Cropping if the DATASEC parameter exists
    ccddata = read_with_datasec(filename)

    # Get the RA/DEC at the corners of the image
    ra_min, dec_min, ra_max, dec_max = get_ccd_bbox(ccddata)

    # Calculate the center RA and DEC
    ra_ctr = (ra_max + ra_min) / 2.0
    dec_ctr = (dec_max + dec_min) / 2.0

    # If the image is larger than the maximum size, crop it.
    if dec_max - dec_min > max_size:
        dec_min = dec_ctr - max_size / 2.
        dec_max = dec_ctr + max_size / 2.
    if ra_max - ra_min > max_size:
        ra_min = ra_ctr - max_size / 2.
        ra_max = ra_ctr + max_size / 2.

    return ccddata, ra_min, dec_min, ra_max, dec_max


def get_ccd_bbox(data):
    """
    Determine the coordinates at the edges of 
    a 2D array with WCS
    
    Parameters
    --------------------
    data : 2D array with WCS

    Output
    ---------------------
    ra_min, ra_max   : Minimum and Maximum RA and DEC
    dec_min, dec_max   in units of degrees
    """

    # Size of the image
    jmax, imax = data.shape[0], data.shape[1]

    # Get the RA/DEC at the corners of the image
    corners = np.array([[0, 0], [jmax, 0], [0, imax], [jmax, imax]])
    corners_wcs = data.wcs.wcs_pix2world(corners, 0).T

    # Get the biggest and smallest RA/DEC values from the corners
    ra_min = np.min(corners_wcs[0])
    ra_max = np.max(corners_wcs[0])
    dec_min = np.min(corners_wcs[1])
    dec_max = np.max(corners_wcs[1])

    return ra_min, dec_min, ra_max, dec_max


def get_ps1_catalog(ra_min, dec_min, ra_max, dec_max):
    """
    Import PS1 table catalog in box form, bounded by coordinates

    Parameters
    --------------------
    ra_min, ra_max   : Minimum and Maximum RA and DEC
    dec_min, dec_max   in units of degrees

    Output
    --------------------
    all_stars : Astropy Table of sources
    """

    # Generate PS1 request
    res = requests.get('http://gsss.stsci.edu/webservices/vo/CatalogSearch.aspx',
                       params={'cat': 'PS1V3OBJECTS', 'format': 'csv', 'mindet': 5,
                               'bbox': '{},{},{},{}'.format(ra_min, dec_min, ra_max, dec_max)})
    # Read in Data
    all_stars = Table.read(res.text, format='csv', header_start=1, data_start=2)

    return all_stars


def calculate_nature(catalog, color, psf_0point=21.50001, difference_0point=0.050001, unknown_radius=0.4):
    """
    Determine if the object is a galaxy or a star based on their Kron luminosity and PSF luminosity 
    using the diagram from https://confluence.stsci.edu/display/PANSTARRS/How+to+separate+stars+and+galaxies
    but cutting it up into radial chunks to give each chunk a probability or it being a galaxy.

    The probability ranges from 0 to 1 in steps of 0.25, where 1 is most likely a galaxy and 0 is most likely a star.

    The function takes in the color of the filter as a string and returns the probability of each object
    It also returns a negative probability if one of the input magnitudes was -999

    Parameters
    ---------------
    catalog           : Input 3PI catalog in http://gsss.stsci.edu/webservices/vo/CatalogSearch.aspx format
    color             : band to use from g, r, i, z, y
    psf_0point        : Breaking point for PSF range
    difference_0point : Breaking point for Delta magnitude
    unknown_radius    : Radius in magnitudes around the 0 point for unknown nature.

    Output
    ---------------
    Object Type from 0 to 1
    """

    # PSF - Kron Magnitude
    kron_magnitude = np.array(catalog['%sMeanKronMag' % color].astype(float))
    psf_magnitude = np.array(catalog['%sMeanPSFMag' % color].astype(float))
    magnitude_difference = psf_magnitude - kron_magnitude

    # Calculate "angle" in the PSF_PSF-Kron diagram to determine the probability
    # that they are galaxies or stars
    angle = np.arctan(
        np.abs(magnitude_difference - difference_0point) / np.abs(psf_magnitude - psf_0point)) * 180 / np.pi

    # Make sure the angle is finite
    good = np.isfinite(angle)

    # Convert from 90 angle to 360 angle
    angle[np.where(
        ((magnitude_difference[good] - difference_0point) > 0) & ((psf_magnitude[good] - psf_0point) < 0))] += 90
    angle[np.where(
        ((magnitude_difference[good] - difference_0point) < 0) & ((psf_magnitude[good] - psf_0point) < 0))] += 180
    angle[np.where(
        ((magnitude_difference[good] - difference_0point) < 0) & ((psf_magnitude[good] - psf_0point) > 0))] += 270

    # Calculate "radius" in the PSF_PSF-Kron diagram
    radius = np.sqrt((magnitude_difference - difference_0point) ** 2 + (psf_magnitude - psf_0point) ** 2)

    # Assign probabilities based on angle
    object_type = - 999 * np.ones(len(radius))
    object_type_good = - 999 * np.ones(len(radius[good]))
    object_type_good[np.where((angle[good] > 0) & (angle[good] <= 25))] = 0.75
    object_type_good[np.where((angle[good] > 25) & (angle[good] <= 150))] = 1.00
    object_type_good[np.where((angle[good] > 150) & (angle[good] <= 170))] = 0.75
    object_type_good[np.where((angle[good] > 170) & (angle[good] <= 180))] = 0.25
    object_type_good[np.where((angle[good] > 180) & (angle[good] <= 270))] = 0.00
    object_type_good[np.where((angle[good] > 270) & (angle[good] <= 310))] = 0.25
    object_type_good[np.where((angle[good] > 310) & (angle[good] <= 360))] = 0.50

    # Override probabilities based on radius, too close to 0 point
    object_type_good[np.where(radius[good] <= unknown_radius)] = 0.50

    # If the object type is not known
    object_type[good] = object_type_good
    object_type[object_type == -999] = 'Nan'

    # For the objects where the magnitudes are not defined make the Probability not a number
    object_type[np.where(magnitude_difference[good] > 900) or np.where(magnitude_difference[good] < 900)] = 'Nan'
    object_type[np.where(psf_magnitude + kron_magnitude == -1998.0)] = 'nan'

    return object_type


def cut_ps1_catalog(complete_catalog, mag_max=21.0, mag_min=16.0, max_galaxy=0.4, deltamag=1.0, mag_filter='r'):
    """
    Refine the input catalog to only the best stars

    Parameters
    ---------------
    complete_catalog : Input catalog
    mag_max          : Maximum allowed magnitude
    mag_min          : Minimum allowed magntidue
    max_galaxy       : Maximum allowed probability the object is a galaxy, lower allows more.
    deltamag         : Minimum allowed delta magnitude at a 10 arcsec separation
    mag_filter       : Color to evalue the magnitude limits on

    Output
    ---------------
    Catalog in the same format as complete_catalog but cut
    """

    # Calculate average nature of the object in all filters
    object_type_g = calculate_nature(complete_catalog, 'g')
    object_type_r = calculate_nature(complete_catalog, 'r')
    object_type_i = calculate_nature(complete_catalog, 'i')
    object_type_z = calculate_nature(complete_catalog, 'z')
    object_type_y = calculate_nature(complete_catalog, 'y')
    object_type = np.nanmean([object_type_g, object_type_r, object_type_i, object_type_z, object_type_y], axis=0)

    # Determine which objects have a PSF - Kron Magnitude < 0.05, call those stars
    if mag_filter in 'grizy':
        psfmag_key = mag_filter + 'MeanPSFMag'
        psf_magnitude = complete_catalog[psfmag_key]
    else:
        print("Filter %s not found, using 'r' instead" % mag_filter)
        mag_filter = 'r'
        psfmag_key = mag_filter + 'MeanPSFMag'
        psf_magnitude = complete_catalog[psfmag_key]

    # Select stars to be considered
    mag_cut = (psf_magnitude < mag_max) & (psf_magnitude > mag_min)
    is_point_source = object_type < max_galaxy
    good_stars = is_point_source & mag_cut

    # Get list of stars that arent nan
    acceptable_stars = psf_magnitude > -100
    accept_ras = complete_catalog['raMean'][acceptable_stars]
    accept_decs = complete_catalog['decMean'][acceptable_stars]
    accept_psfs = psf_magnitude[acceptable_stars]

    # Get RA and DEC of sources
    ras = complete_catalog['raMean'][good_stars]
    decs = complete_catalog['decMean'][good_stars]
    psfs = psf_magnitude[good_stars]

    # Calculate how many "bad stars" are there close to the target star
    bad_stars = np.array([len(np.where((accept_psfs - k_psf) <
                                       (-0.3 * (np.sqrt(
                                           (k_ra - accept_ras) ** 2 + (k_dec - accept_decs) ** 2) * 3600) + (
                                                    deltamag + 3)))[0])
                          for k_psf, k_ra, k_dec in zip(psfs, ras, decs)]) - 1
    best_stars = bad_stars == 0.0

    # Unless there were no stars found, accept some bad stars
    if len(np.where(best_stars == True)[0]) == 0.0:
        print('Accepting some bad stars')
        best_stars = bad_stars == 1.0

    # Return the catalog with the best stars
    catalog = complete_catalog[good_stars][best_stars]

    return catalog


def make_psf(ccddata, catalog_in, plot_action='', plot_name='', box_size=25.0, offset_sigma=1, save_psf=False,
             workdir='.', delete_stars=''):
    """
    Create PSF of stars

    Parameters
    ---------------
    ccddata      : Arary of data
    catalog_in   : PS1 catalog
    plot_action  : empty '' or 'show' or 'save'
    plot_name    : Append at the end of figure name
    box_size     : Search box for centering algorithm
    offset_sigma : Maximum allowed std after centering
    save_psf     : Save output PSF
    workdir      : Directory to save files to
    delete_stars : List of stars to delete from the list

    Output
    ---------------
    epsf                        : PSF model
    fitted_stars.all_good_stars : List of stars
    stars_index                 : Good stars in the list of stars
    catalog_index               : Good stars in the list of input catalog
    bkd_data                    : Model of background data
    """

    # Add 'x' and 'y' pixel coordinates to the catalog
    catalog_in = catalog_in.copy()

    # Remove the stars from the catalog that were listed in delete_stars
    if delete_stars != '':
        stars_delete = np.array(delete_stars.split(',')).astype(int)
        to_keep = [i not in stars_delete for i in range(len(catalog_in))]
        catalog = catalog_in[to_keep]
    else:
        catalog = catalog_in

    # Generate x and y columns
    catalog['x'], catalog['y'] = ccddata.wcs.all_world2pix(catalog['raMean'], catalog['decMean'], 0)

    # Model the background of the image
    bkg_estimator = MedianBackground()
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_model = Background2D(ccddata, 50, filter_size=3, sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    bkd_data = bkg_model.background

    # Subtract off the background from the data
    nddata = NDData(ccddata - bkd_data)

    # Select Only the stars that are within a box_size of the edges
    y_size, x_size = ccddata.shape
    good_stars = np.where(
        (catalog['x'] > (box_size / 2)) & (catalog['x'] < x_size - (box_size / 2)) & (catalog['y'] > (box_size / 2)) & (
                    catalog['y'] < y_size - (box_size / 2)))[0]
    good_catalog = catalog[good_stars]

    # Fit PSF of stars
    stars = psf.extract_stars(nddata, good_catalog, size=box_size)
    epsf_builder = EPSFBuilder(oversampling=1.0)
    epsf, fitted_stars = epsf_builder(stars)

    # Find centers for stars
    x_center = np.array([star.center[0] for star in fitted_stars.all_good_stars])
    y_center = np.array([star.center[1] for star in fitted_stars.all_good_stars])

    # Find which stars correspond to which in the catalog
    id_labels = np.array([star.id_label - 1 for star in fitted_stars.all_good_stars])

    # Good offsets are within some sigma of the mean
    xy_offset = np.sqrt((x_center - good_catalog[id_labels]['x']) ** 2 + (y_center - good_catalog[id_labels]['y']) ** 2)
    good_offset = np.where(xy_offset < np.mean(xy_offset) + offset_sigma * np.std(xy_offset))[0]

    # Version of the catalog with only the best stars
    catalog_index = np.arange(len(catalog))[good_stars][id_labels][good_offset]
    stars_index = np.arange(len(fitted_stars.all_good_stars))[good_offset]

    if plot_action != '':
        # Plot average PSF
        if plot_action == 'save':
            mean_local, median_local, std_local = sigma_clipped_stats(epsf.data, sigma_lower=2.0, sigma_upper=1.0,
                                                                      iters=3)
            plt.imshow(epsf.data, vmin=median_local - std_local * 3.0, vmax=median_local + std_local * 7.0, origin=0)
            plt.savefig(os.path.join(workdir, 'star_psf_%s.jpg' % plot_name), dpi=200)
            plt.clf()
        elif plot_action == 'show':
            plt.figure()
            plt.imshow(epsf.data)

        # Plot the PSF of all stars used
        nrows = int(np.ceil(len(fitted_stars.all_good_stars) ** 0.5))
        if nrows == 1: nrows = 2
        fig, axarr = plt.subplots(nrows, nrows, figsize=(20, 20), squeeze=True)
        plt.subplots_adjust(hspace=0, wspace=0)

        # Plot PSF of stars with a good fit
        for k, ax, star in zip(np.arange(len(catalog))[good_stars][id_labels], axarr.ravel(),
                               fitted_stars.all_good_stars):
            mean_local, median_local, std_local = sigma_clipped_stats(star.data, sigma_lower=2.0, sigma_upper=1.0,
                                                                      iters=3)
            text = ax.annotate(k, xy=(box_size / 2, box_size / 1.25), fontsize=35)
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
            ax.imshow(star, vmin=median_local - std_local * 3.0, vmax=median_local + std_local * 7.0, origin=0)
            ax.plot(star.cutout_center[0], star.cutout_center[1], 'rx', markersize=15)
            ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                           labeltop=False, labelright=False, labelbottom=False)

        # Remove axis labels from the remaining plots
        for k in range(len(fitted_stars.all_good_stars), nrows ** 2):
            ax = axarr.ravel()[k]
            ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                           labeltop=False, labelright=False, labelbottom=False)

        if plot_action == 'save':
            plt.savefig(os.path.join(workdir, 'stars_psf_%s.jpg' % plot_name), bbox_inches='tight', dpi=200)
            plt.clf()
        plt.close('all')

    if save_psf:
        # Save the output psf to file
        science_psf_finalname = os.path.join(workdir, plot_name + '_psf.fits')
        hdu_science = fits.PrimaryHDU()
        hdu_science.data = epsf.data
        hdu_science.writeto(science_psf_finalname, overwrite=True)

    return epsf, fitted_stars.all_good_stars, stars_index, catalog_index, bkd_data


def update_wcs(wcs, p):
    wcs.wcs.crval += p[:2]
    c, s = np.cos(p[2]), np.sin(p[2])
    if wcs.wcs.has_cd():
        wcs.wcs.cd = wcs.wcs.cd @ np.array([[c, -s], [s, c]]) * p[3]
    else:
        wcs.wcs.pc = wcs.wcs.pc @ np.array([[c, -s], [s, c]]) * p[3]


def wcs_offset(p, radec, xy, origwcs):
    wcs = origwcs.deepcopy()
    update_wcs(wcs, p)
    test_xy = wcs.all_world2pix(radec, 0)
    rms = (np.sum((test_xy - xy) ** 2) / len(radec)) ** 0.5
    return rms


def refine_wcs(ccddata, catalog, plot_action='', plot_name='', box_size=25., workdir='.'):
    """
    Modify the data with wcs_in to match the stars in the catalog

    Parameters
    ---------------
    ccddata      : Arary of data
    catalog      : PS1 catalog
    plot_action  : empty '' or 'show' or 'save'
    plot_name    : Append at the end of figure name
    box_size     : Search box for centering algorithm
    workdir      : Directory to save files to

    Output
    ---------------
    No output, but updates the wcs of ccddata
    """

    _, stars, stars_index, catalog_index, _ = make_psf(ccddata, catalog, plot_action=plot_action,
                                                       plot_name=plot_name, box_size=box_size, workdir=workdir)

    # Select only the good PSF stars and the corresponding stars in the catalog
    xy = np.array([star.center for star in stars])[stars_index]
    t_match = catalog[catalog_index]
    radec = np.array([t_match['raMean'], t_match['decMean']]).T

    # Fit the xy to the radec
    res = scipy.optimize.minimize(wcs_offset, [0., 0., 0., 1.], args=(radec, xy, ccddata.wcs),
                                  bounds=[(-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01), (0.95, 1.05)])
    orig_rms = wcs_offset([0., 0., 0., 1.], radec, xy, ccddata.wcs)
    print(' orig_fun: {}'.format(orig_rms))
    print(res)
    update_wcs(ccddata.wcs, res.x)

    return t_match


def download_ps1_image(ra, dec, filt, template_filename=None):
    """
    Download Image from PS1 and correct luptitudes back to a linear scale.

    Parameters
    ---------------
    ra, dec           : Coordinates in degrees
    filt              : Filter color 'g', 'r', 'i', 'z', or 'y'
    template_filename : Path to save template file (default: do not save)

    Output
    ---------------
    ccddata : CCDData format of data with WCS
    """

    # Query a center RA and DEC from PS1 in a specified color
    res = requests.get('http://ps1images.stsci.edu/cgi-bin/ps1filenames.py',
                       params={'ra': ra, 'dec': dec, 'filters': filt})

    # Get the image and save it into hdulist
    t = Table.read(res.text, format='ascii')
    res = requests.get('http://ps1images.stsci.edu' + t['filename'][0])
    hdulist = fits.open(BytesIO(res.content))

    # Linearize from leptitudes    
    boffset = hdulist[1].header['boffset']
    bsoften = hdulist[1].header['bsoften']
    linear = boffset + bsoften * 2 * np.sinh(hdulist[1].data * np.log(10.) / 2.5)
    warnings.simplefilter('ignore')  # ignore nonstandard header keywords in PS1 images
    ccddata = CCDData(linear, wcs=WCS(hdulist[1].header), unit='adu')

    # Save the template to file
    if template_filename is not None:
        ccddata.write(template_filename, overwrite=True)

    return ccddata


def download_references(ra_min, dec_min, ra_max, dec_max, mag_filter, template_basename=None, catalog=None):
    """
    Download 1 to 4 references from PS1 as necessary to cover full RA & dec range

    Parameters
    ---------------
    ra_min, ra_max   : Minimum and Maximum RA and DEC
    dec_min, dec_max   in units of degrees
    mag_filter       : Filter color 'g', 'r', 'i', 'z', or 'y'
    template_basename: Filename of the output(s), to be suffixed by 0.fits, 1.fits, ...
    catalog          : Catalog to which to align the reference image WCS (default: do not align)

    Output
    ---------------
    refdatas   : List of CCDData objects containing the reference images
    psf_catalog: Good stars to be used for generating the PSF (output of refine_wcs)

    """
    # Download 3PI Referece Image
    # Query the lowest RA and DEC coordinates
    print('\nDownloading 3PI Template ...\n')
    refdata0 = download_ps1_image(ra_min, dec_min, mag_filter, template_basename + '0.fits')
    ra_min_ref, dec_min_ref, ra_max_ref, dec_max_ref = get_ccd_bbox(refdata0)

    # Update WCS of reference image
    if catalog is not None:
        print('\nAligning Template ...\n')
        psf_catalog = refine_wcs(refdata0, catalog)
    else:
        psf_catalog = None
    refdatas = [refdata0]

    if (ra_max > ra_max_ref) or (dec_max > dec_max_ref):
        print('\nTemplate too small, downloading offset template ...\n')
        refdata1 = download_ps1_image(ra_max, dec_max, mag_filter, template_basename + '1.fits')
        if catalog is not None:
            refine_wcs(refdata1, catalog)
        refdatas.append(refdata1)

    # If there are three corners missing, download the extra three templates needed
    if (ra_max > ra_max_ref) and (dec_max > dec_max_ref):
        print('\nTemplate really offset, downloading 2 more templates ...\n')
        refdata2 = download_ps1_image(ra_min, dec_max, mag_filter, template_basename + '2.fits')
        refdata3 = download_ps1_image(ra_max, dec_min, mag_filter, template_basename + '3.fits')
        if catalog is not None:
            refine_wcs(refdata2, catalog)
            refine_wcs(refdata3, catalog)
        refdatas += [refdata2, refdata3]

    return refdatas, psf_catalog


def plot_images(catalog, scidata, refdata_reproj, plot_action='', workdir='.'):
    """
    Save the science and template images to a plot, with circles
    for the stars being used.

    Parameters
    ---------------
    catalog        : Catalog table from PS1
    scidata        : CCDdata format data
    refdata_reproj : CCDdata of template
    plot_action    : 'save', 'show', or nothing
    workdir        : Directory to save files to

    Output
    ---------------
    Save a plot
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    x, y = scidata.wcs.all_world2pix(catalog['raMean'], catalog['decMean'], 0.)

    vmin, vmax = np.percentile(scidata.data, (15., 99.5))
    ax1.imshow(scidata.data, vmin=vmin, vmax=vmax, origin=0)
    ax1.plot(x, y, marker='o', mec='r', mfc='none', ls='none', alpha=0.6)
    ax1.set_xlim(0, max(scidata.data.shape[0], scidata.data.shape[1]))
    ax1.set_title('Science')
    ax1.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                    labelright=False, labelbottom=False)

    norm = ImageNormalize(refdata_reproj, PercentileInterval(99.))
    ax2.imshow(refdata_reproj, norm=norm, origin=0)
    ax2.plot(x, y, marker='o', mec='r', mfc='none', ls='none', alpha=0.6)
    ax2.set_xlim(0, max(refdata_reproj.shape[0], refdata_reproj.shape[1]))
    ax2.set_title('Template')
    ax2.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                    labelright=False, labelbottom=False)

    if plot_action == 'save':
        plt.savefig(os.path.join(workdir, 'images.jpg'), dpi=200, bbox_inches='tight')
        plt.clf()
    elif plot_action == 'show':
        plt.show()
    else:
        plt.clf()


def plot_poststamp(template, science, difference, center_ra, center_dec, plot_action='', workdir='.', box_radius=50):
    """
    Plot a poststamp of the template, science, and subtracted images.
    Centered around a specified RA and DEC

    Parameters
    ---------------
    template    : Data array for template with WCS
    science     : Data array for science with WCS
    difference  : Data array for subtracted with WCS
    center_ra   : center RA in degrees
    center_dec  : center DEC in degrees
    plot_action : 'save', 'show', or nothing
    workdir     : Directory to save files to
    box_radius  : Radius of the plot in pixels

    Output
    ---------------
    Save a plot
    """

    # Get coordinates of images
    coord = SkyCoord(center_ra + ' ' + center_dec, unit=(u.hourangle, u.deg))
    science_x, science_y = science.wcs.all_world2pix(coord.ra, coord.dec, 0.)
    template_x, template_y = template.wcs.all_world2pix(coord.ra, coord.dec, 0.)
    difference_x, difference_y = science.wcs.all_world2pix(coord.ra, coord.dec, 0.)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))
    plt.subplots_adjust(hspace=0, wspace=0.01)

    vmin, vmax = np.percentile(science.data, (15., 99.2))
    ax1.imshow(science.data, vmin=vmin, vmax=vmax, origin=0)
    ax1.set_xlim(science_x - box_radius, science_x + box_radius)
    ax1.set_ylim(science_y - box_radius, science_y + box_radius)
    ax1.set_title('Science')
    ax1.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                    labelright=False, labelbottom=False)

    vmin, vmax = np.percentile(template.data, (15., 99.2))
    ax2.imshow(template.data, vmin=vmin, vmax=vmax, origin=0)
    ax2.set_xlim(template_x - box_radius, template_x + box_radius)
    ax2.set_ylim(template_y - box_radius, template_y + box_radius)
    ax2.set_title('Template')
    ax2.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                    labelright=False, labelbottom=False)

    vmin, vmax = np.percentile(difference.data, (15., 99.2))
    ax3.imshow(difference.data, vmin=vmin, vmax=vmax, origin=0)
    ax3.set_xlim(difference_x - box_radius, difference_x + box_radius)
    ax3.set_ylim(difference_y - box_radius, difference_y + box_radius)
    ax3.set_title('Difference')
    ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                    labelright=False, labelbottom=False)

    if plot_action == 'save':
        plt.savefig(os.path.join(workdir, 'poststamp.jpg'), dpi=200, bbox_inches='tight')
        plt.clf()
    elif plot_action == 'show':
        plt.show()
    else:
        plt.clf()


def run_hotpants(workdir='.'):
    """
    Run hotpants, it requires to have hotpants installed.
    the image_subtraction code must alredy have been run and the
    'science.fits' and 'template_project.fits' files must be in the
    workdir folder.
    """

    science = fits.open(os.path.join(workdir, 'science.fits'))
    template = fits.open(os.path.join(workdir, 'template_project.fits'))

    science[0].data[np.where(science[0].data <= 0)] += np.abs(science[0].data[np.where(science[0].data <= 0)]) + 0.1
    template[0].data[np.where(template[0].data <= 0)] += np.abs(template[0].data[np.where(template[0].data <= 0)]) + 0.1

    template_name = os.path.join(workdir, 'hot_template.fits')
    science_name = os.path.join(workdir, 'hot_science.fits')

    template.writeto(template_name, overwrite=True)
    science.writeto(science_name, overwrite=True)

    gain_science = science[0].header['GAIN']
    noise_science = science[0].header['RDNOISE']

    hotpants_command = 'hotpants -inim %s -tmplim %s -outim %s -n i -c t -tu 50000 -iu 50000 -ig %s' % (
    science_name, template_name, os.path.join(workdir, 'hot_output.fits'), gain_science)
    print(hotpants_command)
    os.system(hotpants_command)


def subtract_reference(filename, workdir, plot_action='', align_template=True, catalog_file=None, reference_files=None):
    """
    Do image subtraction on image filename and save everything to workdir directory.

    The first argument can either be a filename or a CCDData object.
    """

    # Make directory if it does not exist
    os.makedirs(workdir, exist_ok=True)

    if isinstance(filename, str):
        scidata, ra_min, dec_min, ra_max, dec_max = get_ccd_bbox_and_data(filename)
    else:
        scidata = filename
        ra_min, dec_min, ra_max, dec_max = get_ccd_bbox(scidata)

    if catalog_file is None:
        # Query 3PI catalog
        print('\nQuerying 3PI ...\n')
        grow = 1 / 60.0  # Query a region 5 arcmin larger
        # Try twice, since sometimes the query fails
        try:
            complete_catalog = get_ps1_catalog(ra_min - grow, dec_min - grow, ra_max + grow, dec_max + grow)
        except:
            complete_catalog = get_ps1_catalog(ra_min - grow, dec_min - grow, ra_max + grow, dec_max + grow)

        complete_catalog.write(os.path.join(workdir, scidata.meta['OBJECT'] + '.cat'), format='ascii', overwrite=True)
    else:
        complete_catalog = Table.read(catalog_file, format='ascii', guess=False)
    mag_filter = scidata.meta['filter'][0]
    catalog = cut_ps1_catalog(complete_catalog, mag_max=19.0, mag_min=16.0, max_galaxy=0.40, deltamag=1.0,
                              mag_filter=mag_filter)

    print('\nAligning Science ...\n')
    # Update the WCS of the science image
    science_psf_catalog = refine_wcs(scidata, catalog, plot_action, plot_name='science', workdir=workdir)
    science_filename = os.path.join(workdir, scidata.meta['OBJECT'] + '_sci.fits')
    scidata.write(science_filename, overwrite=True)

    # Make PSF of science image
    print('\nMaking Science PSF ...\n')
    sci_psf, _, _, _, _ = make_psf(scidata, science_psf_catalog, plot_action, plot_name='science_aligned',
                                   workdir=workdir)

    if reference_files is None:
        refdatas, reference_psf_catalog = download_references(ra_min, dec_min, ra_max, dec_max, mag_filter,
                                                              os.path.join(workdir, scidata.meta['OBJECT'] + '_ref'),
                                                              catalog if align_template else None)
    else:
        refdatas = [CCDData.read(fn, unit='adu') for fn in reference_files]
        reference_psf_catalog = catalog

    # Reproject the Reference image to match the science image
    print('\nReprojecting Template ...\n')
    refdatas_reprojected = []
    refdata_foot = np.zeros_like(scidata.data, float)
    for refdata in refdatas:
        reprojected, foot = reproject_interp((refdata.data, refdata.wcs), scidata.wcs, scidata.shape)
        refdatas_reprojected.append(reprojected)
        refdata_foot += foot
    refdata_reproj = np.nanmean(refdatas_reprojected, axis=0)

    # Save reprojected template
    template_finalname = os.path.join(workdir, scidata.meta['OBJECT'] + '_refalign.fits')
    refdata_reproj[np.isnan(refdata_reproj)] = 0.
    refdata = CCDData(refdata_reproj, wcs=scidata.wcs, mask=refdata_foot == 0., unit='adu')
    refdata.write(template_finalname, overwrite=True)

    # Plot Reference and Science images
    if plot_action:
        plot_images(catalog, scidata, refdata_reproj, plot_action=plot_action, workdir=workdir)

    # Make the PSF for the reference image
    print('\nMaking Template PSF ...\n')
    ref_psf, _, _, _, _ = make_psf(refdata, reference_psf_catalog, plot_action, plot_name='reference_aligned',
                                   workdir=workdir)

    # Subtract the images with PyZOGY
    print('\nSubtracting Images ...\n')
    science = ImageClass(scidata.data, sci_psf.data, header=scidata.header, saturation=65000)
    reference = ImageClass(refdata.data, ref_psf.data, refdata.mask, saturation=65000)
    try:
        difference = subtract.calculate_difference_image(science, reference, show=False)
    except ValueError:
        difference = subtract.calculate_difference_image(science, reference, show=False, size_cut=False)

    # Calculate the zeropoint and save
    difference_zero_point = subtract.calculate_difference_image_zero_point(science, reference)
    normalized_difference = subtract.normalize_difference_image(difference, difference_zero_point, science, reference,
                                                                'i')
    output_filename = os.path.join(workdir, scidata.meta['OBJECT'] + '_diff.fits')
    subtract.save_difference_image_to_file(normalized_difference, fits.open(science_filename)[0], 'i', output_filename)

    # Save Poststamp
    if plot_action:
        center_ra = fits.getval(filename, 'RA')
        center_dec = fits.getval(filename, 'DEC')
        plot_poststamp(refdata, scidata, difference, center_ra, center_dec, plot_action, workdir=workdir, box_radius=50)


if __name__ == '__main':
    workdir, filename = os.path.split(sys.argv[1])
    subtract_reference(filename, workdir)
