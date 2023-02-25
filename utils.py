import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models
from astropy.modeling import functional_models
from astropy import units as u

wavelengths = np.arange(400, 800)*u.nm
FWHM        = 0.5
lambda_Na   = 589.0*u.nm
lambda_Mg   = 518.3*u.nm
width       = 1000
height      = 1500

dir='./dataset/'

#%% Draw the image

def add_background(image_3d):
    '''
    '''
    # TODO: mutate the background
    for i in range(400):
        image_3d[:,:,i] += int(30-0.06*i)
    return image_3d

def add_stars(image_3d, coord_stars, amp_stars, T_stars): 
    '''
    '''
    # psf
    PSF = functional_models.Gaussian2D(x_stddev=FWHM, y_stddev=FWHM)
    # draw N stars with radius FWHM/2
    for [x, y], amp_star, T_star in zip(coord_stars, amp_stars, T_stars): 
        # spectrum
        BB = models.BlackBody(temperature=T_star)
        # add star to the image
        radius = int(5*FWHM)
        xx = np.arange(-radius, radius+1)
        yy = np.arange(-radius, radius+1)
        XX, YY = np.meshgrid(xx, yy)
        star = np.repeat(PSF(XX, YY)[:,:,np.newaxis], 400, axis=2)*BB(wavelengths).value*amp_star
        star = star.astype('uint16')
        image_3d[x-radius:x+radius+1,y-radius:y+radius+1,:] += star
    return image_3d

def add_meteor(image_3d, coord_meteors, amp_meteors, T_meteors, dir_meteors, length_meteors): 
    '''
    '''
    PSF = functional_models.Gaussian2D(x_stddev=FWHM, y_stddev=FWHM)
    for [x, y], amp_meteor, T_meteor, dir_meteor, length_meteor in zip(coord_meteors, amp_meteors, T_meteors, dir_meteors, length_meteors):
        # spectrum - continuous spectrum
        BB = models.BlackBody(temperature=T_meteor)
        meteor_x = BB(wavelengths).value*amp_meteor
        # spectrum - emission line
        amp_Na = amp_meteor*2e-6
        amp_Mg = amp_meteor*1e-6
        emission_Na = functional_models.Gaussian1D(mean=lambda_Na.value, stddev=FWHM)
        emission_Mg = functional_models.Gaussian1D(mean=lambda_Mg.value, stddev=FWHM)
        meteor_x += emission_Na(wavelengths.value)*amp_Na
        meteor_x += emission_Mg(wavelengths.value)*amp_Mg
        # trajectory
        meteor_y = np.arange(length_meteor)
        amplitude = 1e3
        # TODO: use true trajectory
        meteor_y = (meteor_y*10/length_meteor)**(0.1) - meteor_y/length_meteor
        meteor_y = meteor_y*amplitude
        meteor = np.outer(meteor_y, meteor_x)
        # draw meteor
        dx = np.cos(dir_meteor)
        dy = np.sin(dir_meteor)
        for i in range(length_meteor): 
            radius = int(5*FWHM)
            x_plot = int(x + dx*i)
            y_plot = int(y + dy*i)
            # clip the point out of range
            if radius< x_plot and x_plot< width-radius-1 and radius< y_plot and y_plot< height-radius-1: 
                xx = np.arange(-radius, radius+1)
                yy = np.arange(-radius, radius+1)
                XX, YY = np.meshgrid(xx, yy)
                point = np.repeat(PSF(XX, YY)[:,:,np.newaxis], 400, axis=2)*meteor[i, :]
                point = point.astype('uint16')
                image_3d[x_plot-radius:x_plot+radius+1,y_plot-radius:y_plot+radius+1,:] += point
    # plt.imshow(meteor)
    # plt.colorbar()
    # plt.show()
    # TODO: return the bounding box
    return image_3d

def add_landscape(image_3d): 
    '''
    TODO
    '''
    return image_3d

def capture(image_3d, direction, length): 
    '''
    '''
    dx = np.cos(direction)
    dy = np.sin(direction)
    for i in range(400): 
        # print(i, int(dx*length*i/400))
        image_3d[:,:,i] = np.roll(image_3d[:,:,i], axis=0, shift=int(dx*length*i/400))
        image_3d[:,:,i] = np.roll(image_3d[:,:,i], axis=1, shift=int(dy*length*i/400))
    image_rgb = np.zeros([width, height, 3], dtype=np.uint16)
    # RGB filter
    # TODO: adjust the RGB curve of camera
    # TODO: adjust the true slit distortion
    gaussian_r = functional_models.Gaussian1D(mean=600, stddev=30)
    gaussian_g = functional_models.Gaussian1D(mean=540, stddev=30)
    gaussian_b = functional_models.Gaussian1D(mean=480, stddev=30)
    filter_r = gaussian_r(wavelengths.value)
    filter_g = gaussian_g(wavelengths.value)
    filter_b = gaussian_b(wavelengths.value)
    filter_r[200:400] = 1-np.arange(200)**2/200**2
    # generate RGB components
    image_rgb[:, :, 0] = np.average(image_3d*filter_r, axis=2)
    image_rgb[:, :, 1] = np.average(image_3d*filter_g, axis=2)
    image_rgb[:, :, 2] = np.average(image_3d*filter_b, axis=2)
    image_rgb = image_rgb.astype('uint16')
    return image_rgb

def augmentation(image_rgb): 
    '''
    Add the lens effect such as disortion and gaussian noise
    TODO: use `imgaug`
    '''
    return image_rgb

def generate_image(coord_stars, amp_stars, T_stars, coord_meteor, amp_meteor, T_meteor, 
                    angle_meteor, length_meteor, angle_slit, length_slit): 
    '''
    function to generate the image. 
    Args: 
        The position of all the stars, the meteor, etc.: 
        coords_stars    [N, 2]      the coordinates of stars plotted in the diagram. Will also give the number of stars
        TODO...
        angle_slit      [0, 2pi)
        angle_meteor    [0, 2pi)
        length_meteor   int         how long the meteor is 
        length_slit     int         how long the track of spectrum is
    Return: 
        one RGB image
    '''
    image_3d = np.zeros([width, height, 400], dtype=np.uint16)           # 0-65536, valid in 0-256 TODO: change to float16? 
    image_3d = add_background(image_3d)
    image_3d = add_stars(image_3d, coord_stars, amp_stars, T_stars)
    image_3d = add_meteor(image_3d, coord_meteor, amp_meteor, T_meteor, angle_meteor, length_meteor)
    image_rgb = capture(image_3d, angle_slit, length_slit)
    image_rgb = augmentation(image_rgb)
    # clip the image
    image_rgb = np.clip(image_rgb, 0, 255)# including 0 and 255
    # image_rgb = image_rgb.astype('uint8')
    return image_rgb #image_3d[:,:,0]

#%% Annotate labels 

def blank_label(): 
    return {
        'info': {
            'description': 'Meteor spectrum fake data',
            'url': 'https://cheysen.fit',
            'version': '0.1',
            'year': 2023,
            'contributor': 'Yichen Liu',
            'date_created': '2023/02/08'
        },
        'licenses': {
            'url': 'NaN',
            'id': 1,
            'name': 'NaN'
        },
        'images': [],
        'annotations': [],
        'categories': [{
                'id': 1,
                'name': 'meteor',
                'keypoints': [
                    'start_Na',
                    'end_Na',
                    'start_Mg',
                    'end_Mg',
                ],
                'skeleton': [
                    [1, 2],
                    [3, 4],
                ]
            },{
                'id': 2,
                'name': 'star',
                'keypoints': [
                    'start_star',
                    'end_star',
                    'start_', 
                    'end_'
                ],
                'skeleton': [
                    [1, 2],
                ]
            }
        ]
    }

def annotate_image(filename, image_id): 
    return {
        'license': 1,
        'file_name': filename,
        'height': width,
        'width': height,
        'id': image_id
    }

def annotate_meteor(x_meteor, y_meteor, angle_slit, length_slit, angle_meteor, length_meteor, anno_id, image_id): 
    bbox_x1      = int(min(
        x_meteor, 
        x_meteor + np.cos(angle_slit)*length_slit, 
        x_meteor + np.cos(angle_meteor)*length_meteor, 
        x_meteor + np.cos(angle_slit)*length_slit + np.cos(angle_meteor)*length_meteor, 
    ))
    bbox_y1      = int(min(
        y_meteor, 
        y_meteor + np.sin(angle_slit)*length_slit, 
        y_meteor + np.sin(angle_meteor)*length_meteor, 
        y_meteor + np.sin(angle_slit)*length_slit + np.sin(angle_meteor)*length_meteor, 
    ))
    bbox_x2      = int(max(
        x_meteor, 
        x_meteor + np.cos(angle_slit)*length_slit, 
        x_meteor + np.cos(angle_meteor)*length_meteor, 
        x_meteor + np.cos(angle_slit)*length_slit + np.cos(angle_meteor)*length_meteor, 
    ))
    bbox_y2      = int(max(
        y_meteor, 
        y_meteor + np.sin(angle_slit)*length_slit, 
        y_meteor + np.sin(angle_meteor)*length_meteor, 
        y_meteor + np.sin(angle_slit)*length_slit + np.sin(angle_meteor)*length_meteor, 
    ))
    # check if the **bbox** is valid for labelling by **clipping the value out of bound**
    bbox_x1     = max(0, bbox_x1)
    bbox_y1     = max(0, bbox_y1)
    bbox_x2     = min(width, bbox_x2)
    bbox_y2     = min(height, bbox_y2)
    bbox_w      = bbox_x2 - bbox_x1
    bbox_h      = bbox_y2 - bbox_y1
    start_Na_x  = int(x_meteor + np.cos(angle_slit)*length_slit*(lambda_Na.value-400)/400)
    start_Na_y  = int(y_meteor + np.sin(angle_slit)*length_slit*(lambda_Na.value-400)/400)
    end_Na_x    = int(start_Na_x + np.cos(angle_meteor)*length_meteor)
    end_Na_y    = int(start_Na_y + np.sin(angle_meteor)*length_meteor)
    start_Mg_x  = int(x_meteor + np.cos(angle_slit)*length_slit*(lambda_Mg.value-400)/400)
    start_Mg_y  = int(y_meteor + np.sin(angle_slit)*length_slit*(lambda_Mg.value-400)/400)
    end_Mg_x    = int(start_Mg_x + np.cos(angle_meteor)*length_meteor)
    end_Mg_y    = int(start_Mg_y + np.sin(angle_meteor)*length_meteor)
    # check if the **keypoint** is valid for labelling by **clipping the value out of bound**
    flag_start_Na = 0 < start_Na_y and start_Na_y < height and 0 < start_Na_x and start_Na_x < width
    flag_end_Na   = 0 < end_Na_y   and end_Na_y   < height and 0 < end_Na_x   and end_Na_x   < width
    flag_start_Mg = 0 < start_Mg_y and start_Mg_y < height and 0 < start_Mg_x and start_Mg_x < width
    flag_end_Mg   = 0 < end_Mg_y   and end_Mg_y   < height and 0 < end_Mg_x   and end_Mg_x   < width
    return {
        'id': anno_id, 
        'image_id': image_id,
        'category_id': 1,
        'bbox': [
            bbox_y1, bbox_x1, bbox_h, bbox_w
        ],
        'keypoints': [
            start_Na_y, start_Na_x, flag_start_Na*2, end_Na_y, end_Na_x, flag_end_Na*2, 
            start_Mg_y, start_Mg_x, flag_start_Mg*2, end_Mg_y, end_Mg_x, flag_end_Mg*2
        ], 
        'num_keypoints': flag_start_Na + flag_end_Na + flag_start_Mg + flag_end_Mg
    }

def annotate_star(x, y, angle_slit, length_slit, anno_id, image_id): 
    start_star_x    = x
    start_star_y    = y
    end_star_x      = int(x + np.cos(angle_slit)*length_slit)
    end_star_y      = int(y + np.sin(angle_slit)*length_slit)
    bbox_x1 = min(start_star_x, end_star_x)
    bbox_y1 = min(start_star_y, end_star_y)
    bbox_x2 = max(start_star_x, end_star_x)
    bbox_y2 = max(start_star_y, end_star_y)
    bbox_w      = bbox_x2 - bbox_x1
    bbox_h      = bbox_y2 - bbox_y1
    # check if the star is valid for labelling by **ignoring the stars out of bound**
    if 0<=end_star_x and end_star_x<width and 0<=end_star_y and end_star_y<height: 
        return {
            'id': anno_id, 
            'image_id': image_id,
            'category_id': 2,
            'bbox': [
                bbox_y1, bbox_x1, bbox_h, bbox_w
            ],
            'keypoints': [
                start_star_y, start_star_x, 2, end_star_y, end_star_x, 2, 0, 0, 0, 0, 0, 0
            ], 
            'num_keypoints': 2
        }
    else: 
        return None











