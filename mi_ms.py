
import sys, os
import skimage
import skimage.io as skio

#sys.path.insert(0,'/data2/johan/buildse/SimpleITK-build/Wrapping/Python')

import SimpleITK as sitk
 
from multiprocessing import Process, Queue

#def parse_log_str(s):
#    ss = 'Final metric value  = '
#    ind = s.find(ss) + len(ss)
#    print(ind)
#    print(s)
#    ind2 = s[ind:].find('\n')
#    loss_str = s[ind:ind+ind2]
#    return float(loss_str)
   
def parse_log(fn):
    with open(fn) as f:
        lines = f.readlines()
        lines = '\n'.join(lines)        
        s = 'Final metric value  = '
        ind = lines.find(s) + len(s)
        ind2 = lines[ind:].find('\n')
        loss_str = lines[ind:ind+ind2]
        return float(loss_str)

def write_parameter_file(fn, rad, cp):
    if isinstance(rad, tuple) and len(rad) == 1:
        rad = rad[0]
    if isinstance(rad, float):
        with open(fn, 'w') as f:
            f.write(f'(Transform "EulerTransform")\n')
            f.write(f'(NumberOfParameters 3)\n')
            f.write(f'(TransformParameters {rad} 0.0 0.0)\n\n')
            f.write(f'(CenterOfRotationPoint {cp[0]} {cp[1]})\n')
    elif len(rad) == 3:
        with open(fn, 'w') as f:
            f.write(f'(Transform "EulerTransform")\n')
            f.write(f'(NumberOfParameters 6)\n')
            f.write(f'(TransformParameters {rad[0]} {rad[1]} {rad[2]} 0.0 0.0 0.0)\n\n')
            f.write(f'(CenterOfRotationPoint {cp[0]} {cp[1]} {cp[2]})\n')

def reg_one(q, img1, img2, n_res, rot, i, dataroot, shape):
    parameterMap = sitk.GetDefaultParameterMap('rigid')
    parameterMap['ResultImagePixelType'] = ['uint8']
    parameterMap['NumberOfResolutions'] = [str(n_res)]
    parameterMap['MaximumNumberOfIterations'] = ['1024']

    #print('Parameters: ' + str(parameterMap.asdict()))
    #sys.exit(0)
    log_dir = os.path.join(dataroot, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    tform_filename = f'./{log_dir}/rigid{i+1}.txt'
    log_filename = f'./{log_dir}/log{i+1}.txt'
    output_filename = f'./{log_dir}/registered_image{i+1}.tif'
    cp = [(shape[k]-1)/2 for k in range(len(shape))]
    write_parameter_file(tform_filename, rot, cp)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetLogToConsole(False)
    elastixImageFilter.SetFixedImage(img2)
    elastixImageFilter.SetMovingImage(img1)
    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.SetInitialTransformParameterFileName(tform_filename)
    elastixImageFilter.SetLogToFile(True)
    elastixImageFilter.SetLogFileName(log_filename)
    elastixImageFilter.Execute()
#    resultImage = elastixImageFilter.GetResultImage()
#    img1Reg = sitk.GetArrayFromImage(resultImage).astype('uint8')
#    skio.imsave(output_filename, img1Reg)
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()[0].asdict()
    tform_parameter = transformParameterMap['TransformParameters']

    loss = parse_log(log_filename)

    tform_parameter2 = [float(tform_parameter[j]) for j in range(len(tform_parameter))]
    for k in range(len(rot)):
        tform_parameter2[k] += rot[k]
    del tform_parameter
    del parameterMap
    del elastixImageFilter
#    print(f'Initial Rotation: {rot}, Loss: {loss}, Transform: {tform_parameter2}')
    q.put((loss, tform_parameter2))

def register_mi_ms(img1, img2, dataroot, n_res=5, init_rot=[(-0.4,), (0.0,), (0.4,)]):
    img1_arr = img1
    img2_arr = img2
    img1 = sitk.GetImageFromArray(img1)
    img2 = sitk.GetImageFromArray(img2)
    min_loss = None
    best_param = None
    best_im = None
    for i in range(len(init_rot)):
        queue = Queue()
        p = Process(target=reg_one, args=(queue, img1, img2, n_res, init_rot[i], i, dataroot, img1_arr.shape))
        p.start()
        p.join()
        result = queue.get()
        del queue
        del p
        loss, tform_parameter = result
        if min_loss is None or min_loss > loss:
            best_param = tform_parameter
            best_im = img1_arr
            min_loss = loss

    return best_param, min_loss

def main():
    data_root = './Datasets/RIRE_temp/fold1/'
    im1 = skio.imread(data_root + 'A/test/patient003_z0.png')
    im2 = skio.imread(data_root + 'B/test/patient003_z0.png')

    tform, loss = register_mi_ms(im1, im2, dataroot=data_root, n_res=4, init_rot=[(-0.4,), (0.0,), (0.4,)])
    print(tform)
    print(loss)

def main3d():
    im1 = skio.imread('t1vol.tif')
    im2 = skio.imread('pdvol.tif')
    vals = [-0.4, 0.0, 0.4]

    init_rot = [(vals[i], vals[j], vals[k]) for i in range(3) for j in range(3) for k in range(3)]
    im3, tform, loss = register_mi_ms(im1, im2, n_res=7, init_rot=init_rot)
    print(tform)
    print(loss)

def main3d_rire():
    im1 = skio.imread('/data2/jiahao/Registration/Datasets/RIRE_temp/fold1/A/test/patient003_z0.png')
    im2 = skio.imread('/data2/jiahao/Registration/Datasets/RIRE_temp/fold1/B/test/patient003_z0.png')
    print(im1.shape)
    print(im2.shape)
    vals = [-0.4, 0.0, 0.4]

    init_rot = [(vals[i], vals[j], vals[k]) for i in range(3) for j in range(3) for k in range(3)]
    im3, tform, loss = register_mi_ms(im1, im2, n_res=7, init_rot=init_rot)
    print(tform)
    print(loss)

if __name__ == '__main__':
    main()
    #main3d()
#    main3d_rire()

