import os
import cv2
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import open3d as o3d
import pandas as pd
import pyrealsense2
import json

def load_json(path_json):
    import json
    with open(path_json, 'r') as openfile:
        data_raw = json.load(openfile)
    return np.array(eval(data_raw))

def load_image_ply(pathToImages):

    files = os.listdir(pathToImages)
    my_point_cloud_list = []

    for i, filename in enumerate(files):
        my_point_cloud = PyntCloud.from_file(os.path.join(pathToImages, filename))
        my_point_cloud.points.describe()
        my_point_cloud_list.append(my_point_cloud)

    return my_point_cloud_list

def load_open3d(pathToImages):

    files = os.listdir(pathToImages)
    my_point_cloud_list = []
    my_pcd_list = []

    for i, filename in enumerate(files):
        pcd = o3d.io.read_point_cloud(os.path.join(pathToImages, filename), remove_nan_points=True)
        points = np.asarray(pcd.points)
        my_pcd_list.append(pcd)
        my_point_cloud_list.append(points)


    return my_pcd_list, my_point_cloud_list, files

def display_open3d_ply(data):

    o3d.visualization.draw_geometries([data])

    return

def load_images(pathToImages):
    """
    This function loads a sequence of images from disk and returns them as a 3D array.

    Arguments:
        pathToImages {string} -- path to the folder containing the images.

    Returns:
        numpy.array -- 3D array with stacked images.
    """
    files = os.listdir(pathToImages)
    # nImage = len(files)
    # imagesEL = np.zeros((Dim, Dim, nImage))
    # imagesELNormalized = np.zeros((Dim, Dim, nImage))
    images = []
    for i, filename in enumerate(files):
        # name = os.path.join(pathToImages, filename)
        # im = cv2.imread('./PV_images_failures/2019_January_19/Images/Panels_right/DJI_0343.jpg', 0)
        # imagesEL[:, :, i] = cv2.imread(os.path.join(pathToImages, filename), cv2.COLOR_BGR2GRAY)
        if not os.path.exists(pathToImages):
            print("Path does not exist!")
        else:
            images.append(cv2.imread(os.path.join(pathToImages, filename), 0))
            # imagesELNormalized[:, :, i] = cv2.normalize(imagesEL, imagesELNormalized, 0, 255, cv2.NORM_MINMAX)

    return images, files

def save_image_array(output_path, image_array, filename):
    """
    Saves each row of image (electroluminescence) array as new image on disk

    Arguments:
        output_path {str} -- Output path to save images
        image_array {numpy.array} -- Image data
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # for i in range(image_array.shape[2]):
    cv2.imwrite(os.path.join(output_path, filename + '.png'), image_array)

    return

def read_csv(path):

    df = pd.read_csv(path)
    return df

def save_csv(header, data, filename):
    import csv

    # header = ['name', 'area', 'country_code2', 'country_code3']
    # data = ['Afghanistan', 652090, 'AF', 'AFG']

    with open(filename + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)
    print("CSV do %s salvo", filename)

    return

def save_file_pickle(path_out, data, filename):

    import pickle
    with open(path_out + "/" + filename + ".pkl", "wb") as fp:
        pickle.dump(data, fp)
    print("Arquivo Salvo!")

    return

def sift_2D(path_img, filename):

    img_init = cv2.imread(path_img)
    gray = cv2.cvtColor(img_init, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    (kps, descs) = sift.detectAndCompute(gray, None)
    print(kps[0].pt)
    print(kps[1].pt)
    img = cv2.drawKeypoints(gray, kps[0:2], None, color=(0, 255, 0), flags=0)
    img[round(kps[0].pt[1]), round(kps[0].pt[0])] = (0, 255, 0)
    img[round(kps[1].pt[1]), round(kps[1].pt[0])]= (0, 255, 0)
    plt.imshow(img)
    plt.show()

    # cv2.imwrite('scene_1_' + str(i) + '.png', img[i])
    print("SIFT criado com sucesso, imagem ", filename)

    return kps, descs, img

def surf_2D(path_img, filename):

    hessianValue = 400

    img_init = cv2.imread(path_img)
    gray = cv2.cvtColor(img_init, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(hessianValue)

    (kpSurf, descSurf) = surf.detectAndCompute(gray, None)

    img = cv2.drawKeypoints(img_init, kpSurf, None, color=(255, 0, 0), flags=0)

    # cv2.imwrite('scene_1_' + str(i) + '.png', img[i])
    print("SURF criado com sucesso, imagem ", filename)

    return kpSurf, descSurf, img

def coords2D_3D(path_img_depth, path_csv_sift, path_csv_camera_info, index):

    img_depth = cv2.imread(path_img_depth)
    gray = cv2.cvtColor(img_depth, cv2.COLOR_BGR2GRAY)
    data_camera_info = pd.read_csv(path_csv_camera_info)
    coords_sift = pd.read_csv(path_csv_sift)

    x_coords = np.asarray(coords_sift.axes[0])
    y_coords = np.asarray(coords_sift.values)

    CX_DEPTH = data_camera_info.K_2[index]
    CY_DEPTH = data_camera_info.K_5[index]
    FX_DEPTH = data_camera_info.K_0[index]
    FY_DEPTH = data_camera_info.K_4[index]

    pcd = []

    for i in range(len(x_coords)):
        yy = round(float(y_coords[i]))
        xx = round(float(x_coords[i]))
        z = gray[yy][xx]
        x = (float(y_coords[i]) - CX_DEPTH) * z / FX_DEPTH
        y = (float(x_coords[i]) - CY_DEPTH) * z / FY_DEPTH

        pcd.append([x, y, z])
    return pcd

def get_nearest_image(list_images_depth, list_images_rgb, TIMESTAMP):
    list_img_split = []
    list_img_rgb_split = []
    list_time_image = []
    for im_depth in list_images_depth:
        list_img_split.append(float(im_depth.split(".png")[0]))
    for im_rgb in list_images_rgb:
        list_img_rgb_split.append(float(im_rgb.split(".csv")[0]))

    for timest in TIMESTAMP:
        arr_depth = np.asarray(list_img_split)
        index_depth = (np.abs(arr_depth - timest)).argmin()
        arr_rgb = np.asarray(list_img_rgb_split)
        index_rgb = (np.abs(arr_rgb - timest)).argmin()
        list_time_image.append([timest, list_images_depth[index_depth], index_depth, list_images_rgb[index_rgb], index_rgb])

    return list_time_image

def get_depth_lib(cameraInfo, x, y, depth, idx_pcd):
    _intrinsics = pyrealsense2.intrinsics()
    _intrinsics.width = cameraInfo.shape[0]
    _intrinsics.height = cameraInfo.shape[1]
    _intrinsics.ppx = cameraInfo.K_2[idx_pcd]
    _intrinsics.ppy = cameraInfo.K_5[idx_pcd]
    _intrinsics.fx = cameraInfo.K_0[idx_pcd]
    _intrinsics.fy = cameraInfo.K_4[idx_pcd]
    _intrinsics.model = pyrealsense2.distortion.none
    # _intrinsics.coeffs = [i for i in cameraInfo.D_0[idx_pcd]]
    result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)

    return result[2], -result[0], -result[1]

def save_es_vs_json(gng3d, es_vs_path):

    list_json_es = []
    list_json_vs = []

    for j in range(len(gng3d.es.indices)):
        list_json_es.append(list(gng3d.es[j].vertex_tuple[0]['weight']))
        list_json_es.append(list(gng3d.es[j].vertex_tuple[1]['weight']))

    json_object_es = json.dumps(list_json_es)

    with open(es_vs_path+"/es.json", "w") as outfile:
        json.dump(json_object_es, outfile)

    for j in range(len(gng3d.vs.indices)):
        list_json_vs.append(list(gng3d.vs[j]['weight']))

    json_object_vs = json.dumps(list_json_vs)

    with open(es_vs_path+"/vs.json", "w") as outfile:
        json.dump(json_object_vs, outfile)
    # print("gng3d.vs[0]['error']")

    print("Arquivos ES e VS salvos com sucesso!")

    return