# -*- coding: utf-8 -*-
import os

import cv2
import math
import numpy as np
from cytomine import Cytomine
from cytomine.models import Annotation
from shapely.affinity import translate, affine_transform
from shapely.geometry import Point
from shapely.wkt import loads

from io import open_image, files_in_directory, open_image_with_mask

__author__ = "Ulysse Rubens <urubens@student.ulg.ac.be>"
__version__ = "0.1"

IMAGE_INSTANCES_PATH = "imageinstances/"
CROPS_PATH = "crops/"
ROIS_PATH = "rois/"
GROUNDTRUTHS_PATH = "groundtruths/"


def _polygon_box(polygon):
    """From a shapely polygon, return the information about the polygon bounding box.
    These information are offset (x, y), width and height.

    Parameters
    ----------
    polygon: shapely.geometry.Polygon
        The polygon of which the bounding box should be computed

    Returns
    -------
    offset: tuple (int, int)
        The offset of the polygon bounding box
    width: int
        The bounding box width
    height
        The bounding box heigth
    """
    minx, miny, maxx, maxy = polygon.bounds
    fminx, fminy = int(math.floor(minx)), int(math.floor(miny))
    offset = (fminx, fminy)
    width = maxy - miny
    height = maxx - minx
    return offset, int(width), int(height)


def _roi_filename(parent_id, roi_id):
    return "{}_{}".format(parent_id, roi_id)
    # return roi_id


class RegionOfInterest(object):
    def __init__(self, parent_id, roi_id, width, height,
                 image_filename, groundtruth_filename):
        self._parent_id = parent_id
        self._roi_id = roi_id

        self._width = width
        self._height = height

        self._image_filename = image_filename
        self._groundtruth_filename = groundtruth_filename

        self._image = None
        self._groundtruths = None

    @property
    def parent_id(self):
        return self._parent_id

    @parent_id.setter
    def parent_id(self, value):
        self._parent_id = value

    @property
    def roi_id(self):
        return self._roi_id

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def filename(self):
        return _roi_filename(self.parent_id, self.roi_id)

    @property
    def image_filename(self):
        return self._image_filename

    @property
    def image(self):
        self._image = open_image_with_mask(self.image_filename)
        return self._image

    @property
    def groundtruth_filename(self):
        return self._groundtruth_filename

    @property
    def groundtruth(self):
        self._groundtruths = open_image(self.groundtruth_filename, cv2.IMREAD_GRAYSCALE)
        self._groundtruths /= np.max(self._groundtruths)
        return self._groundtruths

    def is_imageinstance(self):
        return self.parent_id == self.roi_id

    def __str__(self):
        return "ROI(Image: #{}, Crop: #{}, [{}x{}])".format(
                self.parent_id, self.roi_id,
                self.width, self.height)

    @classmethod
    def from_image_instance(cls, image_instance, gt_locations, working_path):
        groundtruth = mk_groundtruth_image(gt_locations, image_instance.width,
                                           image_instance.height, image_instance.height)
        return cls(parent_id=image_instance.id,
                   roi_id=image_instance.id,
                   width=image_instance.width,
                   height=image_instance.height,
                   image_filename=image_instance.filename,
                   groundtruth_filename=save_groundtruth_image(
                           groundtruth, working_path, image_instance.project,
                           _roi_filename(image_instance.id, image_instance.id))
                   )

    @classmethod
    def from_annotation_crop(cls, image_instance, crop, gt_locations, working_path):
        polygon = loads(crop.location)
        polygon = affine_transform(polygon, [1, 0, 0, -1, 0, image_instance.height])
        polygon = affine_transform(polygon, [0, 1, 1, 0, 0, 0])
        offset, width, height = _polygon_box(polygon)
        groundtruth = mk_groundtruth_image(gt_locations, width, height,
                                           image_instance.height, offset)
        return cls(parent_id=crop.image,
                   roi_id=crop.id,
                   width=width,
                   height=height,
                   image_filename=crop.filename,
                   groundtruth_filename=save_groundtruth_image(
                           groundtruth, working_path, crop.project,
                           _roi_filename(crop.image, crop.id))
                   )


def mk_groundtruth_image(gt_locations, image_width, image_height, parent_height, offset=(0, 0)):
    points = []
    for pt in gt_locations:
        if not isinstance(pt, Point):
            pt = pt.centroid

        # Transform point to match top-left origin point
        # instead of bottom-left used by Cytomine
        pt = affine_transform(pt, [1, 0, 0, -1, 0, parent_height])
        pt = translate(pt, -offset[1], -offset[0])

        if pt.y < image_height and pt.x < image_width:
            points.append(pt)

    points = np.array([np.array([int(p.y), int(p.x)]) for p in points]).T
    groundtruth = np.zeros((image_height, image_width))
    groundtruth[tuple(points)] = 1
    return groundtruth


def save_groundtruth_image(groundtruth, working_path, id_project, roi_filename):
    path = os.path.join(working_path, GROUNDTRUTHS_PATH, str(id_project))
    if not os.path.exists(path):
        os.makedirs(path)

    filename = os.path.join(path, roi_filename + ".png")
    cv2.imwrite(filename, groundtruth * 255)
    return filename


##########################################################################################


def rois_from_imageinstances(cytomine, working_path, id_project, id_term,
                             id_user=None, reviewed_only=False,
                             force_download=False):
    image_instances = cytomine.dump_project_images(id_project=id_project,
                                                   dest_path="/"+IMAGE_INSTANCES_PATH,
                                                   max_size=True,
                                                   override=force_download)

    rois = []
    for image_instance in image_instances:
        gt_annots = cytomine.get_annotations(id_project=id_project,
                                             id_image=image_instance.id,
                                             id_term=id_term,
                                             id_user=id_user,
                                             reviewed_only=reviewed_only,
                                             showWKT=True).data()
        gt_locations = [loads(gt.location) for gt in gt_annots]
        try:
            roi = RegionOfInterest.from_image_instance(image_instance, gt_locations,
                                                       working_path)
            rois.append(roi)
        except ValueError:
            print "Impossible to create ROI for {}".format(image_instance)

    return rois


def rois_from_annotation_crops(cytomine, working_path, id_project, id_term, id_roi_term,
                               id_user=None, reviewed_only=False,
                               id_roi_user=None, reviewed_only_roi=False,
                               force_download=False):
    # Download annotations
    crops_annotations = cytomine.get_annotations(id_project=id_project,
                                                 id_term=id_roi_term,
                                                 id_user=id_roi_user,
                                                 reviewed_only=reviewed_only_roi,
                                                 showWKT=True,
                                                 showMeta=True)

    # Download crops
    crops = cytomine.dump_annotations(annotations=crops_annotations,
                                      dest_path=os.path.join(working_path,
                                                             CROPS_PATH, str(id_project)),
                                      override=force_download,
                                      desired_zoom=0,
                                      get_image_url_func=Annotation.get_annotation_alpha_crop_url).data()

    rois = []
    for crop in crops:
        gt_annots = cytomine.get_annotations(id_project=id_project,
                                             id_image=crop.image,
                                             id_bbox=crop.id,
                                             id_term=id_term,
                                             id_user=id_user,
                                             reviewed_only=reviewed_only,
                                             showWKT=True).data()
        gt_locations = [loads(gt.location) for gt in gt_annots]
        try:
            roi = RegionOfInterest.from_annotation_crop(cytomine.get_image_instance(crop.image),
                                                        crop, gt_locations, working_path)
            rois.append(roi)
        except ValueError:
            print "Impossible to create ROI for {}".format(crop)

    return rois


def rois_from_cytomine(image_as_roi, cytomine, working_path, id_project, id_term,
                       id_roi_term=None, id_user=None, reviewed_only=None,
                       id_roi_user=None, reviewed_only_roi=False, force_download=False):
    if image_as_roi:
        return rois_from_imageinstances(cytomine, working_path, id_project, id_term,
                                        id_user, reviewed_only, force_download)
    else:
        return rois_from_annotation_crops(cytomine, working_path, id_project, id_term,
                                          id_roi_term, id_user, reviewed_only, id_roi_user,
                                          reviewed_only_roi, force_download)


def rois_from_dataset(dataset_name, working_path="tmp/", force_download=False):
    from cytomine_identifiers import DATASETS, CYTOMINE_KEYS, EXCLUDED
    dataset = DATASETS[dataset_name]
    excluded = EXCLUDED[dataset_name]

    rois = []
    if not force_download:
        if dataset['image_as_roi']:
            path_img = os.path.join(dataset['local_path'], 'images')
            path_gts = os.path.join(dataset['local_path'], 'groundtruths')
            print path_img
            print path_gts
            if os.path.exists(path_img) and os.path.exists(path_gts) \
                    and os.listdir(path_img) and os.listdir(path_gts):
                print 2
                imgs = files_in_directory(path_img)
                for img in imgs:
                    name, ext = os.path.splitext(os.path.basename(img))
                    img_id = int(name)
                    img = os.path.join(path_img, img)
                    gt = os.path.join(path_gts, "{}_{}.png".format(name, name))

                    # XXX: improve, remove image opening
                    image = open_image(img)
                    width = image.shape[1]
                    height = image.shape[0]
                    del image

                    try:
                        if img_id not in excluded:
                            rois.append(RegionOfInterest(img_id, img_id, width, height,
                                                         img, gt))
                    except ValueError:
                        print "Impossible to create ROI for {}".format(name)
                return rois
        else:
            path_img = os.path.join(working_path, CROPS_PATH, str(dataset['id']), str(dataset['roi_term']))
            path_gts = os.path.join(working_path, GROUNDTRUTHS_PATH, str(dataset['id']))
            if os.path.exists(path_img) and os.path.exists(path_gts) \
                    and os.listdir(path_img) and os.listdir(path_gts):
                imgs = files_in_directory(path_img)
                for img in imgs:
                    name, ext = os.path.splitext(os.path.basename(img))
                    image_id, crop_id = name.split('_')
                    image_id, crop_id = int(image_id), int(crop_id)
                    img = os.path.join(path_img, img)
                    gt = os.path.join(path_gts, "{}_{}.png".format(image_id, crop_id))

                    # XXX: improve, remove image opening
                    crop = open_image(gt, cv2.IMREAD_GRAYSCALE)
                    width = crop.shape[1]
                    height = crop.shape[0]
                    del crop

                    try:
                        if image_id not in excluded and crop_id not in excluded:
                            rois.append(RegionOfInterest(image_id, crop_id, width, height,
                                                         img, gt))
                    except ValueError:
                        print "Impossible to create ROI for {}".format(crop_id)
                return rois

    conn = Cytomine(
            dataset['host'],
            public_key=CYTOMINE_KEYS[dataset['host']]['public_key'],
            private_key=CYTOMINE_KEYS[dataset['host']]['private_key'],
            working_path=working_path,
            base_path="/api/",
            verbose=False
    )

    if dataset['image_as_roi']:
        rois = rois_from_imageinstances(conn, working_path,
                                        dataset['id'], dataset['cell_term'],
                                        id_user=dataset['users'], reviewed_only=False)
    else:
        rois = rois_from_annotation_crops(conn, working_path,
                                          dataset['id'], dataset['cell_term'], dataset['roi_term'],
                                          id_user=dataset['users'], reviewed_only=False,
                                          id_roi_user=dataset['users'],
                                          reviewed_only_roi=dataset['reviewed_only'])
    rois = [roi for roi in rois if roi.parent_id not in excluded and roi.roi_id not in excluded]
    return rois


def load_dataset(parameters, cytomine):
    if parameters.dataset:
        rois = rois_from_dataset(parameters.dataset,
                                 working_path=parameters.cytomine_working_path,
                                 force_download=parameters.cytomine_force_download)
    else:
        rois = rois_from_cytomine(parameters.image_as_roi, cytomine,
                                  working_path=parameters.cytomine_working_path,
                                  id_project=parameters.cytomine_project,
                                  id_term=parameters.cytomine_object_term,
                                  id_user=parameters.cytomine_object_user,
                                  reviewed_only=parameters.cytomine_object_reviewed_only,
                                  id_roi_term=parameters.cytomine_roi_term,
                                  id_roi_user=parameters.cytomine_roi_user,
                                  reviewed_only_roi=parameters.cytomine_roi_reviewed_only,
                                  force_download=parameters.cytomine_force_download)

    X = np.array([r.image_filename for r in rois])
    y = np.array([r.groundtruth_filename for r in rois])

    if parameters.cv_labels == 'roi':
        labels = np.array([r.roi_id for r in rois])
    else:
        labels = np.array([r.parent_id for r in rois])

    return X, y, labels


if __name__ == '__main__':
    for d in ['BM GRAZ', 'GANGLIONS', 'ANAPATH', 'BRCA', 'CRC']:
        rois = rois_from_dataset(d, force_download=False)
        print len(rois)
        for r in rois:
            print r
