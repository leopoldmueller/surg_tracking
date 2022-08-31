from mimetypes import init
import cv2
from torchvision.io import read_image
import os
import torch
import numpy as np
from torch import linalg as LA
import math
import time
from scipy import ndimage
from xml.etree.ElementTree import TreeBuilder
import pickle
import time
import torchvision.models as models

class FM_CSRT_Tracker:
    """This is the best performing tracker brougth to you by
    - Leopold Müller
    - Simeon Allmendinger
    - Lars Böcking
    """

    def __init__(self, init_im1, init_im2, init_bbox1, init_bbox2, config):
        """_summary_

        Args:
            init_im1 (_type_): initial left image where bounding box is defined for
            init_im2 (_type_): initial right image where bounding box is defined for
            init_bbox1 (_type_): initial bounding box to follow for left image
            init_bbox2 (_type_): initial bounding box to follow for right image
            config: reference to the config file from where hyperparameters are extracted
        """
        # intialize csrt tracker
        self.t1 = cv2.TrackerCSRT_create()
        self.t2 = cv2.TrackerCSRT_create()
        print('fm_csrt_tracker')

        self.t1.init(init_im1, init_bbox1)
        self.t2.init(init_im2, init_bbox2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # extract the condif hyper parameters
        self.model_choice = config["fm_csrt_tracker"]["model_choice"]
        self.zero_padding = config["fm_csrt_tracker"]["zero_padding"]
        self.surrounding = config["fm_csrt_tracker"]["surrounding"]
        self.zero_mapping_size = config["fm_csrt_tracker"]["zero_mapping_size"]
        self.norm = config["fm_csrt_tracker"]["norm"]
        self.step_length = config["fm_csrt_tracker"]["step_length"]
        self.number_candidates = config["fm_csrt_tracker"]["number_candidates"]
        self.pooling_type = config["fm_csrt_tracker"]["pooling_type"]
        self.pooling_mode = config["fm_csrt_tracker"]["pooling_mode"]
        self.layer_index = config["fm_csrt_tracker"]["layer_index"]

        # prioritized models pretrained torch models https://pytorch.org/vision/0.8/models.html
        if self.model_choice == "alexnet":
            self.model = models.alexnet(pretrained=True)
        elif self.model_choice == "squeezenet1_0":
            self.model = models.squeezenet1_0(pretrained=True)
        elif self.model_choice == "inception_v3":
            self.model = models.inception_v3(pretrained=True)
        elif self.model_choice == "googlenet":
            self.model = models.googlenet(pretrained=True)
        elif self.model_choice == "efficientnet_b0":
            self.model = models.efficientnet_b0(pretrained=True)
        elif self.model_choice == "resnet50":
            self.model = models.resnet50(pretrained=True)

        # deprio models
        elif self.model_choice == "resnet18":
            self.model = models.resnet18(pretrained=True)
        elif self.model_choice == "vgg16":
            self.model = models.vgg16(pretrained=True)
        elif self.model_choice == "densenet161":
            self.model = models.densenet161(pretrained=True)
        elif self.model_choice == "shufflenet_v2_x1_0":
            self.model = models.shufflenet_v2_x1_0(pretrained=True)
        elif self.model_choice == "mobilenet_v2":
            self.model = models.mobilenet_v2(pretrained=True)
        elif self.model_choice == "resnext50_32x4d":
            self.model = models.resnext50_32x4d(pretrained=True)
        elif self.model_choice == "wide_resnet50_2":
            self.model = models.wide_resnet50_2(pretrained=True)
        elif self.model_choice == "mnasnet1_0":
            self.model = models.mnasnet1_0(pretrained=True)
        else:
            self.model = models.resnet50(pretrained=True)

        self.layer_name = [n for n, _ in self.model.named_children()][self.layer_index]

        # Set model to evaluation mode
        self.model.eval()

        # Apply inference preprocessing transforms on image with zero padding
        init_im1_prep_padded = self.extract_bounding_box(
            init_im1, init_bbox1
        ).unsqueeze(0)
        init_im2_prep_padded = self.extract_bounding_box(
            init_im2, init_bbox2
        ).unsqueeze(0)

        # calculate the target feature maps for both initial frames
        self.init_im1_fm = self.compute_fm(init_im1_prep_padded)
        self.init_im2_fm = self.compute_fm(init_im2_prep_padded)

        # save the initial bounding boxes for later reference
        self.init_bbox1 = init_bbox1
        self.init_bbox2 = init_bbox2

        # set the variable for latest bounding box to the inital one
        self.latest_bbox1 = init_bbox1
        self.latest_bbox2 = init_bbox2

        # set the counter for each video side to zero
        self.prediction_counter_bbox1 = 0
        self.prediction_counter_bbox2 = 0

        # set value for experiment start
        self.experiment_time = int(time.time())

    # Method to compute freature map for given image
    def compute_fm(self, img):
        """computing the feature map for a given image

        Args:
            img (_type_): image for which feature maps should be calculated

        Returns:
            _type_: feature map
        """

        # get feature map of predefined layer in network
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        # TODO add variable to access layer by e.g. index
        getattr(self.model, self.layer_name).register_forward_hook(get_activation(self.layer_name))
        #self.model.avgpool.register_forward_hook(get_activation(self.layer_name))

        # Process image
        if torch.cuda.is_available():
            self.model.cuda()(img.float())
        else:
            self.model(img.float())

        # return the feature map
        return activation[self.layer_name]

    # Create zero padding arround given bb
    def extract_bounding_box(self, img, bb):
        """_summary_

        Args:
            img (_type_): image for which the bounding box should be extracted
            bb (_type_): dimensions of shape (u,v,w,h)

        Returns:
            _type_: extracted area of the image
        """
        img = torch.from_numpy(img).to(self.device).permute(2, 0, 1)

        # generate zero map and add bounding box to the center
        if self.zero_padding:
            mask = torch.zeros(3, self.zero_mapping_size, self.zero_mapping_size)

            mask[
                :,
                math.floor(img.size()[1] / 2) : math.floor(img.size()[1] / 2)
                + bb[2]
                + 2 * self.surrounding,
                math.floor(img.size()[2] / 2) : math.floor(img.size()[2] / 2)
                + bb[3]
                + 2 * self.surrounding,
            ] = img[
                :,
                bb[0] - self.surrounding : bb[0] + bb[2] + self.surrounding,
                bb[1] - self.surrounding : bb[1] + bb[3] + self.surrounding,
            ]

            return_image = mask

        # or just return the extracted bounding box are from the image
        else:
            return_image = img[
                :,
                bb[0] - self.surrounding : bb[0] + bb[2] + self.surrounding,
                bb[1] - self.surrounding : bb[1] + bb[3] + self.surrounding,
            ]

        return return_image

    def evaluate_bboxes(self, img, bb, reference_fm):
        """_summary_

        Args:
            img (_type_): image for which the bounding box should be evaluated
            bb (_type_): bounding box that should be evaluated
            reference_fm (_type_): target feature map of the inital image

        Returns:
            _type_: distance and feature map
        """

        img = self.extract_bounding_box(img, bb).unsqueeze(0)
        fm = self.compute_fm(img)

        # euclidian norm
        if self.norm == "euclidian":
            loss = LA.vector_norm(fm - reference_fm, ord=3)

        # cosine norm
        elif self.norm == "cosine":
            if not fm.size()==reference_fm.size():
                print('fm', fm)
                print('fm.size', fm.size())
                print('fm', reference_fm)
                print('fm', reference_fm.size())
            cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
            loss = - cosine(fm, reference_fm).sum()

        # standard deviation of the difference
        elif self.norm == "standard":
            if torch.cuda.is_available():
                loss = (reference_fm - fm).cpu().numpy().std()
            else:
                loss = (reference_fm - fm).numpy().std()

        # absolute norm
        else:
            loss = LA.norm(reference_fm - fm)

        return loss.item(), fm

    # check if a bounding box is valid for a given image examining the size
    def invalid_bb(self, bb, img):
        """_summary_

        Args:
            bb (_type_): bounding box to check
            img (_type_): image to check
            self.surrounding (int, optional):  number of additional pixcels to consider in each dimension. Defaults to 0.

        Returns:
            _type_: boolean if bounding box is within image dimensions
        """

        # TODO self.surrounding shouldn't lead to invalid bounding boxes if bounding box itself is valid --> zero padding
        return (
            bb[0] - self.surrounding < 0
            or bb[1] - self.surrounding < 0
            or (bb[0] + bb[2]) + self.surrounding > img.shape[0]
            or (bb[1] + bb[3]) + self.surrounding > img.shape[1]
        )

    # search for next bounding box
    def update_bb(self, image_side, img, reference_bb, reference_fm, save_results=False):
        """_summary_

        Args:
            image_side (_type_): _description_
            img (_type_): _description_
            reference_bb (_type_): _description_
            reference_fm (_type_): _description_
            save_results (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        best_bb = reference_bb

        # TODO Are we missing the returned feature map anywhere?
        best_distance, _ = self.evaluate_bboxes(
            img=img, bb=best_bb, reference_fm=reference_fm
        )
        # save the feature maps
        if save_results:
            results = {}
            results["image"] = img
            results["reference_fm"] = [reference_fm]
            results["distances"] = []
            results["distance_mapping"] = []
            results["feature_maps"] = []

        if self.pooling_mode:
            distance_candidates = []
            bbox_candidates = []

        # number_candidates fixe Anzahl an kandiaten, search room abgeleitet aus number_candidates * step length
        # 3 abweichungen in jede richtung --> 7x7 kandiaten = 49 kandiaten
        for u_delta in np.arange(
                -self.number_candidates * self.step_length,
                self.number_candidates * self.step_length + 1,
                self.step_length,
            ):
            for v_delta in np.arange(
                    -self.number_candidates * self.step_length,
                    self.number_candidates * self.step_length + 1,
                    self.step_length,
                )[::-1]:
                # TODO add adjustment for bounding box to search space
                w_delta = 0
                h_delta = 0

                new_bb = reference_bb
                new_bb = np.array(new_bb)
                new_bb[0] += u_delta
                new_bb[1] += v_delta
                new_bb[2] += w_delta
                new_bb[3] += h_delta
                new_bb = tuple(new_bb)

                if self.invalid_bb(new_bb, img):
                    if self.pooling_mode:
                        # print("---INVALID BOUNDING BOX----")
                        bbox_candidates.append(new_bb)
                        # If bb is invalid add infinity as distance
                        distance_candidates.append(np.inf)
                    continue

                actual_distance, feature_map = self.evaluate_bboxes(
                    img=img, bb=new_bb, reference_fm=reference_fm
                )

                if save_results:
                    results["distance_mapping"].append((f"{u_delta},{v_delta}"))
                    results["distances"].append(round(actual_distance, 0))
                    results["feature_maps"].append(feature_map)

                if actual_distance < best_distance:
                    best_bb = new_bb
                    best_distance = actual_distance

                if self.pooling_mode:
                    distance_candidates.append(actual_distance)
                    bbox_candidates.append(new_bb)

        # save per bounding box and time stamp
        if save_results:
            # save the distances by u,v coordinates
            results["distance_mapping"] = (
                np.array(results["distance_mapping"])
                .reshape(self.number_candidates * 2 + 1, self.number_candidates * 2 + 1)
                .T
            )
            results["distances"] = (
                np.array(results["distances"])
                .reshape(self.number_candidates * 2 + 1, self.number_candidates * 2 + 1)
                .T
            )
            results["feature_maps"] = (
                np.array(results["feature_maps"])
                .reshape(self.number_candidates * 2 + 1, self.number_candidates * 2 + 1)
                .T
            )

            results["pred_bb"] = best_bb
            results["reference_bb"] = reference_bb

            # save experiments interim results for both frames
            if image_side == 1:
                os.makedirs(
                    os.path.dirname(
                        f"./results/experiments/experiment_{self.experiment_time}/side_{image_side}/"
                    ),
                    exist_ok=True,
                )
                with open(
                    f"./results/experiments/experiment_{self.experiment_time}/side_{image_side}/step_{self.prediction_counter_bbox1}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(results, f)
            else:
                os.makedirs(
                    os.path.dirname(
                        f"./results/experiments/experiment_{self.experiment_time}/side_{image_side}/"
                    ),
                    exist_ok=True,
                )
                with open(
                    f"./results/experiments/experiment_{self.experiment_time}/side_{image_side}/step_{self.prediction_counter_bbox2}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(results, f)

        # upate the counter for predicted bounding boxes after the search
        if image_side == 1:
            self.prediction_counter_bbox1 += 1
        else:
            self.prediction_counter_bbox2 += 1

        # find best bbox for poolingmode
        if self.pooling_mode:
            distance_candidates = (
                np.array(distance_candidates)
                .reshape(self.number_candidates * 2 + 1, self.number_candidates * 2 + 1)
                .T
            )
            k = np.ones((3, 3))

            # average distance_candidates by pooling
            distance_candidates = ndimage.convolve(
                distance_candidates, k, mode=self.pooling_type
            )

            bbox_candidates = np.transpose(
                np.array(bbox_candidates).reshape(
                    self.number_candidates * 2 + 1, self.number_candidates * 2 + 1, 4
                ),
                axes=(1, 0, 2),
            )
            best_bb = bbox_candidates[
                np.unravel_index(
                    np.argmin(distance_candidates),
                    (self.number_candidates * 2 + 1, self.number_candidates * 2 + 1),
                )
            ]

        return best_bb

    def tracker_update(self, new_im1, new_im2):
        """
        Return two bboxes in format (u, v, width, height)

                             (u,)   (u + width,)
                      (0,0)---.--------.---->
                        |
                   (,v) -     x--------.
                        |     |  bbox  |
          (,v + height) -     .________.
                        v
        """

        success1, bbox1 = self.t1.update(new_im1)
        success2, bbox2 = self.t2.update(new_im2)

        if success1:
            if not self.invalid_bb(bbox1, new_im1):
                self.latest_bbox1 = self.update_bb(
                image_side=1,
                img=new_im1,
                reference_bb=bbox1,
                reference_fm=self.init_im1_fm,
                )
            else:
                self.latest_bbox1 = bbox1
        else:
            self.latest_bbox1 = None
            
        if success2:
            if not self.invalid_bb(bbox2, new_im2):
                self.latest_bbox2 = self.update_bb(
                image_side=1,
                img=new_im2,
                reference_bb=bbox2,
                reference_fm=self.init_im2_fm,
                )
            else:
                self.latest_bbox2 = bbox2
        else:
            self.latest_bbox2 = None

        return self.latest_bbox1, self.latest_bbox2
