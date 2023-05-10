# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
import os
from ikomia import utils, dataprocess, core
from ikomia.core.task import TaskParam
from ikomia.dnn import dnntrain
from train_transunet.transunet_utils import my_trainer
from pathlib import Path
import numpy as np
from train_transunet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from train_transunet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ml_collections import ConfigDict
import yaml


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class TrainTransunetParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        # Place default value initialization here
        self.cfg["model_name"]= "TransUNet"
        self.cfg["input_size"] = 256
        self.cfg["patchSize"] = 16
        self.cfg["max_iter"] = 1000
        self.cfg["batch_size"] = 1
        self.cfg["dataset_split_ratio"] = 90
        self.cfg["evalPeriod"] = 100
        self.cfg["use_pretrain"] = True
        self.cfg["output_folder"] = ""
        self.cfg["learning_rate"] = 0.01
        self.cfg["config_file"] = ""
        self.cfg["earlyStopping"] = False
        self.cfg["patience"] = -1

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cfg["input_size"] = int(param_map["input_size"])
        self.cfg["patchSize"] = int(param_map["patchSize"])
        self.cfg["max_iter"] = int(param_map["max_iter"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["dataset_split_ratio"] = int(param_map["dataset_split_ratio"])
        self.cfg["evalPeriod"] = int(param_map["evalPeriod"])
        self.cfg["output_folder"] = param_map["output_folder"]
        self.cfg["learning_rate"] = float(param_map["learning_rate"])
        self.cfg["use_pretrain"] = utils.strtobool(param_map["use_pretrain"])
        self.cfg["config_file"] = param_map["config_file"]
        self.cfg["earlyStopping"] = utils.strtobool(param_map["earlyStopping"])
        self.cfg["patience"] = int(param_map["patience"])


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class TrainTransunet(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name,param)
        self.stop_train = False
        # Create parameters class
        if param is None:
            self.set_param_object(TrainTransunetParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.get_param_object()
        if param is not None:
            return param.cfg["max_iter"]
        else:
            return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        self.stop_train = False
        self.problem = False
        dir_path = os.path.dirname(__file__)
        pretrained_path= os.fspath(Path(dir_path+"/networks/"+"R50+ViT-B_16.npz"))
        # download pretrained weights
        if not(os.path.isfile(pretrained_path)):
            import requests
            print('Downloading weights')

            url = 'https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/R50%2BViT-B_16.npz'
            response = requests.get(url, stream=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(pretrained_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong during downloading")
                self.problem = True
            print('Weights downloaded')

        input = self.get_input(0)
        if len(input.data) == 0:
            print("ERROR, there is no input dataset")
            self.problem = True
        else:
            # complete class names if input dataset has no background class
            if not(input.has_bckgnd_class):
                tmp_dict = {0:"background"}
                for k,name in input.data["metadata"]["category_names"].items():
                    tmp_dict[k+1]=name
                input.data["metadata"]["category_names"] = tmp_dict
                input.has_bckgnd_class = True
            num_classes = len(input.data["metadata"]["category_names"])

        # Get parameters :
        param = self.get_param_object()
        expert_mode = param.cfg["config_file"]
        # current datetime is used as folder name
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")

        if os.path.isfile(expert_mode):
            # load config file
            with open(expert_mode, 'r') as file:
                str = yaml.load(file, Loader=yaml.Loader)
                config_vit = ConfigDict(str)
                pretrained_path = config_vit.pretrained_path
                output_path = config_vit.output_path

        else:
            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
            config_vit.pretrained_path = pretrained_path
            config_vit.batch_size = param.cfg["batch_size"]
            config_vit.img_size = param.cfg["input_size"]
            config_vit.max_iter = param.cfg["max_iter"]
            config_vit.split_train_test = param.cfg["dataset_split_ratio"] / 100
            config_vit.eval_period = param.cfg["evalPeriod"]
            config_vit.base_lr = param.cfg["learning_rate"]
            config_vit.patch_size = 16
            config_vit.n_classes = num_classes
            config_vit.freeze_backbone = True
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(config_vit.img_size / config_vit.patch_size), int(config_vit.img_size / config_vit.patch_size))
            config_vit.class_names = [name for k, name in input.data["metadata"]["category_names"].items()]
            #config_vit.warmup_iters = config_vit.max_iter // 3
            config_vit.warmup_iters = None
            config_vit.warmup_factor = None
            #config_vit.warmup_factor = 0.001
            config_vit.patience = param.cfg["patience"]
            if os.path.isdir(param.cfg["output_folder"]):
                output_path = os.path.join(param.cfg["output_folder"], str_datetime)
            else:
                output_path = os.path.join(dir_path, "output", str_datetime)

            # create output folder
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            config_vit.output_path = output_path

        if config_vit.img_size % config_vit.patch_size != 0:
            print("ERROR, image size must be divisible by patch size")
            self.problem = True

        if not self.problem:
            # initialize model
            model = ViT_seg(config_vit, img_size=config_vit.img_size, num_classes=config_vit.n_classes).cuda()

            # load weights from pretrained path
            if os.path.isfile(config_vit.pretrained_path) :
                with np.load(pretrained_path) as data:
                    pretrained_names = data.files
                    pretrained_dict = {}
                    for i, name in enumerate(pretrained_names):
                        pretrained_dict[Path(name)] = data[pretrained_names[i]]

                model.load_from(weights=pretrained_dict)

            tb_logdir = os.path.join(core.config.main_cfg["tensorboard"]["log_uri"],
                                     param.cfg["model_name"],
                                     str_datetime)
            tb_writer = SummaryWriter(tb_logdir)

            # freeze resnet layers
            if config_vit.freeze_backbone:
                for param in model.transformer.embeddings.hybrid_model.parameters():
                    param.requires_grad = False

            # train model
            my_trainer(model, config_vit, input.data, self.get_stop, self.emit_step_progress, tb_writer)
            with open(os.path.join(output_path, "config.yaml"), 'w') as fp:
                fp.write(config_vit.to_yaml())

        # Call end_task_run to finalize process
        self.end_task_run()

    def get_stop(self):
        return self.stop_train

    def stop(self):
        super().stop()
        self.stop_train=True


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class TrainTransunetFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_transunet"
        self.info.short_description = "Training process for TransUNet model. "
        self.info.description = "Training process for TransUNet model. " \
                                "This Ikomia plugin can train TransUNet model for semantic segmantation. " \
                                "Most common parameters are exposed in the settings window. For expert usage, " \
                                "it is also possible to select a custom configuration file." \
                                "To start your training:" \
                                "create a new workflow, " \
                                "add a task node loading your dataset in Ikomia format " \
                                "(consult the marketplace to check if a suitable dataset loader already exists), " \
                                "add this TransUNet train task, " \
                                "adjust parameters, " \
                                "and click apply to start the training. " \
                                "You are able to monitor your training runs through the TensorBoard dashboard. " \
                                "Compared to original paper, image preprocessing has been changed to match " \
                                "ResNet trained on ImageNet image preprocessing."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.0.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.icon_path = "icons/transunet.png"
        self.info.authors = "Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo,Ehsan Adeli, Yan Wang, Le Lu, Alan L. Yuille, and Yuyin Zhou"
        self.info.article = "TransUNet: Transformers Make StrongEncoders for Medical Image Segmentation"
        self.info.journal = "not published yet"
        self.info.year = 2021
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2102.04306"
        # Code source repository
        self.info.repository = "https://github.com/Beckschen/TransUNet"
        # Keywords used for search
        self.info.keywords = "semantic, segmentation, encoder, decoder, Transformers, U-Net "

    def create(self, param=None):
        # Create process object
        return TrainTransunet(self.info.name, param)
