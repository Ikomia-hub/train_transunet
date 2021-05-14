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

from ikomia import core, dataprocess
import copy
# Your imports below
from ikomia.dnn import datasetio
from ikomia.dnn import dnntrain
import os
import utils
from pathlib import Path
import numpy as np
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ml_collections import ConfigDict
import yaml

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class TransUNet_TrainParam(dnntrain.TrainParam):

    def __init__(self):
        dnntrain.TrainParam.__init__(self)
        # Place default value initialization here
        # Example : self.windowSize = 25
        self.cfg["modelName"]= "TransUNet"
        self.cfg["inputSize"] = 256
        self.cfg["patchSize"] = 16
        self.cfg["maxIter"] = 1000
        self.cfg["batchSize"] = 1
        self.cfg["splitTrainTest"] = 90
        self.cfg["evalPeriod"] = 100
        self.cfg["pretrain"] = True
        self.cfg["outputFolder"] = ""
        self.cfg["baseLearningRate"] = 0.01
        self.cfg["pretrain"] = True
        self.cfg["expertMode"] = ""

    def setParamMap(self, paramMap):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(paramMap["windowSize"])
        self.cfg["inputSize"] = int(paramMap["inputSize"])
        self.cfg["patchSize"] = int(paramMap["patchSize"])
        self.cfg["maxIter"] = int(paramMap["maxIter"])
        self.cfg["batchSize"] = int(paramMap["batchSize"])
        self.cfg["splitTrainTest"] = int(paramMap["splitTrainTest"])
        self.cfg["evalPeriod"] = int(paramMap["evalPeriod"])
        self.cfg["pretrain"] = bool(paramMap["pretrain"])
        self.cfg["outputFolder"] = paramMap["outputFolder"]
        self.cfg["baseLearningRate"] = float(paramMap["baseLearningRate"])
        self.cfg["pretrain"] = bool(paramMap["pretrain"])
        self.cfg["expertMode"] = paramMap["expertMode"]

# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class TransUNet_TrainProcess(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name,param)
        # Add input/output of the process here
        self.addInput(datasetio.IkDatasetIO())
        self.stop_train=False
        # Create parameters class
        if param is None:
            self.setParam(TransUNet_TrainParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.getParam()
        if param is not None:
            return param.cfg["maxIter"]
        else:
            return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        self.stop_train = False
        self.problem = False
        dir_path = os.path.dirname(__file__)
        pretrained_path= os.fspath(Path(dir_path+"/networks/"+"R50+ViT-B_16.npz"))
        print(pretrained_path)
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

        input = self.getInput(0)
        if len(input.data) == 0:
            print("ERROR, there is no input dataset")
            self.problem = True
        else:
            if not(input.has_bckgnd_class):
                tmp_dict = {0:"background"}
                for k,name in input.data["metadata"]["category_names"].items():
                    tmp_dict[k+1]=name
                input.data["metadata"]["category_names"] = tmp_dict
                input.has_bckgnd_class = True
            num_classes = len(input.data["metadata"]["category_names"])

        # Get parameters :
        param = self.getParam()
        expert_mode = param.cfg["expertMode"]

        if os.path.isfile(expert_mode):
            with open(expert_mode, 'r') as file:
                str = yaml.load(file, Loader=yaml.FullLoader)
                config_vit = ConfigDict(str)
                pretrained_path = config_vit.pretrained_path
                output_path = config_vit.output_path
        else:
            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
            config_vit.pretrained_path = pretrained_path
            config_vit.batch_size = param.cfg["batchSize"]
            config_vit.img_size = param.cfg["inputSize"]
            config_vit.max_iter = param.cfg["maxIter"]
            config_vit.split_train_test = param.cfg["splitTrainTest"] / 100
            config_vit.eval_period = param.cfg["evalPeriod"]
            config_vit.base_lr = param.cfg["baseLearningRate"]
            config_vit.patch_size = 16
            config_vit.n_classes = num_classes
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(config_vit.img_size / config_vit.patch_size), int(config_vit.img_size / config_vit.patch_size))
            config_vit.class_names = [name for k, name in input.data["metadata"]["category_names"].items()]
            str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
            if os.path.isdir(param.cfg["outputFolder"]):
                output_path = os.path.join(param.cfg["outputFolder"], str_datetime)
            else:
                output_path = os.path.join(dir_path, "output", str_datetime)

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            config_vit.output_path = output_path

        if config_vit.img_size % config_vit.patch_size != 0:
            print("ERROR, image size must be divisible by patch size")
            self.problem = True

        if not(self.problem):


            model = ViT_seg(config_vit, img_size=config_vit.img_size, num_classes=config_vit.n_classes).cuda()

            if os.path.isfile(config_vit.pretrained_path) :
                with np.load(pretrained_path) as data:
                    pretrained_names = data.files
                    pretrained_dict = {}
                    for i, name in enumerate(pretrained_names):
                        pretrained_dict[Path(name)] = data[pretrained_names[i]]

                model.load_from(weights=pretrained_dict)

            tb_logdir = Path(self.getTensorboardLogDir()+"/"+param.cfg["modelName"]+"/"+str_datetime)
            tb_writer = SummaryWriter(tb_logdir)
            utils.my_trainer(model, config_vit, input.data,self.get_stop,self.emitStepProgress,tb_writer)
            with open(os.path.join(output_path,"config.yaml"), 'w') as fp:
                fp.write(config_vit.to_yaml())

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def get_stop(self):
        return self.stop_train

    def stop(self):
        super().stop()
        self.stop_train=True


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class TransUNet_TrainProcessFactory(dataprocess.CProcessFactory):

    def __init__(self):
        dataprocess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "TransUNet_Train"
        self.info.shortDescription = "your short description"
        self.info.description = "your description"
        self.info.authors = "Plugin authors"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "algorithm author"
        self.info.article = "title of associated research article"
        self.info.journal = "publication journal"
        self.info.year = 2021
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentationLink = ""
        # Code source repository
        self.info.repository = ""
        # Keywords used for search
        self.info.keywords = "your,keywords,here"

    def create(self, param=None):
        # Create process object
        return TransUNet_TrainProcess(self.info.name, param)
