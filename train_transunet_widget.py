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

from ikomia import utils, core, dataprocess
from ikomia.utils import qtconversion, pyqtutils
from train_transunet.train_transunet_process import TrainTransunetParam
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CProtocolTaskWidget from Ikomia API
# --------------------
class TrainTransunetWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = TrainTransunetParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        inputSizeLabel = QLabel("Desired input size:")
        self.inputSizeSpinBox = QSpinBox()
        self.inputSizeSpinBox.setRange(16, 4096)
        self.inputSizeSpinBox.setSingleStep(1)
        self.inputSizeSpinBox.setValue(self.parameters.cfg["input_size"])

        maxIterLabel = QLabel("Max iter:")
        self.maxIterSpinBox = QSpinBox()
        self.maxIterSpinBox.setRange(0, 2147483647)
        self.maxIterSpinBox.setSingleStep(1)
        self.maxIterSpinBox.setValue(self.parameters.cfg["max_iter"])

        batchSizeLabel = QLabel("Batch size:")
        self.batchSizeSpinBox = QSpinBox()
        self.batchSizeSpinBox.setRange(1, 2147483647)
        self.batchSizeSpinBox.setSingleStep(1)
        self.batchSizeSpinBox.setValue(self.parameters.cfg["batch_size"])

        splitTrainTestLabel = QLabel("Train test percentage:")
        self.splitTrainTestSpinBox = QSpinBox()
        self.splitTrainTestSpinBox.setRange(0, 100)
        self.splitTrainTestSpinBox.setSingleStep(1)
        self.splitTrainTestSpinBox.setValue(self.parameters.cfg["dataset_split_ratio"])

        evalPeriodLabel = QLabel("Evaluation period:")
        self.evalPeriodSpinBox = QSpinBox()
        self.evalPeriodSpinBox.setRange(0, 2147483647)
        self.evalPeriodSpinBox.setSingleStep(1)
        self.evalPeriodSpinBox.setValue(self.parameters.cfg["evalPeriod"])

        earlyStoppingLabel = QLabel("Early stopping:")
        self.earlyStoppingCheckBox = QCheckBox()
        self.earlyStoppingCheckBox.setChecked(self.parameters.cfg["earlyStopping"])
        self.earlyStoppingCheckBox.clicked.connect(self.showPatienceSpinBox)

        baseLearningRateLabel = QLabel("Base learning rate:")
        self.baseLearningRateSpinBox = QDoubleSpinBox()
        self.baseLearningRateSpinBox.setRange(0, 10)
        self.baseLearningRateSpinBox.setDecimals(4)
        self.baseLearningRateSpinBox.setSingleStep(0.0001)
        self.baseLearningRateSpinBox.setValue(self.parameters.cfg["learning_rate"])

        self.patienceLabel = QLabel("Patience:")
        self.patienceSpinBox = QSpinBox()
        self.patienceSpinBox.setRange(0, 2147483647)
        self.patienceSpinBox.setSingleStep(1)
        self.patienceSpinBox.setValue(self.parameters.cfg["patience"])
        if self.earlyStoppingCheckBox.isChecked():
            self.patienceLabel.show()
            self.patienceSpinBox.show()
        else:
            self.patienceLabel.hide()
            self.patienceSpinBox.hide()

        # Set widget layout
        self.gridLayout.addWidget(inputSizeLabel, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.inputSizeSpinBox, 0, 1, 1, 2)
        self.gridLayout.addWidget(maxIterLabel, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.maxIterSpinBox, 1, 1, 1, 2)
        self.gridLayout.addWidget(batchSizeLabel, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.batchSizeSpinBox, 2, 1, 1, 2)
        self.gridLayout.addWidget(splitTrainTestLabel, 3, 0, 1, 1)
        self.gridLayout.addWidget(self.splitTrainTestSpinBox, 3, 1, 1, 2)
        self.gridLayout.addWidget(evalPeriodLabel, 4, 0, 1, 1)
        self.gridLayout.addWidget(self.evalPeriodSpinBox, 4, 1, 1, 2)
        self.gridLayout.addWidget(baseLearningRateLabel, 5, 0, 1, 1)
        self.gridLayout.addWidget(self.baseLearningRateSpinBox, 5, 1, 1, 2)
        self.gridLayout.addWidget(earlyStoppingLabel,6,0,1,1)
        self.gridLayout.addWidget(self.earlyStoppingCheckBox,6,1,1,2)
        self.gridLayout.addWidget(self.patienceLabel,7,0,1,1)
        self.gridLayout.addWidget(self.patienceSpinBox,7,1,1,2)

        # Output folder
        self.browse_out_folder = pyqtutils.append_browse_file(self.gridLayout, label="Output folder",
                                                              path=self.parameters.cfg["output_folder"],
                                                              tooltip="Select folder",
                                                              mode=QFileDialog.Directory)

        self.browse_expert_mode = pyqtutils.append_browse_file(self.gridLayout, label="Advanced yaml config",
                                                               path=self.parameters.cfg["config_file"],
                                                               tooltip="Select yaml file",
                                                               mode=QFileDialog.ExistingFile)

        self.set_layout(layout_ptr)

    def showPatienceSpinBox(self):
        if self.earlyStoppingCheckBox.isChecked():
            self.patienceLabel.show()
            self.patienceSpinBox.show()
        else:
            self.patienceLabel.hide()
            self.patienceSpinBox.hide()

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.cfg["input_size"] = self.inputSizeSpinBox.value()
        self.parameters.cfg["batch_size"] = self.batchSizeSpinBox.value()
        self.parameters.cfg["max_iter"] = self.maxIterSpinBox.value()
        self.parameters.cfg["evalPeriod"] = self.evalPeriodSpinBox.value()
        self.parameters.cfg["dataset_split_ratio"] = self.splitTrainTestSpinBox.value()
        self.parameters.cfg["output_folder"] = self.browse_out_folder.path
        self.parameters.cfg["learning_rate"] = self.baseLearningRateSpinBox.value()
        self.parameters.cfg["config_file"] = self.browse_expert_mode.path
        self.parameters.cfg["earlyStopping"] = self.earlyStoppingCheckBox.isChecked()
        if self.earlyStoppingCheckBox.isChecked():
            self.parameters.cfg["patience"] = self.patienceSpinBox.value()
        else:
            self.parameters.cfg["patience"] = -1
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class TrainTransunetWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "train_transunet"

    def create(self, param):
        # Create widget object
        return TrainTransunetWidget(param, None)
