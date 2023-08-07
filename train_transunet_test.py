import logging
from ikomia.utils.tests import run_for_test

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::train transunet =====")
    input_dataset = t.get_input(0)
    params = t.get_parameters()
    params["maxIter"] = "10"
    params["batchSize"] = "1"
    params["evalPeriod"] = "5"
    params["splitTrainTest"] = "50"
    t.set_parameters(params)
    input_dataset.load(data_dict["datasets"]["semantic_segmentation"]["dataset_coco"])
    yield run_for_test(t)