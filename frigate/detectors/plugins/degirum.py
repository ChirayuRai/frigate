import logging
import queue
import time

import degirum as dg
import numpy as np
from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

logger = logging.getLogger(__name__)
DETECTOR_KEY = "degirum"

# TODO: Check yolo (one of all gen), mobilenet, efficient net,
# FOr some reason, if you use docker cp libhailo.so into the container, then run the
# install for hailort.deb again, multi process service works again
# Using an image, ensure that the results returned for yolov6 are the same for both degirum and hailo detector (degirum
# when using local and the custom pp )

# For some other reason, when you launch the dev container, ALL THE PORTS NEEDED for frigate
# are already used on some invisible process. Might be a kernel thing? Have to investigate and
# find out.

#      "PythonFile": "pp.py"
# Add this to POST PROCESS as a line in the yolov6 guy

# Pure hailo detector: average is about 5.3-5.5 ms
# degirum without post processor + returning empty detections immediately == 5.1-5.2 ms
# meaning the bottle neck is when we are packing up the output of the model into the inferenceResults obj +
# also going through pp.py to create a readable format for hte output results
# Also, we have that loop over the array for detections to map it, but I don't think that would matter too much tbh

# how to run this: first you need to sudo apt-get update && sudo apt-get upgrade && sudo apt-get install python3-dev -y
# then, install hailo runtime
# make sure on HOST machine, the hailort service is turned OFF
# That will make sure multiprocess service doesn't get invoked, and things aren't messed up
# TODO: Confirm that output of everything matches, but first I want to make sure I can have the docker file handle as much
# setup as possible --> confirmed, the outputs match :D

"""
[HailoRT] [error] CHECK_SUCCESS failed with status=HAILO_INTERNAL_FAILURE(8)
[HailoRT] [error] CHECK_SUCCESS failed with status=HAILO_INTERNAL_FAILURE(8)
[HailoRT] [error] Non-recoverable Async Infer Pipeline error. status error code: HAILO_INTERNAL_FAILURE(8)
[HailoRT] [error] Shutting down the pipeline with status HAILO_INTERNAL_FAILURE(8)
degirum.exceptions.DegirumException: Failed to perform model 'yolov6n_relu6_coco--640x640_quant_hailort_hailo8_1' inference: Model 'yolov6n_relu6_coco--640x640_quant_hailort_hailo8_1' inference failed: [CRITICAL]Operation failed
HailoRT Runtime Agent: Asynchronous inference failed, status = HAILO_INTERNAL_FAILURE.
hailo_runtime_agent.cpp: 590 [DG::HailoRuntimeAgentImpl::Forward]
When running model 'yolov6n_relu6_coco--640x640_quant_hailort_hailo8_1'

When getting ^^^ errors, just make sure your HOST has hailort service turned OFF (sudo systemctl stop hailort) and it'll work.
"""

"""
Hailo:
    yolo8: good
    yolo11: good
    tiny nas: Good
    scrfd: good
Openvino:
    yolo5: good
    yolo11: good
    yolo10: good
    yolo9: good
Orca:
    efficientdet: good
    mobiledet:  good
    mobilenet ssd coco: good





"""





### STREAM CLASS FROM DG TOOLS ###
class Stream(queue.Queue):
    """Queue-based iterable class with optional item drop"""

    # minimum queue size to avoid deadlocks:
    # one for stray result, one for poison pill in request_stop(),
    # and one for poison pill gizmo_run()
    min_queue_size = 1

    def __init__(self, maxsize=0, allow_drop: bool = False):
        """Constructor

        - maxsize: maximum stream depth; 0 for unlimited depth
        - allow_drop: allow dropping elements on put() when stream is full
        """

        if maxsize < self.min_queue_size and maxsize != 0:
            raise Exception(
                f"Incorrect stream depth: {maxsize}. Should be 0 (unlimited) or at least {self.min_queue_size}"
            )

        super().__init__(maxsize)
        self.allow_drop = allow_drop
        self.dropped_cnt = 0  # number of dropped items

    _poison = None

    def put(self, item, block: bool = True, timeout=None) -> None:
        """Put an item into the stream

        - item: item to put
        If there is no space left, and allow_drop flag is set, then oldest item will
        be popped to free space
        """
        if self.allow_drop:
            while True:
                try:
                    super().put(item, False)
                    break
                except queue.Full:
                    self.dropped_cnt += 1
                    try:
                        self.get_nowait()
                    finally:
                        pass
        else:
            super().put(item, block, timeout)

    def __iter__(self):
        """Iterator method"""
        return iter(self.get, self._poison)

    def close(self):
        """Close stream: put poison pill"""
        self.put(self._poison)


### DETECTOR CONFIG ###
class DGDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    location: str = Field(default=None, title="Inference Location")
    zoo: str = Field(default=None, title="Model Zoo")
    token: str = Field(default=None, title="DeGirum Cloud Token")


class EmptyPreproc:
    def __init__(self):
        self.has_image_inputs = False

    def forward(self, data):
        return dict(
            image_results=data.tobytes(),
            converter=lambda x, y: (x, y),
        )

    def fill_color(self):
        return (0, 0, 0)


### ACTUAL DETECTOR  ###
class DGDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: DGDetectorConfig):
        self._queue = Stream(5, allow_drop=True)
        self._zoo = dg.connect(
            detector_config.location, detector_config.zoo, detector_config.token
        )
        logger.info(f"Models in zoo: {self._zoo.list_models()}")
        self.dg_model = self._zoo.load_model(
            detector_config.model.path,
        )
        self.dg_model.measure_time = True
        # self.dg_model._preprocessor = EmptyPreproc()
        self.dg_model.input_image_format = "RAW"
        self.dg_model._postprocessor = None
        # Openvino tends to have multidevice, and they default to CPU rather than GPU or NPU
        types = self.dg_model.supported_device_types
        for type in types:
            # If openvino is supported, prioritize using gpu, then npu, then cpu
            if "OPENVINO" in type:
                self.dg_model.device_type = [
                    # "OPENVINO/GPU",
                    # "OPENVINO/NPU",
                    "OPENVINO/CPU",
                ]
            elif "HAILORT" in type:
                self.dg_model.device_type = [
                    "HAILORT/HAILO8l",
                    "HAILORT/HAILO8",
                ]
            break
        input_shape = self.dg_model.input_shape[0]
        self.model_height = input_shape[1]
        self.model_width = input_shape[2]


        # initializes model so that we don't have extra overhead here
        # frame = self.dg_model._preprocessor.forward(
        #     np.zeros((10, 10, 3), dtype=np.uint8)
        # )[0]
        # logger.info("Initializing frame")
        frame = np.zeros(
            (detector_config.model.width, detector_config.model.height, 3),
            dtype=np.uint8,
        )
        self.dg_model(frame)
        # logger.info("finished calling dg_model of frame")
        self.prediction = self.prediction_generator()
        self.none_counter = 0
        self.not_none_counter = 0
        self.overall_frame_counter = 0
        self.times = 0

    def prediction_generator(self):
        # logger.debug("Prediction generator was called")
        with self.dg_model as model:
            # data = self._queue.get()
            # result = model.predict(data)
            # return result
            while 1:
                # logger.debug(f"q size before calling get: {self._queue.qsize()}")
                data = self._queue.get()
                # logger.debug(f"q size after calling get: {self._queue.qsize()}")
                # logger.debug(
                #     f"Data we're passing into model predict: {data}, shape of data: {data.shape}"
                # )
                start = time.time_ns()
                result = model.predict(data)
                self.times += (time.time_ns() - start) * 1e-6
                # logger.info(
                #     f"Entire time taken to get result back: {self.times / self.overall_frame_counter}"
                # )
                yield result

    def detect_raw(self, tensor_input):
        # start = time.time_ns()
        self.overall_frame_counter += 1
        truncated_input = tensor_input.reshape(tensor_input.shape[1:])
        # logger.debug(f"Detect raw was called for tensor input: {tensor_input}")

        # add tensor_input to input queue
        self._queue.put(truncated_input)
        # logger.debug(f"Queue size after adding truncated input: {self._queue.qsize()}")

        # define empty detection result
        detections = np.zeros((20, 6), np.float32)
        res = next(self.prediction)
        # result = next(self.prediction)
        # return detections
        # result = self.prediction_generator()
        # logger.info(f"Result: {result}")
        # logger.info(f"Shape of res: {res.results[0]["data"]}")
        # np.savetxt(
        #     "/media/frigate/array_output.txt",
        #     res.results[0]["data"],
        #     fmt="%.5f",
        #     delimiter=",",
        # )
        # logger.debug(f"Queue size after calling for res: {self._queue.qsize()}")
        # logger.debug(f"Output of res in initial next call: {res}")
        # logger.info(
        # f"Overall frame number: {self.overall_frame_counter}, none count: {self.none_counter}, not none count: {self.not_none_counter}, none percentage: {self.none_counter / self.overall_frame_counter}"
        # )
        # logger.info(f"Time stats right after res: {self.dg_model.time_stats()}")
        # start = time.time_ns()


        # res_string = str(res)
        # logger.info(f"Res is: {res_string}")
        # logger.debug(f"Res's list of attributes: {dir(res)}")
        # logger.debug(
        #     f"Res results, {res.results}, length of results: {len(res.results)}"
        # )
        # logger.info(f"Output of res: {res}")
        # res_string = str(res)
        # logger.info(f"Data from array: {res.results}")
        # logger.info(f"First data: {res.results[0]['data']}")
        # logger.info(f"Length of data: {len(res.results[0]['data'][0])}")
        if res is not None and res.results[0].get("category_id") is not None:
        # if result is not None:
            # populate detection result with corresponding inference result information
            self.not_none_counter += 1
            i = 0

            for result in res.results:
                if i > 20:
                    break

                detections[i] = [
                    result["category_id"],
                    float(result["score"]),
                    result["bbox"][1] / self.model_height,
                    result["bbox"][0] / self.model_width,
                    result["bbox"][3] / self.model_height,
                    result["bbox"][2] / self.model_width,
                ]
                i += 1

            # for item in result.results:
            #     # logger.info(f"CURRENT ITEM: {item}")
            #     if (i >= 20):
            #         break

            #     category_id = int(item[5])
            #     score = item[4]
            #     y_min = item[1]
            #     x_min = item[0]
            #     x_max = item[2]
            #     y_max = item[3]
            #     detections[i] = [category_id, score, y_min, x_min, y_max, x_max]
            #     i += 1

        # if (detections[0][1] != 0): # if we have a score, then print detection
        #     logger.info(f"Output of detections: {detections}")
        ## Save the detection results to a file so we can compare
        with open("degirum_local_hailo_detections.txt", "a") as f:
            f.write(f"{detections}")
        # logger.info(f"Overall time took: {(time.time_ns() - start) * 1e-6}ms")
        return detections
