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
        # logger.info(f"Models in zoo: {self._zoo.list_models()}")
        self.dg_model = self._zoo.load_model(
            detector_config.model.path,
        )
        self.dg_model.measure_time = True
        # self.dg_model._preprocessor = EmptyPreproc()
        self.dg_model.input_image_format = "RAW"
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
        self.overall_frame_counter += 1
        truncated_input = tensor_input.reshape(tensor_input.shape[1:])
        # logger.debug(f"Detect raw was called for tensor input: {tensor_input}")
        # logger.info(f"Tensor shape: {tensor_input.shape}")
        # try:
        #     cv2.imwrite(
        #         f"/media/frigate/tensor_images/Tensor__{self.overall_frame_counter}.png",
        #         truncated_input,
        #     )
        # except Exception as e:
        #     logger.info(f"We had an error when trying to write image tensor: {e}")
        # add tensor_input to input queue
        self._queue.put(truncated_input)
        # logger.debug(f"Queue size after adding truncated input: {self._queue.qsize()}")

        # define empty detection result
        detections = np.zeros((20, 6), np.float32)
        # logger.debug("We are about to hit the res initial call ! ! ! !")
        res = next(self.prediction)
        # logger.debug(f"Queue size after calling for res: {self._queue.qsize()}")
        # logger.debug(f"Output of res in initial next call: {res}")
        # logger.info(
        # f"Overall frame number: {self.overall_frame_counter}, none count: {self.none_counter}, not none count: {self.not_none_counter}, none percentage: {self.none_counter / self.overall_frame_counter}"
        # )
        # logger.info(f"Time stats right after res: {self.dg_model.time_stats()}")
        # res_string = str(res)
        # logger.debug(f"Res is: {res_string}")
        # logger.debug(f"Res's list of attributes: {dir(res)}")
        # logger.debug(
        #     f"Res results, {res.results}, length of results: {len(res.results)}"
        # )
        # logger.info(f"Output of res: {res}")
        # res_string = str(res)
        if res is not None and res.results[0].get("category_id") is not None:
            # populate detection result with corresponding inference result information
            self.not_none_counter += 1
            i = 0

            # data = res.results[0]["data"]
            detections = np.zeros((20, 6), dtype=float)

            for result in res.results:
                if i > 20:
                    break

                detections[i] = [
                    result["category_id"],
                    float(result["score"]),
                    result["bbox"][1],
                    result["bbox"][0],
                    result["bbox"][3],
                    result["bbox"][2],
                ]

                i += 1

            # # Iterate over the 100 elements in the array
            # for i in range(data.shape[2]):  # data.shape[2] is 100
            #     if i > 19:
            #         break
            #     category_id = int(
            #         data[0, 0, i, 0]
            #     )  # Assuming 'category_id' is at index 0
            #     score = float(data[0, 0, i, 1])  # Assuming 'score' is at index 1
            #     y_min = float(data[0, 0, i, 2])  # Assuming 'y_min' is at index 2
            #     x_min = float(data[0, 0, i, 3])  # Assuming 'x_min' is at index 3
            #     y_max = float(data[0, 0, i, 4])  # Assuming 'y_max' is at index 4
            #     x_max = float(data[0, 0, i, 5])  # Assuming 'x_max' is at index 5

            #     detections[i] = [category_id, score, y_min, x_min, y_max, x_max]
            #     i += 1

        return detections
