
class PostProcessor:
    def __init__(self, json_config):
        self._num_classes = 80
        self._input_height = 640
        self._input_width = 640
        self._output_conf_threshold = 0.5
        self._label_dictionary = {
            "0": "person",
            "1": "bicycle",
            "2": "car",
            "3": "motorcycle",
            "4": "airplane",
            "5": "bus",
            "6": "train",
            "7": "truck",
            "8": "boat",
            "9": "traffic light",
            "10": "fire hydrant",
            "11": "stop sign",
            "12": "parking meter",
            "13": "bench",
            "14": "bird",
            "15": "cat",
            "16": "dog",
            "17": "horse",
            "18": "sheep",
            "19": "cow",
            "20": "elephant",
            "21": "bear",
            "22": "zebra",
            "23": "giraffe",
            "24": "backpack",
            "25": "umbrella",
            "26": "handbag",
            "27": "tie",
            "28": "suitcase",
            "29": "frisbee",
            "30": "skis",
            "31": "snowboard",
            "32": "sports ball",
            "33": "kite",
            "34": "baseball bat",
            "35": "baseball glove",
            "36": "skateboard",
            "37": "surfboard",
            "38": "tennis racket",
            "39": "bottle",
            "40": "wine glass",
            "41": "cup",
            "42": "fork",
            "43": "knife",
            "44": "spoon",
            "45": "bowl",
            "46": "banana",
            "47": "apple",
            "48": "sandwich",
            "49": "orange",
            "50": "broccoli",
            "51": "carrot",
            "52": "hot dog",
            "53": "pizza",
            "54": "donut",
            "55": "cake",
            "56": "chair",
            "57": "couch",
            "58": "potted plant",
            "59": "bed",
            "60": "dining table",
            "61": "toilet",
            "62": "tv",
            "63": "laptop",
            "64": "mouse",
            "65": "remote",
            "66": "keyboard",
            "67": "cell phone",
            "68": "microwave",
            "69": "oven",
            "70": "toaster",
            "71": "sink",
            "72": "refrigerator",
            "73": "book",
            "74": "clock",
            "75": "vase",
            "76": "scissors",
            "77": "teddy bear",
            "78": "hair drier",
            "79": "toothbrush"
        }



    def forward(self, tensor_list, details_list=None):
        """
        Process the raw output tensor to produce formatted JSON results.

        Parameters:
            tensor_list (list): List of tensors from the model.
            details_list (list): Additional details (unused in this example).

        Returns:
            str: JSON-formatted string containing detection results.
        """
        # Initialize results list
        new_inference_results = []

        # Extract and reshape the raw output tensor
        output_array = tensor_list[0].reshape(-1)

        # Index to parse the array
        index = 0

        # Iterate over classes and parse results
        for class_id in range(self._num_classes):
            # Number of detections for this class
            num_detections = int(output_array[index])
            # logger.info(f"For class_id: {class_id} we have {num_detections} results")
            index += 1  # Move to the next entry

            # Skip if no detections for this class
            if num_detections == 0:
                continue

            # Process each detection for this class
            for _ in range(num_detections):
                if index + 5 > len(output_array):
                    # Safeguard against unexpected array end
                    break

                # Extract score and bounding box in x_center, y_center, width, height format
                score = float(output_array[index + 4])
                y_min, x_min, y_max, x_max = map(float, output_array[index : index + 4])
                index += 5  # Move to the next detection

                # Skip detections below the confidence threshold
                if score < self._output_conf_threshold:
                    continue

                # Convert to x_min, y_min, x_max, y_max format
                x_min = x_min * self._input_width
                y_min = y_min * self._input_height
                x_max = x_max * self._input_width
                y_max = y_max * self._input_height

                # Create a detection result with bbox, score, and class label
                result = {
                    "bbox": [x_min, y_min, x_max, y_max],  # Bounding box in pixel coordinates
                    "score": score,  # Confidence score of the detection
                    "category_id": class_id,  # Class ID of the detected object
                    "label": self._label_dictionary.get(str(class_id), f"class_{class_id}"),  # Class label or fallback
                }
                new_inference_results.append(result)  # Store the formatted detection


            # Stop processing if padded zeros are reached
            slice_end = index + self._num_classes + 2
            arr_slice = output_array[index: slice_end] if slice_end < len(output_array) else output_array[index:]
            if index >= len(output_array) or all(v == 0 for v in arr_slice):
                break

        # Return results as array
        return new_inference_results
