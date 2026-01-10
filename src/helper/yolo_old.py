COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]


def letterbox(img, new_shape=INPUT_SIZE, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = new_shape - nh, new_shape - nw
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, (pad_w // 2, pad_h // 2)


def nms(boxes, scores, iou_threshold):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / (union + 1e-6)


class YoloDetector:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.out_names = [o.name for o in self.session.get_outputs()]

    def infer(self, frame: np.ndarray, display=False):
        img, r, (pad_w, pad_h) = letterbox(frame, INPUT_SIZE)
        img_input = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0  # BGR->RGB, HWC->CHW
        img_input = np.expand_dims(img_input, 0)

        outputs = self.session.run(None, {self.input_name: img_input})
        out = outputs[0]
        # Normalize output to YOLOv8 format: (N, 84) where [cx,cy,w,h, 80 class scores]
        # Common ONNX exports: (1,84,8400) or (1,8400,84)
        if out.ndim == 3:
            if out.shape[1] == 84:      # (1, 84, 8400)
                out = out[0].T          # -> (8400, 84)
            elif out.shape[2] == 84:    # (1, 8400, 84)
                out = out[0]            # -> (8400, 84)
            else:
                out = out.reshape(-1, out.shape[-1])

        # YOLOv8/v11: [x,y,w,h, class_scores...] (no objectness). Some exports emit logits; apply sigmoid fallback.
        boxes = []
        scores = []
        labels = []
        
        if out.size == 0:
            if display:
                cv2.imshow("YOLO Detection", frame)
                cv2.waitKey(1)
            return []

        # Ensure float32
        out = out.astype(np.float32, copy=False)

        # Split boxes and class scores
        bx = out[:, :4]
        cls = out[:, 4:]

        # Sigmoid fallback if values look like logits (outside [0,1])
        if cls.max() > 1.0 or cls.min() < 0.0:
            cls = 1.0 / (1.0 + np.exp(-cls))

        # For each prediction, take top class
        cls_ids = np.argmax(cls, axis=1)
        cls_confs = cls[np.arange(cls.shape[0]), cls_ids]

        # Filter by confidence
        keep_mask = cls_confs >= float(CONF_THRESH)
        bx = bx[keep_mask]
        cls_ids = cls_ids[keep_mask]
        cls_confs = cls_confs[keep_mask]

        # Convert cx,cy,w,h to x1,y1,x2,y2 and map back to original image coords
        if bx.size:
            cx = bx[:, 0]
            cy = bx[:, 1]
            w = bx[:, 2]
            h = bx[:, 3]
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            x1 = (x1 - pad_w) / r
            y1 = (y1 - pad_h) / r
            x2 = (x2 - pad_w) / r
            y2 = (y2 - pad_h) / r
            boxes = np.stack([x1, y1, x2, y2], axis=1)
            scores = cls_confs.astype(np.float32)
            labels = [COCO_NAMES[i] if i < len(COCO_NAMES) else str(i) for i in cls_ids]
        
        if len(boxes) == 0:
            if display:
                cv2.imshow("YOLO Detection", frame)
                cv2.waitKey(1)
            return []

        keep = nms(np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32), IOU_THRESH)
        results = []
        for i in keep:
            results.append({"label": labels[i], "conf": float(scores[i]), "xyxy": boxes[i].tolist()})
        
        # Draw on frame if display enabled
        if display:
            vis_frame = frame.copy()
            for res in results:
                x1, y1, x2, y2 = [int(v) for v in res["xyxy"]]
                label_text = f"{res['label']} {res['conf']:.2f}"
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_frame, label_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("YOLO Detection", vis_frame)
            cv2.waitKey(1)
        
        return results


def choose_target_label(detections, allowed: Optional[List[str]] = None) -> Optional[str]:
    if not detections:
        return None
    if allowed:
        detections = [d for d in detections if d["label"] in allowed]
        if not detections:
            return None
    # pick the highest confidence
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections[0]["label"]
