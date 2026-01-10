import cv2
import numpy as np
import onnxruntime as ort

def letterbox(img, new_shape, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = new_shape - nh, new_shape - nw
    resized = cv2.resize(img, (nw, nh))
    padded = cv2.copyMakeBorder(
        resized,
        pad_h // 2, pad_h - pad_h // 2,
        pad_w // 2, pad_w - pad_w // 2,
        cv2.BORDER_CONSTANT,
        value=color
    )
    return padded, r, (pad_w // 2, pad_h // 2)


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)


def nms(boxes, scores, iou_thresh):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        idxs = idxs[1:][iou(boxes[i], boxes[idxs[1:]]) < iou_thresh]
    return keep


# ==================== YOLO DETECTOR =================
class YoloDetector:
    """
    Correct decoder for YOLOv8 PERSON-ONLY ONNX
    Output shape: (1,5,8400)
    """
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, config, frame, display=False):
        img, r, (pad_w, pad_h) = letterbox(frame, config.get("INPUT_SIZE", 640))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)

        output = self.session.run(None, {self.input_name: img})[0]
        preds = output[0].T.astype(np.float32)  # (8400,5)

        boxes = preds[:, :4]
        scores = preds[:, 4]

        mask = scores >= config.get("CONF_THRESH")
        boxes = boxes[mask]
        scores = scores[mask]

        if boxes.size == 0:
            return []

        cx, cy, w, h = boxes.T
        x1 = (cx - w / 2 - pad_w) / r
        y1 = (cy - h / 2 - pad_h) / r
        x2 = (cx + w / 2 - pad_w) / r
        y2 = (cy + h / 2 - pad_h) / r

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = nms(boxes_xyxy, scores, config.get("IOU_THRESH"))

        results = []
        for i in keep:
            results.append({
                "label": "person",
                "conf": float(scores[i]),
                "xyxy": boxes_xyxy[i].tolist()
            })

        if display:
            vis = frame.copy()
            for r in results:
                x1, y1, x2, y2 = map(int, r["xyxy"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis, f"person {r['conf']:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
            cv2.imshow("Detection", vis)
            cv2.waitKey(1)

        return results


