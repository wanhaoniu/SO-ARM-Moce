import argparse

import cv2
from ultralytics import YOLO


APPLE_CLASS_ID = 47
AUTO_MODEL_CANDIDATES = ("yolo26n.pt", "yolo11n.pt", "yolov8n.pt")


def load_model(model_name: str):
    if model_name != "auto":
        return YOLO(model_name), model_name

    last_err = None
    for name in AUTO_MODEL_CANDIDATES:
        try:
            model = YOLO(name)
            return model, name
        except Exception as e:
            print(f"[WARN] 加载 {name} 失败: {e}")
            last_err = e

    raise RuntimeError(f"无法加载任何候选模型 {AUTO_MODEL_CANDIDATES}: {last_err}")


def main():
    parser = argparse.ArgumentParser(description="Apple detector with Ultralytics YOLO.")
    parser.add_argument("--model", default="auto", help="模型名或权重路径，如 auto/yolo26n.pt/yolo11n.pt/yolov8n.pt")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV 相机索引")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    args = parser.parse_args()

    model, model_name = load_model(args.model)
    print(f"[INFO] 使用模型: {model_name}")

    cap = cv2.VideoCapture(args.camera_index)
    win = "Apple Detection (Ultralytics)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)   # 关键：可调整大小
    cv2.resizeWindow(win, 1280, 720)          # 设定初始窗口大小

    if not cap.isOpened():
        print("错误：无法打开相机。")
        return

    print("正在启动识别，按 'q' 退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 仅检测 COCO 的 apple 类（id=47）
        results = model.predict(
            source=frame,
            conf=args.conf,
            classes=[APPLE_CLASS_ID],
            imgsz=args.imgsz,
            verbose=False,
        )

        annotated_frame = results[0].plot() if results else frame
        cv2.putText(
            annotated_frame,
            f"model: {model_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Apple Detection (Ultralytics)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
