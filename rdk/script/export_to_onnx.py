from ultralytics import YOLO
import onnx

model = YOLO("model/best.pt")

model.export(
    format="onnx",
    imgsz=640,
    opset=11,
    simplify=False,
    dynamic=False,
)

# Patch IR version
m = onnx.load("model/best.onnx")
m.ir_version = 8
onnx.save(m, "model/best_nosimplify.onnx")
print("IR:", m.ir_version)
print("Opset:", m.opset_import[0].version)