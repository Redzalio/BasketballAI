"""Quick structural check of the mobile ONNX after export (input/output shape, opset)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.expanduser("~"),
                                "AppData", "Roaming", "Python", "Python314", "site-packages"))
import onnx

p = os.path.join(os.path.dirname(__file__), "..", "mobile", "www", "models", "detector.onnx")
p = os.path.abspath(p)
m = onnx.load(p)
onnx.checker.check_model(m)


def shp(t):
    return [d.dim_value for d in t.type.tensor_type.shape.dim]


print("VALID ONNX:", os.path.getsize(p) // 1024, "KB")
for i in m.graph.input:
    print("  input ", i.name, shp(i))
for o in m.graph.output:
    print("  output", o.name, shp(o))
print("opset:", m.opset_import[0].version)
