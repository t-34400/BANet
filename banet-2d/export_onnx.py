import argparse, torch
from core.BANet import BANet

def build_model(args, device):
    m = BANet(args)
    # Prefer safe loading; fall back for older PyTorch
    try:
        sd = torch.load(args.restore_ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(args.restore_ckpt, map_location="cpu")
    if "state_dict" in sd: sd = sd["state_dict"]
    m.load_state_dict(sd, strict=False)
    m.eval().to(device)
    return m

def clean_onnx_inputs(path):
    import onnx
    from onnx import helper, TensorProto

    m = onnx.load(path)
    init_names = {i.name for i in m.graph.initializer}

    div2 = "onnx::Div_2"
    if any(vi.name == div2 for vi in m.graph.input) and div2 not in init_names:
        m.graph.initializer.append(
            helper.make_tensor(div2, TensorProto.INT64, (), [2])
        )
        init_names.add(div2)

    kept = [vi for vi in m.graph.input if vi.name not in init_names]
    try:
        m.graph.ClearField("input")
    except Exception:
        while len(m.graph.input):
            m.graph.input.pop()
    m.graph.input.extend(kept)

    onnx.checker.check_model(m)
    onnx.save(m, path)
    return [vi.name for vi in m.graph.input]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--restore_ckpt", required=True)
    p.add_argument("--onnx_out", default="banet2d.onnx")
    p.add_argument("--max_disp", type=int, default=192)
    p.add_argument("--height", type=int, default=960)   # must be divisible by 32
    p.add_argument("--width", type=int, default=1280)   # must be divisible by 32
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args, device)

    b, c, h, w = 1, 3, args.height, args.width
    left  = torch.randn(b, c, h, w, device=device, dtype=torch.float32)
    right = torch.randn(b, c, h, w, device=device, dtype=torch.float32)

    dynamic_axes = {
        "left":  {0: "batch", 2: "height", 3: "width"},
        "right": {0: "batch", 2: "height", 3: "width"},
        "disp":  {0: "batch", 2: "height", 3: "width"},
    }

    torch.onnx.export(
        model,
        (left, right),
        args.onnx_out,
        input_names=["left", "right"],
        output_names=["disp"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        keep_initializers_as_inputs=False
    )
    print(f"[INFO] Exported ONNX -> {args.onnx_out}")

    inputs_after = clean_onnx_inputs(args.onnx_out)
    print(f"[INFO] Cleaned ONNX inputs -> {inputs_after}")
