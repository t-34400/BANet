#!/usr/bin/env python3
import argparse, copy, os, sys, torch

sys.path.append("core")
from core.BANet import BANet
from core.submodule import disparity_regression, context_upsample

def build_model(args, device):
    m = BANet(args)
    try:
        sd = torch.load(args.restore_ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(args.restore_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=False)
    m.eval().to(device)
    return m

def _normalize_feature_attrs(mod):
    # Map L_/R_-prefixed attributes back to default names if present
    for name, child in list(mod._modules.items()):
        if (name.startswith("L_") or name.startswith("R_")) and name[2:] not in mod._modules:
            mod._modules[name[2:]] = child
    for name in list(mod.__dict__.keys()):
        if (name.startswith("L_") or name.startswith("R_")) and not hasattr(mod, name[2:]):
            setattr(mod, name[2:], getattr(mod, name))

def split_fnet_for_export(m: BANet, max_disp: int) -> BANet:
    m.fnet_L = m.fnet
    m.fnet_R = copy.deepcopy(m.fnet)
    delattr(m, "fnet")
    _normalize_feature_attrs(m.fnet_L)
    _normalize_feature_attrs(m.fnet_R)
    m._export_max_disp = int(max_disp)

    def _forward_export(self, left, right):
        import torch.nn.functional as F
        features_left  = self.fnet_L(left)
        features_right = self.fnet_R(right)

        stem_2x = self.stem_2(left);  stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right); stem_4y = self.stem_4(stem_2y)

        features_left[0]  = torch.cat((features_left[0],  stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        max_disp = self._export_max_disp
        corr = self.build_cv(features_left[0], features_right[0], max_disp // 4)
        cv = self.cost_stem(corr)

        spa_att = self.spa_att(features_left)
        cv_0 = self.cost_agg0(spa_att * cv, features_left)
        cv_1 = self.cost_agg1((1.0 - spa_att) * cv, features_left)
        cv = spa_att * cv_0 + (1.0 - spa_att) * cv_1

        prob = torch.softmax(cv, dim=1)
        disp = disparity_regression(prob, max_disp // 4)

        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = torch.softmax(self.spx(xspx), 1)

        disp_up = context_upsample(disp, spx_pred)
        return disp_up * 4.0

    m.forward = _forward_export.__get__(m, BANet)
    return m

def export_pte(model, left, right, out_path, backend):
    from executorch.exir import to_edge_transform_and_lower
    partitioner = None
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        part = XnnpackPartitioner()
    elif backend == "vulkan":
        from executorch.backends.vulkan.partition.vulkan_partitioner import VulkanPartitioner
        part = VulkanPartitioner()
    elif backend == "portable":
        partitioner = None
    else:
        raise RuntimeError(f"Unknown backend: {backend}")

    ep = torch.export.export(model, (left, right))
    try:
        et_prog = to_edge_transform_and_lower(ep, partitioner=partitioner).to_executorch()
    except Exception as e:
        if backend != "portable":
            print(f"[WARN] Backend pre/lowering failed for '{backend}': {e}")
            print("[WARN] Falling back to 'portable'.")
            et_prog = to_edge_transform_and_lower(ep, partitioner=None).to_executorch()
        else:
            raise
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    et_prog.save(out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--restore_ckpt", required=True)
    p.add_argument("--pte_out", default=".local/output/banet2d.pte")
    p.add_argument("--backend", choices=["xnnpack", "vulkan", "portable"], default="xnnpack")
    p.add_argument("--max_disp", type=int, default=192)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    args = p.parse_args()

    if args.height % 32 != 0 or args.width % 32 != 0:
        raise RuntimeError("height and width must be divisible by 32.")
    if args.max_disp % 4 != 0:
        raise RuntimeError("max_disp must be divisible by 4.")

    device = "cpu"
    model = build_model(args, device)
    model = split_fnet_for_export(model, args.max_disp)

    b, c, h, w = 1, 3, args.height, args.width
    left  = torch.randn(b, c, h, w, device=device, dtype=torch.float32)
    right = torch.randn(b, c, h, w, device=device, dtype=torch.float32)

    try:
        export_pte(model, left, right, args.pte_out, args.backend)
        print(f"[INFO] Exported PTE -> {args.pte_out} (backend={args.backend})")
    except Exception as e:
        print(f"[ERROR] PTE export failed: {e}")
        raise
