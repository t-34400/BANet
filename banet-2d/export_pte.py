#!/usr/bin/env python3
import argparse, copy, os, sys
import torch
import torch.nn as nn
from torch.nn.utils import fuse_conv_bn_eval

sys.path.append("core")
from core.BANet import BANet
from core.submodule import disparity_regression, context_upsample

def build_model(args, device):
    m = nn.DataParallel(BANet(args))
    sd = torch.load(args.restore_ckpt)
    m.load_state_dict(sd, strict=True)
    m = m.module
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

def fuse_deconv_bn_eval(deconv: nn.ConvTranspose2d, bn: nn.BatchNorm2d) -> nn.ConvTranspose2d:
    if bn.training:
        raise RuntimeError("BatchNorm must be in eval() mode before folding.")
    if bn.running_mean is None or bn.running_var is None:
        raise RuntimeError("BatchNorm running stats are missing.")

    with torch.no_grad():
        # scale = gamma / sqrt(var + eps), shift handled in bias update
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)  # [Cout]
        Cout, Cin = deconv.out_channels, deconv.in_channels
        g = deconv.groups
        ocpg = Cout // g
        icpg = Cin // g

        # Ensure bias exists
        if deconv.bias is None:
            deconv.bias = nn.Parameter(torch.zeros(Cout, dtype=deconv.weight.dtype, device=deconv.weight.device))

        # W shape: (Cin, Cout/groups, kH, kW). Scale along Cout dimension.
        s = scale.view(g, ocpg)                                  # [g, ocpg]
        s = s.repeat_interleave(icpg, dim=0).view(Cin, ocpg)     # [Cin, ocpg]
        deconv.weight.mul_(s[:, :, None, None])

        # b' = (b - mean) * scale + beta
        deconv.bias.copy_((deconv.bias - bn.running_mean) * scale + bn.bias)

    return deconv

def fuse_all_conv_bn(module: nn.Module):
    # Recurse first
    for name, child in list(module.named_children()):
        fuse_all_conv_bn(child)

        # Case 1: Sequential pipeline
        if isinstance(child, nn.Sequential):
            fused_layers, i = [], 0
            L = len(child)
            while i < L:
                a = child[i]
                b = child[i + 1] if i + 1 < L else None
                if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                    fused_layers.append(fuse_conv_bn_eval(a.eval(), b.eval()))
                    i += 2
                elif isinstance(a, nn.ConvTranspose2d) and isinstance(b, nn.BatchNorm2d):
                    fused_layers.append(fuse_deconv_bn_eval(a.eval(), b.eval()))
                    i += 2
                else:
                    fused_layers.append(a)
                    i += 1
            setattr(module, name, nn.Sequential(*fused_layers))
            continue

        # Case 2: Attribute pattern: self.conv / self.bn
        conv = getattr(child, "conv", None)
        bn   = getattr(child, "bn", None)
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            setattr(child, "conv", fuse_conv_bn_eval(conv.eval(), bn.eval()))
            setattr(child, "bn", nn.Identity())
        elif isinstance(conv, nn.ConvTranspose2d) and isinstance(bn, nn.BatchNorm2d):
            setattr(child, "conv", fuse_deconv_bn_eval(conv.eval(), bn.eval()))
            setattr(child, "bn", nn.Identity())

def export_pte(model, left, right, out_path, backend):
    from executorch.exir import to_edge_transform_and_lower
    partitioner = None
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        partitioner = [XnnpackPartitioner()]
    elif backend == "vulkan":
        from executorch.backends.vulkan.partition.vulkan_partitioner import VulkanPartitioner
        partitioner = [VulkanPartitioner()]
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
            backend=None
            et_prog = to_edge_transform_and_lower(ep, partitioner=None).to_executorch()
        else:
            raise e
        
    for method in et_prog.methods:
        print(f"[Info] Method: {method}")
        
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    et_prog.save(out_path)

    return backend

def export_ops_yml(model_path, ops_yml_path):
    from executorch.codegen.tools.gen_oplist import gen_oplist
    gen_oplist(
        output_path=ops_yml_path,
        model_file_path=model_path
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--restore_ckpt", required=True)
    p.add_argument("--pte_out", default=".local/output/banet2d_xnn.pte")
    p.add_argument("--ops_yml_out", default=".local/output/banet2d_xnn_ops.yml")
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
    fuse_all_conv_bn(model)
    model = split_fnet_for_export(model, args.max_disp)

    b, c, h, w = 1, 3, args.height, args.width
    left  = torch.randn(b, c, h, w, device=device, dtype=torch.float32)
    right = torch.randn(b, c, h, w, device=device, dtype=torch.float32)

    try:
        output_backend = export_pte(model, left, right, args.pte_out, args.backend)
        print(f"[INFO] Exported PTE -> {args.pte_out} (backend={output_backend})")
    except Exception as e:
        print(f"[ERROR] PTE export failed: {e}")
        raise

    export_ops_yml(args.pte_out, args.ops_yml_out)