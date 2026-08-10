# dump_brepmfr_test.py
# S6 推理适配器(BrepMFR)——本文件为本方新增脚本, 官方仓库零改动。
# v1.0(2026-08-09): 复刻官方 brepseg_model.py test_step 前向(brep_encoder→attention→classifier→argmax)
#   + BrepSeg.load_from_checkpoint 官方权重路径, 模型数学 100% 官方;
#   两处刻意偏离(均见协议 §26):
#   [1] DataLoader(shuffle=False, drop_last=False) 全量 8922 —— 官方 get_dataloader 硬编码
#       drop_last=True 丢尾批 26 样本(发现 10);
#   [2] dump 路径改为统一 _s6_preds\brepmfr\ —— 官方硬编码 /home/zhang/...(发现 11);
#   dump 文件名保持官方兼容 feature_<id>.txt(id=bin 文件名末段数字, dataset.py:83);
#   另附 feature_<id>.gt(标签直读, 供 S7 四家标签一致性交叉检查)。
#   best ckpt 自动解析: best-vN 后缀只是去重计数器不是名次, 真 best 读 last.ckpt
#   回调状态里的 best_model_path(PL ModelCheckpoint 官方存档)。
# v1.1(2026-08-09): brepmfr 环境 PyTorch>=2.6 实证 —— torch.load 默认 weights_only=True,
#   官方 ckpt 的 hyperparameters 内含 argparse.Namespace(save_hyperparameters 捕获), 被安全拦截
#   (UnpicklingError, infer_brepmfr_20260809_124301.log)。处置: 白名单放行 argparse.Namespace
#   (自家训练产物, 来源可信), 同时治愈 resolve_best_ckpt 与 BrepSeg.load_from_checkpoint 两处加载。
import argparse
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import CADSynth
from models.brepseg_model import BrepSeg

torch.serialization.add_safe_globals([argparse.Namespace])


def resolve_best_ckpt(run_dir):
    """从 run 目录 last.ckpt 的 ModelCheckpoint 回调状态读取真 best 路径与分数。"""
    last_path = pathlib.Path(run_dir) / "last.ckpt"
    c = torch.load(last_path, map_location="cpu")
    for k, v in c["callbacks"].items():
        if k.startswith("ModelCheckpoint"):
            return v["best_model_path"], v.get("best_model_score")
    raise RuntimeError("last.ckpt 中未找到 ModelCheckpoint 回调状态")


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main():
    parser = argparse.ArgumentParser(description="S6: dump BrepMFR per-face predictions on full test split")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--run_dir", required=True, help="训练 run 目录(含 last.ckpt), 用于自动解析 best ckpt")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path, best_score = resolve_best_ckpt(args.run_dir)
    print(f"[S6] best ckpt = {best_path}")
    print(f"[S6] best eval_loss(score) = {best_score}")
    if not pathlib.Path(best_path).exists():
        raise FileNotFoundError(f"best ckpt 不存在: {best_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[S6] device = {device}")

    # 官方权重加载路径(超参数随 ckpt save_hyperparameters 恢复)
    model = BrepSeg.load_from_checkpoint(best_path, map_location=device)
    model.eval()
    model.to(device)
    num_classes = model.num_classes
    print(f"[S6] num_classes = {num_classes}")

    test_data = CADSynth(root_dir=args.dataset_path, split="test", random_rotate=False, num_class=num_classes)
    print(f"[S6] test samples loaded = {len(test_data)}")

    # 偏离点[1]: 全量评测(不丢尾批), collate 仍用官方 _collate
    loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=test_data._collate,
        drop_last=False,
    )

    n_samples = 0
    n_faces = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            # ===== 官方 test_step 前向逐字复刻(brepseg_model.py:201-216) =====
            node_emb, graph_emb = model.brep_encoder(batch, last_state_only=True)
            node_emb = node_emb[0].permute(1, 0, 2)     # [batch, max_node_num+1, dim] -> 去全局节点
            node_emb = node_emb[:, 1:, :]
            padding_mask = batch["padding_mask"]
            node_pos = torch.where(padding_mask == False)
            node_z = node_emb[node_pos]
            padding_mask_ = ~padding_mask
            num_nodes_per_graph = torch.sum(padding_mask_.long(), dim=-1)
            graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
            z = model.attention([node_z, graph_z])
            node_seg = model.classifier(z)              # [total_nodes, num_classes]
            preds = torch.argmax(node_seg, dim=-1)      # [total_nodes]
            labels = batch["label_feature"].long()
            # ===== 复刻结束; 以下为 dump(偏离点[2]) =====

            counts = num_nodes_per_graph.cpu().tolist()
            ids = batch["id"].long().cpu().tolist()
            offset = 0
            for sid, n in zip(ids, counts):
                p = preds[offset:offset + n].cpu().tolist()
                g = labels[offset:offset + n].cpu().tolist()
                offset += n
                with open(out_dir / f"feature_{sid}.txt", "w") as f:
                    f.write("\n".join(str(int(v)) for v in p) + "\n")
                with open(out_dir / f"feature_{sid}.gt", "w") as f:
                    f.write("\n".join(str(int(v)) for v in g) + "\n")
                n_samples += 1
                n_faces += n

    print(f"[S6] BrepMFR dump done: samples={n_samples}, faces={n_faces}, out={out_dir}")


if __name__ == "__main__":
    main()
