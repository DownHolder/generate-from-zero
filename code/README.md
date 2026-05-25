# 工程路线 — 代码实现

## 学习方式：对比式阅读

每个 stage 只包含**与上一阶段的差异**。学完 stage01 后，打开下一个 stage 目录，一眼就能看到变化了什么。

```
stage01_gan_basics/     ← 从这里开始（完整项目，学习整体流程）
    │
    ├── stage02_dcgan/   只改了 models.py  (MLP → Conv)
    ├── stage03_cgan/    只改了 models.py  (加入 Label Embedding)
    │
    ├── stage04_wgan_gp/ 改了 models.py + train.py  (Wasserstein loss)
    │
    └── stage05_vae/     新基础：Encoder + Decoder + ELBO
```

## 公共模块 (`code/common/`)

```python
common/
├── data.py      # get_dataloader — 数据加载
├── utils.py     # set_seed, get_device, make_noise — 工具函数
├── training.py  # save_checkpoint, save_gan_samples — 训练工具
└── gan.py       # train_bce_gan — BCE GAN 通用训练循环
```

## 如何运行

每个 stage 独立可运行：

```bash
cd code/stageXX_xxx
python main.py       # 训练
python generate.py   # 推理生成
```

## Stage 速查

| Stage | 文件数 | 相对上一阶段的差异 |
|-------|:------:|------------------|
| 01 GAN Basic | 7 | — 基准，完整项目结构 |
| 02 DCGAN | 4 | models: Conv G/D 替代 MLP |
| 03 cGAN | 4 | models: G/D 加入 Label Embedding |
| 04 WGAN-GP | 5 | models: Critic(无BN); train: Wasserstein + GP |
| 05 VAE | 5 | 全链路：Encoder+Decoder+ELBO loss |
