# NinaPro 原始数据与泊松脉冲编码对比

从测试集固定抽取 6 个样本，比较未归一化原始 sEMG、加 128 泊松编码和正负极性分离泊松编码。主对比使用 T=40；时间分辨率图对同一样本比较 T=40、80、120。

原始波形在图中仅为避免 16 条曲线重叠而统一缩放，编码过程直接读取未归一化原始值。

| 测试索引 | 标签 | 动作 | 原始平均绝对幅值 | 加128 T40密度 | 正负分离 T40密度 |
|---:|---:|---|---:|---:|---:|
| 782 | 0 | 食指屈曲 | 13.92 | 19.53% | 13.98% |
| 7033 | 1 | 食指伸展 | 16.62 | 22.50% | 15.62% |
| 5522 | 4 | 无名指屈曲 | 19.65 | 23.28% | 17.81% |
| 3806 | 5 | 无名指伸展 | 8.41 | 24.38% | 11.09% |
| 3190 | 8 | 拇指内收 | 7.95 | 20.78% | 10.39% |
| 8572 | 9 | 拇指外展 | 4.71 | 21.41% | 8.59% |

## 文件

- `comparison_overview_T40.png`
- `sample_test_00782_label_00.png`
- `sample_test_07033_label_01.png`
- `sample_test_05522_label_04.png`
- `sample_test_03806_label_05.png`
- `sample_test_03190_label_08.png`
- `sample_test_08572_label_09.png`
- `time_resolution_comparison.png`
- `selection_manifest.csv`：便于表格查看的样本与脉冲统计。
- `selection_manifest.json`：包含编码参数的完整机器可读记录。
- `plot_poisson_spike_comparison.py`：可复现绘图脚本。
