# Improved DETR with Conditional Cross-Attention for Faster Convergence and Better Localization

This project aims to improve the convergence speed and localization precision of DETR (DEtection TRansformer) for object detection tasks. The components covered in this repository include implementing conditional cross-attention in the decoder, training and evaluating both baseline and improved models on the PASCAL VOC dataset, and conducting ablation studies to validate the architectural improvements.

DETR eliminates hand-designed components in object detection by treating it as a set prediction problem, but it suffers from slow convergence (more than 300 epochs on COCO) and imprecise bounding boxes due to its decoder cross-attention design. This work implements conditional cross-attention into the original DETR codebase, replacing the additive content-position formulation with per-head concatenation of separate content and spatial streams. The improved model converges **1.7× faster** and achieves significantly tighter bounding boxes with an **AP75 improvement of +10.1**, while adding only 1.8M parameters (+4.5%) to the original DETR.

## Tech Stack Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-Attention%20Mechanism-blue?style=for-the-badge)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Object%20Detection-green?style=for-the-badge)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Methodology

1. **Problem Identification**  
   The baseline DETR framework has two key limitations, both originating from its decoder cross-attention mechanism. First, the content and positional embeddings are added together before the attention dot product, producing spurious cross-terms that carry no meaningful signals but still contribute to attention weights, forcing the model to learn and generalize noisy patterns. Second, because all attention heads receive the same added query and key, they tend to attend to similar broad regions rather than specializing in distinct spatial locations. These two issues together lead to slow convergence and imprecise bounding box localization.

2. **Conditional Cross-Attention Implementation**  
   Inspired by Conditional DETR (Meng et al. 2021), three changes are implemented in the original DETR decoder’s cross-attention. First, **Reference Point Prediction** where each object query predicts a 2D coordinate point (normalized to [0, 1]) representing the expected location of its target object, generated through a two-layer MLP network trained end-to-end. Second, **Conditional Spatial Query** where the reference point is converted to a 256-dimensional sine positional embedding and further scaled by the current decoder output in layers 2 to 6, allowing the spatial query to adapt based on what the model has already seen. Third, **Per-Head Concatenation** where content and spatial components are projected separately and concatenated based on the head dimension, allowing each of the 8 attention heads to receive its own independent spatial dimensions and specialize in different spatial regions of the object.

3. **Data Preparation**  
   Both baseline and improved DETR models are trained and evaluated on the PASCAL VOC image set, which contains 16,551 images (VOC 2007 train_val and VOC 2012 train_val) for training and 4,952 images (VOC 2007 test) for testing, across 20 object classes. The VOC annotations are converted to COCO format for compatibility with the original DETR's data pipeline. PASCAL VOC was chosen because it is small enough to train both models completely within a reasonable time and GPU budget, while still containing diverse scenes with a decent amount of objects and classes.

4. **Training Configuration**  
   Both models use identical training settings including 6 encoder/decoder layers, 256 hidden dimension, 8 attention heads, 50 object queries, AdamW optimizer, weight decay of 1e-4, and a batch size of 16. The only differences between the two models are the decoder attention mechanism and the learning rate drop schedule. The improved DETR uses a learning rate drop at epoch 40, while the baseline follows the original setting with a drop at epoch 120. This difference is intentional as the conditional spatial queries train faster during the initial high learning rate phase, enabling an earlier transition to learning rate reduction for refinement.

5. **Performance Analysis**  
   The improved DETR outperforms the baseline across all metrics at epoch 50, achieving an AP of 50.3 (vs 43.5), AP50 of 75.2 (vs 70.8), and AP75 of 54.8 (vs 44.7). The AP75 gain of +10.1 is significantly larger than the overall AP gain of +6.8, indicating that the improvements are most pronounced under strict localization thresholds. Loss component analysis further supports this, showing that the localization losses (BBox and GIoU) both improved by 13%, while classification loss improved by only 10%, confirming that the conditional cross-attention mainly enhances spatial precision.

6. **Convergence Analysis**  
   The improved model reaches AP=50 at epoch 48, while the baseline needs 83 epochs, showing a **1.7× convergence speedup**. Within a 75-epoch window, the improved model achieves a best AP of 52.4, compared to the baseline's best of 49.1. The convergence advantage is phase-dependent: during epochs 1 to 35, both models learn at similar rates because the conditional spatial queries have not yet matured. The advantage materializes after the improved model's learning rate drops at epoch 40, suggesting that the conditional spatial queries mature during the high learning rate phase and benefit from the subsequent reduction.

7. **Learning Rate Drop Analysis**  
   To validate that the convergence advantage is architectural rather than schedule-dependent, the single-epoch AP gain at epoch 39 to 40 was compared. The improved model gained +5.3 AP from the LR drop, while the baseline gained only +2.2 without the drop, a 2.4× larger gain. This suggests that the conditional spatial queries have learned meaningful reference points during the high learning rate phase and are ready for smaller refinement updates when the LR decreases, confirming that the earlier LR drop is enabled by the faster learning of the conditional spatial queries.

8. **Ablation Studies**  
   Two ablation studies were conducted to further investigate potential improvements. First, replacing the standard cross-entropy loss with **Focal Loss** in baseline DETR showed no observable benefit, indicating that the primary bottleneck in DETR's performance is not class imbalance but the convergence behavior of the attention mechanism. Second, **Mixup data augmentation** with a 50% chance and 0.5 blending ratio led to slower initial convergence and only marginal gains (AP: +1.24) even after 270+ epochs, indicating that Mixup provides limited benefit for DETR and that architectural improvements are more effective.

## Key Results

| Metric | Baseline DETR | Improved DETR | Improvement |
|--------|--------------|---------------|-------------|
| AP | 43.5 | 50.3 | +6.8 |
| AP50 | 70.8 | 75.2 | +4.3 |
| AP75 | 44.7 | 54.8 | **+10.1** |
| Epochs to reach AP=50 | 83 | 48 | **1.7× faster** |
| Parameters | 41.3M | 43.1M | +4.5% |

## Future Scope

Future work could explore training the improved model for more epochs to determine whether its performance ceiling can be raised, especially for small object detection where both models still struggle (AP_S below 6). Additionally, training and evaluating on larger-scale image sets such as COCO could further validate the generalizability of these improvements. Running the baseline under an identical LR schedule as the improved model would also provide more definitive proof that the convergence advantage is architectural in nature.

## Academic Context
Completed during my Graduate studies at **Nanyang Technological University (NTU), Singapore**.

* **Semester:** AY 2025/2026, Semester 1

## Usage
**Note for current/future NTU students:** While this repository is public, please ensure you adhere to NTU's Academic Integrity Policy. This is intended as a reference for my personal portfolio; using this code for your own graded assignments is strictly prohibited by the University.
