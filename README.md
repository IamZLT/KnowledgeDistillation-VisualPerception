# Visual Grounding&知识蒸馏研究梳理

本仓库收集了Visual Grounding领域结合知识蒸馏技术的一些重要研究论文。论文附有官方链接或DOI，开源项目则会加入代码链接。

## 2025

- **Align-KD: Distilling Cross-Modal Alignment Knowledge for Mobile VLM** (CVPR 2025) 
  - [论文链接](https://cvpr.thecvf.com/virtual/2025/poster/33163)
  - 针对移动端VLM部署需求，提出在网络浅层对齐视觉和文本特征，让学生模型学会将视觉特征投影到文本语义空间，从7B大型VLM到1.7B移动VLM的知识迁移，提升下游任务性能

- **MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders** (CVPR 2025)
  - [论文链接](https://cvpr.thecvf.com/virtual/2025/poster/32553)
  - 将多个预训练视觉编码器的能力融合到单一模型中，使用低秩适配(LoRA)和专家混合(MoE)机制，对不同输入自适应激活各编码器的专业知识，采用注意力加权的蒸馏策略

- **VL2Lite: Task-Specific Knowledge Distillation from Large VLMs to Lightweight Networks** (CVPR 2025)
  - [论文链接](https://cvpr.thecvf.com/virtual/2025/poster/33217)
  - 提出针对任务的蒸馏框架，将大型视觉语言模型的多模态知识传递给小型网络，直接在训练过程中融合VLM产生的高级视觉-语言特征，让轻量模型学习到丰富的语义表示

- **COSMOS: Cross-Modality Self-Distillation for Vision-Language Pre-training** (CVPR 2025)
  - [论文链接](https://arxiv.org/abs/2412.01814)
  - 引入文本裁剪和跨注意力模块，将图像和文本分别生成全局与局部视图进行多模态自监督学习，通过跨模态自蒸馏损失，使模型学习更加全面的表示

- **DenseGrounding: Improving Dense Language-Vision Semantics for Ego-centric 3D Visual Grounding** (ICLR 2025)
  - [论文链接](https://iclr.cc/virtual/2025/poster/28704)
  - 从视觉和文本两个方面增强语义表示：设计层级场景语义增强模块捕获细粒度全局场景特征，同时借助大语言模型丰富描述信息，在全量和小样本训练集上分别获得了显著的性能提升

## 2024

- **Visual Grounding with Dual Knowledge Distillation (DUET)** (IEEE TCSVT 2024)
  - DOI: [10.1109/TCSVT.2024.3407785](https://doi.org/10.1109/TCSVT.2024.3407785)
  - 提出双向蒸馏框架，通过同时对视觉和语言两路进行蒸馏，缩小跨模态特征差距，提高定位精度

- **SimVG: A Simple Framework for Visual Grounding with Decoupled Multi-modal Fusion** (NeurIPS 2024)
  - [论文链接](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dc6319dde4fb182b22fb902da9418566-Abstract-Conference.html)
  - 采用多分支结构，一支为轻量MLP分支，设计动态权重平衡蒸馏，在保证性能的同时大幅提升推理速度

## 2023

- **Distilling Coarse-to-Fine Semantic Matching Knowledge for Weakly Supervised 3D Visual Grounding** (ICCV 2023)
  - [论文链接](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Distilling_Coarse-to-Fine_Semantic_Matching_Knowledge_for_Weakly_Supervised_3D_Visual_ICCV_2023_paper.pdf)
  - DOI: 10.1109/ICCV.2023.00039
  - 设计粗到细语义匹配模型，将语义匹配知识蒸馏到两阶段3D定位模型中，有效降低推理成本并提升弱监督3D定位性能

- **Weakly Supervised Referring Expression Grounding via Target-Guided Knowledge Distillation** (ICRA 2023)
  - [论文链接](https://colab.ws/articles/10.1109%2Ficra48891.2023.10161294)
  - DOI: 10.1109/ICRA48891.2023.10161294
  - 提出以目标预测结果为导向的蒸馏框架，再激活教师模型对于目标区域的知识，进而提升弱监督指称表达定位性能

- **Pseudo-Query Generation for Semi-Supervised Visual Grounding with Knowledge Distillation** (ICASSP 2023)
  - DOI: 10.1109/ICASSP49357.2023.10095558
  - 利用未标注图像生成伪查询，并结合知识蒸馏技术进行半监督训练，改善了弱标注情况下的视觉定位精度

- **Localized Symbolic Knowledge Distillation for Visual Commonsense Models** (NeurIPS 2023)
  - [论文链接](https://proceedings.neurips.cc/paper_files/paper/2023/hash/257be12f31dfa7cc158dda99822c6fd1-Abstract-Conference.html)
  - DOI: 10.5555/3603796.3603940
  - 构建可输入区域的多模态推理模型，通过从大型语言模型抽取区域级常识知识进行蒸馏，提高了视觉问答和指代推理模型的精度

- **Bridging Modality Gap for Visual Grounding with Effective Cross-modal Distillation** (arXiv 2023)
  - [论文链接](https://arxiv.org/html/2312.17648v2)
  - (EpmVG) 利用预训练多模态模型（如CLIP）蒸馏跨模态信息到视觉定位模型中，缓解图像与文本特征域差距，从而显著提升定位性能

## 2022

- **Look Around and Refer: 2D Synthetic Semantics Knowledge Distillation for 3D Visual Grounding** (NeurIPS 2022)
  - [论文链接](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f0b42291ddab77dcb2ef8a3488301b62-Abstract-Conference.html)
  - DOI: 10.48550/arXiv.2211.14241
  - 生成合成二维视图并蒸馏其知识到3D视觉流，提升了3D场景中以自然语言定位物体的效果和泛化能力

## 2021

- **Improving Weakly Supervised Visual Grounding by Contrastive Knowledge Distillation** (CVPR 2021)
  - [论文链接](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Improving_Weakly_Supervised_Visual_Grounding_by_Contrastive_Knowledge_Distillation_CVPR_2021_paper.pdf)
  - DOI: 10.1109/CVPR46437.2021.00074
  - 通过对象检测器提供的软标签进行对比学习蒸馏，无需检测器推理即可显著提升弱监督短语定位性能

