# Visual Grounding轻量化研究梳理

本仓库收集了Visual Grounding领域以及其结合模型轻量化技术、视觉大模型结合模型轻量化技术的一些研究成果。论文附有官方链接或DOI，开源项目则会加入代码链接。


## 目录

- [Visual Grounding 简介](#visual-grounding-简介)
- [视觉大模型结合模型轻量化技术相关研究成果](#视觉大模型结合模型轻量化技术相关研究成果)



## Visual Grounding 简介

本节多数内容参考自：[Towards Visual Grounding: A Survey](https://arxiv.org/pdf/2412.20206) | [项目主页](https://github.com/linhuixiao/Awesome-Visual-Grounding)

**视觉定位(Visual Grounding)，也称为指代表达文本理解(Referring Expression Comprehension, REC)和短语定位(Phrase Grounding, PG)**，是一项基础性的多模态人工智能任务，**旨在根据给定的文本描述在图像中定位特定区域**。视觉定位的目标是模拟社交对话中普遍存在的指代关系，赋予机器类人的多模态理解能力。因此，视觉定位在各个领域有着广泛的应用。然而，自 2021 年以来，视觉定位取得了重大进展，比如，基于定位的预训练、定位多模态大语言模型、广义视觉定位、多图片定位、千兆像素定位等新概念不断涌现，带来了许多新的挑战。这项任务涉及三种基本数据类型：图像、指代表达式和对应的边界框。

视觉定位的技术方法主要经历了几个阶段：

1. 传统CNN基础的方法
2. 基于Transformer的方法
3. 基于视觉语言预训练的迁移方法
4. 面向接地的预训练方法
5. 接地多模态大语言模型

视觉定位研究始于2014年左右。早期由于配对边界框数据的缺乏，大量研究主要集中在弱监督设置上。2014年，Kazemzadeh等人引入了第一个大规模真实世界表达理解数据集ReferIt Game；2016年，Mao等人提出并重组了基于MS COCO的RefCOCOg数据集，Yu等人同年提出了RefCOCO/+数据集，这三个数据集为后续研究奠定了坚实基础。

传统视觉定位主要根据表达形式和监督方式两个维度进行分类。

**从表达形式角度看**，视觉定位分为Referring Expression Comprehension和Phrase Grounding。**REC任务处理较长、复杂的文本描述，要求模型能够理解语言表达中的细微差别并准确定位对应区域**；而**PG则侧重于通过简短词语进行定位**，通常与ReferIt Game和Flickr30k Entities数据集相关联。这种分类主要基于表达文本的复杂程度，反映了不同任务对语言理解能力的要求。

**从监督方式维度考察，**视觉定位任务形成了一个完整的监督谱系。**全监督视觉定位**是最广泛研究的领域，**使用完整标注的图像-文本-边界框三元组数据**，经过十年发展已形成多个技术分支。与之对应的**弱监督视觉定位只提供图像和文本描述而无边界框标注**，这种设置更符合现实应用场景但技术难度更高。半监督视觉定位利用部分标注数据和大量未标注数据进行训练，是解决标注成本高昂问题的有效方法。**无监督视觉定位完全摆脱了对标注数据的依赖，仅从未标注图像中学习**，通常借助辅助模型如目标检测器辅助学习。**零样本视觉定位则代表了更高层次的泛化能力，模型能够定位训练过程中从未见过的类别**，这方面的研究通常分为两个方向：一是学习基础类别的定位能力并测试其在新类别上的表现；二是利用其他任务预训练的模型，无需特定微调即可评估其定位能力。

**Generalized Visual Grounding**是2023年提出的概念，旨在克服传统视觉定位的局限性。传统视觉定位建立在一个强假设基础上：图像中必须有且仅有一个由文本描述的目标对象。这一假设在现实场景中往往不成立，因此GVG提出了更加灵活的定位范式，包括单一目标定位、多目标定位和无目标定位三种情况。这一概念也被称为泛化指代表达理解或描述目标检测，具有更强的实用价值。例如，在工程建设和交通安全领域，简单查询"无安全帽的个体"可在摄像头视频流中得到广泛应用。GVG的提出使视觉定位更加贴近实际应用需求，但也带来了更复杂的数据集需求和建模挑战。

随着技术发展，视觉定位在多个**专业领域**形成了特定应用方向。例如，**遥感视觉定位**专注于卫星遥感图像，面临大尺度变化和杂乱背景等挑战。与自然场景图像不同，遥感图像由卫星获取，物体外观通常呈现相似的几何形状，需要模型考虑图像中的多尺度信息并解决预训练模型迁移中的领域差异问题。医学视觉定位旨在医学影像中定位与医学查询短语相对应的区域，是医学影像分析和放射学诊断的关键任务。医学放射学图像通常呈现平面、灰度特征，缺乏显著的物体轮廓，需要专业知识才能识别病变和生理区域，这与自然场景图像有显著差异。此外，3D视觉定位将定位任务扩展到三维空间，为虚拟现实和增强现实等应用提供支持。视频目标定位则在时间维度上扩展了视觉定位能力，适用于视频内容分析和理解。这些领域特化的应用展示了视觉定位技术的多样性和适应性。

**视觉定位与其他视觉语言任务的结合形成了多样化的多任务学习框架**。将REC与指代表达生成(REG)结合可实现循环一致性学习，增强模型的鲁棒性；REC与指代表达分割(RES)的结合则提供了更精细的定位能力，从矩形边界框扩展到不规则掩码区域；与图像描述(Image Captioning)结合则增强了模型对视觉内容的语义理解能力。这些多任务设置不仅提高了视觉定位的性能，也促进了相关任务的协同发展。

此外，视觉定位还衍生出一些具有特殊应用价值的任务形式，如Grounded Object Detection和Referring Counting。Grounded Object Detection将单模态检测任务与多模态框架结合，显著增强了模型感知广泛开放和多样化对象的能力。Referring Counting将计数任务与定位任务结合，比传统计数任务更具实用性，能够区分用户所需的特定信息。

综上所述，视觉定位通过多种任务类型展示了其在视觉理解领域的基础地位和广泛应用价值。随着技术不断发展，视觉定位任务类型将进一步丰富和完善，为多模态人工智能领域提供更强大的支持。

## 视觉大模型结合模型轻量化技术相关研究成果

### 2025

- **Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-Language Context Sparsification** (ICLR 2025)

  - [论文链接](https://arxiv.org/pdf/2412.00876) ｜ [开源项目链接](https://github.com/microsoft/Dynamic-LLaVA) ｜ [论文解读](Papers/2025-ICLR-Dynamic-LLaVA-%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-Language%20Context%20Sparsification.md)
  - 创新性提出多模态大语言模型动态视觉-语言上下文稀疏化框架，通过在预填充和解码阶段动态减少视觉和语言上下文的冗余标记，显著降低计算消耗和GPU内存开销
  - 设计了针对不同推理模式的稀疏化推理方案，使用可学习的预测器为图像和输出文本标记生成二进制掩码，通过端到端训练确保模型性能不受影响
- **Align-KD: Distilling Cross-Modal Alignment Knowledge for Mobile VLM** (CVPR 2025)

  - [论文链接](https://cvpr.thecvf.com/virtual/2025/poster/33163) ｜ [开源项目链接](https://github.com/microsoft/Align-KD) ｜ [论文解读](Papers/2025-CVPR-Align-KD-%20Distilling%20Cross-Modal%20Alignment%20Knowledge%20for%20Mobile%20VLM.md)
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

### 2024

- **Visual Grounding with Dual Knowledge Distillation (DUET)** (IEEE TCSVT 2024)

  - DOI: [10.1109/TCSVT.2024.3407785](https://doi.org/10.1109/TCSVT.2024.3407785)
  - 提出双向蒸馏框架，通过同时对视觉和语言两路进行蒸馏，缩小跨模态特征差距，提高定位精度
- **SimVG: A Simple Framework for Visual Grounding with Decoupled Multi-modal Fusion** (NeurIPS 2024)

  - [论文链接](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dc6319dde4fb182b22fb902da9418566-Abstract-Conference.html)
  - 采用多分支结构，一支为轻量MLP分支，设计动态权重平衡蒸馏，在保证性能的同时大幅提升推理速度

### 2023

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

### 2022

- **Look Around and Refer: 2D Synthetic Semantics Knowledge Distillation for 3D Visual Grounding** (NeurIPS 2022)
  - [论文链接](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f0b42291ddab77dcb2ef8a3488301b62-Abstract-Conference.html)
  - DOI: 10.48550/arXiv.2211.14241
  - 生成合成二维视图并蒸馏其知识到3D视觉流，提升了3D场景中以自然语言定位物体的效果和泛化能力

### 2021

- **Improving Weakly Supervised Visual Grounding by Contrastive Knowledge Distillation** (CVPR 2021)
  - [论文链接](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Improving_Weakly_Supervised_Visual_Grounding_by_Contrastive_Knowledge_Distillation_CVPR_2021_paper.pdf)
  - DOI: 10.1109/CVPR46437.2021.00074
  - 通过对象检测器提供的软标签进行对比学习蒸馏，无需检测器推理即可显著提升弱监督短语定位性能

### 视觉大模型+知识蒸馏相关数据集

| 数据集名称                                                                            | 类型             | 任务/用途              | 说明及特点                                                      |
| ------------------------------------------------------------------------------------- | ---------------- | ---------------------- | --------------------------------------------------------------- |
| [656K Mixture Dataset](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception) | 指令微调数据集   | 多模态指令微调训练     | 用于模型训练的混合数据集，类似于LLaVA-1.5所用数据               |
| [VQAv2](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception)                | 视觉问答         | 视觉问答               | 常用视觉问答基准，测试模型对图像内容的理解和问答能力            |
| [GQA](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception)                  | 视觉问答         | 视觉问答               | 关注视觉常识推理，图像理解和复杂问题推理                        |
| [VizWiz](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception)               | 视觉问答         | 视觉问答               | 针对视力障碍者场景的问答数据集，图片质量多样且复杂              |
| [SciQA](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception)                | 专业领域视觉问答 | 视觉科学问答           | 侧重科学领域问题的视觉问答                                      |
| [TextVQA](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception)              | 视觉问答         | 包含文本识别的视觉问答 | 要求模型识别图像中的文本信息并回答相关问题                      |
| [POPE](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception)                 | 视觉问答         | 视觉问答               | 具体任务详情未给，属于视觉理解基准之一                          |
| [MMBench (en)](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception)         | 视觉理解基准     | 多模态模型综合测试     | 英文多模态模型的综合评测基准                                    |
| [SEED (image)](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception)         | 图像相关基准     | 视觉理解               | 具体内容未详细说明，属于视觉基准                                |
| [MM-Vet](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception)               | 视觉理解基准     | 视觉理解               | 具体内容未详细说明，属于视觉基准                                |
| [MMVP](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception)                 | 视觉为中心基准   | 专注于视觉理解         | 视觉为中心的视觉理解基准，用于模型视觉表现测试                  |
| [RealWorldQA](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception)          | 真实世界视觉问答 | 视觉问答               | 真实世界场景下的视觉问答测试                                    |
| [CVBench-2D](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception)           | 视觉理解基准     | 视觉理解               | 2D视觉任务基准，测试模型二维视觉能力                            |
| [LVIS-VQA (单轮/多轮)](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception) | 视觉问答         | 多轮及单轮视觉问答生成 | 基于LVIS-Instruct4V子集，测试模型对复杂交互式视觉问答的生成能力 |
| [ShareGPT4V-VQA](https://github.com/IamZLT/KnowledgeDistillation-VisualPerception)       | 视觉问答生成     | 长文本生成视觉问答     | 基于ShareGPT4V数据集，测试单轮长文本视觉问答生成能力            |
