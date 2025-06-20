# 2022-CVPR-Grounded Language-Image Pre-training

这篇论文提出了一种名为GLIP（Grounded Language-Image Pre-training）的视觉预训练模型，旨在同时学习具备语言理解能力的对象级视觉表示。作者通过将目标检测任务和短语定位任务进行统一建模，使模型不仅能够识别图像中的目标，还能理解文本短语与图像区域之间的语义对应关系。GLIP通过构建一个联合的图像-文本编码框架，实现了视觉特征和语言特征的深度融合，从而显著增强了模型的语义理解能力。该模型在大规模数据集上进行预训练，包括人工标注和从网络爬取的图文对，通过自训练方式自动生成定位框，实现了高效的数据扩展。在实验中，GLIP在COCO、LVIS等标准数据集上展现出优异的零样本和少样本迁移能力，并在13个实际应用场景下取得了与监督模型相当甚至更优的性能，验证了其强大的通用性和部署灵活性。这项研究展示了多模态预训练在视觉理解中的巨大潜力，为构建更泛化、低成本的目标检测系统提供了新路径。

## 摘要

本文提出了一种面向对象级别、具备语言感知能力并富含语义信息的视觉表示学习模型——GLIP（Grounded Language-Image Pre-training）。该模型将目标检测与短语定位任务统一起来进行预训练。这一统一带来了两大优势：

1. GLIP可同时从检测数据和定位数据中学习，从而提升两个任务的性能，并构建出优质的定位模型；
2. GLIP能够通过自训练方式为大量图文对生成定位框，从而学习到语义丰富的表示。

在实验中，GLIP在2700万条定位数据上进行了预训练，其中包括300万条人工标注数据和2400万条网络爬取的图文对。实验结果表明，GLIP在多个对象级识别任务中展现出强大的零样本和少样本迁移能力。

1. 在未使用COCO图像进行预训练的情况下，GLIP在COCO和LVIS数据集上直接评估时分别获得了49.8和26.9的AP，超过了许多监督学习基线；
2. 在COCO上进行微调后，GLIP在验证集和test-dev上分别达到了60.8和61.5的AP，超越了以往的SOTA方法；
3. 在迁移至13个下游目标检测任务时，仅使用1个样本的GLIP就可媲美完全监督的Dynamic Head模型。代码将公开于：https://github.com/microsoft/GLIP。

# 📘 Introduction

视觉识别模型通常被训练用于预测一组预先定义好的固定对象类别，这限制了它们在现实应用中的实用性，因为要泛化到新的视觉概念和领域，往往还需要额外的人工标注数据。CLIP 的研究表明，通过大量原始图文对，可以有效地学习图像级的视觉表示。由于这些配对的文本通常包含比任何预设类别集合都更广泛的视觉概念，CLIP 预训练模型具备非常丰富的语义能力，因此在零样本设置下，能够轻松迁移到图像分类和图文检索等下游任务中。

然而，为了实现对图像更细粒度的理解——这是许多任务所必需的，比如目标检测、图像分割、人体姿态估计、场景理解、行为识别以及视觉语言理解等任务——我们迫切需要对象级的视觉表示。在本研究中，我们提出“短语定位”（phrase grounding）任务作为一种高效且可扩展的预训练手段，用于学习对象级、具备语言感知能力、并富含语义的视觉表示，并据此提出GLIP（Grounded Language-Image Pre-training）模型。我们的方法统一了短语定位任务与目标检测任务：目标检测可以看作是“无上下文”的短语定位，而短语定位可以看作是“有上下文”的目标检测。我们通过将目标检测重构为短语定位，实现了检测与定位任务的统一。

**这种重构改变了检测模型的输入形式：模型的输入不仅包括图像，还包括一个文本提示，用于描述该检测任务中的所有候选类别。**例如，在 COCO 目标检测任务中，文本提示由 80 个类别名称组成的字符串构成，这 80 个类别以句号连接，如图1左侧所示。任何目标检测模型都可以通过将其框分类器中的分类 logits 替换为“词语-区域对齐得分”而转化为短语定位模型。这里的对齐得分指的是区域（或边框）的视觉特征与词语（或短语）的语言特征之间的点积，如图1右侧所示。语言特征由语言模型计算得出，这使得新的检测（或定位）模型具有一个双编码器结构。**与 CLIP 不同，CLIP 仅在最后的点积层融合视觉与语言信息，而我们提出的 GLIP 则在模型中部引入了深层的跨模态融合（如图1中部所示）**，这一点对于学习高质量、具备语言感知能力的视觉表示以及实现更优越的迁移学习性能至关重要。检测与定位任务的统一还使我们能够同时利用这两类数据进行预训练，并且这种方式对两个任务都是有益的。对于目标检测而言，得益于定位数据的引入，视觉概念的种类得到了显著扩展；而对于短语定位而言，目标检测数据提供了更多的边框注释，有助于训练出新的最先进（SoTA）的短语定位模型。

**利用海量图文数据扩展视觉概念。** 一旦拥有了一个性能良好的定位模型（即教师模型），我们便可以通过自动生成定位框的方式，为大量图文对扩充GLIP的预训练数据。这些图文对中的名词短语由自然语言处理工具（NLP parser）识别得到【参考文献2】。因此，我们可以在2700万条定位数据上对学生模型GLIP-L（GLIP-Large）进行预训练，其中包括300万条人工标注的细粒度数据和2400万条从网络爬取的图文对。在这2400万条图文对中，我们共生成了7810万个置信度大于0.5的伪定位框注释，涉及5840万个独特的名词短语。我们在图2中展示了两个由教师模型生成的定位框示例。教师模型可以准确地定位一些具有挑战性的概念，例如“注射器”“疫苗”“美丽的加勒比海蓝绿色”，甚至包括抽象词语（如“风景”）。在如此语义丰富的数据上进行训练，能够得到同样语义丰富的学生模型。相比之下，先前的检测数据扩展方法无法预测超出教师模型预定义词汇表的概念【参考文献68】。**而在本研究中，我们表明，通过扩大短语定位数据的规模这一简单策略，在实际中是非常有效的，尤其在 LVIS 和 13 个下游目标检测任务上带来了显著提升，特别是在识别稀有类别方面（**详见第4.2节和第5节）。当预训练完成的 GLIP-L 模型在 COCO 数据集上进行微调后，在 COCO 2017 的验证集上取得了 60.8 的 AP，在 test-dev 上达到 61.5，超越了当前通过各种方法扩展目标检测数据而得到的公开最先进模型【参考文献9，58】。

**GLIP 的迁移学习能力：一模型通用多任务。** 通过定位重构和语义丰富的预训练，GLIP 极大地提升了跨领域的迁移能力。它可以在仅使用极少甚至完全不使用人工标注的情况下，迁移到多种任务上。当我们在未使用任何 COCO 图像进行预训练的前提下，直接在 COCO 和 LVIS 数据集上评估 GLIP-L 模型时，其在 COCO val2017 上取得了 49.8 的 AP，在 LVIS val 上达到 26.9 的 AP，均超越了许多监督学习的基线模型。当在13个已有的目标检测数据集上进行评估时（这些数据集涵盖了细粒度物种检测、无人机视角检测、第一人称视角检测等多种场景，我们称之为“真实世界目标检测任务”即 ODinW，详见第5.1节），GLIP 展现出了卓越的数据效率。例如，在零样本设置下，GLIP-L 的表现超过了在 Objects365 上预训练并使用10个样本的监督基线模型（Dynamic Head）；而在仅使用1个样本的情况下，GLIP-L 的性能也可以与完全监督的 Dynamic Head 相媲美。此外，当任务的特定标注数据可用时，我们无需微调整个模型，仅需调整任务相关的提示词嵌入（prompt embedding），即可保持模型参数不变地适应新任务。

## Related Work

传统的目标检测系统通常在人工标注的大规模数据集上进行训练，这些数据集预定义了一组固定的目标类别，例如 COCO [32]、OpenImages（OI）[25]、Objects365 [45] 和 Visual Genome（VG）[23]，其中的类别数通常不超过 2000 个。这类人工标注数据的获取与扩展成本非常高。GLIP 提供了一种更具可扩展性的解决方案，即将目标检测任务重构为短语定位（即词语与图像区域的匹配）问题，从而可以有效地利用定位数据以及海量图文对数据。虽然我们当前的实现是基于 Dynamic Head（DyHead）[9] 构建的，但所提出的统一建模方式可以泛化到任何目标检测系统中【如文献 4、5、8、9、31、43、44、67】。

近年来，越来越多的研究开始采用视觉-语言联合方法来解决视觉识别问题，其核心思想是利用自然语言作为监督信号来训练视觉模型。例如，CLIP [40] 和 ALIGN [18] 在数亿级别的图文对上进行跨模态对比学习，从而能够直接实现开放词表的图像分类。ViLD [12] 则将 CLIP/ALIGN 模型中学到的知识蒸馏到两阶段检测器中，推动了零样本目标检测的发展。另一个方向是 MDETR [19]，该方法在现有的多模态数据集上进行端到端训练，这些数据集中明确定义了文本短语与图像中对象之间的对应关系。我们的 GLIP 模型延续了这一类研究中“语义丰富、语言感知”的特性，不仅在目标检测任务中达到了当前最先进的性能，还显著提升了对下游检测任务的迁移能力。

本论文的重点是目标检测任务中的跨领域迁移问题。我们的目标是构建一个预训练模型，能够以零样本或少样本的形式，无缝迁移到各种任务和应用场景中。与传统的零样本检测设定（如文献 [1, 12, 41, 42, 61, 66]）不同，后者通常会将一部分类别设定为“未见类别”或“稀有类别”，并在训练集中显式地排除它们。而我们的做法并未从训练集中刻意排除任何类别，因为我们认为，短语定位数据本身就具备非常丰富的语义信息，能够自然地覆盖大量的稀有类别，我们也确实期望 GLIP 能在这些类别上表现良好（详见第 4.2 节）。这一设定更接近于“开放词表目标检测”任务 [61]，该任务同样希望通过原始图文对数据覆盖尽可能多的稀有类别。除了在稀有类别上的表现之外，我们还特别关注现实应用中的“迁移成本”问题——也就是说，如何在使用最少的数据、最小的训练预算和最低的部署成本的前提下，实现最优性能（详见第5节）。

## Grouned Language Image Pre-training

从概念上看，目标检测和短语定位这两项任务有很大的相似之处：它们都旨在定位图像中的对象，并将其与相应的语义概念对应起来。这种天然的关联性激励我们将传统的目标检测任务重构为一种短语定位问题，并据此提出了统一的任务建模方法（见第 3.1 节）。此外，我们还提出引入图像与文本之间的深度融合机制，使得检测模型具备语言感知能力，从而成为一个强大的短语定位模型（见第 3.2 节）。结合重构与融合，我们可以在语义丰富且具可扩展性的定位数据上对 GLIP 模型进行预训练（见第 3.3 节）。

### 3.1 统一建模（Unified Formulation）

在标准目标检测中，模型通常会将图像输入视觉编码器（$Enc_I$），该编码器通常由 CNN [15, 51] 或 Transformer [34, 60, 62] 构成主干网络，然后提取出区域或边框级的特征$O$。如图 1 所示，模型随后会使用两个预测头：一个是边框分类器 $C$，另一个是边框回归器$R$，分别通过分类损失 $L_{\text{cls}}$ 和定位损失 $L_{\text{loc}}$ 进行训练，联合目标函数为：
$$
L = L_{\text{cls}} + L_{\text{loc}}
$$
在两阶段检测器中，还会加入区域建议网络（RPN），对应一个 RPN 损失 LrpnL_{\text{rpn}}，用于区分前景与背景，并精细调整锚框。但由于该损失与目标类别的语义无关，我们将其并入定位损失中处理。而在一阶段检测器中，定位损失还可能包括中心度损失（centerness loss）[52]。

边框分类器通常是一个线性层，其分类过程如下所示：
$$
O = \text{Enc}_I(\text{Img}), \quad S_{\text{cls}} = O W^\top, \quad L_{\text{cls}} = \text{loss}(S_{\text{cls}}, T)
$$
其中，$O \in \mathbb{R}^{N \times d}$表示图像中 $N$个区域的$d$维特征，$W \in \mathbb{R}^{C \times d}$ 为分类器的权重，$S_{\text{cls}} \in \mathbb{R}^{N \times C}$ 为预测的分类得分矩阵，$T \in \{0, 1\}^{N \times C} $表示每个区域与类别的匹配目标标签。

**将目标检测重构为短语定位。**与其将每个区域直接归类到某一类别，我们提出将检测任务重构为短语定位任务：即将每个图像区域与提示文本中 C 个短语进行对齐。比如对于 COCO 的 80 类检测任务，我们可以构造如下 prompt：

> Detect: person, bicycle, car, ..., toothbrush.

该文本提示中的每个类别名称作为一个短语，用作区域的匹配目标。我们还可以根据语言模型（如 BERT）的特性进一步优化 prompt 的表达形式。实验表明，对于语言编码器 Enc_L 使用 BERT 初始化的情况，将类别用句号隔开构造 prompt（如“person. bicycle. ... toothbrush.”）效果更好。

在这种重构方式下，我们不再使用固定分类器权重矩阵 $W$，而是动态计算图像区域与 prompt 文本中各 token 的相似度（即 region-word alignment）。公式如下：
$$
O = \text{Enc}_I(\text{Img}), \quad P = \text{Enc}_L(\text{Prompt}), \quad S_{\text{ground}} = O P^\top
$$
其中，$P \in \mathbb{R}^{M \times d} $是文本提示中 $M$ 个 token 的语言特征。该对齐得分矩阵 $S_{\text{ground}} \in \mathbb{R}^{N \times M}$ 表示图像中每个区域与每个 token 的匹配分数。为了使用该矩阵进行监督训练，我们将原始目标标签 $T \in \{0, 1\}^{N \times C}$ 扩展为 $T' \in \{0, 1\}^{N \times M}$，这样每个短语的所有子词都被视为正样本 token，特殊符号（如“Detect:”、逗号、“[NoObj]”等）标记为负样本。使用扩展标签后，损失函数形式保持不变：
$$
L = \text{loss}(S_{\text{ground}}, T')
$$
推理阶段中，我们将每个短语中所有 token 的得分进行平均，得到该短语与图像区域的匹配概率。

检测与定位的等价性。上述重构方式使得任何目标检测模型都可以被转化为短语定位模型，且二者在训练与推理中是理论等价的。我们在实验证明中也观察到：当将 SOTA 检测器 DyHead（Swin-Tiny）按本方法重构为定位形式后，其在 COCO val2017 上的性能保持不变。这种统一建模方式的最大优势在于：我们可以利用预训练好的短语定位模型 GLIP，直接迁移到任何目标检测任务中，只需更改输入 prompt 即可实现零样本推理。这为跨任务、跨领域的高效迁移提供了可能。

**相关工作。**我们的短语定位建模方式受到 MDETR [19] 的启发，其提出了细粒度对比损失来训练文本与图像区域的对应关系。与其不同的是，GLIP 进一步发展出一种有效的重构方式，将目标检测统一为短语定位任务，并提出了一个同时适用于检测与定位的统一损失函数。此外，GLIP 的结构也与多个零样本检测方法相似 [1, 12, 41, 42, 66]。

例如，Bansal 等人在开创性研究 [1] 中提出使用预训练的 Glove 词向量作为短语特征 $P \in \mathbb{R}^{C \times d}$，将其直接嵌入检测器，从而实现零样本目标检测。最近的研究则进一步引入了更强大的深度语言模型来提取短语特征，用于开放词表检测任务 [61]。

GLIP 与这些方法的根本区别在于：它不仅支持零样本检测，还提出了一个统一的视角，将“检测”与“定位”这两个任务融合，并引入了两个关键机制：

1. **语言感知的深度融合机制**，实现图像与文本的深层信息交互；
2. **可扩展的图文数据训练策略**，通过自监督方式大规模扩展训练数据。

### 3.2 语言感知的深度融合（Language-Aware Deep Fusion）

在前文的公式 (3) 中，图像和文本是由两个独立的编码器分别进行编码的，最终仅在对齐得分计算阶段才进行融合。这类模型被称为“后融合（late-fusion）”模型。而在视觉-语言相关研究中 [7, 19, 27, 28, 30, 36, 48, 50, 65]，深度融合视觉与语言特征被认为是实现高性能短语定位模型的关键。

我们在图像编码器与语言编码器之间引入了“深度融合”机制，使得图像与文本信息可以在编码器的后几层实现交互，如图1中部所示。具体地说，当我们使用 DyHead [9] 作为图像编码器，BERT [10] 作为文本编码器时，融合过程如下：

- 首先使用跨模态多头注意力（X-MHA）模块计算图像到文本和文本到图像的注意力向量；
- 然后将这些融合信息分别添加到 DyHead 模块和 BERT 层中进行更新；
- 最终输出经过 L 层交互的图像特征 OOO 和文本特征 PPP。

公式表达如下：
$$
O^i_{\text{t2i}}, P^i_{\text{i2t}} = \text{X-MHA}(O^i, P^i), \quad i = 0, 1, ..., L-1
$$

$$
O^{i+1} = \text{DyHeadModule}(O^i + O^i_{\text{t2i}}), \quad O = O^L
$$

$$
P^{i+1} = \text{BERTLayer}(P^i + P^i_{\text{i2t}}), \quad P = P^L
$$

其中，$L$ 表示 DyHead 中 DyHead 模块的层数；BERTLayer 表示在预训练 BERT 基础上新增的层；$O^0$ 和 $P^0$分别表示初始的图像特征和文本 token 特征。跨模态的信息交互通过跨模态多头注意力模块（X-MHA）完成，随后各自通过单模态更新步骤分别融合和更新。

如果不加入这些上下文向量（即 $O^{t2i}*{i} = 0$ 和 $P^{i2t}*{i} = 0$），那么模型就会退化为一个后融合模型。相比之下，我们所提出的**深度融合机制**能实现在图像特征编码过程中引入语言信息，反之亦然，实现真正的图文双向交互。

在跨模态多头注意力模块（X-MHA）（公式 4）中，每个注意力头通过关注另一模态的信息来计算当前模态的上下文向量：
$$
O^{(q)} = O W^{(q,I)}, \quad P^{(q)} = P W^{(q,L)}, \quad \text{Attn} = \frac{O^{(q)} (P^{(q)})^\top}{\sqrt{d}},
$$

$$
P^{(v)} = P W^{(v,L)}, \quad O_{\text{t2i}} = \text{SoftMax}(\text{Attn}) P^{(v)} W^{(\text{out}, I)},
$$

$$
O^{(v)} = O W^{(v,I)}, \quad P_{\text{i2t}} = \text{SoftMax}(\text{Attn}^\top) O^{(v)} W^{(\text{out}, L)},
$$

其中，${W^{(\text{symbol},I)}, W^{(\text{symbol},L)} : \text{symbol} \in {q, v, \text{out}}}$ 是可训练的参数，分别起到多头自注意力机制中 query、value 和 output 线性变换的作用，详见文献 [53]。

**深度融合编码器（公式 4–6）带来了两个好处：**

1. 显著提升了短语定位性能；
2. 使学习到的视觉特征具备语言感知能力，因此模型的预测将根据文本提示进行调整。

这一点对于实现“一模型适配所有下游检测任务”的目标至关重要（详见第 5.2 节）。

### 3.3 利用可扩展且语义丰富的数据进行预训练

人们在收集具有丰富语义且规模庞大的检测数据方面投入了大量努力。然而，人工标注的成本高昂，且数据量始终受限 [13, 25]。已有研究尝试通过**自训练**的方式来扩展数据规模 [68]。这些方法通常采用一个教师模型（即预训练的检测器）对原始图像进行预测，从而生成伪检测标签，用于训练学生模型。

但这类生成的数据在**概念集合的规模**上仍然存在局限，因为教师模型只能预测那些在已有数据集构建的概念集合中定义过的类别标签。相比之下，我们的模型既可以使用检测数据，也可以（更重要地）使用**短语定位数据（grounding data）**进行训练。我们展示了定位数据可以提供更加丰富的语义信息来辅助目标定位任务，并且也可以通过自训练的方式实现规模扩展。

首先，**人工标注的定位数据**所涵盖的视觉概念词汇要远多于现有检测数据。即使是在检测任务中最努力尝试扩展词汇量的工作，也仅能覆盖不超过 2000 个类别 [13, 23]。而通过定位数据，我们可以将词汇扩展至几乎所有出现在图文对字幕中的概念。

例如，Flickr30K [39] 中包含 44,518 个独特短语，而 VG Caption [23] 中则包含 110,689 个独特短语，这些数量级远远超过传统检测数据中的类别数量。我们在第 4.4 节中提供了一个实证研究，证明使用 80 万条人工标注的定位数据，在检测稀有类别方面的性能提升**超过**额外添加 200 万条检测数据的效果。

此外，与其扩大检测数据，不如直接扩展定位数据来获取更语义丰富的数据，这是我们提出的一个更有前景的路径。我们采用了一种受**自训练**启发的简单方法：首先，我们使用人工标注的检测数据与定位数据，预训练一个 GLIP 教师模型；接着，使用该教师模型对网络爬取的图文对进行预测，并借助 NLP 解析器 [2] 从文本中抽取名词短语；最后，我们将人工数据与生成的伪定位数据结合，用于训练学生模型。

如图 2 所示，教师模型能够为语义丰富的实体生成准确的边框。

那么，为什么学生模型可能比教师模型表现更好？尽管在自训练的文献中这一问题仍有广泛讨论 [68]，但在视觉定位任务的背景下，我们认为：**教师模型利用了语言上下文与语言泛化能力，从而能够准确地定位其本身未曾明确学习过的概念**。

例如，在图 2 中，如果“疫苗（vaccine）”和“青绿色（turquoise）”这些词未在人工数据中出现，教师模型可能无法直接识别它们。但丰富的语言上下文（如句法结构）为教师模型提供了强有力的指导，使其能够做出“有根据的猜测”：如果模型能定位到一个小药瓶，它就可能成功定位“疫苗”；如果它能找到“加勒比海”，也就能定位“青绿色”。

当我们训练学生模型时，教师模型的这种“有根据的猜测”就变成了**监督信号**，从而使学生模型最终学会“疫苗”和“青绿色”等概念。