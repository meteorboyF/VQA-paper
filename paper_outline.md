# Paper Outline - Defect-Aware Refusal and Recovery for Assistive VQA

## Working Title

Knowing When and Why to Refuse: Defect-Aware Explanation and Recovery for Assistive Visual Question Answering

## Revised Thesis

For assistive VQA, the most useful reliability layer is not necessarily one that
beats the VQA model's own confidence at risk ranking. Our results show that
frozen VQA confidence remains the strongest selective-prediction signal, while
human-labeled visual defect diagnosis provides something confidence cannot:
interpretable refusal reasons and actionable retake guidance for blind and
low-vision users.

This is the central pivot. The paper should be honest that defect-aware gating
does not significantly improve AURC in the current experiments. The novelty is
the combined refusal-and-recovery layer built from real VizWiz answerability and
quality-issue labels, plus the negative finding that defect labels are useful for
explanation/action but not for improving confidence ranking.

## Core Contributions

1. **Defect-aware refusal layer for assistive VQA.**
   We train lightweight heads on frozen visual embeddings to predict whether a
   VizWiz question is answerable and which human-labeled image-quality defects
   are present.

2. **Actionable Recovery Rate and False Refilm Rate.**
   We introduce ARR/FRR to evaluate whether defect diagnosis leads to useful
   retake advice, not just whether a model abstains.

3. **Confidence-versus-defect diagnostic finding.**
   We show that VQA confidence remains difficult to beat for selective
   prediction: neither predicted defects nor oracle ground-truth defects improve
   AURC over global confidence in this setup. This is a publishable negative
   result if framed carefully.

4. **Backbone and architecture benchmark.**
   We compare CLIP, DINOv2, and MobileNet features under identical heads, and
   quantify joint versus cascade designs for answerability and defect diagnosis.

5. **Optional Phase 2: groundability.**
   E9 should be treated as an optional extension for spatial guidance, not as a
   rescue path for the unsupported AURC claim.

## Claims To Avoid

- Do not claim defect-conditioned calibration or gating beats global confidence.
- Do not make AURC the headline win.
- Do not imply the paper builds a better VQA model.
- Do not use heuristic defect or intent labels; all labels must remain VizWiz
  human annotations.
- Do not frame E9 as required for the paper unless it produces a clear new
  groundability result.

## Key Results From Current Run

| Backbone | Triage AUROC | Defect mAP | ARR | FRR | Joint-Cascade dAUROC |
|---|---:|---:|---:|---:|---:|
| CLIP | 0.7926 | 0.6216 | 0.7196 | 0.7746 | 0.0013 |
| MobileNet | 0.7512 | 0.5801 | 0.6278 | 0.7780 | 0.0037 |
| DINOv2 | 0.7769 | 0.6309 | 0.6907 | 0.7775 | 0.0028 |

Selective prediction diagnostic:

| Backbone | E7 defect AURC delta | p | E7b predicted-risk improvement | p | E7b oracle improvement | p |
|---|---:|---:|---:|---:|---:|---:|
| CLIP | 0.000895 | 0.297 | 0.000701 | 0.798 | -0.005343 | 0.039 |
| MobileNet | 0.000798 | 0.361 | 0.002360 | 0.418 | -0.005343 | 0.039 |
| DINOv2 | -0.000118 | 0.923 | -0.002442 | 0.448 | -0.005343 | 0.039 |

Interpretation: positive improvement means lower AURC than global confidence.
All predicted-defect improvements are statistically unsupported. Oracle
ground-truth defects are worse than global confidence in this setup.

## Manuscript Structure

### Abstract

State the assistive VQA reliability problem: a blind or low-vision user needs to
know not only whether an answer is likely wrong, but also why the image failed
and how to retake it. Summarize the method as a lightweight reliability layer on
frozen visual backbones. Report the strongest numbers: CLIP triage AUROC,
DINOv2 defect mAP, CLIP ARR. Include the negative result: defect-aware risk
ranking does not outperform global VQA confidence, so defect diagnosis should
be used for explanation and recovery rather than confidence replacement.

### 1. Introduction

Motivate from VizWiz: images are captured by blind users, often poor quality,
and questions can be unanswerable. Existing VQA reliability work focuses on
abstention and selective prediction; existing image-quality work labels defects.
The gap is connecting answerability, image-quality defects, and actionable
retake guidance in a single assistive pipeline.

End the introduction with the revised contributions above.

### 2. Related Work

Organize around:

- Assistive VQA and VizWiz.
- Image-quality issues in real-world assistive images.
- Selective prediction and calibration for VQA/VLMs.
- Actionable feedback and recovery for accessibility systems.

Positioning:

- VizWiz VQA establishes unanswerable assistive VQA as a core problem.
- VizWiz-QualityIssues provides human defect labels tied to practical vision
  tasks.
- Selective VQA work studies when to abstain, often via confidence or learned
  selectors.
- Our paper differs by evaluating whether quality-defect diagnosis adds
  explainable refusal and recovery value, and by reporting that it does not beat
  confidence for AURC.

### 3. Data and Labels

Describe:

- VizWiz-VQA answerability and answer annotations.
- VizWiz-QualityIssues defect labels: blur, bright, dark, obstruction, framing,
  rotation, unrecognizable.
- Inner join by image and split.
- Train/cal/report split:
  - train: model fitting
  - cal: thresholds, temperatures, learned selectors
  - report: all reported metrics

Current run:

- 24,319 total rows.
- 20,000 train, 4,319 val.
- Answerable rate: 0.7214.
- High defect prevalence: framing 0.5606, blur 0.4161.

### 4. Method

Pipeline:

1. Frozen visual embeddings from CLIP, MobileNet, DINOv2.
2. Answerability triage head.
3. Multi-label defect diagnosis head.
4. Frozen ViLT confidence harvest for VQA answer confidence.
5. Selective prediction diagnostics.
6. Actionable recovery mapping from predicted defect to retake guidance.

Important framing:

- The reliability layer wraps a frozen VQA model.
- It does not train or fine-tune the VQA model.
- Defect heads are for explanation and recovery, not for replacing VQA
  confidence.

### 5. Metrics

Use:

- Triage: AUROC, AUPRC, F1, majority baseline.
- Defect diagnosis: per-defect AUROC/AUPRC, mAP, macro/micro F1.
- Selective prediction: AURC, risk-coverage curves, ECE, paired bootstrap.
- Recovery: ARR and FRR.
- Architecture: joint versus cascade dAUROC.

Define ARR/FRR clearly:

- ARR measures how often an unanswerable case receives a predicted defect whose
  corrective action matches a true defect.
- FRR measures how often answerable cases would be unnecessarily asked to retake.

### 6. Results

Suggested result order:

1. Data sanity and defect prevalence.
2. Answerability triage works best with CLIP.
3. Defect diagnosis works best with DINOv2 by mAP.
4. Actionable recovery is strongest with CLIP.
5. Joint heads are slightly but consistently better than cascade.
6. Selective prediction negative result: global VQA confidence remains stronger
   than defect-aware risk scores for AURC.

This order makes the negative result part of the scientific story rather than a
collapse of the paper.

### 7. Discussion

Main discussion points:

- Defect labels are semantically meaningful but not necessarily ordered by VQA
  correctness risk once the VQA model already emits confidence.
- Confidence may implicitly encode many visual and linguistic failure modes.
- Defect diagnosis remains valuable because it tells a user what to do next.
- High FRR indicates the recovery policy needs cost-sensitive tuning before
  deployment.
- The strongest deployment setting may be: answer when confidence is high; when
  abstaining, use defect diagnosis to explain and suggest a retake action.

### 8. Limitations

Include:

- Only VizWiz train/val; hidden test answers are not used.
- Frozen VQA model is ViLT, which has low absolute accuracy on this run.
- ARR/FRR depends on a deterministic defect-to-action map.
- Defect labels explain image quality, not all answerability failures.
- Groundability/spatial guidance remains optional Phase 2.

### 9. Conclusion

Conclude that assistive VQA reliability should separate risk ranking from user
recovery. Confidence decides when to trust an answer; defect diagnosis explains
why an answer should be refused and what the user can do next.

## Novelty Position

The defensible novelty is the combination of:

- joining VizWiz-VQA answerability with VizWiz-QualityIssues labels,
- training lightweight reliability heads on frozen modern backbones,
- evaluating actionable recovery through ARR/FRR,
- and reporting a careful negative result for defect-aware selective prediction.

This is more solid than claiming a selective-prediction win that the data does
not support.

## Related Work Anchors To Cite

- VizWiz VQA dataset/task: https://vizwiz.org/tasks-and-datasets/vqa/
- VizWiz-QualityIssues dataset/task: https://vizwiz.org/tasks-and-datasets/image-quality-issues/
- Chiu et al., Assessing Image Quality Issues for Real-World Problems, CVPR 2020.
- VizWiz Grand Challenge / Gurari et al., CVPR 2018.
- Reliable Visual Question Answering: Abstain Rather Than Answer Incorrectly, ECCV 2022.
- Improving Selective Visual Question Answering by Learning From Your Peers, CVPR 2023.
- Selectively Answering Visual Questions, 2024.

## Next Decision

Do not run E9 yet by default. First write the introduction, contributions, and
results narrative around the revised thesis. Run E9 only if the paper needs a
distinct optional contribution on groundability or spatial retake guidance.
