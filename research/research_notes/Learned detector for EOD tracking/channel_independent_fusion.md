# Channel-count-independent multichannel fusion for a multi-electrode EOD salience detector

Scope: architectures that fuse a variable number of sensor channels (2 to 64) without depending on channel count or order, and ways to combine per-channel features with a channel-pooled ("summed") stream. Literature 2017-2026. "Preprint" marks arXiv-only work whose peer-reviewed venue I did not confirm. A few classic references (DeepSets, PNA, Perceiver, PCEN) are cited from my prior knowledge of the papers; I did not re-fetch them this session and mark them "(not re-fetched)".

## 1. Ad-hoc / variable-array speech separation: TAC, its successors, channel attention, VarArray, array-agnostic models

### Takeaway
TAC (Luo et al., ICASSP 2020) is a three-layer DeepSets-style block: a shared per-channel FC, a mean over channels followed by a second FC, and a concatenation of the pooled vector back onto every channel followed by a third shared FC and a residual. It contains no per-channel-index parameters and its only cross-channel operation is a mean, so it is permutation-equivariant and works for any channel count. VarArray (Yoshioka et al., ICASSP 2022) is the closest published template for the project's plan. Its input is "magnitude of the AVERAGE spectrum + per-channel phase relative to that average", it applies per-channel conformer stacks interleaved with TAC, and it merges into one stream early to save compute.

### Cited Findings
**TAC, exact definition (FaSNet-TAC, "End-to-end microphone permutation and number invariant multi-channel speech separation", Luo, Chen, Mesgarani, Yoshioka, ICASSP 2020).** Source: [arXiv 1910.14104](https://arxiv.org/abs/1910.14104). Equations from the PDF:
- Setup: an N-channel array with arbitrary geometry, N ∈ {2,…,Nm}. Each channel is a sequence Z_i ∈ R^{T×K}.
- Transform: f_{i,j} = P(z_{i,j}). P is a fully connected (FC) layer plus PReLU, shared across channels, applied at each time step j and mapping K to D.
- Average: f̂_j = R((1/N) Σ_i f_{i,j}). R is a second FC layer plus PReLU.
- Concatenate: ĝ_{i,j} = S([f_{i,j}; f̂_j]). S is a third FC layer plus PReLU, shared across channels.
- Residual: ẑ_{i,j} = z_{i,j} + ĝ_{i,j}. Output dimension equals input dimension, so TAC "can be inserted into any deep architectures without increasing the optimization difficulty."
- The authors say that with parameter sharing in the transform and concatenate stages and the permutation-invariant average, "TAC guarantees channel permutation and number invariant processing and is always able to make global decisions". They link it explicitly to functions on sets (DeepSets) and graph neural networks.
- Source: [arXiv 1910.14104](https://arxiv.org/abs/1910.14104)

**Where TAC sits in FaSNet.** TAC is inserted into every processing block (DPRNN blocks) of a single-stage FaSNet, which jointly estimates time-domain filter-and-sum beamforming filters for all channels. Cross-channel normalised cross-correlation (NCC) features are mean-pooled over channels, so they too are permutation invariant. [arXiv 1910.14104](https://arxiv.org/abs/1910.14104)

**FaSNet-TAC numbers.**
- Data: simulated noisy, reverberant 2-speaker mixtures from Librispeech, 20,000 / 5,000 / 3,000 4-s utterances, 16 kHz, SNR 10-20 dB.
- Ad-hoc arrays with 2-6 microphones, average SI-SNRi for 2/4/6 mics:
  - two-stage FaSNet: 5.9 / 6.9 / 7.3 dB
  - FaSNet+joint (single-stage): 9.0 / 8.9 / 9.5 dB
  - FaSNet+TAC+joint: 9.8 / 11.2 / 11.7 dB
  - FaSNet+TAC+joint with a 4 ms window: 10.0 / 11.5 / 12.0 dB
- All models have about 2.9-3.0 M parameters.
- The authors say TAC "guarantees that more microphones will not make the performance worse", and that the gain is largest at high overlap. At >75% overlap, joint scores 4.6 / 4.5 / 4.7 dB and TAC+joint 5.5 / 7.2 / 7.6 dB.
- Source: [arXiv 1910.14104](https://arxiv.org/abs/1910.14104)

**Fixed 6-mic circular array, same paper.**
- Average SI-SNRi: TasNet-filter 8.0 dB; with NCC features concatenated by microphone index 6.9 dB (worse); with NCC averaged 8.7 dB; FaSNet 10.0 dB; FaSNet+TAC+joint 11.3 dB; with 4 ms window 11.5 dB.
- By speaker angle, FaSNet vs TAC+joint+4ms: <15°: 8.5 → 9.1 dB; >90°: 11.4 → 13.4 dB. Spatially close sources gain least.
- The authors suggest TAC "is able to learn geometry-dependent information even without explicit geometry-related features."
- Source: [arXiv 1910.14104](https://arxiv.org/abs/1910.14104)

**VarArray ("VarArray: Array-geometry-agnostic continuous speech separation", Yoshioka et al., ICASSP 2022).** Source: [arXiv 2110.05745](https://arxiv.org/abs/2110.05745); details from the PDF:
- Input per microphone m and frame t: z_mt = [ (|X̄_ft|²)_f , ∠(X_mft / X̄_ft)_f ]. X̄ is the average of the complex STFT over microphones. Every channel therefore carries the pooled-spectrum magnitude plus its own phase relative to the pooled spectrum.
- Both features are normalised at the window level. This normalised-IPD set "outperformed the STFT-based method in our preliminary experiment."
- TAC variant: o_out_mt = [ReLU(A o_in_mt), (1/M) Σ_μ ReLU(B o_in_μt)]. A and B halve the dimension, so the output dimension is preserved.
- Network: five conformer blocks, each of five conformer layers with 4 heads, 64-d and kernel size 33. The first three blocks run per channel in parallel with shared weights, with TAC between them. The model then "merge[s] them at an earlier layer… only one stream after the third conformer block. This significantly reduces the network computation cost."
- Training: 3-7 microphones picked at random per mini-batch.
- Source: [arXiv 2110.05745](https://arxiv.org/abs/2110.05745)

**VarArray numbers (%WER).** Columns are BF1 (super-directive beamformer) / SAC (same separator per channel, masks averaged) / Fixed-geometry model / VarArray.
- AMI-dev 8 mics: 25.0 / 18.3 / 18.8 / 17.7
- AMI-dev 4 mics: 25.5 / 18.8 / – / 18.5
- MS 7 mics: 17.6 / 16.0 / 16.0 / 15.5
- MS 3 mics: 17.8 / 16.9 / – / 15.8
- The geometry-agnostic model beat the geometry-specific models, which the authors attribute to its more diverse training data.
- With end-to-end ASR fine-tuning: AMI-dev 17.5%, AMI-eval 20.4% (overlap included).
- Source: [arXiv 2110.05745](https://arxiv.org/abs/2110.05745)

**VarArray's critique of the "run the same model per channel and average outputs" (SAC) approach.** It ignores nonlinear inter-channel correlation, and "the computational cost becomes too large to deploy in practice when the number of channels is large." [arXiv 2110.05745](https://arxiv.org/abs/2110.05745)

**Attention over channels.**
- Multi-channel transformer for ASR (Chang et al., ICASSP 2021). It has channel-wise self-attention (CSA), cross-channel attention (CCA) and multi-channel encoder-decoder attention. It outperformed a single-channel transformer and super-directive / neural beamformers cascaded with transformers on an in-house far-field set. I did not obtain WER numbers. [arXiv 2102.03951](https://arxiv.org/abs/2102.03951)
- Continuous speech separation with ad hoc arrays (Wang, Yoshioka et al., EUSIPCO 2021). It interleaves cross-channel self-attention ("exploits nonlinear spatial correlation between different channels") with cross-frame self-attention, building on a 2020 interleaving design for variable channel counts. [arXiv 2103.02378](https://arxiv.org/abs/2103.02378); earlier work [arXiv 2004.13670](https://arxiv.org/abs/2004.13670)

**Sparsemax channel selection (Chen & Zhang, Interspeech 2021).**
- Replaces the softmax in multichannel stream attention with sparsemax, which forces the weights of noisy channels to exactly zero.
- "Scaling sparsemax" zeroes only very noisy channels.
- Tested on arrays with more than 30 channels and a Conformer ASR model: over 30% relative WER reduction vs softmax on simulated data, over 20% on semi-real data, for both matched and mismatched channel counts.
- Source: [arXiv 2103.15305](https://arxiv.org/abs/2103.15305); [ISCA archive](https://www.isca-archive.org/interspeech_2021/chen21b_interspeech.html)

**Newer array-agnostic separators.**
- UniArray (preprint, 2025): a "virtual microphone estimation" module to cope with different channel counts, a spectral-spatial feature fusion module and a hierarchical dual-path separator. Evaluated only on simulated WSJ0-2mix circular arrays, with reported gains on seen and unseen geometries. [arXiv 2503.05110](https://arxiv.org/abs/2503.05110)
- FlexIO (preprint, Oct 2025): variable microphones (1-5) and speakers (1-3), with an "array-agnostic channel communication mechanism" that builds on TAC. On real CHiME-4 recordings, performance still improved from 4 to 5 microphones although it never saw 5-channel training data. [arXiv 2510.21485](https://arxiv.org/abs/2510.21485)

**SpatialNet (Quan & Li, preprint v2 Dec 2023; journal status not checked) is not channel-agnostic.**
- Each T-F bin's input is the real and imaginary parts of all M mic STFTs, a 2M-dim vector. The paper calls the network "microphone-array-dependent", and separate models are trained per array.
- Its narrow-band blocks process each frequency independently, with multi-head self-attention over time frames, weights shared across frequencies. They are designed to "collect/separate components of the same/different speakers by computing similarity of spatial vectors."
- Attention maps show "the narrow-band MHSA module is indeed performing speaker (and non-speech) clustering based on narrow-band spatial features."
- Cross-band blocks: frequency convolution plus a full-band linear layer.
- Sizes: SpatialNet-small (L=8, C=96) has 1.2-1.6 M params and 23-46 GFLOP/s; large has 6.5-7.3 M params and 119-238 GFLOP/s, measured for a 6-ch, 4-s, 2-speaker input. Narrow-band modules dominate the FLOPs.
- SMS-WSJ 6-ch SI-SDR: large 25.1 dB, TF-GridNet 22.8 dB.
- Source: [arXiv 2307.16516 (HTML v2)](https://arxiv.org/html/2307.16516v2)

**SpatialNet when spatial cues vanish.** With both speakers at the same position, 6-ch SpatialNet-small drops from SDR 15.1 dB to 1.8 dB, and "the two speakers almost cannot be separated." [arXiv 2307.16516](https://arxiv.org/html/2307.16516v2)

### Inferences
- TAC's mean-pool-then-broadcast is exactly the operation needed for "every electrode branch sees a global context" at a cost linear in channel count. VarArray shows the variant in which the pooled stream is an INPUT feature to every channel, which is close to the project's "summed stream + per-electrode stream" idea.
- In VarArray the pooled stream is the complex average spectrum. For fish, the incoherent power sum (current wavetracker) is the right pooled analogue. A complex sum can cancel, because dipole fields give opposite-signed amplitudes on electrodes on either side of the fish's zero-potential plane. This is my physics judgement, not sourced here.
- Concatenating per-channel features by index is worse than pooling even on a fixed array (TAC paper: 6.9 vs 8.7 dB). Early concatenation would also break count independence. Both arguments favour set-style fusion over stacking electrodes as input channels.
- Both TAC (<15° angle) and SpatialNet (same position) show that spatial fusion helps least when sources share a spatial signature. For fish this means two nearby fish with similar orientation, which is precisely when the per-electrode branch would be asked to separate them. In that case only temporal cues (beats, continuity) remain. This is my judgement.
- Of the speech models, only TAC-family designs (FaSNet-TAC, VarArray, FlexIO) and attention-over-channels designs are count-independent by construction. SpatialNet-style per-bin spatial clustering is conceptually the most relevant to "two fish in one bin". Its 2M-dim input would have to be replaced by a set encoder (e.g. TAC over electrodes inside each narrow-band block) to be count-independent.

### Gaps
- I did not obtain layer-level details or numbers for iFaSNet, "channel-attention Conformers", or Chang et al.'s CCA (the query/key/value assignment is unverified).
- I did not confirm UniArray's or FlexIO's venues or their quantitative results.
- No study found compares TAC's mean pooling with attention pooling inside the same separator on large arrays (16-64 channels). The largest count-independent speech results found are VarArray (≤8 mics) and sparsemax stream attention (>30 channels, ASR only).

## 2. Set-based learning (DeepSets, Set Transformer, GNNs over sensor positions) and when mean/max/attention pooling washes out a weak channel

### Takeaway
All count-independent fusion is a variant of "shared per-element encoder + symmetric aggregation", either sum/mean/max (DeepSets, TAC) or attention (Set Transformer PMA/ISAB, Perceiver-style learned queries). Theory says sum-decomposition needs a latent width of at least the set size for universal representation. Empirically, mean pooling dilutes channels that carry signal alone, and learned selective pooling (sparsemax) helps strongly on large arrays. Known geometry should be fed as position encodings; PopT shows dropping them costs 0.08-0.12 AUC.

### Cited Findings
- DeepSets (Zaheer et al., NeurIPS 2017): permutation-invariant functions have the form ρ(Σ φ(x)); equivariant layers combine an element-wise term with a pooled term. (not re-fetched) — [arXiv 1703.06114](https://arxiv.org/abs/1703.06114)
- Wagstaff et al., ICML 2019, "On the limitations of representing functions on sets": for continuous sum-decomposition, a latent dimension of at least the maximum set size N is necessary and sufficient for universal representation (scalar-feature case). Zaheer et al.'s fixed-width argument relies on "highly discontinuous mappings." A 2025 survey reports later bounds of roughly N(D+1) ≤ L for D-dimensional features. [arXiv 1901.09006](https://arxiv.org/abs/1901.09006); survey [arXiv 2501.14991](https://arxiv.org/abs/2501.14991)
- Set Transformer (Lee et al., ICML 2019):
  - Attention encoder and decoder; inducing-point attention (ISAB) "reduces the computation time of self-attention from quadratic to linear in the number of elements."
  - Pooling is done by multi-head attention with learned seed vectors (PMA).
  - The MAB/SAB/ISAB/PMA internals are as I know them from the paper; the abstract page confirms only the linear-complexity claim.
  - Source: [arXiv 1810.00825](https://arxiv.org/abs/1810.00825)
- PNA (Corso et al., NeurIPS 2020): combining several aggregators (mean, max, min, std) with degree scalers outperforms any single aggregator in GNNs. This argues for multi-aggregator pooling over electrodes. (not re-fetched) — [arXiv 2004.05718](https://arxiv.org/abs/2004.05718)
- Perceiver (Jaegle et al., ICML 2021): cross-attention from a small learned latent array to a large input, so cost is linear in input size. (not re-fetched) — [arXiv 2103.03206](https://arxiv.org/abs/2103.03206)
- LUNA (NeurIPS 2025 camera-ready per arXiv comments):
  - "compresses multi-channel EEG into a fixed-size, topology-agnostic latent space via learned queries and cross-attention"; later blocks operate only on the latent, so cost scales "linearly — not quadratically — with channel count."
  - Claims "reducing FLOPs by 300x and trimming GPU memory use by up to 10x" and 0.921 AUROC on TUAR, "consistent across all evaluated electrode configurations."
  - Source: [arXiv 2510.22257](https://arxiv.org/abs/2510.22257); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/115483)
- Selective vs averaging pooling: sparsemax stream attention cut WER by more than 30% (simulated) and more than 20% (semi-real) relative to softmax pooling on arrays of more than 30 channels. Plain sparsemax "punishes the weights of many channels to zero harshly", hence the scaled version. [arXiv 2103.15305](https://arxiv.org/abs/2103.15305)
- Pairwise processing plus mean pooling is a common route to array independence in localisation, e.g. IPDnet (2024 preprint) and the DCASE 2019 pairwise SELD CRNN. [IPDnet arXiv 2405.07021](https://arxiv.org/abs/2405.07021); [arXiv 1910.10049](https://arxiv.org/abs/1910.10049)
- GNNs for distributed microphone networks handle a variable number of input channels (Grinstein et al., ICASSP 2023). [arXiv 2306.16081](https://arxiv.org/abs/2306.16081)
- Position encodings matter: in PopT (ICLR 2025), removing electrode position encoding dropped ROC-AUC on the four iEEG tasks from 0.74/0.87/0.90/0.93 to 0.62/0.76/0.81/0.83. [arXiv 2406.03044 (HTML v4)](https://arxiv.org/html/2406.03044v4)

### Inferences
- Pooling washout for fish, back-of-envelope (my derivation, not from a source). Suppose a faint fish adds power s on only k of N electrodes, and each electrode's noise power bin has mean and std of about n. An incoherent sum has a detection deflection of about k·s / (√N·n), versus s/n on the best electrode.
  - The summed spectrogram therefore beats the best single electrode only if k > √N.
  - For a fish seen on 2 of 64 electrodes, the sum is about 6 dB worse than the best electrode; for 2 of 16 it is about 3 dB worse.
  - Mean pooling of learned per-electrode features inherits the same dilution. Max pooling keeps the best-channel SNR, at a modest false-alarm cost that grows with N.
  - This suggests (a) mean+max (PNA-style) or attention/sparsemax pooling instead of mean only, and (b) faint fish near one electrode may be better found by the per-electrode branch than by the summed stream. The user's design assumes the summed stream finds all ridges, so this is a caveat to test.
- The Wagstaff bound is an expressivity limit for exact representation. For a salience detector, the practical lesson is a pooled width D comfortably larger than the electrode count (e.g. D ≥ 64 for 64 electrodes), or pooling by attention with several seeds (PMA/LUNA queries).
- All listed grids (3x3, 5x3, 1x11, 4x4, 8x8) are regular rectangular lattices. A small 2-D convolution over the electrode lattice, followed by global pooling, is count-independent and exploits spatial smoothness of the dipole pattern; it needs masking for dead electrodes and a spacing input. A GNN over a k-NN graph of electrode positions generalises this to irregular layouts. Learned per-electrode ID embeddings (BIOT/LaBraM style) do not transfer across setups. Position encodings in physical units (cm) relative to the grid centroid, or relative-position terms in attention, are the transferable choice. This is my judgement.

### Gaps
- No empirical study found that directly measures weak-channel washout under mean vs max vs attention pooling for sensor arrays with spatially localised sources. The k/√N argument above is my estimate, not a published result.
- I did not verify the MAB/ISAB/PMA equations from the full Set Transformer text this session.

## 3. EEG/iEEG foundation models, SELD with different array formats, seismic/radio array learning

### Takeaway
Neural-signal foundation models have converged on a shared per-channel temporal encoder plus a transformer (or learned-query cross-attention) over channel tokens with position/ID embeddings. PopT and LUNA are the cleanest count- and layout-agnostic designs: PopT has 3-D coordinate encodings over a frozen per-channel encoder, and LUNA compresses to a fixed latent so cost is linear in channel count. SELD is mostly fixed-format (4-ch FOA/MIC, now stereo) and uses channel-swap augmentation rather than set architectures. Seismology has mature GNNs over variable, time-varying station sets (GENIE).

### Cited Findings
- **PopT, Population Transformer (Chau et al., ICLR 2025).** [arXiv 2406.03044 v4](https://arxiv.org/html/2406.03044v4)
  - Each channel's embedding comes from a frozen pretrained temporal encoder (default BrainBERT; also TOTEM, Chronos, TS2Vec). A sinusoidal position embedding of the electrode's 3-D coordinates (plus ensemble membership) is added. The tokens go through a 6-layer, 8-head transformer with d=512 and a [CLS] token.
  - Pretraining: ensemble-wise discrimination (were two disjoint channel subsets 500 ms apart?) plus channel-wise discrimination (10% of channels swapped with activity from another time). Reconstruction-only pretraining was much worse (0.56/0.65/0.73/0.74 vs 0.74/0.87/0.90/0.93 ROC-AUC).
  - Size and cost: about 20 M trainable parameters; 500 k pretraining steps in about 2 days on one TITAN RTX (24 GB). For comparison, Brant has 500 M params, LaBraM 350 M and BIOT 3 M.
  - Results: on TUAB, PopT reached 0.888 ROC-AUC, vs BIOT 0.8815 and LaBraM 0.9162.
- **BIOT (Yang et al.).**
  - Tokenises each channel into fixed-length segments and adds channel embeddings and relative position embeddings, enabling learning across datasets with mismatched channels, variable lengths and missing values.
  - Results: 3% balanced-accuracy gain over baselines on CHB-MIT, and up to 4% more with pretraining.
  - The arXiv page lists no venue; I believe it appeared at NeurIPS 2023 but did not verify.
  - Source: [arXiv 2305.10351](https://arxiv.org/abs/2305.10351)
- **LaBraM, CBraMod and DIVER-0.** A secondary overview states that LaBraM and CBraMod handle variable channels through positional encodings or patch tokenization but do not enforce permutation equivariance. The HEAR authors note LaBraM's spatial encoding is tied to a fixed channel count. DIVER-0 (ICML 2025 workshop) is described as channel-equivariant (original not checked). [overview](https://aether-sutd.github.io/bfms/brain-fm/sensor-harmonization/channel-topology); [HEAR arXiv 2510.12515](https://arxiv.org/html/2510.12515v1)
- **HEAR** (preprint, Oct 2025): pretrained on 8,782 h from more than 150 electrode layouts, with claimed generalisation to unseen layouts. [arXiv 2510.12515](https://arxiv.org/html/2510.12515v1)
- **EEG-X** (preprint, Nov 2025): location-based channel embedding for different channel numbers and combinations. [link](https://www.emergentmind.com/papers/2511.08861)
- **LUNA**: learned-query cross-attention to a fixed latent; see Section 2. [arXiv 2510.22257](https://arxiv.org/abs/2510.22257)
- **SELD channel swapping.** Audio channel swapping (ACS) permutes channels according to the symmetry of the tetrahedral/spherical array to create new DOA labels, and is widely used in DCASE SELD. In 2025 the task moved to stereo, where ACS reduces to a left/right swap. STARSS23 has only 7 h 22 min of real recordings. [ACS paper arXiv 2101.02919](https://arxiv.org/abs/2101.02919); [DCASE2024 Du report](https://dcase.community/documents/challenge2024/technical_reports/DCASE2024_Du_83_t3.pdf)
- **SALSA-Lite (SELD features).** Using SALSA-Lite raised localisation-dependent F1 and class-dependent localisation recall by 15% and 5% over multichannel log-mel plus GCC features. The search snippet does not establish it as array-agnostic. [arXiv 2111.08192](https://arxiv.org/abs/2111.08192)
- **Seismology, GENIE (McBrearty & Beroza, preprint 2022).** A GNN that uses one graph over stations and one over the source region. It handles "arbitrary geometries and time-varying networks of hundreds of stations." On 100 days of Northern California data it detected about 4.2× as many events as the USGS catalogue. [arXiv 2209.07086](https://arxiv.org/abs/2209.07086)
- **Seismology, location and magnitude GNN (same authors, 2022).** Trained so each input can be a different network geometry. [arXiv 2203.05144](https://arxiv.org/abs/2203.05144)

### Inferences
- PopT's recipe maps directly onto the fish problem: a shared per-electrode spectro-temporal encoder (frozen or jointly trained), position encodings of electrode coordinates, and a small transformer over electrode tokens. Its discrimination-style pretraining (detect swapped channels) also suggests a cheap self-supervised task on unlabelled grid recordings, e.g. "is this electrode's patch from this time or another?". This is my judgement.
- ACS shows that geometry-consistent channel permutations are an effective label-preserving augmentation. For rectangular electrode grids, flips and 180° rotations of the lattice (and 90° rotations on square grids) are the analogue, applied consistently to positions.
- GENIE's two-graph design (sensor graph plus a "source" graph) is analogous to "electrode set" plus "frequency-time candidate set". It is useful if the detector later regresses fish position as an auxiliary output.

### Gaps
- No learned array-processing work in radio astronomy (e.g. variable antenna counts) was found within my search budget.
- I did not verify LaBraM's or Brant's tokenisation details from the primary papers, nor BIOT's venue.
- I found no SELD system that is explicitly trained across different microphone counts and evaluated as array-agnostic (IPDnet and the GNNs are localisation, not SELD).

## 4. Fusing a pooled "sum" stream with per-channel streams (two-stream, cross-attention, early vs late, gating, keeping the per-electrode branch cheap)

### Takeaway
There is direct precedent for each ingredient. VarArray feeds the pooled spectrum into every channel and collapses the channel axis early. FaSNet uses a reference channel and cross-channel features. Stream attention with sparsemax gates channels. LUNA and Perceiver use learned-query cross-attention so the per-channel set never interacts quadratically. I found no published detector that evaluates a per-channel branch only at candidate locations proposed by a pooled stream; that cascade is my proposal. For the fish problem it is the natural way to keep 16-64 electrodes affordable.

### Cited Findings
- VarArray uses pooled-stream-relative features and then early merging. Per channel: |average spectrum|² plus the IPD of each channel relative to the average spectrum. Three per-channel conformer blocks with TAC; the merge to one stream after block 3 "significantly reduces the network computation cost." [arXiv 2110.05745](https://arxiv.org/abs/2110.05745)
- FaSNet two-stage vs single-stage: the original two-stage FaSNet (pre-separation on a reference mic, then filters for the others) "cannot perform reliable beamforming at the second stage in the ad-hoc array configuration." A single-stage joint estimate with TAC was far better (11.7 vs 7.3 dB at 6 mics). [arXiv 1910.14104](https://arxiv.org/abs/1910.14104)
- Cross-channel attention: CSA + CCA layers inside a transformer ASR beat beamformer-then-transformer cascades on far-field data (numbers not obtained). [arXiv 2102.03951](https://arxiv.org/abs/2102.03951)
- Gated / selective fusion: scaling-sparsemax stream attention gives more than 30% / 20% relative WER gains on arrays of more than 30 channels. [arXiv 2103.15305](https://arxiv.org/abs/2103.15305)
- Learned queries over channels: LUNA claims 300× fewer FLOPs and up to 10× less memory by cross-attending learned queries to channels, then working on the fixed latent. [arXiv 2510.22257](https://arxiv.org/abs/2510.22257)
- Narrow-band (per-frequency) spatial processing is where the compute goes: in SpatialNet the narrow-band modules account for 11.0 and 10.1 GFLOP/s vs 1.5 and 0.3 GFLOP/s for the cross-band modules (small model ablation). [arXiv 2307.16516](https://arxiv.org/html/2307.16516v2)

### Inferences

#### Candidate design (my judgement)
1. **Ridge stream (always on, full resolution).** A deep-salience 2-D CNN/U-Net on the electrode-summed power spectrogram, normalised per recording. Its input and output are independent of the electrode count by construction. It produces a per-bin salience map and a candidate mask, e.g. salience above a low threshold dilated by ±3-5 Hz.
2. **Electrode stream (only where needed).**
   - A shared per-electrode encoder is applied to crops around candidates. Its inputs are each electrode's log power, its log ratio to the summed spectrum (VarArray-style "relative to pooled stream", which cancels source power and leaves the spatial pattern), and optionally signed or complex cross-spectral terms (Section 5).
   - Each crop should span roughly ±16 bins (±10 Hz) and ≥ 16-32 frames (≥ 2.5-5 s) at full frame rate. That temporal context matters because the within-bin beat between two fish 0.3-3 Hz apart lives at 0.3-3 Hz, close to the 3.125 Hz Nyquist rate of 0.16 s frames (Section 5). A lower time resolution in this branch would alias exactly the cue needed. The electrode branch can still use LOWER frequency resolution or a narrower band.
3. **Fusion.**
   - Inside the electrode branch, use TAC or channel self-attention between encoder layers so every electrode sees global context, with position encodings of electrode (x, y) in physical units.
   - Then fuse into the ridge stream by cross-attention, with the summed-stream feature at each (t, f) as query and the electrode tokens at the same (t, f) neighbourhood as keys/values.
   - Alternatively, pool with mean+max+attention (PNA/PMA) and add the result through a learned gate initialised near zero, so the network starts as "summed stream only" and learns to use electrodes where they help.
   - This is mid/late fusion. Early fusion (stacking electrodes as input channels) is excluded because it is count-dependent and, per TAC, worse.
4. **Outputs.** Besides salience, an auxiliary "number of sources in this bin" (0/1/2+) or a two-peak salience target trained on synthetic close-pair mixtures gives the electrode branch an explicit job. A per-candidate electrode amplitude vector could also be emitted directly for the tracker. The tracker already uses amplitude vectors, so emitting them from the detector aligns the two stages.
5. **Why this split suits the user's plan.** The summed stream keeps traces gap-free and cheap (one 2-D map regardless of N). The electrode branch costs O(N × candidate area), not O(N × full spectrogram), and candidate area is small: 30 fish × a few harmonics × a few bins out of thousands.

#### Other points
- Risk: candidate gating means fish missed by the summed stream (faint, single-electrode fish; see the k/√N estimate in Section 2) can never be recovered by the electrode branch.
- Mitigation: also propose candidates from a cheap per-electrode max-pooled spectrogram, i.e. the max over electrodes per bin. It costs one extra 2-D map and is count-independent.

### Gaps
- No published two-stream detector (pooled stream proposes, per-channel stream refines) for multichannel spectrograms was found. Evidence for the cascade is by analogy (region-proposal detectors; VarArray's early merge), not direct.
- No ablation found that compares cross-attention fusion with gated additive fusion for pooled+per-channel streams.

## 5. Inter-channel information beyond magnitudes: cross-spectral phase, coherence, spatial covariance — can it separate sources closer than the frequency resolution?

### Takeaway
In speech, adding IPD or spatial-covariance-like features to spectral inputs reliably helps, and narrow-band attention over spatial vectors performs per-frequency source clustering (SpatialNet). That is direct evidence that sources in the SAME frequency bin can be separated when their spatial signatures differ. The benefit collapses when spatial signatures coincide (same position: 15.1 → 1.8 dB SDR). I found no learned-model study of two sinusoids closer than one bin separated by spatial features. For quasi-static electric fields, the physics suggests the per-bin spatial covariance of two fish is rank 2 and their beat modulation is electrode-specific, so separation is in principle possible from frame sequences (my inference).

### Cited Findings
- Multi-channel deep clustering (Wang, Le Roux, Hershey, ICASSP 2018): "simply encoding inter-microphone phase patterns as additional input features during deep clustering provides a significant improvement in separation performance, even with random microphone array geometry." [MERL TR2018-007](https://merl.com/publications/TR2018-007)
- IPD encoding as cos/sin of the phase difference is standard. [arXiv 1910.10912](https://arxiv.org/abs/1910.10912)
- Multi-channel separation "drastically degrades when the speakers locate close to each other… spatial and directional features become less discriminative." A binaural real-time model was better "except when the speaker angle is small (less than 15°)." [arXiv 2002.01626](https://arxiv.org/abs/2002.01626); [arXiv 2002.06637](https://arxiv.org/abs/2002.06637)
- SpatialNet: narrow-band (single-frequency) MHSA clusters speakers by spatial vectors; with co-located speakers SDR falls from 15.1 to 1.8 dB. [arXiv 2307.16516](https://arxiv.org/html/2307.16516v2)
- VarArray: normalised IPD relative to the average spectrum beat raw STFT-coefficient inputs in preliminary experiments. [arXiv 2110.05745](https://arxiv.org/abs/2110.05745)
- Covariance-input DNNs for DOA: several works map the spatial covariance matrix to DOA or a spatial spectrum. One notes that much prior work trained only with large angular separations and equal SNRs; (SP)²-Net (preprint 2025) reports its largest advantage for closely spaced sources. These results are about angular, not spectral, resolution. [Liu 2020, Sci. Rep. (PMC7670414)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7670414); [(SP)²-Net arXiv 2509.15475](https://arxiv.org/html/2509.15475v1)

### Inferences
These points are physics and design judgement, not from sources.

- **Quasi-static field.** At EOD frequencies in water, the field is effectively quasi-static: no propagation delay across a grid, and the wavelength is far larger than the array. A single fish's complex STFT value at electrode m is therefore a REAL, signed amplitude a_m times a common phase. Unlike acoustics, the inter-electrode phase of one fish is only 0 or π, i.e. a sign pattern. IPD features therefore reduce to sign-agreement features; the dipole's sign pattern is lost in power spectrograms.
- **Two fish in one bin (Δf below the bin spacing or main-lobe width).**
  - The per-electrode bin value is X_m(t) = a_m·e^{iφ1(t)} + b_m·e^{iφ2(t)}.
  - The per-bin spatial covariance accumulated over ≥ 1/Δf seconds is approximately a aᵀP1 + b bᵀP2, rank 2 if a and b are not collinear. An eigen-decomposition (MUSIC/PCA-like) recovers both spatial patterns.
  - Even magnitude-only per-electrode spectrograms carry this information: |X_m|² = a_m² + b_m² + 2·a_m·b_m·cos(2πΔf·t + φ0). Each electrode shows a beat at Δf with an electrode-specific baseline and modulation depth, signed by a_m·b_m, so the beat pattern over electrodes and frames identifies the two fish.
  - Beats are sampled at 6.25 frames/s, so Δf up to about 3 Hz is observable without aliasing. This covers the "within 3 Hz → one peak" failure mode.
- **Design consequences.**
  - (a) Keep complex STFTs (or at least signed cross-spectra relative to a reference/sum) available to the electrode branch: storage doubles but the sign pattern is gained.
  - (b) Give the electrode branch temporal context of ≥ 1/Δf_min frames, e.g. ≥ 3-5 s for Δf ≥ 0.3 Hz.
  - (c) A per-bin channel-covariance feature (N×N) is count-dependent. Count-independent alternatives are per-channel features relative to the pooled stream (VarArray style), TAC/attention over electrode tokens (which implicitly computes covariances), or the top-k eigenvalues/eigenvectors of the per-bin covariance. Eigenvalues are count-independent scalars, e.g. λ2/λ1 as a "two-source in this bin" cue.
- **Where this fails.** Fish whose amplitude patterns are nearly collinear (same position and orientation, or both far from the grid), mirroring SpatialNet's co-located failure. Here only frequency/temporal continuity (the tracker) can help.

### Gaps
- No published learned model was found that separates two sinusoids closer than the STFT resolution using multichannel spatial cues. The rank-2 and beat argument above is my analysis and should be tested on synthetic grid mixtures.
- No electric-fish-specific evidence on inter-electrode phase was gathered here; other researchers on the project may cover it.

## 6. Normalisation across electrodes and recordings, and robustness to dead/noisy electrodes (channel dropout)

### Takeaway
Published count-agnostic systems train on random channel subsets: VarArray uses 3-7 mics per batch, PopT swaps channels as a pretext task, and channel ablation is reported in biosignals. They normalise features per window or per channel, and relative features (IPD to the average spectrum) remove common gain. Random channel ablation gave double-digit gains under missing channels in biosignal classification (preprint).

### Cited Findings
- VarArray samples 3-7 microphones at random per mini-batch "to expose the model to a variety of array geometries and inter-microphone spacing patterns". Magnitude and IPD features are normalised at the window level. [arXiv 2110.05745](https://arxiv.org/abs/2110.05745)
- Random Channel Ablation (preprint 2024, multimodal biosignal gesture classification): 12.2% and 24.5% improvement with up to 4 and 8 missing channels vs baseline. [arXiv 2407.10874](https://arxiv.org/html/2407.10874v1)
- EEG augmentation comparison: channel dropout (random zeroing) prevents over-reliance on single channels and improves robustness to corrupted channels. [arXiv 2206.14483](https://arxiv.org/abs/2206.14483)
- HD-sEMG: dropout augmentation over many masks improved performance and reduced overfitting vs a single mask. [arXiv 2309.11086](https://arxiv.org/abs/2309.11086)
- PopT's channel-wise pretext task replaces 10% of channels with activity from another time and trains detection of the swapped tokens, i.e. explicit training to detect anomalous channels. [arXiv 2406.03044](https://arxiv.org/html/2406.03044v4)
- A 2026 robustness study argues EEG models should be tested for both missing channels and channel-order mismatch. [Essex repository](https://repository.essex.ac.uk/43231/)
- PCEN (Wang et al., ICASSP 2017), per-channel energy normalisation, is a learnable automatic gain control on spectrograms used for far-field robustness. (not re-fetched) — [arXiv 1607.05666](https://arxiv.org/abs/1607.05666)

### Inferences
These are my judgement.

- **Normalise each electrode by its own noise floor.** For example, divide by the median over frequency (or over a long time window) per electrode, then take the log. This removes gain differences between electrodes and recordings while keeping relative fish amplitude patterns.
- **Feed log(electrode / sum) as a second feature.** It is a spatial-pattern feature, invariant to source power.
- **Avoid per-electrode z-scoring after the spatial ratio,** since that would erase the amplitude pattern the tracker relies on.
- **Augment:**
  - random electrode subsets, e.g. keep 2..N with geometry intact;
  - random electrode dropout, zeroed and masked;
  - "noisy electrode" injection (broadband noise, 50/60 Hz harmonics, clipping) on 1-2 electrodes;
  - random per-electrode gain of ±10-20 dB;
  - lattice flips/rotations, applied consistently to positions.
- **Use an explicit validity mask** in TAC/attention (exclude dead electrodes from the mean and the softmax) rather than zeros. A zeroed electrode biases a mean pool.
- **Recompute the summed stream from valid electrodes only.** Otherwise a noisy electrode contaminates the ridge stream too.

### Gaps
- No study was found that quantifies channel-dropout benefits for count-agnostic separators specifically (VarArray uses subset sampling but reports no ablation).
- PCEN's behaviour on very-low-frequency-resolution (0.61 Hz) spectrograms is not known.

## 7. Compute and memory for 16-64 channels on a 16 GB GPU with hours-long inputs

### Takeaway
Full-resolution per-electrode processing of whole recordings is memory-infeasible at 64 channels. My estimate is ~24 GB of float32 input per hour before any activations. Published count-agnostic designs all control cost by (a) linear-in-N cross-channel ops (TAC means, inducing points, learned queries), (b) merging the channel axis early (VarArray), and (c) chunked/windowed processing. Candidate-gated electrode processing reduces cost further by the ratio of candidate area to spectrogram area.

### Cited Findings
- TAC adds three FC layers per block and a mean, so cost is linear in N. FaSNet-TAC is ~2.9 M parameters. [arXiv 1910.14104](https://arxiv.org/abs/1910.14104)
- VarArray merges channels after 3 of 5 conformer blocks "to reduce the growth rate of the model evaluation cost with respect to the number of microphones". It processes in sliding windows (1.6 s window, 0.4 s shift) with stitching. [arXiv 2110.05745](https://arxiv.org/abs/2110.05745)
- ISAB inducing points: attention cost becomes linear in set size. [arXiv 1810.00825](https://arxiv.org/abs/1810.00825)
- LUNA: 300× FLOP and up to 10× memory reductions claimed from compressing channels to a fixed latent. [arXiv 2510.22257](https://arxiv.org/abs/2510.22257)
- SpatialNet-small: 1.2-1.6 M params and 23-46 GFLOP per second of 8/16 kHz audio (6 ch). Narrow-band per-frequency modules dominate. [arXiv 2307.16516](https://arxiv.org/html/2307.16516v2)
- PopT: about 20 M params, pretrained on one 24 GB TITAN RTX in about 2 days. [arXiv 2406.03044](https://arxiv.org/html/2406.03044v4)

### Inferences
These are my estimates, assuming 0.61 Hz bins, 0.16 s frames and a 0-2.5 kHz input band.

- **Input size.**
  - About 4,100 bins × 22,500 frames per hour ≈ 92 M values per electrode-hour ≈ 0.37 GB float32.
  - 16 electrodes ≈ 5.9 GB/h; 64 electrodes ≈ 24 GB/h. Complex STFT doubles this.
  - The full per-electrode input must therefore stream from host RAM or disk in chunks, and even host RAM (62 GB) holds only about 2.5 h of 64-ch power spectrograms.
- **Activation memory.** A per-electrode encoder with 32 feature maps over a 10 s chunk (62 frames × 4,100 bins × 16 electrodes) needs about 0.5 GB per layer in forward pass alone. Training at full band is impractical, so training should use crops, e.g. 256 bins × 64 frames.
- **Throughput.**
  - The TF-bin rate is 6.25 × 4,100 ≈ 26 k bins/s, about 1.6× SpatialNet's 16 kHz rate (62.5 × 257 ≈ 16 k).
  - A SpatialNet-small-scale network over the summed stream would cost roughly 75 GFLOP per second of recording, about 270 TFLOP per hour of recording.
  - That is tens of seconds per recording hour on a GPU with a peak on the order of 50 TFLOPS FP32. The RTX 4080 spec is roughly in this range; not verified here.
  - Running the same network on every electrode multiplies this by N (64× → tens of minutes per hour). This is the main quantitative argument for candidate gating.
- **Gating payoff.** With 30 fish × ~3 harmonics × ±16 bins, candidates cover about 3,000 of 4,100 bins in the worst case but typically far fewer in the fundamental band. If the electrode branch runs only on the EODf band (e.g. a 600-1,000 Hz window for a given species mix) and only on candidate crops, its cost falls by one to two orders of magnitude. Exact savings depend on data.
- **Pairwise features scale badly.** N(N−1)/2 pairs = 120 for 16 electrodes and 2,016 for 64, so per-pair cross-spectra should be avoided in favour of per-electrode features relative to the pooled stream (N features) or low-rank summaries.
- **Attention over electrodes.** Full self-attention across 64 electrode tokens costs 64² = 4,096 per (t, f) location. That is affordable only on candidate crops; TAC (linear) or ~4-8 learned queries (LUNA/PMA) are cheaper defaults.

### Gaps
- No published measurement was found of memory or throughput for count-agnostic models at 64 channels on hour-long spectrograms. All figures above are my estimates and should be benchmarked on the RTX 4080.
