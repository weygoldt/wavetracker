# Stack harmonics, fuse electrodes, keep the tracker

The best-supported learned design for wavetracker is a deep-salience-style network that keeps the full 0.61 Hz resolution. It reads the electrode-summed spectrogram through a harmonic stack rebuilt for the linear frequency axis, integrates evidence along time over tens of seconds, and adds a per-electrode branch that does not depend on electrode count (a shared encoder with mean and max pooling, the "TAC" pattern from ad-hoc microphone arrays) to separate fish that share a spectral peak. It outputs a per-bin fundamental salience, a sub-bin frequency offset, and auxiliary "harmonic or stray line" and "two sources here" scores. It hands scored detections to the existing tracker and linker, which keep the job of identity. No learned EOD-frequency detector has been published. The closest precedent is dolphin-whistle extraction: replacing heuristic peak picking with a learned confidence map in front of an unchanged graph tracker raised F1 from 0.634 to 0.884, mostly through precision (0.63 to 0.96) ([Li et al. 2020](https://arxiv.org/pdf/2005.08894)). Music multi-f0 research supplies the architecture but not the frequency grid. Its semitone-based bins are roughly 15 to 40 times coarser than fish at 500 to 2000 Hz need, so harmonic stacking has to be redone as an index gather on the linear axis. Multi-object tracking results for look-alike objects favour tracking-by-detection with score-aware association over end-to-end trackers and appearance embeddings. For faint fish and false lines, training data and labels matter more than model depth. Mixtures should be summed as complex per-electrode STFTs, so that beats between close fish survive. They should include real field noise at very low SNR, keep harmonics as negatives, and mask the benchmark's known truth defects out of the loss. The weakest link is the claim that the per-electrode branch will separate fish within 3 Hz. The physics supports it, but no published model has separated sources closer than one frequency bin using spatial cues, so a cheap diagnostic should test it before the branch is built.

## No EOD detector has been learned, but whistle trackers show the recipe works

Every published pipeline for wave-type fish uses the same three hand-built steps. First, find peaks in a per-electrode or electrode-summed power spectrum. Second, group peaks into a fundamental plus harmonics with rules. Third, link across time with a frequency gate; since Madhav et al. and the wavetracker paper, the linking also compares the power profile across electrodes ([Raab et al. 2022](https://www.frontiersin.org/articles/10.3389/fnint.2022.965211/full); [Madhav et al. 2018](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5895713/fullTextXML)). The grouping rules have changed little since the Darién field work. They use a **1 dB relative peak threshold, an absolute floor of 50 µV²/Hz, candidate fundamentals f_max/n for n ≤ 4, and rejection of any group with fewer than three harmonics or more than a quarter missing** ([Henninger et al. 2020](https://www.biorxiv.org/content/10.1101/550814v2.full)). Since then, all progress has been in linking. On 464 hard assignment conflicts, the spatial profile alone assigned **99.57% correctly, against 94.83% for frequency alone** ([Raab et al. 2022](https://www.frontiersin.org/articles/10.3389/fnint.2022.965211/full)). The only learned spectrogram models in this community are box detectors for chirps, and they need wavetracker's tracks as input ([chirpdetector](https://github.com/weygoldt/chirpdetector); [efishSignalDetector](https://whale.am28.uni-tuebingen.de/git/raab/efishSignalDetector/src/commit/3dbc78685d4c42392ecaa8daaefca7c4c8cdbc24)). The researchers found no public labelled benchmark of EOD frequency tracks, so the project's dense-mixture benchmark has no published counterpart.

The failure cases these papers report match the benchmark's four measured failure modes. For false lines (c), Henninger et al. note that "at times, a Sternopygus harmonic had power similar to an Apteronotus fundamental" ([Henninger et al. 2020](https://www.biorxiv.org/content/10.1101/550814v2.full)). For close lines (b), several groups report hard limits:

- Raab et al. excluded 3 of 45 pairings whose fish were less than 0.5 Hz apart, because their traces "could not be extracted" ([Raab et al. 2021](https://www.biorxiv.org/content/10.1101/2021.02.04.429572v1.full)).
- Dense groups at 670–672 Hz prevent reliable assignment ([Raab et al. 2022](https://www.frontiersin.org/articles/10.3389/fnint.2022.965211/full)).
- Field Eigenmannia "sometimes shared frequencies for tens of seconds" ([Madhav et al. 2018](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5895713/fullTextXML)).
- **142 of 1,119 surface Eigenmannia pairs had a mean frequency difference below 5 Hz, and such pairs were often less than 50 cm apart** ([Fortune et al. 2020](https://www.frontiersin.org/journals/integrative-neuroscience/articles/10.3389/fnint.2020.561524/full)). So the spatial cue is weakest exactly where it is most needed.

For faint fish (a), Henninger et al. name smeared peaks during fast rises and peaks near the noise floor. The researchers add a reading that has not been tested: the "at least three harmonics" rule may itself discard faint fish whose harmonics fall below threshold, and the f_max/n search accepts a strong harmonic of a low fish as a fundamental when that fish's lower harmonics are weak.

Odontocete whistle extraction is the closest working analogue. Li et al. trained a 10-layer, 32-channel residual CNN with **no pooling** to predict a per-bin "whistle confidence map", thresholded it at 0.5, and passed it to the unchanged silbido graph-search tracker. On 911 hard test whistles with many crossings, F1 rose from **0.634 (graph search alone) to 0.884 (P 0.956, R 0.822)**. Training only on synthetic contours blended into real whistle-free noise still reached 0.792 ([Li et al. 2020](https://arxiv.org/pdf/2005.08894)). Packaged as "silbido profundo", the method reached P 96.3 / R 79.7 / F1 87.2 and ran on 487 h of data at 5.5 times real time on one RTX 2080 Ti. Its authors credit the gain to the CNN considering "multiple peaks in context". Its remaining false positives were **narrowband "spectral lines" and runs of short narrowband energy that the gap-bridging tracker chained into long contours** ([Conant et al. 2022](https://research-repository.st-andrews.ac.uk/bitstream/10023/26799/1/Conant_2022_JSA_Silbido_profundo_CC.pdf)). Those are the whistle-world versions of mains combs and of wavetracker's long false identities. A follow-up showed that the CNN-plus-graph pipeline fragments less than a probabilistic multi-target filter: **1.13 segments per true whistle against 1.80 for SMC-PHD** ([Li et al. 2023, JASA](https://repository.library.noaa.gov/view/noaa/65889/noaa_65889_DS1.pdf)).

The lesson for wavetracker is that in the nearest field the gain came from choosing peaks using context, in front of a tracker that was left alone. The precision gain was as large as the recall gain. The project's own benchmark notes point the same way. Halving the bin width to 0.31 Hz added about five recall points but doubled false lines and left IDF1 unchanged. A hand-made faint-peak acceptor raised field IDF1 only from 0.613 to 0.649, and its false additions extended false lines. Adding peaks without context makes identities worse.

## Music salience transfers only once harmonics are stacked on the linear axis

Deep salience is the template. It is a five-layer, fully convolutional network with no pooling, "since we do not want to be invariant to small shifts in time frequency". It maps a six-channel **harmonic CQT (h = 0.5, 1, 2, 3, 4, 5)** to a per-bin sigmoid salience, trained with binary cross-entropy against Gaussian-blurred targets ([Bittner et al. 2017](https://brianmcfee.net/papers/ismir2017_salience.pdf)). The stacking is what lets a small CNN work. Each channel is a CQT whose lowest frequency is multiplied by h, so bin k holds the h-th harmonic of the same candidate f0 in every channel, and one local kernel sees the fundamental and its harmonics together. The sub-harmonic channel separates a fundamental from its first harmonic: a true f0 has little energy at f/2, while a harmonic does. Basic Pitch cut the design to **16,782 parameters** with seven harmonics plus one sub-harmonic. Its ablation found that removing harmonic stacking "substantially reduces performance across all metrics and datasets". Its limited frequency receptive field "removes the need for pitch shifting data augmentations" ([Bittner et al. 2022](https://arxiv.org/abs/2203.09893)). PESTO adds two lessons. It keeps frequency resolution "never reduced" so the model can "distinguish close but different" pitches. And mixing background music into training raised its accuracy at 0 dB **from 50% to about 81–83%** ([Riou et al. 2023](https://arxiv.org/abs/2309.02265)).

The music models also show what to expect. Deep salience had good precision but poor recall on inner voices. Its authors blamed training mixes with "few examples with simultaneous harmonically related pitch content". Its validation-tuned threshold was 0.3, while the best threshold on Bach10 was **0.05** ([Bittner et al. 2017](https://brianmcfee.net/papers/ismir2017_salience.pdf)). This bias toward dominant sources is failure mode (a) unless training mixtures contain many labelled faint lines. The threshold shift warns that one global cut-off will not carry over between lab and field. None of these models was tested near 30 simultaneous f0s; typical test polyphony is one to five. They all handle unknown polyphony the same way, with independent per-bin sigmoids followed by peak picking, so no count head is needed for 0 to 30+ fish. Note-level systems such as Onsets and Frames and MT3 quantise to semitones and emit note events, which do not fit continuous lines. MT3's instrument tokens give class identity, not individual identity ([Gardner et al. 2022](https://arxiv.org/abs/2111.03017v4)).

The frequency grid is what fails to carry over. The figures in this paragraph are the researchers' arithmetic:

- A 20-cent bin is 11.6 Hz wide at 1000 Hz and 23 Hz wide at 2000 Hz.
- Two fish 3 Hz apart at 1000 Hz differ by only 5.2 cents.
- Holding 0.61 Hz at 2000 Hz on a log grid needs about 2,270 bins per octave. That is about 11,500 bins over 60–2000 Hz, against 3,180 linear bins, and it oversamples 60 Hz about 34-fold.

The conclusion is to keep the linear STFT and rebuild the stacking for it. On a linear grid that starts at 0 Hz, harmonic h of bin k sits exactly at bin h·k. A "linear harmonic stack" is therefore a strided gather, H[h, t, k] = S[t, h·k], with no interpolation for integer h. Three adjustments are needed:

- **Pooling.** A fish's f0 is rarely on a bin centre, so harmonic h lands up to h/2 bins away from the gathered index. Each channel should max-pool about h bins around h·k before the stride. This is the same tolerance the classic harmonic-group search uses.
- **Sub-harmonics.** The f/2 and f/3 channels need interpolation.
- **Nyquist.** Channels whose h·f lies above Nyquist must carry an explicit validity mask. At 20 kHz, the h = 5 harmonic of a 2000 Hz fish already sits at Nyquist.

After stacking, shifting a fish by Δf moves its peak by the same number of bins in every channel. Convolutions over the stacked axis are then roughly shift-equivariant in f0, which no kernel on a plain linear spectrogram can be. Mains-hum combs stay at fixed absolute frequencies and do not move with the fish, which is itself a cue. These are the researchers' derivations. They found no published multi-f0 network that stacks harmonics on a linear axis at sub-Hz resolution.

Harmonic evidence is also the most direct tool against false lines. In passive sonar, Lampert's thesis reports that adding prior knowledge of the harmonic process "increases the detection rate by a factor of two" ([Lampert 2010](https://www.bmva-archive.org.uk/theses/2010/2010-lampert.pdf)). In gravitational-wave line searches, a CNN that compares the same band across detectors and classifies it as signal, instrumental line or noise made a fully automated search competitive ([Bayley et al. 2020](https://arxiv.org/abs/2007.08207)). For fish, the f/2 and f/3 channels let the network see that a candidate at 2·f_low has strong energy one octave down. An auxiliary head trained to mark "a line that is not a fundamental" turns that into an explicit output.

Targets and decoding must be re-specified in Hz:

- **Blur width.** A blur that "decays to zero within a quarter-tone" spans ±23 Hz at 800 Hz and would merge fish more than 10 Hz apart. The researchers recommend a Gaussian σ of about 0.6 Hz, defined in Hz.
- **Overlap.** Where fish overlap, merge targets by element-wise maximum, as CenterNet does, so two peaks stay separable ([Zhou et al. 2019](https://arxiv.org/pdf/1904.07850)).
- **Sub-bin frequency.** Recover it with a CenterNet-style offset head (L1 loss at truth bins only) or with a refinement over at most 3 bins. RMVPE's 9-bin weighted average spans 5.5 Hz at this resolution and would pull an estimate toward a neighbouring fish ([Wei et al. 2023](https://arxiv.org/html/2306.15412v1)).

Self-supervision is not a substitute for labels here. PESTO's transposition trick (shift the input, and require the output to shift by the same amount) holds on a linear axis only after harmonic stacking. The self-supervised multi-pitch model SS-MPE trails supervised models by roughly 10 to 25 accuracy points and false-alarms on harmonics ([Cwitkowitz & Duan 2024](https://arxiv.org/html/2402.15569v1)). Its ISMIR 2025 follow-up found that adding the self-supervised objectives to supervised training "simultaneously overfits to the supervised data while degenerating" on the self-supervised data ([Cwitkowitz & Duan 2025](https://arxiv.org/abs/2506.23371v1)). The benchmark can generate unlimited labelled mixtures, so supervised training is the default, and a shift-consistency loss is at most a regulariser.

## Seconds of temporal context recover faint lines; output resolution separates close ones

Deep salience's kernels cover about 15 frames, or **0.17 s at its 11 ms hop**, and its authors named recurrent layers as the next thing to explore ([Bittner et al. 2017, ISMIR archive](https://archives.ismir.net/ismir2017/paper/000085.pdf)). The cleanest ablation the researchers found is Onsets and Frames. Removing its BiLSTMs dropped frame F1 on MAPS from **78.30 to 67.60**, while forward-only (causal) LSTMs cost only about two points ([Hawthorne et al. 2018](https://arxiv.org/pdf/1710.11153)). HPPNet reports a smaller effect of about three points ([Wei et al. 2022](https://arxiv.org/pdf/2208.14339)). Other evidence points the same way:

- CRNNs (a CNN front end followed by a recurrent layer) are the standard for polyphonic sound-event detection ([Çakır et al. 2017](https://www.arxiv-vanity.com/papers/1702.06286/)).
- A Conformer system won DCASE 2020 Task 4, though not in a controlled comparison ([Miyazaki et al. 2020](https://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Miyazaki_108.pdf)).
- Dilated temporal convolutions match LSTMs and show longer effective memory ([Bai et al. 2018](https://arxiv.org/abs/1803.01271)).

Learned detectors of dim lines in passive-sonar spectrograms claim detection near −24 dB ([Han et al. 2019](https://arxiv.org/abs/1912.00605)). Those figures are time-domain SNRs on simulated noise, so they cannot be compared with the benchmark's per-bin "below 6 dB".

Kernel sizes counted in frames have no fixed physical meaning. The deep-salience stack on 0.16 s frames would span 2.4 s, so the receptive field has to be designed in seconds. The physics sets three scales:

- **Persistence.** Fish lines persist for seconds to hours. On the researchers' reasoning, evidence for a stationary faint line grows roughly with the square root of the number of frames integrated.
- **Crossings.** Lines drifting at 0.1–1 Hz/s overlap for seconds to tens of seconds at a crossing. Deciding which branch continues where needs context longer than the overlap, which points to a temporal module spanning 30–60 s.
- **Rises and chirps.** Rises reach **5–68 Hz (mean 17 Hz)** ([Raab et al. 2021](https://www.biorxiv.org/content/10.1101/2021.02.04.429572v1.full)), so the temporal module must not over-smooth in frequency. Rises must be in the training mixtures, and evaluation must report them separately. Small chirps last **under 20 ms** and female long chirps about **162 ms with about 400 Hz excursions** ([Henninger et al. 2018](https://www.biorxiv.org/content/10.1101/114249v2.full)). The 0.61 Hz bins imply an analysis window of about 1.64 s hopped every 0.16 s. A chirp therefore appears as a brief dip and smear in power, not as a resolved jump. The detector should not try to resolve chirps; the tracker should bridge them.

The lowest-risk module is a recurrent or dilated-convolution layer that runs along time separately for each frequency bin, with weights shared across bins. This is the "sub-band, across-time" pattern of the speech separator TF-GridNet ([Wang et al. 2023](https://arxiv.org/abs/2211.12433)). It gives 10 to 100 s of context cheaply.

A short explanation of attention, for readers who know CNNs well. **Attention is a learned, content-dependent weighted average.** Each element of a sequence or set emits three vectors: a query ("what I am looking for"), a key ("what I contain") and a value ("what I pass on"). The weight from element i to element j is a softmax over the similarity of i's query with every key. Element i's output is the weighted sum of the values. Unlike a convolution, which mixes a fixed neighbourhood with fixed weights, attention picks which elements to mix based on their content. Its cost grows with the square of the number of elements. A Conformer interleaves attention along time with depthwise convolution ([Gulati et al. 2020](https://arxiv.org/abs/2005.08100)). Attention along time is cheap here: a 60 s chunk is 375 frames. Attention across the full linear frequency axis (thousands of bins) is expensive, so frequency context should come from convolutions and harmonic stacking. State-space models such as Mamba scale linearly with sequence length. BioMamba matched a transformer on a bioacoustics benchmark with much less GPU memory ([Tang & Baskiyar 2025](https://arxiv.org/html/2512.03563v1)), but the researchers found no state-space model used for pitch or multi-f0.

Temporal context helps three of the four failure modes:

- **(a) Faint fish and (d) fragmentation:** directly. The detector can integrate faint lines and bridge short dropouts.
- **(c) False lines:** partly. A true harmonic co-modulates with its fundamental over a long window (during a rise, the k-th harmonic moves k times as far), and a stray line does not.
- **(b) Close lines:** not on its own. A salience map that cannot represent two peaks within 3 Hz merges them however much time it sees.

The gravitational-wave G2Net competition adds a sobering point. GPU template and dynamic-programming track statistics won at the lowest SNRs. The machine-learning entries were most valuable for robustness to real, non-Gaussian noise once line-like artefacts were injected into training: **the dynamic-programming track-finder's false-positive rate rose about tenfold on real noise, while a U-Net degraded negligibly** ([Tenorio et al. 2025](https://arxiv.org/html/2509.06445v2)). A Viterbi pass over the learned salience map is therefore a reasonable later addition, not a replacement for the network.

## Electrodes are the only cue that separates two fish inside one peak

The summed spectrogram has a blind spot that the researchers quantify with their own back-of-envelope estimate. Suppose a faint fish adds power s on only k of N electrodes, against noise n per electrode. The incoherent sum deflects by about k·s/(√N·n), against s/n on the best single electrode. **For a fish seen on 2 of 64 electrodes, the sum is about 6 dB worse than the best electrode; for 2 of 16, about 3 dB worse.** Mean pooling of learned per-electrode features dilutes the signal in the same way. Max pooling keeps the best-electrode SNR at a false-alarm cost that grows with N. The user's design assumes the summed stream finds every ridge. This estimate says faint fish near a single electrode are a test case for that assumption, and that a max-over-electrodes map belongs next to the sum.

Separating close fish rests on physics argued in the notes but not tested anywhere. At EOD frequencies the field in water is quasi-static: there is no propagation delay across the grid. A single fish's complex STFT value at electrode m is therefore a real, signed amplitude a_m times a common phase. Between electrodes, the phase of one fish is only 0 or π, a sign pattern that power spectrograms throw away. Two fish in the same bin give X_m(t) = a_m·e^{iφ1(t)} + b_m·e^{iφ2(t)}. Even in power only, each electrode then shows a beat at Δf, |X_m|² = a_m² + b_m² + 2·a_m·b_m·cos(2πΔf·t + φ0), with its own baseline and modulation depth. Accumulated over at least one beat period, the per-bin spatial covariance has rank 2 whenever the two amplitude patterns differ. Frames arrive at 6.25 per second, so beats up to about 3 Hz are visible without aliasing, which covers failure mode (b).

Speech separation supports the principle that sources in one frequency bin can be separated by spatial signature. In SpatialNet, attention within each narrow frequency band "is indeed performing speaker … clustering based on narrow-band spatial features". It also shows where the principle fails: **with both speakers at the same position, SDR falls from 15.1 to 1.8 dB** ([Quan & Li 2023](https://arxiv.org/html/2307.16516v2)). Encoding inter-microphone phase as an input "provides a significant improvement … even with random microphone array geometry" ([Wang et al. 2018](https://merl.com/publications/TR2018-007)). For fish, the co-location failure maps onto the Fortune result that low-difference pairs often swim within 50 cm of each other. In those cases only temporal continuity in the tracker remains.

The count-independent architectures all share one pattern. **A network with shared weights processes each electrode, and the only cross-electrode operation is symmetric** (mean, max, or attention).

TAC is the simplest version. It applies one fully connected layer to every channel, averages across channels, passes the average through a second layer, appends it back onto every channel's features, applies a third shared layer, and adds a residual. Because nothing in it is tied to a channel index, it works for 2 or 64 electrodes in any order. On ad-hoc arrays with 2, 4 and 6 microphones, adding TAC raised separation from **9.0/8.9/9.5 dB to 9.8/11.2/11.7 dB**, and the authors state that it "guarantees that more microphones will not make the performance worse". On a fixed array, concatenating cross-channel features by microphone index scored worse than averaging them (**6.9 vs 8.7 dB**) ([Luo et al. 2020](https://arxiv.org/abs/1910.14104)). That is evidence against stacking electrodes as input channels even when the count is fixed. Spatially close sources gained least from TAC: under 15° of separation, the TAC model gained 0.6 dB over plain FaSNet, against 2.0 dB above 90°.

VarArray is the closest published template for the user's plan. Every microphone's input is the **magnitude of the average spectrum plus that channel's phase relative to the average**. Three per-channel Conformer blocks with TAC between them run first, and the model then merges to one stream to save compute. Training uses 3–7 microphones picked at random per mini-batch. The geometry-agnostic model beat geometry-specific ones, for example 17.7% against 18.8% WER on 8-microphone AMI-dev ([Yoshioka et al. 2022](https://arxiv.org/abs/2110.05745)).

Three further results bear on large arrays:

- **Selective pooling.** Replacing softmax channel weighting with sparsemax, which sets noisy channels' weights to exactly zero, cut WER by **more than 30%** relative on arrays of more than 30 channels ([Chen & Zhang 2021](https://arxiv.org/abs/2103.15305)).
- **Pooled width.** For exact representation, sum-pooling needs a feature width at least as large as the set ([Wagstaff et al. 2019](https://arxiv.org/abs/1901.09006)). In practice the pooled width should comfortably exceed the electrode count.
- **Positions.** Electrode positions matter. Removing coordinate encodings in an iEEG population transformer dropped ROC-AUC from 0.74–0.93 to 0.62–0.83 ([Chau et al. 2025](https://arxiv.org/html/2406.03044v4)).

For the electrode set, the attention explained above takes three forms. In self-attention over electrodes, each electrode gathers information from the electrodes most relevant to it. In cross-attention, the summed stream at one time–frequency point asks the electrode set "which of you see this peak, and how?". With learned queries, a few trainable "questions" read a set of any size and always return the same number of answers. That makes learned queries count-independent, and their cost grows only linearly with electrode count ([Lee et al. 2019](https://arxiv.org/abs/1810.00825); [LUNA 2025](https://arxiv.org/abs/2510.22257)).

Compute sets the remaining constraints. The researchers estimate about 92 million values per electrode-hour (4,100 bins × 22,500 frames), which is 0.37 GB in float32. That makes **5.9 GB per hour at 16 electrodes and 24 GB per hour at 64**, before any activations. Running a SpatialNet-sized network on every electrode would multiply a summed-stream cost of tens of seconds per recording-hour by N. The researchers therefore propose two cost controls, neither of which has a published precedent:

- **Candidate gating.** Evaluate the electrode branch only where the summed stream proposes candidates, and also where the max-over-electrodes map does, so that single-electrode fish are not lost.
- **Early merge.** Merge the electrode axis early, as VarArray does.

Training needs random electrode subsets, dropout with an explicit validity mask (a zeroed electrode biases a mean), per-electrode gain jitter, and lattice flips and rotations applied consistently to positions. The flips are the electrode analogue of channel-swap augmentation in sound-event localisation ([ACS paper 2021](https://arxiv.org/abs/2101.02919)).

A subtler cue helps against false lines. A harmonic's electrode pattern should resemble its fundamental's. But the Apteronotus field is spatiotemporally complex, described as an ultrafast travelling wave with two spatial Fourier components ([Shifman et al. 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4626797/)), so harmonic ratios vary across electrodes. The network should learn this similarity rather than have it hard-coded as a veto.

## Identity belongs in the tracker, which needs better scores rather than a new network

No learned method for streaming or voice assignment that the researchers found handles more than a few sources. Liu and Wang tracked two speakers with permutation-invariant training (PIT) and failed on same-gender pairs "because same-gender pairs have very close pitch ranges" ([Liu & Wang 2018](https://pnlwang.github.io/papers/Liu-Wang1.icassp18.pdf)). Vocal-quartet voice assignment uses four fixed output slots and confuses alto with tenor where their ranges overlap ([Cuesta & Gómez 2022](https://transactions.ismir.net/articles/121)). Duan et al. found that "timbre consistency alone is insufficient for effective streaming" ([Duan lab](https://labsites.rochester.edu/air/projects/multipitch/multipitch.html)). Same-species fish a few Hz apart are the extreme case of "same-gender": pitch range carries almost no identity.

Multi-object tracking draws a sharp line between two regimes. On DanceTrack, where dancers look nearly identical, even ground-truth boxes give an association accuracy (AssA) of only 53.6, against 97.3 on MOT17. **Adding appearance to a Kalman tracker made association worse (AssA 29.7 vs 31.0), while better motion models helped (OC-SORT 35.3)**. FairMOT's jointly learned embeddings fell from 59.3 HOTA on MOT17 to 39.7 on DanceTrack ([Sun et al. 2022](https://arxiv.org/html/2111.14690v3)). ByteTrack's two-stage association is the most directly transferable idea. It matches confident detections first, then lets unmatched tracks claim low-score detections. Applied to nine trackers, it **raised IDF1 by 1 to 10 points** ([Zhang et al. 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/315_ECCV_2022_paper.php)).

End-to-end query trackers lead DanceTrack, but three things count against them here:

- MOTRv2 depends on an external detector ([Zhang et al. 2023](https://arxiv.org/pdf/2211.09791)).
- An independent 2026 re-run found SambaMOTR at 53.8 HOTA rather than the reported 69.0 ([arXiv 2609.22291](https://arxiv.org/pdf/2609.22291)).
- A tracking-by-detection method with learned association, CAMELTrack, reached a competitive 69.3 ([CAMELTrack 2025](https://arxiv.org/pdf/2505.01257)).

Cell biology points the same way. Trackastra uses a transformer over all detections in a temporal window to predict pairwise links, after which "simple greedy linking" matches or beats highly tuned trackers ([Gallusser & Weigert 2024](https://arxiv.org/abs/2405.15700)). Ultrack chooses among multiple segmentation hypotheses by global optimisation ([Bragantini et al. 2024](https://arxiv.org/abs/2308.04526v2)).

Wavetracker's pipeline (detections, then a tracker, then a learned fragment linker) already follows the Trackastra and CAMELTrack pattern. The electrode-amplitude cue behaves more like slowly changing position, a motion cue, than like appearance. That is why it already outperforms frequency on conflicts.

Embedding and instance methods are the alternative way to carry identity, and each has a known cost:

- **Per-bin embeddings.** Deep clustering ([Hershey et al. 2016](https://arxiv.org/abs/1508.04306)), associative embedding ([Newell et al. 2017](https://arxiv.org/abs/1611.05424)) and discriminative-loss embeddings ([De Brabandere et al. 2017](https://arxiv.org/abs/1708.02551)) do not fix the number of sources. ConvGRU embeddings even track cells through video ([Payer et al. 2018](https://arxiv.org/abs/1806.02070)). But identity holds only inside the window the network sees.
- **Slot-based outputs.** Solving PIT with the Hungarian algorithm scales to 20 speakers ([Dovrat et al. 2021](https://arxiv.org/pdf/2104.08955)). But in long recordings, "output permutation consistency between chunks is not guaranteed", and stitching chunks needs heavy overlap ([RSAN CSS 2021](https://arxiv.org/pdf/2110.14838)).
- **Curves as objects.** The row-anchor representation from lane detection (for each time column, a distribution over frequency per line instance) suits continuous lines, and CondLaneNet's "find an instance, then trace it" style is close to "find a fish, then follow it" ([Liu et al. 2021](https://arxiv.org/abs/2105.05003)). But lane benchmarks have a handful of lanes that rarely cross.
- **Transformer instance models.** A transformer instance-segmentation model for whistles reached P 89.99 / R 80.65 on held-out sites ([Zhang et al., JASA](https://www.citedrive.com/en/discovery/automated-detection-and-annotation-of-toothed-whale-whistles-using-transformer-based-instance-segmentation/)). TFFR-Net claims that lifting signals into a time–frequency–frequency-rate space cuts ridge crossings from 78.6% to 0.9% ([Pan et al., NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/149622)). Both were seen only as index or poster pages. Neither was tested on 30 crossing lines over hours.

The practical decision follows. **Keep the network's output free of identity.** Hours-long recordings can then be processed in overlapping chunks with only the central part kept (exact for convolutions when the overlap covers the receptive field). There is no stitching, and the tracker, linker, results format and correction GUI stay usable. Then improve the tracker and linker with the new scores. idtracker.ai's contrastive trick supplies free training pairs for a later identity embedding: two fragments present at the same time must be different animals ([Torrents et al. 2025](https://elifesciences.org/articles/107602)).

## Complex-domain mixing and loss masks decide faint-fish recall

On the evidence, how the training mixtures are built matters more than any architectural choice. In speech separation, "dynamic mixing" (making new mixtures every epoch) "often results in a 1.0 to 1.5dB" gain ([Ravenscroft et al. 2023](https://arxiv.org/pdf/2304.07142)), and the gain is largest when clean sources are few ([APSIPA 2022 study](https://aminer.org/pub/63be2deb90e50fcafd0a36cd/an-empirical-study-of-training-mixture-generation-strategies-on-speech-separation-dynamic)). That is the wavetracker situation.

The STFT is linear, but power is not: |X1+X2|² = |X1|² + |X2|² + 2·Re(X1·X2*). Summing power spectrograms replaces the beat-driven cross term with a constant. That would hide from the network the interference patterns that define failure mode (b). The researchers' recipe:

- Cache each train source's per-electrode complex STFT at 8–16 discrete warp factors, since frequency warping must be done in the time domain.
- Draw mixtures on the fly with random gains, constant phase rotations and hop-aligned time offsets, which are all exact in the complex domain.
- Take power only after summing.

They estimate a few GB per source and a few hundred GB for a few dozen sources, unless the band is cropped to what the configured fish range and harmonics actually need.

Background realism was the strongest sim-to-real lever in every field that measured it. Li et al.'s synthetic contours in real ocean noise came within 0.09 F1 of fully supervised training ([Li et al. 2020](https://arxiv.org/pdf/2005.08894)). The G2Net machine-learning teams "injected line-like features into their training data" ([Tenorio et al. 2025](https://arxiv.org/html/2509.06445v2)). DRASDIC's domain-randomised bioacoustic training gives the clearest numbers ([Hoffman et al. 2025](https://arxiv.org/html/2503.00296v2)):

- Full synthetic volume: real-data F1 of **0.445**.
- 220 h of synthetic data: 0.354.
- 22 h of synthetic data: 0.130.
- High-SNR events only: 0.402, below the full SNR range.

For wavetracker this means three things. Add real field-noise STFTs at random gain, with a per-fish SNR distribution that reaches well below 0 dB, because most misses sit below 6 dB. Keep integer harmonics of down-warped low fish as negatives. Add synthetic hum combs with randomised base frequency and per-electrode amplitude, and over-sample close pairs and crossings, since nothing in music training data ever shows two sources in one peak. Volume eventually stops paying off: one study's best result came from 1,000 synthetic soundscapes built from 30 calls, and 8,000 lowered AUC from 0.920 to 0.887 ([Soltero et al. 2025](https://arxiv.org/html/2507.16235v1)). The binding constraint is the number of distinct train sources, not the number of mixtures. The researchers' arithmetic: one 10-minute, 30-fish mixture already holds about 1.1 × 10⁵ labelled fish-frames, the size of Li et al.'s whole positive set.

The three truth defects need three different masks:

- **Real lines missing from the truth** are the most dangerous, because they are systematic. The same faint line, warped, appears in every mixture that uses its source, so the network would learn that "faint stable lines are not fish". That is worse than the random annotation dropout that detectors tolerate; one detector lost only about 5% with 30% of labels dropped ([Wu et al. 2018](https://arxiv.org/abs/1806.06986)). Deep salience avoided the problem by rebuilding mixes "with all 'unknown' pitch content removed" ([Bittner et al. 2017](https://brianmcfee.net/papers/ismir2017_salience.pdf)). Here the unknown lines sit inside the source recordings, so the closest equivalent is an ignore mask. Find residual lines on each source alone, store them in source coordinates so they warp and shift with the source, and give zero loss near them. The mask must **never** cover integer harmonics of labelled fish, which have to stay as negatives for failure mode (c).
- **Truth frames below the noise floor** would teach hallucination if kept as positives, and suppression of real faint fish if flipped to negatives. They should be ignored both ways using the `snr_db` field, and scored in a separate "undetectable" stratum.
- **Ridge-completed frames** (`freqs_detected=False`) are plausible positives with lower confidence and deserve a reduced weight.

Li et al.'s pseudo-label study shows why this care matters. Training naively on tracker-made labels scored **74.14 F1, below the 75.95 of the tracker that produced the labels**. A recall-aware loss lifted it to 86.31, close to full supervision at 87.47 ([Li et al. 2023, JASA](https://repository.library.noaa.gov/view/noaa/65889/noaa_65889_DS1.pdf)).

For the loss itself, the researchers estimate that about 2–3% of target mass per frame is positive with σ ≈ 1 bin. That level of imbalance is moderate, so plain BCE on blurred targets is a sound baseline. CenterNet's penalty-reduced focal loss (α = 2, β = 4, normalised by the number of positives) is the natural upgrade for faint recall ([Zhou et al. 2019](https://arxiv.org/pdf/1904.07850)). Large fixed positive weights should be avoided: a 2025 preprint, seen only through a search summary, reports that Basic Pitch's 0.95 positive weighting caused many false positives ([arXiv 2509.12712](https://arxiv.org/pdf/2509.12712)).

Positive-unlabelled learning is a poor fit. It needs a class prior that varies by recording, and noise-robust classification losses cannot target defects whose location is known ([Kiryo et al. 2017](https://papers.NeurIPS.cc/paper_files/paper/2017/file/7cce53cf90577442771720a370c3c723-Paper.pdf)). Teacher-student self-training on the unlabelled field archive should come last. Naive pseudo-labelling suffers "confirmation bias" unless every batch keeps a minimum share of labelled samples ([Arazo et al. 2020](https://arxiv.org/abs/1908.02983v5)). Adding unlabelled data made a DCASE 2022 system worse ([Li et al. 2022](https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Li_56_t4.pdf)). Local fine-tuning of a bird detector did not carry over to another region ([Lauha et al. 2022](https://jyx.jyu.fi/handle/123456789/83704)). For fish, the specific risk is that a teacher marks harmonics and hum as fish and the student learns to keep doing so.

## Eleven approach families against the six criteria

The table condenses the comparison. "Fit" summarises how well each family matches 30+ near-identical, crossing, sub-Hz lines over hours.

| Family | Examples | Fit to EOD data | Labels needed vs benchmark | Sample-rate / electrode-count independence | Compute on one RTX 4080 | Pipeline plug-in | Failure modes addressed |
|---|---|---|---|---|---|---|---|
| Classic peaks + harmonic rules (today) | wavetracker, thunderfish, Madhav | Proven; fixed thresholds | None | Yes, with Hz/s parameters | Trivial | Is the pipeline | Causes (a), (c) |
| Per-bin salience CNN on a harmonic stack | Deep salience, Basic Pitch, Deep Whistle | Strong after the linear-axis rework; whistle precedent | Per-frame f0 per source: exactly what the benchmark gives | Yes (Hz grid, Nyquist masks); summed stream is count-free | Small (17k to ~0.6M parameters) | Drop-in detector under `detector.method: learned` | (a), (c) strong; (b) weak |
| + temporal module (BiGRU, TCN, Conformer, Mamba) | Onsets and Frames, TF-GridNet, Conformer SED | Lines persist for seconds; rises need care | Same | Yes | Low; time attention is cheap | Inside the detector | (a), (d); (c) partly |
| Count-independent electrode fusion | TAC, VarArray, sparsemax, PopT, LUNA | Only cue for two fish in one peak; fails for co-located fish | Same truth plus electrode augmentation | Count-independent by construction | Linear in N; needs gating at 64 electrodes | Inside the detector | (b); (a) via max pooling; (c) via spatial pattern |
| DP / multi-target filters on the salience map | SOAP Viterbi, GM-PHD, SMC-PHD, AMTC | One track per band, or more fragmentation | Few | Yes | Low | Between detector and tracker | (a) at the lowest SNR; (d) mixed |
| Score-aware association | ByteTrack | Direct fit; tracker change only | None | Yes | Negligible | Tracker config | (a), (c) |
| Learned windowed association + contrastive fragments | Trackastra, CAMELTrack, idtracker.ai | Matches today's linker pattern | Identities free from the truth; co-occurrence pairs free | Yes | Low to moderate | Extends the linker | (d), crossings |
| Joint detection + embedding head | JDE, FairMOT, associative embedding | Look-alike regime: appearance often does not help | Identities free from the truth | Yes | Small | Extra tracker/linker cost | (d), (b) at crossings; weak evidence |
| Per-bin embeddings / PIT slots | Deep clustering, Hungarian PIT, voice assignment | Fixed slots do not scale; clustering brittle at 30 | Per-source masks | Yes | Moderate | Needs cross-window stitching | (b) within a window |
| Curves as objects / instance queries | CondLaneNet, LSTR, TFFR-Net, whistle instance segmentation | Representation fits; 30+ crossing curves untested | Per-source tracks per window | Yes | High | Replaces detector + short-range tracking; still needs the linker | (b), (c), (d) within a window; speculative |
| End-to-end query trackers | MOTR, MOTRv2, MOTIP, SambaMOTR | Hours × 30 lines is an unusual regime; reproducibility doubts | Full identity tracks | Yes | High; fit on 16 GB unknown | Replaces tracker and linker; disrupts the GUI workflow | (d), claimed |
| Self-supervised / self-training | PESTO, SS-MPE, Noisy Student, Mean Teacher | Equivariance valid only after stacking; harmonic false alarms | None, or pseudo-labels | Yes | Moderate to high (SS-MPE: 3–4 days per run) | Training regime only | Residual field gap; risky |

Three families form the core recommendation: the salience CNN on a linear harmonic stack, the temporal module, and count-independent electrode fusion. Each attacks a different failure mode, and all three keep the detector's output a plain set of scored detections. The association families are cheap, well evidenced and plug into existing code, so they belong in the plan as tracker and linker upgrades. The identity-in-network families (embeddings, slots, queries, end-to-end) pay their way only inside a window and force stitching across hours. Embeddings are the one cheap enough to try as an optional extra cost term.

## The recommended design, traced through one frame

Follow one 0.16 s frame through the system. The **summed stream** is the electrode-summed power spectrogram on the existing grid (0.61 Hz × 0.16 s, defined in Hz and seconds so it is the same at 20 and 48 kHz). It is log-compressed and divided by a per-recording noise floor. A strided gather turns it into seven harmonic channels on a fundamental grid that covers only the configured fish range. A residual 2-D CNN with no pooling along frequency turns those channels into features. In parallel, a shared **electrode encoder** reads each electrode's own crop around candidate frequencies, mixes the electrodes through TAC blocks, pools them by mean and max, and adds the result to the trunk's features through a gate that starts at zero. That way the model begins as "summed stream only" and learns to use electrodes where they help. A bidirectional GRU then runs along time at every frequency bin and sees 30–60 s of context. Small heads read out salience, a sub-bin offset, a "line but not a fundamental" score and, later, a source count. Peak picking produces detections in today's format, and the tracker and linker take over from there.

| Component | Recommendation | Basis |
|---|---|---|
| Grid | Existing linear spectrogram, 0.61 Hz bins, 0.16 s frames, defined in physical units; amplitude normalised by the window sum; the model's grid stored with its weights, and a mismatch refused or recomputed | Project rule; normalisation and lock-in are inferences |
| Band | Fundamental grid from the configured fish range (`harmonic_groups.min_freq/max_freq`); spectrum read from min_freq/3 to h_max·max_freq, capped at Nyquist; nothing hard-coded | Project rule; inference |
| Summed stream | Incoherent power sum over valid electrodes only, log-compressed, normalised by a noise floor; a max-over-electrodes map added as a second channel in Stage 3 | Sum: user decision; max channel: inference from the washout arithmetic |
| Harmonic stack | Gather at h ∈ {1/3, 1/2, 1, 2, 3, 4, 5}; max-pool about h bins around h·k; interpolate sub-harmonics; explicit validity mask above Nyquist | Stacking published (deep salience, Basic Pitch ablation); linear gather and pooling are inferences |
| Trunk | Residual 2-D CNN, 32–64 channels, no pooling or stride along frequency, frequency receptive field about ±6 Hz, local time about 2–3 s | No-pooling rule published (deep salience, PESTO, Deep Whistle); Hz/s sizes are inferences |
| Electrode stream | Per electrode: log power over its own noise floor, and log(electrode/sum) as a power-free spatial pattern; electrode (x, y) in cm from the grid centre as position encoding; validity mask | Pooled-relative inputs published (VarArray); fish-specific choices are inferences |
| Electrode fusion | Shared conv encoder with ≥ 3–5 s time context at full frame rate (one beat period at Δf ≥ 0.3 Hz); TAC blocks with mean+max; pooled output added through a zero-initialised gate; gated to candidate regions at inference | TAC, VarArray, PNA-style multi-aggregators, sparsemax published; gating and gate initialisation are inferences |
| Temporal module | Bidirectional GRU (or dilated 1-D convs) along time per frequency bin with shared weights, receptive field 30–60 s, non-causal | CRNN gains published; receptive-field size is an inference |
| Heads | (1) fundamental salience, sigmoid; (2) sub-bin offset in Hz; (3) "line but not a fundamental" (harmonic of a truth fish, hum, stray); (4) from Stage 3, source count 0/1/2+ per bin; (5) from Stage 5, per-peak embedding | (1)–(2) published (deep salience, CenterNet); (3) by analogy with Basic Pitch's multi-head design and Bayley's line class; (4)–(5) inferences |
| Decoding | Peak picking along frequency with about 1-bin minimum spacing; offset refinement; low threshold tuned on dev separately for lab and field-noise variants; continuous salience kept | Threshold non-transfer published; spacing is an inference |
| Interface | Today's detection format (frequency, per-electrode power read at the refined frequency) plus optional salience and auxiliary scores | Project; inference |
| Tracker | Two-threshold association: high-salience detections first, then existing traces absorb low-salience detections near their predicted frequency; traces start only from sustained high salience; optional veto on starts with a high line score; detections may be shared by two traces when the count head says 2+ | ByteTrack and ultrack's multiple hypotheses published; specifics are inferences |
| Linker | Retrained on fragments from the learned detector, with fragment-mean salience and line score as extra features; later, windowed learned association and contrastive embeddings | Trackastra, idtracker.ai published; specifics are inferences |
| Long recordings | About 60 s chunks with overlap ≥ the temporal receptive field on each side; keep the centre; no identity stitching | Inference from the long-form separation literature |

Training follows the evidence in the previous section. Mixtures come from a per-source, per-electrode complex-STFT cache at discrete warp factors spanning the benchmark's warp range. They are drawn on the fly with random gains, phase rotations and hop offsets, mixed with real field-noise STFTs at per-fish SNRs reaching well below 0 dB, and enriched with close pairs, crossings, rises, synthetic hum combs and harmonics of down-warped low fish. Each training example also gets random electrode subsets, dropout with masks, per-electrode gain jitter, a noisy electrode, and lattice flips.

Training uses crops: the full time context (about 375 frames) and a frequency window wider than the receptive field, with harmonic channels gathered from the full spectrum. Mixed precision keeps per-electrode activations within 16 GB (the researchers' estimate, not measured). Hard-example resampling in the style of Li et al.'s recall-guided synthesis is worthwhile, but it must use held-out train sources, never dev; Li et al. used validation data for theirs.

Targets are Gaussians with σ ≈ 0.6 Hz at the truth frequency at the frame centre, merged by maximum. The loss has four parts:

- BCE on salience, upgraded to CenterNet focal loss in Stage 2.
- L1 on the offset at truth bins.
- BCE for the line head.
- Cross-entropy for the count head.

Every term is multiplied by the mask: zero near residual source lines (never near harmonics of labelled fish), zero on truth frames below a chosen `snr_db` cut-off with a ramp of a few dB, and about 0.5 on ridge-completed frames.

The evaluation protocol makes each failure mode a number. All decisions use the dev split. Dev mixtures are generated once from dev sources and frozen. Residual-line masks and hard-example mining see only train sources. The test split, already scored once for the linker, is scored one more time at the end. Thresholds come from threshold-free precision–recall curves on dev, separately for lab and field-noise variants.

Frame-level scoring uses one-to-one Hungarian matching per frame with Hz tolerances of 0.6, 1.2 and 2.5 Hz. The `mir_eval` default of ±0.5 semitone (±29 Hz at 1000 Hz) would credit neighbouring fish ([mir_eval docs](https://mir-eval.readthedocs.io/latest/api/multipitch.html)). Recall is stratified four ways:

- by `snr_db` band;
- by distance to the nearest other line (< 1, 1–3, > 3 Hz);
- by |df/dt| (rises versus steady lines);
- by detected versus ridge-completed truth.

The undetectable stratum is neither rewarded nor penalised. False positives are typed: integer harmonic of a truth fish, hum frequency, known residual source line (an "unknown real" line, reported separately because the truth lacks it), or other.

Track-level scoring runs the unchanged tracker with both the existing and a retrained linker, and reports:

- IDF1;
- HOTA with its detection and association parts, DetA and AssA ([Luiten et al. 2021](https://arxiv.org/abs/2009.07736));
- fragments per truth track;
- the number and total duration of false identities.

Two oracle runs, in the spirit of DanceTrack, split the remaining IDF1 gap between detection and association: truth detections through the tracker and linker, and learned detections with oracle association. Two robustness checks round it off: the same model on electrode subsets (2, 4, 8, all) and on the field-noise variant, and seconds per recording-hour at each electrode count on the RTX 4080. The classic detector is scored first on every one of these measures.

## Six stages, each aimed at one failure mode and one dev number

| Stage | What is added | Failure mode targeted | Dev measurement | Move on when |
|---|---|---|---|---|
| 0. Harness | `detector.method: learned` stub that emits today's detection format plus a salience field; complex-STFT mixer and masks; stratified frame metrics; false-positive typing; oracle-detection and oracle-association runs; classic baseline in every stratum | Diagnosis only | Share of the IDF1 gap due to detection vs association; classic recall per SNR and spacing stratum | Every stratum has a classic number |
| 1. Minimal salience | Summed stream; linear harmonic stack; 6–8-layer CNN with no frequency pooling; σ ≈ 0.6 Hz salience head with ≤ 3-bin refinement; BCE with masks; dynamic mixing with field noise; tracker unchanged; linker scored as-is and retrained | (a) faint misses, (c) false lines | Recall in the < 6 dB strata; harmonic and hum false-positive rates; IDF1 and HOTA vs classic | Faint recall up at an equal or lower harmonic false-positive rate, and IDF1 at least classic |
| 2. Temporal context | Per-bin BiGRU or dilated TCN along time, 30–60 s; focal loss; offset head | (a), (d) fragmentation | < 6 dB recall; fragments per track; rise-stratum recall (no over-smoothing); offset error | Fragmentation down with no loss in rise-stratum recall |
| 3. Electrode stream | 3a: max-over-electrodes channel; 3b: TAC encoder with log(electrode/sum), mean+max pooling, zero-gated fusion, cm position encodings, candidate gating, count head, electrode-subset augmentation | (b) lines within 3 Hz; also (a) for single-electrode fish and (c) via spatial pattern | Recall and swap rate in the < 3 Hz stratum; count-head accuracy vs Δf; scores vs electrode count | The < 3 Hz stratum improves, and fewer electrodes never score below summed-only |
| 4. Score-aware tracking | Two-threshold association; high salience required to start a trace; line-score veto on starts; shared detections when count ≥ 2; linker retrained with salience features (can start right after Stage 1) | (c) long false identities, (d) | Count and duration of false identities; IDF1; AssA | False-identity duration and count drop, and IDF1 rises |
| 5. Identity cues | Per-peak embedding head (discriminative loss, identities from per-source truth) as an extra cost; Trackastra-style windowed linker; contrastive fragment pairs from co-occurrence | (d), swaps at crossings | IDF1 and AssA with the embedding on vs off; swaps at crossings | The embedding ablation shows an IDF1 gain |
| 6. Research branch | Field self-training with track-consistency and harmonic-veto filters; signed or complex electrode inputs; de-mixed per-electrode amplitudes for shared peaks; within-window line queries; Viterbi over salience | Residual field gap; sub-bin pairs | Field-noise variant and held-out field data | Only if Stages 1–5 plateau |

Two remarks on ordering. Stage 4 changes only the tracker, so the classic detector can use it too. Classic peaks can be given a pseudo-salience from their peak SNR, which makes the comparison fair and shows how much of any gain is the tracker's. Stage 3 should begin with a non-learned check that the cue exists. Mix pairs of real train sources at controlled frequency differences (0.3–3 Hz) and spatial separations. Compute the per-bin spatial covariance over a few seconds from the complex per-electrode STFTs. Measure how well the eigenvalue ratio λ2/λ1 separates one fish from two at the benchmark's SNRs. If it fails there, failure mode (b) belongs to the tracker's shared-detection logic, not to the network.

## The main risks lie in data diversity, label defects and the downstream fit

The largest risk is that diversity, not volume, limits generalisation. The train pool contains a limited number of two-fish lab sources, and the dev mixtures come from the same lab. Dev scores will therefore overstate field performance, which is the gap between synthetic and recorded data that the DESED sound-event dataset documents ([Turpault et al. 2020](https://arxiv.org/pdf/2007.03931.pdf)). The field-noise variant is only a proxy. A learning curve over the number of train sources, rather than the number of mixtures, shows whether recording more pairs would pay. A related risk is species. If the train sources are all one species, the network never sees real Sternopygus or Eigenmannia harmonic profiles. Warping sources down creates low fish whose harmonics land in the others' band, but with Apteronotus harmonic ratios.

The second risk is label defects. Ignore masks depend on a residual-line finder, and its errors leak in two directions: missed residual lines become hidden negatives, and masked real harmonics stop being negatives. Evaluation adds a trap of its own, because unknown real lines count as false positives unless they are typed separately. The third risk is inherited recall bias. Music salience retrieved dominant lines and missed inner voices. Some truth frames lie below the noise by construction, so recall in the faintest stratum has a ceiling no method can pass.

The fourth risk is the physics of close pairs. The beat and rank-2 argument is untested. Close fish are often physically close, which is where spatial separation collapses in speech separation. Stage 3 may therefore move (b) less than hoped. The fifth risk is that the temporal module bridges gaps into false identities. The mechanism that recovers faint lines can also carry a line through noise, as gap-bridging did for whistle trackers. False-identity duration has to be watched at every stage.

The remaining risks are operational:

- **Thresholds and resolution.** Thresholds will shift between lab and field. A model trained at 0.61 Hz and 0.16 s is tied to that grid.
- **Sample rate.** At 48 kHz, more harmonic channels are valid than at 20 kHz. Training with random channel masking keeps the model from relying on which channels are present.
- **Downstream fit.** The linker and the tracker's parameters were tuned on classic detections. Without retraining and retuning, a better detector can still lower IDF1.
- **Memory at 64 electrodes.** Candidate gating controls memory but can lose fish the gating misses.
- **Dev overfitting.** Dev is reused at every stage. The test split must stay untouched until the final comparison.

## Open decisions for the user

| Decision | Options | Suggested default and reason |
|---|---|---|
| Temporal module | BiGRU per bin; dilated TCN; Conformer along time; Mamba | BiGRU or TCN first: best evidence, simplest to explain; attention later if needed |
| Mixing domain and storage | Complex per-electrode STFT cache (a few hundred GB); power-only mixing; time-domain bank for dev only | Complex cache for training, because only complex mixing keeps beats for (b) |
| Electrode features | Power and log ratio to sum; add signed or complex terms | Power and ratio first; signed or complex terms as a Stage 6 ablation, since power loses the dipole sign pattern |
| Max-over-electrodes channel | Add to the summed stream, or not | Add as ablation 3a: cheap, count-independent, guards against washout |
| Electrode branch scope at inference | Full band; fundamental band; candidate-gated | Candidate-gated, with the max channel as a safety net; measure speed and recall at 16 and 64 electrodes |
| Detection format | Unchanged; add salience; add line and count scores | Add optional fields: the tracker needs them, and the GUI can display them |
| Tracker changes | None; two-threshold; shared detections | Two-threshold behind config, also for classic; shared detections only after Stage 3 |
| Model grid | Fixed 0.61 Hz × 0.16 s per model; train across resolutions | Fixed per model, refuse or recompute on mismatch; consistent with the physical-units rule |
| Mask parameters | `snr_db` cut-off; ridge-completed weight; residual-line radius | Choose on dev using the stratified metrics; start with weight 0.5 for completed frames |
| Harmonic set | {1/3, 1/2, 1, 2, 3, 4, 5}; more harmonics; fewer | As listed, with Nyquist masks; revisit if harmonic false positives persist |
| Re-synthesised sources | None; add an "EOD-stem-synth" set with exact labels | Defer unless masks fail; exact labels cost domain realism |
| Field evaluation | Dense benchmark only, with the field-noise proxy; add a small curated field set | The user chose benchmark-only. Note that only real field data measures the lab-to-field gap |

## How much rests on evidence, inference or doubtful sources

**Published evidence supports** the following:

- a learned salience map in front of an unchanged tracker (whistles);
- harmonic stacking and no frequency pooling (deep salience, Basic Pitch, PESTO);
- gains from temporal context (Onsets and Frames);
- count-independent fusion with pooled-relative inputs (TAC, VarArray);
- appearance cues being weak for look-alike objects, and score-aware association (DanceTrack, ByteTrack);
- learned windowed linking (Trackastra);
- dynamic mixing, SNR breadth and real-background realism (separation literature, DRASDIC, Li et al.);
- recall-aware and masked training against incomplete labels (Li et al. 2023);
- the risks of self-training.

**Researchers' inferences or arithmetic** carry the following:

- the linear-axis gather with per-harmonic pooling;
- the case against a log axis;
- σ ≈ 0.6 Hz targets;
- the k/√N washout estimate;
- the quasi-static sign, beat and rank-2 argument for separating fish within one bin;
- candidate gating;
- every compute and storage figure;
- the mask recipes;
- the mapping from techniques to failure modes;
- every stage criterion.

**Untested anywhere in the literature found:** linear-axis harmonic stacking at sub-Hz resolution; multi-f0 with more than about five sources; separating sources closer than one bin by spatial cues; learned salience combined with multi-sensor identity cues for tonal tracking; and 30+ crossing curves as instances.

| Source or claim | Issue | How the report treats it |
|---|---|---|
| TFFR-Net, "NeurIPS 2026 poster" (crossing rate 78.6% → 0.9%) | 2026 venue known only from a poster page; synthetic ≤ 6 components | Idea only |
| Independent MOT benchmark, arXiv 2609.22291 (SambaMOTR 53.8 HOTA) | 2026 preprint | Caution flag, not a fact |
| OpenWhistle (NeurIPS 2026); "Mind the Gap" (Marine Mammal Science 2026); arXiv 2609.13281, 2603.14917, 2605.17405 | 2026 venues or preprints; full texts not retrieved | Not used for design |
| Transformer whistle instance segmentation (JASA, doi 10.1121/10.0044187) | Seen only as an index record; year and architecture unknown | Numbers indicative only |
| LSE-Net; SwDCNN; DRNet; Multiple DeepLofargram | Search results or abstracts only; LSE-Net link is a best guess; simulated data | Not used |
| Gruden & White PHD filters; Duan et al. 2014; Paris & Jauffret 2003; multi-target HMM; Sensors 2025 IF U-Net | Full texts not read; search summaries | Background only |
| Harmonic Convolution (ICLR 2020); HarmoF0; MT3 details; Çakır CRNN gain figures | Operator unverified; secondary summaries | Not relied on |
| Training-note numbers marked "via search summary": SepFormer and Wavesplit dynamic-mixing gains, APSIPA 2022, Soft Sampling, Background Recalibration Loss, Unbiased Teacher, Lauha 2022, arXiv 2509.12712 (Basic Pitch false positives) | Not read in full | Direction trusted, magnitudes approximate |
| Raab et al. 2022 combined accuracy "99.87% (462/464)" | Internally inconsistent: 462/464 = 99.57% | Full-recording figures preferred |
| Onsets and Frames parameter count (18M vs 4.3M + 3.5M); Deep Whistle receptive field (46 ms × 2.875 kHz vs 56 ms × 3.125 kHz); SMC-PHD F1 (80.1 vs 83.40); ByteTrack DanceTrack HOTA (47.7 vs 47.1) | Sources disagree | Not load-bearing |
| LUNA (NeurIPS 2025 per arXiv comments); UniArray, FlexIO, HEAR, EEG-X (preprints); DIVER-0 (via an overview); BIOT venue; SpatialNet journal status | Venues unverified | Mechanisms used, not rankings |
| DeepSets, PNA, Perceiver, PCEN | Cited from memory, not re-fetched | Standard background |
| DRASDIC; Soltero et al. | 2025 preprints | Used with that label |
| Kiar et al. 2013; Migliaro et al. 2025 (iScience) | Snippets only | Not used |
| Chirp-type durations (McGill review, Hupe & Lewis) | Secondary summaries | Background only |
| Deep salience ≈ 0.6M parameters; RTX 4080 ≈ 50 TFLOPS FP32; CREPE's Viterbi option | Researchers' estimates or recollection | Labelled as estimates |

## Conclusion

The four failure modes divide cleanly by where their fix lives, and that division sets the order of work. Faint misses and false lines are mostly a matter of what the network is shown and told: harmonic-aligned input, a wide SNR range, hard negatives and honest loss masks. The whistle precedent shows that adding context to peak choice is the cheapest large gain. Fragmentation is mostly downstream, in score-aware association and a linker retrained on the new detections. Close lines are different in kind. They are an information problem: the summed power spectrogram lacks what is needed, because power-domain mixing would erase beats and power discards the dipole sign pattern. That makes (b) the only failure mode that needs both a new way of generating data and the electrode branch.

The first things to build are therefore the complex-domain mixer, the residual-line and SNR masks, and the stratified, oracle-equipped harness. Every later stage needs them, and so does a fair re-score of the classic detector. The electrode branch should wait for a cheap λ2/λ1 test showing that the cue survives at real SNRs and fish spacings. Nothing in music, bioacoustics, sonar or tracking has handled 30 near-identical lines that cross for hours, so the dense benchmark with these strata is a methodological contribution in its own right. Its oracle runs will show how much of today's 0.61–0.65 IDF1 a better detector can recover at all.
