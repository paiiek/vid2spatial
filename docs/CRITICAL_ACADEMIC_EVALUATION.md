# Vid2Spatial: Critical Academic Evaluation

**Evaluator Role**: Independent Academic Reviewer
**Review Date**: 2025-11-30
**System Version**: Vid2Spatial (post-ablation improvements)
**Evaluation Context**: ICASSP 2025 Submission Readiness

---

## Executive Summary

**Overall Assessment**: ⚠️ **PROMISING BUT PRELIMINARY**

Vid2Spatial demonstrates a **technically sound geometric approach** to video-to-spatial audio, but suffers from **limited evaluation scope**, **questionable design choices**, and **missing comparisons with state-of-the-art**. The system shows **potential** but requires significant additional work before publication at a top-tier venue like ICASSP.

**Key Strengths**:
- ✅ Novel integration of depth + tracking + geometric FOA encoding
- ✅ Near real-time performance (0.91x RTF)
- ✅ Thorough ablation study revealing critical insights
- ✅ Clean, modular architecture

**Critical Weaknesses**:
- ❌ **Evaluation limited to 20 samples** (1.1% of FAIR-Play dataset)
- ❌ **No comparison with neural baselines** (VisualEchoes, SpatialAudio-Gen, etc.)
- ❌ **Low absolute performance** (correlation 0.72 is modest)
- ❌ **IR modeling fundamentally broken** (degrades rather than improves)
- ❌ **No user study or perceptual evaluation**
- ❌ **Single-object tracking only** (unrealistic assumption)

**Recommendation**: **MAJOR REVISION REQUIRED** before submission.

---

## 1. Technical Soundness

### 1.1 Approach Validity ✅

**Assessment**: The geometric approach is **theoretically sound**.

**Strengths**:
- First-order Ambisonics encoding is physically grounded
- MiDaS depth estimation is a reasonable proxy
- KCF tracking is established (though outdated)
- Pipeline is interpretable and debuggable

**Concerns**:
- **Depth accuracy**: MiDaS trained on diverse images, not videos. Temporal consistency?
- **Single object assumption**: Real-world audio has multiple sources
- **Tracking initialization**: Requires manual bounding box (not end-to-end)

**Score**: 7/10 (sound but with caveats)

### 1.2 Implementation Quality ✅

**Assessment**: Code is **well-structured** and **reproducible**.

**Strengths**:
- Modular design (vision, spatial, room modules)
- Clear configuration system
- Comprehensive logging
- Good error handling

**Concerns**:
- pyroomacoustics installation fails (C++ compilation issues)
- Fallback to Schroeder IR is **demonstrably harmful**
- No automated tests or CI/CD

**Score**: 8/10 (production-quality code)

### 1.3 Ablation Study ⭐

**Assessment**: **Excellent ablation study** - this is the strongest part of the work.

**Strengths**:
- Tests 5 configurations systematically
- Reveals critical finding: **IR convolution is harmful**
- Clear methodology and statistical reporting
- Honest about failures (IR degradation)

**Impact**:
- This finding alone could be a **short paper contribution**
- Shows that simple geometric approaches can outperform naive room modeling

**Score**: 9/10 (ICASSP-quality analysis)

---

## 2. Evaluation Rigor

### 2.1 Dataset Coverage ❌

**Assessment**: **CRITICALLY INSUFFICIENT**

**Current**:
- 20 samples from FAIR-Play
- **1.1% of available data** (20 / 1,871)
- No cross-dataset validation

**Required for Publication**:
- Minimum 100 samples (5% coverage) for preliminary results
- Ideally 500+ samples (25%) for robust claims
- Cross-validation on other datasets (SoundSpaces, AudioSet, etc.)

**Statistical Significance**:
- n=20 is marginal for statistical tests
- Wide variance in results suggests need for larger sample
- Confidence intervals would be very wide

**Score**: 3/10 (inadequate for publication)

### 2.2 Baseline Comparisons ❌ (IN PROGRESS)

**Assessment**: **MISSING CRITICAL COMPARISONS**

**Currently Evaluated**:
- ✅ Mono (trivial baseline)
- ✅ Simple Pan (basic baseline)
- ✅ Random Pan (sanity check)
- ✅ Ablation configurations (internal comparison)

**MISSING - Essential for Publication**:
- ❌ **VisualEchoes** (Gao & Grauman, ECCV 2020) - geometric baseline
- ❌ **Sep-Stereo** (Xu et al., ICCV 2021) - neural separation
- ❌ **SpatialAudio-Gen** (if available) - neural spatial audio
- ❌ **Mono2Binaural** (Morgado et al., ECCV 2018) - neural baseline

**Impact**:
- **Cannot claim novelty** without comparing to prior work
- **Cannot claim superiority** without beating state-of-the-art
- Reviewers will reject on this basis alone

**Score**: 2/10 (critical gap)

### 2.3 Metrics Selection ✅

**Assessment**: **Appropriate metrics** for spatial audio.

**Included**:
- ✅ Correlation (signal similarity)
- ✅ ILD (interaural level difference)
- ✅ ITD (interaural time difference)
- ✅ SI-SDR (signal quality)
- ✅ Spectral distances (LSD, STFT)

**Missing**:
- ⚠️ **Localization accuracy** (azimuth/elevation error)
  - Authors tried but found binaural-to-angles unreliable
  - Should use FOA-to-angles directly (source DOA estimation)
- ⚠️ **Perceptual metrics** (PEAQ, ViSQOL, PESQ)
- ⚠️ **User study** (subjective quality, preference tests)

**Score**: 7/10 (good but incomplete)

### 2.4 Performance Analysis ✅

**Assessment**: **Thorough performance profiling**.

**Strengths**:
- RTF measurement (0.91x near real-time)
- Component-level timing breakdown
- Ablation impact on speed

**Missing**:
- GPU memory usage
- Scalability to longer videos (tested on ~10s clips only)
- Batch processing capabilities

**Score**: 8/10 (good engineering analysis)

---

## 3. Results Quality

### 3.1 Absolute Performance ⚠️

**Current Results** (improved system, n=20):
- Correlation: **0.72** ± 0.11 (L channel)
- SI-SDR: **+0.7 dB** ± 3.1
- ILD Error: **1.91 dB** ± 1.14

**Interpretation**:

**Correlation 0.72**:
- ⚠️ **Modest** performance
- In audio ML: 0.5-0.7 = weak, 0.7-0.85 = moderate, 0.85+ = strong
- **Not publication-worthy** by itself
- Suggests ~50% variance explained (R²)

**SI-SDR +0.7 dB**:
- ✅ **Positive** is good (better than mono)
- ⚠️ **Small improvement** over reference
- State-of-the-art source separation: 10-15 dB
- **Barely perceptible difference** in blind tests

**ILD Error 1.91 dB**:
- ✅ **Excellent** spatial accuracy
- Human JND for ILD: ~1 dB
- This is the **strongest result**

**Overall**: Results are **acceptable** but **not impressive**. Need comparison to show they're competitive.

**Score**: 6/10 (acceptable but not strong)

### 3.2 Consistency ✅

**Assessment**: **Results are consistent** across samples.

**Evidence**:
- Standard deviations reasonable
- Ablation results replicate across 5 samples
- No catastrophic failures

**Score**: 8/10 (reliable system)

### 3.3 Failure Analysis ⚠️

**Assessment**: **Limited failure mode analysis**.

**Provided**:
- Sample 000005 shows poor correlation (0.15 with IR)
- Acknowledgment of tracking limitations

**Missing**:
- **When does the system fail?**
  - Fast motion?
  - Occlusions?
  - Multiple sources?
  - Small objects?
- **Error distribution analysis**
  - Are errors Gaussian or skewed?
  - Outliers?

**Score**: 5/10 (superficial failure analysis)

---

## 4. Novel Contributions

### 4.1 Claimed Contributions

From the work, I infer these intended contributions:

1. **Integration of depth + tracking for spatial audio**
   - Incremental, not novel (VisualEchoes did this)

2. **Real-time geometric FOA encoding**
   - Engineering contribution, not algorithmic

3. **Ablation showing IR is harmful**
   - ⭐ **Novel empirical finding** (publishable)

4. **Modular pipeline architecture**
   - Engineering, not research contribution

### 4.2 Actual Novelty ⚠️

**Assessment**: **Limited novelty** for top-tier venue.

**What's Novel**:
- ✅ Empirical finding: Simple Schroeder IR degrades performance
- ✅ Specific pipeline design choices validated by ablation
- ✅ Real-time performance on consumer hardware

**What's NOT Novel**:
- ❌ Depth-based spatial audio (done before)
- ❌ Object tracking for audio (done before)
- ❌ FOA encoding (standard technique)
- ❌ KCF tracking (2014 algorithm)

**Publication Viability**:
- ❌ **ICASSP (main track)**: Insufficient novelty
- ⚠️ **ICASSP (demo/late-breaking)**: Possible
- ✅ **WASPAA, AES**: Good fit
- ✅ **ArXiv + Workshop**: Definitely

**Score**: 4/10 (incremental work)

---

## 5. Comparison to State-of-the-Art

### 5.1 Prior Work (What Should Be Compared)

**Geometric Approaches**:
- **VisualEchoes** (Gao & Grauman, ECCV 2020)
  - Uses depth + audio separation
  - Reports azimuth error ~15°
  - **MUST COMPARE**

**Neural Approaches**:
- **Sep-Stereo** (Xu et al., ICCV 2021)
  - Learns to separate and spatialize
  - State-of-the-art on FAIR-Play
  - **MUST COMPARE**

- **Mono2Binaural** (Morgado et al., ECCV 2018)
  - Classic neural baseline
  - Should be easy to compare

**Hybrid Approaches**:
- **Audio-Visual Scene-Aware Dialog** (Alamri et al., CVPR 2019)
  - Uses attention for spatial audio

### 5.2 Reported Performance (from literature)

**FAIR-Play Benchmark**:
- Sep-Stereo: **Correlation ~0.65-0.75** (estimated from paper)
- Random baseline: **Correlation ~0.1-0.2**

**Our System**:
- Correlation: **0.72** ± 0.11

**Interpretation**:
- ⚠️ **Competitive with published results** (if estimates are accurate)
- ❌ **Cannot verify** without direct comparison
- ❌ **Different metrics make comparison unclear**

### 5.3 Missing Comparisons ❌

**Critical Gap**: Zero head-to-head comparisons.

**Required**:
1. Re-implement or use official code for baselines
2. Run on same 20 (or better, 100) samples
3. Report same metrics
4. Statistical significance tests (t-test, Wilcoxon)

**Without this**: Paper will be **desk-rejected** by any serious venue.

**Score**: 0/10 (not done)

---

## 6. Experimental Design Issues

### 6.1 Sample Selection Bias ⚠️

**Concern**: How were 20 samples selected?

**Options**:
- Random (good) - but seed should be reported
- First 20 (bad) - may not be representative
- Cherry-picked (very bad) - biased results

**Recommendation**:
- Report selection method
- Use stratified sampling if categories exist
- Ideally test on standard train/val/test split

### 6.2 Initialization Dependency ⚠️

**Concern**: All tests use **center-frame initialization**.

**Impact**:
- Not realistic (requires manual annotation)
- Biased toward center-frame objects
- Doesn't test tracking robustness

**Recommendation**:
- Test with detector (YOLO, Faster R-CNN)
- Test with random initialization
- Report tracking success rate separately

### 6.3 Single Object Assumption ❌

**Critical Limitation**: Assumes **one sound source**.

**Reality**:
- Most scenes have multiple sources (speech, music, environment)
- FAIR-Play may have background noise
- System has no mechanism for multi-source

**Impact**:
- **Severely limits applicability**
- Real-world use cases require multi-source handling
- Should be acknowledged as major limitation

---

## 7. Writing & Presentation Issues

### 7.1 Documentation Quality ✅

**Assessment**: **Excellent** for a technical system.

**Strengths**:
- Comprehensive evaluation reports
- Clear ablation study document
- Detailed methodology

**For Publication**:
- Needs to be condensed to 4-6 pages
- Clearer contribution statements
- Better related work section

### 7.2 Missing Elements for Publication ❌

**Required for ICASSP**:

1. ❌ **Abstract** (not written)
2. ❌ **Introduction** with clear motivation
3. ❌ **Related Work** section (comprehensive survey)
4. ❌ **Method** description (4-6 pages)
5. ❌ **Experiments** section (current work is close)
6. ⚠️ **Results** (need baselines)
7. ❌ **Discussion** (limitations, future work)
8. ❌ **Conclusion**
9. ❌ **Figures** (architecture diagram, result plots)
10. ❌ **Tables** (comparison with baselines)

**Estimated Completion**: **50%** (mostly experiments, no paper draft)

---

## 8. Specific Technical Concerns

### 8.1 Depth Estimation Reliability ⚠️

**Concern**: MiDaS provides **relative depth**, not **metric depth**.

**Impact**:
- Absolute distance estimates may be wrong
- Affects distance-based gain/filtering
- Ablation shows depth has minimal impact (suspicious)

**Questions**:
- Is depth even being used correctly?
- Should try metric depth (ZoeDepth, Metric3D)
- Could explain why depth ablation shows no benefit

### 8.2 Tracking Robustness ⚠️

**Concern**: KCF is from **2014**, known to fail on:
- Fast motion
- Occlusions
- Scale changes
- Deformable objects

**Evidence**:
- Paper mentions tracking losses
- No quantitative tracking accuracy reported

**Recommendation**:
- Upgrade to modern tracker (SiamRPN++, TransT, OSTrack)
- Report tracking precision/success rate
- Show examples where tracking fails → audio fails

### 8.3 FOA Encoding Assumptions ⚠️

**Concern**: Assumes **point source** at estimated distance/direction.

**Reality**:
- Real audio has spatial extent
- Reverberation creates diffuse field
- FOA W,X,Y,Z may not capture full soundfield

**Question**: Is FOA the right target?
- Alternatives: VBAP, HRTF, learned representation
- Should justify choice of FOA

### 8.4 IR Modeling Failure ❌

**Critical Finding**: IR convolution **degrades** performance.

**Implications**:
1. ✅ **Honest reporting** (good scientific practice)
2. ❌ **Fundamental design flaw** (reverb should help, not hurt)
3. ⚠️ **Mismatched assumptions** (Schroeder RT60 ≠ FAIR-Play acoustics)

**Root Cause Hypotheses**:
- FAIR-Play uses **anechoic** or minimal reverb
- Schroeder model is too simplistic
- RT60=0.6s doesn't match data

**Recommendation**:
- Disable IR by default (done ✓)
- Test with pyroomacoustics if fixable
- OR: Learn IR from data (neural network)
- Could be a **follow-up paper**: "Why does geometric IR fail?"

---

## 9. Ethical & Reproducibility Concerns

### 9.1 Reproducibility ✅

**Assessment**: **Excellent**.

**Provided**:
- ✅ Code available (assumed)
- ✅ Clear configuration
- ✅ Dataset is public (FAIR-Play)
- ✅ Evaluation scripts included

**Missing**:
- ⚠️ Random seeds not documented
- ⚠️ Hardware specs (GPU model, RAM)
- ⚠️ Library versions (requirements.txt incomplete)

**Score**: 8/10 (mostly reproducible)

### 9.2 Ethical Considerations ✅

**Assessment**: No major concerns.

**Potential Issues**:
- Deepfake audio generation (spatial audio could enhance realism)
- Surveillance applications (tracking + audio localization)

**Recommendation**: Add ethics statement in final paper.

---

## 10. Recommendations for Improvement

### 10.1 Must-Have (Before Submission)

**Priority 1 - Comparison with Baselines**:
- [ ] Implement or download **VisualEchoes** baseline
- [ ] Run on same 100 samples
- [ ] Report statistical comparison (t-test, effect size)
- **Estimated Time**: 1 week

**Priority 2 - Larger Evaluation**:
- [ ] Expand to **100+ samples** minimum
- [ ] Report confidence intervals
- [ ] Stratified sampling (if categories available)
- **Estimated Time**: 2 days (just re-run)

**Priority 3 - Write Paper Draft**:
- [ ] Abstract, intro, related work, method, experiments
- [ ] Create architecture figure
- [ ] Create result plots (bar charts, scatter plots)
- **Estimated Time**: 1 week

### 10.2 Should-Have (For Strong Paper)

**Priority 4 - User Study**:
- [ ] ABX perceptual test (20 participants)
- [ ] Compare: mono vs simple-pan vs ours vs ground-truth
- [ ] Measure preference + localization accuracy
- **Estimated Time**: 2 weeks

**Priority 5 - Multi-Source Extension**:
- [ ] Extend to 2-3 sources
- [ ] Use source separation (Demucs) to isolate sources
- [ ] Track multiple objects
- **Estimated Time**: 2-3 weeks

**Priority 6 - Improve IR Modeling**:
- [ ] Fix pyroomacoustics installation
- [ ] Test realistic IR vs Schroeder
- [ ] OR: Learn IR correction network
- **Estimated Time**: 1 week

### 10.3 Nice-to-Have (For Excellent Paper)

**Priority 7 - Cross-Dataset Validation**:
- [ ] Test on SoundSpaces, AudioSet, MUSIC
- [ ] Show generalization
- **Estimated Time**: 1 week

**Priority 8 - Real-Time Demo**:
- [ ] Live webcam + microphone input
- [ ] Process in real-time
- [ ] Deploy as web demo
- **Estimated Time**: 2 weeks

**Priority 9 - Ablation on Tracking**:
- [ ] Test modern trackers (OSTrack, etc.)
- [ ] Show tracking quality → audio quality correlation
- **Estimated Time**: 3 days

---

## 11. Publication Readiness Assessment

### 11.1 Current State

| Criterion | Status | Score |
|-----------|--------|-------|
| **Novel Contribution** | ⚠️ Incremental | 4/10 |
| **Technical Soundness** | ✅ Solid | 7/10 |
| **Evaluation Rigor** | ❌ Insufficient | 3/10 |
| **Baseline Comparison** | ❌ Missing | 0/10 |
| **Writing Quality** | ❌ Not started | 0/10 |
| **Reproducibility** | ✅ Good | 8/10 |
| **Impact** | ⚠️ Limited | 5/10 |

**Overall**: **27/70 = 39%** (Not ready)

### 11.2 Venue Suitability

**ICASSP 2025 (Main Track)**:
- **Acceptance Rate**: ~45%
- **Current Readiness**: ❌ **Not ready** (needs major revision)
- **Required Work**: 3-4 weeks full-time
- **Recommendation**: **Do not submit** without baseline comparisons

**ICASSP 2025 (Late-Breaking Demo)**:
- **Acceptance Rate**: ~70%
- **Current Readiness**: ⚠️ **Marginal** (system works, ablation is interesting)
- **Required Work**: 1 week (write 2-page demo paper)
- **Recommendation**: **Possible** if deadline allows

**WASPAA 2025** (IEEE Workshop on Applications of Signal Processing to Audio and Acoustics):
- **Acceptance Rate**: ~60%
- **Current Readiness**: ⚠️ **Fair** (good fit for workshop)
- **Required Work**: 2 weeks (add baselines, write paper)
- **Recommendation**: **Good target venue**

**ArXiv + ICASSP Workshop**:
- **Current Readiness**: ✅ **Ready** (can submit anytime)
- **Recommendation**: **Immediate option** to establish priority

### 11.3 Timeline Estimate

**If Targeting ICASSP Main Track** (assume 4-week deadline):

| Week | Tasks | Priority |
|------|-------|----------|
| **Week 1** | Baseline comparisons (VisualEchoes, Sep-Stereo) | MUST |
| **Week 2** | Expand evaluation to 100+ samples, user study | MUST |
| **Week 3** | Write full paper draft, create figures | MUST |
| **Week 4** | Revisions, polish, submit | MUST |

**Feasibility**: ⚠️ **Tight** but **possible** with focused effort.

**If Targeting WASPAA** (assume 2-month deadline):

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Phase 1** | 2 weeks | Baseline comparisons, 100-sample eval |
| **Phase 2** | 2 weeks | Multi-source extension, improved IR |
| **Phase 3** | 2 weeks | User study (ABX test) |
| **Phase 4** | 2 weeks | Paper writing, revisions |

**Feasibility**: ✅ **Comfortable** timeline for strong paper.

---

## 12. Comparison with Typical ICASSP Papers

### 12.1 Typical Accepted ICASSP Paper (Audio/Spatial Audio Track)

**Evaluation**:
- Dataset: 100-1000 samples
- Baselines: 3-5 prior methods
- Metrics: 5+ standard metrics
- User study: Often included
- Cross-dataset: Common

**Novelty**:
- New algorithm/architecture
- OR: Significant performance gain (>10% improvement)
- OR: Novel task/dataset

**Writing**:
- 4 pages + 1 reference page
- Clear figures (3-5)
- Statistical significance tests

### 12.2 How Vid2Spatial Compares

| Aspect | ICASSP Typical | Vid2Spatial | Gap |
|--------|----------------|-------------|-----|
| **Dataset Size** | 100-1000 samples | 20 samples | ❌ 5-50× smaller |
| **Baselines** | 3-5 methods | 3 (trivial) | ❌ No SOTA |
| **User Study** | Often | No | ❌ Missing |
| **Novel Algorithm** | Yes | No (geometric) | ❌ Incremental |
| **Performance Gain** | >10% | ~0% (no SOTA) | ❌ Unknown |
| **Writing** | 4 pages | 0 pages | ❌ Not started |

**Verdict**: **Below typical ICASSP standard** in current state.

---

## 13. Strengths to Emphasize (When Ready)

When this work is polished, emphasize:

1. ✅ **Ablation Study Quality**
   - Rigorous methodology
   - Surprising finding about IR
   - Could inspire follow-up work

2. ✅ **Real-Time Performance**
   - 0.91x RTF is impressive
   - Enables practical applications
   - Compare to neural methods (often 10-100× slower)

3. ✅ **Interpretability**
   - Geometric pipeline is explainable
   - Easy to debug and improve
   - Contrast with black-box neural networks

4. ✅ **Modular Design**
   - Can swap components (depth, tracking, IR)
   - Good for future research
   - Extensible to multi-source

5. ✅ **Honest Reporting**
   - Acknowledges IR failure
   - Shows negative results (rare in ML)
   - Builds trust

---

## 14. Final Verdict

### 14.1 Overall Assessment

**System Quality**: ⭐⭐⭐⚪⚪ (3/5)
- Well-implemented, runs reliably
- But limited scope and performance

**Research Quality**: ⭐⭐⚪⚪⚪ (2/5)
- Good ablation study
- But insufficient evaluation and no baselines

**Publication Readiness**: ⭐⚪⚪⚪⚪ (1/5)
- Major work needed before submission

**Potential**: ⭐⭐⭐⭐⚪ (4/5)
- With proper evaluation, could be solid workshop paper
- Ablation findings are interesting

### 14.2 Recommendation Summary

**For ICASSP 2025**: ❌ **Not Recommended** (major revision needed)

**For WASPAA/AES**: ✅ **Recommended** (after baseline comparisons)

**For ArXiv**: ✅ **Recommended** (can submit now as technical report)

**For Workshop**: ✅ **Highly Recommended** (good fit, interesting findings)

### 14.3 Key Improvements Needed

**Critical** (Must-have):
1. ❌ Add baseline comparisons (VisualEchoes, Sep-Stereo)
2. ❌ Expand evaluation to 100+ samples
3. ❌ Write paper draft

**Important** (Should-have):
4. ⚠️ User study (perceptual evaluation)
5. ⚠️ Fix or remove IR modeling
6. ⚠️ Report tracking accuracy

**Nice-to-have**:
7. Multi-source extension
8. Cross-dataset validation
9. Real-time demo

### 14.4 Estimated Time to Publication

**Conservative Estimate**:
- WASPAA-quality paper: **4-6 weeks**
- ICASSP-quality paper: **8-12 weeks**

**Aggressive Estimate** (if focused):
- Workshop paper: **2 weeks**
- ArXiv report: **1 week**

---

## 15. Conclusion

Vid2Spatial is a **solid engineering effort** with **interesting empirical findings** (IR degrades performance), but falls **significantly short** of publication standards for top-tier venues in its current state.

**Key Gaps**:
- Evaluation too small (20 samples)
- No state-of-the-art comparisons
- Limited novelty (geometric approach is incremental)
- Paper not written

**Path Forward**:
1. **Short-term** (1-2 weeks): Complete baselines, expand to 100 samples, submit to ArXiv/workshop
2. **Medium-term** (4-6 weeks): Add user study, improve IR, write WASPAA paper
3. **Long-term** (2-3 months): Multi-source extension, cross-dataset, aim for ICASSP 2026

**Bottom Line**: This is **good work** that needs **more work** before it's **publishable**.

---

**Evaluator**: Independent Academic Reviewer
**Date**: 2025-11-30
**Recommendation**: **Major Revision Required** → Target workshop/ArXiv first, then build toward ICASSP 2026.
