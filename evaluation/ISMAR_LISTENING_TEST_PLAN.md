# ISMAR 2026 — Vid2Spatial 정성평가(Listening Test) 계획

## 1. 평가 목표

**Video-guided spatial audio rendering** 의 지각 품질을 전문가 패널로 평가.
Proposed (HRTF binaural) vs Baseline (stereo pan+reverb) vs Mono (anchor)
3 conditions × 5-point MOS, within-subjects, double-blind.

---

## 2. 테스트 방법론

| 항목 | 선택 | 근거 |
|------|------|------|
| **방법** | MOS (Mean Opinion Score) | 1-5 Likert, ITU-T P.800 기반, ISMAR/CHI 논문에서 표준 |
| **Scale** | 5-point (1=Bad, 2=Poor, 3=Fair, 4=Good, 5=Excellent) | 7-point 대비 응답 부담↓, 통계 해석 용이 |
| **Design** | Within-subjects, double-blind | Proposed/Baseline 순서 랜덤, "Version A/B" 라벨 |
| **Anchor** | Mono (no spatialization) | 하한 레퍼런스, 별도 평가 |
| **비교 단위** | 클립별 3 conditions 각각 독립 평가 | Paired comparison 대비 MOS가 절대 품질 비교 가능 |

---

## 3. 평가 차원 (4개)

| # | 차원 | 질문 (영문) | Scale 앵커 |
|---|------|-------------|-----------|
| Q1 | **Spatial Alignment** | "How well does the sound position match the visual object?" | 1=완전 불일치, 5=정확히 일치 |
| Q2 | **Motion Smoothness** | "How smooth and natural is the spatial audio movement?" | 1=끊김/클릭 심함, 5=완전 자연스러움 |
| Q3 | **Depth Perception** | "How convincing are the near/far distance changes?" | 1=거리감 없음, 5=자연스러운 거리감 |
| Q4 | **Overall Quality** | "Overall, how would you rate this spatial audio?" | 1=Bad, 5=Excellent |

---

## 4. 비디오 클립 세트: 12개

### 선정 기준
- **모노 앵글, 단일 소스** (컷 전환 없음) — 파이프라인 설계에 맞는 평가
- **다양한 motion speed**: slow / moderate / fast
- **다양한 spatial dynamics**: 좌우 sweeping / 근접-원거리 / 중앙 고정 / center crossing
- **다양한 object category**: 동물, 탈것, 사람활동, 악기
- **10초 길이** 통일 (평가 피로 관리)

### 12개 클립 구성

| # | Clip ID | Category | Motion | 핵심 특성 | 음원 (spatamb) |
|---|---------|----------|--------|-----------|----------------|
| 1 | car-10 | Vehicle | Moderate | 좌→우 sweeping, 거리 변화 | Trumpet (Entertainer) |
| 2 | dog-1 | Animal | Fast | 급격한 방향 전환, 5회 center crossing | Violin (Jupiter) |
| 3 | horse-1 | Animal | Fast | 115° wide sweep, 큰 거리 변화 | Flute (Allegro) |
| 4 | motorcycle-6 | Vehicle | Moderate | 우→좌 이동, 2회 center crossing | Saxophone (Entertainer) |
| 5 | skateboard-18 | Sports | Moderate | 빈번한 좌우 진동, 6회 center crossing | Clarinet (Pavane) |
| 6 | train-17 | Vehicle | Slow | 좌→우, 가까운 거리 고정 | Trombone (GString) |
| 7 | guitar-9 | Instrument | Static | 거의 정지, 거리 일정 — negative control | Cello (Jupiter) |
| 8 | drone-2 | Drone | Slow | 미세한 우측, 거리 일정 | Oboe (Maria) |
| 9 | dog-15 | Animal | Moderate | 좌측 체류, 중간 거리 | Viola (Hark) |
| 10 | horse-15 | Animal | Fast | 극적 크기 변화, 빠른 접근 | Horn (Slavonic) |
| 11 | car-5 | Vehicle | Slow | 중앙 부근 안정, 미세한 좌→우 | Bassoon (Fugue) |
| 12 | motorcycle-17 | Vehicle | Moderate | 좌→우 sweeping, 안정적 추적 | Double Bass (Rondeau) |

**음원 할당**: 12개 클립에 **12종 고유 악기** (중복 없음). 모든 소스는 spatamb/Dataset의 스튜디오 녹음 분리 트랙(48kHz mono).

### 클립 분포 확인

| Motion Speed | 클립 수 | 클립 |
|-------------|---------|------|
| **Slow** | 3 | train-17, drone-2, car-5 |
| **Moderate** | 5 | car-10, motorcycle-6, skateboard-18, dog-15, motorcycle-17 |
| **Fast** | 2 | dog-1, horse-1, horse-15 |
| **Static** | 2 | guitar-9, drone-2 |

| Spatial Feature | 클립 |
|-----------------|------|
| Wide L/R sweep | horse-1, car-10, motorcycle-6 |
| Center crossing | dog-1(5), skateboard-18(6), motorcycle-6(2) |
| Large distance change | horse-1, dog-1, car-10 |
| Near-field fixed | train-17, guitar-9 |
| Approach (far→near) | horse-15, dog-1 |
| Recede (near→far) | motorcycle-3, car-10 |

---

## 5. Conditions (3개)

| Condition | Method | Blind Label |
|-----------|--------|-------------|
| **Proposed** | HRTF binaural (KEMAR SOFA, per-20ms block direct convolution) + d_rel gain/LPF + stereo reverb | "Version A" or "B" (random) |
| **Baseline** | Stereo pan (sin law) + d_rel gain/LPF + distance reverb | "Version A" or "B" (random) |
| **Mono** | Dual-mono (no spatialization) | "Mono Reference" (labeled, not blind) |

**Proposed vs Baseline**: double-blind (랜덤 A/B 배정)
**Mono**: open label anchor (하한 레퍼런스로만 사용, MOS 점수 수집은 하지만 주 비교 대상 아님)

---

## 6. 참가자

| 항목 | 내용 |
|------|------|
| **인원** | 10–12명 (목표 10명 유효 데이터) |
| **자격** | 오디오/AR/VR 분야 전문가 또는 관련 연구 경험자 |
| **청력** | 정상 청력 자가 보고 (audiometry 불필요) |
| **헤드폰** | 오버이어 헤드폰 (AKG K371 등 closed-back 권장, 모델 기록) |
| **환경** | 조용한 개인 공간, 외부 소음 최소화 |
| **보상** | 소정의 사례비 또는 기프트카드 |

### 통계적 충분성
- 12 clips × 3 conditions × 4 questions × 10 participants = **1,440 ratings**
- Paired t-test (proposed vs baseline): 12 clips × 10 = 120 paired observations per question
- Cohen's d ≥ 0.5 (medium effect), α=0.05, power=0.80 → 필요 N≈8–10명 (within-subjects)
- 10명이면 충분, 2명 버퍼 포함 12명 모집

---

## 7. 테스트 절차 (1인당 ~20분)

```
Phase 1: 안내 + 연습 (3분)
├── 서면 동의서
├── 헤드폰 착용, 볼륨 조정
├── 연습 클립 1개 (eval set에 미포함)
│   └── Proposed / Baseline / Mono 전부 들어보고 질문 확인
└── 질문 응답 (앵커 의미 확인)

Phase 2: 본 평가 (15분)
├── 12 clips × 3 conditions = 36 trials
├── 클립 순서: Latin square balanced (참가자별 다른 순서)
├── 조건 순서: 클립 내 A/B 랜덤 + Mono 마지막
├── 각 trial:
│   1. 비디오 재생 + 해당 condition 오디오 (10초)
│   2. 4개 질문 각각 1-5 MOS 선택 (5초)
│   3. 필요 시 재청취 가능
│   → trial 당 ~25초, 36 trials = ~15분
└── 중간 휴식 허용 (6번째 클립 후 자유 휴식)

Phase 3: 사후 설문 (2분)
├── 헤드폰 모델
├── 오디오/AR 경험 수준 (1-5)
├── 자유 코멘트
└── 감사 + 보상
```

---

## 7-1. Loudness Normalization (구현 완료)

**문제**: Proposed (HRTF binaural)은 peak-normalized → RMS ~0.10-0.22, Baseline은 pan 방식이라 peak ~0.22-0.41 → **최대 ~11 dB 차이**. 이 상태로 테스트하면 loudness bias 발생.

**해결**: 모든 condition (proposed, baseline, mono)을 **동일한 target RMS = 0.08 (~-22 dBFS)**로 정규화.

| 단계 | 내용 |
|------|------|
| 1. 음원 전처리 | Crest factor > 10인 음원에 soft limiter 적용 (특히 piano: 22.3 → 6.8) |
| 2. 렌더링 | Proposed (HRTF binaural), Baseline (stereo pan+reverb), Mono 각각 렌더 |
| 3. RMS 정규화 | 3개 condition 모두 TARGET_RMS = 0.08로 스케일링 |
| 4. Peak 안전 | peak > 0.99이면 0.99로 soft clip |
| 5. 검증 | 12/12 클립 × 3 conditions = **36개 파일 모두 RMS = 0.080000** (spread = 0) 확인 |

**공정성 보장 사항**:
- Reverb: 동일 (`schroeder_ir(sr, rt60=0.4)`) → Proposed = Baseline
- Gain/LPF: 동일 (`apply_distance_gain_lpf(d_rel_s)`) → Proposed = Baseline
- 유일한 차이: **HRTF binaural rendering vs. stereo pan law**

---

## 8. 구현 플랜

### 8-1. 신규 4개 클립 준비
```
dog-15, horse-15, car-5, motorcycle-3
├── LaSOT에서 10초 구간 추출 (가장 dynamic한 300 프레임)
├── GT bbox → trajectory_3d.json 생성
├── audio_mono.wav 할당 (기존 음원 풀에서)
├── 렌더: proposed.wav, baseline.wav, mono.wav, foa.wav
└── 리뷰 비디오 생성 + 검증
```

### 8-2. Stimuli 패키징
```
evaluation/listening_test/stimuli/
├── config.json (12 clips × 3 conditions × 4 questions)
├── clip_01/ (video.mp4, proposed.wav, baseline.wav, mono.wav)
├── clip_02/
├── ...
└── clip_12/
```

### 8-3. 웹 인터페이스 업데이트
- 기존 `evaluation/listening_test/index.html` 수정
- 7-point → **5-point scale** 변경
- 질문 텍스트 업데이트
- Latin square 클립 순서 구현
- 연습 trial 추가
- 서버 응답 저장 (server.py)

### 8-4. 분석 파이프라인
```python
# 주요 분석
1. Per-question MOS: mean ± 95% CI (proposed vs baseline vs mono)
2. Paired t-test: proposed vs baseline (per question)
3. Wilcoxon signed-rank test: 비모수 대안
4. Effect size: Cohen's d
5. Per-clip breakdown: 어떤 motion type에서 차이가 큰지
6. Radar chart: 4 dimensions × 3 conditions
```

---

## 9. 예상 결과 보고 (논문용)

### Table 형식
```
┌──────────────────┬──────────┬──────────┬──────────┐
│ Dimension        │ Proposed │ Baseline │ Mono     │
├──────────────────┼──────────┼──────────┼──────────┤
│ Spatial Alignment│ 4.2±0.3  │ 3.5±0.4  │ 1.8±0.3  │
│ Motion Smoothness│ 4.0±0.3  │ 3.8±0.3  │ 2.0±0.4  │
│ Depth Perception │ 3.8±0.4  │ 3.2±0.4  │ 1.5±0.3  │
│ Overall Quality  │ 4.1±0.3  │ 3.4±0.4  │ 1.7±0.3  │
└──────────────────┴──────────┴──────────┴──────────┘
* p < 0.01 (paired t-test, proposed vs baseline)
```

### Figure 구성
1. **Bar chart**: 4 dimensions × 3 conditions with 95% CI error bars
2. **Radar chart**: 4-axis spider plot, 3 conditions 겹쳐서
3. **Per-motion-type breakdown**: slow/moderate/fast 별 MOS 비교
4. **Box plot**: 참가자 간 variation 시각화

---

## 10. 컷 전환/멀티 소스 필요 여부

**아니오 — 모노 앵글, 단일 소스로 충분.**

근거:
- Vid2Spatial의 핵심 기여는 **단일 비디오에서 단일 객체의 3D trajectory 추출 → spatial audio rendering**
- 컷 전환/멀티 소스는 별도 확장(multi_source.py)이며 핵심 평가 대상이 아님
- 단일 소스만으로 HRTF vs pan 차이가 명확히 드러남
- 학회 리뷰어도 핵심 파이프라인의 single-source 평가를 기대

**단, 1-2개 "보너스 클립"으로 multi-source scene을 추가하면 demo 목적으로 효과적** (평가 대상은 아니고 논문 figure/demo 용도)

---

## 11. 타임라인

| 단계 | 소요 | 내용 |
|------|------|------|
| 1. 신규 4클립 준비 | 1일 | bbox 추출, trajectory, 렌더, 검증 |
| 2. 웹 인터페이스 업데이트 | 1일 | 5-point, 12 clips, Latin square |
| 3. 파일럿 테스트 | 1일 | 본인 + 1-2명 dry run |
| 4. 본 평가 실시 | 3-5일 | 10-12명 스케줄링 |
| 5. 분석 + 논문 반영 | 1일 | 통계 + figure 생성 |
| **Total** | **~7-9일** | |
