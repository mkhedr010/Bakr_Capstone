# CAPSTONE MODEL STUDY PLAN: PointNet, PointNet++, RandLA-Net

> **Audience:** Claude Code (coding agent) working in a local workspace.
> **Operator:** a university capstone student ("Student C"), computer hardware engineering background, **no prior ML/deep-learning experience**.
> **Read this whole file before doing anything.** Then follow the phases in order. Keep `PROGRESS.md` updated (see §3).

---

## 0. TL;DR (what you are doing)

You will **study three foundational point-cloud deep-learning models**: **PointNet**, **PointNet++**, **RandLA-Net**. You will:

1. Read their papers and code and document the **architecture, procedure, requirements, segmentation capabilities, and how each captures geometric context**.
2. **Actually run them** (smoke tests, then training/evaluation on a public mobile-LiDAR dataset, then compatibility tests on the lab's own data).
3. **Measure** feasibility: accuracy (esp. thin classes like road markings), memory, speed, scalability, setup pain.
4. Produce a **comparison and "what to use / what to merge" analysis** that lets the team choose a baseline and design its project contribution.

The output is a **set of reports + reproducible runbooks + result tables** (§12). Everything must be **evidence-backed and honestly labeled**. A wrong-but-confident report is worse than an incomplete honest one.

---

## 1. Project context (so you understand why)

**The team's capstone** (group of four): automatic **feature extraction / semantic segmentation of road features** from **mobile laser scanning (MLS)** point clouds, using ML and deep learning.

**Data source:** a university lab's mobile mapping system: LiDAR + GNSS/GPS + orientation/position sensors (IMU). Time-stamped sensor data is processed into **georeferenced, dense 3D point clouds** (likely LAS/LAZ/PLY or similar; **format, density, attributes, and label status are NOT yet confirmed. Phase 0 must discover them**). Previous student teams collected field-trip data; the student has access to a good amount of it.

**Road features of interest** (taxonomy is being defined by a teammate; assume at least): road surface, **road markings**, curbs, poles (light/utility/sign), traffic signs, traffic lights, guardrails/barriers, fences, buildings, vegetation, cars, utility lines. Thin/small classes (markings, poles, signs, fences) are known to be the hardest.

**Why these three models:** they are the foundational lineage the rest of the field builds on:
- **PointNet** (Qi et al., CVPR 2017): consumes raw unordered points; shared MLP + max-pool; **no local structure**.
- **PointNet++** (Qi et al., NeurIPS 2017): hierarchical local grouping (set abstraction); handles non-uniform density; **slow FPS sampling**.
- **RandLA-Net** (Hu et al., CVPR 2020; TPAMI version arXiv:2107.02389): **random sampling + local feature aggregation** for **whole large scenes in one pass**; evaluated on Toronto-3D, an MLS road dataset, in the TPAMI version.

**What the student already established from the literature (use as hypotheses to test, not as facts):**
- Large classes (road, building, vegetation, car) are largely solved; **thin classes are weak**. On Toronto-3D, published baselines score poorly on road marking and fence, and the dataset authors say intensity/color would help.
- Published Toronto-3D numbers (re-verify in the papers): RandLA-Net XYZ-only ≈ 77.7 mIoU / 92.95 OA, road-marking IoU ≈ 42.6, fence ≈ 37.1; with RGB ≈ 81.8 mIoU, road-marking ≈ 64.2. PointNet++ ≈ 56.55 mIoU in the Toronto-3D paper. Label revisions can change numbers.
- Cross-sensor/cross-domain generalization is a recognized weakness of the field (important because the lab's sensor differs from benchmark sensors).
- Newer backbones (Point Transformer V3, Sonata self-supervised pretraining) exist but are **out of scope for this study** (other phases will cover them).

---

## 2. Mission: the questions this study must answer

At the end, the team must be able to answer, **with evidence**, each of the following. Every report section maps to these.

### Per-model questions (Q-M)
- **QM1 Architecture:** What exactly are the blocks, tensor shapes, parameter count, and data flow? (verified by instantiating the model, not just reading)
- **QM2 Procedure:** What are the exact steps to go from raw point cloud → training-ready data → trained model → predictions → metrics? (a runbook that actually worked)
- **QM3 Requirements:** Software stack and versions, hardware (GPU memory, RAM, disk), custom CUDA ops, input format and feature channels, labels format, licenses, time to train.
- **QM4 Segmentation capabilities:** What can it do (classification / part / semantic scene / instance)? At what scene scale? What can it NOT do?
- **QM5 Geometric context:** How does it see neighborhood geometry (global feature? ball query radii? kNN + relative position encoding?), how large is its **effective receptive field in practice**, and how does it behave when density or scene scale changes?
- **QM6 Feasibility:** Measured accuracy (OA, mIoU, per-class IoU), thin-class performance, peak GPU memory, throughput, max points per pass, training time, inference time per million points and extrapolated per kilometer of road.
- **QM7 Behavior on our kind of data:** Does it run on the lab's data? What preprocessing is required? What breaks?
- **QM8 Failure modes:** Where and why does it fail (which classes, which confusions, which conditions)?
- **QM9 Reusable ideas:** Which components are worth borrowing or combining?

### Cross-model questions (Q-X)
- **QX1** Side-by-side comparison matrix (accuracy, cost, complexity, scalability, thin-class ability, ease of setup, risk).
- **QX2** Which model is the best **baseline** for the project, and why?
- **QX3** **Merge hypotheses:** which components of which models could be combined to fix which measured weakness (e.g., PointNet++ multi-scale density handling + RandLA-Net's efficient sampling/aggregation; adding intensity as input; geometric post-processing). Each hypothesis states rationale, evidence level, cost, risk, and a test plan.
- **QX4** What gaps remain that these models do NOT address (feed into the project-definition phase)?

---

## 3. Operating rules (non-negotiable)

### 3.1 Honesty and evidence labels
Every factual claim and number in every report must carry one label:

| Label | Meaning |
|---|---|
| `[RUN]` | Produced by a run you executed. Must point to a log/CSV path. |
| `[CODE]` | Read directly from source code (cite file and line/commit). |
| `[PAPER]` | Stated in the paper (cite paper + section/table). |
| `[UNVERIFIED]` | From memory or secondary source; not checked. Avoid; if used, flag clearly. |

Rules:
- **Never fabricate or "estimate silently."** If a run was not done, write `NOT RUN` and why.
- **Never report a number that is not traceable** to a file in `results/` or `logs/`.
- If your result disagrees with the paper, **report both** and investigate; do not adjust to match.
- Distinguish **official code** from **ports**; record every deviation from the paper (framework, hyperparameters, sampling, loss weights).
- "Reproduced" means within **±3 mIoU** of the published figure on the same split and features; otherwise write "not reproduced" and list suspected causes.

### 3.2 Safety and hygiene
- **Do not modify, move, or delete the student's raw data.** Work on copies/derived files under `workspace/data/derived/`. Treat the raw path as read-only.
- Do not delete anything outside your own working directories. Do not `rm -rf` broadly.
- Use an isolated environment per implementation (`venv`/`conda`). Never install system-wide packages without telling the human first.
- Do not push to any remote. Do not upload data anywhere. Do not commit large files (datasets, checkpoints) to git; use `.gitignore`.
- Before downloading anything large (> **5 GB**) or with an unclear license, stop and ask the human.
- **Respect licenses.** Note each repo's and dataset's license in the requirements report (e.g., the official RandLA-Net repo is CC BY-NC-SA 4.0: non-commercial; verify).
- If network access is restricted, do not try to bypass it. State what is blocked and what the human must download manually (exact URLs and target paths).

### 3.3 Working style
- **Time-box** every task (see §11). If a task exceeds its box, stop, write what you learned, and move on or ask.
- **Commit after each phase** (local git in `workspace/`), message format `phaseN: <summary>`.
- Keep a chronological **`journal/JOURNAL.md`**: every command you ran that matters, what happened, what you decided and why. This is the source of truth for runbooks.
- Maintain **`PROGRESS.md`** at the workspace root: a checklist mirroring this plan (checked only when the phase's acceptance criteria are met), plus "Blockers" and "Questions for human" sections.
- The human is a beginner in ML. Every report starts with a **plain-English summary** and defines jargon on first use. Add terms to `reports/GLOSSARY.md`.
- When blocked on something only the human can decide or provide (data location, GPU access, label status), write the question in `PROGRESS.md → Questions for human`, continue with everything that does not depend on the answer, and mention it in your end-of-turn message.
- You may parallelize independent reading/analysis tasks (e.g., one subagent per model for Phase 2) if your tooling allows; merge their outputs into the common templates and re-verify numbers yourself.

---

## 4. Inputs the human provides (fill in before starting)

The human will fill these in or tell you. If any is missing, **discover it in Phase 0** and record it.

```yaml
RAW_LAB_DATA_PATH: <D:\SEMESTER 7\Capstone project\Data>
LAB_DATA_FORMAT_HINT: <unknown | las | laz | ply | e57 | txt | other>
LAB_DATA_HAS_LABELS: <unknown | yes | no>
LAB_SENSOR_SPEC_AVAILABLE: <unknown | path to datasheet>
GPU_AVAILABLE: <unknown | description>          # Phase 0 will verify with nvidia-smi
WORKSPACE_DIR: ./workspace                        # create if absent
PUBLIC_DATASET_PREFERRED: Toronto-3D              # MLS road dataset, Toronto, Canada
```

---

## 5. Workspace layout (create this)

```
workspace/
├── PROGRESS.md
├── journal/JOURNAL.md
├── env/                    # environment captures (nvidia-smi, pip freeze, conda export, versions)
├── repos/                  # cloned model code (official + ports), one subfolder each
├── data/
│   ├── raw_links/          # symlinks or path notes to student's raw data (read-only)
│   ├── public/toronto3d/   # downloaded public dataset (original files untouched)
│   └── derived/            # converted/tiled/subsampled data produced by you
├── src/                    # YOUR code: loaders, converters, metrics, probes, benchmark harness
├── configs/                # one config per experiment (yaml/json), seeds recorded
├── runs/                   # per-run output: logs, curves, checkpoints (gitignored), predictions
├── results/                # tidy CSV/JSON summaries (the only source for numbers in reports)
├── figures/                # plots, error maps, diagrams
└── reports/
    ├── 00_EXECUTIVE_SUMMARY.md
    ├── 01_model_pointnet.md
    ├── 02_model_pointnet2.md
    ├── 03_model_randlanet.md
    ├── 04_comparison_matrix.md
    ├── 05_merge_hypotheses.md
    ├── 06_lab_data_compatibility.md
    ├── 07_runbooks/ (RUNBOOK_pointnet.md, RUNBOOK_pointnet2.md, RUNBOOK_randlanet.md, RUNBOOK_data_prep.md)
    ├── 08_requirements_table.md
    ├── 09_MCR_STUDENT_C_SUMMARY.md
    ├── HUMAN_STUDY_GUIDE.md
    ├── GLOSSARY.md
    └── REFERENCES.md
```

---

## 6. PHASE 0: Discovery (environment + data)  *[time-box: 2–3 h]*

**Goal:** know exactly what hardware, software, and data you have, so later choices are grounded.

### 6.1 Environment audit → `env/ENVIRONMENT.md`
Run and record outputs (do not install anything yet):
```bash
uname -a; cat /etc/os-release
nvidia-smi                      # GPU model, VRAM, driver, max supported CUDA
nvcc --version                  # may not exist; record
python --version; which python; conda --version 2>/dev/null
free -h; df -h .; nproc
git --version; gcc --version; cmake --version 2>/dev/null
pip --version
```
Record: GPU model + VRAM, CPU cores, RAM, free disk, OS, CUDA driver/toolkit, available Python versions, whether internet works (`curl -sI https://github.com | head -1`, `pip download --no-deps -d /tmp/x tqdm`).

**Decision tree: compute:**
| Finding | Action |
|---|---|
| NVIDIA GPU ≥ 8 GB VRAM | Proceed with full plan. |
| NVIDIA GPU < 8 GB | Full plan but smaller batch/points; document limits; the max-points test (Probe E) becomes more important. |
| No GPU | Do Phases 1–3 fully, Phase 4 **smoke tests on CPU only** (tiny subsets), and write in `PROGRESS.md → Questions for human` asking for GPU access (lab server, university cluster, Colab). Do not pretend CPU results are comparable. |
| Very new GPU (e.g., Ada/Blackwell) | Old TensorFlow 1.x stacks will **not** run. Plan on PyTorch ports (§7). |

### 6.2 Data audit → `reports/06_lab_data_compatibility.md` (section "Raw data audit")
Find and inspect the raw lab data at `RAW_LAB_DATA_PATH` (read-only!):
- File inventory: count, total size, extensions, folder structure (which trip/date/area, if inferable).
- For a representative sample of files (at least 3, varied sizes), read **headers/metadata only first**, then a small subsample of points:
  - Format and version (LAS/LAZ version, point format ID; PLY properties; etc.). Use `laspy`, `plyfile`, or `open3d`; if a library is missing, check `pip` access; if unavailable ask the human.
  - **Number of points**, **bounding box / extent**, **CRS/coordinate magnitude** (are X/Y in UTM-sized numbers like 6-7 digit values? These cause float32 precision problems; note offset/scale).
  - **Available attributes**: XYZ, intensity (range? bit depth?), RGB, return number/number of returns, GPS time, scan angle, classification field (is it populated? values?), user data, point source ID.
  - **Point density**: points/m² on the road surface and overall; point spacing; density variation along the trajectory (speed-dependent) and with range.
  - **Classification/labels:** is any label field populated with meaningful, consistent values? Any companion label files? Document honestly; do NOT assume.
  - Trajectory/GNSS files present? (Note existence only; not needed for this study.)
- Produce a **table**: file/tile → points, extent, attributes present, density, labeled?
- Answer: *"Can this data be used for supervised training now? If not, what is missing?"* Also list what a human would need to do to label a small tile (e.g., CloudCompare) without doing it yourself.

**Acceptance (Phase 0):** `env/ENVIRONMENT.md` and the raw data audit exist, with tables and no guesses. `PROGRESS.md` shows decisions and open questions. Commit.

---

## 7. PHASE 1: Acquire code & choose implementations  *[time-box: 3–4 h]*

**Goal:** have, for each model, (a) **reference code** (official, for fidelity and reading) and (b) a **runnable implementation** on the available hardware.

### 7.1 Candidate sources (verify existence, maintenance, license, and last commit; these are candidates, not guarantees)
| Model | Official (reference) | Likely runnable alternatives to evaluate |
|---|---|---|
| PointNet | `github.com/charlesq34/pointnet` (TensorFlow 1.x) | PyTorch ports, e.g., `yanx27/Pointnet_Pointnet2_pytorch`; Open3D-ML (`isl-org/Open3D-ML`) if it includes it; others you find |
| PointNet++ | `github.com/charlesq34/pointnet2` (TF 1.x, **custom CUDA ops**) | same PyTorch port family; Pointcept (`Pointcept/Pointcept`) if it includes a PointNet++ config; others |
| RandLA-Net | `github.com/QingyongHu/RandLA-Net` (TF 1.11, Python 3.5, CUDA 9.0, tested on Ubuntu 16.04; needs compile step; license CC BY-NC-SA 4.0) | Toronto-3D–specific forks: `github.com/WeikaiTan/RandLA-Net`, `github.com/Me710/RandLA-Net` (Toronto-3D training scripts/results); PyTorch ports; Open3D-ML (verify it has RandLA-Net + Toronto3D support) |
| Dataset | Toronto-3D: `github.com/WeikaiTan/Toronto-3D` (README has download links, class list, attributes, tips) | Paris-Lille-3D, Semantic3D, SemanticKITTI as backups only if Toronto-3D unobtainable |

### 7.2 Selection rubric (record in `reports/07_runbooks/RUNBOOK_*.md → "Implementation choice"`)
Score each candidate on: **fidelity to the paper** (architecture + hyperparameters match?), **runs on this machine's CUDA/driver**, **maintenance/last commit**, **dataset support (Toronto-3D)**, **license**, **code clarity**, **reproduces published numbers**.
- Clone **both** official and the chosen runnable port into `repos/` (shallow clone ok). Record **commit hashes** in `env/REPOS.md`.
- Always read the official code for the architecture even if you run a port. Log every discrepancy between official and port (layer widths, K, radii, loss, sampling, normalization).
- If the official stack can be built (e.g., in a container with old CUDA) in **≤ 3 h**, try it for fidelity; otherwise document why it was abandoned (that itself is a feasibility finding for QM3).

### 7.3 Environments
Create one environment per implementation family. Capture `pip freeze`/`conda env export` to `env/`. Compile custom ops if needed (PointNet++ FPS/ball-query; RandLA-Net `compile_op.sh`); record every error and fix verbatim in the journal; these become the "requirements & gotchas" section.

**Acceptance (Phase 1):** for each model: repo(s) cloned with commit hashes; an environment where `import`+a forward pass on random data works; implementation-choice table filled. Commit.

---

## 8. PHASE 2: Static study (architecture, procedure, requirements, capabilities, context) *[time-box: 1.5–2 days; no GPU needed]*

For each model, fill `reports/0X_model_*.md` using the template in §12.1. Do the following **with code, not just reading**.

### 8.1 Cross-check list (seed facts from the papers; **verify each against paper and code; mark `[PAPER]`/`[CODE]`/mismatch**)

**PointNet** (arXiv:1612.00593)
- Per-point shared MLPs; **max-pool** as the symmetric function → global feature (1024-d); permutation invariance by symmetric function.
- **T-Net**: input transform (3×3) and feature transform (64×64) with orthogonality regularizer (reported weight 0.001).
- Segmentation head concatenates **per-point local feature (64-d) + global feature (1024-d) = 1088-d**, then MLP (512,256,128) → (128, m).
- S3DIS protocol in paper: 1 m × 1 m blocks, **4096 points sampled per block**, 9-d input (XYZ, RGB, normalized room location); Adam lr 0.001, batch 32, lr halved every 20 epochs.
- Reported: S3DIS mIoU 47.71 / OA 78.62; robust to missing points; no local neighborhood modeling.

**PointNet++** (arXiv:1706.02413)
- **Set abstraction (SA)** = Sampling (FPS) + Grouping (ball query) + mini-PointNet; relative coordinates within groups.
- **MSG** (multi-scale grouping) and **MRG** (multi-resolution grouping); **random input dropout** (ratio up to 0.95) during training for density robustness.
- Segmentation via **feature propagation**: inverse-distance weighted interpolation (k=3, p=2) + skip links + "unit PointNet".
- Ball query vs kNN: paper prefers ball query for local pattern generalization; MSG ≈ 2× cost of single-scale grouping.
- ScanNet protocol: 1.5 × 1.5 × 3 m cubes, 8192 points; predictions merged by majority vote across cubes.
- **FPS is O(N²)**; RandLA-Net paper reports ~200 s to sample 10% of 1M points. **Check in code why and how it is implemented.**

**RandLA-Net** (arXiv:1911.11236; TPAMI version arXiv:2107.02389)
- **Random sampling** (O(1)) instead of FPS; **local feature aggregation (LFA)** = Local Spatial Encoding (LocSE: relative position encoding of K=16 neighbors using `[p_i, p_i^k, p_i−p_i^k, ‖p_i−p_i^k‖]`) + **attentive pooling** + **dilated residual block** (2 units chained, receptive field ≈ K² neighbors).
- Encoder: 4 layers, 4× decimation each (N → N/4 → … → N/256), feature dims 8→32→128→256→512; decoder: nearest-neighbor upsampling + skip connections + shared MLPs; head FC 64→32→n_class with dropout 0.5.
- Training: Adam lr 0.01, 5% decay per epoch, ~10⁵ points per training sample; whole cloud at test time without block partition.
- Reported: 1.24 M params; ~1.03 M points in one pass; 77.4 mIoU Semantic3D, 53.9 SemanticKITTI, 70.0 S3DIS; weak on minority classes/small objects.
- Official repo setup: Python 3.5, TF 1.11, CUDA 9.0, cuDNN 7.4.1, Ubuntu 16.04; compile ops; >64 GB RAM preferred for Semantic3D.

### 8.2 Required static analyses (per model; commit scripts in `src/analysis/`)

**A. Architecture trace (QM1)**
1. Instantiate the model with the **dataset's real configuration** (n_classes=8 for Toronto-3D, input feature dim for each feature set).
2. Print a **layer-by-layer table**: layer name, operation, input shape, output shape, #params. Use forward hooks (or `torchinfo` if available). For TF models, use the framework's summary or manual shape tracing.
3. Report **total trainable parameters**; compare to paper (PointNet ≈ 3.5 M for classification incl. T-Nets, vanilla 0.8 M; RandLA-Net 1.24 M) and explain differences.
4. Produce an architecture diagram (Mermaid in markdown **and** a PNG in `figures/`), annotated with shapes for a concrete example input (e.g., N = 4096 for PointNet; N = 16384 / 40960 for RandLA-Net; record what you use).
5. Write the **forward pass in plain English**, step by step, for one input (what happens to a single point from input to logits).

**B. Input/output contract (QM2, QM3)**
- Exact tensors: names, shapes, dtypes, value ranges, normalization expected (e.g., unit sphere, centered, local coordinates), required neighbor-search precomputations (kNN indices computed in the dataloader?), required labels format (int per point, ignore index?).
- Which **feature channels** are supported (XYZ only? +RGB? +intensity?) and what code changes are needed to add intensity.
- Loss function, class weighting, ignore-label handling, optimizer, scheduler, augmentation list, default hyperparameters (table).

**C. Procedure summary (QM2)** → `RUNBOOK_*.md` outline (commands filled in Phases 3–4):
1. Environment setup → 2. Data preparation → 3. Training → 4. Validation/checkpointing → 5. Inference on whole scenes (tiling/voting if applicable) → 6. Evaluation → 7. Visualization/export of predictions.

**D. Capabilities matrix (QM4)** (per model, with evidence label):
| Capability | Supported? | Notes / how |
|---|---|---|
| Object/shape classification | | |
| Part segmentation | | |
| Semantic segmentation (scene) | | block-based vs whole-scene |
| Instance segmentation | | (RandLA-Net authors list it as future work; check) |
| Whole-scene single pass | | max N measured in Probe E |
| Handles intensity/RGB | | |
| Handles variable density | | |
| Needs pre-partitioning | | block size, overlap, voting |
| Multi-GPU / batching limits | | |

**E. Geometric-context mechanism (QM5): "geometric context memory"**
Describe, per model, **how local/global geometry is captured and retained**:
- PointNet: global max-pooled descriptor concatenated per point; critical-point set; no neighborhoods.
- PointNet++: nested neighborhoods (radii per level), MSG/MRG, interpolation back to all points.
- RandLA-Net: kNN (K=16) per layer, relative position encoding, attentive pooling, dilation by stacking, receptive field growth through decimation (4 layers).
Produce a **table**: neighborhood definition (kNN vs ball), radii/K per level, #levels, theoretical receptive field (points and meters where inferable), what is lost (e.g., random sampling dropping points), and the **empirical receptive-field probe** (Probe A, §10).

**F. Requirements table (QM3)** → fill `reports/08_requirements_table.md`, one column per model: language/framework + version, Python, CUDA/cuDNN, GPU VRAM needed for training (measured), RAM, disk, custom ops (compile?), dataset prep scripts, training time per epoch (measured), pretrained weights available?, license, last upstream commit, **setup difficulty score (1–5) with justification**, known issues (read GitHub issues of each repo; summarize the top recurring build/run problems).

**Acceptance (Phase 2):** each model report has sections 1–6 and 11 of the template filled, with every number labeled; parameter counts verified by instantiation (or `NOT POSSIBLE` with reason). Commit.

---

## 9. PHASE 3: Data pipeline  *[time-box: 1–1.5 days]*

### 9.1 Public dataset: Toronto-3D (the benchmark for real runs)
- Follow the dataset repo README for download; **check download size and license; ask human if > 5 GB** or if links are broken (the human may need to download manually).
- Expected (verify): 4 tiles (L001–L004), ~78.3 M points, **8 classes + unclassified(0)**: Road, Road marking, Natural, Building, Utility line, Pole, Car, Fence. Attributes: XYZ, RGB, Intensity, GPS time, Scan angle rank. **XY stored in UTM**: subtract an offset before float32 operations (the README warns about grid-subsampling precision issues).
- Standard protocol (verify in paper/README): **train on L001, L003, L004; test on L002**.
- **Validation without test leakage:** hold out a spatially contiguous slice (e.g., last 10–15% along the road axis) of the training tiles for validation/checkpoint selection. Report L002 **once** per final model. (Many papers validate on L002; if you also report that convention, label it clearly as "L002 used for selection".)
- Write `src/data/toronto3d_loader.py` + stats script. Output `results/dataset_stats_toronto3d.csv`: points per class per tile, class frequencies, intensity range/histogram, density, extent. **Class imbalance table** is critical (it drives loss weights and thin-class findings).

### 9.2 Lab data conversion (same pipeline, for compatibility tests)
- Build `src/data/lab_to_common.py`: read lab format → common internal representation: `float32 xyz` (offset-subtracted, offset saved), `float32 intensity` (normalized with the **method recorded**; note intensity needs calibration/normalization across range/angle, so just document raw vs normalized ranges), optional `uint8 rgb`, optional `int label`.
- Record offset, scale, and normalization in a sidecar JSON per tile.
- Tile a few representative lab sections (e.g., 50–100 m of road) into `data/derived/lab_tiles/`. Never overwrite raw.
- **Compare lab vs Toronto-3D** in `reports/06_lab_data_compatibility.md`: density (pts/m²), intensity distribution, attribute availability, extent per tile, scanner/trajectory differences (if spec known). Flag domain-shift risks.

### 9.3 Per-model preprocessing (implement in `src/data/`, document in `RUNBOOK_data_prep.md`)
| Model | Required preprocessing (verify against code) |
|---|---|
| PointNet | Block partition (start 1 m × 1 m as in paper; also test larger blocks suited to road scale, e.g., 2–4 m, documented as a deviation), sample fixed N (4096 default) per block, per-block normalization, overlap + voting at inference |
| PointNet++ | Cube/block partition (paper: 1.5 × 1.5 × 3 m, 8192 points), fixed N sampling, overlap + majority vote at inference |
| RandLA-Net | Grid subsampling (record voxel size, e.g., values used by the Toronto-3D scripts), build kd-tree/KNN structures, sample ~10⁵ points per training sample from whole tiles, full-cloud inference with projection of predictions back to the original points |

Create **identical train/val/test splits** and the **same feature sets** for all models:
- **F0** = XYZ; **F1** = XYZ + intensity; **F2** = XYZ + RGB; **F3** = XYZ + RGB + intensity. (Point coordinates must be offset-normalized the way each model expects; document differences.)

**Acceptance (Phase 3):** dataset stats table; loaders for each model produce a valid batch (assert shapes, dtype, label range, no NaN); lab tile conversion works end-to-end; `RUNBOOK_data_prep.md` written with commands that actually ran. Commit.

---

## 10. PHASE 4: Experiments  *[time-box: 3–5 days wall-clock, compute-dependent]*

### 10.1 Experiment harness (build once, `src/bench/`)
Common utilities used by all models:
- `metrics.py`: confusion matrix → **OA, mIoU, per-class IoU, per-class precision/recall/F1**; ignore label 0 (unclassified). Unit-test it on a tiny hand-computed example (include test).
- `instrument.py`: wall-clock per epoch/iteration; **peak GPU memory** (`torch.cuda.max_memory_allocated` or TF equivalent / `nvidia-smi` polling); throughput (points/s); params count.
- `logger.py`: writes every run to `runs/<exp_id>/` with `config.yaml` (full hyperparameters + seed + git commit of repo and `src/`), `train_log.csv`, `val_metrics.csv`, `test_metrics.json`, `confusion_matrix.csv`, best/last checkpoint (gitignored), `predictions/*.ply` or `.npy`.
- `aggregate.py`: builds `results/summary_all_runs.csv` (**the only source for numbers in reports**).
- Seeds fixed and recorded; note non-determinism (random sampling, CUDA ops).

### 10.2 Stage S: Smoke and sanity tests (each model) [P0]
1. **Forward/backward on random data**: no errors, loss finite.
2. **Overfit test**: train on a **single small block/tile** (e.g., ≤ 20k points) until train accuracy ≳ 95–99%. If the model cannot overfit a tiny set, there is a bug; fix before any real run. Save curve in `figures/`.
3. **Determinism check**: same seed twice → compare first-iteration loss.
4. Record time per iteration and memory to size later runs.

### 10.3 Stage T: Toronto-3D benchmark runs

**Fairness protocol (write in `configs/PROTOCOL.md`):**
- Same train/val/test split and class set for all models; same feature sets per comparison.
- Use each model's **paper/default hyperparameters first** (document); do not tune per model except documented, equal-effort changes (e.g., learning-rate sanity). Where a model **cannot** consume the same input (e.g., block-based), document the adaptation.
- Same stopping rule: train until validation mIoU has not improved for *P* evaluations (choose P, record) or a **maximum wall-clock budget** (record). Report epochs actually run.
- **Class imbalance:** use each model's default loss first (RandLA-Net uses class-weighted loss; check code). In P1, add an ablation with/without weighting.
- Evaluate the **best-validation checkpoint on L002 once**.

**Experiment matrix (priorities):**

| ID | Model | Features | Priority |
|---|---|---|---|
| T1 | RandLA-Net | F0 (XYZ) | **P0** |
| T2 | PointNet++ | F0 | **P0** |
| T3 | PointNet | F0 | **P0** |
| T4 | RandLA-Net | F3 (XYZ+RGB+I) | P1 |
| T5 | PointNet++ | F3 | P1 |
| T6 | PointNet | F3 | P1 |
| T7 | RandLA-Net | F1 (XYZ+I) | P1 *(key question: does intensity alone rescue road markings without RGB? Our lab data may lack RGB)* |
| T8 | RandLA-Net | F2 (XYZ+RGB) | P2 |
| T9 | PointNet++/PointNet | F1 | P2 |
| T10 | repeat T1–T3 with 3 seeds | F0 | P2 (report mean ± std) |
| T11 | RandLA-Net, weighted vs unweighted loss | F0 | P2 |

**Reproduction check (important):** compare T1 (and T4 if RGB) with the published Toronto-3D numbers (re-read them from the papers). Within ±3 mIoU → "reproduced". Else document the gap and the suspects (label version, subsampling, epochs, sampling randomness).

### 10.4 Stage L: Lab data runs
- **If lab data has trustworthy labels:** map lab classes → Toronto-3D class set (document mapping); evaluate best Toronto-3D-trained models on the lab tiles **(cross-domain test)**; then, if budget allows, train/fine-tune on lab data (small split) and compare. Report the **domain-shift drop** per class.
- **If no labels:** run **inference only** with the trained models on lab tiles. Produce: colored prediction PLYs, rendered screenshots (top-down and oblique) in `figures/`, class-distribution of predictions, runtime/memory, and a list of **visible failure patterns** (qualitative; label as `[RUN]-qualitative`, never as accuracy). Note which features were available (e.g., no RGB → use F0/F1 models only).
- Extrapolate inference cost: **seconds and GPU-memory per km of road** at the lab's density.
- Write the exact steps a human would take to **label a small lab tile** for future quantitative evaluation (tool suggestions, class scheme, time estimate), without doing the labeling.

### 10.5 Analysis probes (run on trained models; save to `results/probes/`)

| Probe | Question | Method |
|---|---|---|
| **A: Effective receptive field** (QM5) | How many points / how far in meters actually influence a point's prediction? | Pick ≥ 30 target points (stratified: include thin-class points). With model in eval mode and **fixed sampling seed**, take the logit of the true (or predicted) class at the target point, backprop to the **input tensor** (xyz + features). Influence set = input points with gradient magnitude > ε·max (ε documented, e.g., 1e-6); report count, max distance, distance histogram. For PointNet expect influence via critical points of the global feature; PointNet++/RandLA-Net expect growth with levels. State limitations (non-differentiable neighbor indices, randomness of sampling) in the report. |
| **B: Density robustness** | How do models degrade when density drops? | Test-time random point dropout at 0/25/50/75 % and voxel-size sweep (e.g., 3 sizes); report mIoU and thin-class IoU. (Relevant because the lab's density differs from Toronto-3D and varies with vehicle speed.) |
| **C: Feature ablation** | Do intensity/RGB fix thin classes? | From matrix T1/T4/T7/T8 etc.; table of per-class IoU (road marking, pole, fence, utility line) per feature set. |
| **D: Block/scene scale** | Does context size matter? | For block-based models, vary block size (e.g., 1 m, 2 m, 4 m) at fixed N; for RandLA-Net vary points per sample / subsample voxel size. |
| **E: Scalability** | What is the max N per pass and how do memory/latency scale? | Inference on N ∈ {2¹³, 2¹⁴, …, up to OOM} random subsets of a tile; record peak memory, latency, success/OOM → `figures/scalability_*.png`. Include the sampling/neighbor-search time breakdown if profile tools allow. |
| **F: Failure analysis** | Where and why does it fail? | Confusion matrices; top confused class pairs; **spatial error maps** (PLY with error coloring; screenshots); per-class recall vs object size; qualitative notes on road markings (e.g., markings predicted as road because geometry is coplanar), poles vs tree trunks, fences vs buildings, utility lines at tile borders. |
| **G (P2): Component ablations** | Which component matters? | PointNet: with/without T-Net. PointNet++: SSG vs MSG (and MRG if available). RandLA-Net: max-pool vs attentive pooling; with/without LocSE if code permits cheaply. Short runs only; label as such. |

**Acceptance (Phase 4):** `results/summary_all_runs.csv` exists with at least P0 runs; every number in reports traceable to it; reproduction check written; Probes A, B, E done at minimum (others as time allows). Commit.

---

## 11. Prioritization, time-boxing, and blocker handling

**Order of work** (so usable outputs appear early): Phase 0 → 1 → (Phase 2 static study **in parallel with** Phase 3 data pipeline) → Phase 4 S → Phase 4 T (P0) → probes A/B/E → Phase 4 L → Phase 4 T (P1) → remaining probes/P2 → Phase 6 deliverables. **Write report sections as soon as the evidence exists**, not only at the end.

**Minimum viable deliverable (if time or compute runs out):**
- Phase 2 reports complete for all three models (static, code-verified).
- At least T1–T3 run, **or** a clear, evidenced explanation of why not.
- Requirements table + runbooks (as far as they worked).
- Honest status of what was NOT done.

**Blocker playbook:**
| Blocker | Do |
|---|---|
| Old TF stack cannot build | Log the exact errors; switch to PyTorch port; record fidelity checks; this goes into the requirements report as a finding. |
| Custom CUDA op compile fails (PointNet++) | Try the port's pure-PyTorch ops; document speed penalty; compare with paper's FPS cost. |
| Toronto-3D download blocked | Ask human for manual download with exact instructions; meanwhile run smoke tests on any available public sample or synthetic data. |
| OOM | Reduce points/batch; use gradient accumulation if appropriate; record the limits (Probe E). |
| Results far below paper | Check label mapping, ignore-label handling, coordinate offsets, normalization, class weights, subsampling; run overfit test again; document. |
| A run is >2× over its time box | Stop it, save partial logs, label results "partial/early-stopped", continue. |
| Ambiguity about what the human wants | Write the question in `PROGRESS.md`, choose the most reasonable default, state it, proceed. |

---

## 12. PHASE 6: Deliverables (templates and acceptance)

### 12.1 Template for `reports/0X_model_*.md`
1. **Plain-English summary** (≤ 10 lines, no jargon; what problem it solves, the key trick, its main weakness).
2. **Position in the lineage** (what it fixes from the predecessor, what it leaves unsolved).
3. **Architecture** (diagram, layer table with shapes, parameter count; `[CODE]`/`[RUN]` verified).
4. **Input/output contract and preprocessing** (exact tensors; block/scene handling).
5. **Procedure** (setup → data prep → train → infer → evaluate; point to RUNBOOK; list defaults).
6. **Requirements** (software, hardware, licenses, setup difficulty 1–5 with justification).
7. **Segmentation capabilities** (matrix from §8.2-D).
8. **Geometric context** (mechanism table + Probe A results).
9. **Measured results** (table from `summary_all_runs.csv`: OA, mIoU, per-class IoU for Toronto-3D; feature-set comparisons; reproduction check vs paper; resources: peak memory, time/epoch, throughput, max N).
10. **Behavior on lab data** (compatibility, qualitative results, cost per km).
11. **Failure modes** (with figures).
12. **Strengths / weaknesses for *our* project** and **feasibility verdict** 🟢/🟡/🔴 with reasons.
13. **Reusable components** (ideas worth borrowing; cost to integrate).
14. **Open issues / not done.**
15. **Sources** (papers, repos with commit hashes).

### 12.2 `reports/04_comparison_matrix.md`
A single table, models as columns, with evidence labels. Rows at least: year/venue; core idea; local context mechanism; sampling; scene handling (block vs whole); input features supported; params; setup difficulty; measured mIoU (F0 and F3); **road marking / pole / fence / utility-line IoU**; peak GPU memory; training time; max N per pass; inference time per 1M points and per km; density robustness (Probe B); effective receptive field (Probe A); license; failure modes; feasibility verdict. End with **ranked recommendation of baseline** and the reasoning (weights stated explicitly, e.g., thin-class accuracy, scalability, setup cost, risk).

### 12.3 `reports/05_merge_hypotheses.md`
For each hypothesis (aim for 5–8, prioritized):
`ID | Weakness observed (with evidence pointer) | Proposed combination (component from model X + component from model Y / extra input / post-processing) | Why it should help | Evidence level (measured / literature / speculation) | Implementation cost (days) | Risk | Minimal test plan | Expected metric change`
Seed ideas to evaluate (do not assume they are good; test or reason from your measurements):
- Add **intensity** (normalized) as an input channel to RandLA-Net (cheap; test via T7).
- **PointNet++-style multi-scale/density-adaptive grouping** inside a **RandLA-Net-style efficient pipeline** to handle varying MLS density.
- **PointNet-style global context** vector concatenated into RandLA-Net's decoder to give whole-tile context.
- **Two-stage pipeline:** classical road-surface extraction (ground/road mask) → network focuses on road-plane features (markings) using intensity; non-road classes via the general model.
- **Class-balanced sampling/loss** targeted at thin classes.
- **Geometric post-processing** (e.g., clustering + shape fitting for poles; regularization for lane markings).
- **Self-supervised pretraining** on unlabeled lab data (note: pretraining backbones like Sonata are covered in another phase; here only list as direction).
- Use **trajectory/GNSS-derived road direction/heading** as extra context (state clearly if no evidence it helps: hypothesis only).

### 12.4 `reports/07_runbooks/RUNBOOK_*.md`
**Only commands that you actually ran successfully.** Each runbook: prerequisites, environment creation, data prep, training command with config, inference, evaluation, expected outputs and runtime, common errors and fixes (from the journal). A human must be able to follow it cold.

### 12.5 `reports/09_MCR_STUDENT_C_SUMMARY.md`
One page, plain language, structured for the student's progress report (sections: *Progress made*, *Difficulties encountered and plan*, *Tasks for next period*). Must reflect **only what was actually done** (list runs with IDs; explicitly list anything NOT done). Include IEEE-style in-text citations [n] and a numbered reference list (PointNet, PointNet++, RandLA-Net CVPR 2020 pp. 11108–11117, RandLA-Net TPAMI arXiv:2107.02389, Toronto-3D CVPRW 2020 pp. 202–203, plus any other papers you rely on, verifying author lists and venues from the sources).

### 12.6 `reports/HUMAN_STUDY_GUIDE.md`
The student must be able to **explain and defend** this work. Provide, per model, a 1-page study sheet: reading order (which paper sections matter), 10 self-test questions with short answers (e.g., "Why does max-pooling give permutation invariance?", "Why is FPS slow on a million points?", "What does the dilated residual block do and what is K²?", "Why do road markings fail with XYZ-only input?"), a 5-minute explanation script, and a glossary-linked list of terms (mIoU, IoU, OA, FPS, kNN, ball query, receptive field, T-Net, attention, class imbalance, etc.).

### 12.7 `reports/00_EXECUTIVE_SUMMARY.md`
≤ 2 pages: what was studied, what was run, **key findings** (numbers with labels), the recommended baseline, top merge hypotheses, risks, and what the team should do next. Also list **"what we still do not know."**

### 12.8 `reports/08_requirements_table.md`, `06_lab_data_compatibility.md`, `REFERENCES.md`, `GLOSSARY.md`
As specified above. `REFERENCES.md`: IEEE format; every reference must be verified from its source page; mark any unverified field `[verify]`.

---

## 13. Definition of done (checklist for `PROGRESS.md`)

- [ ] Phase 0: environment + raw data audits written, no guesses.
- [ ] Phase 1: repos cloned (hashes), environments working (forward pass).
- [ ] Phase 2: three model reports (sections 1–8 + 12–15), parameter counts verified, capabilities matrices, requirements table.
- [ ] Phase 3: Toronto-3D loaded + stats; per-model data prep; lab conversion; compatibility report.
- [ ] Phase 4: smoke + overfit tests pass for all models; T1–T3 complete; reproduction check; Probes A, B, E; lab inference/qualitative or cross-domain results.
- [ ] Phase 4 (P1): feature-set comparison (T4–T7), Probes C, D, F.
- [ ] Phase 6: comparison matrix, merge hypotheses, runbooks, executive summary, MCR summary, study guide, references, glossary.
- [ ] Final audit: every number in reports traced to `results/`; every claim labeled; list of NOT DONE items; `git log` clean; environment captured.
- [ ] End-of-work message to the human: what is done, what is not, what you need from them.

---

## 14. Appendix

### A. Common pitfalls to watch for
- **Float precision with UTM coordinates:** subtract offsets before casting to float32; verify subsampling and kNN behave.
- **Label 0 ("unclassified")** must be ignored in loss and metrics; verify the code does so consistently.
- **Class imbalance:** road markings, poles, fences are a tiny share of points; unweighted training may ignore them.
- **Data leakage:** don't select checkpoints or tune on the test tile; keep a held-out validation slice.
- **Block-based inference:** use overlap and voting; edge effects at block borders hurt thin classes.
- **Random sampling** (RandLA-Net) and FPS start point (PointNet++) cause run-to-run variation: use multiple seeds when budget allows.
- **Intensity** is not calibrated across range/angle on most MLS data; do not assume lab intensity matches Toronto-3D intensity (record distributions side by side).
- **TF1-era repos on modern GPUs** often cannot build; prefer ports but verify fidelity.
- **Metric details:** report per-class IoU, not only mIoU; thin classes are the point of this study.

### B. Metric definitions
- **IoU_c** = TP_c / (TP_c + FP_c + FN_c). **mIoU** = mean over valid classes (state whether "unclassified" is excluded; it should be). **OA** = correct points / all labeled points.

### C. Starting reference numbers (from literature; **re-verify at the source, then compare with your runs**)
| Model / config (Toronto-3D) | mIoU | Notes |
|---|---|---|
| PointNet++ | 56.55 | Toronto-3D paper baseline list |
| RandLA-Net, XYZ only | 77.71 (OA 92.95) | road marking 42.62; fence 37.12 |
| RandLA-Net, +RGB | 81.77 | road marking 64.21 |
| (PointNet on Toronto-3D) | not known to us | determine from your run; check whether published |

### D. Key sources to read in full (not only abstracts)
1. Qi et al., *PointNet*, CVPR 2017 (arXiv:1612.00593), §4 and supplementary.
2. Qi et al., *PointNet++*, NeurIPS 2017 (arXiv:1706.02413), §3 and supplementary B–C.
3. Hu et al., *RandLA-Net*, CVPR 2020 (arXiv:1911.11236), §3, §4.4, appendices B–D, G.
4. Hu et al., *Learning Semantic Segmentation of Large-Scale Point Clouds With Random Sampling* (arXiv:2107.02389) for Toronto-3D results.
5. Tan et al., *Toronto-3D*, CVPRW 2020 (arXiv:2003.08284), §3–5.
6. The GitHub READMEs and **open issues** of every repo you use.

### E. Questions you may need to ask the human (collect in `PROGRESS.md`)
- Where is the lab data? Is any of it labeled? Which classes?
- Is a GPU available (which one), and are there time limits (shared server, cluster, Colab)?
- Is internet access to GitHub/Google Drive/PyPI available from this machine?
- Preferred output language/format for the MCR summary and whether the supervisor requires specific metrics.
- Whether the lab sensor datasheet can be shared.

---
*End of plan. Start with Phase 0. When Phase 0 is done, summarize findings and any blocking questions to the human before proceeding.*
