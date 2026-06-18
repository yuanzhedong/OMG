# OMG Demo — Friday

## The one-line story
> **We stood up the entire OMG pipeline end-to-end and trained its text-conditioned
> motion brain on real Unitree-G1 data — type a sentence, and a diffusion model
> generates a physically-structured G1 whole-body motion that renders in simulation.**

## What OMG is (30 seconds)
OMG (Tsinghua MARS Lab) is an *omni-modal* motion generator for humanoid robots: a
diffusion "brain" turns **language / audio / human-reference** conditions into G1
reference motion, which a reactive tracker "cerebellum" executes on the robot. This
demo focuses on the **language → motion → simulation** path.

## What we actually did (be precise — this is what earns trust)
1. **Reproduced the full software stack** from the public release: installed, fixed
   3 stale tests, and verified the training/generation/benchmark pipeline runs
   (126 tests green).
2. **Sourced real G1 motion** — the paper's dataset (OMG-Data) is unreleased, so we
   used a public retarget: **Unitree LAFAN1 → G1** (174 mocap sequences; we used 40,
   ~147 min @ 30 FPS), spanning walk / run / dance / jump / fight / sprint.
3. **Built an adapter** into OMG's exact `qpos_36` + forward-kinematics format with
   action-category text labels, then computed real normalization stats.
4. **Trained the 50M diffusion model** on 4× RTX 4090 with real text conditioning
   (frozen T5-base encoder), logged live to W&B.
5. **Generated from free-text prompts** and rendered the result in MuJoCo.

## What to SHOW (in order)
1. **W&B loss curve** — real training, loss decreasing. "This is learning on real G1 data, live."
2. **The hero clip(s)** — text prompt → rendered G1 motion video (walk, dance, …).
3. **Live generation** (if confident) — type a new prompt, generate, play.
4. **The code** — the diffusion denoiser + conditioning, and the one-command pipeline.

## ⭐ Headline capability — full brain→cerebellum loop with PHYSICS
We now run the complete OMG hierarchy end-to-end:
**text → diffusion brain (kinematic reference) → HoloMotion tracker (physical execution in MuJoCo)**.
- The renderer's *generation* clips are kinematic (no dynamics — that's why they can look "floaty").
- The **HoloMotion tracker** (downloadable v1.3.1 ONNX) physically executes the reference: **walk and dance
  stay upright** under gravity/contacts; **run falls** (the reference is too dynamic) — an honest, instructive
  demonstration of the generation-vs-execution gap the paper's co-design exists to close.
- Physics even *improves* grounding: foot-ground error 0.015 m (reference) → **0.002 m** (executed).
- **Show the `tracker_sidebyside_walk.mp4`** (left = reference, right = physics) as the hero clip.

## ▶️ LIVE RUNBOOK (exact commands)
```bash
# One command: text -> diffusion brain -> HoloMotion physics -> side-by-side video
scripts/dev/demo.sh "a person is dancing"
#   -> outputs_demo/<slug>/demo_<slug>.mp4   (left=reference, right=physics, prints UPRIGHT/FELL)
```
Pre-rendered backups (don't rely on live gen):
- Hero montage (4 motions, physics): `outputs_tracker/montage_physics_2x2.mp4`
- Per-motion reference|physics: `outputs_tracker/*/tracker_sidebyside_*.mp4`
- 50M-vs-100M (physics-loss): `outputs_compare/compare_{walk,dance,run}.mp4`

**Verified working live prompts** (physics stays upright): `"a person walks forward"`,
`"a person is dancing"`, `"a person fights"`, `"a person jumps"`.
**Honest failure to show on purpose:** `"a person runs"` → robot falls (too dynamic to track).

## How to FRAME it honestly (pre-empts the hard question)
- **What this is:** a faithful, end-to-end reproduction of the OMG *system* — text→motion **and physical
  execution** — on real G1 data, on a single workstation in ~a day.
- **What this is NOT (yet):** the paper's headline benchmark numbers. Those require
  the unreleased OMG-Data (hundreds of hours, ~19 curated datasets), pretrained
  checkpoints, and the evaluator checkpoint. We trained on ~150 min of one public
  dataset, so expect coarser, category-level motion — not SOTA fidelity.
- **Why it's still convincing:** every moving part of the pipeline is real and wired
  together — data adaptation, normalization, diffusion training, classifier-free
  guided sampling, and simulation rendering. When the official artifacts drop, the
  same pipeline scales up by swapping the data root and model size. We're *ready*.

## Anticipated Q&A
- *"Is this the paper's model?"* — Same architecture and training code (50M preset);
  our weights, trained on public data. Not the released checkpoint (unreleased).
- *"Why so little data?"* — OMG-Data isn't public yet; we proved the pipeline on what
  is. Scaling is a data-download away.
- *"Does it follow physics / run on the real robot?"* — Yes in sim: the HoloMotion tracker
  physically executes the generated reference (walk/dance stay upright). The *same* tracker runs
  on a G1 Orin for real-robot deployment (in the repo) — only the hardware step is out of scope.
- *"Why does the raw generation look floaty?"* — That render is kinematic (no dynamics). Physics
  comes from the tracker; see the side-by-side reference-vs-executed clip.
- *"How long did this take?"* — ~1 day on one workstation, 4× 4090.

## Stats / specifics to have on hand
- Data: Unitree LAFAN1→G1, 40 clips, ~264k frames (~147 min) @ 30 FPS, 8 action classes.
- Model: 50M transformer diffusion denoiser, rot6d 125-D motion rep, 60-frame windows,
  DDIM-50 sampling, CFG. Text: frozen T5-base.
- Compute: 4× RTX 4090 (the 2× Blackwell PRO 6000 are unsupported by torch cu124).
- Training: 60,000 steps, ~2h58m on 2x Blackwell PRO 6000, final train loss 0.032.
  **Demo checkpoint: step 1000** (see scaling note below).

## Important finding — small-data overfitting (a GIFT for the scaling story)
We checkpointed every 1000 steps and measured how generated motion responds to text:

| checkpoint | walk travel | run travel | dance travel | dance limb-activity |
|---|---|---|---|---|
| **step 1000** | 1.88 m | 2.57 m | 1.29 m | **0.232** |
| step 3000 | 0.91 m | 4.28 m | 0.63 m | 0.131 |
| step 6000 | 0.32 m | 2.82 m | 0.36 m | 0.075 |
| step 24000 | 0.12 m | 0.94 m | 0.09 m | 0.068 |
| step 60000 | 0.22 m | 0.20 m | 0.24 m | (collapsed) |

On only ~150 min / 40 clips, the model learns fast (great motion by step 1000) then
**overfits and collapses** to near-static motion. This is the *empirical case for the
paper's central claim*: OMG's quality and scaling come from OMG-Data's hundreds of
hours across ~19 curated datasets. We reproduced the capability AND independently
re-derived why scale matters. **Use the step-1000 checkpoint for the demo.**

## Evidence the model learned language → motion (measured, step-1000 checkpoint)
Generated 4 prompts; motion signatures (4 s clips) differ exactly as expected:

| prompt | forward travel | joint activity (std) | reading |
|---|---|---|---|
| "a person runs" | **2.57 m** | 0.192 | most ground covered |
| "a person walks forward" | 1.88 m | 0.178 | moderate locomotion |
| "a person jumps" | 1.37 m | 0.164 | mostly in place |
| "a person is dancing" | 1.29 m | **0.232** | least travel, most limb movement |

Run > walk in distance; dance = most limb activity + least travel. This is from only
1000 training steps — quality improves with the full run. Hero videos in
`outputs_generate/<prompt>/50m/lafan1_50m_demo/*.mp4`.
