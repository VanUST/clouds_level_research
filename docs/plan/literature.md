# CBH by passive fisheye stereo — literature & open-source reconnaissance

Scope note: URLs only where a claim is made. "Implementation" = a public, usable repo/toolbox or an explicit statement that code exists. PDFs could generally not be fetched by the research tool (only HTML/XML/ar5iv), so for several papers the numbers below come from their abstract/HTML full text or from another paper's citable summary of them.

---

## 1. Stereo cloud base height from ground-based cameras

**Beekmans, Schneider, Läbe, Lennefer, Stachniss, Simmer (2016), "Cloud photogrammetry with dense stereo for fisheye cameras", ACP 16, 14231–14248.**
https://acp.copernicus.org/articles/16/14231/2016/ — full text at https://acp.copernicus.org/articles/16/14231/2016/acp-16-14231-2016.xml
- Closest analogue to this project: two hemispheric fisheye imagers, 300 m baseline, epipolar rectification designed for fisheye so that an off-the-shelf dense (semi-global) matcher can be used, dense 3-D cloud point cloud over ~10×10 km within ~5 km range. Exterior orientation from **sensed stars** (Stellarium/NAOJ catalogue), exterior positions from GPS (2–3 m, cm with D-GPS), interior from a printed half-cube test field bundle adjustment. Validation case: stratocumulus compared against **cloud radar reflectivity + lidar-ceilometer cloud base**; second case an altocumulus stratiformis at ~3000 m. Crispel & Roberts summarise the result as CBH relative errors **< 5 %**. Explicitly notes that dense stereo recovers low-contrast regions better than sparse feature methods — directly relevant to the current LoFTR/HDBSCAN design.
- Implementation: none published. Method (fisheye rectification) is fully described and reproducible.

**Crispel & Roberts (2018), "All-sky photogrammetry techniques to georeference a cloud field", AMT 11, 593–609.**
https://amt.copernicus.org/articles/11/593/2018/ (preprint: https://amt.copernicus.org/preprints/amt-2017-203/)
- 150 m baseline, VIVOTEK 180° fisheye, 2944×2944. Intrinsics via **OcamCalib/Scaramuzza** model. Relative orientation **without a target**: GPS + bubble level + vertical sights, then SIFT matches → essential matrix (several subsets tested, coherence check R≃Id, T≃(b,0,0)), then rectification and **OpenCV dense block matching** for height, cross-correlation block tracking for velocity. Two validation cases against a **Vaisala CL31 ceilometer**: cumulus at 1500 m a.g.l., and two layers at 1000 m + 2300 m. Height uncertainty estimated by comparison with the ceilometer; horizontal/velocity uncertainty propagated theoretically from height and orientation uncertainty.
- Directly relevant caveats: block matching "generates noise/speckles in the weakly textured image part"; and the pinhole disparity relation "becomes less relevant for larger incident angles for which larger horizontal errors occur".
- Implementation: none published; relies on OpenCV + OcamCalib.

**Öktem, Prabhat, Lee, Thomas, Zuidema, Romps (2014), "Stereophotogrammetry of Oceanic Clouds", J. Atmos. Oceanic Technol. 31(7).**
https://journals.ametsoc.org/view/journals/atot/31/7/jtech-d-13-00224_1.xml (DOI 10.1175/JTECH-D-13-00224.1; the AMS host returned HTTP 403 twice — see "could not find"; author page: https://davidromps.com/papers/pubs-2011-velocity.html)
- Two cameras ~873 m apart oriented toward the horizon; orientation from the **sun position and horizon line**. Agreement with lidar: errors mostly **< 2 % for shallow clouds**, up to **8 % for high cirrocumulus** (summarised in Crispel & Roberts, above). Also compares camera-derived cloud motion of individual layers with **radiosonde wind** — the earliest found explicit wind/velocity cross-check of a stereo-camera retrieval.
- Implementation: none published.

**Seiz et al. (2006/2007)** — two whole-sky imagers, 5 km baseline; standard stereo matching largely failed at that separation; **> 50 % of estimated CBH within 5 % of a lidar-ceilometer**. Cited and characterised in Beekmans et al. 2016; source: https://misr.jpl.nasa.gov/media/documents/seiz2006b.pdf

**Blum, Nouri, Wilbert, Schmidt, Lünsdorf, Stührenberg, Heinemann, Kazantzidis, Pitz-Paal (2021), "Cloud height measurement by a network of all-sky imagers", AMT 14, 5199–5224.**
https://amt.copernicus.org/articles/14/5199/2021/ (XML: https://amt.copernicus.org/articles/14/5199/2021/amt-14-5199-2021.xml)
- Seven ASIs → 42 ordered camera pairs, baselines **0.8–5.7 km**, three months, validated against a reference ceilometer. Per-pair deviations are compiled into **empirical conditional probabilities P(CBH_pair | h_true)** and fused by a modified maximum-likelihood / cumulative-likelihood intersection to give the *likeliest* CBH. Findings: pairs with baseline < **1.2 km** are accurate for **CBH < 4 km**; baselines > **3 km** are better for high cloud; **no single pair is best at all heights**, and the network beats any single pair. Also documents that "there is no generally accepted, quantifiable definition" of ceilometer accuracy and that instrument types can disagree by kilometres in heavy rain.
- Implementation: none published; the conditional-probability fusion is simple enough to reimplement.

**Herrero del Barrio et al. (2026), "Cloud height mapping using multi simultaneous sky images from an all-sky camera network", AMT 19, 5785–5814.**
https://amt.copernicus.org/articles/19/5785/2026/ (preprint: https://egusphere.copernicus.org/preprints/2026/egusphere-2026-2694/)
- 20 all-sky cameras (GOA-SCAN, Valladolid), baselines >50 km down to <200 m; per-pixel RGB block correlation along epipolar rows; up to four camera pairs combined by **per-pixel median**; 3×3 median-filter outlier rejection; layer selection from a 1 km-bin height histogram. Heights < ground or >15 km rejected, and a distance-dependent admissibility band H_min=d/tan 80°, H_max=d/tan 10°.
- **Per-pixel uncertainty is obtained by shifting the correlation peak ±1 px and taking the max height change**; a ±1 px shift after downsampling corresponds to ±10 px in the original image; pixels whose uncertainty exceeds 1 km are discarded. (This is exactly the kind of cheap per-estimate uncertainty the current pipeline lacks.)
- **Validation: 2.5 years against a collocated ceilometer — determination coefficient r² = 0.93 for median CBH within 150 m**, r² = 0.72 for cloud-top (which overestimates). Sentinel-2 scene classification used as an independent map check; discrepancies at cloud edges and thin cloud.
- Implementation: none published.

**Andreev/Chulichkov and Nikitin/Chulichkov (the project's cited refs), extended context:**
- Chulichkov et al., "On cloud bottom boundary determination by digital stereo photography from the Earth's surface" — https://www.semanticscholar.org/paper/fad113d2ecc86690d88b6e9e0388a19e029f1081 and https://bakhtiniada.ru/1024-8560/article/view/188035
- Related AMT discussion that explicitly attributes a **systematic offset to unmodelled 3-D cloud-side geometry** (relevant to the geometric-vs-optical offset this project expects): https://amt.copernicus.org/preprints/amt-2018-220/amt-2018-220.pdf ("We consider 3D cloud geometry itself to be the most likely reason for the offset… cloud side geometry is not considered complete").
- Andreev & Chulichkov SPIE 2015 abstract: https://ui.adsabs.harvard.edu/abs/2015SPIE.9680E..4NC/abstract

**Other measured-agreement work:**
- "Developments of low-cost procedure to estimate cloud base height based on a digital camera", Measurement (2010): https://www.sciencedirect.com/science/article/abs/pii/S0263224110000138
- "Cloud height and tracking accuracy of three all sky imager systems for individual clouds", Solar Energy (2019): https://www.sciencedirect.com/science/article/abs/pii/S0038092X18310570
- "Cloud base height estimates from sky imagery and a network of pyranometers": https://www.sciencedirect.com/science/article/abs/pii/S0038092X19303305
- "Cloud base height estimates by stereoscopy" background (DOAJ record): https://doaj.org/article/dc552d6fec53445b81ff09092353ccce

**Accuracy/range summary that emerges:** reported relative CBH errors cluster at **2 % (low cloud) to 8 % (high/thin cloud)**, with 5 % a common headline number; absolute scatter of order **0.6–1.4 km RMSD** depending on height and baseline; usable range is baseline-limited (150 m→~1.5–2.5 km; 300 m→~5 km; ~900 m–1.2 km→higher clouds). Ceilometer comparison is the standard validation and **all authors accept the geometric-vs-optical offset implicitly**; only the amt-2018-220 discussion names 3-D cloud geometry as its cause.

---

## 2. Alternative feature matchers for low-texture, semi-transparent targets

**EfficientLoFTR (Wang, He, Peng, Tan, Zhou), CVPR 2024.**
Paper: https://ar5iv.labs.arxiv.org/html/2403.04765v1 · Code: https://github.com/zju3dv/EfficientLoFTR
- Explicitly identifies the defect of LoFTR relevant here: *"there can be spatial variance in the matching refinement phase of LoFTR, which is caused by the expectation over the entire correlation patch when noisy feature correlation exists."* Its fix — **two-stage correlation**: stage 1 does mutual-nearest-neighbour pixel-level matching on fine patches; stage 2 correlates a tiny 3×3 patch and takes the softmax expectation for sub-pixel. Reported to give "accurate subpixel correspondences", ~2.5× faster than LoFTR, and better relative-pose AUC than LoFTR/QuadTree/MatchFormer/TopicFM/AspanFormer on MegaDepth (56.4 vs 52.8 AUC@5°) while still below the dense matchers (DKM 60.4, RoMa 62.6). This is the single most directly citable statement about **why LoFTR-style refinement is not reliably sub-pixel** and how to fix it.

**LoFTR (Sun et al.), CVPR 2021.**
https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.pdf · Code: https://github.com/zju3dv/LoFTR
- Original coarse-to-fine design: coarse matches at 1/8 resolution, then the location in one image is fixed and the sub-pixel correspondence in the other is found by correlation + **soft-argmax expectation over the whole fine patch**. That asymmetry (anchor fixed on image A, expectation on image B) is the source of the "location variance" EfficientLoFTR quantifies. The paper reports strong performance on texture-poor indoor pairs (ScanNet) but publishes **only pose/homography metrics, no per-match pixel error and no uncertainty**.

**RoMa (Edstedt et al.), CVPR 2024.**
https://arxiv.org/abs/2305.15404 · Code: https://github.com/Parskatt/RoMa · RoMa v2: https://ar5iv.labs.arxiv.org/html/2511.15706
- Dense matcher built on DINOv2 features + a learned regression head; best accuracy in the EfficientLoFTR comparison table and strong zero-shot robustness, but ~7.5× slower than EfficientLoFTR (302 ms/pair at 640×480). No uncertainty output.

**DKM (Edstedt et al.), CVPR 2023.**
Paper/README: https://github.com/Parskatt/DKM
- Dense kernelized matching with Gaussian-process-like iterative refinement; best-in-class geometry accuracy at the time, slower; **no per-match uncertainty published**.

**OmniGlue (Jiang et al.), CVPR 2024.**
https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_OmniGlue_Generalizable_Feature_Matching_with_Foundation_Model_Guidance_CVPR_2024_paper.html · Code: https://github.com/google-research/omniglue
- Detector-based but foundation-model-guided; improves generalization across domains; evaluated on sparse keypoints, not on semi-transparent targets. No uncertainty.

**MatchAnything (Zhu et al., 2025) / ELoFTR weights.**
https://arxiv.org/abs/2501.07556 · Weights: https://huggingface.co/zju-community/matchanything_eloftr
- Cross-modality (thermal/optical, night) pre-training; built on ELoFTR, so it inherits the two-stage sub-pixel refinement. No uncertainty; no cloud-specific evaluation found.

**Sub-pixel localisation of learned features — "Learning to Make Keypoints Sub-Pixel Accurate" (Kim, Pollefeys, Barath), ECCV 2024.**
https://ar5iv.labs.arxiv.org/html/2407.11668 · Code: https://github.com/KimSinjeong/keypt2subpx
- States plainly that learned detectors (SuperPoint, ALIKED) **lack sub-pixel precision and lag classical SIFT**, and provides a detector-agnostic learned offset module (+~7 ms) that minimises epipolar error. Measured gains are pose AUC, not pixel RMSE — see "could not find".

**ConDL (Kwiatkowski, Matern, Hellwich), 2024.**
https://ar5iv.labs.arxiv.org/html/2408.02766
- One of the few works that **measures match error in pixels**: inliers counted at reprojection thresholds 0.1 px, 1 px, 10 px on synthetic SIDAR pairs. Qualitative result directly relevant to cloud matching: **LoFTR produces few matches but almost all are inliers (high precision, low recall), whereas dense contrastive matching produces many inliers plus many false positives**. No sub-pixel-bias analysis and no cloud imagery.

**HSROSS benchmark (high-resolution optical satellite, challenging scenarios).**
https://ieeexplore.ieee.org/document/11184119
- Independent head-to-head: detector-free dense methods dominate; LoFTR ~0.99 average, GIM-DKM ~0.99. Useful as evidence that detector-free matchers beat sparse ones in hard/low-texture regimes, but the domain is satellite, not sky.

**Verdict for topic 2:** all five families are usable drop-ins, all are texture-poor-capable, **none except SURE (topic 3) exposes per-match uncertainty**; LoFTR's fine stage is *nominally* sub-pixel (soft-argmax) but demonstrably location-biased, and EfficientLoFTR is the only public matcher that documents and fixes this. Published pixel-level localisation error for LoFTR-style matching on cloud-like input does not exist (see final section).

---

## 3. Uncertainty quantification for stereo / dense matching

**SURE: Semi-dense Uncertainty-REfined Feature Matching (Li, Gu, Zhang, Guo, Jiang, Cheng), ICRA 2026.**
https://arxiv.org/abs/2603.04869 · Code (promised): https://github.com/LSC-ALAN/SURE
- The most on-point result found: a semi-dense matcher that **jointly predicts correspondences and per-match confidence by modelling both aleatoric and epistemic uncertainty**, via an evidential head for coordinate regression plus a light spatial fusion. Motivation is literally "incorrect correspondences may still receive high similarity scores … overconfident errors" in **textureless regions**. Measured on standard matching benchmarks (accuracy/efficiency), not metrology. Code is announced but, as of the arXiv v1, not yet public.

**ELFNet: Evidential Local-global Fusion for Stereo Matching (Lou et al.), ICCV 2023.**
https://arxiv.org/abs/2308.00728 · code: https://github.com/jimmy19991222/ELFNet
- Evidential deep learning gives a per-pixel disparity plus an uncertainty/evidence map; fuses local and global cost volumes. Reusable recipe for turning a stereo head into a probabilistic one.

**"Uncertainty estimation in transformer-based stereo matching", ICCV 2023 (A*STAR).**
https://oar.a-star.edu.sg/storage/8/8dxwnp0g72/iccv-2023-depth-1.pdf
- Applies/benchmarks uncertainty estimators on transformer stereo (probabilistic + ensembling); directly transferable to any attention-based matcher backbone.

**"Joint estimation of depth and its uncertainty from stereo images using Bayesian deep learning", ISPRS Annals V-2-2022.**
https://isprs-annals.copernicus.org/articles/V-2-2022/69/2022/
- Predicts a per-pixel depth mean and variance (aleatoric + epistemic via dropout), validated against propagation; a clean template for a probabilistic disparity head.

**VMI-Stereo: Variational Bayesian Inference Based Multivariate Information Fusion Stereo Matching.**
https://digital-library.theiet.org/doi/pdf/10.1049/ipr2.70386
- Variational Bayesian stereo with uncertainty from multi-source fusion; recent, implementation not checked.

**"Not All Pixels Are Equal: Confidence-Guided Attention for Feature Matching" (2025).**
https://browse-export.arxiv.org/pdf/2505.02161
- Uses a confidence map to reweight matching attention and analyses sensitivity to imperfect confidence maps — useful if confidence is used to gate clusters rather than as a variance.

**Ensembling / MC-dropout practical comparisons** (bachelor thesis, RUG 2025, *Not Just Seeing, Understanding What's Uncertain*): https://fse.studenttheses.ub.rug.nl/36742/1/bAI2025vanLoonSA.pdf — compares ensembles, MC-dropout and Flipout for stereo depth; useful for expected achievable calibration, though it is a thesis not a peer-reviewed paper. Foundational, still-citable: deep ensembles (Lakshminarayanan et al. 2017, https://arxiv.org/abs/1612.01474) and MC-dropout (Gal & Ghahramani 2016, https://arxiv.org/abs/1506.02142).

**Classical probabilistic matching / robust fitting (reusable):**
- Asymmetrical Gauss mixture model point matching (CVPR 2014): https://www.openaccess.thecvf.com/content_cvpr_2014/papers/Tao_Asymmetrical_Gauss_Mixture_2014_CVPR_paper.pdf
- Bayesian feature matching (Mohr et al., INRIA): https://inria.hal.science/file/index/docid/548350/filename/mohr97bayesian-1.pdf
- MAGSAC++ (marginalising sample consensus, infers the inlier noise scale σ instead of a fixed threshold; in OpenCV and a standalone repo): https://github.com/danini/magsac — directly usable to replace HDBSCAN/ fixed inlier thresholds with a noise-scale estimate.

---

## 4. Multi-camera calibration without a target, and calibration drift

**SuMo — "Geometric Calibration of All-Sky Cameras Using Sun and Moon Positions" (Blum et al., DLR), EMS 2025 / Solar Energy 2025.**
https://elib.dlr.de/217695/ · journal version: https://www.sciencedirect.com/science/article/pii/S0038092X25002397
- Python tool that calibrates intrinsics **and** external orientation from ordinary day/night images of Sun and Moon, no on-site handwork, and — explicitly — **can continuously monitor an ASI's geometric calibration**. Validated on five camera types at three sites: a single month of images gives **RMSE ≤ 0.14°**; across all cameras 0.14°–0.38°; degraded only to 0.32–0.38° under high turbidity / heavy cloud. Cross-validated against the star-based ORION method. Published as an **open-source package together with >2 years of ASI images and irradiance data**. This is the strongest single hit for target-free calibration *and* drift monitoring of exactly this camera class.

**ORION — "ORION software tool for the geometrical calibration of all-sky cameras" (Calle et al.), PLOS ONE 2022.**
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265959 · app: https://goa.uva.es/orion-app/
- Open-source star-based geometric calibration (azimuth/zenith + intrinsics) for all-sky cameras; the reference method SuMo cross-checks against. Star-based exterior orientation is also what Beekmans et al. use, so this is the citable, implemented version of the "infinity reference" idea.

**Sun-based calibration (non-ASI, classic):** Lalonde et al., "Camera calibration and light source orientation from solar shadows", IJCV 2010 — https://vision.gel.ulaval.ca/~jflalonde/assets/pubs/papers/lalonde_ijcv_10.pdf. Shows sun/sky radiometry can recover both intrinsics and light-source direction; complements the "Sun at infinity" constraint.

**Horizon / landmark references:** Öktem et al. (2014) use sun + horizon; Crispel & Roberts (2018) use GPS+level+sights then **SIFT→essential matrix auto-calibration**; Beekmans et al. (2016) use stars. Together these are the principal target-free infinity/scene references in this exact literature.

**SLAM-based extrinsic calibration of a multi-camera rig (Carrera et al., ICRA 2011).**
https://ieeexplore.ieee.org/document/5980294
- Calibrates rig extrinsics from natural visual features while mapping — a target-free self-calibration route that also directly measures reprojection residuals (see topic 7).

**Kalibr.**
https://github.com/ethz-asl/kalibr
- The de-facto multi-camera / camera-IMU calibration toolbox; supports pinhole-equidistant ("fisheye") and omni models, gives covariances, and is the standard way to produce the calibration covariance needed in topic 4's propagation question. (Target-based, but the covariance machinery is what matters.)

**Calibration drift for fixed outdoor rigs (thermal, wind, long-term stability):**
- Detchev & Habib, FIG 2017, "Geometrical calibration and online stability" — long-term stability of a camera system over repeated calibration sessions: http://eib.fig.net/resources/proceedings/fig_proceedings/fig2017/papers/ts05f/TS05F_detchev_habib_et_al_8565.pdf
- "Methodology for Automatically Detecting Pan-Tilt-Zoom CCTV Camera Drift in Advanced Traffic Management System Networks", MDPI 2024: https://www.mdpi.com/2673-7590/4/4/62
- "An online thermal drift calibration method for vision-based displacement measurement", Optics and Laser Technology 2026: https://www.sciencedirect.com/science/article/abs/pii/S014381662600117X
- "Characterization and modeling of image drift in vision-based structural monitoring through sensitivity analysis and experiments", Measurement 2026: https://www.sciencedirect.com/science/article/abs/pii/S0263224126028587
- These establish that fixed-camera extrinsic/intrinsic drift is real and monitorable, but none is a cloud rig — see "could not find".

**Uncertainty propagation into a triangulated range:**
- "Propagation of uncertainty through stereo triangulation" (IEEE 2010): https://ieeexplore.ieee.org/abstract/document/5488057 — closed-form/analytic propagation of pixel and calibration covariance into range uncertainty; the canonical citation for this question.
- "Factorial-Design-Based Uncertainty Analysis for Stereo-Vision Photogrammetry" (IEEE TIM 2023): https://ieeexplore.ieee.org/document/10301504 — decomposes how much of the depth variance comes from calibration vs matching vs geometry; exactly the sensitivity analysis needed to decide whether to spend effort on calibration drift or on matching.
- "Uncertainty evaluation of camera model parameters" (IEEE 2011): https://ieeexplore.ieee.org/document/5944307/references
- "Covariance estimation for minimal geometry solvers via scaled unscented transformation" (CVIU 2014): https://www.sciencedirect.com/science/article/abs/pii/S1077314214001817 — covariance of essential-matrix/pose solutions, i.e. how calibration/orientation uncertainty enters.
- NASA EOL "Cal and Alignment" (fisheye calibration + alignment error budget): https://ntrs.nasa.gov/api/citations/20250000681/downloads/EOL_Cal_and_Alignment_AAS_GNC_Paper_011725.pdf

---

## 5. Fisheye / omnidirectional stereo geometry

**Kannala & Brandt (2006), "A generic camera model and calibration method for conventional, wide-angle, and fish-eye lenses", IEEE TPAMI 28(8).**
https://pubmed.ncbi.nlm.nih.gov/16886867/
- The equidistant polynomial model r(θ)=k1θ+k2θ³+k3θ⁵+k4θ⁷ that OpenCV's `fisheye` module and Kalibr's "pinhole + equidistant" both implement. This is the correct baseline model for a 180° lens; "equisolid" r=2f·sin(θ/2) and "stereographic" r=2f·tan(θ/2) are the competing closed forms, and the choice matters most at the periphery.

**OpenCV fisheye model (Kannala–Brandt).**
https://docs.opencv.org/4.10.0/db/d58/group__calib3d__fisheye.html
- Implemented `calibrate`, `undistort`, `stereoCalibrate`, `stereoRectify`, `initUndistortRectifyMap` for fisheye. Usable, but note it rectifies to a pinhole image (so the periphery is stretched/undersampled and the epipolar geometry becomes non-uniform in resolution).

**Scaramuzza OcamCalib (unified/spherical single-viewpoint model).**
https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab
- The model used by Crispel & Roberts (2018) for their 180° fisheye. Works from a flat chessboard; well suited to peripheral accuracy.

**Kalibr.** https://github.com/ethz-asl/kalibr — pinhole-equidistant and (via extension) omni models; gives camera-rig extrinsics and covariances.

**Basalt / Basalt-headers — double-sphere camera model.** https://github.com/flexivrobotics/basalt · https://vladyslavusenko.gitlab.io/basalt-headers/index.html
- The "double sphere" model, a compact alternative to Kannala–Brandt that is closed-form invertible; useful if the current warp is affine/approximate.

**COLMAP fisheye camera model.** https://colmap.github.io/cameras.html — `OPENCV_FISHEYE`, `RADIAL_FISHEYE`, `FOV`; the reference SfM implementation if bundle-adjusting the rig against the sky is ever wanted.

**Eising (2022), "Direct Triangulation with Spherical Projection for Omnidirectional Cameras".**
https://ar5iv.labs.arxiv.org/html/2206.03928
- **Closed-form, non-iterative triangulation directly on the unit sphere**: map each pixel to a unit ray, then (a) spherical midpoint, (b) spherical linear triangulation, (c) optimal epipolar-plane optimisation (roots of a quadratic), numerically equivalent to the iterative Förstner–Wrobel spherical triangulation. This is the correct replacement for the paraxial D = ξ₀B/(π|d|) formula: it uses the true ray directions, so it stays valid at large zenith angles and gives a reprojection residual on the sphere as a by-product (topic 7). Applies equally to narrow-FOV cameras. Implementation: method is self-contained; no separate repo identified (see final section).

**Beekmans et al. (2016) fisheye epipolar rectification + angle-based triangulation.**
https://acp.copernicus.org/articles/16/14231/2016/acp-16-14231-2016.xml
- Gives the explicit rectified coordinates u_V = c·ψ + u₀,V, v_V = c·β + v₀,V (angles scaled by the camera constant) and the exact triangulation relation s·sin(ψ_R + π/2) = b·sin(γ), with γ = ψ_L − ψ_R, P = s·x″_L. This is the rigorous angle-domain alternative to the disparity formula for a fisheye pair.

**Crispel & Roberts (2018)** explicitly warn that the pinhole disparity/height relation "becomes less relevant for larger incident angles for which larger horizontal errors occur", and that their disparity bounds ("related to height detection bounds") also degrade with incidence angle. https://amt.copernicus.org/articles/11/593/2018/

**Kang et al., CVPR 2022, "Uniform Subdivision of Omnidirectional Camera Space for Efficient Spherical Stereo".**
https://openaccess.thecvf.com/content/CVPR2022/papers/Kang_Uniform_Subdivision_of_Omnidirectional_Camera_Space_for_Efficient_Spherical_Stereo_CVPR_2022_paper.pdf
- Shows how to discretise the sphere uniformly for stereo, and why equirectangular/planar sampling is non-uniform — relevant to disparity-space clustering in HDBSCAN.

**Practical takeaway:** the projection model should be Kannala–Brandt (equidistant) or Scaramuzza, not equisolid, unless the lens is explicitly equisolid; triangulation should be done on unit-sphere rays (Eising/Beekmans) rather than with the paraxial disparity formula; and the residual of the two-ray intersection is a free, physically meaningful validation signal.

---

## 6. Temporal / multi-layer cloud tracking and wind advection

**Blum et al. (2021), AMT 14, 5199.** https://amt.copernicus.org/articles/14/5199/2021/
- Statistical (conditional-probability, modified-MLE) fusion of many camera pairs — an alternative to treating layers as targets. Baseline-dependent accuracy means camera-pair selection carries information about height, which is a usable prior for temporal association.

**Herrero del Barrio et al. (2026), AMT 19, 5785.** https://amt.copernicus.org/articles/19/5785/2026/
- Explicit temporal/spatial filtering chain: 3×3 median filter with a 1 km deviation gate, then a **1 km-bin height histogram selecting up to four layers** each holding ≥2 % of valid pixels, pixels >0.7 km from a layer centre dropped. A concrete, citable multi-layer selection scheme.

**Ceilometer / lidar layer detection (reusable implementations):**
- ARM `ACT` Python package, wavelet covariance transform PBL/layer detection: https://arm-doe.github.io/ACT/API/generated/act.retrievals.calculate_wavelet_pbl.html · repo: https://github.com/ARM-Development/ACT
- WCT applied to range-corrected backscatter (ACP 2019): https://acp.copernicus.org/preprints/acp-2019-565/acp-2019-565-manuscript-version6.pdf
- "A Threshold-Based Method for Cloud Base Height Detection using Ceilometers" (EGUsphere 2026): https://egusphere.copernicus.org/preprints/2026/egusphere-2026-3317/
- CBASE — cloud-base height + uncertainty from CALIOP, with an explicit uncertainty treatment: https://essd.copernicus.org/preprints/essd-2018-43/essd-2018-43.pdf
- PBL-height retrieval from ceilometer with uncertainty attribution: https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/178514/3/acp-2016-962.pdf

**Wind advection and instrument lag:**
- "Comparison of Cloud Motion Vector Profiles Derived from Ground-Based Hemispheric Stereocameras to Wind-Lidar Observations": https://beta.schweizerbart.de/content/papers/download/99766 — directly measures camera-derived cloud motion against a wind lidar, i.e. the observable that decouples the two instruments spatially.
- "Optimizing cloud motion estimation on the edge with phase correlation and optical flow", AMT 2022: https://amt.copernicus.org/preprints/amt-2022-159/ (also ARM: https://www.arm.gov/publications/other/P013077.pdf) — phase-correlation + optical flow for cloud velocity, and it explicitly matches cloud-motion vectors to winds at the appropriate altitude.
- Crispel & Roberts (2018) compute a cloud-base velocity field by block cross-correlation between successive images and convert to velocity using the height map (their Eq. 13/14).
- Öktem et al. (2014) compare layer motion to radiosonde winds.
- General MHT/IMM references (not cloud-specific): MH-IMM filter, http://fusion.isif.org/proceedings/fusion01CD/fusion/searchengine/pdf/TuB21.pdf ; multi-target lidar tracking framework, https://radar.inria.fr/rapportsactivite/intranet/PDF-2013/resultats-tri-.pdf

**On the specific "co-located but separated instruments, advection lag" question:** no paper was found that models the wind-advection time lag between a pair of separated ground cameras and a zenith ceilometer and corrects the CBH comparison for it. The closest tools are the camera-vs-wind-lidar comparisons above and the general cross-correlation-with-lag approach; see "could not find".

---

## 7. Falsifiable validation without ground truth

**Leclerc & Luong (2000), "Measuring the Self-Consistency of Stereo Algorithms", ECCV.** https://rd.springer.com/chapter/10.1007/3-540-45054-8_19
- Formalises self-consistency: if the same estimator is applied to a perturbed/related image pair (or to sub-regions) the results must agree; it provides a measurable, reference-free score. The canonical citation for "self-consistency validation of stereo depth".

**Stereo confidence / left-right consistency (standard, reusable):**
- Stereo confidence measures on KITTI, "Analysis of KITTI Data for Stereo Analysis with Stereo Confidence Measures": https://rd.springer.com/chapter/10.1007/978-3-642-33868-7_16
- Middlebury stereo evaluation (Scharstein & Szeliski), the reference for LR-consistency and confidence definitions: https://vision.middlebury.edu/stereo/
- Reprojection error from `stereoCalibrate()` as a rig-consistency metric: https://link-hkg.springer.com/content/pdf/10.1007/s11042-025-20846-7.pdf
- "Characterizing the Performance of Multiple-Image Point-Correspondence Algorithms Using Self-Consistency": https://dl.acm.org/doi/abs/10.5555/646271.685633

**Imagery-only checks already in the CBH literature:**
- Herrero del Barrio et al. (2026) use: physically-plausible height band (0–15 km), distance-dependent angular admissibility band, ±1 px shift → height-uncertainty gate, correlation-coefficient threshold, cloud-mask restriction, 3×3 median smoothness, and layer-histogram consistency; plus an independent Sentinel-2 scene-classification map check. https://amt.copernicus.org/articles/19/5785/2026/
- Crispel & Roberts (2018) reject essential-matrix solutions inconsistent with the known rig geometry (R≃Id, T≃(b,0,0)) — a strong physical prior used as a falsification test. https://amt.copernicus.org/articles/11/593/2018/
- Beekmans et al. (2016) use the star-derived orientation as an absolute reference and the lidar/radar as comparison; the star/sun/moon positions are themselves an imagery-only geometric check (SuMo/ORION).

**Geometry-residual checks that are free with unit-sphere triangulation:**
- Eising (2022) spherical triangulation yields the residual of the two-ray intersection directly; a large residual at a matched cloud pixel flags a bad match or a wrong layer hypothesis. https://ar5iv.labs.arxiv.org/html/2206.03928
- MAGSAC++ gives a data-driven inlier noise scale σ for the epipolar constraint, a quantitative self-consistency statistic. https://github.com/danini/magsac

**Cross-region / sub-region agreement:** the median-across-pairs fusion in Blum 2021 and Herrero del Barrio 2026 is effectively a multi-region agreement test; Blum's conditional probabilities turn pair disagreement into an actual likelihood over height.

---

## Highest-value candidates to reuse

1. **SuMo (DLR) — sun/moon auto-calibration with continuous drift monitoring** (https://elib.dlr.de/217695/ ; journal version https://www.sciencedirect.com/science/article/pii/S0038092X25002397). Directly replaces the hand-rolled "sun at infinity" check with a published, open-source, sub-0.4° tool that also *monitors* the fixed rig over time; the 2-year open dataset makes it verifiable. Highest benefit because the entire pipeline's absolute geometry rests on this calibration.
2. **EfficientLoFTR + two-stage correlation refinement** (https://github.com/zju3dv/EfficientLoFTR). Drop-in replacement for LoFTR that explicitly diagnoses and fixes the fine-stage location variance that limits sub-pixel disparity, at lower cost. Best matcher-side accuracy gain with least integration risk.
3. **SURE — evidential per-match aleatoric + epistemic uncertainty** (https://arxiv.org/abs/2603.04869 , code announced at https://github.com/LSC-ALAN/SURE). The only semi-dense matcher found that emits usable per-match confidence/variance, which is exactly what per-cluster weighting and the Kalman filter need; caveat: code not yet released at v1.
4. **Eising spherical triangulation** (https://ar5iv.labs.arxiv.org/html/2206.03928). Replaces the paraxial D = ξ₀B/(π|d|) formula with closed-form unit-sphere ray triangulation that stays valid at large zenith angles and yields a triangulation residual for free.
5. **ORION star-based all-sky calibration** (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265959 , https://goa.uva.es/orion-app/). Open-source implementation of exactly the star/infinity-reference extrinsic method, and the cross-check for SuMo; gives an independent absolute orientation for night-time calibration and drift detection.
6. **Blum et al. 2021 conditional-probability CBH fusion** (https://amt.copernicus.org/articles/14/5199/2021/). Reusable statistical fusion of many estimates into a likeliest height with empirical, height-dependent error weights — a principled upgrade over fixed per-frame cluster selection and a natural way to combine multiple image sub-regions.

Honourable mentions: ARM `ACT` wavelet covariance transform for ceiling-layer ground truth (https://github.com/ARM-Development/ACT); MAGSAC++ for a data-driven inlier noise scale (https://github.com/danini/magsac); Kalibr for calibration covariances (https://github.com/ethz-asl/kalibr).

---

## What I could not find

- **No per-match or per-pixel uncertainty in any mainstream detector-free matcher other than SURE.** LoFTR, EfficientLoFTR, RoMa, DKM, OmniGlue and MatchAnything all publish pose/homography or AUC metrics only; none exposes a confidence-calibrated variance per match. SURE (ICRA 2026) is the exception, and its code was still "will be available" at arXiv v1.
- **No published quantification of LoFTR-style fine-stage sub-pixel localisation error in pixels on textureless/low-contrast imagery.** The only explicit mechanism-level statement is EfficientLoFTR's "spatial variance … caused by the expectation over the entire correlation patch"; no paper reports a measured px-level bias/σ for the soft-argmax refinement. ConDL measures reprojection error in px but on synthetic SIDAR pairs and does not isolate sub-pixel bias. This is a genuine gap.
- **No calibration-drift study for ground-based fisheye cloud-camera rigs** (thermal/wind). Drift work exists for structural-monitoring cameras, PTZ CCTV and displacement metrology, and SuMo can *monitor* drift, but no published drift budget/rate for an all-sky cloud stereo rig was found.
- **No application of IMM / multi-hypothesis tracking / Kalman filtering to ceilometer cloud-layer time series specifically.** General MHT/IMM literature and lidar multi-target tracking exist; PBL-height retrieval uses wavelets/thresholds, not target tracking. The project's per-frame cluster selection has no direct literature template found.
- **No explicit treatment of wind-advection lag between two separated ground cameras and a collocated zenith ceilometer.** Camera-derived motion is compared to wind lidars/radiosondes, but no paper found corrects a camera-vs-ceilometer CBH comparison for the advection time offset. Cross-correlation with variable lag and advection estimation from image motion are the available pieces, not a published method.
- **No open-source end-to-end CBH-from-fisheye-stereo pipeline.** Beekmans 2016, Crispel & Roberts 2018, Blum 2021 and Herrero del Barrio 2026 all describe bespoke pipelines with no released code. Reusable components exist (OcamCalib, OpenCV, ORION, SuMo, ACT) but not the assembled system.
- **No cloud-specific self-consistency validation framework**; the usable ideas are borrowed from generic stereo (Leclerc & Luong) and from the ad-hoc filters in the CBH papers.
- **Unfetched sources (stated limits):** Öktem et al. 2014 returned HTTP 403 from the AMS host on both attempts (numbers taken from Crispel & Roberts' summary instead); the All-sky-photogrammetry HAL PDF and the amt-2017-203 PDF failed to fetch (PDF content type unsupported / connection failure) after 2 attempts each, so those were read via their Copernicus HTML/XML counterparts. The `web_search` backend also failed intermittently (~half of batched calls) and was retried.

---

## Availability checked (2025-10-13)

Checked against the project's own environment (`conda` env `clouds`) and PyPI,
so the plan does not depend on a tool that cannot be obtained:

| Candidate | Availability | Note |
|---|---|---|
| **SuMo / Sky-Imaging** (DLR, Sun+Moon calibration) | **not on PyPI** (`pip index versions sky-imaging` → no matching distribution) | Published as an open-source package alongside the paper ([elib.dlr.de/218121](https://elib.dlr.de/218121/)); source would have to be obtained and installed manually. |
| **ORION** (star calibration) | source released on Zenodo ([zenodo.org/records/5595851](https://zenodo.org/records/5595851)) | MATLAB/standalone app; not a Python dependency. |
| **EfficientLoFTR** | weights on HuggingFace ([zju-community/efficientloftr](https://huggingface.co/zju-community/efficientloftr)); code at [github.com/zju3dv/EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR) | Not in the installed kornia (kornia exposes `LoFTR`, `LightGlue`, `DISK`, `ALIKED`, `DeDoDe`, `match_adalam`, …). Requires a separate install or a HF-weights path. |
| **kornia built-ins** | present in the installed env | `LoFTR` (current backend) plus `LightGlue`, `DISK`, `ALIKED`, `DeDoDe` — usable for a matcher comparison without new dependencies. |
| **Eising spherical triangulation** | method only, no repository identified | Closed-form; small enough to implement and check against the paraxial limit in the image centre. |

Consequence for the plan: the infinity-reference check (W1) must be
**implemented in-project** using the solar-ephemeris approach, because neither
SuMo nor ORION is a drop-in dependency here. Their published methods define the
target accuracy (SuMo reports RMSE ≤ 0.14°). The matcher comparison (W3) can
start from what kornia already provides.
