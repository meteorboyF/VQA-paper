"""E0 - Environment & schema audit (CPU). Stages data, prints real JSON
schemas, validates VQA/Quality joinability, writes audit.json."""
import json
import os

from src import config, env, expstate, progress, resultlog, staging

EXP = "E0"


def _peek_json(path, label=""):
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return {}
    obj = json.load(open(path))
    base = os.path.basename(path)
    split_hint = "train" if "train" in base else "val" if "val" in base else "test"
    recs = staging.annotation_records(obj, split_hint)
    first = recs[0] if recs else (obj if isinstance(obj, dict) else {})
    print(f"\n=== {label or path} ===")
    print(f"  type={type(obj).__name__}  records={len(recs)}  "
          f"top_len={len(obj) if hasattr(obj, '__len__') else '?'}")
    print(f"  keys of first item: {list(first.keys())}")
    print(f"  sample:\n{json.dumps(first, indent=4)[:600]}")
    return first


def main():
    progress.install_error_hook("E0 schema audit")
    env.seed_everything()
    env.check_gpu(EXP)
    env.mount_drive()
    config.ensure_output_dirs()
    config.print_output_locations()

    audit_path = os.path.join(config.RESULTS_E0, "audit.json")
    if expstate.is_done(EXP, config.RESULTS_E0, required=[audit_path]):
        expstate.skip_banner(EXP, config.RESULTS_E0)
        # Still make sure the *pending* experiments on this fresh runtime
        # have their local data (cheap no-op on a warm runtime).
        staging.stage_for_pending()
        return

    pbar = progress.notebook_bar("E0 schema audit", total=7)
    progress.step(pbar, "Drive mounted, output dirs ready")

    # ── 1. Stage all raw zips to local disk (auto-discovery on Drive) ──
    print("\n[E0] Staging data zips to local disk...")
    staged = staging.stage_kinds(
        list(staging.ANNOTATION_KINDS) + list(staging.IMAGE_KINDS))
    progress.step(pbar, "Raw zips staged")

    # ── 2. Count local images per split ──
    print("\n[E0] Counting local images...")
    img_counts = {}
    for split in ("images_train", "images_val"):
        d = os.path.join(config.LOCAL_BASE, split)
        if os.path.exists(d):
            n = env.count_images(d)
            img_counts[split] = n
            print(f"  {split}: {n} images")
    progress.step(pbar, "Local image counts completed")

    # ── 3. Print REAL JSON schemas (the bug-catcher) ──
    schema_info = {}
    ann_paths = {}
    for split in ("train", "val"):
        p = staging.find_annotation_json("vqa", split)
        if p:
            ann_paths[f"vqa_{split}"] = p
            schema_info[f"vqa_{split}"] = _peek_json(p, label=f"VQA {split}")
    for split in ("train", "val", "test"):
        p = staging.find_annotation_json("quality", split)
        if p:
            ann_paths[f"qual_{split}"] = p
            schema_info[f"qual_{split}"] = _peek_json(p, label=f"Quality {split}")
    progress.step(pbar, "Annotation schemas printed")

    # ── 4. VQA <-> Quality image-ID overlap ──
    print("\n[E0] Checking VQA <-> Quality image-ID overlap...")
    overlap_info = {}
    for split in ("train", "val"):
        vqa_path = ann_paths.get(f"vqa_{split}")
        qual_path = ann_paths.get(f"qual_{split}")
        if not (vqa_path and qual_path):
            continue
        vqa_data = staging.annotation_records(json.load(open(vqa_path)), split)
        qual_data = staging.annotation_records(json.load(open(qual_path)), split)
        vqa_images = {os.path.basename(str(it.get("image", it.get("file_name", ""))))
                      for it in vqa_data}
        qual_images = {os.path.basename(str(it.get("image", it.get("file_name", ""))))
                       for it in qual_data}
        overlap = vqa_images & qual_images
        overlap_info[split] = {
            "n_vqa": len(vqa_images),
            "n_quality": len(qual_images),
            "overlap": len(overlap),
            "pct_vqa_covered": len(overlap) / max(len(vqa_images), 1),
        }
        print(f"  {split}: VQA={len(vqa_images)}  Quality={len(qual_images)}  "
              f"Overlap={len(overlap)} "
              f"({100 * len(overlap) / max(len(vqa_images), 1):.1f}% of VQA)")
    progress.step(pbar, "VQA/quality image overlap checked")

    # ── 5. Fatal validation: E0 must not pass unless both datasets join ──
    fatal_issues = []
    for required_key in ("vqa_train", "vqa_val", "qual_train", "qual_val"):
        if required_key not in schema_info:
            fatal_issues.append(f"missing annotation schema: {required_key}")
    for split in ("train", "val"):
        if img_counts.get(f"images_{split}", 0) <= 0:
            fatal_issues.append(f"no local image files staged for split={split}")
        ov = overlap_info.get(split)
        if ov is None:
            fatal_issues.append(f"cannot compute VQA/Quality overlap for split={split}")
        elif ov.get("overlap", 0) <= 0:
            fatal_issues.append(f"zero VQA/Quality overlap for split={split}")
        elif ov.get("pct_vqa_covered", 0) < 0.50:
            fatal_issues.append(
                f"low VQA coverage by Quality labels for split={split}: "
                f"{ov.get('pct_vqa_covered', 0):.1%}")
    if fatal_issues:
        print("\n[E0 FATAL] Dataset staging/schema validation failed:")
        for issue in fatal_issues:
            print(f"  - {issue}")
        print("\n[E0 FATAL] Do not run E1. Fix Drive zip paths/files, then rerun E0.")
        raise RuntimeError("E0 dataset validation failed; see fatal issues above")

    # ── 6. Cross-check: overlap vs local image count ──
    for split in ("train", "val"):
        img_n = img_counts.get(f"images_{split}", 0)
        ovlp_n = overlap_info.get(split, {}).get("overlap", 0)
        if img_n > 0 and ovlp_n > 0 and abs(img_n - ovlp_n) / max(img_n, 1) > 0.10:
            print(f"\n[E0] WARN: {split} image-file count ({img_n}) differs by "
                  f">10% from annotation overlap ({ovlp_n}). Check zip structure.")
        else:
            print(f"[E0] {split}: image files ({img_n}) ~ annotation overlap "
                  f"({ovlp_n}) - OK")

    # ── 7. Write audit.json ──
    audit = {
        "image_counts_local": img_counts,
        "annotation_overlap": overlap_info,
        "sample_schemas": {k: list(v.keys()) for k, v in schema_info.items() if v},
        "annotation_paths": ann_paths,
        "staged_dirs": staged,
    }
    os.makedirs(config.RESULTS_E0, exist_ok=True)
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    progress.step(pbar, "audit.json written")

    resultlog.log_run(EXP, metrics=audit, params={"seed": config.SEED},
                      results_dir=config.RESULTS_E0, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, config.RESULTS_E0, artifacts=[audit_path])
    progress.step(pbar, "E0 result logged")
    pbar.close()

    print("\n[E0 DONE] audit.json written. Check the schemas above before running E1.")
    print("ACTION: If the field names printed above differ from what data_assembly.py")
    print("        expects, edit src/data_assembly.py FIELD_MAP_VQA / FIELD_MAP_QUALITY.")
