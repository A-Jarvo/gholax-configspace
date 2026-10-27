"""
Run gholax on N noisy mock data vectors at the fiducial cosmology and summarise
parameter recovery in sigma units.

Usage (from the repo root):
    python run_mock_suite.py --config example_configs/config_configspace_emulator.yaml \
        --theory theory_fid.npy --n 10
    python run_mock_suite.py ... --dry-run        # print what would run, run nothing

For each seed 1..N it:
  1. calls make_dv.py   -> example_configs/dv_seed{k}.h5   (noisy, frac errors)
  2. writes a config copy with that data vector and output_file chains/mock_seed{k}
  3. runs `run-gholax` on it (skipped if chains/mock_seed{k}.samples.0.0.txt exists)
Then it writes runs.json and calls analyze_runs.py.
Rerun the same command to resume after an interruption.
"""
import argparse
import json
import os
import subprocess
import sys
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--theory", required=True)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--frac", type=float, default=0.10)
ap.add_argument("--dvdir", default="example_configs")
ap.add_argument("--chaindir", default="chains")
ap.add_argument("--tag", default="mock",
                help="name for this suite; files become chains/<tag>_seed<k>, runs_<tag>.json (default 'mock' keeps the old names)")
ap.add_argument("--zmin-proj", type=float, default=-1,
                help="override Nx2PTCorrelationFunction.zmin_proj in the config copies (-1 keeps the config value)")
ap.add_argument("--w-theta-max", type=float, default=-1,
                help="override the w_theta upper theta cut [rad] in the config copies (-1 keeps the config value)")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

# True values of the free parameters (As in units of 1e-9). Edit if the theory
# vector was made with different biases.
TRUTH = {"As": 2.107, "omch2": 0.11923, "b_1_0": 1.5, "b_1_1": 1.7}
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(args.chaindir, exist_ok=True)

base = yaml.safe_load(open(args.config))
runs = []
for k in range(1, args.n + 1):
    legacy = args.tag == "mock"
    dv = os.path.join(args.dvdir, f"dv_seed{k}.h5" if legacy else f"dv_{args.tag}_seed{k}.h5")
    cfg_path = os.path.join(args.dvdir, f"config_{args.tag}_seed{k}.yaml")
    prefix = os.path.join(args.chaindir, f"{args.tag}_seed{k}")
    runs.append({"label": f"seed{k}", "prefix": prefix, "truth": TRUTH})

    if os.path.exists(f"{prefix}.samples.0.0.txt"):
        print(f"[seed {k}] chains exist, skipping")
        continue

    cmd_dv = [sys.executable, os.path.join(here, "make_dv.py"), "--theory", args.theory,
              "--out", dv, "--frac", str(args.frac), "--seed", str(k)]
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["output_file"] = prefix
    cfg["likelihood"]["Nx2PTCorrelationFunction"]["data_vector"]["data_vector_info_filename"] = dv
    nx = cfg["likelihood"]["Nx2PTCorrelationFunction"]
    if args.zmin_proj >= 0:
        nx["zmin_proj"] = args.zmin_proj
    if args.w_theta_max >= 0:
        for pair in nx["data_vector"]["scale_cuts"].get("w_theta", {}):
            nx["data_vector"]["scale_cuts"]["w_theta"][pair][1] = args.w_theta_max
    cmd_run = ["run-gholax", cfg_path]

    print(f"[seed {k}] {' '.join(cmd_dv)}\n[seed {k}] {' '.join(cmd_run)}")
    if args.dry_run:
        continue
    subprocess.run(cmd_dv, check=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    subprocess.run(cmd_run, check=True)

runs_file = "runs.json" if args.tag == "mock" else f"runs_{args.tag}.json"
with open(runs_file, "w") as f:
    json.dump(runs, f, indent=1)
print("Wrote", runs_file)
if not args.dry_run:
    subprocess.run([sys.executable, os.path.join(here, "analyze_runs.py"), runs_file])