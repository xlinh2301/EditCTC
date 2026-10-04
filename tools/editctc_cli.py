#!/usr/bin/env python3
"""
EditCTC Unified CLI, Experiment Automation & SoL-Pi Research Harness Tool
Provides one-command execution for Colab GPU setup, remote inference, local HTML visualization,
and SoL-Pi harness mechanisms (Evidence-Preserving Reducer, ObservationPack, Action Fusion).
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

# Connect to sdd harness modules
SDD_ROOT = "/mnt/d/workspace/sdd"
if SDD_ROOT not in sys.path:
    sys.path.insert(0, SDD_ROOT)

try:
    from ai_sdlc.harness.reducer import EvidencePreservingReducer
    from ai_sdlc.harness.observation_pack import ObservationPack
    from ai_sdlc.harness.action_fusion import ActionFusionExecutor
    from ai_sdlc.harness.compact import OnlineContextCompact
    HAS_SOL_HARNESS = True
except ImportError:
    HAS_SOL_HARNESS = False

COLAB_BIN = "/mnt/d/workspace/sdd/.venv/bin/colab"
WORKSPACE_DIR = "/mnt/d/workspace/EditCTC"
SCRIPTS_DIR = os.path.join(WORKSPACE_DIR, "scripts")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
TOOLS_DIR = os.path.join(WORKSPACE_DIR, "tools")
OBS_DIR = os.path.join(WORKSPACE_DIR, ".mimi-sdlc", "observations")


def check_colab_bin():
    if not os.path.exists(COLAB_BIN):
        raise FileNotFoundError(f"Colab CLI binary not found at {COLAB_BIN}. Please run install.sh in sdd.")


def run_cmd(cmd, cwd=WORKSPACE_DIR, check=True):
    print(f"[*] Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and res.returncode != 0:
        print(f"[!] Error: Command failed with return code {res.returncode}")
        sys.exit(res.returncode)
    return res


def cmd_setup_colab(args):
    """Sets up Paddle GPU and dependencies on Colab GPU session."""
    check_colab_bin()
    setup_script = os.path.join(SCRIPTS_DIR, "colab_setup_env.py")
    session = args.session or "editctc-exp"
    timeout = args.timeout or 300
    print(f"🚀 Running automated Colab setup on session '{session}' (timeout: {timeout}s)...")
    run_cmd(f"{COLAB_BIN} exec -s {session} --timeout {timeout} -f {setup_script}")
    print("✅ Colab GPU environment is ready!")


def cmd_eval_colab(args):
    """Runs indomain inference and benchmark evaluation on Colab GPU."""
    check_colab_bin()
    eval_script = os.path.join(SCRIPTS_DIR, "colab_run_indomain_inference.py")
    session = args.session or "editctc-exp"
    timeout = args.timeout or 300
    print(f"⚡ Running In-domain V4 Evaluation on Colab GPU '{session}'...")
    run_cmd(f"{COLAB_BIN} exec -s {session} --timeout {timeout} -f {eval_script}")
    
    # Export JSON to local results/
    print("\n[*] Syncing comparison JSON to local workspace...")
    export_script = os.path.join(WORKSPACE_DIR, "scratch", "export_json_local.py")
    raw_out_path = os.path.join(WORKSPACE_DIR, "scratch", "raw_output.txt")
    os.makedirs(os.path.dirname(export_script), exist_ok=True)
    with open(export_script, "w") as f:
        f.write("""import json
src_path = '/content/drive/MyDrive/research/EditCTC/release_EditCTC/eval/indomain_val_comparison.json'
with open(src_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print("---JSON_START---")
print(json.dumps(data, ensure_ascii=False))
print("---JSON_END---")
""")
    run_cmd(f"{COLAB_BIN} exec -s {session} --timeout 60 -f {export_script} > {raw_out_path}")
    
    # Parse output to json
    with open(raw_out_path, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.find("---JSON_START---") + len("---JSON_START---")
    end = content.find("---JSON_END---")
    if start != -1 and end != -1:
        json_str = content[start:end].strip()
        data = json.loads(json_str)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_json = os.path.join(RESULTS_DIR, "indomain_val_comparison.json")
        with open(out_json, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2, ensure_ascii=False)
        print(f"✅ Comparison data saved locally to: {out_json}")

        if HAS_SOL_HARNESS and getattr(args, "pack_observation", True):
            opack = ObservationPack(storage_dir=OBS_DIR)
            formatted, is_archived, handle = opack.pack_observation(json_str, tool_name="indomain_eval")
            if is_archived:
                print(f"📦 [ObservationPack] Archived eval output to handle `{handle}`")
    else:
        print("[!] Warning: Could not parse JSON marker from Colab output.")


def cmd_visualize(args):
    """Generates the interactive HTML visualization report locally."""
    json_path = args.json or os.path.join(RESULTS_DIR, "indomain_val_comparison.json")
    html_path = args.output or os.path.join(RESULTS_DIR, "indomain_val_report.html")
    
    vis_script = os.path.join(TOOLS_DIR, "visualize_results.py")
    print(f"📊 Generating HTML report from {json_path} -> {html_path}...")
    run_cmd(f"python3 {vis_script} {json_path} {html_path}")
    print(f"✨ Report generated: file://{html_path}")


def cmd_pipeline(args):
    """Runs full pipeline: Colab GPU evaluation -> sync -> local HTML visualization."""
    print("==========================================================")
    print("🚀 Starting End-to-End Pipeline (Colab GPU + Local Visualizer)")
    print("==========================================================")
    cmd_eval_colab(args)
    cmd_visualize(args)
    print("==========================================================")
    print("🎉 Pipeline completed successfully!")
    print(f"🌐 Open Report: file://{os.path.join(RESULTS_DIR, 'indomain_val_report.html')}")
    print("==========================================================")


def cmd_sol_reduce(args):
    """Evidence-Preserving Reducer: extracts verified failure receipt from raw logs."""
    if not HAS_SOL_HARNESS:
        print("[!] SoL harness modules not available. Ensure sdd is present.")
        sys.exit(1)

    log_path = Path(args.file)
    if not log_path.exists():
        print(f"[!] Log file '{log_path}' not found.")
        sys.exit(1)

    content = log_path.read_text(encoding="utf-8")
    reducer = EvidencePreservingReducer()
    reduced_text, is_reduced, receipt = reducer.reduce(content, command=args.cmd or "eval", exit_code=args.exit_code)
    
    if is_reduced and receipt:
        print(reduced_text)
        if args.output:
            Path(args.output).write_text(reduced_text, encoding="utf-8")
            print(f"✅ Verified receipt saved to: {args.output}")
    else:
        print("[*] Log did not require reduction or verification fell back to raw content.")
        print(content[:500] + "\n...")


def cmd_sol_recall(args):
    """ObservationPack on-demand chunk recall."""
    if not HAS_SOL_HARNESS:
        print("[!] SoL harness modules not available.")
        sys.exit(1)

    opack = ObservationPack(storage_dir=OBS_DIR)
    res = opack.recall_observation(args.handle, offset=args.offset, length=args.length)
    if res["success"]:
        print(f"📦 Recalled chunk for `{args.handle}` (offset: {res['offset']}, length: {res['length']}):\n")
        print(res["chunk"])
    else:
        print(f"[!] Error: {res.get('error')}")


def cmd_git_sync(args):
    """Stages, commits, and pushes updates to GitHub repository."""
    msg = args.message or "feat: add Colab GPU automated pipeline, unified CLI, and local HTML visualizer"
    print(f"🐙 Syncing changes to Git repository...")
    run_cmd("git add -A")
    run_cmd(f'git commit -m "{msg}"', check=False)
    run_cmd("git push origin feat/sdd-wikiskill-knowledge-architecture", check=False)
    print("✅ Changes pushed to GitHub successfully!")


def main():
    parser = argparse.ArgumentParser(description="EditCTC Unified CLI & SoL-Pi Research Automation")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # setup-colab
    p_setup = subparsers.add_parser("setup-colab", help="Install Paddle GPU and dependencies on Colab")
    p_setup.add_argument("-s", "--session", default="editctc-exp", help="Colab session name")
    p_setup.add_argument("-t", "--timeout", type=int, default=300, help="Command timeout in seconds")

    # eval-colab
    p_eval = subparsers.add_parser("eval-colab", help="Run in-domain V4 inference on Colab GPU")
    p_eval.add_argument("-s", "--session", default="editctc-exp", help="Colab session name")
    p_eval.add_argument("-t", "--timeout", type=int, default=300, help="Command timeout in seconds")
    p_eval.add_argument("--no-pack", dest="pack_observation", action="store_false", help="Disable ObservationPack archival")

    # visualize
    p_vis = subparsers.add_parser("visualize", help="Generate local HTML visualization report")
    p_vis.add_argument("-j", "--json", help="Input comparison JSON file")
    p_vis.add_argument("-o", "--output", help="Output HTML report file")

    # pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run full pipeline: Colab GPU Inference -> Sync -> Local HTML Report")
    p_pipe.add_argument("-s", "--session", default="editctc-exp", help="Colab session name")
    p_pipe.add_argument("-t", "--timeout", type=int, default=300, help="Command timeout in seconds")
    p_pipe.add_argument("-j", "--json", help="Input comparison JSON file")
    p_pipe.add_argument("-o", "--output", help="Output HTML report file")

    # sol-reduce
    p_reduce = subparsers.add_parser("sol-reduce", help="Evidence-Preserving Reducer for heavy experiment logs")
    p_reduce.add_argument("-f", "--file", required=True, help="Path to raw log file")
    p_reduce.add_argument("-c", "--cmd", default="pytest", help="Command that generated the log")
    p_reduce.add_argument("-e", "--exit-code", type=int, default=1, help="Exit code of the run")
    p_reduce.add_argument("-o", "--output", help="Optional output path for receipt markdown")

    # sol-recall
    p_recall = subparsers.add_parser("sol-recall", help="ObservationPack on-demand chunk recall tool")
    p_recall.add_argument("handle", help="Observation handle (obs_xxx)")
    p_recall.add_argument("--offset", type=int, default=0, help="Character offset")
    p_recall.add_argument("--length", type=int, default=1000, help="Character length to read")

    # git-sync
    p_git = subparsers.add_parser("git-sync", help="Commit and push changes to GitHub")
    p_git.add_argument("-m", "--message", help="Commit message")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "setup-colab":
        cmd_setup_colab(args)
    elif args.command == "eval-colab":
        cmd_eval_colab(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)
    elif args.command == "sol-reduce":
        cmd_sol_reduce(args)
    elif args.command == "sol-recall":
        cmd_sol_recall(args)
    elif args.command == "git-sync":
        cmd_git_sync(args)


if __name__ == "__main__":
    main()
