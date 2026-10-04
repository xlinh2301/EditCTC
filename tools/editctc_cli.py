#!/usr/bin/env python3
"""
EditCTC Unified CLI & Experiment Automation Tool
Provides one-command execution for Colab GPU setup, remote inference, local HTML visualization, and Git synchronization.
"""

import os
import sys
import argparse
import subprocess
import json

COLAB_BIN = "/mnt/d/workspace/sdd/.venv/bin/colab"
WORKSPACE_DIR = "/mnt/d/workspace/EditCTC"
SCRIPTS_DIR = os.path.join(WORKSPACE_DIR, "scripts")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
TOOLS_DIR = os.path.join(WORKSPACE_DIR, "tools")

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

def cmd_git_sync(args):
    """Stages, commits, and pushes updates to GitHub repository."""
    msg = args.message or "feat: add Colab GPU automated pipeline, unified CLI, and local HTML visualizer"
    print(f"🐙 Syncing changes to Git repository...")
    run_cmd("git add -A")
    run_cmd(f'git commit -m "{msg}"', check=False)
    run_cmd("git push origin feat/sdd-wikiskill-knowledge-architecture", check=False)
    print("✅ Changes pushed to GitHub successfully!")

def main():
    parser = argparse.ArgumentParser(description="EditCTC Unified CLI & Colab GPU Experiment Automation")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # setup-colab
    p_setup = subparsers.add_parser("setup-colab", help="Install Paddle GPU and dependencies on Colab")
    p_setup.add_argument("-s", "--session", default="editctc-exp", help="Colab session name")
    p_setup.add_argument("-t", "--timeout", type=int, default=300, help="Command timeout in seconds")

    # eval-colab
    p_eval = subparsers.add_parser("eval-colab", help="Run in-domain V4 inference on Colab GPU")
    p_eval.add_argument("-s", "--session", default="editctc-exp", help="Colab session name")
    p_eval.add_argument("-t", "--timeout", type=int, default=300, help="Command timeout in seconds")

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
    elif args.command == "git-sync":
        cmd_git_sync(args)

if __name__ == "__main__":
    main()
