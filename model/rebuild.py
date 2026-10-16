"""One-shot rebuild of the Edge Board: refresh data, re-run both models, rebuild the page.
Usage: python3 rebuild.py [--skip-fetch]"""
import subprocess, sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
def run(cmd):
    t = time.time(); print(">>", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=HERE, capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-6:])
    print(tail, f"\n   ({time.time()-t:.0f}s, exit {r.returncode})", flush=True)
    if r.returncode: sys.exit(f"step failed: {cmd}")
if "--skip-fetch" not in sys.argv:
    run("bash fetch_data.sh > /dev/null")
run("python3 run_2026.py")
run("python3 ncaa_model.py")
run("python3 build_bundle.py")
run("python3 build_page.py")
