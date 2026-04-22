"""
顺序运行重庆 4 个实验. 每个实验独立日志 + 汇总日志.
"""
import os
import subprocess
import sys
import time
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(HERE, "logs_cq")

EXPERIMENTS = [
    "configs_chongqing/e1_3h_to_3h.yaml",
    "configs_chongqing/e3_mstc_nodetach.yaml",
    "configs_chongqing/e4_mstc_detach.yaml",
    "configs_chongqing/e2_21h_to_3h.yaml",   # 放最后, 最占显存
]


def main():
    os.makedirs(LOG_ROOT, exist_ok=True)
    summary_path = os.path.join(LOG_ROOT, "run_all_summary.txt")
    with open(summary_path, "a", encoding="utf-8") as sfile:
        sfile.write(f"\n\n==== run_chongqing_all.py 启动于 "
                    f"{datetime.datetime.now()} ====\n")

    for cfg in EXPERIMENTS:
        tag = os.path.splitext(os.path.basename(cfg))[0]
        log_path = os.path.join(LOG_ROOT, f"{tag}.stdout.log")
        print(f"\n\n{'#'*70}\n# RUN {cfg}\n#  日志: {log_path}\n{'#'*70}",
              flush=True)
        t0 = time.time()
        cmd = [sys.executable, "-u", "train_chongqing.py", "--config", cfg]
        with open(log_path, "a", encoding="utf-8", errors="replace") as lf:
            lf.write(f"\n==== {datetime.datetime.now()} ====\n")
            ret = subprocess.call(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
        dt = time.time() - t0

        status = {0: "OK", 2: "OOM"}.get(ret, f"ERR{ret}")
        line = f"[{status}] {cfg}  time={dt/60:.1f}min"
        print(line, flush=True)
        with open(summary_path, "a", encoding="utf-8") as sfile:
            sfile.write(line + "\n")


if __name__ == "__main__":
    main()
