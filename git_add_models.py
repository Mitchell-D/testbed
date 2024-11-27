#!/nas/rhome/mdodson/.micromamba/envs/learn3/bin/python
""" Simple script for adding ModelDir serial skeletons to the git repo """
import sys
import shlex
import subprocess
from pathlib import Path

if __name__=="__main__":
    if len(sys.argv) > 1:
        model_parent_dir = Path(sys.argv[1])
    else:
        model_parent_dir = Path(f"data/models/new/")

    substrings = [".png", "_config.json", "_prog.csv", "_summary.txt"]
    model_dirs = list(model_parent_dir.iterdir())
    md_substrings = [(m,s) for m in model_dirs for s in substrings]
    output = []
    for md in model_dirs:
        for ss in substrings:
            tmp_cmd = shlex.split(f"git add -f {md.as_posix()}/{md.name}{ss}")
            print(tmp_cmd)
            output.append(subprocess.run( tmp_cmd, capture_output=True))

    for md,op in zip(model_dirs,output):
        print(f"\n{md}\n{op}")
