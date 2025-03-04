
import json
from pathlib import Path

from tracktrain import ModelDir,ModelSet

if __name__=="__main__":
    model_root_dir = Path("data/models/new")
    model_dirs = [ModelDir(p) for p in model_root_dir.iterdir()]
    model_dirs = [md for md in model_dirs if md.config["model_type"]=="accfnn"]

    for md in sorted(model_dirs, key=lambda a:int(a.name.split("-")[-1])):
        print(md.name, md.config.get("notes"))
