#!/Users/mtdodson/.micromamba/envs/aes/bin/python
"""
Script for parsing individual class or method contexts from python files

 - Provide only 1 or more files as positional arguments to print a list of
   valid context names associated with each file.
 - Pass a valid context name after the "-c" or "--context" flag to search for
   and return a specific namespace.
 - Pass the "-g" or "--globals" flag to also print all the zero-indented lines
   that are not associated with a method or class context
"""
import argparse
import glob
import subprocess
import sys
from pathlib import Path
from pprint import pprint

## list of initial strings that indicate a zero-indented line opens a context,
## mapped to characters that can terminate the name portion of a context.
scope_openers = {"def":("(",), "class":(":","(")}

def extract_contexts(python_path:Path, indent_size=4):
    """
    Separates python source files into distinct contexts base on indent rules

    :@param python_path: Path to a valid ".py" source file to parse
    :@param indent_size: Width of each indentation level
    """
    contexts = []
    names = []
    global_vars = []
    in_context = False
    with python_path.open("r") as fp:
        lines = [l.replace("\n","") for l in fp.readlines() if l != "\n"]
    for i,l in enumerate(lines):
        tmp_opener = l.split(" ")[0]
        if tmp_opener in scope_openers.keys():
            tmp_name = l.split(" ")[1]
            name_terminal = min(
                    tmp_name.index(o)
                    for o in scope_openers[tmp_opener]
                    if o in tmp_name
                    )
            names.append(tmp_name[:name_terminal])
            contexts.append([l])
            in_context = True
        elif l[:indent_size] == indent_size*" ":
            if in_context:
                contexts[-1].append(l)
        else:
            global_vars.append(l)
            in_context = False
    global_vars = "\n".join(global_vars)
    return {n:"\n".join(ns) for n,ns in zip(names,contexts)}, global_vars

def get_user_args():
    """
    Parse arguments relevant for specifying files, context names, and globals
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "src_path",
            metavar="python_source_file",
            nargs="*",
            type=str,
            default="",
            help="Path of existing python source files to search for " \
                    "contexts. Accepts * as wildcard."
            )
    parser.add_argument(
            "-c", "--context",
            metavar="context_to_return",
            required=False,
            type=str,
            default=None,
            help="Optional signature of a method or class context to " \
                    "print out if found",
            )
    parser.add_argument(
            "-g", "--globals",
            required=False,
            action="store_true",
            default=False,
            help="If flag provided, global values from each file will " \
                    "be printed as well."
            )
    args = parser.parse_args()
    if args.src_path == "":
        file_args = sys.stdin.read().split(" ")
    else:
        file_args = args.src_path
    in_files = list(map(lambda p:Path(p.replace("\n","")), file_args))
    return_context = args.context
    return_globals = args.globals
    for f in in_files:
        assert f.exists(), f"File doesn't exist: {f.as_posix()}"
    return in_files, return_context, return_globals


if __name__=="__main__":
    ## Parse command line arguments
    in_files,return_context,return_globals = get_user_args()
    ## iterate over provided files
    for f in in_files:
        tmp_contexts,tmp_globals = extract_contexts(f)
        ## If no specific context is provided, print all valid context names
        if return_context is None:
            print("\n" + 8*"=" + f"( {f.name} )" + 8*"=" + "\n")
            for l in tmp_contexts.keys():
                print(l)
        ## If a valid context was provided print out the full namespace
        elif return_context in tmp_contexts.keys():
            print("\n" + 8*"=" + f"( {f.name} )" + 8*"=" + "\n")
            print(tmp_contexts[return_context])
        ## Print globals as well if requested
        if return_globals:
            print(tmp_globals)
