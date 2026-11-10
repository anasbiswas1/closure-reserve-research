"""Single commit step for this repo. Run from a notebook's final cell only:

    subprocess.run(["python", "tools/commit_cell.py", "message"])

Restores git identity and credentials from the Drive project folder (the project root, or its
.secrets folder as used by this repo's first notebooks), strips all notebook outputs, refuses to
stage data, credential or oversized files, refuses any staged file containing a secret token
pattern, commits if anything is staged, and always pushes so unpushed commits never remain local.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path

DRIVE_PROJECT_ROOT = Path(os.environ.get("CLOSURE_DRIVE_ROOT", "/content/drive/MyDrive/CLOSURE_Research"))
REPO_ROOT = Path(os.environ.get("CLOSURE_REPO_ROOT", str(DRIVE_PROJECT_ROOT / "closure-reserve-research")))
GIT_NAME = "Md Anas Biswas"
GIT_EMAIL = "anasbiswas@gmail.com"
FORBIDDEN_NAMES = {".git-credentials", ".gitconfig", "git-credentials", "gitconfig", "kaggle.json", ".env"}
FORBIDDEN_PREFIXES = ("data/", "external/", ".secrets/")
MAX_FILE_MB = 50
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),            # Hugging Face
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),           # GitHub classic PAT
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),   # GitHub fine-grained PAT
    re.compile(r"sk-[A-Za-z0-9]{32,}"),             # OpenAI-style keys
    re.compile(r"AKIA[0-9A-Z]{16}"),               # AWS access key id
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),         # Google API key
]
TEXT_EXT = {".py", ".ipynb", ".md", ".txt", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh"}


def run(args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"{' '.join(args)}\n{r.stdout}{r.stderr}")
    return r.stdout


def restore_credentials():
    for dotfile, plain in ((".gitconfig", "gitconfig"), (".git-credentials", "git-credentials")):
        for source in (DRIVE_PROJECT_ROOT / dotfile, DRIVE_PROJECT_ROOT / ".secrets" / plain):
            if source.exists():
                shutil.copy(source, Path.home() / dotfile)
                break
    credentials = Path.home() / ".git-credentials"
    if credentials.exists():
        os.chmod(credentials, 0o600)


def strip_outputs():
    notebooks = sorted(glob(str(REPO_ROOT / "notebooks" / "*.ipynb")))
    for notebook in notebooks:
        r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert",
                            "--ClearOutputPreprocessor.enabled=True", "--inplace", notebook],
                           capture_output=True, text=True)
        if r.returncode != 0:
            import nbformat
            nb = nbformat.read(notebook, as_version=4)
            for cell in nb.cells:
                if cell.cell_type == "code":
                    cell["outputs"] = []; cell["execution_count"] = None
            nbformat.write(nb, notebook)
    import nbformat
    for notebook in notebooks:
        nb = nbformat.read(notebook, as_version=4)
        if "widgets" in nb.metadata:
            del nb.metadata["widgets"]; nbformat.write(nb, notebook)
    print(f"outputs: stripped {len(notebooks)} notebook(s)")


def secret_scan(paths):
    hits = []
    for p in paths:
        path = REPO_ROOT / p
        if path.suffix.lower() not in TEXT_EXT or not path.is_file():
            continue
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(txt):
                hits.append(f"{p} ({pat.pattern[:14]}...)")
                break
    return hits


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        raise SystemExit('usage: python tools/commit_cell.py "commit message"')
    message = sys.argv[1].strip()

    restore_credentials()
    os.chdir(REPO_ROOT)
    run(["git", "config", "user.name", GIT_NAME])
    run(["git", "config", "user.email", GIT_EMAIL])
    run(["git", "config", "credential.helper", "store"])

    strip_outputs()

    run(["git", "add", "-A"])
    staged = [l for l in run(["git", "diff", "--cached", "--name-only"]).splitlines() if l.strip()]
    bad = [p for p in staged if Path(p).name in FORBIDDEN_NAMES or p.startswith(FORBIDDEN_PREFIXES)]
    big = [p for p in staged if Path(p).exists() and Path(p).stat().st_size > MAX_FILE_MB * 1024 * 1024]
    secrets = secret_scan(staged)
    if bad or big or secrets:
        run(["git", "reset", "-q"], check=False)
        raise SystemExit(f"refused: forbidden={bad} oversized={big} secrets_in={secrets}")

    if staged:
        commit = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
        print(commit.stdout or commit.stderr)
    else:
        print("nothing new to commit; pushing any unpushed commits")

    push = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
    print(push.stdout or push.stderr)
    print(run(["git", "log", "--oneline", "-1"]).strip())
    print(run(["git", "status", "--short"]).strip() or "working tree clean")


if __name__ == "__main__":
    main()
