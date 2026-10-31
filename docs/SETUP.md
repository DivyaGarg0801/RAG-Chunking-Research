# Windows Setup Guide

This guide helps a new collaborator prepare a separate local copy of the research project. It assumes Windows, PowerShell, Git, and Python 3.13 are installed.

## 1. Check Python and clone the repository

In PowerShell, verify that Python 3.13 is available:

```powershell
py -3.13 --version
```

The output should report Python 3.13.x. This project has been developed with Python 3.13.

Clone the GitHub repository and enter its folder:

```powershell
git clone https://github.com/DivyaGarg0801/RAG-Chunking-Research.git
Set-Location RAG-Chunking-Research
```

`git clone` creates a new local copy. Each collaborator should create their own virtual environment; do not copy another person's `.venv`.

## 2. Create and activate an independent virtual environment

From the project folder, create `.venv` using Python 3.13:

```powershell
py -3.13 -m venv .venv
```

A virtual environment keeps this project's Python packages separate from other projects and from system-wide Python packages.

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

The terminal prompt usually gains a `(.venv)` prefix. You can confirm which Python is active with:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

The executable path should point into this project's `.venv\Scripts` folder.

### If PowerShell blocks activation

Some computers prevent PowerShell scripts from running. Do **not** change the system execution policy for this project. You can run commands through the virtual environment's Python directly instead; this does not require activation:

```powershell
.\.venv\Scripts\python.exe --version
```

Use the same `.\.venv\Scripts\python.exe` prefix for the installation, verification, and test commands below.

Alternatively, make the virtual environment's Scripts folder first on `PATH` for only the current PowerShell session:

```powershell
$env:Path = "$PWD\.venv\Scripts;$env:Path"
python --version
```

This change ends when that terminal session closes and does not change Windows execution-policy settings.

## 3. Install and verify project dependencies

With the environment activated, install the dependencies listed by the project:

```powershell
python -m pip install -r requirements.txt
```

If activation was blocked, use the environment's Python explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The current requirements file pins PyMuPDF. Verify that it imports and print its version:

```powershell
python -c "import pymupdf; print(pymupdf.__version__)"
```

Or, without activation:

```powershell
.\.venv\Scripts\python.exe -c "import pymupdf; print(pymupdf.__version__)"
```

## 4. Run the tests

Run the complete test suite using Python's built-in `unittest` runner:

```powershell
python -m unittest discover -s tests -v
```

Without activation, run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Successful output lists the tests and ends with `OK`.

## 5. Obtain the source manual independently

The Computer Networks Lab Manual is intentionally not included in the Git repository. Obtain it independently from an official source, such as the institution or publisher that issued it. If you cannot identify an official source, ask the project supervisor or instructor where it may be obtained.

Before downloading or using the manual, check its licence, access conditions, and other usage terms. Do not assume that the document may be redistributed or shared with collaborators.

Place your permitted local copy at this exact path in the cloned project:

```text
data\raw\computer_networks_lab_manual.pdf
```

For example, if the file is in your Downloads folder:

```powershell
Copy-Item "$HOME\Downloads\computer_networks_lab_manual.pdf" "data\raw\computer_networks_lab_manual.pdf"
```

The copy command does not download the manual; it only copies a file you already obtained. Keep the original unchanged.

## 6. Generated and processed data

The original PDF and generated fixed-size preview files are intentionally excluded from Git. The PDF may have usage restrictions, and the preview contains text extracted from that source document. The preview output folder is:

```text
experiments\results\fixed_size_preview\
```

After placing an authorized local source copy in `data\raw`, the exploratory fixed-size preview can be regenerated with:

```powershell
python -m scripts.preview_fixed_size
```

Or without activation:

```powershell
.\.venv\Scripts\python.exe -m scripts.preview_fixed_size
```

The script writes its chunk text and summary files under the excluded preview folder and refuses to overwrite existing output files. Processed data and experiment outputs should be reproducible from the authorized source documents, project scripts, and documented settings. Do not treat generated outputs as a substitute for preserving the source-document permissions.

## 7. Check out the recursive chunking branch

The default clone checks out the repository's default branch. To test the recursive chunking feature branch used for its open pull request, first fetch the latest remote branch information:

```powershell
git fetch origin
```

If you do not already have a local branch with that name, create one that tracks the remote branch:

```powershell
git switch --track -c feature/recursive-chunking origin/feature/recursive-chunking
```

If the local branch already exists, switch to it instead:

```powershell
git switch feature/recursive-chunking
```

Confirm the checked-out branch with:

```powershell
git branch --show-current
```
