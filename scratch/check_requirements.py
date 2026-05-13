import ast
import os

imported_packages = set()

for root, dirs, files in os.walk("."):
    dirs[:] = [
        d for d in dirs
        if d not in ["venv", "venv312", ".venv", ".git", "__pycache__",
                     "node_modules", ".pytest_cache", "scratch"]
    ]
    for fname in files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(root, fname)
        try:
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_packages.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_packages.add(node.module.split(".")[0])
        except Exception:
            pass

with open("requirements.txt") as f:
    requirements = set()
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("-"):
            pkg = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
            requirements.add(pkg.lower().replace("-", "_"))

stdlib = {
    "os", "sys", "json", "re", "time", "datetime", "math", "random",
    "collections", "itertools", "functools", "pathlib", "typing",
    "abc", "copy", "io", "gc", "threading", "subprocess", "traceback",
    "warnings", "logging", "hashlib", "base64", "urllib", "http",
    "email", "html", "xml", "csv", "sqlite3", "decimal", "fractions",
    "statistics", "struct", "array", "queue", "heapq", "bisect",
    "contextlib", "dataclasses", "enum", "inspect", "ast", "dis",
    "importlib", "pkgutil", "platform", "signal", "socket", "ssl",
    "uuid", "secrets", "string", "textwrap", "unicodedata", "codecs",
    "atexit", "builtins",
}

third_party = imported_packages - stdlib
not_in_requirements = {
    p for p in third_party
    if p.lower().replace("-", "_") not in requirements
    and not p.startswith("_")
    and p not in {"__future__", "typing_extensions"}
}

# Filter out local project modules
local_modules = {"data", "pipeline", "agents", "api", "app", "scripts", "scheduler"}
not_in_requirements = {p for p in not_in_requirements if p not in local_modules}

print("=== Packages imported but potentially missing from requirements.txt ===")
for pkg in sorted(not_in_requirements):
    print(f"  {pkg}")

print(f"\n=== Summary ===")
print(f"Total third-party imports found: {len(third_party)}")
print(f"Potentially missing from requirements.txt: {len(not_in_requirements)}")
print(f"\nAll found packages: {sorted(third_party)}")
