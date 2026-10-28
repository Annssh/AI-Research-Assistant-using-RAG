"""Rewrite requirements.txt with the exact versions installed in this environment."""
from importlib.metadata import version

with open("requirements.txt") as f:
    names = [line.split(">=")[0].split("==")[0].strip()
             for line in f if line.strip() and not line.startswith("#")]

pinned = [f"{name}=={version(name.split('[')[0])}" for name in names]

with open("requirements.txt", "w") as f:
    f.write("\n".join(pinned) + "\n")
print("\n".join(pinned))