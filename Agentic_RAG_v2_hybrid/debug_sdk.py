
import sys
import os

# Try various potential paths for cactus
potential_paths = [
    "cactus/python/src",
    "../cactus/python/src",
    "../../cactus/python/src",
]

found = False
for p in potential_paths:
    full_path = os.path.abspath(p)
    if os.path.exists(full_path):
        print(f"Adding {full_path} to sys.path")
        sys.path.insert(0, full_path)
        found = True

try:
    import cactus
    print("\n--- Members of cactus ---")
    for member in dir(cactus):
        print(member)
    print("------------------------\n")
except ImportError as e:
    print(f"ImportError: {e}")
    print(f"Current sys.path: {sys.path}")
