import os
import sys
import shutil

src_dir = 'tools/hooks'
dist_dir = '.git/hooks'

if not os.path.exists(dist_dir):
    print("Error: this directory is not a git repository!")
    sys.exit(1)

for file in os.listdir(src_dir):
    shutil.copy(os.path.join(src_dir, file), dist_dir)

print("Copied tools/hooks/* to .git/hooks")
