# asm-mips

The files in this directory are almost direct copies from Android NDK r12, with
the exception of changing the include guards to Breakpad ones. They are copied
from the MIPS asm/ directory, but are meant to be used as replacements for both
asm/ and machine/ includes since the files in each are largely duplicates.

Some MIPS asm/ and all machine/ headers were removed in the move to unified NDK
headers, so Breakpad fails to compile on newer NDK versions without these files.