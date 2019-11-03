import binascii
import os.path
import sys


def tof(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    content = content.replace("0x", "")
    content = content.split(',')
    for i in range(len(content)):
        if len(content[i]) == 1:
            content[i] = "0" + content[i]
    content = "".join(content)
    with open(filepath + ".file", 'wb') as f:
        content = f.write(content.decode("hex"))
    print(os.path.basename(filepath) + ".file created.")
    exit(0)


def toa(filepath):
    with open(filepath, 'rb') as f:
        content = f.read()
    content = binascii.hexlify(content)
    content = [content[i:i + 2] for i in range(0, len(content), 2)]
    content = ",0x".join(content)
    content = "0x" + content
    content = content.replace("0x00", "0x0")
    with open(filepath + ".array", 'w') as f:
        content = f.write(content)
    print(os.path.basename(filepath) + ".array created.")
    exit(0)


def usage():
    print("========================================================\n\
#\n\
# Usage: python file-hex-array.py [action] [option]\n\
#\n\
# Arguments:\n\
#          action ==>   toa   # convert file to array [option is file path]\n\
#                       tof   # convert array to file [option is array file path]\n\
#\n\
# Example : python file-hex-array.py toa 1.png\n\
#\n\
========================================================")
    exit(1)

if len(sys.argv) != 3:
    usage()
if sys.argv[1] == "toa" and os.path.isfile(sys.argv[2]):
    toa(sys.argv[2])
elif sys.argv[1] == "tof" and os.path.isfile(sys.argv[2]):
    tof(sys.argv[2])
else:
    usage()
