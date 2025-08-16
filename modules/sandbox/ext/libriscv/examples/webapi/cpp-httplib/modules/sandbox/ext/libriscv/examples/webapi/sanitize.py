#!/usr/bin/python3
import sys
import subprocess
import os

# Linux compiler detection, looking for riscv64-linux-gnu-g++-[10-14]
# Starting from the top (to get the latest version)
riscv_cross_compiler = "riscv64-linux-gnu-g++"
for i in range(15, 9, -1):
	riscv_cross_compiler = "riscv64-linux-gnu-g++-" + str(i)
	rc = subprocess.call(['which', riscv_cross_compiler])
	if rc == 0:
		break

project_base = sys.argv[1]
project_dir  = sys.argv[2]
method       = sys.argv[3]
# relative local filesystem paths
os.chdir(project_base + "/" + project_dir)
python_codefile = "code.cpp"
python_status   = "status.txt"
# docker volume paths
dc_codefile = python_codefile
dc_binary   = "binary"
dc_symmap   = "symbols.map"

fo = open("symbols.map", "w")
fo.write("main")
fo.close()

# sanitize the code here
sanitized = ""
with open(python_codefile) as fp:
	for line in fp:
		# no sanitation atm
		sanitized += str(line)
#print(sanitized)

# overwrite with sanitized text
fo = open(python_codefile, "w")
fo.write(sanitized)
fo.close()

# docker outside & inside shared folder
local_dir = project_base
dc_shared = "/usr/outside"

dc_extra = []
if method == "linux":
	dc_instance = "linux-rv64gc"
	dc_gnucpp = riscv_cross_compiler
	dc_extra = ["-pthread"]
else:
	dc_instance = "newlib-rv64gc"
	dc_gnucpp = "riscv64-unknown-elf-g++"

# compile the code
cmd = [ #"docker", "exec", dc_instance,
		dc_gnucpp, "-static"] + dc_extra + [
		"-std=c++20", "-O2", dc_codefile, "-o", dc_binary,
		# Fixes a g++ bug where thread::join jumps to 0x0 (missing pthread_join)
		"-Wl,--undefined=pthread_join"] # DO NOT REMOVE
print(cmd)

result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = result.communicate()
returncode = result.returncode

print(stdout)
print(stderr)

fo = open(python_status, "w")
fo.write(stderr.decode("utf-8"))
fo.close()

exit(returncode)
