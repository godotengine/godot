#!/usr/bin/env python

import sys
import os

includes = sys.argv[1]
output_file = sys.argv[2]
input_file = sys.argv[3]

can_remove = {}

lipo_command = ''

exit_code = 1

for arch in ['32', '64']:
    if arch == '32' and input_file.endswith('x86_64.asm'):
        can_remove[arch] = False
    else:
        command = 'yasm ' + includes + ' -f macho' + arch + ' -D X86_' + arch + ' -o ' + output_file + '.' + arch + ' ' + input_file
        print(command)
        if os.system(command) == 0:
            lipo_command += output_file + '.' + arch + ' '
            can_remove[arch] = True
        else:
            can_remove[arch] = False

if lipo_command != '':
    lipo_command = 'lipo -create ' + lipo_command + '-output ' + output_file
    print(lipo_command)
    if os.system(lipo_command) == 0:
        exit_code = 0

for arch in ['32', '64']:
    if can_remove[arch]:
        os.remove(output_file + '.' + arch)

sys.exit(exit_code)
