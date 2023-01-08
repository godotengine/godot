import time
import random

random.seed(int(time.time()))

s = ' '
for i in range(0, 31):
	s += "{:.3f}".format(random.random()) + 'F, '
s += "{:.3f}".format(random.random()) + 'F '

code_gen = '''// markers.gen.h
/* THIS FILE IS GENERATED DO NOT EDIT */

#ifndef MARKERS_GEN_H
#define MARKERS_GEN_H

double * markers_f64 = {%s};

#endif

''' % (s)

print(code_gen)
