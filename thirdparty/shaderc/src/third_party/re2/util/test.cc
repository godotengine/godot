// Copyright 2009 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>

#include "util/test.h"

DEFINE_string(test_tmpdir, "/var/tmp", "temp directory");

struct Test {
  void (*fn)(void);
  const char *name;
};

static Test tests[10000];
static int ntests;

void RegisterTest(void (*fn)(void), const char *name) {
  tests[ntests].fn = fn;
  tests[ntests++].name = name;
}

int main(int argc, char** argv) {
  for (int i = 0; i < ntests; i++) {
    printf("%s\n", tests[i].name);
    tests[i].fn();
  }
  printf("PASS\n");
  return 0;
}
