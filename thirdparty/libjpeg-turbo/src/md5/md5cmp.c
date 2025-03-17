/*
 * Copyright (C)2013, 2016 D. R. Commander.  All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the libjpeg-turbo Project nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS",
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <string.h>
#include "./md5.h"
#include "../tjutil.h"

int main(int argc, char *argv[])
{
  char *md5sum = NULL, buf[65];

  if (argc < 3) {
    fprintf(stderr, "USAGE: %s <correct MD5 sum> <file>\n", argv[0]);
    return -1;
  }

  if (strlen(argv[1]) != 32)
    fprintf(stderr, "WARNING: MD5 hash size is wrong.\n");

  md5sum = MD5File(argv[2], buf);
  if (!md5sum) {
    perror("Could not obtain MD5 sum");
    return -1;
  }

  if (!strcasecmp(md5sum, argv[1])) {
    fprintf(stderr, "%s: OK\n", argv[2]);
    return 0;
  } else {
    fprintf(stderr, "%s: FAILED.  Checksum is %s\n", argv[2], md5sum);
    return -1;
  }
}
