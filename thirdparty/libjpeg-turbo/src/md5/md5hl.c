/* mdXhl.c
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <phk@FreeBSD.org> wrote this file.  As long as you retain this notice you
 * can do whatever you want with this stuff. If we meet some day, and you think
 * this stuff is worth it, you can buy me a beer in return.   Poul-Henning Kamp
 * ----------------------------------------------------------------------------
 * libjpeg-turbo Modifications:
 * Copyright (C)2016, 2018-2019, 2022, 2024 D. R. Commander.
 *                                          All Rights Reserved.
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
 * ----------------------------------------------------------------------------
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#define close  _close
#define fstat  _fstat
#define lseek  _lseek
#define read  _read
#define stat  _stat
#else
#include <unistd.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#define LENGTH  16

#include "./md5.h"

static char *MD5End(MD5_CTX *ctx, char *buf)
{
  int i;
  unsigned char digest[LENGTH];
  static const char hex[] = "0123456789abcdef";

  if (!buf)
    buf = malloc(2 * LENGTH + 1);
  if (!buf)
    return 0;
  MD5Final(digest, ctx);
  for (i = 0; i < LENGTH; i++) {
    buf[i + i] = hex[digest[i] >> 4];
    buf[i + i + 1] = hex[digest[i] & 0x0f];
  }
  buf[i + i] = '\0';
  return buf;
}

char *MD5File(const char *filename, char *buf)
{
  return (MD5FileChunk(filename, buf, 0, 0));
}

char *MD5FileChunk(const char *filename, char *buf, off_t ofs, off_t len)
{
  unsigned char buffer[BUFSIZ];
  MD5_CTX ctx;
  struct stat stbuf;
  int f, i, e;
  off_t n;

  MD5Init(&ctx);
#ifdef _WIN32
  f = _open(filename, O_RDONLY | O_BINARY);
#else
  f = open(filename, O_RDONLY);
#endif
  if (f < 0)
    return 0;
  if (fstat(f, &stbuf) < 0)
    return 0;
  if (ofs > stbuf.st_size)
    ofs = stbuf.st_size;
  if ((len == 0) || (len > stbuf.st_size - ofs))
    len = stbuf.st_size - ofs;
  if (lseek(f, ofs, SEEK_SET) < 0)
    return 0;
  n = len;
  i = 0;
  while (n > 0) {
    if ((size_t)n > sizeof(buffer))
      i = read(f, buffer, sizeof(buffer));
    else
      i = read(f, buffer, n);
    if (i < 0)
      break;
    MD5Update(&ctx, buffer, i);
    n -= i;
  }
  e = errno;
  close(f);
  errno = e;
  if (i < 0)
    return 0;
  return (MD5End(&ctx, buf));
}
