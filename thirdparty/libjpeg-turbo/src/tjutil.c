/*
 * Copyright (C)2011, 2019 D. R. Commander.  All Rights Reserved.
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

#ifdef _WIN32

#include <windows.h>
#include "tjutil.h"

static double getFreq(void)
{
  LARGE_INTEGER freq;

  if (!QueryPerformanceFrequency(&freq)) return 0.0;
  return (double)freq.QuadPart;
}

static double f = -1.0;

double getTime(void)
{
  LARGE_INTEGER t;

  if (f < 0.0) f = getFreq();
  if (f == 0.0) return (double)GetTickCount() / 1000.;
  else {
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / f;
  }
}

#else

#include <stdlib.h>
#include <sys/time.h>
#include "tjutil.h"

double getTime(void)
{
  struct timeval tv;

  if (gettimeofday(&tv, NULL) < 0) return 0.0;
  else return (double)tv.tv_sec + ((double)tv.tv_usec / 1000000.);
}

#endif
