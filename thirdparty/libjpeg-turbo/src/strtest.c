/*
 * Copyright (C)2022-2023 D. R. Commander.  All Rights Reserved.
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

#include "jinclude.h"
#include <errno.h>


#define CHECK_VALUE(actual, expected, desc) \
  if (actual != expected) { \
    printf("ERROR in line %d: " desc " is %d, should be %d\n", \
           __LINE__, actual, expected); \
    return -1; \
  }

#define CHECK_ERRNO(errno_return, expected_errno) \
  CHECK_VALUE(errno_return, expected_errno, "Return value") \
  CHECK_VALUE(errno, expected_errno, "errno") \


#ifdef _MSC_VER

void invalid_parameter_handler(const wchar_t *expression,
                               const wchar_t *function, const wchar_t *file,
                               unsigned int line, uintptr_t pReserved)
{
}

#endif


int main(int argc, char **argv)
{
#if !defined(NO_GETENV) || !defined(NO_PUTENV)
  int err;
#endif
#ifndef NO_GETENV
  char env[3];
#endif

#ifdef _MSC_VER
  _set_invalid_parameter_handler(invalid_parameter_handler);
#endif

  /***************************************************************************/

#ifndef NO_PUTENV

  printf("PUTENV_S():\n");

  errno = 0;
  err = PUTENV_S(NULL, "12");
  CHECK_ERRNO(err, EINVAL);

  errno = 0;
  err = PUTENV_S("TESTENV", NULL);
  CHECK_ERRNO(err, EINVAL);

  errno = 0;
  err = PUTENV_S("TESTENV", "12");
  CHECK_ERRNO(err, 0);

  printf("SUCCESS!\n\n");

#endif

  /***************************************************************************/

#ifndef NO_GETENV

  printf("GETENV_S():\n");

  errno = 0;
  env[0] = 1;
  env[1] = 2;
  env[2] = 3;
  err = GETENV_S(env, 3, NULL);
  CHECK_ERRNO(err, 0);
  CHECK_VALUE(env[0], 0, "env[0]");
  CHECK_VALUE(env[1], 2, "env[1]");
  CHECK_VALUE(env[2], 3, "env[2]");

  errno = 0;
  env[0] = 1;
  env[1] = 2;
  env[2] = 3;
  err = GETENV_S(env, 3, "TESTENV2");
  CHECK_ERRNO(err, 0);
  CHECK_VALUE(env[0], 0, "env[0]");
  CHECK_VALUE(env[1], 2, "env[1]");
  CHECK_VALUE(env[2], 3, "env[2]");

  errno = 0;
  err = GETENV_S(NULL, 3, "TESTENV");
  CHECK_ERRNO(err, EINVAL);

  errno = 0;
  err = GETENV_S(NULL, 0, "TESTENV");
  CHECK_ERRNO(err, 0);

  errno = 0;
  env[0] = 1;
  err = GETENV_S(env, 0, "TESTENV");
  CHECK_ERRNO(err, EINVAL);
  CHECK_VALUE(env[0], 1, "env[0]");

  errno = 0;
  env[0] = 1;
  env[1] = 2;
  env[2] = 3;
  err = GETENV_S(env, 1, "TESTENV");
  CHECK_VALUE(err, ERANGE, "Return value");
  CHECK_VALUE(errno, 0, "errno");
  CHECK_VALUE(env[0], 0, "env[0]");
  CHECK_VALUE(env[1], 2, "env[1]");
  CHECK_VALUE(env[2], 3, "env[2]");

  errno = 0;
  env[0] = 1;
  env[1] = 2;
  env[2] = 3;
  err = GETENV_S(env, 2, "TESTENV");
  CHECK_VALUE(err, ERANGE, "Return value");
  CHECK_VALUE(errno, 0, "errno");
  CHECK_VALUE(env[0], 0, "env[0]");
  CHECK_VALUE(env[1], 2, "env[1]");
  CHECK_VALUE(env[2], 3, "env[2]");

  errno = 0;
  env[0] = 1;
  env[1] = 2;
  env[2] = 3;
  err = GETENV_S(env, 3, "TESTENV");
  CHECK_ERRNO(err, 0);
  CHECK_VALUE(env[0], '1', "env[0]");
  CHECK_VALUE(env[1], '2', "env[1]");
  CHECK_VALUE(env[2], 0, "env[2]");

  printf("SUCCESS!\n\n");

#endif

  /***************************************************************************/

  return 0;
}
