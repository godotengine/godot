/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/
#include "tool_setup.h"

#define ENABLE_CURLX_PRINTF
/* use our own printf() functions */
#include "curlx.h"

#include "tool_cfgable.h"
#include "tool_cb_see.h"

#include "memdebug.h" /* keep this as LAST include */

/* OUR_MAX_SEEK_L has 'long' data type, OUR_MAX_SEEK_O has 'curl_off_t,
   both represent the same value. Maximum offset used here when we lseek
   using a 'long' data type offset */

#define OUR_MAX_SEEK_L  2147483647L - 1L
#define OUR_MAX_SEEK_O  CURL_OFF_T_C(0x7FFFFFFF) - CURL_OFF_T_C(0x1)

/*
** callback for CURLOPT_SEEKFUNCTION
**
** Notice that this is not supposed to return the resulting offset. This
** shall only return CURL_SEEKFUNC_* return codes.
*/

int tool_seek_cb(void *userdata, curl_off_t offset, int whence)
{
  struct InStruct *in = userdata;

#if(SIZEOF_CURL_OFF_T > SIZEOF_OFF_T) && !defined(USE_WIN32_LARGE_FILES)

  /* The offset check following here is only interesting if curl_off_t is
     larger than off_t and we are not using the WIN32 large file support
     macros that provide the support to do 64bit seeks correctly */

  if(offset > OUR_MAX_SEEK_O) {
    /* Some precaution code to work around problems with different data sizes
       to allow seeking >32bit even if off_t is 32bit. Should be very rare and
       is really valid on weirdo-systems. */
    curl_off_t left = offset;

    if(whence != SEEK_SET)
      /* this code path doesn't support other types */
      return CURL_SEEKFUNC_FAIL;

    if(LSEEK_ERROR == lseek(in->fd, 0, SEEK_SET))
      /* couldn't rewind to beginning */
      return CURL_SEEKFUNC_FAIL;

    while(left) {
      long step = (left > OUR_MAX_SEEK_O) ? OUR_MAX_SEEK_L : (long)left;
      if(LSEEK_ERROR == lseek(in->fd, step, SEEK_CUR))
        /* couldn't seek forwards the desired amount */
        return CURL_SEEKFUNC_FAIL;
      left -= step;
    }
    return CURL_SEEKFUNC_OK;
  }
#endif

  if(LSEEK_ERROR == lseek(in->fd, offset, whence))
    /* couldn't rewind, the reason is in errno but errno is just not portable
       enough and we don't actually care that much why we failed. We'll let
       libcurl know that it may try other means if it wants to. */
    return CURL_SEEKFUNC_CANTSEEK;

  return CURL_SEEKFUNC_OK;
}

#ifdef USE_TOOL_FTRUNCATE

#ifdef __BORLANDC__
/* 64-bit lseek-like function unavailable */
#  define _lseeki64(hnd,ofs,whence) lseek(hnd,ofs,whence)
#endif

#ifdef __POCC__
#  if(__POCC__ < 450)
/* 64-bit lseek-like function unavailable */
#    define _lseeki64(hnd,ofs,whence) _lseek(hnd,ofs,whence)
#  else
#    undef _lseeki64
#    define _lseeki64(hnd,ofs,whence) _lseek64(hnd,ofs,whence)
#  endif
#endif

#ifdef _WIN32_WCE
/* 64-bit lseek-like function unavailable */
#  undef _lseeki64
#  define _lseeki64(hnd,ofs,whence) lseek(hnd,ofs,whence)
#  undef _get_osfhandle
#  define _get_osfhandle(fd) (fd)
#endif

/*
 * Truncate a file handle at a 64-bit position 'where'.
 */

int tool_ftruncate64(int fd, curl_off_t where)
{
  intptr_t handle = _get_osfhandle(fd);

  if(_lseeki64(fd, where, SEEK_SET) < 0)
    return -1;

  if(!SetEndOfFile((HANDLE)handle))
    return -1;

  return 0;
}

#endif /* USE_TOOL_FTRUNCATE */
