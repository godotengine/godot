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

#include "curl_setup.h"

#ifdef __AMIGA__
#  include "amigaos.h"
#  if defined(HAVE_PROTO_BSDSOCKET_H) && !defined(USE_AMISSL)
#    include <amitcp/socketbasetags.h>
#  endif
#  ifdef __libnix__
#    include <stabs.h>
#  endif
#endif

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

#ifdef __AMIGA__
#if defined(HAVE_PROTO_BSDSOCKET_H) && !defined(USE_AMISSL)
struct Library *SocketBase = NULL;
extern int errno, h_errno;

#ifdef __libnix__
void __request(const char *msg);
#else
# define __request(msg)       Printf(msg "\n\a")
#endif

void Curl_amiga_cleanup()
{
  if(SocketBase) {
    CloseLibrary(SocketBase);
    SocketBase = NULL;
  }
}

bool Curl_amiga_init()
{
  if(!SocketBase)
    SocketBase = OpenLibrary("bsdsocket.library", 4);

  if(!SocketBase) {
    __request("No TCP/IP Stack running!");
    return FALSE;
  }

  if(SocketBaseTags(SBTM_SETVAL(SBTC_ERRNOPTR(sizeof(errno))), (ULONG) &errno,
                    SBTM_SETVAL(SBTC_LOGTAGPTR), (ULONG) "curl",
                    TAG_DONE)) {
    __request("SocketBaseTags ERROR");
    return FALSE;
  }

#ifndef __libnix__
  atexit(Curl_amiga_cleanup);
#endif

  return TRUE;
}

#ifdef __libnix__
ADD2EXIT(Curl_amiga_cleanup, -50);
#endif

#endif /* HAVE_PROTO_BSDSOCKET_H */

#ifdef USE_AMISSL
void Curl_amiga_X509_free(X509 *a)
{
  X509_free(a);
}

/* AmiSSL replaces many functions with macros. Curl requires pointer
 * to some of these functions. Thus, we have to encapsulate these macros.
 */

#include "warnless.h"

int (SHA256_Init)(SHA256_CTX *c)
{
  return SHA256_Init(c);
};

int (SHA256_Update)(SHA256_CTX *c, const void *data, size_t len)
{
  return SHA256_Update(c, data, curlx_uztoui(len));
};

int (SHA256_Final)(unsigned char *md, SHA256_CTX *c)
{
  return SHA256_Final(md, c);
};

void (X509_INFO_free)(X509_INFO *a)
{
  X509_INFO_free(a);
};

#endif /* USE_AMISSL */
#endif /* __AMIGA__ */

