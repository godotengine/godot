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

#ifdef CURLDEBUG

#include <curl/curl.h>

#include "urldata.h"

#define MEMDEBUG_NODEFINES /* don't redefine the standard functions */

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

struct memdebug {
  size_t size;
  union {
    curl_off_t o;
    double d;
    void *p;
  } mem[1];
  /* I'm hoping this is the thing with the strictest alignment
   * requirements.  That also means we waste some space :-( */
};

/*
 * Note that these debug functions are very simple and they are meant to
 * remain so. For advanced analysis, record a log file and write perl scripts
 * to analyze them!
 *
 * Don't use these with multithreaded test programs!
 */

FILE *curl_dbg_logfile = NULL;
static bool registered_cleanup = FALSE; /* atexit registered cleanup */
static bool memlimit = FALSE; /* enable memory limit */
static long memsize = 0;  /* set number of mallocs allowed */

/* LeakSantizier (LSAN) calls _exit() instead of exit() when a leak is detected
   on exit so the logfile must be closed explicitly or data could be lost.
   Though _exit() does not call atexit handlers such as this, LSAN's call to
   _exit() comes after the atexit handlers are called. curl/curl#6620 */
static void curl_dbg_cleanup(void)
{
  if(curl_dbg_logfile &&
     curl_dbg_logfile != stderr &&
     curl_dbg_logfile != stdout) {
    fclose(curl_dbg_logfile);
  }
  curl_dbg_logfile = NULL;
}

/* this sets the log file name */
void curl_dbg_memdebug(const char *logname)
{
  if(!curl_dbg_logfile) {
    if(logname && *logname)
      curl_dbg_logfile = fopen(logname, FOPEN_WRITETEXT);
    else
      curl_dbg_logfile = stderr;
#ifdef MEMDEBUG_LOG_SYNC
    /* Flush the log file after every line so the log isn't lost in a crash */
    if(curl_dbg_logfile)
      setbuf(curl_dbg_logfile, (char *)NULL);
#endif
  }
  if(!registered_cleanup)
    registered_cleanup = !atexit(curl_dbg_cleanup);
}

/* This function sets the number of malloc() calls that should return
   successfully! */
void curl_dbg_memlimit(long limit)
{
  if(!memlimit) {
    memlimit = TRUE;
    memsize = limit;
  }
}

/* returns TRUE if this isn't allowed! */
static bool countcheck(const char *func, int line, const char *source)
{
  /* if source is NULL, then the call is made internally and this check
     should not be made */
  if(memlimit && source) {
    if(!memsize) {
      /* log to file */
      curl_dbg_log("LIMIT %s:%d %s reached memlimit\n",
                   source, line, func);
      /* log to stderr also */
      fprintf(stderr, "LIMIT %s:%d %s reached memlimit\n",
              source, line, func);
      fflush(curl_dbg_logfile); /* because it might crash now */
      errno = ENOMEM;
      return TRUE; /* RETURN ERROR! */
    }
    else
      memsize--; /* countdown */


  }

  return FALSE; /* allow this */
}

void *curl_dbg_malloc(size_t wantedsize, int line, const char *source)
{
  struct memdebug *mem;
  size_t size;

  DEBUGASSERT(wantedsize != 0);

  if(countcheck("malloc", line, source))
    return NULL;

  /* alloc at least 64 bytes */
  size = sizeof(struct memdebug) + wantedsize;

  mem = (Curl_cmalloc)(size);
  if(mem) {
    mem->size = wantedsize;
  }

  if(source)
    curl_dbg_log("MEM %s:%d malloc(%zu) = %p\n",
                 source, line, wantedsize,
                 mem ? (void *)mem->mem : (void *)0);

  return (mem ? mem->mem : NULL);
}

void *curl_dbg_calloc(size_t wanted_elements, size_t wanted_size,
                      int line, const char *source)
{
  struct memdebug *mem;
  size_t size, user_size;

  DEBUGASSERT(wanted_elements != 0);
  DEBUGASSERT(wanted_size != 0);

  if(countcheck("calloc", line, source))
    return NULL;

  /* alloc at least 64 bytes */
  user_size = wanted_size * wanted_elements;
  size = sizeof(struct memdebug) + user_size;

  mem = (Curl_ccalloc)(1, size);
  if(mem)
    mem->size = user_size;

  if(source)
    curl_dbg_log("MEM %s:%d calloc(%zu,%zu) = %p\n",
                 source, line, wanted_elements, wanted_size,
                 mem ? (void *)mem->mem : (void *)0);

  return (mem ? mem->mem : NULL);
}

char *curl_dbg_strdup(const char *str, int line, const char *source)
{
  char *mem;
  size_t len;

  DEBUGASSERT(str != NULL);

  if(countcheck("strdup", line, source))
    return NULL;

  len = strlen(str) + 1;

  mem = curl_dbg_malloc(len, 0, NULL); /* NULL prevents logging */
  if(mem)
    memcpy(mem, str, len);

  if(source)
    curl_dbg_log("MEM %s:%d strdup(%p) (%zu) = %p\n",
                 source, line, (const void *)str, len, (const void *)mem);

  return mem;
}

#if defined(WIN32) && defined(UNICODE)
wchar_t *curl_dbg_wcsdup(const wchar_t *str, int line, const char *source)
{
  wchar_t *mem;
  size_t wsiz, bsiz;

  DEBUGASSERT(str != NULL);

  if(countcheck("wcsdup", line, source))
    return NULL;

  wsiz = wcslen(str) + 1;
  bsiz = wsiz * sizeof(wchar_t);

  mem = curl_dbg_malloc(bsiz, 0, NULL); /* NULL prevents logging */
  if(mem)
    memcpy(mem, str, bsiz);

  if(source)
    curl_dbg_log("MEM %s:%d wcsdup(%p) (%zu) = %p\n",
                source, line, (void *)str, bsiz, (void *)mem);

  return mem;
}
#endif

/* We provide a realloc() that accepts a NULL as pointer, which then
   performs a malloc(). In order to work with ares. */
void *curl_dbg_realloc(void *ptr, size_t wantedsize,
                      int line, const char *source)
{
  struct memdebug *mem = NULL;

  size_t size = sizeof(struct memdebug) + wantedsize;

  DEBUGASSERT(wantedsize != 0);

  if(countcheck("realloc", line, source))
    return NULL;

#ifdef __INTEL_COMPILER
#  pragma warning(push)
#  pragma warning(disable:1684)
   /* 1684: conversion from pointer to same-sized integral type */
#endif

  if(ptr)
    mem = (void *)((char *)ptr - offsetof(struct memdebug, mem));

#ifdef __INTEL_COMPILER
#  pragma warning(pop)
#endif

  mem = (Curl_crealloc)(mem, size);
  if(source)
    curl_dbg_log("MEM %s:%d realloc(%p, %zu) = %p\n",
                source, line, (void *)ptr, wantedsize,
                mem ? (void *)mem->mem : (void *)0);

  if(mem) {
    mem->size = wantedsize;
    return mem->mem;
  }

  return NULL;
}

void curl_dbg_free(void *ptr, int line, const char *source)
{
  if(ptr) {
    struct memdebug *mem;

#ifdef __INTEL_COMPILER
#  pragma warning(push)
#  pragma warning(disable:1684)
   /* 1684: conversion from pointer to same-sized integral type */
#endif

    mem = (void *)((char *)ptr - offsetof(struct memdebug, mem));

#ifdef __INTEL_COMPILER
#  pragma warning(pop)
#endif

    /* free for real */
    (Curl_cfree)(mem);
  }

  if(source && ptr)
    curl_dbg_log("MEM %s:%d free(%p)\n", source, line, (void *)ptr);
}

curl_socket_t curl_dbg_socket(int domain, int type, int protocol,
                             int line, const char *source)
{
  const char *fmt = (sizeof(curl_socket_t) == sizeof(int)) ?
    "FD %s:%d socket() = %d\n" :
    (sizeof(curl_socket_t) == sizeof(long)) ?
    "FD %s:%d socket() = %ld\n" :
    "FD %s:%d socket() = %zd\n";

  curl_socket_t sockfd;

  if(countcheck("socket", line, source))
    return CURL_SOCKET_BAD;

  sockfd = socket(domain, type, protocol);

  if(source && (sockfd != CURL_SOCKET_BAD))
    curl_dbg_log(fmt, source, line, sockfd);

  return sockfd;
}

SEND_TYPE_RETV curl_dbg_send(SEND_TYPE_ARG1 sockfd,
                            SEND_QUAL_ARG2 SEND_TYPE_ARG2 buf,
                            SEND_TYPE_ARG3 len, SEND_TYPE_ARG4 flags, int line,
                            const char *source)
{
  SEND_TYPE_RETV rc;
  if(countcheck("send", line, source))
    return -1;
  rc = send(sockfd, buf, len, flags);
  if(source)
    curl_dbg_log("SEND %s:%d send(%lu) = %ld\n",
                source, line, (unsigned long)len, (long)rc);
  return rc;
}

RECV_TYPE_RETV curl_dbg_recv(RECV_TYPE_ARG1 sockfd, RECV_TYPE_ARG2 buf,
                            RECV_TYPE_ARG3 len, RECV_TYPE_ARG4 flags, int line,
                            const char *source)
{
  RECV_TYPE_RETV rc;
  if(countcheck("recv", line, source))
    return -1;
  rc = recv(sockfd, buf, len, flags);
  if(source)
    curl_dbg_log("RECV %s:%d recv(%lu) = %ld\n",
                source, line, (unsigned long)len, (long)rc);
  return rc;
}

#ifdef HAVE_SOCKETPAIR
int curl_dbg_socketpair(int domain, int type, int protocol,
                       curl_socket_t socket_vector[2],
                       int line, const char *source)
{
  const char *fmt = (sizeof(curl_socket_t) == sizeof(int)) ?
    "FD %s:%d socketpair() = %d %d\n" :
    (sizeof(curl_socket_t) == sizeof(long)) ?
    "FD %s:%d socketpair() = %ld %ld\n" :
    "FD %s:%d socketpair() = %zd %zd\n";

  int res = socketpair(domain, type, protocol, socket_vector);

  if(source && (0 == res))
    curl_dbg_log(fmt, source, line, socket_vector[0], socket_vector[1]);

  return res;
}
#endif

curl_socket_t curl_dbg_accept(curl_socket_t s, void *saddr, void *saddrlen,
                             int line, const char *source)
{
  const char *fmt = (sizeof(curl_socket_t) == sizeof(int)) ?
    "FD %s:%d accept() = %d\n" :
    (sizeof(curl_socket_t) == sizeof(long)) ?
    "FD %s:%d accept() = %ld\n" :
    "FD %s:%d accept() = %zd\n";

  struct sockaddr *addr = (struct sockaddr *)saddr;
  curl_socklen_t *addrlen = (curl_socklen_t *)saddrlen;

  curl_socket_t sockfd = accept(s, addr, addrlen);

  if(source && (sockfd != CURL_SOCKET_BAD))
    curl_dbg_log(fmt, source, line, sockfd);

  return sockfd;
}

/* separate function to allow libcurl to mark a "faked" close */
void curl_dbg_mark_sclose(curl_socket_t sockfd, int line, const char *source)
{
  const char *fmt = (sizeof(curl_socket_t) == sizeof(int)) ?
    "FD %s:%d sclose(%d)\n":
    (sizeof(curl_socket_t) == sizeof(long)) ?
    "FD %s:%d sclose(%ld)\n":
    "FD %s:%d sclose(%zd)\n";

  if(source)
    curl_dbg_log(fmt, source, line, sockfd);
}

/* this is our own defined way to close sockets on *ALL* platforms */
int curl_dbg_sclose(curl_socket_t sockfd, int line, const char *source)
{
  int res = sclose(sockfd);
  curl_dbg_mark_sclose(sockfd, line, source);
  return res;
}

FILE *curl_dbg_fopen(const char *file, const char *mode,
                    int line, const char *source)
{
  FILE *res = fopen(file, mode);

  if(source)
    curl_dbg_log("FILE %s:%d fopen(\"%s\",\"%s\") = %p\n",
                source, line, file, mode, (void *)res);

  return res;
}

FILE *curl_dbg_fdopen(int filedes, const char *mode,
                      int line, const char *source)
{
  FILE *res = fdopen(filedes, mode);
  if(source)
    curl_dbg_log("FILE %s:%d fdopen(\"%d\",\"%s\") = %p\n",
                 source, line, filedes, mode, (void *)res);
  return res;
}

int curl_dbg_fclose(FILE *file, int line, const char *source)
{
  int res;

  DEBUGASSERT(file != NULL);

  if(source)
    curl_dbg_log("FILE %s:%d fclose(%p)\n",
                 source, line, (void *)file);

  res = fclose(file);

  return res;
}

#define LOGLINE_BUFSIZE  1024

/* this does the writing to the memory tracking log file */
void curl_dbg_log(const char *format, ...)
{
  char *buf;
  int nchars;
  va_list ap;

  if(!curl_dbg_logfile)
    return;

  buf = (Curl_cmalloc)(LOGLINE_BUFSIZE);
  if(!buf)
    return;

  va_start(ap, format);
  nchars = mvsnprintf(buf, LOGLINE_BUFSIZE, format, ap);
  va_end(ap);

  if(nchars > LOGLINE_BUFSIZE - 1)
    nchars = LOGLINE_BUFSIZE - 1;

  if(nchars > 0)
    fwrite(buf, 1, (size_t)nchars, curl_dbg_logfile);

  (Curl_cfree)(buf);
}

#endif /* CURLDEBUG */
