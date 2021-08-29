/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2020, Daniel Stenberg, <daniel@haxx.se>, et al.
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

#include <curl/curl.h>

#if defined(USE_THREADS_POSIX)
#  ifdef HAVE_PTHREAD_H
#    include <pthread.h>
#  endif
#elif defined(USE_THREADS_WIN32)
#  ifdef HAVE_PROCESS_H
#    include <process.h>
#  endif
#endif

#include "curl_threads.h"
#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

#if defined(USE_THREADS_POSIX)

struct Curl_actual_call {
  unsigned int (*func)(void *);
  void *arg;
};

static void *curl_thread_create_thunk(void *arg)
{
  struct Curl_actual_call *ac = arg;
  unsigned int (*func)(void *) = ac->func;
  void *real_arg = ac->arg;

  free(ac);

  (*func)(real_arg);

  return 0;
}

curl_thread_t Curl_thread_create(unsigned int (*func) (void *), void *arg)
{
  curl_thread_t t = malloc(sizeof(pthread_t));
  struct Curl_actual_call *ac = malloc(sizeof(struct Curl_actual_call));
  if(!(ac && t))
    goto err;

  ac->func = func;
  ac->arg = arg;

  if(pthread_create(t, NULL, curl_thread_create_thunk, ac) != 0)
    goto err;

  return t;

err:
  free(t);
  free(ac);
  return curl_thread_t_null;
}

void Curl_thread_destroy(curl_thread_t hnd)
{
  if(hnd != curl_thread_t_null) {
    pthread_detach(*hnd);
    free(hnd);
  }
}

int Curl_thread_join(curl_thread_t *hnd)
{
  int ret = (pthread_join(**hnd, NULL) == 0);

  free(*hnd);
  *hnd = curl_thread_t_null;

  return ret;
}

#elif defined(USE_THREADS_WIN32)

/* !checksrc! disable SPACEBEFOREPAREN 1 */
curl_thread_t Curl_thread_create(unsigned int (CURL_STDCALL *func) (void *),
                                 void *arg)
{
#ifdef _WIN32_WCE
  typedef HANDLE curl_win_thread_handle_t;
#elif defined(__MINGW32__) && !defined(__MINGW64_VERSION_MAJOR)
  typedef unsigned long curl_win_thread_handle_t;
#else
  typedef uintptr_t curl_win_thread_handle_t;
#endif
  curl_thread_t t;
  curl_win_thread_handle_t thread_handle;
#ifdef _WIN32_WCE
  thread_handle = CreateThread(NULL, 0, func, arg, 0, NULL);
#else
  thread_handle = _beginthreadex(NULL, 0, func, arg, 0, NULL);
#endif
  t = (curl_thread_t)thread_handle;
  if((t == 0) || (t == LongToHandle(-1L))) {
#ifdef _WIN32_WCE
    DWORD gle = GetLastError();
    errno = ((gle == ERROR_ACCESS_DENIED ||
              gle == ERROR_NOT_ENOUGH_MEMORY) ?
             EACCES : EINVAL);
#endif
    return curl_thread_t_null;
  }
  return t;
}

void Curl_thread_destroy(curl_thread_t hnd)
{
  CloseHandle(hnd);
}

int Curl_thread_join(curl_thread_t *hnd)
{
#if !defined(_WIN32_WINNT) || !defined(_WIN32_WINNT_VISTA) || \
    (_WIN32_WINNT < _WIN32_WINNT_VISTA)
  int ret = (WaitForSingleObject(*hnd, INFINITE) == WAIT_OBJECT_0);
#else
  int ret = (WaitForSingleObjectEx(*hnd, INFINITE, FALSE) == WAIT_OBJECT_0);
#endif

  Curl_thread_destroy(*hnd);

  *hnd = curl_thread_t_null;

  return ret;
}

#endif /* USE_THREADS_* */
