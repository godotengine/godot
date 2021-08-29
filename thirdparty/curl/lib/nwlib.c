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

#ifdef NETWARE /* Novell NetWare */

#ifdef __NOVELL_LIBC__
/* For native LibC-based NLM we need to register as a real lib. */
#include <library.h>
#include <netware.h>
#include <screen.h>
#include <nks/thread.h>
#include <nks/synch.h>

#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

struct libthreaddata {
  int     _errno;
  void    *twentybytes;
};

struct libdata {
  int         x;
  int         y;
  int         z;
  void        *tenbytes;
  NXKey_t     perthreadkey;   /* if -1, no key obtained... */
  NXMutex_t   *lock;
};

int         gLibId      = -1;
void        *gLibHandle = (void *) NULL;
rtag_t      gAllocTag   = (rtag_t) NULL;
NXMutex_t   *gLibLock   = (NXMutex_t *) NULL;

/* internal library function prototypes... */
int  DisposeLibraryData(void *);
void DisposeThreadData(void *);
int  GetOrSetUpData(int id, struct libdata **data,
                    struct libthreaddata **threaddata);


int _NonAppStart(void        *NLMHandle,
                 void        *errorScreen,
                 const char  *cmdLine,
                 const char  *loadDirPath,
                 size_t      uninitializedDataLength,
                 void        *NLMFileHandle,
                 int         (*readRoutineP)(int conn,
                                             void *fileHandle, size_t offset,
                                             size_t nbytes,
                                             size_t *bytesRead,
                                             void *buffer),
                  size_t      customDataOffset,
                  size_t      customDataSize,
                  int         messageCount,
                  const char  **messages)
{
  NX_LOCK_INFO_ALLOC(liblock, "Per-Application Data Lock", 0);

#ifndef __GNUC__
#pragma unused(cmdLine)
#pragma unused(loadDirPath)
#pragma unused(uninitializedDataLength)
#pragma unused(NLMFileHandle)
#pragma unused(readRoutineP)
#pragma unused(customDataOffset)
#pragma unused(customDataSize)
#pragma unused(messageCount)
#pragma unused(messages)
#endif

  /*
   * Here we process our command line, post errors (to the error screen),
   * perform initializations and anything else we need to do before being able
   * to accept calls into us. If we succeed, we return non-zero and the NetWare
   * Loader will leave us up, otherwise we fail to load and get dumped.
   */
  gAllocTag = AllocateResourceTag(NLMHandle,
                                  "<library-name> memory allocations",
                                  AllocSignature);

  if(!gAllocTag) {
    OutputToScreen(errorScreen, "Unable to allocate resource tag for "
                   "library memory allocations.\n");
    return -1;
  }

  gLibId = register_library(DisposeLibraryData);

  if(gLibId < -1) {
    OutputToScreen(errorScreen, "Unable to register library with kernel.\n");
    return -1;
  }

  gLibHandle = NLMHandle;

  gLibLock = NXMutexAlloc(0, 0, &liblock);

  if(!gLibLock) {
    OutputToScreen(errorScreen, "Unable to allocate library data lock.\n");
    return -1;
  }

  return 0;
}

/*
 * Here we clean up any resources we allocated. Resource tags is a big part
 * of what we created, but NetWare doesn't ask us to free those.
 */
void _NonAppStop(void)
{
  (void) unregister_library(gLibId);
  NXMutexFree(gLibLock);
}

/*
 * This function cannot be the first in the file for if the file is linked
 * first, then the check-unload function's offset will be nlmname.nlm+0
 * which is how to tell that there isn't one. When the check function is
 * first in the linked objects, it is ambiguous. For this reason, we will
 * put it inside this file after the stop function.
 *
 * Here we check to see if it's alright to ourselves to be unloaded. If not,
 * we return a non-zero value. Right now, there isn't any reason not to allow
 * it.
 */
int _NonAppCheckUnload(void)
{
    return 0;
}

int GetOrSetUpData(int id, struct libdata **appData,
                   struct libthreaddata **threadData)
{
  int                 err;
  struct libdata      *app_data;
  struct libthreaddata *thread_data;
  NXKey_t             key;
  NX_LOCK_INFO_ALLOC(liblock, "Application Data Lock", 0);

  err         = 0;
  thread_data = (struct libthreaddata_t *) NULL;

  /*
   * Attempt to get our data for the application calling us. This is where we
   * store whatever application-specific information we need to carry in
   * support of calling applications.
   */
  app_data = (struct libdata *) get_app_data(id);

  if(!app_data) {
    /*
     * This application hasn't called us before; set up application AND
     * per-thread data. Of course, just in case a thread from this same
     * application is calling us simultaneously, we better lock our application
     * data-creation mutex. We also need to recheck for data after we acquire
     * the lock because WE might be that other thread that was too late to
     * create the data and the first thread in will have created it.
     */
    NXLock(gLibLock);

    app_data = (struct libdata *) get_app_data(id);
    if(!app_data) {
      app_data = calloc(1, sizeof(struct libdata));

      if(app_data) {
        app_data->tenbytes = malloc(10);
        app_data->lock     = NXMutexAlloc(0, 0, &liblock);

        if(!app_data->tenbytes || !app_data->lock) {
          if(app_data->lock)
            NXMutexFree(app_data->lock);
          free(app_data->tenbytes);
          free(app_data);
          app_data = (libdata_t *) NULL;
          err      = ENOMEM;
        }

        if(app_data) {
          /*
           * Here we burn in the application data that we were trying to get
           * by calling get_app_data(). Next time we call the first function,
           * we'll get this data we're just now setting. We also go on here to
           * establish the per-thread data for the calling thread, something
           * we'll have to do on each application thread the first time
           * it calls us.
           */
          err = set_app_data(gLibId, app_data);

          if(err) {
            if(app_data->lock)
              NXMutexFree(app_data->lock);
            free(app_data->tenbytes);
            free(app_data);
            app_data = (libdata_t *) NULL;
            err      = ENOMEM;
          }
          else {
            /* create key for thread-specific data... */
            err = NXKeyCreate(DisposeThreadData, (void *) NULL, &key);

            if(err)                /* (no more keys left?) */
              key = -1;

            app_data->perthreadkey = key;
          }
        }
      }
    }

    NXUnlock(gLibLock);
  }

  if(app_data) {
    key = app_data->perthreadkey;

    if(key != -1 /* couldn't create a key? no thread data */
        && !(err = NXKeyGetValue(key, (void **) &thread_data))
        && !thread_data) {
      /*
       * Allocate the per-thread data for the calling thread. Regardless of
       * whether there was already application data or not, this may be the
       * first call by a new thread. The fact that we allocation 20 bytes on
       * a pointer is not very important, this just helps to demonstrate that
       * we can have arbitrarily complex per-thread data.
       */
      thread_data = malloc(sizeof(struct libthreaddata));

      if(thread_data) {
        thread_data->_errno      = 0;
        thread_data->twentybytes = malloc(20);

        if(!thread_data->twentybytes) {
          free(thread_data);
          thread_data = (struct libthreaddata *) NULL;
          err         = ENOMEM;
        }

        err = NXKeySetValue(key, thread_data);
        if(err) {
          free(thread_data->twentybytes);
          free(thread_data);
          thread_data = (struct libthreaddata *) NULL;
        }
      }
    }
  }

  if(appData)
    *appData = app_data;

  if(threadData)
    *threadData = thread_data;

  return err;
}

int DisposeLibraryData(void *data)
{
  if(data) {
    void *tenbytes = ((libdata_t *) data)->tenbytes;

    free(tenbytes);
    free(data);
  }

  return 0;
}

void DisposeThreadData(void *data)
{
  if(data) {
    void *twentybytes = ((struct libthreaddata *) data)->twentybytes;

    free(twentybytes);
    free(data);
  }
}

#else /* __NOVELL_LIBC__ */
/* For native CLib-based NLM seems we can do a bit more simple. */
#include <nwthread.h>

int main(void)
{
  /* initialize any globals here... */

  /* do this if any global initializing was done
  SynchronizeStart();
  */
  ExitThread(TSR_THREAD, 0);
  return 0;
}

#endif /* __NOVELL_LIBC__ */

#else /* NETWARE */

#ifdef __POCC__
#  pragma warn(disable:2024)  /* Disable warning #2024: Empty input file */
#endif

#endif /* NETWARE */
