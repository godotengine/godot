#ifndef HEADER_CURL_SIGPIPE_H
#define HEADER_CURL_SIGPIPE_H
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

#if defined(HAVE_SIGNAL_H) && defined(HAVE_SIGACTION) &&        \
  (defined(USE_OPENSSL) || defined(USE_MBEDTLS) || defined(USE_WOLFSSL))
#include <signal.h>

struct sigpipe_ignore {
  struct sigaction old_pipe_act;
  bool no_signal;
};

#define SIGPIPE_VARIABLE(x) struct sigpipe_ignore x

/*
 * sigpipe_ignore() makes sure we ignore SIGPIPE while running libcurl
 * internals, and then sigpipe_restore() will restore the situation when we
 * return from libcurl again.
 */
static void sigpipe_ignore(struct Curl_easy *data,
                           struct sigpipe_ignore *ig)
{
  /* get a local copy of no_signal because the Curl_easy might not be
     around when we restore */
  ig->no_signal = data->set.no_signal;
  if(!data->set.no_signal) {
    struct sigaction action;
    /* first, extract the existing situation */
    memset(&ig->old_pipe_act, 0, sizeof(struct sigaction));
    sigaction(SIGPIPE, NULL, &ig->old_pipe_act);
    action = ig->old_pipe_act;
    /* ignore this signal */
    action.sa_handler = SIG_IGN;
    sigaction(SIGPIPE, &action, NULL);
  }
}

/*
 * sigpipe_restore() puts back the outside world's opinion of signal handler
 * and SIGPIPE handling. It MUST only be called after a corresponding
 * sigpipe_ignore() was used.
 */
static void sigpipe_restore(struct sigpipe_ignore *ig)
{
  if(!ig->no_signal)
    /* restore the outside state */
    sigaction(SIGPIPE, &ig->old_pipe_act, NULL);
}

#else
/* for systems without sigaction */
#define sigpipe_ignore(x,y) Curl_nop_stmt
#define sigpipe_restore(x)  Curl_nop_stmt
#define SIGPIPE_VARIABLE(x)
#endif

#endif /* HEADER_CURL_SIGPIPE_H */
