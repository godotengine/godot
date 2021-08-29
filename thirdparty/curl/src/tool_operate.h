#ifndef HEADER_CURL_TOOL_OPERATE_H
#define HEADER_CURL_TOOL_OPERATE_H
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
#include "tool_cb_hdr.h"
#include "tool_cb_prg.h"
#include "tool_sdecls.h"

struct per_transfer {
  /* double linked */
  struct per_transfer *next;
  struct per_transfer *prev;
  struct OperationConfig *config; /* for this transfer */
  CURL *curl;
  long retry_numretries;
  long retry_sleep_default;
  long retry_sleep;
  struct timeval retrystart;
  char *this_url;
  unsigned int urlnum; /* the index of the given URL */
  char *outfile;
  bool infdopen; /* TRUE if infd needs closing */
  int infd;
  bool noprogress;
  struct ProgressData progressbar;
  struct OutStruct outs;
  struct OutStruct heads;
  struct OutStruct etag_save;
  struct InStruct input;
  struct HdrCbData hdrcbdata;
  long num_headers;
  bool was_last_header_empty;
  char errorbuffer[CURL_ERROR_SIZE];

  bool added; /* set TRUE when added to the multi handle */
  time_t startat; /* when doing parallel transfers, this is a retry transfer
                     that has been set to sleep until this time before it
                     should get started (again) */
  bool abort; /* when doing parallel transfers and this is TRUE then a critical
                 error (eg --fail-early) has occurred in another transfer and
                 this transfer will be aborted in the progress callback */

  /* for parallel progress bar */
  curl_off_t dltotal;
  curl_off_t dlnow;
  curl_off_t ultotal;
  curl_off_t ulnow;
  bool dltotal_added; /* if the total has been added from this */
  bool ultotal_added;

  /* NULL or malloced */
  char *separator_err;
  char *separator;
  char *uploadfile;
};

CURLcode operate(struct GlobalConfig *config, int argc, argv_item_t argv[]);

extern struct per_transfer *transfers; /* first node */

#endif /* HEADER_CURL_TOOL_OPERATE_H */
