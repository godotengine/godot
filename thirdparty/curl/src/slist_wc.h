#ifndef HEADER_CURL_SLIST_WC_H
#define HEADER_CURL_SLIST_WC_H
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

#include "tool_setup.h"
#ifndef CURL_DISABLE_LIBCURL_OPTION

/* linked-list structure with last node cache for easysrc */
struct slist_wc {
  struct curl_slist *first;
  struct curl_slist *last;
};

/*
 * NAME curl_slist_wc_append()
 *
 * DESCRIPTION
 *
 * Appends a string to a linked list. If no list exists, it will be created
 * first. Returns the new list, after appending.
 */
struct slist_wc *slist_wc_append(struct slist_wc *, const char *);

/*
 * NAME curl_slist_free_all()
 *
 * DESCRIPTION
 *
 * free a previously built curl_slist_wc.
 */
void slist_wc_free_all(struct slist_wc *);

#endif /* CURL_DISABLE_LIBCURL_OPTION */

#endif /* HEADER_CURL_SLIST_WC_H */
