#ifndef HEADER_CURL_BUFREF_H
#define HEADER_CURL_BUFREF_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
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

/*
 * Generic buffer reference.
 */
struct bufref {
  void (*dtor)(void *);         /* Associated destructor. */
  const unsigned char *ptr;     /* Referenced data buffer. */
  size_t len;                   /* The data size in bytes. */
#ifdef DEBUGBUILD
  int signature;                /* Detect API use mistakes. */
#endif
};


void Curl_bufref_init(struct bufref *br);
void Curl_bufref_set(struct bufref *br, const void *ptr, size_t len,
                     void (*dtor)(void *));
const unsigned char *Curl_bufref_ptr(const struct bufref *br);
size_t Curl_bufref_len(const struct bufref *br);
CURLcode Curl_bufref_memdup(struct bufref *br, const void *ptr, size_t len);
void Curl_bufref_free(struct bufref *br);

#endif
