#ifndef HEADER_CURL_TOOL_URLGLOB_H
#define HEADER_CURL_TOOL_URLGLOB_H
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

typedef enum {
  UPTSet = 1,
  UPTCharRange,
  UPTNumRange
} URLPatternType;

struct URLPattern {
  URLPatternType type;
  int globindex; /* the number of this particular glob or -1 if not used
                    within {} or [] */
  union {
    struct {
      char **elements;
      int size;
      int ptr_s;
    } Set;
    struct {
      char min_c;
      char max_c;
      char ptr_c;
      int step;
    } CharRange;
    struct {
      unsigned long min_n;
      unsigned long max_n;
      int padlength;
      unsigned long ptr_n;
      unsigned long step;
    } NumRange;
  } content;
};

/* the total number of globs supported */
#define GLOB_PATTERN_NUM 100

struct URLGlob {
  struct URLPattern pattern[GLOB_PATTERN_NUM];
  size_t size;
  size_t urllen;
  char *glob_buffer;
  char beenhere;
  const char *error; /* error message */
  size_t pos;        /* column position of error or 0 */
};

CURLcode glob_url(struct URLGlob**, char *, unsigned long *, FILE *);
CURLcode glob_next_url(char **, struct URLGlob *);
CURLcode glob_match_url(char **, char *, struct URLGlob *);
void glob_cleanup(struct URLGlob *glob);

#endif /* HEADER_CURL_TOOL_URLGLOB_H */
