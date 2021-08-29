#ifndef HEADER_CURL_TOOL_SDECLS_H
#define HEADER_CURL_TOOL_SDECLS_H
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

/*
 * OutStruct variables keep track of information relative to curl's
 * output writing, which may take place to a standard stream or a file.
 *
 * 'filename' member is either a pointer to a file name string or NULL
 * when dealing with a standard stream.
 *
 * 'alloc_filename' member is TRUE when string pointed by 'filename' has been
 * dynamically allocated and 'belongs' to this OutStruct, otherwise FALSE.
 *
 * 'is_cd_filename' member is TRUE when string pointed by 'filename' has been
 * set using a server-specified Content-Disposition filename, otherwise FALSE.
 *
 * 's_isreg' member is TRUE when output goes to a regular file, this also
 * implies that output is 'seekable' and 'appendable' and also that member
 * 'filename' points to file name's string. For any standard stream member
 * 's_isreg' will be FALSE.
 *
 * 'fopened' member is TRUE when output goes to a regular file and it
 * has been fopen'ed, requiring it to be closed later on. In any other
 * case this is FALSE.
 *
 * 'stream' member is a pointer to a stream controlling object as returned
 * from a 'fopen' call or a standard stream.
 *
 * 'config' member is a pointer to associated 'OperationConfig' struct.
 *
 * 'bytes' member represents amount written so far.
 *
 * 'init' member holds original file size or offset at which truncation is
 * taking place. Always zero unless appending to a non-empty regular file.
 *
 */

struct OutStruct {
  char *filename;
  bool alloc_filename;
  bool is_cd_filename;
  bool s_isreg;
  bool fopened;
  FILE *stream;
  curl_off_t bytes;
  curl_off_t init;
};


/*
 * InStruct variables keep track of information relative to curl's
 * input reading, which may take place from stdin or from some file.
 *
 * 'fd' member is either 'stdin' file descriptor number STDIN_FILENO
 * or a file descriptor as returned from an 'open' call for some file.
 *
 * 'config' member is a pointer to associated 'OperationConfig' struct.
 */

struct InStruct {
  int fd;
  struct OperationConfig *config;
};


/*
 * A linked list of these 'getout' nodes contain URL's to fetch,
 * as well as information relative to where URL contents should
 * be stored or which file should be uploaded.
 */

struct getout {
  struct getout *next;      /* next one */
  char          *url;       /* the URL we deal with */
  char          *outfile;   /* where to store the output */
  char          *infile;    /* file to upload, if GETOUT_UPLOAD is set */
  int            flags;     /* options - composed of GETOUT_* bits */
  int            num;       /* which URL number in an invocation */
};

#define GETOUT_OUTFILE    (1<<0)  /* set when outfile is deemed done */
#define GETOUT_URL        (1<<1)  /* set when URL is deemed done */
#define GETOUT_USEREMOTE  (1<<2)  /* use remote file name locally */
#define GETOUT_UPLOAD     (1<<3)  /* if set, -T has been used */
#define GETOUT_NOUPLOAD   (1<<4)  /* if set, -T "" has been used */

/*
 * 'trace' enumeration represents curl's output look'n feel possibilities.
 */

typedef enum {
  TRACE_NONE,  /* no trace/verbose output at all */
  TRACE_BIN,   /* tcpdump inspired look */
  TRACE_ASCII, /* like *BIN but without the hex output */
  TRACE_PLAIN  /* -v/--verbose type */
} trace;


/*
 * 'HttpReq' enumeration represents HTTP request types.
 */

typedef enum {
  HTTPREQ_UNSPEC,  /* first in list */
  HTTPREQ_GET,
  HTTPREQ_HEAD,
  HTTPREQ_MIMEPOST,
  HTTPREQ_SIMPLEPOST
} HttpReq;


/*
 * Complete struct declarations which have OperationConfig struct members,
 * just in case this header is directly included in some source file.
 */

#include "tool_cfgable.h"

#endif /* HEADER_CURL_TOOL_SDECLS_H */
