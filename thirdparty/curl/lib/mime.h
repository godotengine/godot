#ifndef HEADER_CURL_MIME_H
#define HEADER_CURL_MIME_H
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

#define MIME_RAND_BOUNDARY_CHARS        16  /* Nb. of random boundary chars. */
#define MAX_ENCODED_LINE_LENGTH         76  /* Maximum encoded line length. */
#define ENCODING_BUFFER_SIZE            256 /* Encoding temp buffers size. */

/* Part flags. */
#define MIME_USERHEADERS_OWNER  (1 << 0)
#define MIME_BODY_ONLY          (1 << 1)
#define MIME_FAST_READ          (1 << 2)

#define FILE_CONTENTTYPE_DEFAULT        "application/octet-stream"
#define MULTIPART_CONTENTTYPE_DEFAULT   "multipart/mixed"
#define DISPOSITION_DEFAULT             "attachment"

/* Part source kinds. */
enum mimekind {
  MIMEKIND_NONE = 0,            /* Part not set. */
  MIMEKIND_DATA,                /* Allocated mime data. */
  MIMEKIND_FILE,                /* Data from file. */
  MIMEKIND_CALLBACK,            /* Data from `read' callback. */
  MIMEKIND_MULTIPART,           /* Data is a mime subpart. */
  MIMEKIND_LAST
};

/* Readback state tokens. */
enum mimestate {
  MIMESTATE_BEGIN,              /* Readback has not yet started. */
  MIMESTATE_CURLHEADERS,        /* In curl-generated headers. */
  MIMESTATE_USERHEADERS,        /* In caller's supplied headers. */
  MIMESTATE_EOH,                /* End of headers. */
  MIMESTATE_BODY,               /* Placeholder. */
  MIMESTATE_BOUNDARY1,          /* In boundary prefix. */
  MIMESTATE_BOUNDARY2,          /* In boundary. */
  MIMESTATE_CONTENT,            /* In content. */
  MIMESTATE_END,                /* End of part reached. */
  MIMESTATE_LAST
};

/* Mime headers strategies. */
enum mimestrategy {
  MIMESTRATEGY_MAIL,            /* Mime mail. */
  MIMESTRATEGY_FORM,            /* HTTP post form. */
  MIMESTRATEGY_LAST
};

/* Content transfer encoder. */
struct mime_encoder {
  const char *   name;          /* Encoding name. */
  size_t         (*encodefunc)(char *buffer, size_t size, bool ateof,
                               curl_mimepart *part);  /* Encoded read. */
  curl_off_t     (*sizefunc)(curl_mimepart *part);  /* Encoded size. */
};

/* Content transfer encoder state. */
struct mime_encoder_state {
  size_t         pos;           /* Position on output line. */
  size_t         bufbeg;        /* Next data index in input buffer. */
  size_t         bufend;        /* First unused byte index in input buffer. */
  char           buf[ENCODING_BUFFER_SIZE]; /* Input buffer. */
};

/* Mime readback state. */
struct mime_state {
  enum mimestate state;       /* Current state token. */
  void *ptr;                  /* State-dependent pointer. */
  curl_off_t offset;          /* State-dependent offset. */
};

/* minimum buffer size for the boundary string */
#define MIME_BOUNDARY_LEN (24 + MIME_RAND_BOUNDARY_CHARS + 1)

/* A mime multipart. */
struct curl_mime {
  struct Curl_easy *easy;          /* The associated easy handle. */
  curl_mimepart *parent;           /* Parent part. */
  curl_mimepart *firstpart;        /* First part. */
  curl_mimepart *lastpart;         /* Last part. */
  char boundary[MIME_BOUNDARY_LEN]; /* The part boundary. */
  struct mime_state state;         /* Current readback state. */
};

/* A mime part. */
struct curl_mimepart {
  struct Curl_easy *easy;          /* The associated easy handle. */
  curl_mime *parent;               /* Parent mime structure. */
  curl_mimepart *nextpart;         /* Forward linked list. */
  enum mimekind kind;              /* The part kind. */
  unsigned int flags;              /* Flags. */
  char *data;                      /* Memory data or file name. */
  curl_read_callback readfunc;     /* Read function. */
  curl_seek_callback seekfunc;     /* Seek function. */
  curl_free_callback freefunc;     /* Argument free function. */
  void *arg;                       /* Argument to callback functions. */
  FILE *fp;                        /* File pointer. */
  struct curl_slist *curlheaders;  /* Part headers. */
  struct curl_slist *userheaders;  /* Part headers. */
  char *mimetype;                  /* Part mime type. */
  char *filename;                  /* Remote file name. */
  char *name;                      /* Data name. */
  curl_off_t datasize;             /* Expected data size. */
  struct mime_state state;         /* Current readback state. */
  const struct mime_encoder *encoder; /* Content data encoder. */
  struct mime_encoder_state encstate; /* Data encoder state. */
  size_t lastreadstatus;           /* Last read callback returned status. */
};

CURLcode Curl_mime_add_header(struct curl_slist **slp, const char *fmt, ...);

#if (!defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_MIME)) ||     \
  !defined(CURL_DISABLE_SMTP) || !defined(CURL_DISABLE_IMAP)

/* Prototypes. */
void Curl_mime_initpart(struct curl_mimepart *part, struct Curl_easy *easy);
void Curl_mime_cleanpart(struct curl_mimepart *part);
CURLcode Curl_mime_duppart(struct curl_mimepart *dst,
                           const curl_mimepart *src);
CURLcode Curl_mime_set_subparts(struct curl_mimepart *part,
                                struct curl_mime *subparts,
                                int take_ownership);
CURLcode Curl_mime_prepare_headers(struct curl_mimepart *part,
                                   const char *contenttype,
                                   const char *disposition,
                                   enum mimestrategy strategy);
curl_off_t Curl_mime_size(struct curl_mimepart *part);
size_t Curl_mime_read(char *buffer, size_t size, size_t nitems,
                      void *instream);
CURLcode Curl_mime_rewind(struct curl_mimepart *part);
const char *Curl_mime_contenttype(const char *filename);
void Curl_mime_unpause(struct curl_mimepart *part);

#else
/* if disabled */
#define Curl_mime_initpart(x,y)
#define Curl_mime_cleanpart(x)
#define Curl_mime_duppart(x,y) CURLE_OK /* Nothing to duplicate. Succeed */
#define Curl_mime_set_subparts(a,b,c) CURLE_NOT_BUILT_IN
#define Curl_mime_prepare_headers(a,b,c,d) CURLE_NOT_BUILT_IN
#define Curl_mime_size(x) (curl_off_t) -1
#define Curl_mime_read NULL
#define Curl_mime_rewind(x) ((void)x, CURLE_NOT_BUILT_IN)
#define Curl_mime_unpause(x)
#endif


#endif /* HEADER_CURL_MIME_H */
