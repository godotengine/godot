#ifndef HEADER_CURL_HTTP_CHUNKS_H
#define HEADER_CURL_HTTP_CHUNKS_H
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

struct connectdata;

/*
 * The longest possible hexadecimal number we support in a chunked transfer.
 * Neither RFC2616 nor the later HTTP specs define a maximum chunk size.
 * For 64 bit curl_off_t we support 16 digits. For 32 bit, 8 digits.
 */
#define CHUNK_MAXNUM_LEN (SIZEOF_CURL_OFF_T * 2)

typedef enum {
  /* await and buffer all hexadecimal digits until we get one that isn't a
     hexadecimal digit. When done, we go CHUNK_LF */
  CHUNK_HEX,

  /* wait for LF, ignore all else */
  CHUNK_LF,

  /* We eat the amount of data specified. When done, we move on to the
     POST_CR state. */
  CHUNK_DATA,

  /* POSTLF should get a CR and then a LF and nothing else, then move back to
     HEX as the CRLF combination marks the end of a chunk. A missing CR is no
     big deal. */
  CHUNK_POSTLF,

  /* Used to mark that we're out of the game.  NOTE: that there's a 'datasize'
     field in the struct that will tell how many bytes that were not passed to
     the client in the end of the last buffer! */
  CHUNK_STOP,

  /* At this point optional trailer headers can be found, unless the next line
     is CRLF */
  CHUNK_TRAILER,

  /* A trailer CR has been found - next state is CHUNK_TRAILER_POSTCR.
     Next char must be a LF */
  CHUNK_TRAILER_CR,

  /* A trailer LF must be found now, otherwise CHUNKE_BAD_CHUNK will be
     signalled If this is an empty trailer CHUNKE_STOP will be signalled.
     Otherwise the trailer will be broadcasted via Curl_client_write() and the
     next state will be CHUNK_TRAILER */
  CHUNK_TRAILER_POSTCR
} ChunkyState;

typedef enum {
  CHUNKE_STOP = -1,
  CHUNKE_OK = 0,
  CHUNKE_TOO_LONG_HEX = 1,
  CHUNKE_ILLEGAL_HEX,
  CHUNKE_BAD_CHUNK,
  CHUNKE_BAD_ENCODING,
  CHUNKE_OUT_OF_MEMORY,
  CHUNKE_PASSTHRU_ERROR, /* Curl_httpchunk_read() returns a CURLcode to use */
  CHUNKE_LAST
} CHUNKcode;

const char *Curl_chunked_strerror(CHUNKcode code);

struct Curl_chunker {
  curl_off_t datasize;
  ChunkyState state;
  unsigned char hexindex;
  char hexbuffer[ CHUNK_MAXNUM_LEN + 1]; /* +1 for null-terminator */
};

/* The following functions are defined in http_chunks.c */
void Curl_httpchunk_init(struct Curl_easy *data);
CHUNKcode Curl_httpchunk_read(struct Curl_easy *data, char *datap,
                              ssize_t length, ssize_t *wrote,
                              CURLcode *passthru);

#endif /* HEADER_CURL_HTTP_CHUNKS_H */
