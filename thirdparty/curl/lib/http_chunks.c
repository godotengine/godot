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

#ifndef CURL_DISABLE_HTTP

#include "urldata.h" /* it includes http_chunks.h */
#include "sendf.h"   /* for the client write stuff */
#include "dynbuf.h"
#include "content_encoding.h"
#include "http.h"
#include "non-ascii.h" /* for Curl_convert_to_network prototype */
#include "strtoofft.h"
#include "warnless.h"

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

/*
 * Chunk format (simplified):
 *
 * <HEX SIZE>[ chunk extension ] CRLF
 * <DATA> CRLF
 *
 * Highlights from RFC2616 section 3.6 say:

   The chunked encoding modifies the body of a message in order to
   transfer it as a series of chunks, each with its own size indicator,
   followed by an OPTIONAL trailer containing entity-header fields. This
   allows dynamically produced content to be transferred along with the
   information necessary for the recipient to verify that it has
   received the full message.

       Chunked-Body   = *chunk
                        last-chunk
                        trailer
                        CRLF

       chunk          = chunk-size [ chunk-extension ] CRLF
                        chunk-data CRLF
       chunk-size     = 1*HEX
       last-chunk     = 1*("0") [ chunk-extension ] CRLF

       chunk-extension= *( ";" chunk-ext-name [ "=" chunk-ext-val ] )
       chunk-ext-name = token
       chunk-ext-val  = token | quoted-string
       chunk-data     = chunk-size(OCTET)
       trailer        = *(entity-header CRLF)

   The chunk-size field is a string of hex digits indicating the size of
   the chunk. The chunked encoding is ended by any chunk whose size is
   zero, followed by the trailer, which is terminated by an empty line.

 */

#ifdef CURL_DOES_CONVERSIONS
/* Check for an ASCII hex digit.
   We avoid the use of ISXDIGIT to accommodate non-ASCII hosts. */
static bool isxdigit_ascii(char digit)
{
  return (digit >= 0x30 && digit <= 0x39) /* 0-9 */
    || (digit >= 0x41 && digit <= 0x46) /* A-F */
    || (digit >= 0x61 && digit <= 0x66); /* a-f */
}
#else
#define isxdigit_ascii(x) Curl_isxdigit(x)
#endif

void Curl_httpchunk_init(struct Curl_easy *data)
{
  struct connectdata *conn = data->conn;
  struct Curl_chunker *chunk = &conn->chunk;
  chunk->hexindex = 0;      /* start at 0 */
  chunk->state = CHUNK_HEX; /* we get hex first! */
  Curl_dyn_init(&conn->trailer, DYN_H1_TRAILER);
}

/*
 * chunk_read() returns a OK for normal operations, or a positive return code
 * for errors. STOP means this sequence of chunks is complete.  The 'wrote'
 * argument is set to tell the caller how many bytes we actually passed to the
 * client (for byte-counting and whatever).
 *
 * The states and the state-machine is further explained in the header file.
 *
 * This function always uses ASCII hex values to accommodate non-ASCII hosts.
 * For example, 0x0d and 0x0a are used instead of '\r' and '\n'.
 */
CHUNKcode Curl_httpchunk_read(struct Curl_easy *data,
                              char *datap,
                              ssize_t datalen,
                              ssize_t *wrotep,
                              CURLcode *extrap)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  struct Curl_chunker *ch = &conn->chunk;
  struct SingleRequest *k = &data->req;
  size_t piece;
  curl_off_t length = (curl_off_t)datalen;
  size_t *wrote = (size_t *)wrotep;

  *wrote = 0; /* nothing's written yet */

  /* the original data is written to the client, but we go on with the
     chunk read process, to properly calculate the content length*/
  if(data->set.http_te_skip && !k->ignorebody) {
    result = Curl_client_write(data, CLIENTWRITE_BODY, datap, datalen);
    if(result) {
      *extrap = result;
      return CHUNKE_PASSTHRU_ERROR;
    }
  }

  while(length) {
    switch(ch->state) {
    case CHUNK_HEX:
      if(isxdigit_ascii(*datap)) {
        if(ch->hexindex < CHUNK_MAXNUM_LEN) {
          ch->hexbuffer[ch->hexindex] = *datap;
          datap++;
          length--;
          ch->hexindex++;
        }
        else {
          return CHUNKE_TOO_LONG_HEX; /* longer hex than we support */
        }
      }
      else {
        char *endptr;
        if(0 == ch->hexindex)
          /* This is illegal data, we received junk where we expected
             a hexadecimal digit. */
          return CHUNKE_ILLEGAL_HEX;

        /* length and datap are unmodified */
        ch->hexbuffer[ch->hexindex] = 0;

        /* convert to host encoding before calling strtoul */
        result = Curl_convert_from_network(data, ch->hexbuffer, ch->hexindex);
        if(result) {
          /* Curl_convert_from_network calls failf if unsuccessful */
          /* Treat it as a bad hex character */
          return CHUNKE_ILLEGAL_HEX;
        }

        if(curlx_strtoofft(ch->hexbuffer, &endptr, 16, &ch->datasize))
          return CHUNKE_ILLEGAL_HEX;
        ch->state = CHUNK_LF; /* now wait for the CRLF */
      }
      break;

    case CHUNK_LF:
      /* waiting for the LF after a chunk size */
      if(*datap == 0x0a) {
        /* we're now expecting data to come, unless size was zero! */
        if(0 == ch->datasize) {
          ch->state = CHUNK_TRAILER; /* now check for trailers */
        }
        else
          ch->state = CHUNK_DATA;
      }

      datap++;
      length--;
      break;

    case CHUNK_DATA:
      /* We expect 'datasize' of data. We have 'length' right now, it can be
         more or less than 'datasize'. Get the smallest piece.
      */
      piece = curlx_sotouz((ch->datasize >= length)?length:ch->datasize);

      /* Write the data portion available */
      if(!data->set.http_te_skip && !k->ignorebody) {
        if(!data->set.http_ce_skip && k->writer_stack)
          result = Curl_unencode_write(data, k->writer_stack, datap, piece);
        else
          result = Curl_client_write(data, CLIENTWRITE_BODY, datap, piece);

        if(result) {
          *extrap = result;
          return CHUNKE_PASSTHRU_ERROR;
        }
      }

      *wrote += piece;
      ch->datasize -= piece; /* decrease amount left to expect */
      datap += piece;    /* move read pointer forward */
      length -= piece;   /* decrease space left in this round */

      if(0 == ch->datasize)
        /* end of data this round, we now expect a trailing CRLF */
        ch->state = CHUNK_POSTLF;
      break;

    case CHUNK_POSTLF:
      if(*datap == 0x0a) {
        /* The last one before we go back to hex state and start all over. */
        Curl_httpchunk_init(data); /* sets state back to CHUNK_HEX */
      }
      else if(*datap != 0x0d)
        return CHUNKE_BAD_CHUNK;
      datap++;
      length--;
      break;

    case CHUNK_TRAILER:
      if((*datap == 0x0d) || (*datap == 0x0a)) {
        char *tr = Curl_dyn_ptr(&conn->trailer);
        /* this is the end of a trailer, but if the trailer was zero bytes
           there was no trailer and we move on */

        if(tr) {
          size_t trlen;
          result = Curl_dyn_add(&conn->trailer, (char *)"\x0d\x0a");
          if(result)
            return CHUNKE_OUT_OF_MEMORY;

          tr = Curl_dyn_ptr(&conn->trailer);
          trlen = Curl_dyn_len(&conn->trailer);
          /* Convert to host encoding before calling Curl_client_write */
          result = Curl_convert_from_network(data, tr, trlen);
          if(result)
            /* Curl_convert_from_network calls failf if unsuccessful */
            /* Treat it as a bad chunk */
            return CHUNKE_BAD_CHUNK;

          if(!data->set.http_te_skip) {
            result = Curl_client_write(data, CLIENTWRITE_HEADER, tr, trlen);
            if(result) {
              *extrap = result;
              return CHUNKE_PASSTHRU_ERROR;
            }
          }
          Curl_dyn_reset(&conn->trailer);
          ch->state = CHUNK_TRAILER_CR;
          if(*datap == 0x0a)
            /* already on the LF */
            break;
        }
        else {
          /* no trailer, we're on the final CRLF pair */
          ch->state = CHUNK_TRAILER_POSTCR;
          break; /* don't advance the pointer */
        }
      }
      else {
        result = Curl_dyn_addn(&conn->trailer, datap, 1);
        if(result)
          return CHUNKE_OUT_OF_MEMORY;
      }
      datap++;
      length--;
      break;

    case CHUNK_TRAILER_CR:
      if(*datap == 0x0a) {
        ch->state = CHUNK_TRAILER_POSTCR;
        datap++;
        length--;
      }
      else
        return CHUNKE_BAD_CHUNK;
      break;

    case CHUNK_TRAILER_POSTCR:
      /* We enter this state when a CR should arrive so we expect to
         have to first pass a CR before we wait for LF */
      if((*datap != 0x0d) && (*datap != 0x0a)) {
        /* not a CR then it must be another header in the trailer */
        ch->state = CHUNK_TRAILER;
        break;
      }
      if(*datap == 0x0d) {
        /* skip if CR */
        datap++;
        length--;
      }
      /* now wait for the final LF */
      ch->state = CHUNK_STOP;
      break;

    case CHUNK_STOP:
      if(*datap == 0x0a) {
        length--;

        /* Record the length of any data left in the end of the buffer
           even if there's no more chunks to read */
        ch->datasize = curlx_sotouz(length);

        return CHUNKE_STOP; /* return stop */
      }
      else
        return CHUNKE_BAD_CHUNK;
    }
  }
  return CHUNKE_OK;
}

const char *Curl_chunked_strerror(CHUNKcode code)
{
  switch(code) {
  default:
    return "OK";
  case CHUNKE_TOO_LONG_HEX:
    return "Too long hexadecimal number";
  case CHUNKE_ILLEGAL_HEX:
    return "Illegal or missing hexadecimal sequence";
  case CHUNKE_BAD_CHUNK:
    return "Malformed encoding found";
  case CHUNKE_PASSTHRU_ERROR:
    DEBUGASSERT(0); /* never used */
    return "";
  case CHUNKE_BAD_ENCODING:
    return "Bad content-encoding found";
  case CHUNKE_OUT_OF_MEMORY:
    return "Out of memory";
  }
}

#endif /* CURL_DISABLE_HTTP */
