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

#ifdef CURL_DOES_CONVERSIONS

#include <curl/curl.h>

#include "non-ascii.h"
#include "formdata.h"
#include "sendf.h"
#include "urldata.h"
#include "multiif.h"
#include "strerror.h"

#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

#ifdef HAVE_ICONV
#include <iconv.h>
/* set default codesets for iconv */
#ifndef CURL_ICONV_CODESET_OF_NETWORK
#define CURL_ICONV_CODESET_OF_NETWORK "ISO8859-1"
#endif
#ifndef CURL_ICONV_CODESET_FOR_UTF8
#define CURL_ICONV_CODESET_FOR_UTF8   "UTF-8"
#endif
#define ICONV_ERROR  (size_t)-1
#endif /* HAVE_ICONV */

/*
 * Curl_convert_clone() returns a malloced copy of the source string (if
 * returning CURLE_OK), with the data converted to network format.
 */
CURLcode Curl_convert_clone(struct Curl_easy *data,
                           const char *indata,
                           size_t insize,
                           char **outbuf)
{
  char *convbuf;
  CURLcode result;

  convbuf = malloc(insize);
  if(!convbuf)
    return CURLE_OUT_OF_MEMORY;

  memcpy(convbuf, indata, insize);
  result = Curl_convert_to_network(data, convbuf, insize);
  if(result) {
    free(convbuf);
    return result;
  }

  *outbuf = convbuf; /* return the converted buffer */

  return CURLE_OK;
}

/*
 * Curl_convert_to_network() is an internal function for performing ASCII
 * conversions on non-ASCII platforms. It converts the buffer _in place_.
 */
CURLcode Curl_convert_to_network(struct Curl_easy *data,
                                 char *buffer, size_t length)
{
  if(data && data->set.convtonetwork) {
    /* use translation callback */
    CURLcode result;
    Curl_set_in_callback(data, true);
    result = data->set.convtonetwork(buffer, length);
    Curl_set_in_callback(data, false);
    if(result) {
      failf(data,
            "CURLOPT_CONV_TO_NETWORK_FUNCTION callback returned %d: %s",
            (int)result, curl_easy_strerror(result));
    }

    return result;
  }
  else {
#ifdef HAVE_ICONV
    /* do the translation ourselves */
    iconv_t tmpcd = (iconv_t) -1;
    iconv_t *cd = &tmpcd;
    char *input_ptr, *output_ptr;
    size_t in_bytes, out_bytes, rc;
    char ebuffer[STRERROR_LEN];

    /* open an iconv conversion descriptor if necessary */
    if(data)
      cd = &data->outbound_cd;
    if(*cd == (iconv_t)-1) {
      *cd = iconv_open(CURL_ICONV_CODESET_OF_NETWORK,
                       CURL_ICONV_CODESET_OF_HOST);
      if(*cd == (iconv_t)-1) {
        failf(data,
              "The iconv_open(\"%s\", \"%s\") call failed with errno %i: %s",
              CURL_ICONV_CODESET_OF_NETWORK,
              CURL_ICONV_CODESET_OF_HOST,
              errno, Curl_strerror(errno, ebuffer, sizeof(ebuffer)));
        return CURLE_CONV_FAILED;
      }
    }
    /* call iconv */
    input_ptr = output_ptr = buffer;
    in_bytes = out_bytes = length;
    rc = iconv(*cd, &input_ptr, &in_bytes,
               &output_ptr, &out_bytes);
    if(!data)
      iconv_close(tmpcd);
    if((rc == ICONV_ERROR) || (in_bytes)) {
      failf(data,
            "The Curl_convert_to_network iconv call failed with errno %i: %s",
            errno, Curl_strerror(errno, ebuffer, sizeof(ebuffer)));
      return CURLE_CONV_FAILED;
    }
#else
    failf(data, "CURLOPT_CONV_TO_NETWORK_FUNCTION callback required");
    return CURLE_CONV_REQD;
#endif /* HAVE_ICONV */
  }

  return CURLE_OK;
}

/*
 * Curl_convert_from_network() is an internal function for performing ASCII
 * conversions on non-ASCII platforms. It converts the buffer _in place_.
 */
CURLcode Curl_convert_from_network(struct Curl_easy *data,
                                   char *buffer, size_t length)
{
  if(data && data->set.convfromnetwork) {
    /* use translation callback */
    CURLcode result;
    Curl_set_in_callback(data, true);
    result = data->set.convfromnetwork(buffer, length);
    Curl_set_in_callback(data, false);
    if(result) {
      failf(data,
            "CURLOPT_CONV_FROM_NETWORK_FUNCTION callback returned %d: %s",
            (int)result, curl_easy_strerror(result));
    }

    return result;
  }
  else {
#ifdef HAVE_ICONV
    /* do the translation ourselves */
    iconv_t tmpcd = (iconv_t) -1;
    iconv_t *cd = &tmpcd;
    char *input_ptr, *output_ptr;
    size_t in_bytes, out_bytes, rc;
    char ebuffer[STRERROR_LEN];

    /* open an iconv conversion descriptor if necessary */
    if(data)
      cd = &data->inbound_cd;
    if(*cd == (iconv_t)-1) {
      *cd = iconv_open(CURL_ICONV_CODESET_OF_HOST,
                       CURL_ICONV_CODESET_OF_NETWORK);
      if(*cd == (iconv_t)-1) {
        failf(data,
              "The iconv_open(\"%s\", \"%s\") call failed with errno %i: %s",
              CURL_ICONV_CODESET_OF_HOST,
              CURL_ICONV_CODESET_OF_NETWORK,
              errno, Curl_strerror(errno, ebuffer, sizeof(ebuffer)));
        return CURLE_CONV_FAILED;
      }
    }
    /* call iconv */
    input_ptr = output_ptr = buffer;
    in_bytes = out_bytes = length;
    rc = iconv(*cd, &input_ptr, &in_bytes,
               &output_ptr, &out_bytes);
    if(!data)
      iconv_close(tmpcd);
    if((rc == ICONV_ERROR) || (in_bytes)) {
      failf(data,
            "Curl_convert_from_network iconv call failed with errno %i: %s",
            errno, Curl_strerror(errno, ebuffer, sizeof(ebuffer)));
      return CURLE_CONV_FAILED;
    }
#else
    failf(data, "CURLOPT_CONV_FROM_NETWORK_FUNCTION callback required");
    return CURLE_CONV_REQD;
#endif /* HAVE_ICONV */
  }

  return CURLE_OK;
}

/*
 * Curl_convert_from_utf8() is an internal function for performing UTF-8
 * conversions on non-ASCII platforms.
 */
CURLcode Curl_convert_from_utf8(struct Curl_easy *data,
                                char *buffer, size_t length)
{
  if(data && data->set.convfromutf8) {
    /* use translation callback */
    CURLcode result;
    Curl_set_in_callback(data, true);
    result = data->set.convfromutf8(buffer, length);
    Curl_set_in_callback(data, false);
    if(result) {
      failf(data,
            "CURLOPT_CONV_FROM_UTF8_FUNCTION callback returned %d: %s",
            (int)result, curl_easy_strerror(result));
    }

    return result;
  }
  else {
#ifdef HAVE_ICONV
    /* do the translation ourselves */
    iconv_t tmpcd = (iconv_t) -1;
    iconv_t *cd = &tmpcd;
    char *input_ptr;
    char *output_ptr;
    size_t in_bytes, out_bytes, rc;
    char ebuffer[STRERROR_LEN];

    /* open an iconv conversion descriptor if necessary */
    if(data)
      cd = &data->utf8_cd;
    if(*cd == (iconv_t)-1) {
      *cd = iconv_open(CURL_ICONV_CODESET_OF_HOST,
                       CURL_ICONV_CODESET_FOR_UTF8);
      if(*cd == (iconv_t)-1) {
        failf(data,
              "The iconv_open(\"%s\", \"%s\") call failed with errno %i: %s",
              CURL_ICONV_CODESET_OF_HOST,
              CURL_ICONV_CODESET_FOR_UTF8,
              errno, Curl_strerror(errno, ebuffer, sizeof(ebuffer)));
        return CURLE_CONV_FAILED;
      }
    }
    /* call iconv */
    input_ptr = output_ptr = buffer;
    in_bytes = out_bytes = length;
    rc = iconv(*cd, &input_ptr, &in_bytes,
               &output_ptr, &out_bytes);
    if(!data)
      iconv_close(tmpcd);
    if((rc == ICONV_ERROR) || (in_bytes)) {
      failf(data,
            "The Curl_convert_from_utf8 iconv call failed with errno %i: %s",
            errno, Curl_strerror(errno, ebuffer, sizeof(ebuffer)));
      return CURLE_CONV_FAILED;
    }
    if(output_ptr < input_ptr) {
      /* null terminate the now shorter output string */
      *output_ptr = 0x00;
    }
#else
    failf(data, "CURLOPT_CONV_FROM_UTF8_FUNCTION callback required");
    return CURLE_CONV_REQD;
#endif /* HAVE_ICONV */
  }

  return CURLE_OK;
}

/*
 * Init conversion stuff for a Curl_easy
 */
void Curl_convert_init(struct Curl_easy *data)
{
#if defined(CURL_DOES_CONVERSIONS) && defined(HAVE_ICONV)
  /* conversion descriptors for iconv calls */
  data->outbound_cd = (iconv_t)-1;
  data->inbound_cd  = (iconv_t)-1;
  data->utf8_cd     = (iconv_t)-1;
#else
  (void)data;
#endif /* CURL_DOES_CONVERSIONS && HAVE_ICONV */
}

/*
 * Setup conversion stuff for a Curl_easy
 */
void Curl_convert_setup(struct Curl_easy *data)
{
  data->inbound_cd = iconv_open(CURL_ICONV_CODESET_OF_HOST,
                                CURL_ICONV_CODESET_OF_NETWORK);
  data->outbound_cd = iconv_open(CURL_ICONV_CODESET_OF_NETWORK,
                                 CURL_ICONV_CODESET_OF_HOST);
  data->utf8_cd = iconv_open(CURL_ICONV_CODESET_OF_HOST,
                             CURL_ICONV_CODESET_FOR_UTF8);
}

/*
 * Close conversion stuff for a Curl_easy
 */

void Curl_convert_close(struct Curl_easy *data)
{
#ifdef HAVE_ICONV
  /* close iconv conversion descriptors */
  if(data->inbound_cd != (iconv_t)-1) {
    iconv_close(data->inbound_cd);
  }
  if(data->outbound_cd != (iconv_t)-1) {
    iconv_close(data->outbound_cd);
  }
  if(data->utf8_cd != (iconv_t)-1) {
    iconv_close(data->utf8_cd);
  }
#else
  (void)data;
#endif /* HAVE_ICONV */
}

#endif /* CURL_DOES_CONVERSIONS */
