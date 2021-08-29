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

#ifdef CURL_DOES_CONVERSIONS

#ifdef HAVE_ICONV
#  include <iconv.h>
#endif

#include "tool_convert.h"

#include "memdebug.h" /* keep this as LAST include */

#ifdef HAVE_ICONV

/* curl tool iconv conversion descriptors */
static iconv_t inbound_cd  = (iconv_t)-1;
static iconv_t outbound_cd = (iconv_t)-1;

/* set default codesets for iconv */
#ifndef CURL_ICONV_CODESET_OF_NETWORK
#  define CURL_ICONV_CODESET_OF_NETWORK "ISO8859-1"
#endif

/*
 * convert_to_network() is a curl tool function to convert
 * from the host encoding to ASCII on non-ASCII platforms.
 */
CURLcode convert_to_network(char *buffer, size_t length)
{
  /* translate from the host encoding to the network encoding */
  char *input_ptr, *output_ptr;
  size_t res, in_bytes, out_bytes;

  /* open an iconv conversion descriptor if necessary */
  if(outbound_cd == (iconv_t)-1) {
    outbound_cd = iconv_open(CURL_ICONV_CODESET_OF_NETWORK,
                             CURL_ICONV_CODESET_OF_HOST);
    if(outbound_cd == (iconv_t)-1) {
      return CURLE_CONV_FAILED;
    }
  }
  /* call iconv */
  input_ptr = output_ptr = buffer;
  in_bytes = out_bytes = length;
  res = iconv(outbound_cd, &input_ptr,  &in_bytes,
              &output_ptr, &out_bytes);
  if((res == (size_t)-1) || (in_bytes)) {
    return CURLE_CONV_FAILED;
  }

  return CURLE_OK;
}

/*
 * convert_from_network() is a curl tool function
 * for performing ASCII conversions on non-ASCII platforms.
 */
CURLcode convert_from_network(char *buffer, size_t length)
{
  /* translate from the network encoding to the host encoding */
  char *input_ptr, *output_ptr;
  size_t res, in_bytes, out_bytes;

  /* open an iconv conversion descriptor if necessary */
  if(inbound_cd == (iconv_t)-1) {
    inbound_cd = iconv_open(CURL_ICONV_CODESET_OF_HOST,
                            CURL_ICONV_CODESET_OF_NETWORK);
    if(inbound_cd == (iconv_t)-1) {
      return CURLE_CONV_FAILED;
    }
  }
  /* call iconv */
  input_ptr = output_ptr = buffer;
  in_bytes = out_bytes = length;
  res = iconv(inbound_cd, &input_ptr,  &in_bytes,
              &output_ptr, &out_bytes);
  if((res == (size_t)-1) || (in_bytes)) {
    return CURLE_CONV_FAILED;
  }

  return CURLE_OK;
}

void convert_cleanup(void)
{
  /* close iconv conversion descriptors */
  if(inbound_cd != (iconv_t)-1)
    (void)iconv_close(inbound_cd);
  if(outbound_cd != (iconv_t)-1)
    (void)iconv_close(outbound_cd);
}

#endif /* HAVE_ICONV */

char convert_char(curl_infotype infotype, char this_char)
{
/* determine how this specific character should be displayed */
  switch(infotype) {
  case CURLINFO_DATA_IN:
  case CURLINFO_DATA_OUT:
  case CURLINFO_SSL_DATA_IN:
  case CURLINFO_SSL_DATA_OUT:
    /* data, treat as ASCII */
    if(this_char < 0x20 || this_char >= 0x7f) {
      /* non-printable ASCII, use a replacement character */
      return UNPRINTABLE_CHAR;
    }
    /* printable ASCII hex value: convert to host encoding */
    (void)convert_from_network(&this_char, 1);
    /* FALLTHROUGH */
  default:
    /* treat as host encoding */
    if(ISPRINT(this_char)
       &&  (this_char != '\t')
       &&  (this_char != '\r')
       &&  (this_char != '\n')) {
      /* printable characters excluding tabs and line end characters */
      return this_char;
    }
    break;
  }
  /* non-printable, use a replacement character  */
  return UNPRINTABLE_CHAR;
}

#endif /* CURL_DOES_CONVERSIONS */
