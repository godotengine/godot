#ifndef HEADER_CURL_ENDIAN_H
#define HEADER_CURL_ENDIAN_H
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

/* Converts a 16-bit integer from little endian */
unsigned short Curl_read16_le(const unsigned char *buf);

/* Converts a 32-bit integer from little endian */
unsigned int Curl_read32_le(const unsigned char *buf);

/* Converts a 16-bit integer from big endian */
unsigned short Curl_read16_be(const unsigned char *buf);

#if (SIZEOF_CURL_OFF_T > 4)
/* Converts a 64-bit integer to little endian */
#if defined(HAVE_LONGLONG)
void Curl_write64_le(const long long value, unsigned char *buffer);
#else
void Curl_write64_le(const __int64 value, unsigned char *buffer);
#endif
#endif

#endif /* HEADER_CURL_ENDIAN_H */
