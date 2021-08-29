#ifndef HEADER_CURL_KEYLOG_H
#define HEADER_CURL_KEYLOG_H
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
#include "curl_setup.h"

/*
 * Opens the TLS key log file if requested by the user. The SSLKEYLOGFILE
 * environment variable specifies the output file.
 */
void Curl_tls_keylog_open(void);

/*
 * Closes the TLS key log file if not already.
 */
void Curl_tls_keylog_close(void);

/*
 * Returns true if the user successfully enabled the TLS key log file.
 */
bool Curl_tls_keylog_enabled(void);

/*
 * Appends a key log file entry.
 * Returns true iff the key log file is open and a valid entry was provided.
 */
bool Curl_tls_keylog_write(const char *label,
                           const unsigned char client_random[32],
                           const unsigned char *secret, size_t secretlen);

/*
 * Appends a line to the key log file, ensure it is terminated by a LF.
 * Returns true iff the key log file is open and a valid line was provided.
 */
bool Curl_tls_keylog_write_line(const char *line);

#endif /* HEADER_CURL_KEYLOG_H */
