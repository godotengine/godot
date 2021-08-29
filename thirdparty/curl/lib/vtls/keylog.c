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

#include "keylog.h"

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

#define KEYLOG_LABEL_MAXLEN (sizeof("CLIENT_HANDSHAKE_TRAFFIC_SECRET") - 1)

#define CLIENT_RANDOM_SIZE  32

/*
 * The master secret in TLS 1.2 and before is always 48 bytes. In TLS 1.3, the
 * secret size depends on the cipher suite's hash function which is 32 bytes
 * for SHA-256 and 48 bytes for SHA-384.
 */
#define SECRET_MAXLEN       48


/* The fp for the open SSLKEYLOGFILE, or NULL if not open */
static FILE *keylog_file_fp;

void
Curl_tls_keylog_open(void)
{
  char *keylog_file_name;

  if(!keylog_file_fp) {
    keylog_file_name = curl_getenv("SSLKEYLOGFILE");
    if(keylog_file_name) {
      keylog_file_fp = fopen(keylog_file_name, FOPEN_APPENDTEXT);
      if(keylog_file_fp) {
#ifdef WIN32
        if(setvbuf(keylog_file_fp, NULL, _IONBF, 0))
#else
        if(setvbuf(keylog_file_fp, NULL, _IOLBF, 4096))
#endif
        {
          fclose(keylog_file_fp);
          keylog_file_fp = NULL;
        }
      }
      Curl_safefree(keylog_file_name);
    }
  }
}

void
Curl_tls_keylog_close(void)
{
  if(keylog_file_fp) {
    fclose(keylog_file_fp);
    keylog_file_fp = NULL;
  }
}

bool
Curl_tls_keylog_enabled(void)
{
  return keylog_file_fp != NULL;
}

bool
Curl_tls_keylog_write_line(const char *line)
{
  /* The current maximum valid keylog line length LF and NUL is 195. */
  size_t linelen;
  char buf[256];

  if(!keylog_file_fp || !line) {
    return false;
  }

  linelen = strlen(line);
  if(linelen == 0 || linelen > sizeof(buf) - 2) {
    /* Empty line or too big to fit in a LF and NUL. */
    return false;
  }

  memcpy(buf, line, linelen);
  if(line[linelen - 1] != '\n') {
    buf[linelen++] = '\n';
  }
  buf[linelen] = '\0';

  /* Using fputs here instead of fprintf since libcurl's fprintf replacement
     may not be thread-safe. */
  fputs(buf, keylog_file_fp);
  return true;
}

bool
Curl_tls_keylog_write(const char *label,
                      const unsigned char client_random[CLIENT_RANDOM_SIZE],
                      const unsigned char *secret, size_t secretlen)
{
  const char *hex = "0123456789ABCDEF";
  size_t pos, i;
  char line[KEYLOG_LABEL_MAXLEN + 1 + 2 * CLIENT_RANDOM_SIZE + 1 +
            2 * SECRET_MAXLEN + 1 + 1];

  if(!keylog_file_fp) {
    return false;
  }

  pos = strlen(label);
  if(pos > KEYLOG_LABEL_MAXLEN || !secretlen || secretlen > SECRET_MAXLEN) {
    /* Should never happen - sanity check anyway. */
    return false;
  }

  memcpy(line, label, pos);
  line[pos++] = ' ';

  /* Client Random */
  for(i = 0; i < CLIENT_RANDOM_SIZE; i++) {
    line[pos++] = hex[client_random[i] >> 4];
    line[pos++] = hex[client_random[i] & 0xF];
  }
  line[pos++] = ' ';

  /* Secret */
  for(i = 0; i < secretlen; i++) {
    line[pos++] = hex[secret[i] >> 4];
    line[pos++] = hex[secret[i] & 0xF];
  }
  line[pos++] = '\n';
  line[pos] = '\0';

  /* Using fputs here instead of fprintf since libcurl's fprintf replacement
     may not be thread-safe. */
  fputs(line, keylog_file_fp);
  return true;
}
