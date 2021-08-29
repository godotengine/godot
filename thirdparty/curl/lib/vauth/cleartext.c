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
 * RFC4616 PLAIN authentication
 * Draft   LOGIN SASL Mechanism <draft-murchison-sasl-login-00.txt>
 *
 ***************************************************************************/

#include "curl_setup.h"

#if !defined(CURL_DISABLE_IMAP) || !defined(CURL_DISABLE_SMTP) ||       \
  !defined(CURL_DISABLE_POP3)

#include <curl/curl.h>
#include "urldata.h"

#include "vauth/vauth.h"
#include "curl_md5.h"
#include "warnless.h"
#include "strtok.h"
#include "sendf.h"
#include "curl_printf.h"

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

/*
 * Curl_auth_create_plain_message()
 *
 * This is used to generate an already encoded PLAIN message ready
 * for sending to the recipient.
 *
 * Parameters:
 *
 * authzid [in]     - The authorization identity.
 * authcid [in]     - The authentication identity.
 * passwd  [in]     - The password.
 * out     [out]    - The result storage.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_plain_message(const char *authzid,
                                        const char *authcid,
                                        const char *passwd,
                                        struct bufref *out)
{
  char *plainauth;
  size_t plainlen;
  size_t zlen;
  size_t clen;
  size_t plen;

  zlen = (authzid == NULL ? 0 : strlen(authzid));
  clen = strlen(authcid);
  plen = strlen(passwd);

  /* Compute binary message length. Check for overflows. */
  if((zlen > SIZE_T_MAX/4) || (clen > SIZE_T_MAX/4) ||
     (plen > (SIZE_T_MAX/2 - 2)))
    return CURLE_OUT_OF_MEMORY;
  plainlen = zlen + clen + plen + 2;

  plainauth = malloc(plainlen + 1);
  if(!plainauth)
    return CURLE_OUT_OF_MEMORY;

  /* Calculate the reply */
  if(zlen)
    memcpy(plainauth, authzid, zlen);
  plainauth[zlen] = '\0';
  memcpy(plainauth + zlen + 1, authcid, clen);
  plainauth[zlen + clen + 1] = '\0';
  memcpy(plainauth + zlen + clen + 2, passwd, plen);
  plainauth[plainlen] = '\0';
  Curl_bufref_set(out, plainauth, plainlen, curl_free);
  return CURLE_OK;
}

/*
 * Curl_auth_create_login_message()
 *
 * This is used to generate an already encoded LOGIN message containing the
 * user name or password ready for sending to the recipient.
 *
 * Parameters:
 *
 * valuep  [in]     - The user name or user's password.
 * out     [out]    - The result storage.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_login_message(const char *valuep, struct bufref *out)
{
  Curl_bufref_set(out, valuep, strlen(valuep), NULL);
  return CURLE_OK;
}

/*
 * Curl_auth_create_external_message()
 *
 * This is used to generate an already encoded EXTERNAL message containing
 * the user name ready for sending to the recipient.
 *
 * Parameters:
 *
 * user    [in]     - The user name.
 * out     [out]    - The result storage.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_auth_create_external_message(const char *user,
                                           struct bufref *out)
{
  /* This is the same formatting as the login message */
  return Curl_auth_create_login_message(user, out);
}

#endif /* if no users */
