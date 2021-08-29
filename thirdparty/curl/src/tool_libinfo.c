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

#include "strcase.h"

#define ENABLE_CURLX_PRINTF
/* use our own printf() functions */
#include "curlx.h"

#include "tool_libinfo.h"

#include "memdebug.h" /* keep this as LAST include */

/* global variable definitions, for libcurl run-time info */

curl_version_info_data *curlinfo = NULL;
long built_in_protos = 0;

static struct proto_name_pattern {
  const char *proto_name;
  long        proto_pattern;
} const possibly_built_in[] = {
  { "dict",   CURLPROTO_DICT   },
  { "file",   CURLPROTO_FILE   },
  { "ftp",    CURLPROTO_FTP    },
  { "ftps",   CURLPROTO_FTPS   },
  { "gopher", CURLPROTO_GOPHER },
  { "gophers",CURLPROTO_GOPHERS},
  { "http",   CURLPROTO_HTTP   },
  { "https",  CURLPROTO_HTTPS  },
  { "imap",   CURLPROTO_IMAP   },
  { "imaps",  CURLPROTO_IMAPS  },
  { "ldap",   CURLPROTO_LDAP   },
  { "ldaps",  CURLPROTO_LDAPS  },
  { "mqtt",   CURLPROTO_MQTT   },
  { "pop3",   CURLPROTO_POP3   },
  { "pop3s",  CURLPROTO_POP3S  },
  { "rtmp",   CURLPROTO_RTMP   },
  { "rtmps",  CURLPROTO_RTMPS  },
  { "rtsp",   CURLPROTO_RTSP   },
  { "scp",    CURLPROTO_SCP    },
  { "sftp",   CURLPROTO_SFTP   },
  { "smb",    CURLPROTO_SMB    },
  { "smbs",   CURLPROTO_SMBS   },
  { "smtp",   CURLPROTO_SMTP   },
  { "smtps",  CURLPROTO_SMTPS  },
  { "telnet", CURLPROTO_TELNET },
  { "tftp",   CURLPROTO_TFTP   },
  {  NULL,    0                }
};

/*
 * libcurl_info_init: retrieves run-time information about libcurl,
 * setting a global pointer 'curlinfo' to libcurl's run-time info
 * struct, and a global bit pattern 'built_in_protos' composed of
 * CURLPROTO_* bits indicating which protocols are actually built
 * into library being used.
 */

CURLcode get_libcurl_info(void)
{
  const char *const *proto;

  /* Pointer to libcurl's run-time version information */
  curlinfo = curl_version_info(CURLVERSION_NOW);
  if(!curlinfo)
    return CURLE_FAILED_INIT;

  /* Build CURLPROTO_* bit pattern with libcurl's built-in protocols */
  built_in_protos = 0;
  if(curlinfo->protocols) {
    for(proto = curlinfo->protocols; *proto; proto++) {
      struct proto_name_pattern const *p;
      for(p = possibly_built_in; p->proto_name; p++) {
        if(curl_strequal(*proto, p->proto_name)) {
          built_in_protos |= p->proto_pattern;
          break;
        }
      }
    }
  }

  return CURLE_OK;
}

/*
 * scheme2protocol() returns the protocol bit for the specified URL scheme
 */
long scheme2protocol(const char *scheme)
{
  struct proto_name_pattern const *p;
  for(p = possibly_built_in; p->proto_name; p++) {
    if(curl_strequal(scheme, p->proto_name))
      return p->proto_pattern;
  }
  return 0;
}
