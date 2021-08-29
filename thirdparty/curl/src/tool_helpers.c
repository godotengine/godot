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

#include "tool_cfgable.h"
#include "tool_msgs.h"
#include "tool_getparam.h"
#include "tool_helpers.h"

#include "memdebug.h" /* keep this as LAST include */

/*
** Helper functions that are used from more than one source file.
*/

const char *param2text(int res)
{
  ParameterError error = (ParameterError)res;
  switch(error) {
  case PARAM_GOT_EXTRA_PARAMETER:
    return "had unsupported trailing garbage";
  case PARAM_OPTION_UNKNOWN:
    return "is unknown";
  case PARAM_OPTION_AMBIGUOUS:
    return "is ambiguous";
  case PARAM_REQUIRES_PARAMETER:
    return "requires parameter";
  case PARAM_BAD_USE:
    return "is badly used here";
  case PARAM_BAD_NUMERIC:
    return "expected a proper numerical parameter";
  case PARAM_NEGATIVE_NUMERIC:
    return "expected a positive numerical parameter";
  case PARAM_LIBCURL_DOESNT_SUPPORT:
    return "the installed libcurl version doesn't support this";
  case PARAM_LIBCURL_UNSUPPORTED_PROTOCOL:
    return "a specified protocol is unsupported by libcurl";
  case PARAM_NO_MEM:
    return "out of memory";
  case PARAM_NO_PREFIX:
    return "the given option can't be reversed with a --no- prefix";
  case PARAM_NUMBER_TOO_LARGE:
    return "too large number";
  case PARAM_NO_NOT_BOOLEAN:
    return "used '--no-' for option that isn't a boolean";
  case PARAM_CONTDISP_SHOW_HEADER:
    return "showing headers and --remote-header-name cannot be combined";
  case PARAM_CONTDISP_RESUME_FROM:
    return "--continue-at and --remote-header-name cannot be combined";
  default:
    return "unknown error";
  }
}

int SetHTTPrequest(struct OperationConfig *config, HttpReq req, HttpReq *store)
{
  /* this mirrors the HttpReq enum in tool_sdecls.h */
  const char *reqname[]= {
    "", /* unspec */
    "GET (-G, --get)",
    "HEAD (-I, --head)",
    "multipart formpost (-F, --form)",
    "POST (-d, --data)"
  };

  if((*store == HTTPREQ_UNSPEC) ||
     (*store == req)) {
    *store = req;
    return 0;
  }
  warnf(config->global, "You can only select one HTTP request method! "
        "You asked for both %s and %s.\n",
        reqname[req], reqname[*store]);

  return 1;
}

void customrequest_helper(struct OperationConfig *config, HttpReq req,
                          char *method)
{
  /* this mirrors the HttpReq enum in tool_sdecls.h */
  const char *dflt[]= {
    "GET",
    "GET",
    "HEAD",
    "POST",
    "POST"
  };

  if(!method)
    ;
  else if(curl_strequal(method, dflt[req])) {
    notef(config->global, "Unnecessary use of -X or --request, %s is already "
          "inferred.\n", dflt[req]);
  }
  else if(curl_strequal(method, "head")) {
    warnf(config->global,
          "Setting custom HTTP method to HEAD with -X/--request may not work "
          "the way you want. Consider using -I/--head instead.\n");
  }
}
