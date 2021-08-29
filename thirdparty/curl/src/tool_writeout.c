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
#define ENABLE_CURLX_PRINTF
/* use our own printf() functions */
#include "curlx.h"
#include "tool_cfgable.h"
#include "tool_writeout.h"
#include "tool_writeout_json.h"

#include "memdebug.h" /* keep this as LAST include */

static int writeTime(FILE *stream, const struct writeoutvar *wovar,
                     struct per_transfer *per, CURLcode per_result,
                     bool use_json);

static int writeString(FILE *stream, const struct writeoutvar *wovar,
                       struct per_transfer *per, CURLcode per_result,
                       bool use_json);

static int writeLong(FILE *stream, const struct writeoutvar *wovar,
                     struct per_transfer *per, CURLcode per_result,
                     bool use_json);

static int writeOffset(FILE *stream, const struct writeoutvar *wovar,
                       struct per_transfer *per, CURLcode per_result,
                       bool use_json);

struct httpmap {
  const char *str;
  int num;
};

static const struct httpmap http_version[] = {
  { "0",   CURL_HTTP_VERSION_NONE},
  { "1",   CURL_HTTP_VERSION_1_0},
  { "1.1", CURL_HTTP_VERSION_1_1},
  { "2",   CURL_HTTP_VERSION_2},
  { "3",   CURL_HTTP_VERSION_3},
  { NULL, 0} /* end of list */
};

/* The designated write function should be the same as the CURLINFO return type
   with exceptions special cased in the respective function. For example,
   http_version uses CURLINFO_HTTP_VERSION which returns the version as a long,
   however it is output as a string and therefore is handled in writeString.

   Yes: "http_version": "1.1"
   No:  "http_version": 1.1

   Variable names should be in alphabetical order.
   */
static const struct writeoutvar variables[] = {
  {"content_type", VAR_CONTENT_TYPE, CURLINFO_CONTENT_TYPE, writeString},
  {"errormsg", VAR_ERRORMSG, 0, writeString},
  {"exitcode", VAR_EXITCODE, 0, writeLong},
  {"filename_effective", VAR_EFFECTIVE_FILENAME, 0, writeString},
  {"ftp_entry_path", VAR_FTP_ENTRY_PATH, CURLINFO_FTP_ENTRY_PATH, writeString},
  {"http_code", VAR_HTTP_CODE, CURLINFO_RESPONSE_CODE, writeLong},
  {"http_connect", VAR_HTTP_CODE_PROXY, CURLINFO_HTTP_CONNECTCODE, writeLong},
  {"http_version", VAR_HTTP_VERSION, CURLINFO_HTTP_VERSION, writeString},
  {"json", VAR_JSON, 0, NULL},
  {"local_ip", VAR_LOCAL_IP, CURLINFO_LOCAL_IP, writeString},
  {"local_port", VAR_LOCAL_PORT, CURLINFO_LOCAL_PORT, writeLong},
  {"method", VAR_EFFECTIVE_METHOD, CURLINFO_EFFECTIVE_METHOD, writeString},
  {"num_connects", VAR_NUM_CONNECTS, CURLINFO_NUM_CONNECTS, writeLong},
  {"num_headers", VAR_NUM_HEADERS, 0, writeLong},
  {"num_redirects", VAR_REDIRECT_COUNT, CURLINFO_REDIRECT_COUNT, writeLong},
  {"onerror", VAR_ONERROR, 0, NULL},
  {"proxy_ssl_verify_result", VAR_PROXY_SSL_VERIFY_RESULT,
   CURLINFO_PROXY_SSL_VERIFYRESULT, writeLong},
  {"redirect_url", VAR_REDIRECT_URL, CURLINFO_REDIRECT_URL, writeString},
  {"referer", VAR_REFERER, CURLINFO_REFERER, writeString},
  {"remote_ip", VAR_PRIMARY_IP, CURLINFO_PRIMARY_IP, writeString},
  {"remote_port", VAR_PRIMARY_PORT, CURLINFO_PRIMARY_PORT, writeLong},
  {"response_code", VAR_HTTP_CODE, CURLINFO_RESPONSE_CODE, writeLong},
  {"scheme", VAR_SCHEME, CURLINFO_SCHEME, writeString},
  {"size_download", VAR_SIZE_DOWNLOAD, CURLINFO_SIZE_DOWNLOAD_T, writeOffset},
  {"size_header", VAR_HEADER_SIZE, CURLINFO_HEADER_SIZE, writeLong},
  {"size_request", VAR_REQUEST_SIZE, CURLINFO_REQUEST_SIZE, writeLong},
  {"size_upload", VAR_SIZE_UPLOAD, CURLINFO_SIZE_UPLOAD_T, writeOffset},
  {"speed_download", VAR_SPEED_DOWNLOAD, CURLINFO_SPEED_DOWNLOAD_T,
   writeOffset},
  {"speed_upload", VAR_SPEED_UPLOAD, CURLINFO_SPEED_UPLOAD_T, writeOffset},
  {"ssl_verify_result", VAR_SSL_VERIFY_RESULT, CURLINFO_SSL_VERIFYRESULT,
   writeLong},
  {"stderr", VAR_STDERR, 0, NULL},
  {"stdout", VAR_STDOUT, 0, NULL},
  {"time_appconnect", VAR_APPCONNECT_TIME, CURLINFO_APPCONNECT_TIME_T,
   writeTime},
  {"time_connect", VAR_CONNECT_TIME, CURLINFO_CONNECT_TIME_T, writeTime},
  {"time_namelookup", VAR_NAMELOOKUP_TIME, CURLINFO_NAMELOOKUP_TIME_T,
   writeTime},
  {"time_pretransfer", VAR_PRETRANSFER_TIME, CURLINFO_PRETRANSFER_TIME_T,
   writeTime},
  {"time_redirect", VAR_REDIRECT_TIME, CURLINFO_REDIRECT_TIME_T, writeTime},
  {"time_starttransfer", VAR_STARTTRANSFER_TIME, CURLINFO_STARTTRANSFER_TIME_T,
   writeTime},
  {"time_total", VAR_TOTAL_TIME, CURLINFO_TOTAL_TIME_T, writeTime},
  {"url", VAR_INPUT_URL, 0, writeString},
  {"url_effective", VAR_EFFECTIVE_URL, CURLINFO_EFFECTIVE_URL, writeString},
  {"urlnum", VAR_URLNUM, 0, writeLong},
  {NULL, VAR_NONE, 0, NULL}
};

static int writeTime(FILE *stream, const struct writeoutvar *wovar,
                     struct per_transfer *per, CURLcode per_result,
                     bool use_json)
{
  bool valid = false;
  curl_off_t us = 0;

  (void)per;
  (void)per_result;
  DEBUGASSERT(wovar->writefunc == writeTime);

  if(wovar->ci) {
    if(!curl_easy_getinfo(per->curl, wovar->ci, &us))
      valid = true;
  }
  else {
    DEBUGASSERT(0);
  }

  if(valid) {
    curl_off_t secs = us / 1000000;
    us %= 1000000;

    if(use_json)
      fprintf(stream, "\"%s\":", wovar->name);

    fprintf(stream, "%" CURL_FORMAT_CURL_OFF_TU
            ".%06" CURL_FORMAT_CURL_OFF_TU, secs, us);
  }
  else {
    if(use_json)
      fprintf(stream, "\"%s\":null", wovar->name);
  }

  return 1; /* return 1 if anything was written */
}

static int writeString(FILE *stream, const struct writeoutvar *wovar,
                       struct per_transfer *per, CURLcode per_result,
                       bool use_json)
{
  bool valid = false;
  const char *strinfo = NULL;

  DEBUGASSERT(wovar->writefunc == writeString);

  if(wovar->ci) {
    if(wovar->ci == CURLINFO_HTTP_VERSION) {
      long version = 0;
      if(!curl_easy_getinfo(per->curl, CURLINFO_HTTP_VERSION, &version)) {
        const struct httpmap *m = &http_version[0];
        while(m->str) {
          if(m->num == version) {
            strinfo = m->str;
            valid = true;
            break;
          }
          m++;
        }
      }
    }
    else {
      if(!curl_easy_getinfo(per->curl, wovar->ci, &strinfo) && strinfo)
        valid = true;
    }
  }
  else {
    switch(wovar->id) {
    case VAR_ERRORMSG:
      if(per_result) {
        strinfo = per->errorbuffer[0] ? per->errorbuffer :
                  curl_easy_strerror(per_result);
        valid = true;
      }
      break;
    case VAR_EFFECTIVE_FILENAME:
      if(per->outs.filename) {
        strinfo = per->outs.filename;
        valid = true;
      }
      break;
    case VAR_INPUT_URL:
      if(per->this_url) {
        strinfo = per->this_url;
        valid = true;
      }
      break;
    default:
      DEBUGASSERT(0);
      break;
    }
  }

  if(valid) {
    DEBUGASSERT(strinfo);
    if(use_json) {
      fprintf(stream, "\"%s\":\"", wovar->name);
      jsonWriteString(stream, strinfo);
      fputs("\"", stream);
    }
    else
      fputs(strinfo, stream);
  }
  else {
    if(use_json)
      fprintf(stream, "\"%s\":null", wovar->name);
  }

  return 1; /* return 1 if anything was written */
}

static int writeLong(FILE *stream, const struct writeoutvar *wovar,
                     struct per_transfer *per, CURLcode per_result,
                     bool use_json)
{
  bool valid = false;
  long longinfo = 0;

  DEBUGASSERT(wovar->writefunc == writeLong);

  if(wovar->ci) {
    if(!curl_easy_getinfo(per->curl, wovar->ci, &longinfo))
      valid = true;
  }
  else {
    switch(wovar->id) {
    case VAR_NUM_HEADERS:
      longinfo = per->num_headers;
      valid = true;
      break;
    case VAR_EXITCODE:
      longinfo = per_result;
      valid = true;
      break;
    case VAR_URLNUM:
      if(per->urlnum <= INT_MAX) {
        longinfo = (long)per->urlnum;
        valid = true;
      }
      break;
    default:
      DEBUGASSERT(0);
      break;
    }
  }

  if(valid) {
    if(use_json)
      fprintf(stream, "\"%s\":%ld", wovar->name, longinfo);
    else {
      if(wovar->id == VAR_HTTP_CODE || wovar->id == VAR_HTTP_CODE_PROXY)
        fprintf(stream, "%03ld", longinfo);
      else
        fprintf(stream, "%ld", longinfo);
    }
  }
  else {
    if(use_json)
      fprintf(stream, "\"%s\":null", wovar->name);
  }

  return 1; /* return 1 if anything was written */
}

static int writeOffset(FILE *stream, const struct writeoutvar *wovar,
                       struct per_transfer *per, CURLcode per_result,
                       bool use_json)
{
  bool valid = false;
  curl_off_t offinfo = 0;

  (void)per;
  (void)per_result;
  DEBUGASSERT(wovar->writefunc == writeOffset);

  if(wovar->ci) {
    if(!curl_easy_getinfo(per->curl, wovar->ci, &offinfo))
      valid = true;
  }
  else {
    DEBUGASSERT(0);
  }

  if(valid) {
    if(use_json)
      fprintf(stream, "\"%s\":", wovar->name);

    fprintf(stream, "%" CURL_FORMAT_CURL_OFF_T, offinfo);
  }
  else {
    if(use_json)
      fprintf(stream, "\"%s\":null", wovar->name);
  }

  return 1; /* return 1 if anything was written */
}

void ourWriteOut(const char *writeinfo, struct per_transfer *per,
                 CURLcode per_result)
{
  FILE *stream = stdout;
  const char *ptr = writeinfo;
  bool done = FALSE;

  while(ptr && *ptr && !done) {
    if('%' == *ptr && ptr[1]) {
      if('%' == ptr[1]) {
        /* an escaped %-letter */
        fputc('%', stream);
        ptr += 2;
      }
      else {
        /* this is meant as a variable to output */
        char *end;
        if('{' == ptr[1]) {
          char keepit;
          int i;
          bool match = FALSE;
          end = strchr(ptr, '}');
          ptr += 2; /* pass the % and the { */
          if(!end) {
            fputs("%{", stream);
            continue;
          }
          keepit = *end;
          *end = 0; /* null-terminate */
          for(i = 0; variables[i].name; i++) {
            if(curl_strequal(ptr, variables[i].name)) {
              match = TRUE;
              switch(variables[i].id) {
              case VAR_ONERROR:
                if(per_result == CURLE_OK)
                  /* this isn't error so skip the rest */
                  done = TRUE;
                break;
              case VAR_STDOUT:
                stream = stdout;
                break;
              case VAR_STDERR:
                stream = stderr;
                break;
              case VAR_JSON:
                ourWriteOutJSON(stream, variables, per, per_result);
                break;
              default:
                (void)variables[i].writefunc(stream, &variables[i],
                                             per, per_result, false);
                break;
              }
              break;
            }
          }
          if(!match) {
            fprintf(stderr, "curl: unknown --write-out variable: '%s'\n", ptr);
          }
          ptr = end + 1; /* pass the end */
          *end = keepit;
        }
        else {
          /* illegal syntax, then just output the characters that are used */
          fputc('%', stream);
          fputc(ptr[1], stream);
          ptr += 2;
        }
      }
    }
    else if('\\' == *ptr && ptr[1]) {
      switch(ptr[1]) {
      case 'r':
        fputc('\r', stream);
        break;
      case 'n':
        fputc('\n', stream);
        break;
      case 't':
        fputc('\t', stream);
        break;
      default:
        /* unknown, just output this */
        fputc(*ptr, stream);
        fputc(ptr[1], stream);
        break;
      }
      ptr += 2;
    }
    else {
      fputc(*ptr, stream);
      ptr++;
    }
  }
}
