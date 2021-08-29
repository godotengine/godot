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
#include "tool_setup.h"

#define ENABLE_CURLX_PRINTF
/* use our own printf() functions */
#include "curlx.h"

#include "tool_cfgable.h"
#include "tool_convert.h"
#include "tool_msgs.h"
#include "tool_cb_dbg.h"
#include "tool_util.h"

#include "memdebug.h" /* keep this as LAST include */

static void dump(const char *timebuf, const char *text,
                 FILE *stream, const unsigned char *ptr, size_t size,
                 trace tracetype, curl_infotype infotype);

/*
** callback for CURLOPT_DEBUGFUNCTION
*/

int tool_debug_cb(CURL *handle, curl_infotype type,
                  char *data, size_t size,
                  void *userdata)
{
  struct OperationConfig *operation = userdata;
  struct GlobalConfig *config = operation->global;
  FILE *output = config->errors;
  const char *text;
  struct timeval tv;
  char timebuf[20];
  time_t secs;

  (void)handle; /* not used */

  if(config->tracetime) {
    struct tm *now;
    static time_t epoch_offset;
    static int    known_offset;
    tv = tvnow();
    if(!known_offset) {
      epoch_offset = time(NULL) - tv.tv_sec;
      known_offset = 1;
    }
    secs = epoch_offset + tv.tv_sec;
    /* !checksrc! disable BANNEDFUNC 1 */
    now = localtime(&secs);  /* not thread safe but we don't care */
    msnprintf(timebuf, sizeof(timebuf), "%02d:%02d:%02d.%06ld ",
              now->tm_hour, now->tm_min, now->tm_sec, (long)tv.tv_usec);
  }
  else
    timebuf[0] = 0;

  if(!config->trace_stream) {
    /* open for append */
    if(!strcmp("-", config->trace_dump))
      config->trace_stream = stdout;
    else if(!strcmp("%", config->trace_dump))
      /* Ok, this is somewhat hackish but we do it undocumented for now */
      config->trace_stream = config->errors;  /* aka stderr */
    else {
      config->trace_stream = fopen(config->trace_dump, FOPEN_WRITETEXT);
      config->trace_fopened = TRUE;
    }
  }

  if(config->trace_stream)
    output = config->trace_stream;

  if(!output) {
    warnf(config, "Failed to create/open output");
    return 0;
  }

  if(config->tracetype == TRACE_PLAIN) {
    /*
     * This is the trace look that is similar to what libcurl makes on its
     * own.
     */
    static const char * const s_infotype[] = {
      "*", "<", ">", "{", "}", "{", "}"
    };
    static bool newl = FALSE;
    static bool traced_data = FALSE;

    switch(type) {
    case CURLINFO_HEADER_OUT:
      if(size > 0) {
        size_t st = 0;
        size_t i;
        for(i = 0; i < size - 1; i++) {
          if(data[i] == '\n') { /* LF */
            if(!newl) {
              fprintf(output, "%s%s ", timebuf, s_infotype[type]);
            }
            (void)fwrite(data + st, i - st + 1, 1, output);
            st = i + 1;
            newl = FALSE;
          }
        }
        if(!newl)
          fprintf(output, "%s%s ", timebuf, s_infotype[type]);
        (void)fwrite(data + st, i - st + 1, 1, output);
      }
      newl = (size && (data[size - 1] != '\n')) ? TRUE : FALSE;
      traced_data = FALSE;
      break;
    case CURLINFO_TEXT:
    case CURLINFO_HEADER_IN:
      if(!newl)
        fprintf(output, "%s%s ", timebuf, s_infotype[type]);
      (void)fwrite(data, size, 1, output);
      newl = (size && (data[size - 1] != '\n')) ? TRUE : FALSE;
      traced_data = FALSE;
      break;
    case CURLINFO_DATA_OUT:
    case CURLINFO_DATA_IN:
    case CURLINFO_SSL_DATA_IN:
    case CURLINFO_SSL_DATA_OUT:
      if(!traced_data) {
        /* if the data is output to a tty and we're sending this debug trace
           to stderr or stdout, we don't display the alert about the data not
           being shown as the data _is_ shown then just not via this
           function */
        if(!config->isatty || ((output != stderr) && (output != stdout))) {
          if(!newl)
            fprintf(output, "%s%s ", timebuf, s_infotype[type]);
          fprintf(output, "[%zu bytes data]\n", size);
          newl = FALSE;
          traced_data = TRUE;
        }
      }
      break;
    default: /* nada */
      newl = FALSE;
      traced_data = FALSE;
      break;
    }

    return 0;
  }

#ifdef CURL_DOES_CONVERSIONS
  /* Special processing is needed for CURLINFO_HEADER_OUT blocks
   * if they contain both headers and data (separated by CRLFCRLF).
   * We dump the header text and then switch type to CURLINFO_DATA_OUT.
   */
  if((type == CURLINFO_HEADER_OUT) && (size > 4)) {
    size_t i;
    for(i = 0; i < size - 4; i++) {
      if(memcmp(&data[i], "\r\n\r\n", 4) == 0) {
        /* dump everything through the CRLFCRLF as a sent header */
        text = "=> Send header";
        dump(timebuf, text, output, (unsigned char *)data, i + 4,
             config->tracetype, type);
        data += i + 3;
        size -= i + 4;
        type = CURLINFO_DATA_OUT;
        data += 1;
        break;
      }
    }
  }
#endif /* CURL_DOES_CONVERSIONS */

  switch(type) {
  case CURLINFO_TEXT:
    fprintf(output, "%s== Info: %.*s", timebuf, (int)size, data);
    /* FALLTHROUGH */
  default: /* in case a new one is introduced to shock us */
    return 0;

  case CURLINFO_HEADER_OUT:
    text = "=> Send header";
    break;
  case CURLINFO_DATA_OUT:
    text = "=> Send data";
    break;
  case CURLINFO_HEADER_IN:
    text = "<= Recv header";
    break;
  case CURLINFO_DATA_IN:
    text = "<= Recv data";
    break;
  case CURLINFO_SSL_DATA_IN:
    text = "<= Recv SSL data";
    break;
  case CURLINFO_SSL_DATA_OUT:
    text = "=> Send SSL data";
    break;
  }

  dump(timebuf, text, output, (unsigned char *) data, size, config->tracetype,
       type);
  return 0;
}

static void dump(const char *timebuf, const char *text,
                 FILE *stream, const unsigned char *ptr, size_t size,
                 trace tracetype, curl_infotype infotype)
{
  size_t i;
  size_t c;

  unsigned int width = 0x10;

  if(tracetype == TRACE_ASCII)
    /* without the hex output, we can fit more on screen */
    width = 0x40;

  fprintf(stream, "%s%s, %zu bytes (0x%zx)\n", timebuf, text, size, size);

  for(i = 0; i < size; i += width) {

    fprintf(stream, "%04zx: ", i);

    if(tracetype == TRACE_BIN) {
      /* hex not disabled, show it */
      for(c = 0; c < width; c++)
        if(i + c < size)
          fprintf(stream, "%02x ", ptr[i + c]);
        else
          fputs("   ", stream);
    }

    for(c = 0; (c < width) && (i + c < size); c++) {
      /* check for 0D0A; if found, skip past and start a new line of output */
      if((tracetype == TRACE_ASCII) &&
         (i + c + 1 < size) && (ptr[i + c] == 0x0D) &&
         (ptr[i + c + 1] == 0x0A)) {
        i += (c + 2 - width);
        break;
      }
#ifdef CURL_DOES_CONVERSIONS
      /* repeat the 0D0A check above but use the host encoding for CRLF */
      if((tracetype == TRACE_ASCII) &&
         (i + c + 1 < size) && (ptr[i + c] == '\r') &&
         (ptr[i + c + 1] == '\n')) {
        i += (c + 2 - width);
        break;
      }
      /* convert to host encoding and print this character */
      fprintf(stream, "%c", convert_char(infotype, ptr[i + c]));
#else
      (void)infotype;
      fprintf(stream, "%c", ((ptr[i + c] >= 0x20) && (ptr[i + c] < 0x80)) ?
              ptr[i + c] : UNPRINTABLE_CHAR);
#endif /* CURL_DOES_CONVERSIONS */
      /* check again for 0D0A, to avoid an extra \n if it's at width */
      if((tracetype == TRACE_ASCII) &&
         (i + c + 2 < size) && (ptr[i + c + 1] == 0x0D) &&
         (ptr[i + c + 2] == 0x0A)) {
        i += (c + 3 - width);
        break;
      }
    }
    fputc('\n', stream); /* newline */
  }
  fflush(stream);
}
