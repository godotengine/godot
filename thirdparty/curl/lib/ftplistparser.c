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

/**
 * Now implemented:
 *
 * 1) Unix version 1
 * drwxr-xr-x 1 user01 ftp  512 Jan 29 23:32 prog
 * 2) Unix version 2
 * drwxr-xr-x 1 user01 ftp  512 Jan 29 1997  prog
 * 3) Unix version 3
 * drwxr-xr-x 1      1   1  512 Jan 29 23:32 prog
 * 4) Unix symlink
 * lrwxr-xr-x 1 user01 ftp  512 Jan 29 23:32 prog -> prog2000
 * 5) DOS style
 * 01-29-97 11:32PM <DIR> prog
 */

#include "curl_setup.h"

#ifndef CURL_DISABLE_FTP

#include <curl/curl.h>

#include "urldata.h"
#include "fileinfo.h"
#include "llist.h"
#include "strtoofft.h"
#include "ftp.h"
#include "ftplistparser.h"
#include "curl_fnmatch.h"
#include "curl_memory.h"
#include "multiif.h"
/* The last #include file should be: */
#include "memdebug.h"

/* allocs buffer which will contain one line of LIST command response */
#define FTP_BUFFER_ALLOCSIZE 160

typedef enum {
  PL_UNIX_TOTALSIZE = 0,
  PL_UNIX_FILETYPE,
  PL_UNIX_PERMISSION,
  PL_UNIX_HLINKS,
  PL_UNIX_USER,
  PL_UNIX_GROUP,
  PL_UNIX_SIZE,
  PL_UNIX_TIME,
  PL_UNIX_FILENAME,
  PL_UNIX_SYMLINK
} pl_unix_mainstate;

typedef union {
  enum {
    PL_UNIX_TOTALSIZE_INIT = 0,
    PL_UNIX_TOTALSIZE_READING
  } total_dirsize;

  enum {
    PL_UNIX_HLINKS_PRESPACE = 0,
    PL_UNIX_HLINKS_NUMBER
  } hlinks;

  enum {
    PL_UNIX_USER_PRESPACE = 0,
    PL_UNIX_USER_PARSING
  } user;

  enum {
    PL_UNIX_GROUP_PRESPACE = 0,
    PL_UNIX_GROUP_NAME
  } group;

  enum {
    PL_UNIX_SIZE_PRESPACE = 0,
    PL_UNIX_SIZE_NUMBER
  } size;

  enum {
    PL_UNIX_TIME_PREPART1 = 0,
    PL_UNIX_TIME_PART1,
    PL_UNIX_TIME_PREPART2,
    PL_UNIX_TIME_PART2,
    PL_UNIX_TIME_PREPART3,
    PL_UNIX_TIME_PART3
  } time;

  enum {
    PL_UNIX_FILENAME_PRESPACE = 0,
    PL_UNIX_FILENAME_NAME,
    PL_UNIX_FILENAME_WINDOWSEOL
  } filename;

  enum {
    PL_UNIX_SYMLINK_PRESPACE = 0,
    PL_UNIX_SYMLINK_NAME,
    PL_UNIX_SYMLINK_PRETARGET1,
    PL_UNIX_SYMLINK_PRETARGET2,
    PL_UNIX_SYMLINK_PRETARGET3,
    PL_UNIX_SYMLINK_PRETARGET4,
    PL_UNIX_SYMLINK_TARGET,
    PL_UNIX_SYMLINK_WINDOWSEOL
  } symlink;
} pl_unix_substate;

typedef enum {
  PL_WINNT_DATE = 0,
  PL_WINNT_TIME,
  PL_WINNT_DIRORSIZE,
  PL_WINNT_FILENAME
} pl_winNT_mainstate;

typedef union {
  enum {
    PL_WINNT_TIME_PRESPACE = 0,
    PL_WINNT_TIME_TIME
  } time;
  enum {
    PL_WINNT_DIRORSIZE_PRESPACE = 0,
    PL_WINNT_DIRORSIZE_CONTENT
  } dirorsize;
  enum {
    PL_WINNT_FILENAME_PRESPACE = 0,
    PL_WINNT_FILENAME_CONTENT,
    PL_WINNT_FILENAME_WINEOL
  } filename;
} pl_winNT_substate;

/* This struct is used in wildcard downloading - for parsing LIST response */
struct ftp_parselist_data {
  enum {
    OS_TYPE_UNKNOWN = 0,
    OS_TYPE_UNIX,
    OS_TYPE_WIN_NT
  } os_type;

  union {
    struct {
      pl_unix_mainstate main;
      pl_unix_substate sub;
    } UNIX;

    struct {
      pl_winNT_mainstate main;
      pl_winNT_substate sub;
    } NT;
  } state;

  CURLcode error;
  struct fileinfo *file_data;
  unsigned int item_length;
  size_t item_offset;
  struct {
    size_t filename;
    size_t user;
    size_t group;
    size_t time;
    size_t perm;
    size_t symlink_target;
  } offsets;
};

struct ftp_parselist_data *Curl_ftp_parselist_data_alloc(void)
{
  return calloc(1, sizeof(struct ftp_parselist_data));
}


void Curl_ftp_parselist_data_free(struct ftp_parselist_data **parserp)
{
  struct ftp_parselist_data *parser = *parserp;
  if(parser)
    Curl_fileinfo_cleanup(parser->file_data);
  free(parser);
  *parserp = NULL;
}


CURLcode Curl_ftp_parselist_geterror(struct ftp_parselist_data *pl_data)
{
  return pl_data->error;
}


#define FTP_LP_MALFORMATED_PERM 0x01000000

static int ftp_pl_get_permission(const char *str)
{
  int permissions = 0;
  /* USER */
  if(str[0] == 'r')
    permissions |= 1 << 8;
  else if(str[0] != '-')
    permissions |= FTP_LP_MALFORMATED_PERM;
  if(str[1] == 'w')
    permissions |= 1 << 7;
  else if(str[1] != '-')
    permissions |= FTP_LP_MALFORMATED_PERM;

  if(str[2] == 'x')
    permissions |= 1 << 6;
  else if(str[2] == 's') {
    permissions |= 1 << 6;
    permissions |= 1 << 11;
  }
  else if(str[2] == 'S')
    permissions |= 1 << 11;
  else if(str[2] != '-')
    permissions |= FTP_LP_MALFORMATED_PERM;
  /* GROUP */
  if(str[3] == 'r')
    permissions |= 1 << 5;
  else if(str[3] != '-')
    permissions |= FTP_LP_MALFORMATED_PERM;
  if(str[4] == 'w')
    permissions |= 1 << 4;
  else if(str[4] != '-')
    permissions |= FTP_LP_MALFORMATED_PERM;
  if(str[5] == 'x')
    permissions |= 1 << 3;
  else if(str[5] == 's') {
    permissions |= 1 << 3;
    permissions |= 1 << 10;
  }
  else if(str[5] == 'S')
    permissions |= 1 << 10;
  else if(str[5] != '-')
    permissions |= FTP_LP_MALFORMATED_PERM;
  /* others */
  if(str[6] == 'r')
    permissions |= 1 << 2;
  else if(str[6] != '-')
    permissions |= FTP_LP_MALFORMATED_PERM;
  if(str[7] == 'w')
    permissions |= 1 << 1;
  else if(str[7] != '-')
      permissions |= FTP_LP_MALFORMATED_PERM;
  if(str[8] == 'x')
    permissions |= 1;
  else if(str[8] == 't') {
    permissions |= 1;
    permissions |= 1 << 9;
  }
  else if(str[8] == 'T')
    permissions |= 1 << 9;
  else if(str[8] != '-')
    permissions |= FTP_LP_MALFORMATED_PERM;

  return permissions;
}

static CURLcode ftp_pl_insert_finfo(struct Curl_easy *data,
                                    struct fileinfo *infop)
{
  curl_fnmatch_callback compare;
  struct WildcardData *wc = &data->wildcard;
  struct ftp_wc *ftpwc = wc->protdata;
  struct Curl_llist *llist = &wc->filelist;
  struct ftp_parselist_data *parser = ftpwc->parser;
  bool add = TRUE;
  struct curl_fileinfo *finfo = &infop->info;

  /* move finfo pointers to b_data */
  char *str = finfo->b_data;
  finfo->filename       = str + parser->offsets.filename;
  finfo->strings.group  = parser->offsets.group ?
                          str + parser->offsets.group : NULL;
  finfo->strings.perm   = parser->offsets.perm ?
                          str + parser->offsets.perm : NULL;
  finfo->strings.target = parser->offsets.symlink_target ?
                          str + parser->offsets.symlink_target : NULL;
  finfo->strings.time   = str + parser->offsets.time;
  finfo->strings.user   = parser->offsets.user ?
                          str + parser->offsets.user : NULL;

  /* get correct fnmatch callback */
  compare = data->set.fnmatch;
  if(!compare)
    compare = Curl_fnmatch;

  /* filter pattern-corresponding filenames */
  Curl_set_in_callback(data, true);
  if(compare(data->set.fnmatch_data, wc->pattern,
             finfo->filename) == 0) {
    /* discard symlink which is containing multiple " -> " */
    if((finfo->filetype == CURLFILETYPE_SYMLINK) && finfo->strings.target &&
       (strstr(finfo->strings.target, " -> "))) {
      add = FALSE;
    }
  }
  else {
    add = FALSE;
  }
  Curl_set_in_callback(data, false);

  if(add) {
    Curl_llist_insert_next(llist, llist->tail, finfo, &infop->list);
  }
  else {
    Curl_fileinfo_cleanup(infop);
  }

  ftpwc->parser->file_data = NULL;
  return CURLE_OK;
}

size_t Curl_ftp_parselist(char *buffer, size_t size, size_t nmemb,
                          void *connptr)
{
  size_t bufflen = size*nmemb;
  struct Curl_easy *data = (struct Curl_easy *)connptr;
  struct ftp_wc *ftpwc = data->wildcard.protdata;
  struct ftp_parselist_data *parser = ftpwc->parser;
  struct fileinfo *infop;
  struct curl_fileinfo *finfo;
  unsigned long i = 0;
  CURLcode result;
  size_t retsize = bufflen;

  if(parser->error) { /* error in previous call */
    /* scenario:
     * 1. call => OK..
     * 2. call => OUT_OF_MEMORY (or other error)
     * 3. (last) call => is skipped RIGHT HERE and the error is hadled later
     *    in wc_statemach()
     */
    goto fail;
  }

  if(parser->os_type == OS_TYPE_UNKNOWN && bufflen > 0) {
    /* considering info about FILE response format */
    parser->os_type = (buffer[0] >= '0' && buffer[0] <= '9') ?
                       OS_TYPE_WIN_NT : OS_TYPE_UNIX;
  }

  while(i < bufflen) { /* FSM */

    char c = buffer[i];
    if(!parser->file_data) { /* tmp file data is not allocated yet */
      parser->file_data = Curl_fileinfo_alloc();
      if(!parser->file_data) {
        parser->error = CURLE_OUT_OF_MEMORY;
        goto fail;
      }
      parser->file_data->info.b_data = malloc(FTP_BUFFER_ALLOCSIZE);
      if(!parser->file_data->info.b_data) {
        parser->error = CURLE_OUT_OF_MEMORY;
        goto fail;
      }
      parser->file_data->info.b_size = FTP_BUFFER_ALLOCSIZE;
      parser->item_offset = 0;
      parser->item_length = 0;
    }

    infop = parser->file_data;
    finfo = &infop->info;
    finfo->b_data[finfo->b_used++] = c;

    if(finfo->b_used >= finfo->b_size - 1) {
      /* if it is important, extend buffer space for file data */
      char *tmp = realloc(finfo->b_data,
                          finfo->b_size + FTP_BUFFER_ALLOCSIZE);
      if(tmp) {
        finfo->b_size += FTP_BUFFER_ALLOCSIZE;
        finfo->b_data = tmp;
      }
      else {
        Curl_fileinfo_cleanup(parser->file_data);
        parser->file_data = NULL;
        parser->error = CURLE_OUT_OF_MEMORY;
        goto fail;
      }
    }

    switch(parser->os_type) {
    case OS_TYPE_UNIX:
      switch(parser->state.UNIX.main) {
      case PL_UNIX_TOTALSIZE:
        switch(parser->state.UNIX.sub.total_dirsize) {
        case PL_UNIX_TOTALSIZE_INIT:
          if(c == 't') {
            parser->state.UNIX.sub.total_dirsize = PL_UNIX_TOTALSIZE_READING;
            parser->item_length++;
          }
          else {
            parser->state.UNIX.main = PL_UNIX_FILETYPE;
            /* start FSM again not considering size of directory */
            finfo->b_used = 0;
            continue;
          }
          break;
        case PL_UNIX_TOTALSIZE_READING:
          parser->item_length++;
          if(c == '\r') {
            parser->item_length--;
            finfo->b_used--;
          }
          else if(c == '\n') {
            finfo->b_data[parser->item_length - 1] = 0;
            if(strncmp("total ", finfo->b_data, 6) == 0) {
              char *endptr = finfo->b_data + 6;
              /* here we can deal with directory size, pass the leading
                 whitespace and then the digits */
              while(ISSPACE(*endptr))
                endptr++;
              while(ISDIGIT(*endptr))
                endptr++;
              if(*endptr) {
                parser->error = CURLE_FTP_BAD_FILE_LIST;
                goto fail;
              }
              parser->state.UNIX.main = PL_UNIX_FILETYPE;
              finfo->b_used = 0;
            }
            else {
              parser->error = CURLE_FTP_BAD_FILE_LIST;
              goto fail;
            }
          }
          break;
        }
        break;
      case PL_UNIX_FILETYPE:
        switch(c) {
        case '-':
          finfo->filetype = CURLFILETYPE_FILE;
          break;
        case 'd':
          finfo->filetype = CURLFILETYPE_DIRECTORY;
          break;
        case 'l':
          finfo->filetype = CURLFILETYPE_SYMLINK;
          break;
        case 'p':
          finfo->filetype = CURLFILETYPE_NAMEDPIPE;
          break;
        case 's':
          finfo->filetype = CURLFILETYPE_SOCKET;
          break;
        case 'c':
          finfo->filetype = CURLFILETYPE_DEVICE_CHAR;
          break;
        case 'b':
          finfo->filetype = CURLFILETYPE_DEVICE_BLOCK;
          break;
        case 'D':
          finfo->filetype = CURLFILETYPE_DOOR;
          break;
        default:
          parser->error = CURLE_FTP_BAD_FILE_LIST;
          goto fail;
        }
        parser->state.UNIX.main = PL_UNIX_PERMISSION;
        parser->item_length = 0;
        parser->item_offset = 1;
        break;
      case PL_UNIX_PERMISSION:
        parser->item_length++;
        if(parser->item_length <= 9) {
          if(!strchr("rwx-tTsS", c)) {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
        }
        else if(parser->item_length == 10) {
          unsigned int perm;
          if(c != ' ') {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          finfo->b_data[10] = 0; /* terminate permissions */
          perm = ftp_pl_get_permission(finfo->b_data + parser->item_offset);
          if(perm & FTP_LP_MALFORMATED_PERM) {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          parser->file_data->info.flags |= CURLFINFOFLAG_KNOWN_PERM;
          parser->file_data->info.perm = perm;
          parser->offsets.perm = parser->item_offset;

          parser->item_length = 0;
          parser->state.UNIX.main = PL_UNIX_HLINKS;
          parser->state.UNIX.sub.hlinks = PL_UNIX_HLINKS_PRESPACE;
        }
        break;
      case PL_UNIX_HLINKS:
        switch(parser->state.UNIX.sub.hlinks) {
        case PL_UNIX_HLINKS_PRESPACE:
          if(c != ' ') {
            if(c >= '0' && c <= '9') {
              parser->item_offset = finfo->b_used - 1;
              parser->item_length = 1;
              parser->state.UNIX.sub.hlinks = PL_UNIX_HLINKS_NUMBER;
            }
            else {
              parser->error = CURLE_FTP_BAD_FILE_LIST;
              goto fail;
            }
          }
          break;
        case PL_UNIX_HLINKS_NUMBER:
          parser->item_length ++;
          if(c == ' ') {
            char *p;
            long int hlinks;
            finfo->b_data[parser->item_offset + parser->item_length - 1] = 0;
            hlinks = strtol(finfo->b_data + parser->item_offset, &p, 10);
            if(p[0] == '\0' && hlinks != LONG_MAX && hlinks != LONG_MIN) {
              parser->file_data->info.flags |= CURLFINFOFLAG_KNOWN_HLINKCOUNT;
              parser->file_data->info.hardlinks = hlinks;
            }
            parser->item_length = 0;
            parser->item_offset = 0;
            parser->state.UNIX.main = PL_UNIX_USER;
            parser->state.UNIX.sub.user = PL_UNIX_USER_PRESPACE;
          }
          else if(c < '0' || c > '9') {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          break;
        }
        break;
      case PL_UNIX_USER:
        switch(parser->state.UNIX.sub.user) {
        case PL_UNIX_USER_PRESPACE:
          if(c != ' ') {
            parser->item_offset = finfo->b_used - 1;
            parser->item_length = 1;
            parser->state.UNIX.sub.user = PL_UNIX_USER_PARSING;
          }
          break;
        case PL_UNIX_USER_PARSING:
          parser->item_length++;
          if(c == ' ') {
            finfo->b_data[parser->item_offset + parser->item_length - 1] = 0;
            parser->offsets.user = parser->item_offset;
            parser->state.UNIX.main = PL_UNIX_GROUP;
            parser->state.UNIX.sub.group = PL_UNIX_GROUP_PRESPACE;
            parser->item_offset = 0;
            parser->item_length = 0;
          }
          break;
        }
        break;
      case PL_UNIX_GROUP:
        switch(parser->state.UNIX.sub.group) {
        case PL_UNIX_GROUP_PRESPACE:
          if(c != ' ') {
            parser->item_offset = finfo->b_used - 1;
            parser->item_length = 1;
            parser->state.UNIX.sub.group = PL_UNIX_GROUP_NAME;
          }
          break;
        case PL_UNIX_GROUP_NAME:
          parser->item_length++;
          if(c == ' ') {
            finfo->b_data[parser->item_offset + parser->item_length - 1] = 0;
            parser->offsets.group = parser->item_offset;
            parser->state.UNIX.main = PL_UNIX_SIZE;
            parser->state.UNIX.sub.size = PL_UNIX_SIZE_PRESPACE;
            parser->item_offset = 0;
            parser->item_length = 0;
          }
          break;
        }
        break;
      case PL_UNIX_SIZE:
        switch(parser->state.UNIX.sub.size) {
        case PL_UNIX_SIZE_PRESPACE:
          if(c != ' ') {
            if(c >= '0' && c <= '9') {
              parser->item_offset = finfo->b_used - 1;
              parser->item_length = 1;
              parser->state.UNIX.sub.size = PL_UNIX_SIZE_NUMBER;
            }
            else {
              parser->error = CURLE_FTP_BAD_FILE_LIST;
              goto fail;
            }
          }
          break;
        case PL_UNIX_SIZE_NUMBER:
          parser->item_length++;
          if(c == ' ') {
            char *p;
            curl_off_t fsize;
            finfo->b_data[parser->item_offset + parser->item_length - 1] = 0;
            if(!curlx_strtoofft(finfo->b_data + parser->item_offset,
                                &p, 10, &fsize)) {
              if(p[0] == '\0' && fsize != CURL_OFF_T_MAX &&
                 fsize != CURL_OFF_T_MIN) {
                parser->file_data->info.flags |= CURLFINFOFLAG_KNOWN_SIZE;
                parser->file_data->info.size = fsize;
              }
              parser->item_length = 0;
              parser->item_offset = 0;
              parser->state.UNIX.main = PL_UNIX_TIME;
              parser->state.UNIX.sub.time = PL_UNIX_TIME_PREPART1;
            }
          }
          else if(!ISDIGIT(c)) {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          break;
        }
        break;
      case PL_UNIX_TIME:
        switch(parser->state.UNIX.sub.time) {
        case PL_UNIX_TIME_PREPART1:
          if(c != ' ') {
            if(ISALNUM(c)) {
              parser->item_offset = finfo->b_used -1;
              parser->item_length = 1;
              parser->state.UNIX.sub.time = PL_UNIX_TIME_PART1;
            }
            else {
              parser->error = CURLE_FTP_BAD_FILE_LIST;
              goto fail;
            }
          }
          break;
        case PL_UNIX_TIME_PART1:
          parser->item_length++;
          if(c == ' ') {
            parser->state.UNIX.sub.time = PL_UNIX_TIME_PREPART2;
          }
          else if(!ISALNUM(c) && c != '.') {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          break;
        case PL_UNIX_TIME_PREPART2:
          parser->item_length++;
          if(c != ' ') {
            if(ISALNUM(c)) {
              parser->state.UNIX.sub.time = PL_UNIX_TIME_PART2;
            }
            else {
              parser->error = CURLE_FTP_BAD_FILE_LIST;
              goto fail;
            }
          }
          break;
        case PL_UNIX_TIME_PART2:
          parser->item_length++;
          if(c == ' ') {
            parser->state.UNIX.sub.time = PL_UNIX_TIME_PREPART3;
          }
          else if(!ISALNUM(c) && c != '.') {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          break;
        case PL_UNIX_TIME_PREPART3:
          parser->item_length++;
          if(c != ' ') {
            if(ISALNUM(c)) {
              parser->state.UNIX.sub.time = PL_UNIX_TIME_PART3;
            }
            else {
              parser->error = CURLE_FTP_BAD_FILE_LIST;
              goto fail;
            }
          }
          break;
        case PL_UNIX_TIME_PART3:
          parser->item_length++;
          if(c == ' ') {
            finfo->b_data[parser->item_offset + parser->item_length -1] = 0;
            parser->offsets.time = parser->item_offset;
            /*
              if(ftp_pl_gettime(parser, finfo->b_data + parser->item_offset)) {
                parser->file_data->flags |= CURLFINFOFLAG_KNOWN_TIME;
              }
            */
            if(finfo->filetype == CURLFILETYPE_SYMLINK) {
              parser->state.UNIX.main = PL_UNIX_SYMLINK;
              parser->state.UNIX.sub.symlink = PL_UNIX_SYMLINK_PRESPACE;
            }
            else {
              parser->state.UNIX.main = PL_UNIX_FILENAME;
              parser->state.UNIX.sub.filename = PL_UNIX_FILENAME_PRESPACE;
            }
          }
          else if(!ISALNUM(c) && c != '.' && c != ':') {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          break;
        }
        break;
      case PL_UNIX_FILENAME:
        switch(parser->state.UNIX.sub.filename) {
        case PL_UNIX_FILENAME_PRESPACE:
          if(c != ' ') {
            parser->item_offset = finfo->b_used - 1;
            parser->item_length = 1;
            parser->state.UNIX.sub.filename = PL_UNIX_FILENAME_NAME;
          }
          break;
        case PL_UNIX_FILENAME_NAME:
          parser->item_length++;
          if(c == '\r') {
            parser->state.UNIX.sub.filename = PL_UNIX_FILENAME_WINDOWSEOL;
          }
          else if(c == '\n') {
            finfo->b_data[parser->item_offset + parser->item_length - 1] = 0;
            parser->offsets.filename = parser->item_offset;
            parser->state.UNIX.main = PL_UNIX_FILETYPE;
            result = ftp_pl_insert_finfo(data, infop);
            if(result) {
              parser->error = result;
              goto fail;
            }
          }
          break;
        case PL_UNIX_FILENAME_WINDOWSEOL:
          if(c == '\n') {
            finfo->b_data[parser->item_offset + parser->item_length - 1] = 0;
            parser->offsets.filename = parser->item_offset;
            parser->state.UNIX.main = PL_UNIX_FILETYPE;
            result = ftp_pl_insert_finfo(data, infop);
            if(result) {
              parser->error = result;
              goto fail;
            }
          }
          else {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          break;
        }
        break;
      case PL_UNIX_SYMLINK:
        switch(parser->state.UNIX.sub.symlink) {
        case PL_UNIX_SYMLINK_PRESPACE:
          if(c != ' ') {
            parser->item_offset = finfo->b_used - 1;
            parser->item_length = 1;
            parser->state.UNIX.sub.symlink = PL_UNIX_SYMLINK_NAME;
          }
          break;
        case PL_UNIX_SYMLINK_NAME:
          parser->item_length++;
          if(c == ' ') {
            parser->state.UNIX.sub.symlink = PL_UNIX_SYMLINK_PRETARGET1;
          }
          else if(c == '\r' || c == '\n') {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          break;
        case PL_UNIX_SYMLINK_PRETARGET1:
          parser->item_length++;
          if(c == '-') {
            parser->state.UNIX.sub.symlink = PL_UNIX_SYMLINK_PRETARGET2;
          }
          else if(c == '\r' || c == '\n') {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          else {
            parser->state.UNIX.sub.symlink = PL_UNIX_SYMLINK_NAME;
          }
          break;
        case PL_UNIX_SYMLINK_PRETARGET2:
          parser->item_length++;
          if(c == '>') {
            parser->state.UNIX.sub.symlink = PL_UNIX_SYMLINK_PRETARGET3;
          }
          else if(c == '\r' || c == '\n') {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          else {
            parser->state.UNIX.sub.symlink = PL_UNIX_SYMLINK_NAME;
          }
          break;
        case PL_UNIX_SYMLINK_PRETARGET3:
          parser->item_length++;
          if(c == ' ') {
            parser->state.UNIX.sub.symlink = PL_UNIX_SYMLINK_PRETARGET4;
            /* now place where is symlink following */
            finfo->b_data[parser->item_offset + parser->item_length - 4] = 0;
            parser->offsets.filename = parser->item_offset;
            parser->item_length = 0;
            parser->item_offset = 0;
          }
          else if(c == '\r' || c == '\n') {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          else {
            parser->state.UNIX.sub.symlink = PL_UNIX_SYMLINK_NAME;
          }
          break;
        case PL_UNIX_SYMLINK_PRETARGET4:
          if(c != '\r' && c != '\n') {
            parser->state.UNIX.sub.symlink = PL_UNIX_SYMLINK_TARGET;
            parser->item_offset = finfo->b_used - 1;
            parser->item_length = 1;
          }
          else {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          break;
        case PL_UNIX_SYMLINK_TARGET:
          parser->item_length++;
          if(c == '\r') {
            parser->state.UNIX.sub.symlink = PL_UNIX_SYMLINK_WINDOWSEOL;
          }
          else if(c == '\n') {
            finfo->b_data[parser->item_offset + parser->item_length - 1] = 0;
            parser->offsets.symlink_target = parser->item_offset;
            result = ftp_pl_insert_finfo(data, infop);
            if(result) {
              parser->error = result;
              goto fail;
            }
            parser->state.UNIX.main = PL_UNIX_FILETYPE;
          }
          break;
        case PL_UNIX_SYMLINK_WINDOWSEOL:
          if(c == '\n') {
            finfo->b_data[parser->item_offset + parser->item_length - 1] = 0;
            parser->offsets.symlink_target = parser->item_offset;
            result = ftp_pl_insert_finfo(data, infop);
            if(result) {
              parser->error = result;
              goto fail;
            }
            parser->state.UNIX.main = PL_UNIX_FILETYPE;
          }
          else {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          break;
        }
        break;
      }
      break;
    case OS_TYPE_WIN_NT:
      switch(parser->state.NT.main) {
      case PL_WINNT_DATE:
        parser->item_length++;
        if(parser->item_length < 9) {
          if(!strchr("0123456789-", c)) { /* only simple control */
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
        }
        else if(parser->item_length == 9) {
          if(c == ' ') {
            parser->state.NT.main = PL_WINNT_TIME;
            parser->state.NT.sub.time = PL_WINNT_TIME_PRESPACE;
          }
          else {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
        }
        else {
          parser->error = CURLE_FTP_BAD_FILE_LIST;
          goto fail;
        }
        break;
      case PL_WINNT_TIME:
        parser->item_length++;
        switch(parser->state.NT.sub.time) {
        case PL_WINNT_TIME_PRESPACE:
          if(!ISSPACE(c)) {
            parser->state.NT.sub.time = PL_WINNT_TIME_TIME;
          }
          break;
        case PL_WINNT_TIME_TIME:
          if(c == ' ') {
            parser->offsets.time = parser->item_offset;
            finfo->b_data[parser->item_offset + parser->item_length -1] = 0;
            parser->state.NT.main = PL_WINNT_DIRORSIZE;
            parser->state.NT.sub.dirorsize = PL_WINNT_DIRORSIZE_PRESPACE;
            parser->item_length = 0;
          }
          else if(!strchr("APM0123456789:", c)) {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          break;
        }
        break;
      case PL_WINNT_DIRORSIZE:
        switch(parser->state.NT.sub.dirorsize) {
        case PL_WINNT_DIRORSIZE_PRESPACE:
          if(c != ' ') {
            parser->item_offset = finfo->b_used - 1;
            parser->item_length = 1;
            parser->state.NT.sub.dirorsize = PL_WINNT_DIRORSIZE_CONTENT;
          }
          break;
        case PL_WINNT_DIRORSIZE_CONTENT:
          parser->item_length ++;
          if(c == ' ') {
            finfo->b_data[parser->item_offset + parser->item_length - 1] = 0;
            if(strcmp("<DIR>", finfo->b_data + parser->item_offset) == 0) {
              finfo->filetype = CURLFILETYPE_DIRECTORY;
              finfo->size = 0;
            }
            else {
              char *endptr;
              if(curlx_strtoofft(finfo->b_data +
                                 parser->item_offset,
                                 &endptr, 10, &finfo->size)) {
                parser->error = CURLE_FTP_BAD_FILE_LIST;
                goto fail;
              }
              /* correct file type */
              parser->file_data->info.filetype = CURLFILETYPE_FILE;
            }

            parser->file_data->info.flags |= CURLFINFOFLAG_KNOWN_SIZE;
            parser->item_length = 0;
            parser->state.NT.main = PL_WINNT_FILENAME;
            parser->state.NT.sub.filename = PL_WINNT_FILENAME_PRESPACE;
          }
          break;
        }
        break;
      case PL_WINNT_FILENAME:
        switch(parser->state.NT.sub.filename) {
        case PL_WINNT_FILENAME_PRESPACE:
          if(c != ' ') {
            parser->item_offset = finfo->b_used -1;
            parser->item_length = 1;
            parser->state.NT.sub.filename = PL_WINNT_FILENAME_CONTENT;
          }
          break;
        case PL_WINNT_FILENAME_CONTENT:
          parser->item_length++;
          if(c == '\r') {
            parser->state.NT.sub.filename = PL_WINNT_FILENAME_WINEOL;
            finfo->b_data[finfo->b_used - 1] = 0;
          }
          else if(c == '\n') {
            parser->offsets.filename = parser->item_offset;
            finfo->b_data[finfo->b_used - 1] = 0;
            result = ftp_pl_insert_finfo(data, infop);
            if(result) {
              parser->error = result;
              goto fail;
            }
            parser->state.NT.main = PL_WINNT_DATE;
            parser->state.NT.sub.filename = PL_WINNT_FILENAME_PRESPACE;
          }
          break;
        case PL_WINNT_FILENAME_WINEOL:
          if(c == '\n') {
            parser->offsets.filename = parser->item_offset;
            result = ftp_pl_insert_finfo(data, infop);
            if(result) {
              parser->error = result;
              goto fail;
            }
            parser->state.NT.main = PL_WINNT_DATE;
            parser->state.NT.sub.filename = PL_WINNT_FILENAME_PRESPACE;
          }
          else {
            parser->error = CURLE_FTP_BAD_FILE_LIST;
            goto fail;
          }
          break;
        }
        break;
      }
      break;
    default:
      retsize = bufflen + 1;
      goto fail;
    }

    i++;
  }
  return retsize;

fail:

  /* Clean up any allocated memory. */
  if(parser->file_data) {
    Curl_fileinfo_cleanup(parser->file_data);
    parser->file_data = NULL;
  }

  return retsize;
}

#endif /* CURL_DISABLE_FTP */
