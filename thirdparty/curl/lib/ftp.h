#ifndef HEADER_CURL_FTP_H
#define HEADER_CURL_FTP_H
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

#include "pingpong.h"

#ifndef CURL_DISABLE_FTP
extern const struct Curl_handler Curl_handler_ftp;

#ifdef USE_SSL
extern const struct Curl_handler Curl_handler_ftps;
#endif

CURLcode Curl_GetFTPResponse(struct Curl_easy *data, ssize_t *nread,
                             int *ftpcode);
#endif /* CURL_DISABLE_FTP */

/****************************************************************************
 * FTP unique setup
 ***************************************************************************/
typedef enum {
  FTP_STOP,    /* do nothing state, stops the state machine */
  FTP_WAIT220, /* waiting for the initial 220 response immediately after
                  a connect */
  FTP_AUTH,
  FTP_USER,
  FTP_PASS,
  FTP_ACCT,
  FTP_PBSZ,
  FTP_PROT,
  FTP_CCC,
  FTP_PWD,
  FTP_SYST,
  FTP_NAMEFMT,
  FTP_QUOTE, /* waiting for a response to a command sent in a quote list */
  FTP_RETR_PREQUOTE,
  FTP_STOR_PREQUOTE,
  FTP_POSTQUOTE,
  FTP_CWD,  /* change dir */
  FTP_MKD,  /* if the dir didn't exist */
  FTP_MDTM, /* to figure out the datestamp */
  FTP_TYPE, /* to set type when doing a head-like request */
  FTP_LIST_TYPE, /* set type when about to do a dir list */
  FTP_RETR_TYPE, /* set type when about to RETR a file */
  FTP_STOR_TYPE, /* set type when about to STOR a file */
  FTP_SIZE, /* get the remote file's size for head-like request */
  FTP_RETR_SIZE, /* get the remote file's size for RETR */
  FTP_STOR_SIZE, /* get the size for STOR */
  FTP_REST, /* when used to check if the server supports it in head-like */
  FTP_RETR_REST, /* when asking for "resume" in for RETR */
  FTP_PORT, /* generic state for PORT, LPRT and EPRT, check count1 */
  FTP_PRET, /* generic state for PRET RETR, PRET STOR and PRET LIST/NLST */
  FTP_PASV, /* generic state for PASV and EPSV, check count1 */
  FTP_LIST, /* generic state for LIST, NLST or a custom list command */
  FTP_RETR,
  FTP_STOR, /* generic state for STOR and APPE */
  FTP_QUIT,
  FTP_LAST  /* never used */
} ftpstate;

struct ftp_parselist_data; /* defined later in ftplistparser.c */

struct ftp_wc {
  struct ftp_parselist_data *parser;

  struct {
    curl_write_callback write_function;
    FILE *file_descriptor;
  } backup;
};

typedef enum {
  FTPFILE_MULTICWD  = 1, /* as defined by RFC1738 */
  FTPFILE_NOCWD     = 2, /* use SIZE / RETR / STOR on the full path */
  FTPFILE_SINGLECWD = 3  /* make one CWD, then SIZE / RETR / STOR on the
                            file */
} curl_ftpfile;

/* This FTP struct is used in the Curl_easy. All FTP data that is
   connection-oriented must be in FTP_conn to properly deal with the fact that
   perhaps the Curl_easy is changed between the times the connection is
   used. */
struct FTP {
  char *path;    /* points to the urlpieces struct field */
  char *pathalloc; /* if non-NULL a pointer to an allocated path */

  /* transfer a file/body or not, done as a typedefed enum just to make
     debuggers display the full symbol and not just the numerical value */
  curl_pp_transfer transfer;
  curl_off_t downloadsize;
};


/* ftp_conn is used for struct connection-oriented data in the connectdata
   struct */
struct ftp_conn {
  struct pingpong pp;
  char *entrypath; /* the PWD reply when we logged on */
  char *file;    /* url-decoded file name (or path) */
  char **dirs;   /* realloc()ed array for path components */
  int dirdepth;  /* number of entries used in the 'dirs' array */
  bool dont_check;  /* Set to TRUE to prevent the final (post-transfer)
                       file size and 226/250 status check. It should still
                       read the line, just ignore the result. */
  bool ctl_valid;   /* Tells Curl_ftp_quit() whether or not to do anything. If
                       the connection has timed out or been closed, this
                       should be FALSE when it gets to Curl_ftp_quit() */
  bool cwddone;     /* if it has been determined that the proper CWD combo
                       already has been done */
  int cwdcount;     /* number of CWD commands issued */
  bool cwdfail;     /* set TRUE if a CWD command fails, as then we must prevent
                       caching the current directory */
  bool wait_data_conn; /* this is set TRUE if data connection is waited */
  /* newhost is the (allocated) IP addr or host name to connect the data
     connection to */
  unsigned short newport;
  char *newhost;
  char *prevpath;   /* url-decoded conn->path from the previous transfer */
  char transfertype; /* set by ftp_transfertype for use by Curl_client_write()a
                        and others (A/I or zero) */
  int count1; /* general purpose counter for the state machine */
  int count2; /* general purpose counter for the state machine */
  int count3; /* general purpose counter for the state machine */
  ftpstate state; /* always use ftp.c:state() to change state! */
  ftpstate state_saved; /* transfer type saved to be reloaded after
                           data connection is established */
  curl_off_t retr_size_saved; /* Size of retrieved file saved */
  char *server_os;     /* The target server operating system. */
  curl_off_t known_filesize; /* file size is different from -1, if wildcard
                                LIST parsing was done and wc_statemach set
                                it */
};

#define DEFAULT_ACCEPT_TIMEOUT   60000 /* milliseconds == one minute */

#endif /* HEADER_CURL_FTP_H */
