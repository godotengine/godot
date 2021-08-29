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

#include "curl_setup.h"

#ifndef CURL_DISABLE_FILE

#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifdef HAVE_NET_IF_H
#include <net/if.h>
#endif
#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#include "strtoofft.h"
#include "urldata.h"
#include <curl/curl.h>
#include "progress.h"
#include "sendf.h"
#include "escape.h"
#include "file.h"
#include "speedcheck.h"
#include "getinfo.h"
#include "transfer.h"
#include "url.h"
#include "parsedate.h" /* for the week day and month names */
#include "warnless.h"
#include "curl_range.h"
/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

#if defined(WIN32) || defined(MSDOS) || defined(__EMX__)
#define DOS_FILESYSTEM 1
#endif

#ifdef OPEN_NEEDS_ARG3
#  define open_readonly(p,f) open((p),(f),(0))
#else
#  define open_readonly(p,f) open((p),(f))
#endif

/*
 * Forward declarations.
 */

static CURLcode file_do(struct Curl_easy *data, bool *done);
static CURLcode file_done(struct Curl_easy *data,
                          CURLcode status, bool premature);
static CURLcode file_connect(struct Curl_easy *data, bool *done);
static CURLcode file_disconnect(struct Curl_easy *data,
                                struct connectdata *conn,
                                bool dead_connection);
static CURLcode file_setup_connection(struct Curl_easy *data,
                                      struct connectdata *conn);

/*
 * FILE scheme handler.
 */

const struct Curl_handler Curl_handler_file = {
  "FILE",                               /* scheme */
  file_setup_connection,                /* setup_connection */
  file_do,                              /* do_it */
  file_done,                            /* done */
  ZERO_NULL,                            /* do_more */
  file_connect,                         /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  ZERO_NULL,                            /* proto_getsock */
  ZERO_NULL,                            /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  file_disconnect,                      /* disconnect */
  ZERO_NULL,                            /* readwrite */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  0,                                    /* defport */
  CURLPROTO_FILE,                       /* protocol */
  CURLPROTO_FILE,                       /* family */
  PROTOPT_NONETWORK | PROTOPT_NOURLQUERY /* flags */
};


static CURLcode file_setup_connection(struct Curl_easy *data,
                                      struct connectdata *conn)
{
  (void)conn;
  /* allocate the FILE specific struct */
  data->req.p.file = calloc(1, sizeof(struct FILEPROTO));
  if(!data->req.p.file)
    return CURLE_OUT_OF_MEMORY;

  return CURLE_OK;
}

/*
 * file_connect() gets called from Curl_protocol_connect() to allow us to
 * do protocol-specific actions at connect-time.  We emulate a
 * connect-then-transfer protocol and "connect" to the file here
 */
static CURLcode file_connect(struct Curl_easy *data, bool *done)
{
  char *real_path;
  struct FILEPROTO *file = data->req.p.file;
  int fd;
#ifdef DOS_FILESYSTEM
  size_t i;
  char *actual_path;
#endif
  size_t real_path_len;

  CURLcode result = Curl_urldecode(data, data->state.up.path, 0, &real_path,
                                   &real_path_len, REJECT_ZERO);
  if(result)
    return result;

#ifdef DOS_FILESYSTEM
  /* If the first character is a slash, and there's
     something that looks like a drive at the beginning of
     the path, skip the slash.  If we remove the initial
     slash in all cases, paths without drive letters end up
     relative to the current directory which isn't how
     browsers work.

     Some browsers accept | instead of : as the drive letter
     separator, so we do too.

     On other platforms, we need the slash to indicate an
     absolute pathname.  On Windows, absolute paths start
     with a drive letter.
  */
  actual_path = real_path;
  if((actual_path[0] == '/') &&
      actual_path[1] &&
     (actual_path[2] == ':' || actual_path[2] == '|')) {
    actual_path[2] = ':';
    actual_path++;
    real_path_len--;
  }

  /* change path separators from '/' to '\\' for DOS, Windows and OS/2 */
  for(i = 0; i < real_path_len; ++i)
    if(actual_path[i] == '/')
      actual_path[i] = '\\';
    else if(!actual_path[i]) { /* binary zero */
      Curl_safefree(real_path);
      return CURLE_URL_MALFORMAT;
    }

  fd = open_readonly(actual_path, O_RDONLY|O_BINARY);
  file->path = actual_path;
#else
  if(memchr(real_path, 0, real_path_len)) {
    /* binary zeroes indicate foul play */
    Curl_safefree(real_path);
    return CURLE_URL_MALFORMAT;
  }

  fd = open_readonly(real_path, O_RDONLY);
  file->path = real_path;
#endif
  file->freepath = real_path; /* free this when done */

  file->fd = fd;
  if(!data->set.upload && (fd == -1)) {
    failf(data, "Couldn't open file %s", data->state.up.path);
    file_done(data, CURLE_FILE_COULDNT_READ_FILE, FALSE);
    return CURLE_FILE_COULDNT_READ_FILE;
  }
  *done = TRUE;

  return CURLE_OK;
}

static CURLcode file_done(struct Curl_easy *data,
                          CURLcode status, bool premature)
{
  struct FILEPROTO *file = data->req.p.file;
  (void)status; /* not used */
  (void)premature; /* not used */

  if(file) {
    Curl_safefree(file->freepath);
    file->path = NULL;
    if(file->fd != -1)
      close(file->fd);
    file->fd = -1;
  }

  return CURLE_OK;
}

static CURLcode file_disconnect(struct Curl_easy *data,
                                struct connectdata *conn,
                                bool dead_connection)
{
  (void)dead_connection; /* not used */
  (void)conn;
  return file_done(data, 0, 0);
}

#ifdef DOS_FILESYSTEM
#define DIRSEP '\\'
#else
#define DIRSEP '/'
#endif

static CURLcode file_upload(struct Curl_easy *data)
{
  struct FILEPROTO *file = data->req.p.file;
  const char *dir = strchr(file->path, DIRSEP);
  int fd;
  int mode;
  CURLcode result = CURLE_OK;
  char *buf = data->state.buffer;
  curl_off_t bytecount = 0;
  struct_stat file_stat;
  const char *buf2;

  /*
   * Since FILE: doesn't do the full init, we need to provide some extra
   * assignments here.
   */
  data->req.upload_fromhere = buf;

  if(!dir)
    return CURLE_FILE_COULDNT_READ_FILE; /* fix: better error code */

  if(!dir[1])
    return CURLE_FILE_COULDNT_READ_FILE; /* fix: better error code */

#ifdef O_BINARY
#define MODE_DEFAULT O_WRONLY|O_CREAT|O_BINARY
#else
#define MODE_DEFAULT O_WRONLY|O_CREAT
#endif

  if(data->state.resume_from)
    mode = MODE_DEFAULT|O_APPEND;
  else
    mode = MODE_DEFAULT|O_TRUNC;

  fd = open(file->path, mode, data->set.new_file_perms);
  if(fd < 0) {
    failf(data, "Can't open %s for writing", file->path);
    return CURLE_WRITE_ERROR;
  }

  if(-1 != data->state.infilesize)
    /* known size of data to "upload" */
    Curl_pgrsSetUploadSize(data, data->state.infilesize);

  /* treat the negative resume offset value as the case of "-" */
  if(data->state.resume_from < 0) {
    if(fstat(fd, &file_stat)) {
      close(fd);
      failf(data, "Can't get the size of %s", file->path);
      return CURLE_WRITE_ERROR;
    }
    data->state.resume_from = (curl_off_t)file_stat.st_size;
  }

  while(!result) {
    size_t nread;
    size_t nwrite;
    size_t readcount;
    result = Curl_fillreadbuffer(data, data->set.buffer_size, &readcount);
    if(result)
      break;

    if(!readcount)
      break;

    nread = readcount;

    /*skip bytes before resume point*/
    if(data->state.resume_from) {
      if((curl_off_t)nread <= data->state.resume_from) {
        data->state.resume_from -= nread;
        nread = 0;
        buf2 = buf;
      }
      else {
        buf2 = buf + data->state.resume_from;
        nread -= (size_t)data->state.resume_from;
        data->state.resume_from = 0;
      }
    }
    else
      buf2 = buf;

    /* write the data to the target */
    nwrite = write(fd, buf2, nread);
    if(nwrite != nread) {
      result = CURLE_SEND_ERROR;
      break;
    }

    bytecount += nread;

    Curl_pgrsSetUploadCounter(data, bytecount);

    if(Curl_pgrsUpdate(data))
      result = CURLE_ABORTED_BY_CALLBACK;
    else
      result = Curl_speedcheck(data, Curl_now());
  }
  if(!result && Curl_pgrsUpdate(data))
    result = CURLE_ABORTED_BY_CALLBACK;

  close(fd);

  return result;
}

/*
 * file_do() is the protocol-specific function for the do-phase, separated
 * from the connect-phase above. Other protocols merely setup the transfer in
 * the do-phase, to have it done in the main transfer loop but since some
 * platforms we support don't allow select()ing etc on file handles (as
 * opposed to sockets) we instead perform the whole do-operation in this
 * function.
 */
static CURLcode file_do(struct Curl_easy *data, bool *done)
{
  /* This implementation ignores the host name in conformance with
     RFC 1738. Only local files (reachable via the standard file system)
     are supported. This means that files on remotely mounted directories
     (via NFS, Samba, NT sharing) can be accessed through a file:// URL
  */
  CURLcode result = CURLE_OK;
  struct_stat statbuf; /* struct_stat instead of struct stat just to allow the
                          Windows version to have a different struct without
                          having to redefine the simple word 'stat' */
  curl_off_t expected_size = -1;
  bool size_known;
  bool fstated = FALSE;
  char *buf = data->state.buffer;
  curl_off_t bytecount = 0;
  int fd;
  struct FILEPROTO *file;

  *done = TRUE; /* unconditionally */

  Curl_pgrsStartNow(data);

  if(data->set.upload)
    return file_upload(data);

  file = data->req.p.file;

  /* get the fd from the connection phase */
  fd = file->fd;

  /* VMS: This only works reliable for STREAMLF files */
  if(-1 != fstat(fd, &statbuf)) {
    if(!S_ISDIR(statbuf.st_mode))
      expected_size = statbuf.st_size;
    /* and store the modification time */
    data->info.filetime = statbuf.st_mtime;
    fstated = TRUE;
  }

  if(fstated && !data->state.range && data->set.timecondition) {
    if(!Curl_meets_timecondition(data, data->info.filetime)) {
      *done = TRUE;
      return CURLE_OK;
    }
  }

  if(fstated) {
    time_t filetime;
    struct tm buffer;
    const struct tm *tm = &buffer;
    char header[80];
    int headerlen;
    char accept_ranges[24]= { "Accept-ranges: bytes\r\n" };
    if(expected_size >= 0) {
      headerlen = msnprintf(header, sizeof(header),
                "Content-Length: %" CURL_FORMAT_CURL_OFF_T "\r\n",
                expected_size);
      result = Curl_client_write(data, CLIENTWRITE_HEADER, header, headerlen);
      if(result)
        return result;

      result = Curl_client_write(data, CLIENTWRITE_HEADER,
                                 accept_ranges, strlen(accept_ranges));
      if(result != CURLE_OK)
        return result;
    }

    filetime = (time_t)statbuf.st_mtime;
    result = Curl_gmtime(filetime, &buffer);
    if(result)
      return result;

    /* format: "Tue, 15 Nov 1994 12:45:26 GMT" */
    headerlen = msnprintf(header, sizeof(header),
              "Last-Modified: %s, %02d %s %4d %02d:%02d:%02d GMT\r\n%s",
              Curl_wkday[tm->tm_wday?tm->tm_wday-1:6],
              tm->tm_mday,
              Curl_month[tm->tm_mon],
              tm->tm_year + 1900,
              tm->tm_hour,
              tm->tm_min,
              tm->tm_sec,
              data->set.opt_no_body ? "": "\r\n");
    result = Curl_client_write(data, CLIENTWRITE_HEADER, header, headerlen);
    if(result)
      return result;
    /* set the file size to make it available post transfer */
    Curl_pgrsSetDownloadSize(data, expected_size);
    if(data->set.opt_no_body)
      return result;
  }

  /* Check whether file range has been specified */
  result = Curl_range(data);
  if(result)
    return result;

  /* Adjust the start offset in case we want to get the N last bytes
   * of the stream if the filesize could be determined */
  if(data->state.resume_from < 0) {
    if(!fstated) {
      failf(data, "Can't get the size of file.");
      return CURLE_READ_ERROR;
    }
    data->state.resume_from += (curl_off_t)statbuf.st_size;
  }

  if(data->state.resume_from > 0) {
    /* We check explicitly if we have a start offset, because
     * expected_size may be -1 if we don't know how large the file is,
     * in which case we should not adjust it. */
    if(data->state.resume_from <= expected_size)
      expected_size -= data->state.resume_from;
    else {
      failf(data, "failed to resume file:// transfer");
      return CURLE_BAD_DOWNLOAD_RESUME;
    }
  }

  /* A high water mark has been specified so we obey... */
  if(data->req.maxdownload > 0)
    expected_size = data->req.maxdownload;

  if(!fstated || (expected_size <= 0))
    size_known = FALSE;
  else
    size_known = TRUE;

  /* The following is a shortcut implementation of file reading
     this is both more efficient than the former call to download() and
     it avoids problems with select() and recv() on file descriptors
     in Winsock */
  if(size_known)
    Curl_pgrsSetDownloadSize(data, expected_size);

  if(data->state.resume_from) {
    if(data->state.resume_from !=
       lseek(fd, data->state.resume_from, SEEK_SET))
      return CURLE_BAD_DOWNLOAD_RESUME;
  }

  Curl_pgrsTime(data, TIMER_STARTTRANSFER);

  while(!result) {
    ssize_t nread;
    /* Don't fill a whole buffer if we want less than all data */
    size_t bytestoread;

    if(size_known) {
      bytestoread = (expected_size < data->set.buffer_size) ?
        curlx_sotouz(expected_size) : (size_t)data->set.buffer_size;
    }
    else
      bytestoread = data->set.buffer_size-1;

    nread = read(fd, buf, bytestoread);

    if(nread > 0)
      buf[nread] = 0;

    if(nread <= 0 || (size_known && (expected_size == 0)))
      break;

    bytecount += nread;
    if(size_known)
      expected_size -= nread;

    result = Curl_client_write(data, CLIENTWRITE_BODY, buf, nread);
    if(result)
      return result;

    Curl_pgrsSetDownloadCounter(data, bytecount);

    if(Curl_pgrsUpdate(data))
      result = CURLE_ABORTED_BY_CALLBACK;
    else
      result = Curl_speedcheck(data, Curl_now());
  }
  if(Curl_pgrsUpdate(data))
    result = CURLE_ABORTED_BY_CALLBACK;

  return result;
}

#endif
