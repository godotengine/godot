#ifndef CURLINC_CURL_H
#define CURLINC_CURL_H
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

/*
 * If you have libcurl problems, all docs and details are found here:
 *   https://curl.se/libcurl/
 */

#ifdef CURL_NO_OLDIES
#define CURL_STRICTER
#endif

#include "curlver.h"         /* libcurl version defines   */
#include "system.h"          /* determine things run-time */

/*
 * Define CURL_WIN32 when build target is Win32 API
 */

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32)) &&        \
  !defined(__SYMBIAN32__)
#define CURL_WIN32
#endif

#include <stdio.h>
#include <limits.h>

#if (defined(__FreeBSD__) && (__FreeBSD__ >= 2)) || defined(__MidnightBSD__)
/* Needed for __FreeBSD_version or __MidnightBSD_version symbol definition */
#include <osreldate.h>
#endif

/* The include stuff here below is mainly for time_t! */
#include <sys/types.h>
#include <time.h>

#if defined(CURL_WIN32) && !defined(_WIN32_WCE) && !defined(__CYGWIN__)
#if !(defined(_WINSOCKAPI_) || defined(_WINSOCK_H) || \
      defined(__LWIP_OPT_H__) || defined(LWIP_HDR_OPT_H))
/* The check above prevents the winsock2 inclusion if winsock.h already was
   included, since they can't co-exist without problems */
#include <winsock2.h>
#include <ws2tcpip.h>
#endif
#endif

/* HP-UX systems version 9, 10 and 11 lack sys/select.h and so does oldish
   libc5-based Linux systems. Only include it on systems that are known to
   require it! */
#if defined(_AIX) || defined(__NOVELL_LIBC__) || defined(__NetBSD__) || \
    defined(__minix) || defined(__SYMBIAN32__) || defined(__INTEGRITY) || \
    defined(ANDROID) || defined(__ANDROID__) || defined(__OpenBSD__) || \
    defined(__CYGWIN__) || defined(AMIGA) || defined(__NuttX__) || \
   (defined(__FreeBSD_version) && (__FreeBSD_version < 800000)) || \
   (defined(__MidnightBSD_version) && (__MidnightBSD_version < 100000)) || \
    defined(__VXWORKS__)
#include <sys/select.h>
#endif

#if !defined(CURL_WIN32) && !defined(_WIN32_WCE)
#include <sys/socket.h>
#endif

#if !defined(CURL_WIN32) && !defined(__WATCOMC__) && !defined(__VXWORKS__)
#include <sys/time.h>
#endif

#ifdef __BEOS__
#include <support/SupportDefs.h>
#endif

/* Compatibility for non-Clang compilers */
#ifndef __has_declspec_attribute
#  define __has_declspec_attribute(x) 0
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#if defined(BUILDING_LIBCURL) || defined(CURL_STRICTER)
typedef struct Curl_easy CURL;
typedef struct Curl_share CURLSH;
#else
typedef void CURL;
typedef void CURLSH;
#endif

/*
 * libcurl external API function linkage decorations.
 */

#ifdef CURL_STATICLIB
#  define CURL_EXTERN
#elif defined(CURL_WIN32) || defined(__SYMBIAN32__) || \
     (__has_declspec_attribute(dllexport) && \
      __has_declspec_attribute(dllimport))
#  if defined(BUILDING_LIBCURL)
#    define CURL_EXTERN  __declspec(dllexport)
#  else
#    define CURL_EXTERN  __declspec(dllimport)
#  endif
#elif defined(BUILDING_LIBCURL) && defined(CURL_HIDDEN_SYMBOLS)
#  define CURL_EXTERN CURL_EXTERN_SYMBOL
#else
#  define CURL_EXTERN
#endif

#ifndef curl_socket_typedef
/* socket typedef */
#if defined(CURL_WIN32) && !defined(__LWIP_OPT_H__) && !defined(LWIP_HDR_OPT_H)
typedef SOCKET curl_socket_t;
#define CURL_SOCKET_BAD INVALID_SOCKET
#else
typedef int curl_socket_t;
#define CURL_SOCKET_BAD -1
#endif
#define curl_socket_typedef
#endif /* curl_socket_typedef */

/* enum for the different supported SSL backends */
typedef enum {
  CURLSSLBACKEND_NONE = 0,
  CURLSSLBACKEND_OPENSSL = 1,
  CURLSSLBACKEND_GNUTLS = 2,
  CURLSSLBACKEND_NSS = 3,
  CURLSSLBACKEND_OBSOLETE4 = 4,  /* Was QSOSSL. */
  CURLSSLBACKEND_GSKIT = 5,
  CURLSSLBACKEND_POLARSSL = 6,
  CURLSSLBACKEND_WOLFSSL = 7,
  CURLSSLBACKEND_SCHANNEL = 8,
  CURLSSLBACKEND_SECURETRANSPORT = 9,
  CURLSSLBACKEND_AXTLS = 10, /* never used since 7.63.0 */
  CURLSSLBACKEND_MBEDTLS = 11,
  CURLSSLBACKEND_MESALINK = 12,
  CURLSSLBACKEND_BEARSSL = 13,
  CURLSSLBACKEND_RUSTLS = 14
} curl_sslbackend;

/* aliases for library clones and renames */
#define CURLSSLBACKEND_LIBRESSL CURLSSLBACKEND_OPENSSL
#define CURLSSLBACKEND_BORINGSSL CURLSSLBACKEND_OPENSSL

/* deprecated names: */
#define CURLSSLBACKEND_CYASSL CURLSSLBACKEND_WOLFSSL
#define CURLSSLBACKEND_DARWINSSL CURLSSLBACKEND_SECURETRANSPORT

struct curl_httppost {
  struct curl_httppost *next;       /* next entry in the list */
  char *name;                       /* pointer to allocated name */
  long namelength;                  /* length of name length */
  char *contents;                   /* pointer to allocated data contents */
  long contentslength;              /* length of contents field, see also
                                       CURL_HTTPPOST_LARGE */
  char *buffer;                     /* pointer to allocated buffer contents */
  long bufferlength;                /* length of buffer field */
  char *contenttype;                /* Content-Type */
  struct curl_slist *contentheader; /* list of extra headers for this form */
  struct curl_httppost *more;       /* if one field name has more than one
                                       file, this link should link to following
                                       files */
  long flags;                       /* as defined below */

/* specified content is a file name */
#define CURL_HTTPPOST_FILENAME (1<<0)
/* specified content is a file name */
#define CURL_HTTPPOST_READFILE (1<<1)
/* name is only stored pointer do not free in formfree */
#define CURL_HTTPPOST_PTRNAME (1<<2)
/* contents is only stored pointer do not free in formfree */
#define CURL_HTTPPOST_PTRCONTENTS (1<<3)
/* upload file from buffer */
#define CURL_HTTPPOST_BUFFER (1<<4)
/* upload file from pointer contents */
#define CURL_HTTPPOST_PTRBUFFER (1<<5)
/* upload file contents by using the regular read callback to get the data and
   pass the given pointer as custom pointer */
#define CURL_HTTPPOST_CALLBACK (1<<6)
/* use size in 'contentlen', added in 7.46.0 */
#define CURL_HTTPPOST_LARGE (1<<7)

  char *showfilename;               /* The file name to show. If not set, the
                                       actual file name will be used (if this
                                       is a file part) */
  void *userp;                      /* custom pointer used for
                                       HTTPPOST_CALLBACK posts */
  curl_off_t contentlen;            /* alternative length of contents
                                       field. Used if CURL_HTTPPOST_LARGE is
                                       set. Added in 7.46.0 */
};


/* This is a return code for the progress callback that, when returned, will
   signal libcurl to continue executing the default progress function */
#define CURL_PROGRESSFUNC_CONTINUE 0x10000001

/* This is the CURLOPT_PROGRESSFUNCTION callback prototype. It is now
   considered deprecated but was the only choice up until 7.31.0 */
typedef int (*curl_progress_callback)(void *clientp,
                                      double dltotal,
                                      double dlnow,
                                      double ultotal,
                                      double ulnow);

/* This is the CURLOPT_XFERINFOFUNCTION callback prototype. It was introduced
   in 7.32.0, avoids the use of floating point numbers and provides more
   detailed information. */
typedef int (*curl_xferinfo_callback)(void *clientp,
                                      curl_off_t dltotal,
                                      curl_off_t dlnow,
                                      curl_off_t ultotal,
                                      curl_off_t ulnow);

#ifndef CURL_MAX_READ_SIZE
  /* The maximum receive buffer size configurable via CURLOPT_BUFFERSIZE. */
#define CURL_MAX_READ_SIZE 524288
#endif

#ifndef CURL_MAX_WRITE_SIZE
  /* Tests have proven that 20K is a very bad buffer size for uploads on
     Windows, while 16K for some odd reason performed a lot better.
     We do the ifndef check to allow this value to easier be changed at build
     time for those who feel adventurous. The practical minimum is about
     400 bytes since libcurl uses a buffer of this size as a scratch area
     (unrelated to network send operations). */
#define CURL_MAX_WRITE_SIZE 16384
#endif

#ifndef CURL_MAX_HTTP_HEADER
/* The only reason to have a max limit for this is to avoid the risk of a bad
   server feeding libcurl with a never-ending header that will cause reallocs
   infinitely */
#define CURL_MAX_HTTP_HEADER (100*1024)
#endif

/* This is a magic return code for the write callback that, when returned,
   will signal libcurl to pause receiving on the current transfer. */
#define CURL_WRITEFUNC_PAUSE 0x10000001

typedef size_t (*curl_write_callback)(char *buffer,
                                      size_t size,
                                      size_t nitems,
                                      void *outstream);

/* This callback will be called when a new resolver request is made */
typedef int (*curl_resolver_start_callback)(void *resolver_state,
                                            void *reserved, void *userdata);

/* enumeration of file types */
typedef enum {
  CURLFILETYPE_FILE = 0,
  CURLFILETYPE_DIRECTORY,
  CURLFILETYPE_SYMLINK,
  CURLFILETYPE_DEVICE_BLOCK,
  CURLFILETYPE_DEVICE_CHAR,
  CURLFILETYPE_NAMEDPIPE,
  CURLFILETYPE_SOCKET,
  CURLFILETYPE_DOOR, /* is possible only on Sun Solaris now */

  CURLFILETYPE_UNKNOWN /* should never occur */
} curlfiletype;

#define CURLFINFOFLAG_KNOWN_FILENAME    (1<<0)
#define CURLFINFOFLAG_KNOWN_FILETYPE    (1<<1)
#define CURLFINFOFLAG_KNOWN_TIME        (1<<2)
#define CURLFINFOFLAG_KNOWN_PERM        (1<<3)
#define CURLFINFOFLAG_KNOWN_UID         (1<<4)
#define CURLFINFOFLAG_KNOWN_GID         (1<<5)
#define CURLFINFOFLAG_KNOWN_SIZE        (1<<6)
#define CURLFINFOFLAG_KNOWN_HLINKCOUNT  (1<<7)

/* Information about a single file, used when doing FTP wildcard matching */
struct curl_fileinfo {
  char *filename;
  curlfiletype filetype;
  time_t time; /* always zero! */
  unsigned int perm;
  int uid;
  int gid;
  curl_off_t size;
  long int hardlinks;

  struct {
    /* If some of these fields is not NULL, it is a pointer to b_data. */
    char *time;
    char *perm;
    char *user;
    char *group;
    char *target; /* pointer to the target filename of a symlink */
  } strings;

  unsigned int flags;

  /* used internally */
  char *b_data;
  size_t b_size;
  size_t b_used;
};

/* return codes for CURLOPT_CHUNK_BGN_FUNCTION */
#define CURL_CHUNK_BGN_FUNC_OK      0
#define CURL_CHUNK_BGN_FUNC_FAIL    1 /* tell the lib to end the task */
#define CURL_CHUNK_BGN_FUNC_SKIP    2 /* skip this chunk over */

/* if splitting of data transfer is enabled, this callback is called before
   download of an individual chunk started. Note that parameter "remains" works
   only for FTP wildcard downloading (for now), otherwise is not used */
typedef long (*curl_chunk_bgn_callback)(const void *transfer_info,
                                        void *ptr,
                                        int remains);

/* return codes for CURLOPT_CHUNK_END_FUNCTION */
#define CURL_CHUNK_END_FUNC_OK      0
#define CURL_CHUNK_END_FUNC_FAIL    1 /* tell the lib to end the task */

/* If splitting of data transfer is enabled this callback is called after
   download of an individual chunk finished.
   Note! After this callback was set then it have to be called FOR ALL chunks.
   Even if downloading of this chunk was skipped in CHUNK_BGN_FUNC.
   This is the reason why we don't need "transfer_info" parameter in this
   callback and we are not interested in "remains" parameter too. */
typedef long (*curl_chunk_end_callback)(void *ptr);

/* return codes for FNMATCHFUNCTION */
#define CURL_FNMATCHFUNC_MATCH    0 /* string corresponds to the pattern */
#define CURL_FNMATCHFUNC_NOMATCH  1 /* pattern doesn't match the string */
#define CURL_FNMATCHFUNC_FAIL     2 /* an error occurred */

/* callback type for wildcard downloading pattern matching. If the
   string matches the pattern, return CURL_FNMATCHFUNC_MATCH value, etc. */
typedef int (*curl_fnmatch_callback)(void *ptr,
                                     const char *pattern,
                                     const char *string);

/* These are the return codes for the seek callbacks */
#define CURL_SEEKFUNC_OK       0
#define CURL_SEEKFUNC_FAIL     1 /* fail the entire transfer */
#define CURL_SEEKFUNC_CANTSEEK 2 /* tell libcurl seeking can't be done, so
                                    libcurl might try other means instead */
typedef int (*curl_seek_callback)(void *instream,
                                  curl_off_t offset,
                                  int origin); /* 'whence' */

/* This is a return code for the read callback that, when returned, will
   signal libcurl to immediately abort the current transfer. */
#define CURL_READFUNC_ABORT 0x10000000
/* This is a return code for the read callback that, when returned, will
   signal libcurl to pause sending data on the current transfer. */
#define CURL_READFUNC_PAUSE 0x10000001

/* Return code for when the trailing headers' callback has terminated
   without any errors*/
#define CURL_TRAILERFUNC_OK 0
/* Return code for when was an error in the trailing header's list and we
  want to abort the request */
#define CURL_TRAILERFUNC_ABORT 1

typedef size_t (*curl_read_callback)(char *buffer,
                                      size_t size,
                                      size_t nitems,
                                      void *instream);

typedef int (*curl_trailer_callback)(struct curl_slist **list,
                                      void *userdata);

typedef enum {
  CURLSOCKTYPE_IPCXN,  /* socket created for a specific IP connection */
  CURLSOCKTYPE_ACCEPT, /* socket created by accept() call */
  CURLSOCKTYPE_LAST    /* never use */
} curlsocktype;

/* The return code from the sockopt_callback can signal information back
   to libcurl: */
#define CURL_SOCKOPT_OK 0
#define CURL_SOCKOPT_ERROR 1 /* causes libcurl to abort and return
                                CURLE_ABORTED_BY_CALLBACK */
#define CURL_SOCKOPT_ALREADY_CONNECTED 2

typedef int (*curl_sockopt_callback)(void *clientp,
                                     curl_socket_t curlfd,
                                     curlsocktype purpose);

struct curl_sockaddr {
  int family;
  int socktype;
  int protocol;
  unsigned int addrlen; /* addrlen was a socklen_t type before 7.18.0 but it
                           turned really ugly and painful on the systems that
                           lack this type */
  struct sockaddr addr;
};

typedef curl_socket_t
(*curl_opensocket_callback)(void *clientp,
                            curlsocktype purpose,
                            struct curl_sockaddr *address);

typedef int
(*curl_closesocket_callback)(void *clientp, curl_socket_t item);

typedef enum {
  CURLIOE_OK,            /* I/O operation successful */
  CURLIOE_UNKNOWNCMD,    /* command was unknown to callback */
  CURLIOE_FAILRESTART,   /* failed to restart the read */
  CURLIOE_LAST           /* never use */
} curlioerr;

typedef enum {
  CURLIOCMD_NOP,         /* no operation */
  CURLIOCMD_RESTARTREAD, /* restart the read stream from start */
  CURLIOCMD_LAST         /* never use */
} curliocmd;

typedef curlioerr (*curl_ioctl_callback)(CURL *handle,
                                         int cmd,
                                         void *clientp);

#ifndef CURL_DID_MEMORY_FUNC_TYPEDEFS
/*
 * The following typedef's are signatures of malloc, free, realloc, strdup and
 * calloc respectively.  Function pointers of these types can be passed to the
 * curl_global_init_mem() function to set user defined memory management
 * callback routines.
 */
typedef void *(*curl_malloc_callback)(size_t size);
typedef void (*curl_free_callback)(void *ptr);
typedef void *(*curl_realloc_callback)(void *ptr, size_t size);
typedef char *(*curl_strdup_callback)(const char *str);
typedef void *(*curl_calloc_callback)(size_t nmemb, size_t size);

#define CURL_DID_MEMORY_FUNC_TYPEDEFS
#endif

/* the kind of data that is passed to information_callback*/
typedef enum {
  CURLINFO_TEXT = 0,
  CURLINFO_HEADER_IN,    /* 1 */
  CURLINFO_HEADER_OUT,   /* 2 */
  CURLINFO_DATA_IN,      /* 3 */
  CURLINFO_DATA_OUT,     /* 4 */
  CURLINFO_SSL_DATA_IN,  /* 5 */
  CURLINFO_SSL_DATA_OUT, /* 6 */
  CURLINFO_END
} curl_infotype;

typedef int (*curl_debug_callback)
       (CURL *handle,      /* the handle/transfer this concerns */
        curl_infotype type, /* what kind of data */
        char *data,        /* points to the data */
        size_t size,       /* size of the data pointed to */
        void *userptr);    /* whatever the user please */

/* This is the CURLOPT_PREREQFUNCTION callback prototype. */
typedef int (*curl_prereq_callback)(void *clientp,
                                    char *conn_primary_ip,
                                    char *conn_local_ip,
                                    int conn_primary_port,
                                    int conn_local_port);

/* Return code for when the pre-request callback has terminated without
   any errors */
#define CURL_PREREQFUNC_OK 0
/* Return code for when the pre-request callback wants to abort the
   request */
#define CURL_PREREQFUNC_ABORT 1

/* All possible error codes from all sorts of curl functions. Future versions
   may return other values, stay prepared.

   Always add new return codes last. Never *EVER* remove any. The return
   codes must remain the same!
 */

typedef enum {
  CURLE_OK = 0,
  CURLE_UNSUPPORTED_PROTOCOL,    /* 1 */
  CURLE_FAILED_INIT,             /* 2 */
  CURLE_URL_MALFORMAT,           /* 3 */
  CURLE_NOT_BUILT_IN,            /* 4 - [was obsoleted in August 2007 for
                                    7.17.0, reused in April 2011 for 7.21.5] */
  CURLE_COULDNT_RESOLVE_PROXY,   /* 5 */
  CURLE_COULDNT_RESOLVE_HOST,    /* 6 */
  CURLE_COULDNT_CONNECT,         /* 7 */
  CURLE_WEIRD_SERVER_REPLY,      /* 8 */
  CURLE_REMOTE_ACCESS_DENIED,    /* 9 a service was denied by the server
                                    due to lack of access - when login fails
                                    this is not returned. */
  CURLE_FTP_ACCEPT_FAILED,       /* 10 - [was obsoleted in April 2006 for
                                    7.15.4, reused in Dec 2011 for 7.24.0]*/
  CURLE_FTP_WEIRD_PASS_REPLY,    /* 11 */
  CURLE_FTP_ACCEPT_TIMEOUT,      /* 12 - timeout occurred accepting server
                                    [was obsoleted in August 2007 for 7.17.0,
                                    reused in Dec 2011 for 7.24.0]*/
  CURLE_FTP_WEIRD_PASV_REPLY,    /* 13 */
  CURLE_FTP_WEIRD_227_FORMAT,    /* 14 */
  CURLE_FTP_CANT_GET_HOST,       /* 15 */
  CURLE_HTTP2,                   /* 16 - A problem in the http2 framing layer.
                                    [was obsoleted in August 2007 for 7.17.0,
                                    reused in July 2014 for 7.38.0] */
  CURLE_FTP_COULDNT_SET_TYPE,    /* 17 */
  CURLE_PARTIAL_FILE,            /* 18 */
  CURLE_FTP_COULDNT_RETR_FILE,   /* 19 */
  CURLE_OBSOLETE20,              /* 20 - NOT USED */
  CURLE_QUOTE_ERROR,             /* 21 - quote command failure */
  CURLE_HTTP_RETURNED_ERROR,     /* 22 */
  CURLE_WRITE_ERROR,             /* 23 */
  CURLE_OBSOLETE24,              /* 24 - NOT USED */
  CURLE_UPLOAD_FAILED,           /* 25 - failed upload "command" */
  CURLE_READ_ERROR,              /* 26 - couldn't open/read from file */
  CURLE_OUT_OF_MEMORY,           /* 27 */
  /* Note: CURLE_OUT_OF_MEMORY may sometimes indicate a conversion error
           instead of a memory allocation error if CURL_DOES_CONVERSIONS
           is defined
  */
  CURLE_OPERATION_TIMEDOUT,      /* 28 - the timeout time was reached */
  CURLE_OBSOLETE29,              /* 29 - NOT USED */
  CURLE_FTP_PORT_FAILED,         /* 30 - FTP PORT operation failed */
  CURLE_FTP_COULDNT_USE_REST,    /* 31 - the REST command failed */
  CURLE_OBSOLETE32,              /* 32 - NOT USED */
  CURLE_RANGE_ERROR,             /* 33 - RANGE "command" didn't work */
  CURLE_HTTP_POST_ERROR,         /* 34 */
  CURLE_SSL_CONNECT_ERROR,       /* 35 - wrong when connecting with SSL */
  CURLE_BAD_DOWNLOAD_RESUME,     /* 36 - couldn't resume download */
  CURLE_FILE_COULDNT_READ_FILE,  /* 37 */
  CURLE_LDAP_CANNOT_BIND,        /* 38 */
  CURLE_LDAP_SEARCH_FAILED,      /* 39 */
  CURLE_OBSOLETE40,              /* 40 - NOT USED */
  CURLE_FUNCTION_NOT_FOUND,      /* 41 - NOT USED starting with 7.53.0 */
  CURLE_ABORTED_BY_CALLBACK,     /* 42 */
  CURLE_BAD_FUNCTION_ARGUMENT,   /* 43 */
  CURLE_OBSOLETE44,              /* 44 - NOT USED */
  CURLE_INTERFACE_FAILED,        /* 45 - CURLOPT_INTERFACE failed */
  CURLE_OBSOLETE46,              /* 46 - NOT USED */
  CURLE_TOO_MANY_REDIRECTS,      /* 47 - catch endless re-direct loops */
  CURLE_UNKNOWN_OPTION,          /* 48 - User specified an unknown option */
  CURLE_SETOPT_OPTION_SYNTAX,    /* 49 - Malformed setopt option */
  CURLE_OBSOLETE50,              /* 50 - NOT USED */
  CURLE_OBSOLETE51,              /* 51 - NOT USED */
  CURLE_GOT_NOTHING,             /* 52 - when this is a specific error */
  CURLE_SSL_ENGINE_NOTFOUND,     /* 53 - SSL crypto engine not found */
  CURLE_SSL_ENGINE_SETFAILED,    /* 54 - can not set SSL crypto engine as
                                    default */
  CURLE_SEND_ERROR,              /* 55 - failed sending network data */
  CURLE_RECV_ERROR,              /* 56 - failure in receiving network data */
  CURLE_OBSOLETE57,              /* 57 - NOT IN USE */
  CURLE_SSL_CERTPROBLEM,         /* 58 - problem with the local certificate */
  CURLE_SSL_CIPHER,              /* 59 - couldn't use specified cipher */
  CURLE_PEER_FAILED_VERIFICATION, /* 60 - peer's certificate or fingerprint
                                     wasn't verified fine */
  CURLE_BAD_CONTENT_ENCODING,    /* 61 - Unrecognized/bad encoding */
  CURLE_LDAP_INVALID_URL,        /* 62 - Invalid LDAP URL */
  CURLE_FILESIZE_EXCEEDED,       /* 63 - Maximum file size exceeded */
  CURLE_USE_SSL_FAILED,          /* 64 - Requested FTP SSL level failed */
  CURLE_SEND_FAIL_REWIND,        /* 65 - Sending the data requires a rewind
                                    that failed */
  CURLE_SSL_ENGINE_INITFAILED,   /* 66 - failed to initialise ENGINE */
  CURLE_LOGIN_DENIED,            /* 67 - user, password or similar was not
                                    accepted and we failed to login */
  CURLE_TFTP_NOTFOUND,           /* 68 - file not found on server */
  CURLE_TFTP_PERM,               /* 69 - permission problem on server */
  CURLE_REMOTE_DISK_FULL,        /* 70 - out of disk space on server */
  CURLE_TFTP_ILLEGAL,            /* 71 - Illegal TFTP operation */
  CURLE_TFTP_UNKNOWNID,          /* 72 - Unknown transfer ID */
  CURLE_REMOTE_FILE_EXISTS,      /* 73 - File already exists */
  CURLE_TFTP_NOSUCHUSER,         /* 74 - No such user */
  CURLE_CONV_FAILED,             /* 75 - conversion failed */
  CURLE_CONV_REQD,               /* 76 - caller must register conversion
                                    callbacks using curl_easy_setopt options
                                    CURLOPT_CONV_FROM_NETWORK_FUNCTION,
                                    CURLOPT_CONV_TO_NETWORK_FUNCTION, and
                                    CURLOPT_CONV_FROM_UTF8_FUNCTION */
  CURLE_SSL_CACERT_BADFILE,      /* 77 - could not load CACERT file, missing
                                    or wrong format */
  CURLE_REMOTE_FILE_NOT_FOUND,   /* 78 - remote file not found */
  CURLE_SSH,                     /* 79 - error from the SSH layer, somewhat
                                    generic so the error message will be of
                                    interest when this has happened */

  CURLE_SSL_SHUTDOWN_FAILED,     /* 80 - Failed to shut down the SSL
                                    connection */
  CURLE_AGAIN,                   /* 81 - socket is not ready for send/recv,
                                    wait till it's ready and try again (Added
                                    in 7.18.2) */
  CURLE_SSL_CRL_BADFILE,         /* 82 - could not load CRL file, missing or
                                    wrong format (Added in 7.19.0) */
  CURLE_SSL_ISSUER_ERROR,        /* 83 - Issuer check failed.  (Added in
                                    7.19.0) */
  CURLE_FTP_PRET_FAILED,         /* 84 - a PRET command failed */
  CURLE_RTSP_CSEQ_ERROR,         /* 85 - mismatch of RTSP CSeq numbers */
  CURLE_RTSP_SESSION_ERROR,      /* 86 - mismatch of RTSP Session Ids */
  CURLE_FTP_BAD_FILE_LIST,       /* 87 - unable to parse FTP file list */
  CURLE_CHUNK_FAILED,            /* 88 - chunk callback reported error */
  CURLE_NO_CONNECTION_AVAILABLE, /* 89 - No connection available, the
                                    session will be queued */
  CURLE_SSL_PINNEDPUBKEYNOTMATCH, /* 90 - specified pinned public key did not
                                     match */
  CURLE_SSL_INVALIDCERTSTATUS,   /* 91 - invalid certificate status */
  CURLE_HTTP2_STREAM,            /* 92 - stream error in HTTP/2 framing layer
                                    */
  CURLE_RECURSIVE_API_CALL,      /* 93 - an api function was called from
                                    inside a callback */
  CURLE_AUTH_ERROR,              /* 94 - an authentication function returned an
                                    error */
  CURLE_HTTP3,                   /* 95 - An HTTP/3 layer problem */
  CURLE_QUIC_CONNECT_ERROR,      /* 96 - QUIC connection error */
  CURLE_PROXY,                   /* 97 - proxy handshake error */
  CURLE_SSL_CLIENTCERT,          /* 98 - client-side certificate required */
  CURL_LAST /* never use! */
} CURLcode;

#ifndef CURL_NO_OLDIES /* define this to test if your app builds with all
                          the obsolete stuff removed! */

/* Previously obsolete error code re-used in 7.38.0 */
#define CURLE_OBSOLETE16 CURLE_HTTP2

/* Previously obsolete error codes re-used in 7.24.0 */
#define CURLE_OBSOLETE10 CURLE_FTP_ACCEPT_FAILED
#define CURLE_OBSOLETE12 CURLE_FTP_ACCEPT_TIMEOUT

/*  compatibility with older names */
#define CURLOPT_ENCODING CURLOPT_ACCEPT_ENCODING
#define CURLE_FTP_WEIRD_SERVER_REPLY CURLE_WEIRD_SERVER_REPLY

/* The following were added in 7.62.0 */
#define CURLE_SSL_CACERT CURLE_PEER_FAILED_VERIFICATION

/* The following were added in 7.21.5, April 2011 */
#define CURLE_UNKNOWN_TELNET_OPTION CURLE_UNKNOWN_OPTION

/* Added for 7.78.0 */
#define CURLE_TELNET_OPTION_SYNTAX CURLE_SETOPT_OPTION_SYNTAX

/* The following were added in 7.17.1 */
/* These are scheduled to disappear by 2009 */
#define CURLE_SSL_PEER_CERTIFICATE CURLE_PEER_FAILED_VERIFICATION

/* The following were added in 7.17.0 */
/* These are scheduled to disappear by 2009 */
#define CURLE_OBSOLETE CURLE_OBSOLETE50 /* no one should be using this! */
#define CURLE_BAD_PASSWORD_ENTERED CURLE_OBSOLETE46
#define CURLE_BAD_CALLING_ORDER CURLE_OBSOLETE44
#define CURLE_FTP_USER_PASSWORD_INCORRECT CURLE_OBSOLETE10
#define CURLE_FTP_CANT_RECONNECT CURLE_OBSOLETE16
#define CURLE_FTP_COULDNT_GET_SIZE CURLE_OBSOLETE32
#define CURLE_FTP_COULDNT_SET_ASCII CURLE_OBSOLETE29
#define CURLE_FTP_WEIRD_USER_REPLY CURLE_OBSOLETE12
#define CURLE_FTP_WRITE_ERROR CURLE_OBSOLETE20
#define CURLE_LIBRARY_NOT_FOUND CURLE_OBSOLETE40
#define CURLE_MALFORMAT_USER CURLE_OBSOLETE24
#define CURLE_SHARE_IN_USE CURLE_OBSOLETE57
#define CURLE_URL_MALFORMAT_USER CURLE_NOT_BUILT_IN

#define CURLE_FTP_ACCESS_DENIED CURLE_REMOTE_ACCESS_DENIED
#define CURLE_FTP_COULDNT_SET_BINARY CURLE_FTP_COULDNT_SET_TYPE
#define CURLE_FTP_QUOTE_ERROR CURLE_QUOTE_ERROR
#define CURLE_TFTP_DISKFULL CURLE_REMOTE_DISK_FULL
#define CURLE_TFTP_EXISTS CURLE_REMOTE_FILE_EXISTS
#define CURLE_HTTP_RANGE_ERROR CURLE_RANGE_ERROR
#define CURLE_FTP_SSL_FAILED CURLE_USE_SSL_FAILED

/* The following were added earlier */

#define CURLE_OPERATION_TIMEOUTED CURLE_OPERATION_TIMEDOUT

#define CURLE_HTTP_NOT_FOUND CURLE_HTTP_RETURNED_ERROR
#define CURLE_HTTP_PORT_FAILED CURLE_INTERFACE_FAILED
#define CURLE_FTP_COULDNT_STOR_FILE CURLE_UPLOAD_FAILED

#define CURLE_FTP_PARTIAL_FILE CURLE_PARTIAL_FILE
#define CURLE_FTP_BAD_DOWNLOAD_RESUME CURLE_BAD_DOWNLOAD_RESUME

/* This was the error code 50 in 7.7.3 and a few earlier versions, this
   is no longer used by libcurl but is instead #defined here only to not
   make programs break */
#define CURLE_ALREADY_COMPLETE 99999

/* Provide defines for really old option names */
#define CURLOPT_FILE CURLOPT_WRITEDATA /* name changed in 7.9.7 */
#define CURLOPT_INFILE CURLOPT_READDATA /* name changed in 7.9.7 */
#define CURLOPT_WRITEHEADER CURLOPT_HEADERDATA

/* Since long deprecated options with no code in the lib that does anything
   with them. */
#define CURLOPT_WRITEINFO CURLOPT_OBSOLETE40
#define CURLOPT_CLOSEPOLICY CURLOPT_OBSOLETE72

#endif /*!CURL_NO_OLDIES*/

/*
 * Proxy error codes. Returned in CURLINFO_PROXY_ERROR if CURLE_PROXY was
 * return for the transfers.
 */
typedef enum {
  CURLPX_OK,
  CURLPX_BAD_ADDRESS_TYPE,
  CURLPX_BAD_VERSION,
  CURLPX_CLOSED,
  CURLPX_GSSAPI,
  CURLPX_GSSAPI_PERMSG,
  CURLPX_GSSAPI_PROTECTION,
  CURLPX_IDENTD,
  CURLPX_IDENTD_DIFFER,
  CURLPX_LONG_HOSTNAME,
  CURLPX_LONG_PASSWD,
  CURLPX_LONG_USER,
  CURLPX_NO_AUTH,
  CURLPX_RECV_ADDRESS,
  CURLPX_RECV_AUTH,
  CURLPX_RECV_CONNECT,
  CURLPX_RECV_REQACK,
  CURLPX_REPLY_ADDRESS_TYPE_NOT_SUPPORTED,
  CURLPX_REPLY_COMMAND_NOT_SUPPORTED,
  CURLPX_REPLY_CONNECTION_REFUSED,
  CURLPX_REPLY_GENERAL_SERVER_FAILURE,
  CURLPX_REPLY_HOST_UNREACHABLE,
  CURLPX_REPLY_NETWORK_UNREACHABLE,
  CURLPX_REPLY_NOT_ALLOWED,
  CURLPX_REPLY_TTL_EXPIRED,
  CURLPX_REPLY_UNASSIGNED,
  CURLPX_REQUEST_FAILED,
  CURLPX_RESOLVE_HOST,
  CURLPX_SEND_AUTH,
  CURLPX_SEND_CONNECT,
  CURLPX_SEND_REQUEST,
  CURLPX_UNKNOWN_FAIL,
  CURLPX_UNKNOWN_MODE,
  CURLPX_USER_REJECTED,
  CURLPX_LAST /* never use */
} CURLproxycode;

/* This prototype applies to all conversion callbacks */
typedef CURLcode (*curl_conv_callback)(char *buffer, size_t length);

typedef CURLcode (*curl_ssl_ctx_callback)(CURL *curl,    /* easy handle */
                                          void *ssl_ctx, /* actually an OpenSSL
                                                            or WolfSSL SSL_CTX,
                                                            or an mbedTLS
                                                          mbedtls_ssl_config */
                                          void *userptr);

typedef enum {
  CURLPROXY_HTTP = 0,   /* added in 7.10, new in 7.19.4 default is to use
                           CONNECT HTTP/1.1 */
  CURLPROXY_HTTP_1_0 = 1,   /* added in 7.19.4, force to use CONNECT
                               HTTP/1.0  */
  CURLPROXY_HTTPS = 2, /* added in 7.52.0 */
  CURLPROXY_SOCKS4 = 4, /* support added in 7.15.2, enum existed already
                           in 7.10 */
  CURLPROXY_SOCKS5 = 5, /* added in 7.10 */
  CURLPROXY_SOCKS4A = 6, /* added in 7.18.0 */
  CURLPROXY_SOCKS5_HOSTNAME = 7 /* Use the SOCKS5 protocol but pass along the
                                   host name rather than the IP address. added
                                   in 7.18.0 */
} curl_proxytype;  /* this enum was added in 7.10 */

/*
 * Bitmasks for CURLOPT_HTTPAUTH and CURLOPT_PROXYAUTH options:
 *
 * CURLAUTH_NONE         - No HTTP authentication
 * CURLAUTH_BASIC        - HTTP Basic authentication (default)
 * CURLAUTH_DIGEST       - HTTP Digest authentication
 * CURLAUTH_NEGOTIATE    - HTTP Negotiate (SPNEGO) authentication
 * CURLAUTH_GSSNEGOTIATE - Alias for CURLAUTH_NEGOTIATE (deprecated)
 * CURLAUTH_NTLM         - HTTP NTLM authentication
 * CURLAUTH_DIGEST_IE    - HTTP Digest authentication with IE flavour
 * CURLAUTH_NTLM_WB      - HTTP NTLM authentication delegated to winbind helper
 * CURLAUTH_BEARER       - HTTP Bearer token authentication
 * CURLAUTH_ONLY         - Use together with a single other type to force no
 *                         authentication or just that single type
 * CURLAUTH_ANY          - All fine types set
 * CURLAUTH_ANYSAFE      - All fine types except Basic
 */

#define CURLAUTH_NONE         ((unsigned long)0)
#define CURLAUTH_BASIC        (((unsigned long)1)<<0)
#define CURLAUTH_DIGEST       (((unsigned long)1)<<1)
#define CURLAUTH_NEGOTIATE    (((unsigned long)1)<<2)
/* Deprecated since the advent of CURLAUTH_NEGOTIATE */
#define CURLAUTH_GSSNEGOTIATE CURLAUTH_NEGOTIATE
/* Used for CURLOPT_SOCKS5_AUTH to stay terminologically correct */
#define CURLAUTH_GSSAPI CURLAUTH_NEGOTIATE
#define CURLAUTH_NTLM         (((unsigned long)1)<<3)
#define CURLAUTH_DIGEST_IE    (((unsigned long)1)<<4)
#define CURLAUTH_NTLM_WB      (((unsigned long)1)<<5)
#define CURLAUTH_BEARER       (((unsigned long)1)<<6)
#define CURLAUTH_AWS_SIGV4    (((unsigned long)1)<<7)
#define CURLAUTH_ONLY         (((unsigned long)1)<<31)
#define CURLAUTH_ANY          (~CURLAUTH_DIGEST_IE)
#define CURLAUTH_ANYSAFE      (~(CURLAUTH_BASIC|CURLAUTH_DIGEST_IE))

#define CURLSSH_AUTH_ANY       ~0     /* all types supported by the server */
#define CURLSSH_AUTH_NONE      0      /* none allowed, silly but complete */
#define CURLSSH_AUTH_PUBLICKEY (1<<0) /* public/private key files */
#define CURLSSH_AUTH_PASSWORD  (1<<1) /* password */
#define CURLSSH_AUTH_HOST      (1<<2) /* host key files */
#define CURLSSH_AUTH_KEYBOARD  (1<<3) /* keyboard interactive */
#define CURLSSH_AUTH_AGENT     (1<<4) /* agent (ssh-agent, pageant...) */
#define CURLSSH_AUTH_GSSAPI    (1<<5) /* gssapi (kerberos, ...) */
#define CURLSSH_AUTH_DEFAULT CURLSSH_AUTH_ANY

#define CURLGSSAPI_DELEGATION_NONE        0      /* no delegation (default) */
#define CURLGSSAPI_DELEGATION_POLICY_FLAG (1<<0) /* if permitted by policy */
#define CURLGSSAPI_DELEGATION_FLAG        (1<<1) /* delegate always */

#define CURL_ERROR_SIZE 256

enum curl_khtype {
  CURLKHTYPE_UNKNOWN,
  CURLKHTYPE_RSA1,
  CURLKHTYPE_RSA,
  CURLKHTYPE_DSS,
  CURLKHTYPE_ECDSA,
  CURLKHTYPE_ED25519
};

struct curl_khkey {
  const char *key; /* points to a null-terminated string encoded with base64
                      if len is zero, otherwise to the "raw" data */
  size_t len;
  enum curl_khtype keytype;
};

/* this is the set of return values expected from the curl_sshkeycallback
   callback */
enum curl_khstat {
  CURLKHSTAT_FINE_ADD_TO_FILE,
  CURLKHSTAT_FINE,
  CURLKHSTAT_REJECT, /* reject the connection, return an error */
  CURLKHSTAT_DEFER,  /* do not accept it, but we can't answer right now so
                        this causes a CURLE_DEFER error but otherwise the
                        connection will be left intact etc */
  CURLKHSTAT_FINE_REPLACE, /* accept and replace the wrong key*/
  CURLKHSTAT_LAST    /* not for use, only a marker for last-in-list */
};

/* this is the set of status codes pass in to the callback */
enum curl_khmatch {
  CURLKHMATCH_OK,       /* match */
  CURLKHMATCH_MISMATCH, /* host found, key mismatch! */
  CURLKHMATCH_MISSING,  /* no matching host/key found */
  CURLKHMATCH_LAST      /* not for use, only a marker for last-in-list */
};

typedef int
  (*curl_sshkeycallback) (CURL *easy,     /* easy handle */
                          const struct curl_khkey *knownkey, /* known */
                          const struct curl_khkey *foundkey, /* found */
                          enum curl_khmatch, /* libcurl's view on the keys */
                          void *clientp); /* custom pointer passed from app */

/* parameter for the CURLOPT_USE_SSL option */
typedef enum {
  CURLUSESSL_NONE,    /* do not attempt to use SSL */
  CURLUSESSL_TRY,     /* try using SSL, proceed anyway otherwise */
  CURLUSESSL_CONTROL, /* SSL for the control connection or fail */
  CURLUSESSL_ALL,     /* SSL for all communication or fail */
  CURLUSESSL_LAST     /* not an option, never use */
} curl_usessl;

/* Definition of bits for the CURLOPT_SSL_OPTIONS argument: */

/* - ALLOW_BEAST tells libcurl to allow the BEAST SSL vulnerability in the
   name of improving interoperability with older servers. Some SSL libraries
   have introduced work-arounds for this flaw but those work-arounds sometimes
   make the SSL communication fail. To regain functionality with those broken
   servers, a user can this way allow the vulnerability back. */
#define CURLSSLOPT_ALLOW_BEAST (1<<0)

/* - NO_REVOKE tells libcurl to disable certificate revocation checks for those
   SSL backends where such behavior is present. */
#define CURLSSLOPT_NO_REVOKE (1<<1)

/* - NO_PARTIALCHAIN tells libcurl to *NOT* accept a partial certificate chain
   if possible. The OpenSSL backend has this ability. */
#define CURLSSLOPT_NO_PARTIALCHAIN (1<<2)

/* - REVOKE_BEST_EFFORT tells libcurl to ignore certificate revocation offline
   checks and ignore missing revocation list for those SSL backends where such
   behavior is present. */
#define CURLSSLOPT_REVOKE_BEST_EFFORT (1<<3)

/* - CURLSSLOPT_NATIVE_CA tells libcurl to use standard certificate store of
   operating system. Currently implemented under MS-Windows. */
#define CURLSSLOPT_NATIVE_CA (1<<4)

/* - CURLSSLOPT_AUTO_CLIENT_CERT tells libcurl to automatically locate and use
   a client certificate for authentication. (Schannel) */
#define CURLSSLOPT_AUTO_CLIENT_CERT (1<<5)

/* The default connection attempt delay in milliseconds for happy eyeballs.
   CURLOPT_HAPPY_EYEBALLS_TIMEOUT_MS.3 and happy-eyeballs-timeout-ms.d document
   this value, keep them in sync. */
#define CURL_HET_DEFAULT 200L

/* The default connection upkeep interval in milliseconds. */
#define CURL_UPKEEP_INTERVAL_DEFAULT 60000L

#ifndef CURL_NO_OLDIES /* define this to test if your app builds with all
                          the obsolete stuff removed! */

/* Backwards compatibility with older names */
/* These are scheduled to disappear by 2009 */

#define CURLFTPSSL_NONE CURLUSESSL_NONE
#define CURLFTPSSL_TRY CURLUSESSL_TRY
#define CURLFTPSSL_CONTROL CURLUSESSL_CONTROL
#define CURLFTPSSL_ALL CURLUSESSL_ALL
#define CURLFTPSSL_LAST CURLUSESSL_LAST
#define curl_ftpssl curl_usessl
#endif /*!CURL_NO_OLDIES*/

/* parameter for the CURLOPT_FTP_SSL_CCC option */
typedef enum {
  CURLFTPSSL_CCC_NONE,    /* do not send CCC */
  CURLFTPSSL_CCC_PASSIVE, /* Let the server initiate the shutdown */
  CURLFTPSSL_CCC_ACTIVE,  /* Initiate the shutdown */
  CURLFTPSSL_CCC_LAST     /* not an option, never use */
} curl_ftpccc;

/* parameter for the CURLOPT_FTPSSLAUTH option */
typedef enum {
  CURLFTPAUTH_DEFAULT, /* let libcurl decide */
  CURLFTPAUTH_SSL,     /* use "AUTH SSL" */
  CURLFTPAUTH_TLS,     /* use "AUTH TLS" */
  CURLFTPAUTH_LAST /* not an option, never use */
} curl_ftpauth;

/* parameter for the CURLOPT_FTP_CREATE_MISSING_DIRS option */
typedef enum {
  CURLFTP_CREATE_DIR_NONE,  /* do NOT create missing dirs! */
  CURLFTP_CREATE_DIR,       /* (FTP/SFTP) if CWD fails, try MKD and then CWD
                               again if MKD succeeded, for SFTP this does
                               similar magic */
  CURLFTP_CREATE_DIR_RETRY, /* (FTP only) if CWD fails, try MKD and then CWD
                               again even if MKD failed! */
  CURLFTP_CREATE_DIR_LAST   /* not an option, never use */
} curl_ftpcreatedir;

/* parameter for the CURLOPT_FTP_FILEMETHOD option */
typedef enum {
  CURLFTPMETHOD_DEFAULT,   /* let libcurl pick */
  CURLFTPMETHOD_MULTICWD,  /* single CWD operation for each path part */
  CURLFTPMETHOD_NOCWD,     /* no CWD at all */
  CURLFTPMETHOD_SINGLECWD, /* one CWD to full dir, then work on file */
  CURLFTPMETHOD_LAST       /* not an option, never use */
} curl_ftpmethod;

/* bitmask defines for CURLOPT_HEADEROPT */
#define CURLHEADER_UNIFIED  0
#define CURLHEADER_SEPARATE (1<<0)

/* CURLALTSVC_* are bits for the CURLOPT_ALTSVC_CTRL option */
#define CURLALTSVC_READONLYFILE (1<<2)
#define CURLALTSVC_H1           (1<<3)
#define CURLALTSVC_H2           (1<<4)
#define CURLALTSVC_H3           (1<<5)


struct curl_hstsentry {
  char *name;
  size_t namelen;
  unsigned int includeSubDomains:1;
  char expire[18]; /* YYYYMMDD HH:MM:SS [null-terminated] */
};

struct curl_index {
  size_t index; /* the provided entry's "index" or count */
  size_t total; /* total number of entries to save */
};

typedef enum {
  CURLSTS_OK,
  CURLSTS_DONE,
  CURLSTS_FAIL
} CURLSTScode;

typedef CURLSTScode (*curl_hstsread_callback)(CURL *easy,
                                              struct curl_hstsentry *e,
                                              void *userp);
typedef CURLSTScode (*curl_hstswrite_callback)(CURL *easy,
                                               struct curl_hstsentry *e,
                                               struct curl_index *i,
                                               void *userp);

/* CURLHSTS_* are bits for the CURLOPT_HSTS option */
#define CURLHSTS_ENABLE       (long)(1<<0)
#define CURLHSTS_READONLYFILE (long)(1<<1)

/* CURLPROTO_ defines are for the CURLOPT_*PROTOCOLS options */
#define CURLPROTO_HTTP   (1<<0)
#define CURLPROTO_HTTPS  (1<<1)
#define CURLPROTO_FTP    (1<<2)
#define CURLPROTO_FTPS   (1<<3)
#define CURLPROTO_SCP    (1<<4)
#define CURLPROTO_SFTP   (1<<5)
#define CURLPROTO_TELNET (1<<6)
#define CURLPROTO_LDAP   (1<<7)
#define CURLPROTO_LDAPS  (1<<8)
#define CURLPROTO_DICT   (1<<9)
#define CURLPROTO_FILE   (1<<10)
#define CURLPROTO_TFTP   (1<<11)
#define CURLPROTO_IMAP   (1<<12)
#define CURLPROTO_IMAPS  (1<<13)
#define CURLPROTO_POP3   (1<<14)
#define CURLPROTO_POP3S  (1<<15)
#define CURLPROTO_SMTP   (1<<16)
#define CURLPROTO_SMTPS  (1<<17)
#define CURLPROTO_RTSP   (1<<18)
#define CURLPROTO_RTMP   (1<<19)
#define CURLPROTO_RTMPT  (1<<20)
#define CURLPROTO_RTMPE  (1<<21)
#define CURLPROTO_RTMPTE (1<<22)
#define CURLPROTO_RTMPS  (1<<23)
#define CURLPROTO_RTMPTS (1<<24)
#define CURLPROTO_GOPHER (1<<25)
#define CURLPROTO_SMB    (1<<26)
#define CURLPROTO_SMBS   (1<<27)
#define CURLPROTO_MQTT   (1<<28)
#define CURLPROTO_GOPHERS (1<<29)
#define CURLPROTO_ALL    (~0) /* enable everything */

/* long may be 32 or 64 bits, but we should never depend on anything else
   but 32 */
#define CURLOPTTYPE_LONG          0
#define CURLOPTTYPE_OBJECTPOINT   10000
#define CURLOPTTYPE_FUNCTIONPOINT 20000
#define CURLOPTTYPE_OFF_T         30000
#define CURLOPTTYPE_BLOB          40000

/* *STRINGPOINT is an alias for OBJECTPOINT to allow tools to extract the
   string options from the header file */


#define CURLOPT(na,t,nu) na = t + nu

/* CURLOPT aliases that make no run-time difference */

/* 'char *' argument to a string with a trailing zero */
#define CURLOPTTYPE_STRINGPOINT CURLOPTTYPE_OBJECTPOINT

/* 'struct curl_slist *' argument */
#define CURLOPTTYPE_SLISTPOINT  CURLOPTTYPE_OBJECTPOINT

/* 'void *' argument passed untouched to callback */
#define CURLOPTTYPE_CBPOINT     CURLOPTTYPE_OBJECTPOINT

/* 'long' argument with a set of values/bitmask */
#define CURLOPTTYPE_VALUES      CURLOPTTYPE_LONG

/*
 * All CURLOPT_* values.
 */

typedef enum {
  /* This is the FILE * or void * the regular output should be written to. */
  CURLOPT(CURLOPT_WRITEDATA, CURLOPTTYPE_CBPOINT, 1),

  /* The full URL to get/put */
  CURLOPT(CURLOPT_URL, CURLOPTTYPE_STRINGPOINT, 2),

  /* Port number to connect to, if other than default. */
  CURLOPT(CURLOPT_PORT, CURLOPTTYPE_LONG, 3),

  /* Name of proxy to use. */
  CURLOPT(CURLOPT_PROXY, CURLOPTTYPE_STRINGPOINT, 4),

  /* "user:password;options" to use when fetching. */
  CURLOPT(CURLOPT_USERPWD, CURLOPTTYPE_STRINGPOINT, 5),

  /* "user:password" to use with proxy. */
  CURLOPT(CURLOPT_PROXYUSERPWD, CURLOPTTYPE_STRINGPOINT, 6),

  /* Range to get, specified as an ASCII string. */
  CURLOPT(CURLOPT_RANGE, CURLOPTTYPE_STRINGPOINT, 7),

  /* not used */

  /* Specified file stream to upload from (use as input): */
  CURLOPT(CURLOPT_READDATA, CURLOPTTYPE_CBPOINT, 9),

  /* Buffer to receive error messages in, must be at least CURL_ERROR_SIZE
   * bytes big. */
  CURLOPT(CURLOPT_ERRORBUFFER, CURLOPTTYPE_OBJECTPOINT, 10),

  /* Function that will be called to store the output (instead of fwrite). The
   * parameters will use fwrite() syntax, make sure to follow them. */
  CURLOPT(CURLOPT_WRITEFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 11),

  /* Function that will be called to read the input (instead of fread). The
   * parameters will use fread() syntax, make sure to follow them. */
  CURLOPT(CURLOPT_READFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 12),

  /* Time-out the read operation after this amount of seconds */
  CURLOPT(CURLOPT_TIMEOUT, CURLOPTTYPE_LONG, 13),

  /* If the CURLOPT_INFILE is used, this can be used to inform libcurl about
   * how large the file being sent really is. That allows better error
   * checking and better verifies that the upload was successful. -1 means
   * unknown size.
   *
   * For large file support, there is also a _LARGE version of the key
   * which takes an off_t type, allowing platforms with larger off_t
   * sizes to handle larger files.  See below for INFILESIZE_LARGE.
   */
  CURLOPT(CURLOPT_INFILESIZE, CURLOPTTYPE_LONG, 14),

  /* POST static input fields. */
  CURLOPT(CURLOPT_POSTFIELDS, CURLOPTTYPE_OBJECTPOINT, 15),

  /* Set the referrer page (needed by some CGIs) */
  CURLOPT(CURLOPT_REFERER, CURLOPTTYPE_STRINGPOINT, 16),

  /* Set the FTP PORT string (interface name, named or numerical IP address)
     Use i.e '-' to use default address. */
  CURLOPT(CURLOPT_FTPPORT, CURLOPTTYPE_STRINGPOINT, 17),

  /* Set the User-Agent string (examined by some CGIs) */
  CURLOPT(CURLOPT_USERAGENT, CURLOPTTYPE_STRINGPOINT, 18),

  /* If the download receives less than "low speed limit" bytes/second
   * during "low speed time" seconds, the operations is aborted.
   * You could i.e if you have a pretty high speed connection, abort if
   * it is less than 2000 bytes/sec during 20 seconds.
   */

  /* Set the "low speed limit" */
  CURLOPT(CURLOPT_LOW_SPEED_LIMIT, CURLOPTTYPE_LONG, 19),

  /* Set the "low speed time" */
  CURLOPT(CURLOPT_LOW_SPEED_TIME, CURLOPTTYPE_LONG, 20),

  /* Set the continuation offset.
   *
   * Note there is also a _LARGE version of this key which uses
   * off_t types, allowing for large file offsets on platforms which
   * use larger-than-32-bit off_t's.  Look below for RESUME_FROM_LARGE.
   */
  CURLOPT(CURLOPT_RESUME_FROM, CURLOPTTYPE_LONG, 21),

  /* Set cookie in request: */
  CURLOPT(CURLOPT_COOKIE, CURLOPTTYPE_STRINGPOINT, 22),

  /* This points to a linked list of headers, struct curl_slist kind. This
     list is also used for RTSP (in spite of its name) */
  CURLOPT(CURLOPT_HTTPHEADER, CURLOPTTYPE_SLISTPOINT, 23),

  /* This points to a linked list of post entries, struct curl_httppost */
  CURLOPT(CURLOPT_HTTPPOST, CURLOPTTYPE_OBJECTPOINT, 24),

  /* name of the file keeping your private SSL-certificate */
  CURLOPT(CURLOPT_SSLCERT, CURLOPTTYPE_STRINGPOINT, 25),

  /* password for the SSL or SSH private key */
  CURLOPT(CURLOPT_KEYPASSWD, CURLOPTTYPE_STRINGPOINT, 26),

  /* send TYPE parameter? */
  CURLOPT(CURLOPT_CRLF, CURLOPTTYPE_LONG, 27),

  /* send linked-list of QUOTE commands */
  CURLOPT(CURLOPT_QUOTE, CURLOPTTYPE_SLISTPOINT, 28),

  /* send FILE * or void * to store headers to, if you use a callback it
     is simply passed to the callback unmodified */
  CURLOPT(CURLOPT_HEADERDATA, CURLOPTTYPE_CBPOINT, 29),

  /* point to a file to read the initial cookies from, also enables
     "cookie awareness" */
  CURLOPT(CURLOPT_COOKIEFILE, CURLOPTTYPE_STRINGPOINT, 31),

  /* What version to specifically try to use.
     See CURL_SSLVERSION defines below. */
  CURLOPT(CURLOPT_SSLVERSION, CURLOPTTYPE_VALUES, 32),

  /* What kind of HTTP time condition to use, see defines */
  CURLOPT(CURLOPT_TIMECONDITION, CURLOPTTYPE_VALUES, 33),

  /* Time to use with the above condition. Specified in number of seconds
     since 1 Jan 1970 */
  CURLOPT(CURLOPT_TIMEVALUE, CURLOPTTYPE_LONG, 34),

  /* 35 = OBSOLETE */

  /* Custom request, for customizing the get command like
     HTTP: DELETE, TRACE and others
     FTP: to use a different list command
     */
  CURLOPT(CURLOPT_CUSTOMREQUEST, CURLOPTTYPE_STRINGPOINT, 36),

  /* FILE handle to use instead of stderr */
  CURLOPT(CURLOPT_STDERR, CURLOPTTYPE_OBJECTPOINT, 37),

  /* 38 is not used */

  /* send linked-list of post-transfer QUOTE commands */
  CURLOPT(CURLOPT_POSTQUOTE, CURLOPTTYPE_SLISTPOINT, 39),

   /* OBSOLETE, do not use! */
  CURLOPT(CURLOPT_OBSOLETE40, CURLOPTTYPE_OBJECTPOINT, 40),

  /* talk a lot */
  CURLOPT(CURLOPT_VERBOSE, CURLOPTTYPE_LONG, 41),

  /* throw the header out too */
  CURLOPT(CURLOPT_HEADER, CURLOPTTYPE_LONG, 42),

  /* shut off the progress meter */
  CURLOPT(CURLOPT_NOPROGRESS, CURLOPTTYPE_LONG, 43),

  /* use HEAD to get http document */
  CURLOPT(CURLOPT_NOBODY, CURLOPTTYPE_LONG, 44),

  /* no output on http error codes >= 400 */
  CURLOPT(CURLOPT_FAILONERROR, CURLOPTTYPE_LONG, 45),

  /* this is an upload */
  CURLOPT(CURLOPT_UPLOAD, CURLOPTTYPE_LONG, 46),

  /* HTTP POST method */
  CURLOPT(CURLOPT_POST, CURLOPTTYPE_LONG, 47),

  /* bare names when listing directories */
  CURLOPT(CURLOPT_DIRLISTONLY, CURLOPTTYPE_LONG, 48),

  /* Append instead of overwrite on upload! */
  CURLOPT(CURLOPT_APPEND, CURLOPTTYPE_LONG, 50),

  /* Specify whether to read the user+password from the .netrc or the URL.
   * This must be one of the CURL_NETRC_* enums below. */
  CURLOPT(CURLOPT_NETRC, CURLOPTTYPE_VALUES, 51),

  /* use Location: Luke! */
  CURLOPT(CURLOPT_FOLLOWLOCATION, CURLOPTTYPE_LONG, 52),

   /* transfer data in text/ASCII format */
  CURLOPT(CURLOPT_TRANSFERTEXT, CURLOPTTYPE_LONG, 53),

  /* HTTP PUT */
  CURLOPT(CURLOPT_PUT, CURLOPTTYPE_LONG, 54),

  /* 55 = OBSOLETE */

  /* DEPRECATED
   * Function that will be called instead of the internal progress display
   * function. This function should be defined as the curl_progress_callback
   * prototype defines. */
  CURLOPT(CURLOPT_PROGRESSFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 56),

  /* Data passed to the CURLOPT_PROGRESSFUNCTION and CURLOPT_XFERINFOFUNCTION
     callbacks */
  CURLOPT(CURLOPT_XFERINFODATA, CURLOPTTYPE_CBPOINT, 57),
#define CURLOPT_PROGRESSDATA CURLOPT_XFERINFODATA

  /* We want the referrer field set automatically when following locations */
  CURLOPT(CURLOPT_AUTOREFERER, CURLOPTTYPE_LONG, 58),

  /* Port of the proxy, can be set in the proxy string as well with:
     "[host]:[port]" */
  CURLOPT(CURLOPT_PROXYPORT, CURLOPTTYPE_LONG, 59),

  /* size of the POST input data, if strlen() is not good to use */
  CURLOPT(CURLOPT_POSTFIELDSIZE, CURLOPTTYPE_LONG, 60),

  /* tunnel non-http operations through a HTTP proxy */
  CURLOPT(CURLOPT_HTTPPROXYTUNNEL, CURLOPTTYPE_LONG, 61),

  /* Set the interface string to use as outgoing network interface */
  CURLOPT(CURLOPT_INTERFACE, CURLOPTTYPE_STRINGPOINT, 62),

  /* Set the krb4/5 security level, this also enables krb4/5 awareness.  This
   * is a string, 'clear', 'safe', 'confidential' or 'private'.  If the string
   * is set but doesn't match one of these, 'private' will be used.  */
  CURLOPT(CURLOPT_KRBLEVEL, CURLOPTTYPE_STRINGPOINT, 63),

  /* Set if we should verify the peer in ssl handshake, set 1 to verify. */
  CURLOPT(CURLOPT_SSL_VERIFYPEER, CURLOPTTYPE_LONG, 64),

  /* The CApath or CAfile used to validate the peer certificate
     this option is used only if SSL_VERIFYPEER is true */
  CURLOPT(CURLOPT_CAINFO, CURLOPTTYPE_STRINGPOINT, 65),

  /* 66 = OBSOLETE */
  /* 67 = OBSOLETE */

  /* Maximum number of http redirects to follow */
  CURLOPT(CURLOPT_MAXREDIRS, CURLOPTTYPE_LONG, 68),

  /* Pass a long set to 1 to get the date of the requested document (if
     possible)! Pass a zero to shut it off. */
  CURLOPT(CURLOPT_FILETIME, CURLOPTTYPE_LONG, 69),

  /* This points to a linked list of telnet options */
  CURLOPT(CURLOPT_TELNETOPTIONS, CURLOPTTYPE_SLISTPOINT, 70),

  /* Max amount of cached alive connections */
  CURLOPT(CURLOPT_MAXCONNECTS, CURLOPTTYPE_LONG, 71),

  /* OBSOLETE, do not use! */
  CURLOPT(CURLOPT_OBSOLETE72, CURLOPTTYPE_LONG, 72),

  /* 73 = OBSOLETE */

  /* Set to explicitly use a new connection for the upcoming transfer.
     Do not use this unless you're absolutely sure of this, as it makes the
     operation slower and is less friendly for the network. */
  CURLOPT(CURLOPT_FRESH_CONNECT, CURLOPTTYPE_LONG, 74),

  /* Set to explicitly forbid the upcoming transfer's connection to be re-used
     when done. Do not use this unless you're absolutely sure of this, as it
     makes the operation slower and is less friendly for the network. */
  CURLOPT(CURLOPT_FORBID_REUSE, CURLOPTTYPE_LONG, 75),

  /* Set to a file name that contains random data for libcurl to use to
     seed the random engine when doing SSL connects. */
  CURLOPT(CURLOPT_RANDOM_FILE, CURLOPTTYPE_STRINGPOINT, 76),

  /* Set to the Entropy Gathering Daemon socket pathname */
  CURLOPT(CURLOPT_EGDSOCKET, CURLOPTTYPE_STRINGPOINT, 77),

  /* Time-out connect operations after this amount of seconds, if connects are
     OK within this time, then fine... This only aborts the connect phase. */
  CURLOPT(CURLOPT_CONNECTTIMEOUT, CURLOPTTYPE_LONG, 78),

  /* Function that will be called to store headers (instead of fwrite). The
   * parameters will use fwrite() syntax, make sure to follow them. */
  CURLOPT(CURLOPT_HEADERFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 79),

  /* Set this to force the HTTP request to get back to GET. Only really usable
     if POST, PUT or a custom request have been used first.
   */
  CURLOPT(CURLOPT_HTTPGET, CURLOPTTYPE_LONG, 80),

  /* Set if we should verify the Common name from the peer certificate in ssl
   * handshake, set 1 to check existence, 2 to ensure that it matches the
   * provided hostname. */
  CURLOPT(CURLOPT_SSL_VERIFYHOST, CURLOPTTYPE_LONG, 81),

  /* Specify which file name to write all known cookies in after completed
     operation. Set file name to "-" (dash) to make it go to stdout. */
  CURLOPT(CURLOPT_COOKIEJAR, CURLOPTTYPE_STRINGPOINT, 82),

  /* Specify which SSL ciphers to use */
  CURLOPT(CURLOPT_SSL_CIPHER_LIST, CURLOPTTYPE_STRINGPOINT, 83),

  /* Specify which HTTP version to use! This must be set to one of the
     CURL_HTTP_VERSION* enums set below. */
  CURLOPT(CURLOPT_HTTP_VERSION, CURLOPTTYPE_VALUES, 84),

  /* Specifically switch on or off the FTP engine's use of the EPSV command. By
     default, that one will always be attempted before the more traditional
     PASV command. */
  CURLOPT(CURLOPT_FTP_USE_EPSV, CURLOPTTYPE_LONG, 85),

  /* type of the file keeping your SSL-certificate ("DER", "PEM", "ENG") */
  CURLOPT(CURLOPT_SSLCERTTYPE, CURLOPTTYPE_STRINGPOINT, 86),

  /* name of the file keeping your private SSL-key */
  CURLOPT(CURLOPT_SSLKEY, CURLOPTTYPE_STRINGPOINT, 87),

  /* type of the file keeping your private SSL-key ("DER", "PEM", "ENG") */
  CURLOPT(CURLOPT_SSLKEYTYPE, CURLOPTTYPE_STRINGPOINT, 88),

  /* crypto engine for the SSL-sub system */
  CURLOPT(CURLOPT_SSLENGINE, CURLOPTTYPE_STRINGPOINT, 89),

  /* set the crypto engine for the SSL-sub system as default
     the param has no meaning...
   */
  CURLOPT(CURLOPT_SSLENGINE_DEFAULT, CURLOPTTYPE_LONG, 90),

  /* Non-zero value means to use the global dns cache */
  /* DEPRECATED, do not use! */
  CURLOPT(CURLOPT_DNS_USE_GLOBAL_CACHE, CURLOPTTYPE_LONG, 91),

  /* DNS cache timeout */
  CURLOPT(CURLOPT_DNS_CACHE_TIMEOUT, CURLOPTTYPE_LONG, 92),

  /* send linked-list of pre-transfer QUOTE commands */
  CURLOPT(CURLOPT_PREQUOTE, CURLOPTTYPE_SLISTPOINT, 93),

  /* set the debug function */
  CURLOPT(CURLOPT_DEBUGFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 94),

  /* set the data for the debug function */
  CURLOPT(CURLOPT_DEBUGDATA, CURLOPTTYPE_CBPOINT, 95),

  /* mark this as start of a cookie session */
  CURLOPT(CURLOPT_COOKIESESSION, CURLOPTTYPE_LONG, 96),

  /* The CApath directory used to validate the peer certificate
     this option is used only if SSL_VERIFYPEER is true */
  CURLOPT(CURLOPT_CAPATH, CURLOPTTYPE_STRINGPOINT, 97),

  /* Instruct libcurl to use a smaller receive buffer */
  CURLOPT(CURLOPT_BUFFERSIZE, CURLOPTTYPE_LONG, 98),

  /* Instruct libcurl to not use any signal/alarm handlers, even when using
     timeouts. This option is useful for multi-threaded applications.
     See libcurl-the-guide for more background information. */
  CURLOPT(CURLOPT_NOSIGNAL, CURLOPTTYPE_LONG, 99),

  /* Provide a CURLShare for mutexing non-ts data */
  CURLOPT(CURLOPT_SHARE, CURLOPTTYPE_OBJECTPOINT, 100),

  /* indicates type of proxy. accepted values are CURLPROXY_HTTP (default),
     CURLPROXY_HTTPS, CURLPROXY_SOCKS4, CURLPROXY_SOCKS4A and
     CURLPROXY_SOCKS5. */
  CURLOPT(CURLOPT_PROXYTYPE, CURLOPTTYPE_VALUES, 101),

  /* Set the Accept-Encoding string. Use this to tell a server you would like
     the response to be compressed. Before 7.21.6, this was known as
     CURLOPT_ENCODING */
  CURLOPT(CURLOPT_ACCEPT_ENCODING, CURLOPTTYPE_STRINGPOINT, 102),

  /* Set pointer to private data */
  CURLOPT(CURLOPT_PRIVATE, CURLOPTTYPE_OBJECTPOINT, 103),

  /* Set aliases for HTTP 200 in the HTTP Response header */
  CURLOPT(CURLOPT_HTTP200ALIASES, CURLOPTTYPE_SLISTPOINT, 104),

  /* Continue to send authentication (user+password) when following locations,
     even when hostname changed. This can potentially send off the name
     and password to whatever host the server decides. */
  CURLOPT(CURLOPT_UNRESTRICTED_AUTH, CURLOPTTYPE_LONG, 105),

  /* Specifically switch on or off the FTP engine's use of the EPRT command (
     it also disables the LPRT attempt). By default, those ones will always be
     attempted before the good old traditional PORT command. */
  CURLOPT(CURLOPT_FTP_USE_EPRT, CURLOPTTYPE_LONG, 106),

  /* Set this to a bitmask value to enable the particular authentications
     methods you like. Use this in combination with CURLOPT_USERPWD.
     Note that setting multiple bits may cause extra network round-trips. */
  CURLOPT(CURLOPT_HTTPAUTH, CURLOPTTYPE_VALUES, 107),

  /* Set the ssl context callback function, currently only for OpenSSL or
     WolfSSL ssl_ctx, or mbedTLS mbedtls_ssl_config in the second argument.
     The function must match the curl_ssl_ctx_callback prototype. */
  CURLOPT(CURLOPT_SSL_CTX_FUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 108),

  /* Set the userdata for the ssl context callback function's third
     argument */
  CURLOPT(CURLOPT_SSL_CTX_DATA, CURLOPTTYPE_CBPOINT, 109),

  /* FTP Option that causes missing dirs to be created on the remote server.
     In 7.19.4 we introduced the convenience enums for this option using the
     CURLFTP_CREATE_DIR prefix.
  */
  CURLOPT(CURLOPT_FTP_CREATE_MISSING_DIRS, CURLOPTTYPE_LONG, 110),

  /* Set this to a bitmask value to enable the particular authentications
     methods you like. Use this in combination with CURLOPT_PROXYUSERPWD.
     Note that setting multiple bits may cause extra network round-trips. */
  CURLOPT(CURLOPT_PROXYAUTH, CURLOPTTYPE_VALUES, 111),

  /* FTP option that changes the timeout, in seconds, associated with
     getting a response.  This is different from transfer timeout time and
     essentially places a demand on the FTP server to acknowledge commands
     in a timely manner. */
  CURLOPT(CURLOPT_FTP_RESPONSE_TIMEOUT, CURLOPTTYPE_LONG, 112),
#define CURLOPT_SERVER_RESPONSE_TIMEOUT CURLOPT_FTP_RESPONSE_TIMEOUT

  /* Set this option to one of the CURL_IPRESOLVE_* defines (see below) to
     tell libcurl to use those IP versions only. This only has effect on
     systems with support for more than one, i.e IPv4 _and_ IPv6. */
  CURLOPT(CURLOPT_IPRESOLVE, CURLOPTTYPE_VALUES, 113),

  /* Set this option to limit the size of a file that will be downloaded from
     an HTTP or FTP server.

     Note there is also _LARGE version which adds large file support for
     platforms which have larger off_t sizes.  See MAXFILESIZE_LARGE below. */
  CURLOPT(CURLOPT_MAXFILESIZE, CURLOPTTYPE_LONG, 114),

  /* See the comment for INFILESIZE above, but in short, specifies
   * the size of the file being uploaded.  -1 means unknown.
   */
  CURLOPT(CURLOPT_INFILESIZE_LARGE, CURLOPTTYPE_OFF_T, 115),

  /* Sets the continuation offset.  There is also a CURLOPTTYPE_LONG version
   * of this; look above for RESUME_FROM.
   */
  CURLOPT(CURLOPT_RESUME_FROM_LARGE, CURLOPTTYPE_OFF_T, 116),

  /* Sets the maximum size of data that will be downloaded from
   * an HTTP or FTP server.  See MAXFILESIZE above for the LONG version.
   */
  CURLOPT(CURLOPT_MAXFILESIZE_LARGE, CURLOPTTYPE_OFF_T, 117),

  /* Set this option to the file name of your .netrc file you want libcurl
     to parse (using the CURLOPT_NETRC option). If not set, libcurl will do
     a poor attempt to find the user's home directory and check for a .netrc
     file in there. */
  CURLOPT(CURLOPT_NETRC_FILE, CURLOPTTYPE_STRINGPOINT, 118),

  /* Enable SSL/TLS for FTP, pick one of:
     CURLUSESSL_TRY     - try using SSL, proceed anyway otherwise
     CURLUSESSL_CONTROL - SSL for the control connection or fail
     CURLUSESSL_ALL     - SSL for all communication or fail
  */
  CURLOPT(CURLOPT_USE_SSL, CURLOPTTYPE_VALUES, 119),

  /* The _LARGE version of the standard POSTFIELDSIZE option */
  CURLOPT(CURLOPT_POSTFIELDSIZE_LARGE, CURLOPTTYPE_OFF_T, 120),

  /* Enable/disable the TCP Nagle algorithm */
  CURLOPT(CURLOPT_TCP_NODELAY, CURLOPTTYPE_LONG, 121),

  /* 122 OBSOLETE, used in 7.12.3. Gone in 7.13.0 */
  /* 123 OBSOLETE. Gone in 7.16.0 */
  /* 124 OBSOLETE, used in 7.12.3. Gone in 7.13.0 */
  /* 125 OBSOLETE, used in 7.12.3. Gone in 7.13.0 */
  /* 126 OBSOLETE, used in 7.12.3. Gone in 7.13.0 */
  /* 127 OBSOLETE. Gone in 7.16.0 */
  /* 128 OBSOLETE. Gone in 7.16.0 */

  /* When FTP over SSL/TLS is selected (with CURLOPT_USE_SSL), this option
     can be used to change libcurl's default action which is to first try
     "AUTH SSL" and then "AUTH TLS" in this order, and proceed when a OK
     response has been received.

     Available parameters are:
     CURLFTPAUTH_DEFAULT - let libcurl decide
     CURLFTPAUTH_SSL     - try "AUTH SSL" first, then TLS
     CURLFTPAUTH_TLS     - try "AUTH TLS" first, then SSL
  */
  CURLOPT(CURLOPT_FTPSSLAUTH, CURLOPTTYPE_VALUES, 129),

  CURLOPT(CURLOPT_IOCTLFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 130),
  CURLOPT(CURLOPT_IOCTLDATA, CURLOPTTYPE_CBPOINT, 131),

  /* 132 OBSOLETE. Gone in 7.16.0 */
  /* 133 OBSOLETE. Gone in 7.16.0 */

  /* null-terminated string for pass on to the FTP server when asked for
     "account" info */
  CURLOPT(CURLOPT_FTP_ACCOUNT, CURLOPTTYPE_STRINGPOINT, 134),

  /* feed cookie into cookie engine */
  CURLOPT(CURLOPT_COOKIELIST, CURLOPTTYPE_STRINGPOINT, 135),

  /* ignore Content-Length */
  CURLOPT(CURLOPT_IGNORE_CONTENT_LENGTH, CURLOPTTYPE_LONG, 136),

  /* Set to non-zero to skip the IP address received in a 227 PASV FTP server
     response. Typically used for FTP-SSL purposes but is not restricted to
     that. libcurl will then instead use the same IP address it used for the
     control connection. */
  CURLOPT(CURLOPT_FTP_SKIP_PASV_IP, CURLOPTTYPE_LONG, 137),

  /* Select "file method" to use when doing FTP, see the curl_ftpmethod
     above. */
  CURLOPT(CURLOPT_FTP_FILEMETHOD, CURLOPTTYPE_VALUES, 138),

  /* Local port number to bind the socket to */
  CURLOPT(CURLOPT_LOCALPORT, CURLOPTTYPE_LONG, 139),

  /* Number of ports to try, including the first one set with LOCALPORT.
     Thus, setting it to 1 will make no additional attempts but the first.
  */
  CURLOPT(CURLOPT_LOCALPORTRANGE, CURLOPTTYPE_LONG, 140),

  /* no transfer, set up connection and let application use the socket by
     extracting it with CURLINFO_LASTSOCKET */
  CURLOPT(CURLOPT_CONNECT_ONLY, CURLOPTTYPE_LONG, 141),

  /* Function that will be called to convert from the
     network encoding (instead of using the iconv calls in libcurl) */
  CURLOPT(CURLOPT_CONV_FROM_NETWORK_FUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 142),

  /* Function that will be called to convert to the
     network encoding (instead of using the iconv calls in libcurl) */
  CURLOPT(CURLOPT_CONV_TO_NETWORK_FUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 143),

  /* Function that will be called to convert from UTF8
     (instead of using the iconv calls in libcurl)
     Note that this is used only for SSL certificate processing */
  CURLOPT(CURLOPT_CONV_FROM_UTF8_FUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 144),

  /* if the connection proceeds too quickly then need to slow it down */
  /* limit-rate: maximum number of bytes per second to send or receive */
  CURLOPT(CURLOPT_MAX_SEND_SPEED_LARGE, CURLOPTTYPE_OFF_T, 145),
  CURLOPT(CURLOPT_MAX_RECV_SPEED_LARGE, CURLOPTTYPE_OFF_T, 146),

  /* Pointer to command string to send if USER/PASS fails. */
  CURLOPT(CURLOPT_FTP_ALTERNATIVE_TO_USER, CURLOPTTYPE_STRINGPOINT, 147),

  /* callback function for setting socket options */
  CURLOPT(CURLOPT_SOCKOPTFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 148),
  CURLOPT(CURLOPT_SOCKOPTDATA, CURLOPTTYPE_CBPOINT, 149),

  /* set to 0 to disable session ID re-use for this transfer, default is
     enabled (== 1) */
  CURLOPT(CURLOPT_SSL_SESSIONID_CACHE, CURLOPTTYPE_LONG, 150),

  /* allowed SSH authentication methods */
  CURLOPT(CURLOPT_SSH_AUTH_TYPES, CURLOPTTYPE_VALUES, 151),

  /* Used by scp/sftp to do public/private key authentication */
  CURLOPT(CURLOPT_SSH_PUBLIC_KEYFILE, CURLOPTTYPE_STRINGPOINT, 152),
  CURLOPT(CURLOPT_SSH_PRIVATE_KEYFILE, CURLOPTTYPE_STRINGPOINT, 153),

  /* Send CCC (Clear Command Channel) after authentication */
  CURLOPT(CURLOPT_FTP_SSL_CCC, CURLOPTTYPE_LONG, 154),

  /* Same as TIMEOUT and CONNECTTIMEOUT, but with ms resolution */
  CURLOPT(CURLOPT_TIMEOUT_MS, CURLOPTTYPE_LONG, 155),
  CURLOPT(CURLOPT_CONNECTTIMEOUT_MS, CURLOPTTYPE_LONG, 156),

  /* set to zero to disable the libcurl's decoding and thus pass the raw body
     data to the application even when it is encoded/compressed */
  CURLOPT(CURLOPT_HTTP_TRANSFER_DECODING, CURLOPTTYPE_LONG, 157),
  CURLOPT(CURLOPT_HTTP_CONTENT_DECODING, CURLOPTTYPE_LONG, 158),

  /* Permission used when creating new files and directories on the remote
     server for protocols that support it, SFTP/SCP/FILE */
  CURLOPT(CURLOPT_NEW_FILE_PERMS, CURLOPTTYPE_LONG, 159),
  CURLOPT(CURLOPT_NEW_DIRECTORY_PERMS, CURLOPTTYPE_LONG, 160),

  /* Set the behavior of POST when redirecting. Values must be set to one
     of CURL_REDIR* defines below. This used to be called CURLOPT_POST301 */
  CURLOPT(CURLOPT_POSTREDIR, CURLOPTTYPE_VALUES, 161),

  /* used by scp/sftp to verify the host's public key */
  CURLOPT(CURLOPT_SSH_HOST_PUBLIC_KEY_MD5, CURLOPTTYPE_STRINGPOINT, 162),

  /* Callback function for opening socket (instead of socket(2)). Optionally,
     callback is able change the address or refuse to connect returning
     CURL_SOCKET_BAD.  The callback should have type
     curl_opensocket_callback */
  CURLOPT(CURLOPT_OPENSOCKETFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 163),
  CURLOPT(CURLOPT_OPENSOCKETDATA, CURLOPTTYPE_CBPOINT, 164),

  /* POST volatile input fields. */
  CURLOPT(CURLOPT_COPYPOSTFIELDS, CURLOPTTYPE_OBJECTPOINT, 165),

  /* set transfer mode (;type=<a|i>) when doing FTP via an HTTP proxy */
  CURLOPT(CURLOPT_PROXY_TRANSFER_MODE, CURLOPTTYPE_LONG, 166),

  /* Callback function for seeking in the input stream */
  CURLOPT(CURLOPT_SEEKFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 167),
  CURLOPT(CURLOPT_SEEKDATA, CURLOPTTYPE_CBPOINT, 168),

  /* CRL file */
  CURLOPT(CURLOPT_CRLFILE, CURLOPTTYPE_STRINGPOINT, 169),

  /* Issuer certificate */
  CURLOPT(CURLOPT_ISSUERCERT, CURLOPTTYPE_STRINGPOINT, 170),

  /* (IPv6) Address scope */
  CURLOPT(CURLOPT_ADDRESS_SCOPE, CURLOPTTYPE_LONG, 171),

  /* Collect certificate chain info and allow it to get retrievable with
     CURLINFO_CERTINFO after the transfer is complete. */
  CURLOPT(CURLOPT_CERTINFO, CURLOPTTYPE_LONG, 172),

  /* "name" and "pwd" to use when fetching. */
  CURLOPT(CURLOPT_USERNAME, CURLOPTTYPE_STRINGPOINT, 173),
  CURLOPT(CURLOPT_PASSWORD, CURLOPTTYPE_STRINGPOINT, 174),

    /* "name" and "pwd" to use with Proxy when fetching. */
  CURLOPT(CURLOPT_PROXYUSERNAME, CURLOPTTYPE_STRINGPOINT, 175),
  CURLOPT(CURLOPT_PROXYPASSWORD, CURLOPTTYPE_STRINGPOINT, 176),

  /* Comma separated list of hostnames defining no-proxy zones. These should
     match both hostnames directly, and hostnames within a domain. For
     example, local.com will match local.com and www.local.com, but NOT
     notlocal.com or www.notlocal.com. For compatibility with other
     implementations of this, .local.com will be considered to be the same as
     local.com. A single * is the only valid wildcard, and effectively
     disables the use of proxy. */
  CURLOPT(CURLOPT_NOPROXY, CURLOPTTYPE_STRINGPOINT, 177),

  /* block size for TFTP transfers */
  CURLOPT(CURLOPT_TFTP_BLKSIZE, CURLOPTTYPE_LONG, 178),

  /* Socks Service */
  /* DEPRECATED, do not use! */
  CURLOPT(CURLOPT_SOCKS5_GSSAPI_SERVICE, CURLOPTTYPE_STRINGPOINT, 179),

  /* Socks Service */
  CURLOPT(CURLOPT_SOCKS5_GSSAPI_NEC, CURLOPTTYPE_LONG, 180),

  /* set the bitmask for the protocols that are allowed to be used for the
     transfer, which thus helps the app which takes URLs from users or other
     external inputs and want to restrict what protocol(s) to deal
     with. Defaults to CURLPROTO_ALL. */
  CURLOPT(CURLOPT_PROTOCOLS, CURLOPTTYPE_LONG, 181),

  /* set the bitmask for the protocols that libcurl is allowed to follow to,
     as a subset of the CURLOPT_PROTOCOLS ones. That means the protocol needs
     to be set in both bitmasks to be allowed to get redirected to. */
  CURLOPT(CURLOPT_REDIR_PROTOCOLS, CURLOPTTYPE_LONG, 182),

  /* set the SSH knownhost file name to use */
  CURLOPT(CURLOPT_SSH_KNOWNHOSTS, CURLOPTTYPE_STRINGPOINT, 183),

  /* set the SSH host key callback, must point to a curl_sshkeycallback
     function */
  CURLOPT(CURLOPT_SSH_KEYFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 184),

  /* set the SSH host key callback custom pointer */
  CURLOPT(CURLOPT_SSH_KEYDATA, CURLOPTTYPE_CBPOINT, 185),

  /* set the SMTP mail originator */
  CURLOPT(CURLOPT_MAIL_FROM, CURLOPTTYPE_STRINGPOINT, 186),

  /* set the list of SMTP mail receiver(s) */
  CURLOPT(CURLOPT_MAIL_RCPT, CURLOPTTYPE_SLISTPOINT, 187),

  /* FTP: send PRET before PASV */
  CURLOPT(CURLOPT_FTP_USE_PRET, CURLOPTTYPE_LONG, 188),

  /* RTSP request method (OPTIONS, SETUP, PLAY, etc...) */
  CURLOPT(CURLOPT_RTSP_REQUEST, CURLOPTTYPE_VALUES, 189),

  /* The RTSP session identifier */
  CURLOPT(CURLOPT_RTSP_SESSION_ID, CURLOPTTYPE_STRINGPOINT, 190),

  /* The RTSP stream URI */
  CURLOPT(CURLOPT_RTSP_STREAM_URI, CURLOPTTYPE_STRINGPOINT, 191),

  /* The Transport: header to use in RTSP requests */
  CURLOPT(CURLOPT_RTSP_TRANSPORT, CURLOPTTYPE_STRINGPOINT, 192),

  /* Manually initialize the client RTSP CSeq for this handle */
  CURLOPT(CURLOPT_RTSP_CLIENT_CSEQ, CURLOPTTYPE_LONG, 193),

  /* Manually initialize the server RTSP CSeq for this handle */
  CURLOPT(CURLOPT_RTSP_SERVER_CSEQ, CURLOPTTYPE_LONG, 194),

  /* The stream to pass to INTERLEAVEFUNCTION. */
  CURLOPT(CURLOPT_INTERLEAVEDATA, CURLOPTTYPE_CBPOINT, 195),

  /* Let the application define a custom write method for RTP data */
  CURLOPT(CURLOPT_INTERLEAVEFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 196),

  /* Turn on wildcard matching */
  CURLOPT(CURLOPT_WILDCARDMATCH, CURLOPTTYPE_LONG, 197),

  /* Directory matching callback called before downloading of an
     individual file (chunk) started */
  CURLOPT(CURLOPT_CHUNK_BGN_FUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 198),

  /* Directory matching callback called after the file (chunk)
     was downloaded, or skipped */
  CURLOPT(CURLOPT_CHUNK_END_FUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 199),

  /* Change match (fnmatch-like) callback for wildcard matching */
  CURLOPT(CURLOPT_FNMATCH_FUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 200),

  /* Let the application define custom chunk data pointer */
  CURLOPT(CURLOPT_CHUNK_DATA, CURLOPTTYPE_CBPOINT, 201),

  /* FNMATCH_FUNCTION user pointer */
  CURLOPT(CURLOPT_FNMATCH_DATA, CURLOPTTYPE_CBPOINT, 202),

  /* send linked-list of name:port:address sets */
  CURLOPT(CURLOPT_RESOLVE, CURLOPTTYPE_SLISTPOINT, 203),

  /* Set a username for authenticated TLS */
  CURLOPT(CURLOPT_TLSAUTH_USERNAME, CURLOPTTYPE_STRINGPOINT, 204),

  /* Set a password for authenticated TLS */
  CURLOPT(CURLOPT_TLSAUTH_PASSWORD, CURLOPTTYPE_STRINGPOINT, 205),

  /* Set authentication type for authenticated TLS */
  CURLOPT(CURLOPT_TLSAUTH_TYPE, CURLOPTTYPE_STRINGPOINT, 206),

  /* Set to 1 to enable the "TE:" header in HTTP requests to ask for
     compressed transfer-encoded responses. Set to 0 to disable the use of TE:
     in outgoing requests. The current default is 0, but it might change in a
     future libcurl release.

     libcurl will ask for the compressed methods it knows of, and if that
     isn't any, it will not ask for transfer-encoding at all even if this
     option is set to 1.

  */
  CURLOPT(CURLOPT_TRANSFER_ENCODING, CURLOPTTYPE_LONG, 207),

  /* Callback function for closing socket (instead of close(2)). The callback
     should have type curl_closesocket_callback */
  CURLOPT(CURLOPT_CLOSESOCKETFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 208),
  CURLOPT(CURLOPT_CLOSESOCKETDATA, CURLOPTTYPE_CBPOINT, 209),

  /* allow GSSAPI credential delegation */
  CURLOPT(CURLOPT_GSSAPI_DELEGATION, CURLOPTTYPE_VALUES, 210),

  /* Set the name servers to use for DNS resolution */
  CURLOPT(CURLOPT_DNS_SERVERS, CURLOPTTYPE_STRINGPOINT, 211),

  /* Time-out accept operations (currently for FTP only) after this amount
     of milliseconds. */
  CURLOPT(CURLOPT_ACCEPTTIMEOUT_MS, CURLOPTTYPE_LONG, 212),

  /* Set TCP keepalive */
  CURLOPT(CURLOPT_TCP_KEEPALIVE, CURLOPTTYPE_LONG, 213),

  /* non-universal keepalive knobs (Linux, AIX, HP-UX, more) */
  CURLOPT(CURLOPT_TCP_KEEPIDLE, CURLOPTTYPE_LONG, 214),
  CURLOPT(CURLOPT_TCP_KEEPINTVL, CURLOPTTYPE_LONG, 215),

  /* Enable/disable specific SSL features with a bitmask, see CURLSSLOPT_* */
  CURLOPT(CURLOPT_SSL_OPTIONS, CURLOPTTYPE_VALUES, 216),

  /* Set the SMTP auth originator */
  CURLOPT(CURLOPT_MAIL_AUTH, CURLOPTTYPE_STRINGPOINT, 217),

  /* Enable/disable SASL initial response */
  CURLOPT(CURLOPT_SASL_IR, CURLOPTTYPE_LONG, 218),

  /* Function that will be called instead of the internal progress display
   * function. This function should be defined as the curl_xferinfo_callback
   * prototype defines. (Deprecates CURLOPT_PROGRESSFUNCTION) */
  CURLOPT(CURLOPT_XFERINFOFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 219),

  /* The XOAUTH2 bearer token */
  CURLOPT(CURLOPT_XOAUTH2_BEARER, CURLOPTTYPE_STRINGPOINT, 220),

  /* Set the interface string to use as outgoing network
   * interface for DNS requests.
   * Only supported by the c-ares DNS backend */
  CURLOPT(CURLOPT_DNS_INTERFACE, CURLOPTTYPE_STRINGPOINT, 221),

  /* Set the local IPv4 address to use for outgoing DNS requests.
   * Only supported by the c-ares DNS backend */
  CURLOPT(CURLOPT_DNS_LOCAL_IP4, CURLOPTTYPE_STRINGPOINT, 222),

  /* Set the local IPv6 address to use for outgoing DNS requests.
   * Only supported by the c-ares DNS backend */
  CURLOPT(CURLOPT_DNS_LOCAL_IP6, CURLOPTTYPE_STRINGPOINT, 223),

  /* Set authentication options directly */
  CURLOPT(CURLOPT_LOGIN_OPTIONS, CURLOPTTYPE_STRINGPOINT, 224),

  /* Enable/disable TLS NPN extension (http2 over ssl might fail without) */
  CURLOPT(CURLOPT_SSL_ENABLE_NPN, CURLOPTTYPE_LONG, 225),

  /* Enable/disable TLS ALPN extension (http2 over ssl might fail without) */
  CURLOPT(CURLOPT_SSL_ENABLE_ALPN, CURLOPTTYPE_LONG, 226),

  /* Time to wait for a response to a HTTP request containing an
   * Expect: 100-continue header before sending the data anyway. */
  CURLOPT(CURLOPT_EXPECT_100_TIMEOUT_MS, CURLOPTTYPE_LONG, 227),

  /* This points to a linked list of headers used for proxy requests only,
     struct curl_slist kind */
  CURLOPT(CURLOPT_PROXYHEADER, CURLOPTTYPE_SLISTPOINT, 228),

  /* Pass in a bitmask of "header options" */
  CURLOPT(CURLOPT_HEADEROPT, CURLOPTTYPE_VALUES, 229),

  /* The public key in DER form used to validate the peer public key
     this option is used only if SSL_VERIFYPEER is true */
  CURLOPT(CURLOPT_PINNEDPUBLICKEY, CURLOPTTYPE_STRINGPOINT, 230),

  /* Path to Unix domain socket */
  CURLOPT(CURLOPT_UNIX_SOCKET_PATH, CURLOPTTYPE_STRINGPOINT, 231),

  /* Set if we should verify the certificate status. */
  CURLOPT(CURLOPT_SSL_VERIFYSTATUS, CURLOPTTYPE_LONG, 232),

  /* Set if we should enable TLS false start. */
  CURLOPT(CURLOPT_SSL_FALSESTART, CURLOPTTYPE_LONG, 233),

  /* Do not squash dot-dot sequences */
  CURLOPT(CURLOPT_PATH_AS_IS, CURLOPTTYPE_LONG, 234),

  /* Proxy Service Name */
  CURLOPT(CURLOPT_PROXY_SERVICE_NAME, CURLOPTTYPE_STRINGPOINT, 235),

  /* Service Name */
  CURLOPT(CURLOPT_SERVICE_NAME, CURLOPTTYPE_STRINGPOINT, 236),

  /* Wait/don't wait for pipe/mutex to clarify */
  CURLOPT(CURLOPT_PIPEWAIT, CURLOPTTYPE_LONG, 237),

  /* Set the protocol used when curl is given a URL without a protocol */
  CURLOPT(CURLOPT_DEFAULT_PROTOCOL, CURLOPTTYPE_STRINGPOINT, 238),

  /* Set stream weight, 1 - 256 (default is 16) */
  CURLOPT(CURLOPT_STREAM_WEIGHT, CURLOPTTYPE_LONG, 239),

  /* Set stream dependency on another CURL handle */
  CURLOPT(CURLOPT_STREAM_DEPENDS, CURLOPTTYPE_OBJECTPOINT, 240),

  /* Set E-xclusive stream dependency on another CURL handle */
  CURLOPT(CURLOPT_STREAM_DEPENDS_E, CURLOPTTYPE_OBJECTPOINT, 241),

  /* Do not send any tftp option requests to the server */
  CURLOPT(CURLOPT_TFTP_NO_OPTIONS, CURLOPTTYPE_LONG, 242),

  /* Linked-list of host:port:connect-to-host:connect-to-port,
     overrides the URL's host:port (only for the network layer) */
  CURLOPT(CURLOPT_CONNECT_TO, CURLOPTTYPE_SLISTPOINT, 243),

  /* Set TCP Fast Open */
  CURLOPT(CURLOPT_TCP_FASTOPEN, CURLOPTTYPE_LONG, 244),

  /* Continue to send data if the server responds early with an
   * HTTP status code >= 300 */
  CURLOPT(CURLOPT_KEEP_SENDING_ON_ERROR, CURLOPTTYPE_LONG, 245),

  /* The CApath or CAfile used to validate the proxy certificate
     this option is used only if PROXY_SSL_VERIFYPEER is true */
  CURLOPT(CURLOPT_PROXY_CAINFO, CURLOPTTYPE_STRINGPOINT, 246),

  /* The CApath directory used to validate the proxy certificate
     this option is used only if PROXY_SSL_VERIFYPEER is true */
  CURLOPT(CURLOPT_PROXY_CAPATH, CURLOPTTYPE_STRINGPOINT, 247),

  /* Set if we should verify the proxy in ssl handshake,
     set 1 to verify. */
  CURLOPT(CURLOPT_PROXY_SSL_VERIFYPEER, CURLOPTTYPE_LONG, 248),

  /* Set if we should verify the Common name from the proxy certificate in ssl
   * handshake, set 1 to check existence, 2 to ensure that it matches
   * the provided hostname. */
  CURLOPT(CURLOPT_PROXY_SSL_VERIFYHOST, CURLOPTTYPE_LONG, 249),

  /* What version to specifically try to use for proxy.
     See CURL_SSLVERSION defines below. */
  CURLOPT(CURLOPT_PROXY_SSLVERSION, CURLOPTTYPE_VALUES, 250),

  /* Set a username for authenticated TLS for proxy */
  CURLOPT(CURLOPT_PROXY_TLSAUTH_USERNAME, CURLOPTTYPE_STRINGPOINT, 251),

  /* Set a password for authenticated TLS for proxy */
  CURLOPT(CURLOPT_PROXY_TLSAUTH_PASSWORD, CURLOPTTYPE_STRINGPOINT, 252),

  /* Set authentication type for authenticated TLS for proxy */
  CURLOPT(CURLOPT_PROXY_TLSAUTH_TYPE, CURLOPTTYPE_STRINGPOINT, 253),

  /* name of the file keeping your private SSL-certificate for proxy */
  CURLOPT(CURLOPT_PROXY_SSLCERT, CURLOPTTYPE_STRINGPOINT, 254),

  /* type of the file keeping your SSL-certificate ("DER", "PEM", "ENG") for
     proxy */
  CURLOPT(CURLOPT_PROXY_SSLCERTTYPE, CURLOPTTYPE_STRINGPOINT, 255),

  /* name of the file keeping your private SSL-key for proxy */
  CURLOPT(CURLOPT_PROXY_SSLKEY, CURLOPTTYPE_STRINGPOINT, 256),

  /* type of the file keeping your private SSL-key ("DER", "PEM", "ENG") for
     proxy */
  CURLOPT(CURLOPT_PROXY_SSLKEYTYPE, CURLOPTTYPE_STRINGPOINT, 257),

  /* password for the SSL private key for proxy */
  CURLOPT(CURLOPT_PROXY_KEYPASSWD, CURLOPTTYPE_STRINGPOINT, 258),

  /* Specify which SSL ciphers to use for proxy */
  CURLOPT(CURLOPT_PROXY_SSL_CIPHER_LIST, CURLOPTTYPE_STRINGPOINT, 259),

  /* CRL file for proxy */
  CURLOPT(CURLOPT_PROXY_CRLFILE, CURLOPTTYPE_STRINGPOINT, 260),

  /* Enable/disable specific SSL features with a bitmask for proxy, see
     CURLSSLOPT_* */
  CURLOPT(CURLOPT_PROXY_SSL_OPTIONS, CURLOPTTYPE_LONG, 261),

  /* Name of pre proxy to use. */
  CURLOPT(CURLOPT_PRE_PROXY, CURLOPTTYPE_STRINGPOINT, 262),

  /* The public key in DER form used to validate the proxy public key
     this option is used only if PROXY_SSL_VERIFYPEER is true */
  CURLOPT(CURLOPT_PROXY_PINNEDPUBLICKEY, CURLOPTTYPE_STRINGPOINT, 263),

  /* Path to an abstract Unix domain socket */
  CURLOPT(CURLOPT_ABSTRACT_UNIX_SOCKET, CURLOPTTYPE_STRINGPOINT, 264),

  /* Suppress proxy CONNECT response headers from user callbacks */
  CURLOPT(CURLOPT_SUPPRESS_CONNECT_HEADERS, CURLOPTTYPE_LONG, 265),

  /* The request target, instead of extracted from the URL */
  CURLOPT(CURLOPT_REQUEST_TARGET, CURLOPTTYPE_STRINGPOINT, 266),

  /* bitmask of allowed auth methods for connections to SOCKS5 proxies */
  CURLOPT(CURLOPT_SOCKS5_AUTH, CURLOPTTYPE_LONG, 267),

  /* Enable/disable SSH compression */
  CURLOPT(CURLOPT_SSH_COMPRESSION, CURLOPTTYPE_LONG, 268),

  /* Post MIME data. */
  CURLOPT(CURLOPT_MIMEPOST, CURLOPTTYPE_OBJECTPOINT, 269),

  /* Time to use with the CURLOPT_TIMECONDITION. Specified in number of
     seconds since 1 Jan 1970. */
  CURLOPT(CURLOPT_TIMEVALUE_LARGE, CURLOPTTYPE_OFF_T, 270),

  /* Head start in milliseconds to give happy eyeballs. */
  CURLOPT(CURLOPT_HAPPY_EYEBALLS_TIMEOUT_MS, CURLOPTTYPE_LONG, 271),

  /* Function that will be called before a resolver request is made */
  CURLOPT(CURLOPT_RESOLVER_START_FUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 272),

  /* User data to pass to the resolver start callback. */
  CURLOPT(CURLOPT_RESOLVER_START_DATA, CURLOPTTYPE_CBPOINT, 273),

  /* send HAProxy PROXY protocol header? */
  CURLOPT(CURLOPT_HAPROXYPROTOCOL, CURLOPTTYPE_LONG, 274),

  /* shuffle addresses before use when DNS returns multiple */
  CURLOPT(CURLOPT_DNS_SHUFFLE_ADDRESSES, CURLOPTTYPE_LONG, 275),

  /* Specify which TLS 1.3 ciphers suites to use */
  CURLOPT(CURLOPT_TLS13_CIPHERS, CURLOPTTYPE_STRINGPOINT, 276),
  CURLOPT(CURLOPT_PROXY_TLS13_CIPHERS, CURLOPTTYPE_STRINGPOINT, 277),

  /* Disallow specifying username/login in URL. */
  CURLOPT(CURLOPT_DISALLOW_USERNAME_IN_URL, CURLOPTTYPE_LONG, 278),

  /* DNS-over-HTTPS URL */
  CURLOPT(CURLOPT_DOH_URL, CURLOPTTYPE_STRINGPOINT, 279),

  /* Preferred buffer size to use for uploads */
  CURLOPT(CURLOPT_UPLOAD_BUFFERSIZE, CURLOPTTYPE_LONG, 280),

  /* Time in ms between connection upkeep calls for long-lived connections. */
  CURLOPT(CURLOPT_UPKEEP_INTERVAL_MS, CURLOPTTYPE_LONG, 281),

  /* Specify URL using CURL URL API. */
  CURLOPT(CURLOPT_CURLU, CURLOPTTYPE_OBJECTPOINT, 282),

  /* add trailing data just after no more data is available */
  CURLOPT(CURLOPT_TRAILERFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 283),

  /* pointer to be passed to HTTP_TRAILER_FUNCTION */
  CURLOPT(CURLOPT_TRAILERDATA, CURLOPTTYPE_CBPOINT, 284),

  /* set this to 1L to allow HTTP/0.9 responses or 0L to disallow */
  CURLOPT(CURLOPT_HTTP09_ALLOWED, CURLOPTTYPE_LONG, 285),

  /* alt-svc control bitmask */
  CURLOPT(CURLOPT_ALTSVC_CTRL, CURLOPTTYPE_LONG, 286),

  /* alt-svc cache file name to possibly read from/write to */
  CURLOPT(CURLOPT_ALTSVC, CURLOPTTYPE_STRINGPOINT, 287),

  /* maximum age (idle time) of a connection to consider it for reuse
   * (in seconds) */
  CURLOPT(CURLOPT_MAXAGE_CONN, CURLOPTTYPE_LONG, 288),

  /* SASL authorisation identity */
  CURLOPT(CURLOPT_SASL_AUTHZID, CURLOPTTYPE_STRINGPOINT, 289),

  /* allow RCPT TO command to fail for some recipients */
  CURLOPT(CURLOPT_MAIL_RCPT_ALLLOWFAILS, CURLOPTTYPE_LONG, 290),

  /* the private SSL-certificate as a "blob" */
  CURLOPT(CURLOPT_SSLCERT_BLOB, CURLOPTTYPE_BLOB, 291),
  CURLOPT(CURLOPT_SSLKEY_BLOB, CURLOPTTYPE_BLOB, 292),
  CURLOPT(CURLOPT_PROXY_SSLCERT_BLOB, CURLOPTTYPE_BLOB, 293),
  CURLOPT(CURLOPT_PROXY_SSLKEY_BLOB, CURLOPTTYPE_BLOB, 294),
  CURLOPT(CURLOPT_ISSUERCERT_BLOB, CURLOPTTYPE_BLOB, 295),

  /* Issuer certificate for proxy */
  CURLOPT(CURLOPT_PROXY_ISSUERCERT, CURLOPTTYPE_STRINGPOINT, 296),
  CURLOPT(CURLOPT_PROXY_ISSUERCERT_BLOB, CURLOPTTYPE_BLOB, 297),

  /* the EC curves requested by the TLS client (RFC 8422, 5.1);
   * OpenSSL support via 'set_groups'/'set_curves':
   * https://www.openssl.org/docs/manmaster/man3/SSL_CTX_set1_groups.html
   */
  CURLOPT(CURLOPT_SSL_EC_CURVES, CURLOPTTYPE_STRINGPOINT, 298),

  /* HSTS bitmask */
  CURLOPT(CURLOPT_HSTS_CTRL, CURLOPTTYPE_LONG, 299),
  /* HSTS file name */
  CURLOPT(CURLOPT_HSTS, CURLOPTTYPE_STRINGPOINT, 300),

  /* HSTS read callback */
  CURLOPT(CURLOPT_HSTSREADFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 301),
  CURLOPT(CURLOPT_HSTSREADDATA, CURLOPTTYPE_CBPOINT, 302),

  /* HSTS write callback */
  CURLOPT(CURLOPT_HSTSWRITEFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 303),
  CURLOPT(CURLOPT_HSTSWRITEDATA, CURLOPTTYPE_CBPOINT, 304),

  /* Parameters for V4 signature */
  CURLOPT(CURLOPT_AWS_SIGV4, CURLOPTTYPE_STRINGPOINT, 305),

  /* Same as CURLOPT_SSL_VERIFYPEER but for DoH (DNS-over-HTTPS) servers. */
  CURLOPT(CURLOPT_DOH_SSL_VERIFYPEER, CURLOPTTYPE_LONG, 306),

  /* Same as CURLOPT_SSL_VERIFYHOST but for DoH (DNS-over-HTTPS) servers. */
  CURLOPT(CURLOPT_DOH_SSL_VERIFYHOST, CURLOPTTYPE_LONG, 307),

  /* Same as CURLOPT_SSL_VERIFYSTATUS but for DoH (DNS-over-HTTPS) servers. */
  CURLOPT(CURLOPT_DOH_SSL_VERIFYSTATUS, CURLOPTTYPE_LONG, 308),

  /* The CA certificates as "blob" used to validate the peer certificate
     this option is used only if SSL_VERIFYPEER is true */
  CURLOPT(CURLOPT_CAINFO_BLOB, CURLOPTTYPE_BLOB, 309),

  /* The CA certificates as "blob" used to validate the proxy certificate
     this option is used only if PROXY_SSL_VERIFYPEER is true */
  CURLOPT(CURLOPT_PROXY_CAINFO_BLOB, CURLOPTTYPE_BLOB, 310),

  /* used by scp/sftp to verify the host's public key */
  CURLOPT(CURLOPT_SSH_HOST_PUBLIC_KEY_SHA256, CURLOPTTYPE_STRINGPOINT, 311),

  /* Function that will be called immediately before the initial request
     is made on a connection (after any protocol negotiation step).  */
  CURLOPT(CURLOPT_PREREQFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 312),

  /* Data passed to the CURLOPT_PREREQFUNCTION callback */
  CURLOPT(CURLOPT_PREREQDATA, CURLOPTTYPE_CBPOINT, 313),

  /* maximum age (since creation) of a connection to consider it for reuse
   * (in seconds) */
  CURLOPT(CURLOPT_MAXLIFETIME_CONN, CURLOPTTYPE_LONG, 314),

  /* Set MIME option flags. */
  CURLOPT(CURLOPT_MIME_OPTIONS, CURLOPTTYPE_LONG, 315),

  CURLOPT_LASTENTRY /* the last unused */
} CURLoption;

#ifndef CURL_NO_OLDIES /* define this to test if your app builds with all
                          the obsolete stuff removed! */

/* Backwards compatibility with older names */
/* These are scheduled to disappear by 2011 */

/* This was added in version 7.19.1 */
#define CURLOPT_POST301 CURLOPT_POSTREDIR

/* These are scheduled to disappear by 2009 */

/* The following were added in 7.17.0 */
#define CURLOPT_SSLKEYPASSWD CURLOPT_KEYPASSWD
#define CURLOPT_FTPAPPEND CURLOPT_APPEND
#define CURLOPT_FTPLISTONLY CURLOPT_DIRLISTONLY
#define CURLOPT_FTP_SSL CURLOPT_USE_SSL

/* The following were added earlier */

#define CURLOPT_SSLCERTPASSWD CURLOPT_KEYPASSWD
#define CURLOPT_KRB4LEVEL CURLOPT_KRBLEVEL

#else
/* This is set if CURL_NO_OLDIES is defined at compile-time */
#undef CURLOPT_DNS_USE_GLOBAL_CACHE /* soon obsolete */
#endif


  /* Below here follows defines for the CURLOPT_IPRESOLVE option. If a host
     name resolves addresses using more than one IP protocol version, this
     option might be handy to force libcurl to use a specific IP version. */
#define CURL_IPRESOLVE_WHATEVER 0 /* default, uses addresses to all IP
                                     versions that your system allows */
#define CURL_IPRESOLVE_V4       1 /* uses only IPv4 addresses/connections */
#define CURL_IPRESOLVE_V6       2 /* uses only IPv6 addresses/connections */

  /* three convenient "aliases" that follow the name scheme better */
#define CURLOPT_RTSPHEADER CURLOPT_HTTPHEADER

  /* These enums are for use with the CURLOPT_HTTP_VERSION option. */
enum {
  CURL_HTTP_VERSION_NONE, /* setting this means we don't care, and that we'd
                             like the library to choose the best possible
                             for us! */
  CURL_HTTP_VERSION_1_0,  /* please use HTTP 1.0 in the request */
  CURL_HTTP_VERSION_1_1,  /* please use HTTP 1.1 in the request */
  CURL_HTTP_VERSION_2_0,  /* please use HTTP 2 in the request */
  CURL_HTTP_VERSION_2TLS, /* use version 2 for HTTPS, version 1.1 for HTTP */
  CURL_HTTP_VERSION_2_PRIOR_KNOWLEDGE,  /* please use HTTP 2 without HTTP/1.1
                                           Upgrade */
  CURL_HTTP_VERSION_3 = 30, /* Makes use of explicit HTTP/3 without fallback.
                               Use CURLOPT_ALTSVC to enable HTTP/3 upgrade */
  CURL_HTTP_VERSION_LAST /* *ILLEGAL* http version */
};

/* Convenience definition simple because the name of the version is HTTP/2 and
   not 2.0. The 2_0 version of the enum name was set while the version was
   still planned to be 2.0 and we stick to it for compatibility. */
#define CURL_HTTP_VERSION_2 CURL_HTTP_VERSION_2_0

/*
 * Public API enums for RTSP requests
 */
enum {
    CURL_RTSPREQ_NONE, /* first in list */
    CURL_RTSPREQ_OPTIONS,
    CURL_RTSPREQ_DESCRIBE,
    CURL_RTSPREQ_ANNOUNCE,
    CURL_RTSPREQ_SETUP,
    CURL_RTSPREQ_PLAY,
    CURL_RTSPREQ_PAUSE,
    CURL_RTSPREQ_TEARDOWN,
    CURL_RTSPREQ_GET_PARAMETER,
    CURL_RTSPREQ_SET_PARAMETER,
    CURL_RTSPREQ_RECORD,
    CURL_RTSPREQ_RECEIVE,
    CURL_RTSPREQ_LAST /* last in list */
};

  /* These enums are for use with the CURLOPT_NETRC option. */
enum CURL_NETRC_OPTION {
  CURL_NETRC_IGNORED,     /* The .netrc will never be read.
                           * This is the default. */
  CURL_NETRC_OPTIONAL,    /* A user:password in the URL will be preferred
                           * to one in the .netrc. */
  CURL_NETRC_REQUIRED,    /* A user:password in the URL will be ignored.
                           * Unless one is set programmatically, the .netrc
                           * will be queried. */
  CURL_NETRC_LAST
};

enum {
  CURL_SSLVERSION_DEFAULT,
  CURL_SSLVERSION_TLSv1, /* TLS 1.x */
  CURL_SSLVERSION_SSLv2,
  CURL_SSLVERSION_SSLv3,
  CURL_SSLVERSION_TLSv1_0,
  CURL_SSLVERSION_TLSv1_1,
  CURL_SSLVERSION_TLSv1_2,
  CURL_SSLVERSION_TLSv1_3,

  CURL_SSLVERSION_LAST /* never use, keep last */
};

enum {
  CURL_SSLVERSION_MAX_NONE =     0,
  CURL_SSLVERSION_MAX_DEFAULT =  (CURL_SSLVERSION_TLSv1   << 16),
  CURL_SSLVERSION_MAX_TLSv1_0 =  (CURL_SSLVERSION_TLSv1_0 << 16),
  CURL_SSLVERSION_MAX_TLSv1_1 =  (CURL_SSLVERSION_TLSv1_1 << 16),
  CURL_SSLVERSION_MAX_TLSv1_2 =  (CURL_SSLVERSION_TLSv1_2 << 16),
  CURL_SSLVERSION_MAX_TLSv1_3 =  (CURL_SSLVERSION_TLSv1_3 << 16),

  /* never use, keep last */
  CURL_SSLVERSION_MAX_LAST =     (CURL_SSLVERSION_LAST    << 16)
};

enum CURL_TLSAUTH {
  CURL_TLSAUTH_NONE,
  CURL_TLSAUTH_SRP,
  CURL_TLSAUTH_LAST /* never use, keep last */
};

/* symbols to use with CURLOPT_POSTREDIR.
   CURL_REDIR_POST_301, CURL_REDIR_POST_302 and CURL_REDIR_POST_303
   can be bitwise ORed so that CURL_REDIR_POST_301 | CURL_REDIR_POST_302
   | CURL_REDIR_POST_303 == CURL_REDIR_POST_ALL */

#define CURL_REDIR_GET_ALL  0
#define CURL_REDIR_POST_301 1
#define CURL_REDIR_POST_302 2
#define CURL_REDIR_POST_303 4
#define CURL_REDIR_POST_ALL \
    (CURL_REDIR_POST_301|CURL_REDIR_POST_302|CURL_REDIR_POST_303)

typedef enum {
  CURL_TIMECOND_NONE,

  CURL_TIMECOND_IFMODSINCE,
  CURL_TIMECOND_IFUNMODSINCE,
  CURL_TIMECOND_LASTMOD,

  CURL_TIMECOND_LAST
} curl_TimeCond;

/* Special size_t value signaling a null-terminated string. */
#define CURL_ZERO_TERMINATED ((size_t) -1)

/* curl_strequal() and curl_strnequal() are subject for removal in a future
   release */
CURL_EXTERN int curl_strequal(const char *s1, const char *s2);
CURL_EXTERN int curl_strnequal(const char *s1, const char *s2, size_t n);

/* Mime/form handling support. */
typedef struct curl_mime      curl_mime;      /* Mime context. */
typedef struct curl_mimepart  curl_mimepart;  /* Mime part context. */

/* CURLMIMEOPT_ defines are for the CURLOPT_MIME_OPTIONS option. */
#define CURLMIMEOPT_FORMESCAPE  (1<<0) /* Use backslash-escaping for forms. */

/*
 * NAME curl_mime_init()
 *
 * DESCRIPTION
 *
 * Create a mime context and return its handle. The easy parameter is the
 * target handle.
 */
CURL_EXTERN curl_mime *curl_mime_init(CURL *easy);

/*
 * NAME curl_mime_free()
 *
 * DESCRIPTION
 *
 * release a mime handle and its substructures.
 */
CURL_EXTERN void curl_mime_free(curl_mime *mime);

/*
 * NAME curl_mime_addpart()
 *
 * DESCRIPTION
 *
 * Append a new empty part to the given mime context and return a handle to
 * the created part.
 */
CURL_EXTERN curl_mimepart *curl_mime_addpart(curl_mime *mime);

/*
 * NAME curl_mime_name()
 *
 * DESCRIPTION
 *
 * Set mime/form part name.
 */
CURL_EXTERN CURLcode curl_mime_name(curl_mimepart *part, const char *name);

/*
 * NAME curl_mime_filename()
 *
 * DESCRIPTION
 *
 * Set mime part remote file name.
 */
CURL_EXTERN CURLcode curl_mime_filename(curl_mimepart *part,
                                        const char *filename);

/*
 * NAME curl_mime_type()
 *
 * DESCRIPTION
 *
 * Set mime part type.
 */
CURL_EXTERN CURLcode curl_mime_type(curl_mimepart *part, const char *mimetype);

/*
 * NAME curl_mime_encoder()
 *
 * DESCRIPTION
 *
 * Set mime data transfer encoder.
 */
CURL_EXTERN CURLcode curl_mime_encoder(curl_mimepart *part,
                                       const char *encoding);

/*
 * NAME curl_mime_data()
 *
 * DESCRIPTION
 *
 * Set mime part data source from memory data,
 */
CURL_EXTERN CURLcode curl_mime_data(curl_mimepart *part,
                                    const char *data, size_t datasize);

/*
 * NAME curl_mime_filedata()
 *
 * DESCRIPTION
 *
 * Set mime part data source from named file.
 */
CURL_EXTERN CURLcode curl_mime_filedata(curl_mimepart *part,
                                        const char *filename);

/*
 * NAME curl_mime_data_cb()
 *
 * DESCRIPTION
 *
 * Set mime part data source from callback function.
 */
CURL_EXTERN CURLcode curl_mime_data_cb(curl_mimepart *part,
                                       curl_off_t datasize,
                                       curl_read_callback readfunc,
                                       curl_seek_callback seekfunc,
                                       curl_free_callback freefunc,
                                       void *arg);

/*
 * NAME curl_mime_subparts()
 *
 * DESCRIPTION
 *
 * Set mime part data source from subparts.
 */
CURL_EXTERN CURLcode curl_mime_subparts(curl_mimepart *part,
                                        curl_mime *subparts);
/*
 * NAME curl_mime_headers()
 *
 * DESCRIPTION
 *
 * Set mime part headers.
 */
CURL_EXTERN CURLcode curl_mime_headers(curl_mimepart *part,
                                       struct curl_slist *headers,
                                       int take_ownership);

typedef enum {
  CURLFORM_NOTHING,        /********* the first one is unused ************/
  CURLFORM_COPYNAME,
  CURLFORM_PTRNAME,
  CURLFORM_NAMELENGTH,
  CURLFORM_COPYCONTENTS,
  CURLFORM_PTRCONTENTS,
  CURLFORM_CONTENTSLENGTH,
  CURLFORM_FILECONTENT,
  CURLFORM_ARRAY,
  CURLFORM_OBSOLETE,
  CURLFORM_FILE,

  CURLFORM_BUFFER,
  CURLFORM_BUFFERPTR,
  CURLFORM_BUFFERLENGTH,

  CURLFORM_CONTENTTYPE,
  CURLFORM_CONTENTHEADER,
  CURLFORM_FILENAME,
  CURLFORM_END,
  CURLFORM_OBSOLETE2,

  CURLFORM_STREAM,
  CURLFORM_CONTENTLEN, /* added in 7.46.0, provide a curl_off_t length */

  CURLFORM_LASTENTRY /* the last unused */
} CURLformoption;

/* structure to be used as parameter for CURLFORM_ARRAY */
struct curl_forms {
  CURLformoption option;
  const char     *value;
};

/* use this for multipart formpost building */
/* Returns code for curl_formadd()
 *
 * Returns:
 * CURL_FORMADD_OK             on success
 * CURL_FORMADD_MEMORY         if the FormInfo allocation fails
 * CURL_FORMADD_OPTION_TWICE   if one option is given twice for one Form
 * CURL_FORMADD_NULL           if a null pointer was given for a char
 * CURL_FORMADD_MEMORY         if the allocation of a FormInfo struct failed
 * CURL_FORMADD_UNKNOWN_OPTION if an unknown option was used
 * CURL_FORMADD_INCOMPLETE     if the some FormInfo is not complete (or error)
 * CURL_FORMADD_MEMORY         if a curl_httppost struct cannot be allocated
 * CURL_FORMADD_MEMORY         if some allocation for string copying failed.
 * CURL_FORMADD_ILLEGAL_ARRAY  if an illegal option is used in an array
 *
 ***************************************************************************/
typedef enum {
  CURL_FORMADD_OK, /* first, no error */

  CURL_FORMADD_MEMORY,
  CURL_FORMADD_OPTION_TWICE,
  CURL_FORMADD_NULL,
  CURL_FORMADD_UNKNOWN_OPTION,
  CURL_FORMADD_INCOMPLETE,
  CURL_FORMADD_ILLEGAL_ARRAY,
  CURL_FORMADD_DISABLED, /* libcurl was built with this disabled */

  CURL_FORMADD_LAST /* last */
} CURLFORMcode;

/*
 * NAME curl_formadd()
 *
 * DESCRIPTION
 *
 * Pretty advanced function for building multi-part formposts. Each invoke
 * adds one part that together construct a full post. Then use
 * CURLOPT_HTTPPOST to send it off to libcurl.
 */
CURL_EXTERN CURLFORMcode curl_formadd(struct curl_httppost **httppost,
                                      struct curl_httppost **last_post,
                                      ...);

/*
 * callback function for curl_formget()
 * The void *arg pointer will be the one passed as second argument to
 *   curl_formget().
 * The character buffer passed to it must not be freed.
 * Should return the buffer length passed to it as the argument "len" on
 *   success.
 */
typedef size_t (*curl_formget_callback)(void *arg, const char *buf,
                                        size_t len);

/*
 * NAME curl_formget()
 *
 * DESCRIPTION
 *
 * Serialize a curl_httppost struct built with curl_formadd().
 * Accepts a void pointer as second argument which will be passed to
 * the curl_formget_callback function.
 * Returns 0 on success.
 */
CURL_EXTERN int curl_formget(struct curl_httppost *form, void *arg,
                             curl_formget_callback append);
/*
 * NAME curl_formfree()
 *
 * DESCRIPTION
 *
 * Free a multipart formpost previously built with curl_formadd().
 */
CURL_EXTERN void curl_formfree(struct curl_httppost *form);

/*
 * NAME curl_getenv()
 *
 * DESCRIPTION
 *
 * Returns a malloc()'ed string that MUST be curl_free()ed after usage is
 * complete. DEPRECATED - see lib/README.curlx
 */
CURL_EXTERN char *curl_getenv(const char *variable);

/*
 * NAME curl_version()
 *
 * DESCRIPTION
 *
 * Returns a static ascii string of the libcurl version.
 */
CURL_EXTERN char *curl_version(void);

/*
 * NAME curl_easy_escape()
 *
 * DESCRIPTION
 *
 * Escapes URL strings (converts all letters consider illegal in URLs to their
 * %XX versions). This function returns a new allocated string or NULL if an
 * error occurred.
 */
CURL_EXTERN char *curl_easy_escape(CURL *handle,
                                   const char *string,
                                   int length);

/* the previous version: */
CURL_EXTERN char *curl_escape(const char *string,
                              int length);


/*
 * NAME curl_easy_unescape()
 *
 * DESCRIPTION
 *
 * Unescapes URL encoding in strings (converts all %XX codes to their 8bit
 * versions). This function returns a new allocated string or NULL if an error
 * occurred.
 * Conversion Note: On non-ASCII platforms the ASCII %XX codes are
 * converted into the host encoding.
 */
CURL_EXTERN char *curl_easy_unescape(CURL *handle,
                                     const char *string,
                                     int length,
                                     int *outlength);

/* the previous version */
CURL_EXTERN char *curl_unescape(const char *string,
                                int length);

/*
 * NAME curl_free()
 *
 * DESCRIPTION
 *
 * Provided for de-allocation in the same translation unit that did the
 * allocation. Added in libcurl 7.10
 */
CURL_EXTERN void curl_free(void *p);

/*
 * NAME curl_global_init()
 *
 * DESCRIPTION
 *
 * curl_global_init() should be invoked exactly once for each application that
 * uses libcurl and before any call of other libcurl functions.
 *
 * This function is not thread-safe!
 */
CURL_EXTERN CURLcode curl_global_init(long flags);

/*
 * NAME curl_global_init_mem()
 *
 * DESCRIPTION
 *
 * curl_global_init() or curl_global_init_mem() should be invoked exactly once
 * for each application that uses libcurl.  This function can be used to
 * initialize libcurl and set user defined memory management callback
 * functions.  Users can implement memory management routines to check for
 * memory leaks, check for mis-use of the curl library etc.  User registered
 * callback routines will be invoked by this library instead of the system
 * memory management routines like malloc, free etc.
 */
CURL_EXTERN CURLcode curl_global_init_mem(long flags,
                                          curl_malloc_callback m,
                                          curl_free_callback f,
                                          curl_realloc_callback r,
                                          curl_strdup_callback s,
                                          curl_calloc_callback c);

/*
 * NAME curl_global_cleanup()
 *
 * DESCRIPTION
 *
 * curl_global_cleanup() should be invoked exactly once for each application
 * that uses libcurl
 */
CURL_EXTERN void curl_global_cleanup(void);

/* linked-list structure for the CURLOPT_QUOTE option (and other) */
struct curl_slist {
  char *data;
  struct curl_slist *next;
};

/*
 * NAME curl_global_sslset()
 *
 * DESCRIPTION
 *
 * When built with multiple SSL backends, curl_global_sslset() allows to
 * choose one. This function can only be called once, and it must be called
 * *before* curl_global_init().
 *
 * The backend can be identified by the id (e.g. CURLSSLBACKEND_OPENSSL). The
 * backend can also be specified via the name parameter (passing -1 as id).
 * If both id and name are specified, the name will be ignored. If neither id
 * nor name are specified, the function will fail with
 * CURLSSLSET_UNKNOWN_BACKEND and set the "avail" pointer to the
 * NULL-terminated list of available backends.
 *
 * Upon success, the function returns CURLSSLSET_OK.
 *
 * If the specified SSL backend is not available, the function returns
 * CURLSSLSET_UNKNOWN_BACKEND and sets the "avail" pointer to a NULL-terminated
 * list of available SSL backends.
 *
 * The SSL backend can be set only once. If it has already been set, a
 * subsequent attempt to change it will result in a CURLSSLSET_TOO_LATE.
 */

struct curl_ssl_backend {
  curl_sslbackend id;
  const char *name;
};
typedef struct curl_ssl_backend curl_ssl_backend;

typedef enum {
  CURLSSLSET_OK = 0,
  CURLSSLSET_UNKNOWN_BACKEND,
  CURLSSLSET_TOO_LATE,
  CURLSSLSET_NO_BACKENDS /* libcurl was built without any SSL support */
} CURLsslset;

CURL_EXTERN CURLsslset curl_global_sslset(curl_sslbackend id, const char *name,
                                          const curl_ssl_backend ***avail);

/*
 * NAME curl_slist_append()
 *
 * DESCRIPTION
 *
 * Appends a string to a linked list. If no list exists, it will be created
 * first. Returns the new list, after appending.
 */
CURL_EXTERN struct curl_slist *curl_slist_append(struct curl_slist *,
                                                 const char *);

/*
 * NAME curl_slist_free_all()
 *
 * DESCRIPTION
 *
 * free a previously built curl_slist.
 */
CURL_EXTERN void curl_slist_free_all(struct curl_slist *);

/*
 * NAME curl_getdate()
 *
 * DESCRIPTION
 *
 * Returns the time, in seconds since 1 Jan 1970 of the time string given in
 * the first argument. The time argument in the second parameter is unused
 * and should be set to NULL.
 */
CURL_EXTERN time_t curl_getdate(const char *p, const time_t *unused);

/* info about the certificate chain, only for OpenSSL, GnuTLS, Schannel, NSS
   and GSKit builds. Asked for with CURLOPT_CERTINFO / CURLINFO_CERTINFO */
struct curl_certinfo {
  int num_of_certs;             /* number of certificates with information */
  struct curl_slist **certinfo; /* for each index in this array, there's a
                                   linked list with textual information in the
                                   format "name: value" */
};

/* Information about the SSL library used and the respective internal SSL
   handle, which can be used to obtain further information regarding the
   connection. Asked for with CURLINFO_TLS_SSL_PTR or CURLINFO_TLS_SESSION. */
struct curl_tlssessioninfo {
  curl_sslbackend backend;
  void *internals;
};

#define CURLINFO_STRING   0x100000
#define CURLINFO_LONG     0x200000
#define CURLINFO_DOUBLE   0x300000
#define CURLINFO_SLIST    0x400000
#define CURLINFO_PTR      0x400000 /* same as SLIST */
#define CURLINFO_SOCKET   0x500000
#define CURLINFO_OFF_T    0x600000
#define CURLINFO_MASK     0x0fffff
#define CURLINFO_TYPEMASK 0xf00000

typedef enum {
  CURLINFO_NONE, /* first, never use this */
  CURLINFO_EFFECTIVE_URL    = CURLINFO_STRING + 1,
  CURLINFO_RESPONSE_CODE    = CURLINFO_LONG   + 2,
  CURLINFO_TOTAL_TIME       = CURLINFO_DOUBLE + 3,
  CURLINFO_NAMELOOKUP_TIME  = CURLINFO_DOUBLE + 4,
  CURLINFO_CONNECT_TIME     = CURLINFO_DOUBLE + 5,
  CURLINFO_PRETRANSFER_TIME = CURLINFO_DOUBLE + 6,
  CURLINFO_SIZE_UPLOAD      = CURLINFO_DOUBLE + 7,
  CURLINFO_SIZE_UPLOAD_T    = CURLINFO_OFF_T  + 7,
  CURLINFO_SIZE_DOWNLOAD    = CURLINFO_DOUBLE + 8,
  CURLINFO_SIZE_DOWNLOAD_T  = CURLINFO_OFF_T  + 8,
  CURLINFO_SPEED_DOWNLOAD   = CURLINFO_DOUBLE + 9,
  CURLINFO_SPEED_DOWNLOAD_T = CURLINFO_OFF_T  + 9,
  CURLINFO_SPEED_UPLOAD     = CURLINFO_DOUBLE + 10,
  CURLINFO_SPEED_UPLOAD_T   = CURLINFO_OFF_T  + 10,
  CURLINFO_HEADER_SIZE      = CURLINFO_LONG   + 11,
  CURLINFO_REQUEST_SIZE     = CURLINFO_LONG   + 12,
  CURLINFO_SSL_VERIFYRESULT = CURLINFO_LONG   + 13,
  CURLINFO_FILETIME         = CURLINFO_LONG   + 14,
  CURLINFO_FILETIME_T       = CURLINFO_OFF_T  + 14,
  CURLINFO_CONTENT_LENGTH_DOWNLOAD   = CURLINFO_DOUBLE + 15,
  CURLINFO_CONTENT_LENGTH_DOWNLOAD_T = CURLINFO_OFF_T  + 15,
  CURLINFO_CONTENT_LENGTH_UPLOAD     = CURLINFO_DOUBLE + 16,
  CURLINFO_CONTENT_LENGTH_UPLOAD_T   = CURLINFO_OFF_T  + 16,
  CURLINFO_STARTTRANSFER_TIME = CURLINFO_DOUBLE + 17,
  CURLINFO_CONTENT_TYPE     = CURLINFO_STRING + 18,
  CURLINFO_REDIRECT_TIME    = CURLINFO_DOUBLE + 19,
  CURLINFO_REDIRECT_COUNT   = CURLINFO_LONG   + 20,
  CURLINFO_PRIVATE          = CURLINFO_STRING + 21,
  CURLINFO_HTTP_CONNECTCODE = CURLINFO_LONG   + 22,
  CURLINFO_HTTPAUTH_AVAIL   = CURLINFO_LONG   + 23,
  CURLINFO_PROXYAUTH_AVAIL  = CURLINFO_LONG   + 24,
  CURLINFO_OS_ERRNO         = CURLINFO_LONG   + 25,
  CURLINFO_NUM_CONNECTS     = CURLINFO_LONG   + 26,
  CURLINFO_SSL_ENGINES      = CURLINFO_SLIST  + 27,
  CURLINFO_COOKIELIST       = CURLINFO_SLIST  + 28,
  CURLINFO_LASTSOCKET       = CURLINFO_LONG   + 29,
  CURLINFO_FTP_ENTRY_PATH   = CURLINFO_STRING + 30,
  CURLINFO_REDIRECT_URL     = CURLINFO_STRING + 31,
  CURLINFO_PRIMARY_IP       = CURLINFO_STRING + 32,
  CURLINFO_APPCONNECT_TIME  = CURLINFO_DOUBLE + 33,
  CURLINFO_CERTINFO         = CURLINFO_PTR    + 34,
  CURLINFO_CONDITION_UNMET  = CURLINFO_LONG   + 35,
  CURLINFO_RTSP_SESSION_ID  = CURLINFO_STRING + 36,
  CURLINFO_RTSP_CLIENT_CSEQ = CURLINFO_LONG   + 37,
  CURLINFO_RTSP_SERVER_CSEQ = CURLINFO_LONG   + 38,
  CURLINFO_RTSP_CSEQ_RECV   = CURLINFO_LONG   + 39,
  CURLINFO_PRIMARY_PORT     = CURLINFO_LONG   + 40,
  CURLINFO_LOCAL_IP         = CURLINFO_STRING + 41,
  CURLINFO_LOCAL_PORT       = CURLINFO_LONG   + 42,
  CURLINFO_TLS_SESSION      = CURLINFO_PTR    + 43,
  CURLINFO_ACTIVESOCKET     = CURLINFO_SOCKET + 44,
  CURLINFO_TLS_SSL_PTR      = CURLINFO_PTR    + 45,
  CURLINFO_HTTP_VERSION     = CURLINFO_LONG   + 46,
  CURLINFO_PROXY_SSL_VERIFYRESULT = CURLINFO_LONG + 47,
  CURLINFO_PROTOCOL         = CURLINFO_LONG   + 48,
  CURLINFO_SCHEME           = CURLINFO_STRING + 49,
  CURLINFO_TOTAL_TIME_T     = CURLINFO_OFF_T + 50,
  CURLINFO_NAMELOOKUP_TIME_T = CURLINFO_OFF_T + 51,
  CURLINFO_CONNECT_TIME_T   = CURLINFO_OFF_T + 52,
  CURLINFO_PRETRANSFER_TIME_T = CURLINFO_OFF_T + 53,
  CURLINFO_STARTTRANSFER_TIME_T = CURLINFO_OFF_T + 54,
  CURLINFO_REDIRECT_TIME_T  = CURLINFO_OFF_T + 55,
  CURLINFO_APPCONNECT_TIME_T = CURLINFO_OFF_T + 56,
  CURLINFO_RETRY_AFTER      = CURLINFO_OFF_T + 57,
  CURLINFO_EFFECTIVE_METHOD = CURLINFO_STRING + 58,
  CURLINFO_PROXY_ERROR      = CURLINFO_LONG + 59,
  CURLINFO_REFERER          = CURLINFO_STRING + 60,

  CURLINFO_LASTONE          = 60
} CURLINFO;

/* CURLINFO_RESPONSE_CODE is the new name for the option previously known as
   CURLINFO_HTTP_CODE */
#define CURLINFO_HTTP_CODE CURLINFO_RESPONSE_CODE

typedef enum {
  CURLCLOSEPOLICY_NONE, /* first, never use this */

  CURLCLOSEPOLICY_OLDEST,
  CURLCLOSEPOLICY_LEAST_RECENTLY_USED,
  CURLCLOSEPOLICY_LEAST_TRAFFIC,
  CURLCLOSEPOLICY_SLOWEST,
  CURLCLOSEPOLICY_CALLBACK,

  CURLCLOSEPOLICY_LAST /* last, never use this */
} curl_closepolicy;

#define CURL_GLOBAL_SSL (1<<0) /* no purpose since since 7.57.0 */
#define CURL_GLOBAL_WIN32 (1<<1)
#define CURL_GLOBAL_ALL (CURL_GLOBAL_SSL|CURL_GLOBAL_WIN32)
#define CURL_GLOBAL_NOTHING 0
#define CURL_GLOBAL_DEFAULT CURL_GLOBAL_ALL
#define CURL_GLOBAL_ACK_EINTR (1<<2)


/*****************************************************************************
 * Setup defines, protos etc for the sharing stuff.
 */

/* Different data locks for a single share */
typedef enum {
  CURL_LOCK_DATA_NONE = 0,
  /*  CURL_LOCK_DATA_SHARE is used internally to say that
   *  the locking is just made to change the internal state of the share
   *  itself.
   */
  CURL_LOCK_DATA_SHARE,
  CURL_LOCK_DATA_COOKIE,
  CURL_LOCK_DATA_DNS,
  CURL_LOCK_DATA_SSL_SESSION,
  CURL_LOCK_DATA_CONNECT,
  CURL_LOCK_DATA_PSL,
  CURL_LOCK_DATA_LAST
} curl_lock_data;

/* Different lock access types */
typedef enum {
  CURL_LOCK_ACCESS_NONE = 0,   /* unspecified action */
  CURL_LOCK_ACCESS_SHARED = 1, /* for read perhaps */
  CURL_LOCK_ACCESS_SINGLE = 2, /* for write perhaps */
  CURL_LOCK_ACCESS_LAST        /* never use */
} curl_lock_access;

typedef void (*curl_lock_function)(CURL *handle,
                                   curl_lock_data data,
                                   curl_lock_access locktype,
                                   void *userptr);
typedef void (*curl_unlock_function)(CURL *handle,
                                     curl_lock_data data,
                                     void *userptr);


typedef enum {
  CURLSHE_OK,  /* all is fine */
  CURLSHE_BAD_OPTION, /* 1 */
  CURLSHE_IN_USE,     /* 2 */
  CURLSHE_INVALID,    /* 3 */
  CURLSHE_NOMEM,      /* 4 out of memory */
  CURLSHE_NOT_BUILT_IN, /* 5 feature not present in lib */
  CURLSHE_LAST        /* never use */
} CURLSHcode;

typedef enum {
  CURLSHOPT_NONE,  /* don't use */
  CURLSHOPT_SHARE,   /* specify a data type to share */
  CURLSHOPT_UNSHARE, /* specify which data type to stop sharing */
  CURLSHOPT_LOCKFUNC,   /* pass in a 'curl_lock_function' pointer */
  CURLSHOPT_UNLOCKFUNC, /* pass in a 'curl_unlock_function' pointer */
  CURLSHOPT_USERDATA,   /* pass in a user data pointer used in the lock/unlock
                           callback functions */
  CURLSHOPT_LAST  /* never use */
} CURLSHoption;

CURL_EXTERN CURLSH *curl_share_init(void);
CURL_EXTERN CURLSHcode curl_share_setopt(CURLSH *, CURLSHoption option, ...);
CURL_EXTERN CURLSHcode curl_share_cleanup(CURLSH *);

/****************************************************************************
 * Structures for querying information about the curl library at runtime.
 */

typedef enum {
  CURLVERSION_FIRST,
  CURLVERSION_SECOND,
  CURLVERSION_THIRD,
  CURLVERSION_FOURTH,
  CURLVERSION_FIFTH,
  CURLVERSION_SIXTH,
  CURLVERSION_SEVENTH,
  CURLVERSION_EIGHTH,
  CURLVERSION_NINTH,
  CURLVERSION_TENTH,
  CURLVERSION_LAST /* never actually use this */
} CURLversion;

/* The 'CURLVERSION_NOW' is the symbolic name meant to be used by
   basically all programs ever that want to get version information. It is
   meant to be a built-in version number for what kind of struct the caller
   expects. If the struct ever changes, we redefine the NOW to another enum
   from above. */
#define CURLVERSION_NOW CURLVERSION_TENTH

struct curl_version_info_data {
  CURLversion age;          /* age of the returned struct */
  const char *version;      /* LIBCURL_VERSION */
  unsigned int version_num; /* LIBCURL_VERSION_NUM */
  const char *host;         /* OS/host/cpu/machine when configured */
  int features;             /* bitmask, see defines below */
  const char *ssl_version;  /* human readable string */
  long ssl_version_num;     /* not used anymore, always 0 */
  const char *libz_version; /* human readable string */
  /* protocols is terminated by an entry with a NULL protoname */
  const char * const *protocols;

  /* The fields below this were added in CURLVERSION_SECOND */
  const char *ares;
  int ares_num;

  /* This field was added in CURLVERSION_THIRD */
  const char *libidn;

  /* These field were added in CURLVERSION_FOURTH */

  /* Same as '_libiconv_version' if built with HAVE_ICONV */
  int iconv_ver_num;

  const char *libssh_version; /* human readable string */

  /* These fields were added in CURLVERSION_FIFTH */
  unsigned int brotli_ver_num; /* Numeric Brotli version
                                  (MAJOR << 24) | (MINOR << 12) | PATCH */
  const char *brotli_version; /* human readable string. */

  /* These fields were added in CURLVERSION_SIXTH */
  unsigned int nghttp2_ver_num; /* Numeric nghttp2 version
                                   (MAJOR << 16) | (MINOR << 8) | PATCH */
  const char *nghttp2_version; /* human readable string. */
  const char *quic_version;    /* human readable quic (+ HTTP/3) library +
                                  version or NULL */

  /* These fields were added in CURLVERSION_SEVENTH */
  const char *cainfo;          /* the built-in default CURLOPT_CAINFO, might
                                  be NULL */
  const char *capath;          /* the built-in default CURLOPT_CAPATH, might
                                  be NULL */

  /* These fields were added in CURLVERSION_EIGHTH */
  unsigned int zstd_ver_num; /* Numeric Zstd version
                                  (MAJOR << 24) | (MINOR << 12) | PATCH */
  const char *zstd_version; /* human readable string. */

  /* These fields were added in CURLVERSION_NINTH */
  const char *hyper_version; /* human readable string. */

  /* These fields were added in CURLVERSION_TENTH */
  const char *gsasl_version; /* human readable string. */
};
typedef struct curl_version_info_data curl_version_info_data;

#define CURL_VERSION_IPV6         (1<<0)  /* IPv6-enabled */
#define CURL_VERSION_KERBEROS4    (1<<1)  /* Kerberos V4 auth is supported
                                             (deprecated) */
#define CURL_VERSION_SSL          (1<<2)  /* SSL options are present */
#define CURL_VERSION_LIBZ         (1<<3)  /* libz features are present */
#define CURL_VERSION_NTLM         (1<<4)  /* NTLM auth is supported */
#define CURL_VERSION_GSSNEGOTIATE (1<<5)  /* Negotiate auth is supported
                                             (deprecated) */
#define CURL_VERSION_DEBUG        (1<<6)  /* Built with debug capabilities */
#define CURL_VERSION_ASYNCHDNS    (1<<7)  /* Asynchronous DNS resolves */
#define CURL_VERSION_SPNEGO       (1<<8)  /* SPNEGO auth is supported */
#define CURL_VERSION_LARGEFILE    (1<<9)  /* Supports files larger than 2GB */
#define CURL_VERSION_IDN          (1<<10) /* Internationized Domain Names are
                                             supported */
#define CURL_VERSION_SSPI         (1<<11) /* Built against Windows SSPI */
#define CURL_VERSION_CONV         (1<<12) /* Character conversions supported */
#define CURL_VERSION_CURLDEBUG    (1<<13) /* Debug memory tracking supported */
#define CURL_VERSION_TLSAUTH_SRP  (1<<14) /* TLS-SRP auth is supported */
#define CURL_VERSION_NTLM_WB      (1<<15) /* NTLM delegation to winbind helper
                                             is supported */
#define CURL_VERSION_HTTP2        (1<<16) /* HTTP2 support built-in */
#define CURL_VERSION_GSSAPI       (1<<17) /* Built against a GSS-API library */
#define CURL_VERSION_KERBEROS5    (1<<18) /* Kerberos V5 auth is supported */
#define CURL_VERSION_UNIX_SOCKETS (1<<19) /* Unix domain sockets support */
#define CURL_VERSION_PSL          (1<<20) /* Mozilla's Public Suffix List, used
                                             for cookie domain verification */
#define CURL_VERSION_HTTPS_PROXY  (1<<21) /* HTTPS-proxy support built-in */
#define CURL_VERSION_MULTI_SSL    (1<<22) /* Multiple SSL backends available */
#define CURL_VERSION_BROTLI       (1<<23) /* Brotli features are present. */
#define CURL_VERSION_ALTSVC       (1<<24) /* Alt-Svc handling built-in */
#define CURL_VERSION_HTTP3        (1<<25) /* HTTP3 support built-in */
#define CURL_VERSION_ZSTD         (1<<26) /* zstd features are present */
#define CURL_VERSION_UNICODE      (1<<27) /* Unicode support on Windows */
#define CURL_VERSION_HSTS         (1<<28) /* HSTS is supported */
#define CURL_VERSION_GSASL        (1<<29) /* libgsasl is supported */

 /*
 * NAME curl_version_info()
 *
 * DESCRIPTION
 *
 * This function returns a pointer to a static copy of the version info
 * struct. See above.
 */
CURL_EXTERN curl_version_info_data *curl_version_info(CURLversion);

/*
 * NAME curl_easy_strerror()
 *
 * DESCRIPTION
 *
 * The curl_easy_strerror function may be used to turn a CURLcode value
 * into the equivalent human readable error string.  This is useful
 * for printing meaningful error messages.
 */
CURL_EXTERN const char *curl_easy_strerror(CURLcode);

/*
 * NAME curl_share_strerror()
 *
 * DESCRIPTION
 *
 * The curl_share_strerror function may be used to turn a CURLSHcode value
 * into the equivalent human readable error string.  This is useful
 * for printing meaningful error messages.
 */
CURL_EXTERN const char *curl_share_strerror(CURLSHcode);

/*
 * NAME curl_easy_pause()
 *
 * DESCRIPTION
 *
 * The curl_easy_pause function pauses or unpauses transfers. Select the new
 * state by setting the bitmask, use the convenience defines below.
 *
 */
CURL_EXTERN CURLcode curl_easy_pause(CURL *handle, int bitmask);

#define CURLPAUSE_RECV      (1<<0)
#define CURLPAUSE_RECV_CONT (0)

#define CURLPAUSE_SEND      (1<<2)
#define CURLPAUSE_SEND_CONT (0)

#define CURLPAUSE_ALL       (CURLPAUSE_RECV|CURLPAUSE_SEND)
#define CURLPAUSE_CONT      (CURLPAUSE_RECV_CONT|CURLPAUSE_SEND_CONT)

#ifdef  __cplusplus
}
#endif

/* unfortunately, the easy.h and multi.h include files need options and info
  stuff before they can be included! */
#include "easy.h" /* nothing in curl is fun without the easy stuff */
#include "multi.h"
#include "urlapi.h"
#include "options.h"

/* the typechecker doesn't work in C++ (yet) */
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && \
    ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3)) && \
    !defined(__cplusplus) && !defined(CURL_DISABLE_TYPECHECK)
#include "typecheck-gcc.h"
#else
#if defined(__STDC__) && (__STDC__ >= 1)
/* This preprocessor magic that replaces a call with the exact same call is
   only done to make sure application authors pass exactly three arguments
   to these functions. */
#define curl_easy_setopt(handle,opt,param) curl_easy_setopt(handle,opt,param)
#define curl_easy_getinfo(handle,info,arg) curl_easy_getinfo(handle,info,arg)
#define curl_share_setopt(share,opt,param) curl_share_setopt(share,opt,param)
#define curl_multi_setopt(handle,opt,param) curl_multi_setopt(handle,opt,param)
#endif /* __STDC__ >= 1 */
#endif /* gcc >= 4.3 && !__cplusplus */

#endif /* CURLINC_CURL_H */
