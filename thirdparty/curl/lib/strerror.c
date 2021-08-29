/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2004 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
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

#ifdef HAVE_STRERROR_R
#  if (!defined(HAVE_POSIX_STRERROR_R) && \
       !defined(HAVE_GLIBC_STRERROR_R) && \
       !defined(HAVE_VXWORKS_STRERROR_R)) || \
      (defined(HAVE_POSIX_STRERROR_R) && defined(HAVE_VXWORKS_STRERROR_R)) || \
      (defined(HAVE_GLIBC_STRERROR_R) && defined(HAVE_VXWORKS_STRERROR_R)) || \
      (defined(HAVE_POSIX_STRERROR_R) && defined(HAVE_GLIBC_STRERROR_R))
#    error "strerror_r MUST be either POSIX, glibc or vxworks-style"
#  endif
#endif

#include <curl/curl.h>

#ifdef USE_LIBIDN2
#include <idn2.h>
#endif

#ifdef USE_WINDOWS_SSPI
#include "curl_sspi.h"
#endif

#include "strerror.h"
/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

#if defined(WIN32) || defined(_WIN32_WCE)
#define PRESERVE_WINDOWS_ERROR_CODE
#endif

const char *
curl_easy_strerror(CURLcode error)
{
#ifndef CURL_DISABLE_VERBOSE_STRINGS
  switch(error) {
  case CURLE_OK:
    return "No error";

  case CURLE_UNSUPPORTED_PROTOCOL:
    return "Unsupported protocol";

  case CURLE_FAILED_INIT:
    return "Failed initialization";

  case CURLE_URL_MALFORMAT:
    return "URL using bad/illegal format or missing URL";

  case CURLE_NOT_BUILT_IN:
    return "A requested feature, protocol or option was not found built-in in"
      " this libcurl due to a build-time decision.";

  case CURLE_COULDNT_RESOLVE_PROXY:
    return "Couldn't resolve proxy name";

  case CURLE_COULDNT_RESOLVE_HOST:
    return "Couldn't resolve host name";

  case CURLE_COULDNT_CONNECT:
    return "Couldn't connect to server";

  case CURLE_WEIRD_SERVER_REPLY:
    return "Weird server reply";

  case CURLE_REMOTE_ACCESS_DENIED:
    return "Access denied to remote resource";

  case CURLE_FTP_ACCEPT_FAILED:
    return "FTP: The server failed to connect to data port";

  case CURLE_FTP_ACCEPT_TIMEOUT:
    return "FTP: Accepting server connect has timed out";

  case CURLE_FTP_PRET_FAILED:
    return "FTP: The server did not accept the PRET command.";

  case CURLE_FTP_WEIRD_PASS_REPLY:
    return "FTP: unknown PASS reply";

  case CURLE_FTP_WEIRD_PASV_REPLY:
    return "FTP: unknown PASV reply";

  case CURLE_FTP_WEIRD_227_FORMAT:
    return "FTP: unknown 227 response format";

  case CURLE_FTP_CANT_GET_HOST:
    return "FTP: can't figure out the host in the PASV response";

  case CURLE_HTTP2:
    return "Error in the HTTP2 framing layer";

  case CURLE_FTP_COULDNT_SET_TYPE:
    return "FTP: couldn't set file type";

  case CURLE_PARTIAL_FILE:
    return "Transferred a partial file";

  case CURLE_FTP_COULDNT_RETR_FILE:
    return "FTP: couldn't retrieve (RETR failed) the specified file";

  case CURLE_QUOTE_ERROR:
    return "Quote command returned error";

  case CURLE_HTTP_RETURNED_ERROR:
    return "HTTP response code said error";

  case CURLE_WRITE_ERROR:
    return "Failed writing received data to disk/application";

  case CURLE_UPLOAD_FAILED:
    return "Upload failed (at start/before it took off)";

  case CURLE_READ_ERROR:
    return "Failed to open/read local data from file/application";

  case CURLE_OUT_OF_MEMORY:
    return "Out of memory";

  case CURLE_OPERATION_TIMEDOUT:
    return "Timeout was reached";

  case CURLE_FTP_PORT_FAILED:
    return "FTP: command PORT failed";

  case CURLE_FTP_COULDNT_USE_REST:
    return "FTP: command REST failed";

  case CURLE_RANGE_ERROR:
    return "Requested range was not delivered by the server";

  case CURLE_HTTP_POST_ERROR:
    return "Internal problem setting up the POST";

  case CURLE_SSL_CONNECT_ERROR:
    return "SSL connect error";

  case CURLE_BAD_DOWNLOAD_RESUME:
    return "Couldn't resume download";

  case CURLE_FILE_COULDNT_READ_FILE:
    return "Couldn't read a file:// file";

  case CURLE_LDAP_CANNOT_BIND:
    return "LDAP: cannot bind";

  case CURLE_LDAP_SEARCH_FAILED:
    return "LDAP: search failed";

  case CURLE_FUNCTION_NOT_FOUND:
    return "A required function in the library was not found";

  case CURLE_ABORTED_BY_CALLBACK:
    return "Operation was aborted by an application callback";

  case CURLE_BAD_FUNCTION_ARGUMENT:
    return "A libcurl function was given a bad argument";

  case CURLE_INTERFACE_FAILED:
    return "Failed binding local connection end";

  case CURLE_TOO_MANY_REDIRECTS :
    return "Number of redirects hit maximum amount";

  case CURLE_UNKNOWN_OPTION:
    return "An unknown option was passed in to libcurl";

  case CURLE_SETOPT_OPTION_SYNTAX :
    return "Malformed option provided in a setopt";

  case CURLE_GOT_NOTHING:
    return "Server returned nothing (no headers, no data)";

  case CURLE_SSL_ENGINE_NOTFOUND:
    return "SSL crypto engine not found";

  case CURLE_SSL_ENGINE_SETFAILED:
    return "Can not set SSL crypto engine as default";

  case CURLE_SSL_ENGINE_INITFAILED:
    return "Failed to initialise SSL crypto engine";

  case CURLE_SEND_ERROR:
    return "Failed sending data to the peer";

  case CURLE_RECV_ERROR:
    return "Failure when receiving data from the peer";

  case CURLE_SSL_CERTPROBLEM:
    return "Problem with the local SSL certificate";

  case CURLE_SSL_CIPHER:
    return "Couldn't use specified SSL cipher";

  case CURLE_PEER_FAILED_VERIFICATION:
    return "SSL peer certificate or SSH remote key was not OK";

  case CURLE_SSL_CACERT_BADFILE:
    return "Problem with the SSL CA cert (path? access rights?)";

  case CURLE_BAD_CONTENT_ENCODING:
    return "Unrecognized or bad HTTP Content or Transfer-Encoding";

  case CURLE_LDAP_INVALID_URL:
    return "Invalid LDAP URL";

  case CURLE_FILESIZE_EXCEEDED:
    return "Maximum file size exceeded";

  case CURLE_USE_SSL_FAILED:
    return "Requested SSL level failed";

  case CURLE_SSL_SHUTDOWN_FAILED:
    return "Failed to shut down the SSL connection";

  case CURLE_SSL_CRL_BADFILE:
    return "Failed to load CRL file (path? access rights?, format?)";

  case CURLE_SSL_ISSUER_ERROR:
    return "Issuer check against peer certificate failed";

  case CURLE_SEND_FAIL_REWIND:
    return "Send failed since rewinding of the data stream failed";

  case CURLE_LOGIN_DENIED:
    return "Login denied";

  case CURLE_TFTP_NOTFOUND:
    return "TFTP: File Not Found";

  case CURLE_TFTP_PERM:
    return "TFTP: Access Violation";

  case CURLE_REMOTE_DISK_FULL:
    return "Disk full or allocation exceeded";

  case CURLE_TFTP_ILLEGAL:
    return "TFTP: Illegal operation";

  case CURLE_TFTP_UNKNOWNID:
    return "TFTP: Unknown transfer ID";

  case CURLE_REMOTE_FILE_EXISTS:
    return "Remote file already exists";

  case CURLE_TFTP_NOSUCHUSER:
    return "TFTP: No such user";

  case CURLE_CONV_FAILED:
    return "Conversion failed";

  case CURLE_CONV_REQD:
    return "Caller must register CURLOPT_CONV_ callback options";

  case CURLE_REMOTE_FILE_NOT_FOUND:
    return "Remote file not found";

  case CURLE_SSH:
    return "Error in the SSH layer";

  case CURLE_AGAIN:
    return "Socket not ready for send/recv";

  case CURLE_RTSP_CSEQ_ERROR:
    return "RTSP CSeq mismatch or invalid CSeq";

  case CURLE_RTSP_SESSION_ERROR:
    return "RTSP session error";

  case CURLE_FTP_BAD_FILE_LIST:
    return "Unable to parse FTP file list";

  case CURLE_CHUNK_FAILED:
    return "Chunk callback failed";

  case CURLE_NO_CONNECTION_AVAILABLE:
    return "The max connection limit is reached";

  case CURLE_SSL_PINNEDPUBKEYNOTMATCH:
    return "SSL public key does not match pinned public key";

  case CURLE_SSL_INVALIDCERTSTATUS:
    return "SSL server certificate status verification FAILED";

  case CURLE_HTTP2_STREAM:
    return "Stream error in the HTTP/2 framing layer";

  case CURLE_RECURSIVE_API_CALL:
    return "API function called from within callback";

  case CURLE_AUTH_ERROR:
    return "An authentication function returned an error";

  case CURLE_HTTP3:
    return "HTTP/3 error";

  case CURLE_QUIC_CONNECT_ERROR:
    return "QUIC connection error";

  case CURLE_PROXY:
    return "proxy handshake error";

  case CURLE_SSL_CLIENTCERT:
    return "SSL Client Certificate required";

    /* error codes not used by current libcurl */
  case CURLE_OBSOLETE20:
  case CURLE_OBSOLETE24:
  case CURLE_OBSOLETE29:
  case CURLE_OBSOLETE32:
  case CURLE_OBSOLETE40:
  case CURLE_OBSOLETE44:
  case CURLE_OBSOLETE46:
  case CURLE_OBSOLETE50:
  case CURLE_OBSOLETE51:
  case CURLE_OBSOLETE57:
  case CURL_LAST:
    break;
  }
  /*
   * By using a switch, gcc -Wall will complain about enum values
   * which do not appear, helping keep this function up-to-date.
   * By using gcc -Wall -Werror, you can't forget.
   *
   * A table would not have the same benefit.  Most compilers will
   * generate code very similar to a table in any case, so there
   * is little performance gain from a table.  And something is broken
   * for the user's application, anyways, so does it matter how fast
   * it _doesn't_ work?
   *
   * The line number for the error will be near this comment, which
   * is why it is here, and not at the start of the switch.
   */
  return "Unknown error";
#else
  if(!error)
    return "No error";
  else
    return "Error";
#endif
}

const char *
curl_multi_strerror(CURLMcode error)
{
#ifndef CURL_DISABLE_VERBOSE_STRINGS
  switch(error) {
  case CURLM_CALL_MULTI_PERFORM:
    return "Please call curl_multi_perform() soon";

  case CURLM_OK:
    return "No error";

  case CURLM_BAD_HANDLE:
    return "Invalid multi handle";

  case CURLM_BAD_EASY_HANDLE:
    return "Invalid easy handle";

  case CURLM_OUT_OF_MEMORY:
    return "Out of memory";

  case CURLM_INTERNAL_ERROR:
    return "Internal error";

  case CURLM_BAD_SOCKET:
    return "Invalid socket argument";

  case CURLM_UNKNOWN_OPTION:
    return "Unknown option";

  case CURLM_ADDED_ALREADY:
    return "The easy handle is already added to a multi handle";

  case CURLM_RECURSIVE_API_CALL:
    return "API function called from within callback";

  case CURLM_WAKEUP_FAILURE:
    return "Wakeup is unavailable or failed";

  case CURLM_BAD_FUNCTION_ARGUMENT:
    return "A libcurl function was given a bad argument";

  case CURLM_ABORTED_BY_CALLBACK:
    return "Operation was aborted by an application callback";

  case CURLM_LAST:
    break;
  }

  return "Unknown error";
#else
  if(error == CURLM_OK)
    return "No error";
  else
    return "Error";
#endif
}

const char *
curl_share_strerror(CURLSHcode error)
{
#ifndef CURL_DISABLE_VERBOSE_STRINGS
  switch(error) {
  case CURLSHE_OK:
    return "No error";

  case CURLSHE_BAD_OPTION:
    return "Unknown share option";

  case CURLSHE_IN_USE:
    return "Share currently in use";

  case CURLSHE_INVALID:
    return "Invalid share handle";

  case CURLSHE_NOMEM:
    return "Out of memory";

  case CURLSHE_NOT_BUILT_IN:
    return "Feature not enabled in this library";

  case CURLSHE_LAST:
    break;
  }

  return "CURLSHcode unknown";
#else
  if(error == CURLSHE_OK)
    return "No error";
  else
    return "Error";
#endif
}

const char *
curl_url_strerror(CURLUcode error)
{
#ifndef CURL_DISABLE_VERBOSE_STRINGS
  switch(error) {
  case CURLUE_OK:
    return "No error";

  case CURLUE_BAD_HANDLE:
    return "An invalid CURLU pointer was passed as argument";

  case CURLUE_BAD_PARTPOINTER:
    return "An invalid 'part' argument was passed as argument";

  case CURLUE_MALFORMED_INPUT:
    return "Malformed input to a URL function";

  case CURLUE_BAD_PORT_NUMBER:
    return "Port number was not a decimal number between 0 and 65535";

  case CURLUE_UNSUPPORTED_SCHEME:
    return "This libcurl build doesn't support the given URL scheme";

  case CURLUE_URLDECODE:
    return "URL decode error, most likely because of rubbish in the input";

  case CURLUE_OUT_OF_MEMORY:
    return "A memory function failed";

  case CURLUE_USER_NOT_ALLOWED:
    return "Credentials was passed in the URL when prohibited";

  case CURLUE_UNKNOWN_PART:
    return "An unknown part ID was passed to a URL API function";

  case CURLUE_NO_SCHEME:
    return "No scheme part in the URL";

  case CURLUE_NO_USER:
    return "No user part in the URL";

  case CURLUE_NO_PASSWORD:
    return "No password part in the URL";

  case CURLUE_NO_OPTIONS:
    return "No options part in the URL";

  case CURLUE_NO_HOST:
    return "No host part in the URL";

  case CURLUE_NO_PORT:
    return "No port part in the URL";

  case CURLUE_NO_QUERY:
    return "No query part in the URL";

  case CURLUE_NO_FRAGMENT:
    return "No fragment part in the URL";

  case CURLUE_NO_ZONEID:
    return "No zoneid part in the URL";

  case CURLUE_BAD_LOGIN:
    return "Bad login part";

  case CURLUE_BAD_IPV6:
    return "Bad IPv6 address";

  case CURLUE_BAD_HOSTNAME:
    return "Bad hostname";

  case CURLUE_BAD_FILE_URL:
    return "Bad file:// URL";

  case CURLUE_BAD_SLASHES:
    return "Unsupported number of slashes";

  case CURLUE_BAD_SCHEME:
    return "Bad scheme";

  case CURLUE_BAD_PATH:
    return "Bad path";

  case CURLUE_BAD_FRAGMENT:
    return "Bad fragment";

  case CURLUE_BAD_QUERY:
    return "Bad query";

  case CURLUE_BAD_PASSWORD:
    return "Bad password";

  case CURLUE_BAD_USER:
    return "Bad user";

  case CURLUE_LAST:
    break;
  }

  return "CURLUcode unknown";
#else
  if(error == CURLUE_OK)
    return "No error";
  else
    return "Error";
#endif
}

#ifdef USE_WINSOCK
/* This is a helper function for Curl_strerror that converts Winsock error
 * codes (WSAGetLastError) to error messages.
 * Returns NULL if no error message was found for error code.
 */
static const char *
get_winsock_error (int err, char *buf, size_t len)
{
#ifndef CURL_DISABLE_VERBOSE_STRINGS
  const char *p;
#endif

  if(!len)
    return NULL;

  *buf = '\0';

#ifdef CURL_DISABLE_VERBOSE_STRINGS
  (void)err;
  return NULL;
#else
  switch(err) {
  case WSAEINTR:
    p = "Call interrupted";
    break;
  case WSAEBADF:
    p = "Bad file";
    break;
  case WSAEACCES:
    p = "Bad access";
    break;
  case WSAEFAULT:
    p = "Bad argument";
    break;
  case WSAEINVAL:
    p = "Invalid arguments";
    break;
  case WSAEMFILE:
    p = "Out of file descriptors";
    break;
  case WSAEWOULDBLOCK:
    p = "Call would block";
    break;
  case WSAEINPROGRESS:
  case WSAEALREADY:
    p = "Blocking call in progress";
    break;
  case WSAENOTSOCK:
    p = "Descriptor is not a socket";
    break;
  case WSAEDESTADDRREQ:
    p = "Need destination address";
    break;
  case WSAEMSGSIZE:
    p = "Bad message size";
    break;
  case WSAEPROTOTYPE:
    p = "Bad protocol";
    break;
  case WSAENOPROTOOPT:
    p = "Protocol option is unsupported";
    break;
  case WSAEPROTONOSUPPORT:
    p = "Protocol is unsupported";
    break;
  case WSAESOCKTNOSUPPORT:
    p = "Socket is unsupported";
    break;
  case WSAEOPNOTSUPP:
    p = "Operation not supported";
    break;
  case WSAEAFNOSUPPORT:
    p = "Address family not supported";
    break;
  case WSAEPFNOSUPPORT:
    p = "Protocol family not supported";
    break;
  case WSAEADDRINUSE:
    p = "Address already in use";
    break;
  case WSAEADDRNOTAVAIL:
    p = "Address not available";
    break;
  case WSAENETDOWN:
    p = "Network down";
    break;
  case WSAENETUNREACH:
    p = "Network unreachable";
    break;
  case WSAENETRESET:
    p = "Network has been reset";
    break;
  case WSAECONNABORTED:
    p = "Connection was aborted";
    break;
  case WSAECONNRESET:
    p = "Connection was reset";
    break;
  case WSAENOBUFS:
    p = "No buffer space";
    break;
  case WSAEISCONN:
    p = "Socket is already connected";
    break;
  case WSAENOTCONN:
    p = "Socket is not connected";
    break;
  case WSAESHUTDOWN:
    p = "Socket has been shut down";
    break;
  case WSAETOOMANYREFS:
    p = "Too many references";
    break;
  case WSAETIMEDOUT:
    p = "Timed out";
    break;
  case WSAECONNREFUSED:
    p = "Connection refused";
    break;
  case WSAELOOP:
    p = "Loop??";
    break;
  case WSAENAMETOOLONG:
    p = "Name too long";
    break;
  case WSAEHOSTDOWN:
    p = "Host down";
    break;
  case WSAEHOSTUNREACH:
    p = "Host unreachable";
    break;
  case WSAENOTEMPTY:
    p = "Not empty";
    break;
  case WSAEPROCLIM:
    p = "Process limit reached";
    break;
  case WSAEUSERS:
    p = "Too many users";
    break;
  case WSAEDQUOT:
    p = "Bad quota";
    break;
  case WSAESTALE:
    p = "Something is stale";
    break;
  case WSAEREMOTE:
    p = "Remote error";
    break;
#ifdef WSAEDISCON  /* missing in SalfordC! */
  case WSAEDISCON:
    p = "Disconnected";
    break;
#endif
    /* Extended Winsock errors */
  case WSASYSNOTREADY:
    p = "Winsock library is not ready";
    break;
  case WSANOTINITIALISED:
    p = "Winsock library not initialised";
    break;
  case WSAVERNOTSUPPORTED:
    p = "Winsock version not supported";
    break;

    /* getXbyY() errors (already handled in herrmsg):
     * Authoritative Answer: Host not found */
  case WSAHOST_NOT_FOUND:
    p = "Host not found";
    break;

    /* Non-Authoritative: Host not found, or SERVERFAIL */
  case WSATRY_AGAIN:
    p = "Host not found, try again";
    break;

    /* Non recoverable errors, FORMERR, REFUSED, NOTIMP */
  case WSANO_RECOVERY:
    p = "Unrecoverable error in call to nameserver";
    break;

    /* Valid name, no data record of requested type */
  case WSANO_DATA:
    p = "No data record of requested type";
    break;

  default:
    return NULL;
  }
  strncpy(buf, p, len);
  buf [len-1] = '\0';
  return buf;
#endif
}
#endif   /* USE_WINSOCK */

#if defined(WIN32) || defined(_WIN32_WCE)
/* This is a helper function for Curl_strerror that converts Windows API error
 * codes (GetLastError) to error messages.
 * Returns NULL if no error message was found for error code.
 */
static const char *
get_winapi_error(int err, char *buf, size_t buflen)
{
  char *p;
  wchar_t wbuf[256];

  if(!buflen)
    return NULL;

  *buf = '\0';
  *wbuf = L'\0';

  /* We return the local codepage version of the error string because if it is
     output to the user's terminal it will likely be with functions which
     expect the local codepage (eg fprintf, failf, infof).
     FormatMessageW -> wcstombs is used for Windows CE compatibility. */
  if(FormatMessageW((FORMAT_MESSAGE_FROM_SYSTEM |
                     FORMAT_MESSAGE_IGNORE_INSERTS), NULL, err,
                    LANG_NEUTRAL, wbuf, sizeof(wbuf)/sizeof(wchar_t), NULL)) {
    size_t written = wcstombs(buf, wbuf, buflen - 1);
    if(written != (size_t)-1)
      buf[written] = '\0';
    else
      *buf = '\0';
  }

  /* Truncate multiple lines */
  p = strchr(buf, '\n');
  if(p) {
    if(p > buf && *(p-1) == '\r')
      *(p-1) = '\0';
    else
      *p = '\0';
  }

  return (*buf ? buf : NULL);
}
#endif /* WIN32 || _WIN32_WCE */

/*
 * Our thread-safe and smart strerror() replacement.
 *
 * The 'err' argument passed in to this function MUST be a true errno number
 * as reported on this system. We do no range checking on the number before
 * we pass it to the "number-to-message" conversion function and there might
 * be systems that don't do proper range checking in there themselves.
 *
 * We don't do range checking (on systems other than Windows) since there is
 * no good reliable and portable way to do it.
 *
 * On Windows different types of error codes overlap. This function has an
 * order of preference when trying to match error codes:
 * CRT (errno), Winsock (WSAGetLastError), Windows API (GetLastError).
 *
 * It may be more correct to call one of the variant functions instead:
 * Call Curl_sspi_strerror if the error code is definitely Windows SSPI.
 * Call Curl_winapi_strerror if the error code is definitely Windows API.
 */
const char *Curl_strerror(int err, char *buf, size_t buflen)
{
#ifdef PRESERVE_WINDOWS_ERROR_CODE
  DWORD old_win_err = GetLastError();
#endif
  int old_errno = errno;
  char *p;
  size_t max;

  if(!buflen)
    return NULL;

#ifndef WIN32
  DEBUGASSERT(err >= 0);
#endif

  max = buflen - 1;
  *buf = '\0';

#if defined(WIN32) || defined(_WIN32_WCE)
#if defined(WIN32)
  /* 'sys_nerr' is the maximum errno number, it is not widely portable */
  if(err >= 0 && err < sys_nerr)
    strncpy(buf, sys_errlist[err], max);
  else
#endif
  {
    if(
#ifdef USE_WINSOCK
       !get_winsock_error(err, buf, max) &&
#endif
       !get_winapi_error((DWORD)err, buf, max))
      msnprintf(buf, max, "Unknown error %d (%#x)", err, err);
  }
#else /* not Windows coming up */

#if defined(HAVE_STRERROR_R) && defined(HAVE_POSIX_STRERROR_R)
 /*
  * The POSIX-style strerror_r() may set errno to ERANGE if insufficient
  * storage is supplied via 'strerrbuf' and 'buflen' to hold the generated
  * message string, or EINVAL if 'errnum' is not a valid error number.
  */
  if(0 != strerror_r(err, buf, max)) {
    if('\0' == buf[0])
      msnprintf(buf, max, "Unknown error %d", err);
  }
#elif defined(HAVE_STRERROR_R) && defined(HAVE_GLIBC_STRERROR_R)
 /*
  * The glibc-style strerror_r() only *might* use the buffer we pass to
  * the function, but it always returns the error message as a pointer,
  * so we must copy that string unconditionally (if non-NULL).
  */
  {
    char buffer[256];
    char *msg = strerror_r(err, buffer, sizeof(buffer));
    if(msg)
      strncpy(buf, msg, max);
    else
      msnprintf(buf, max, "Unknown error %d", err);
  }
#elif defined(HAVE_STRERROR_R) && defined(HAVE_VXWORKS_STRERROR_R)
 /*
  * The vxworks-style strerror_r() does use the buffer we pass to the function.
  * The buffer size should be at least NAME_MAX (256)
  */
  {
    char buffer[256];
    if(OK == strerror_r(err, buffer))
      strncpy(buf, buffer, max);
    else
      msnprintf(buf, max, "Unknown error %d", err);
  }
#else
  {
    /* !checksrc! disable STRERROR 1 */
    const char *msg = strerror(err);
    if(msg)
      strncpy(buf, msg, max);
    else
      msnprintf(buf, max, "Unknown error %d", err);
  }
#endif

#endif /* end of not Windows */

  buf[max] = '\0'; /* make sure the string is null-terminated */

  /* strip trailing '\r\n' or '\n'. */
  p = strrchr(buf, '\n');
  if(p && (p - buf) >= 2)
    *p = '\0';
  p = strrchr(buf, '\r');
  if(p && (p - buf) >= 1)
    *p = '\0';

  if(errno != old_errno)
    errno = old_errno;

#ifdef PRESERVE_WINDOWS_ERROR_CODE
  if(old_win_err != GetLastError())
    SetLastError(old_win_err);
#endif

  return buf;
}

/*
 * Curl_winapi_strerror:
 * Variant of Curl_strerror if the error code is definitely Windows API.
 */
#if defined(WIN32) || defined(_WIN32_WCE)
const char *Curl_winapi_strerror(DWORD err, char *buf, size_t buflen)
{
#ifdef PRESERVE_WINDOWS_ERROR_CODE
  DWORD old_win_err = GetLastError();
#endif
  int old_errno = errno;

  if(!buflen)
    return NULL;

  *buf = '\0';

#ifndef CURL_DISABLE_VERBOSE_STRINGS
  if(!get_winapi_error(err, buf, buflen)) {
    msnprintf(buf, buflen, "Unknown error %u (0x%08X)", err, err);
  }
#else
  {
    const char *txt = (err == ERROR_SUCCESS) ? "No error" : "Error";
    strncpy(buf, txt, buflen);
    buf[buflen - 1] = '\0';
  }
#endif

  if(errno != old_errno)
    errno = old_errno;

#ifdef PRESERVE_WINDOWS_ERROR_CODE
  if(old_win_err != GetLastError())
    SetLastError(old_win_err);
#endif

  return buf;
}
#endif /* WIN32 || _WIN32_WCE */

#ifdef USE_WINDOWS_SSPI
/*
 * Curl_sspi_strerror:
 * Variant of Curl_strerror if the error code is definitely Windows SSPI.
 */
const char *Curl_sspi_strerror(int err, char *buf, size_t buflen)
{
#ifdef PRESERVE_WINDOWS_ERROR_CODE
  DWORD old_win_err = GetLastError();
#endif
  int old_errno = errno;
  const char *txt;

  if(!buflen)
    return NULL;

  *buf = '\0';

#ifndef CURL_DISABLE_VERBOSE_STRINGS

  switch(err) {
    case SEC_E_OK:
      txt = "No error";
      break;
#define SEC2TXT(sec) case sec: txt = #sec; break
    SEC2TXT(CRYPT_E_REVOKED);
    SEC2TXT(SEC_E_ALGORITHM_MISMATCH);
    SEC2TXT(SEC_E_BAD_BINDINGS);
    SEC2TXT(SEC_E_BAD_PKGID);
    SEC2TXT(SEC_E_BUFFER_TOO_SMALL);
    SEC2TXT(SEC_E_CANNOT_INSTALL);
    SEC2TXT(SEC_E_CANNOT_PACK);
    SEC2TXT(SEC_E_CERT_EXPIRED);
    SEC2TXT(SEC_E_CERT_UNKNOWN);
    SEC2TXT(SEC_E_CERT_WRONG_USAGE);
    SEC2TXT(SEC_E_CONTEXT_EXPIRED);
    SEC2TXT(SEC_E_CROSSREALM_DELEGATION_FAILURE);
    SEC2TXT(SEC_E_CRYPTO_SYSTEM_INVALID);
    SEC2TXT(SEC_E_DECRYPT_FAILURE);
    SEC2TXT(SEC_E_DELEGATION_POLICY);
    SEC2TXT(SEC_E_DELEGATION_REQUIRED);
    SEC2TXT(SEC_E_DOWNGRADE_DETECTED);
    SEC2TXT(SEC_E_ENCRYPT_FAILURE);
    SEC2TXT(SEC_E_ILLEGAL_MESSAGE);
    SEC2TXT(SEC_E_INCOMPLETE_CREDENTIALS);
    SEC2TXT(SEC_E_INCOMPLETE_MESSAGE);
    SEC2TXT(SEC_E_INSUFFICIENT_MEMORY);
    SEC2TXT(SEC_E_INTERNAL_ERROR);
    SEC2TXT(SEC_E_INVALID_HANDLE);
    SEC2TXT(SEC_E_INVALID_PARAMETER);
    SEC2TXT(SEC_E_INVALID_TOKEN);
    SEC2TXT(SEC_E_ISSUING_CA_UNTRUSTED);
    SEC2TXT(SEC_E_ISSUING_CA_UNTRUSTED_KDC);
    SEC2TXT(SEC_E_KDC_CERT_EXPIRED);
    SEC2TXT(SEC_E_KDC_CERT_REVOKED);
    SEC2TXT(SEC_E_KDC_INVALID_REQUEST);
    SEC2TXT(SEC_E_KDC_UNABLE_TO_REFER);
    SEC2TXT(SEC_E_KDC_UNKNOWN_ETYPE);
    SEC2TXT(SEC_E_LOGON_DENIED);
    SEC2TXT(SEC_E_MAX_REFERRALS_EXCEEDED);
    SEC2TXT(SEC_E_MESSAGE_ALTERED);
    SEC2TXT(SEC_E_MULTIPLE_ACCOUNTS);
    SEC2TXT(SEC_E_MUST_BE_KDC);
    SEC2TXT(SEC_E_NOT_OWNER);
    SEC2TXT(SEC_E_NO_AUTHENTICATING_AUTHORITY);
    SEC2TXT(SEC_E_NO_CREDENTIALS);
    SEC2TXT(SEC_E_NO_IMPERSONATION);
    SEC2TXT(SEC_E_NO_IP_ADDRESSES);
    SEC2TXT(SEC_E_NO_KERB_KEY);
    SEC2TXT(SEC_E_NO_PA_DATA);
    SEC2TXT(SEC_E_NO_S4U_PROT_SUPPORT);
    SEC2TXT(SEC_E_NO_TGT_REPLY);
    SEC2TXT(SEC_E_OUT_OF_SEQUENCE);
    SEC2TXT(SEC_E_PKINIT_CLIENT_FAILURE);
    SEC2TXT(SEC_E_PKINIT_NAME_MISMATCH);
    SEC2TXT(SEC_E_POLICY_NLTM_ONLY);
    SEC2TXT(SEC_E_QOP_NOT_SUPPORTED);
    SEC2TXT(SEC_E_REVOCATION_OFFLINE_C);
    SEC2TXT(SEC_E_REVOCATION_OFFLINE_KDC);
    SEC2TXT(SEC_E_SECPKG_NOT_FOUND);
    SEC2TXT(SEC_E_SECURITY_QOS_FAILED);
    SEC2TXT(SEC_E_SHUTDOWN_IN_PROGRESS);
    SEC2TXT(SEC_E_SMARTCARD_CERT_EXPIRED);
    SEC2TXT(SEC_E_SMARTCARD_CERT_REVOKED);
    SEC2TXT(SEC_E_SMARTCARD_LOGON_REQUIRED);
    SEC2TXT(SEC_E_STRONG_CRYPTO_NOT_SUPPORTED);
    SEC2TXT(SEC_E_TARGET_UNKNOWN);
    SEC2TXT(SEC_E_TIME_SKEW);
    SEC2TXT(SEC_E_TOO_MANY_PRINCIPALS);
    SEC2TXT(SEC_E_UNFINISHED_CONTEXT_DELETED);
    SEC2TXT(SEC_E_UNKNOWN_CREDENTIALS);
    SEC2TXT(SEC_E_UNSUPPORTED_FUNCTION);
    SEC2TXT(SEC_E_UNSUPPORTED_PREAUTH);
    SEC2TXT(SEC_E_UNTRUSTED_ROOT);
    SEC2TXT(SEC_E_WRONG_CREDENTIAL_HANDLE);
    SEC2TXT(SEC_E_WRONG_PRINCIPAL);
    SEC2TXT(SEC_I_COMPLETE_AND_CONTINUE);
    SEC2TXT(SEC_I_COMPLETE_NEEDED);
    SEC2TXT(SEC_I_CONTEXT_EXPIRED);
    SEC2TXT(SEC_I_CONTINUE_NEEDED);
    SEC2TXT(SEC_I_INCOMPLETE_CREDENTIALS);
    SEC2TXT(SEC_I_LOCAL_LOGON);
    SEC2TXT(SEC_I_NO_LSA_CONTEXT);
    SEC2TXT(SEC_I_RENEGOTIATE);
    SEC2TXT(SEC_I_SIGNATURE_NEEDED);
    default:
      txt = "Unknown error";
  }

  if(err == SEC_E_ILLEGAL_MESSAGE) {
    msnprintf(buf, buflen,
              "SEC_E_ILLEGAL_MESSAGE (0x%08X) - This error usually occurs "
              "when a fatal SSL/TLS alert is received (e.g. handshake failed)."
              " More detail may be available in the Windows System event log.",
              err);
  }
  else {
    char txtbuf[80];
    char msgbuf[256];

    msnprintf(txtbuf, sizeof(txtbuf), "%s (0x%08X)", txt, err);

    if(get_winapi_error(err, msgbuf, sizeof(msgbuf)))
      msnprintf(buf, buflen, "%s - %s", txtbuf, msgbuf);
    else {
      strncpy(buf, txtbuf, buflen);
      buf[buflen - 1] = '\0';
    }
  }

#else
  if(err == SEC_E_OK)
    txt = "No error";
  else
    txt = "Error";
  strncpy(buf, txt, buflen);
  buf[buflen - 1] = '\0';
#endif

  if(errno != old_errno)
    errno = old_errno;

#ifdef PRESERVE_WINDOWS_ERROR_CODE
  if(old_win_err != GetLastError())
    SetLastError(old_win_err);
#endif

  return buf;
}
#endif /* USE_WINDOWS_SSPI */
