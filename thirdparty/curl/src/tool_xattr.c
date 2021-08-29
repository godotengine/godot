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

#ifdef HAVE_FSETXATTR
#  include <sys/xattr.h> /* header from libc, not from libattr */
#  define USE_XATTR
#elif (defined(__FreeBSD_version) && (__FreeBSD_version > 500000)) || \
      defined(__MidnightBSD_version)
#  include <sys/types.h>
#  include <sys/extattr.h>
#  define USE_XATTR
#endif

#include "tool_xattr.h"

#include "memdebug.h" /* keep this as LAST include */

#ifdef USE_XATTR

/* mapping table of curl metadata to extended attribute names */
static const struct xattr_mapping {
  const char *attr; /* name of the xattr */
  CURLINFO info;
} mappings[] = {
  /* mappings proposed by
   * https://freedesktop.org/wiki/CommonExtendedAttributes/
   */
  { "user.xdg.referrer.url", CURLINFO_REFERER },
  { "user.xdg.origin.url",   CURLINFO_EFFECTIVE_URL },
  { "user.mime_type",        CURLINFO_CONTENT_TYPE },
  { NULL,                    CURLINFO_NONE } /* last element, abort here */
};

/* returns TRUE if a new URL is returned, that then needs to be freed */
/* @unittest: 1621 */
#ifdef UNITTESTS
bool stripcredentials(char **url);
#else
static
#endif
bool stripcredentials(char **url)
{
  CURLU *u;
  CURLUcode uc;
  char *nurl;
  u = curl_url();
  if(u) {
    uc = curl_url_set(u, CURLUPART_URL, *url, 0);
    if(uc)
      goto error;

    uc = curl_url_set(u, CURLUPART_USER, NULL, 0);
    if(uc)
      goto error;

    uc = curl_url_set(u, CURLUPART_PASSWORD, NULL, 0);
    if(uc)
      goto error;

    uc = curl_url_get(u, CURLUPART_URL, &nurl, 0);
    if(uc)
      goto error;

    curl_url_cleanup(u);

    *url = nurl;
    return TRUE;
  }
  error:
  curl_url_cleanup(u);
  return FALSE;
}

/* store metadata from the curl request alongside the downloaded
 * file using extended attributes
 */
int fwrite_xattr(CURL *curl, int fd)
{
  int i = 0;
  int err = 0;

  /* loop through all xattr-curlinfo pairs and abort on a set error */
  while(err == 0 && mappings[i].attr != NULL) {
    char *value = NULL;
    CURLcode result = curl_easy_getinfo(curl, mappings[i].info, &value);
    if(!result && value) {
      bool freeptr = FALSE;
      if(CURLINFO_EFFECTIVE_URL == mappings[i].info)
        freeptr = stripcredentials(&value);
      if(value) {
#ifdef HAVE_FSETXATTR_6
        err = fsetxattr(fd, mappings[i].attr, value, strlen(value), 0, 0);
#elif defined(HAVE_FSETXATTR_5)
        err = fsetxattr(fd, mappings[i].attr, value, strlen(value), 0);
#elif defined(__FreeBSD_version) || defined(__MidnightBSD_version)
        {
          ssize_t rc = extattr_set_fd(fd, EXTATTR_NAMESPACE_USER,
                                      mappings[i].attr, value, strlen(value));
          /* FreeBSD's extattr_set_fd returns the length of the extended
             attribute */
          err = (rc < 0 ? -1 : 0);
        }
#endif
        if(freeptr)
          curl_free(value);
      }
    }
    i++;
  }

  return err;
}
#else
int fwrite_xattr(CURL *curl, int fd)
{
  (void)curl;
  (void)fd;

  return 0;
}
#endif
