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

#ifdef HAVE_PWD_H
#  undef __NO_NET_API /* required for building for AmigaOS */
#  include <pwd.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#include <curl/mprintf.h>

#include "tool_findfile.h"

#include "memdebug.h" /* keep this as LAST include */

struct finder {
  const char *env;
  const char *append;
};

static const struct finder list[] = {
  { "CURL_HOME", NULL },
  { "XDG_CONFIG_HOME", NULL },
  { "HOME", NULL },
#ifdef WIN32
  { "USERPROFILE", NULL },
  { "APPDATA", NULL },
  { "USERPROFILE", "\\Application Data"},
#endif
  { NULL, NULL }
};

static char *checkhome(const char *home, const char *fname, bool dotscore)
{
  const char pref[2] = { '.', '_' };
  int i;
  for(i = 0; i < (dotscore ? 2 : 1); i++) {
    char *c;
    if(dotscore)
      c = curl_maprintf("%s" DIR_CHAR "%c%s", home, pref[i], &fname[1]);
    else
      c = curl_maprintf("%s" DIR_CHAR "%s", home, fname);
    if(c) {
      int fd = open(c, O_RDONLY);
      if(fd >= 0) {
        char *path = strdup(c);
        close(fd);
        curl_free(c);
        return path;
      }
      curl_free(c);
    }
  }
  return NULL;
}

/*
 * findfile() - return the full path name of the file.
 *
 * If 'dotscore' is TRUE, then check for the file first with a leading dot
 * and then with a leading underscore.
 *
 * 1. Iterate over the environment variables in order, and if set, check for
 *    the given file to be accessed there, then it is a match.
 * 2. Non-windows: try getpwuid
 */
char *findfile(const char *fname, bool dotscore)
{
  int i;
  DEBUGASSERT(fname && fname[0]);
  DEBUGASSERT(!dotscore || (fname[0] == '.'));

  if(!fname[0])
    return NULL;

  for(i = 0; list[i].env; i++) {
    char *home = curl_getenv(list[i].env);
    if(home) {
      char *path;
      if(!home[0]) {
        curl_free(home);
        continue;
      }
      if(list[i].append) {
        char *c = curl_maprintf("%s%s", home, list[i].append);
        curl_free(home);
        if(!c)
          return NULL;
        home = c;
      }
      path = checkhome(home, fname, dotscore);
      curl_free(home);
      if(path)
        return path;
    }
  }
#if defined(HAVE_GETPWUID) && defined(HAVE_GETEUID)
  {
    struct passwd *pw = getpwuid(geteuid());
    if(pw) {
      char *home = pw->pw_dir;
      if(home && home[0])
        return checkhome(home, fname, FALSE);
    }
  }
#endif /* PWD-stuff */
  return NULL;
}
