/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#ifdef SDL_FILESYSTEM_UNIX

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
// System dependent filesystem routines

#include "../SDL_sysfilesystem.h"

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(SDL_PLATFORM_FREEBSD) || defined(SDL_PLATFORM_OPENBSD)
#include <sys/sysctl.h>
#endif

static char *readSymLink(const char *path)
{
    char *result = NULL;
    ssize_t len = 64;
    ssize_t rc = -1;

    while (1) {
        char *ptr = (char *)SDL_realloc(result, (size_t)len);
        if (!ptr) {
            break;
        }

        result = ptr;

        rc = readlink(path, result, len);
        if (rc == -1) {
            break; // not a symlink, i/o error, etc.
        } else if (rc < len) {
            result[rc] = '\0'; // readlink doesn't null-terminate.
            return result;     // we're good to go.
        }

        len *= 2; // grow buffer, try again.
    }

    SDL_free(result);
    return NULL;
}

#ifdef SDL_PLATFORM_OPENBSD
static char *search_path_for_binary(const char *bin)
{
    const char *envr_real = SDL_getenv("PATH");
    char *envr;
    size_t alloc_size;
    char *exe = NULL;
    char *start = envr;
    char *ptr;

    if (!envr_real) {
        SDL_SetError("No $PATH set");
        return NULL;
    }

    envr = SDL_strdup(envr_real);
    if (!envr) {
        return NULL;
    }

    SDL_assert(bin != NULL);

    alloc_size = SDL_strlen(bin) + SDL_strlen(envr) + 2;
    exe = (char *)SDL_malloc(alloc_size);

    do {
        ptr = SDL_strchr(start, ':'); // find next $PATH separator.
        if (ptr != start) {
            if (ptr) {
                *ptr = '\0';
            }

            // build full binary path...
            SDL_snprintf(exe, alloc_size, "%s%s%s", start, (ptr && (ptr[-1] == '/')) ? "" : "/", bin);

            if (access(exe, X_OK) == 0) { // Exists as executable? We're done.
                SDL_free(envr);
                return exe;
            }
        }
        start = ptr + 1; // start points to beginning of next element.
    } while (ptr);

    SDL_free(envr);
    SDL_free(exe);

    SDL_SetError("Process not found in $PATH");
    return NULL; // doesn't exist in path.
}
#endif

char *SDL_SYS_GetBasePath(void)
{
    char *result = NULL;

#ifdef SDL_PLATFORM_FREEBSD
    char fullpath[PATH_MAX];
    size_t buflen = sizeof(fullpath);
    const int mib[] = { CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1 };
    if (sysctl(mib, SDL_arraysize(mib), fullpath, &buflen, NULL, 0) != -1) {
        result = SDL_strdup(fullpath);
        if (!result) {
            return NULL;
        }
    }
#endif
#ifdef SDL_PLATFORM_OPENBSD
    // Please note that this will fail if the process was launched with a relative path and $PWD + the cwd have changed, or argv is altered. So don't do that. Or add a new sysctl to OpenBSD.
    char **cmdline;
    size_t len;
    const int mib[] = { CTL_KERN, KERN_PROC_ARGS, getpid(), KERN_PROC_ARGV };
    if (sysctl(mib, 4, NULL, &len, NULL, 0) != -1) {
        char *exe, *pwddst;
        char *realpathbuf = (char *)SDL_malloc(PATH_MAX + 1);
        if (!realpathbuf) {
            return NULL;
        }

        cmdline = SDL_malloc(len);
        if (!cmdline) {
            SDL_free(realpathbuf);
            return NULL;
        }

        sysctl(mib, 4, cmdline, &len, NULL, 0);

        exe = cmdline[0];
        pwddst = NULL;
        if (SDL_strchr(exe, '/') == NULL) { // not a relative or absolute path, check $PATH for it
            exe = search_path_for_binary(cmdline[0]);
        } else {
            if (exe && *exe == '.') {
                const char *pwd = SDL_getenv("PWD");
                if (pwd && *pwd) {
                    SDL_asprintf(&pwddst, "%s/%s", pwd, exe);
                }
            }
        }

        if (exe) {
            if (!pwddst) {
                if (realpath(exe, realpathbuf) != NULL) {
                    result = realpathbuf;
                }
            } else {
                if (realpath(pwddst, realpathbuf) != NULL) {
                    result = realpathbuf;
                }
                SDL_free(pwddst);
            }

            if (exe != cmdline[0]) {
                SDL_free(exe);
            }
        }

        if (!result) {
            SDL_free(realpathbuf);
        }

        SDL_free(cmdline);
    }
#endif

    // is a Linux-style /proc filesystem available?
    if (!result && (access("/proc", F_OK) == 0)) {
        /* !!! FIXME: after 2.0.6 ships, let's delete this code and just
                      use the /proc/%llu version. There's no reason to have
                      two copies of this plus all the #ifdefs. --ryan. */
#ifdef SDL_PLATFORM_FREEBSD
        result = readSymLink("/proc/curproc/file");
#elif defined(SDL_PLATFORM_NETBSD)
        result = readSymLink("/proc/curproc/exe");
#elif defined(SDL_PLATFORM_SOLARIS)
        result = readSymLink("/proc/self/path/a.out");
#else
        result = readSymLink("/proc/self/exe"); // linux.
        if (!result) {
            // older kernels don't have /proc/self ... try PID version...
            char path[64];
            const int rc = SDL_snprintf(path, sizeof(path),
                                        "/proc/%llu/exe",
                                        (unsigned long long)getpid());
            if ((rc > 0) && (rc < sizeof(path))) {
                result = readSymLink(path);
            }
        }
#endif
    }

#ifdef SDL_PLATFORM_SOLARIS  // try this as a fallback if /proc didn't pan out
    if (!result) {
        const char *path = getexecname();
        if ((path) && (path[0] == '/')) { // must be absolute path...
            result = SDL_strdup(path);
            if (!result) {
                return NULL;
            }
        }
    }
#endif
    /* If we had access to argv[0] here, we could check it for a path,
        or troll through $PATH looking for it, too. */

    if (result) { // chop off filename.
        char *ptr = SDL_strrchr(result, '/');
        if (ptr) {
            *(ptr + 1) = '\0';
        } else { // shouldn't happen, but just in case...
            SDL_free(result);
            result = NULL;
        }
    }

    if (result) {
        // try to shrink buffer...
        char *ptr = (char *)SDL_realloc(result, SDL_strlen(result) + 1);
        if (ptr) {
            result = ptr; // oh well if it failed.
        }
    }

    return result;
}

char *SDL_SYS_GetPrefPath(const char *org, const char *app)
{
    /*
     * We use XDG's base directory spec, even if you're not on Linux.
     *  This isn't strictly correct, but the results are relatively sane
     *  in any case.
     *
     * http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
     */
    const char *envr = SDL_getenv("XDG_DATA_HOME");
    const char *append;
    char *result = NULL;
    char *ptr = NULL;
    size_t len = 0;

    if (!app) {
        SDL_InvalidParamError("app");
        return NULL;
    }
    if (!org) {
        org = "";
    }

    if (!envr) {
        // You end up with "$HOME/.local/share/Game Name 2"
        envr = SDL_getenv("HOME");
        if (!envr) {
            // we could take heroic measures with /etc/passwd, but oh well.
            SDL_SetError("neither XDG_DATA_HOME nor HOME environment is set");
            return NULL;
        }
        append = "/.local/share/";
    } else {
        append = "/";
    }

    len = SDL_strlen(envr);
    if (envr[len - 1] == '/') {
        append += 1;
    }

    len += SDL_strlen(append) + SDL_strlen(org) + SDL_strlen(app) + 3;
    result = (char *)SDL_malloc(len);
    if (!result) {
        return NULL;
    }

    if (*org) {
        (void)SDL_snprintf(result, len, "%s%s%s/%s/", envr, append, org, app);
    } else {
        (void)SDL_snprintf(result, len, "%s%s%s/", envr, append, app);
    }

    for (ptr = result + 1; *ptr; ptr++) {
        if (*ptr == '/') {
            *ptr = '\0';
            if (mkdir(result, 0700) != 0 && errno != EEXIST) {
                goto error;
            }
            *ptr = '/';
        }
    }
    if (mkdir(result, 0700) != 0 && errno != EEXIST) {
    error:
        SDL_SetError("Couldn't create directory '%s': '%s'", result, strerror(errno));
        SDL_free(result);
        return NULL;
    }

    return result;
}

/*
  The two functions below (prefixed with `xdg_`) have been copied from:
  https://gitlab.freedesktop.org/xdg/xdg-user-dirs/-/blob/master/xdg-user-dir-lookup.c
  and have been adapted to work with SDL. They are licensed under the following
  terms:

  Copyright (c) 2007 Red Hat, Inc.

  Permission is hereby granted, free of charge, to any person
  obtaining a copy of this software and associated documentation files
  (the "Software"), to deal in the Software without restriction,
  including without limitation the rights to use, copy, modify, merge,
  publish, distribute, sublicense, and/or sell copies of the Software,
  and to permit persons to whom the Software is furnished to do so,
  subject to the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
  ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/
static char *xdg_user_dir_lookup_with_fallback (const char *type, const char *fallback)
{
  FILE *file;
  const char *home_dir, *config_home;
  char *config_file;
  char buffer[512];
  char *user_dir;
  char *p, *d;
  int len;
  int relative;
  size_t l;

  home_dir = SDL_getenv("HOME");

  if (!home_dir)
    goto error;

  config_home = SDL_getenv("XDG_CONFIG_HOME");
  if (!config_home || config_home[0] == 0)
    {
      l = SDL_strlen (home_dir) + SDL_strlen ("/.config/user-dirs.dirs") + 1;
      config_file = (char*) SDL_malloc (l);
      if (!config_file)
        goto error;

      SDL_strlcpy (config_file, home_dir, l);
      SDL_strlcat (config_file, "/.config/user-dirs.dirs", l);
    }
  else
    {
      l = SDL_strlen (config_home) + SDL_strlen ("/user-dirs.dirs") + 1;
      config_file = (char*) SDL_malloc (l);
      if (!config_file)
        goto error;

      SDL_strlcpy (config_file, config_home, l);
      SDL_strlcat (config_file, "/user-dirs.dirs", l);
    }

  file = fopen (config_file, "r");
  SDL_free (config_file);
  if (!file)
    goto error;

  user_dir = NULL;
  while (fgets (buffer, sizeof (buffer), file))
    {
      // Remove newline at end
      len = SDL_strlen (buffer);
      if (len > 0 && buffer[len-1] == '\n')
        buffer[len-1] = 0;

      p = buffer;
      while (*p == ' ' || *p == '\t')
        p++;

      if (SDL_strncmp (p, "XDG_", 4) != 0)
        continue;
      p += 4;
      if (SDL_strncmp (p, type, SDL_strlen (type)) != 0)
        continue;
      p += SDL_strlen (type);
      if (SDL_strncmp (p, "_DIR", 4) != 0)
        continue;
      p += 4;

      while (*p == ' ' || *p == '\t')
        p++;

      if (*p != '=')
        continue;
      p++;

      while (*p == ' ' || *p == '\t')
        p++;

      if (*p != '"')
        continue;
      p++;

      relative = 0;
      if (SDL_strncmp (p, "$HOME/", 6) == 0)
        {
          p += 6;
          relative = 1;
        }
      else if (*p != '/')
        continue;

      SDL_free (user_dir);
      if (relative)
        {
          l = SDL_strlen (home_dir) + 1 + SDL_strlen (p) + 1;
          user_dir = (char*) SDL_malloc (l);
          if (!user_dir)
            goto error2;

          SDL_strlcpy (user_dir, home_dir, l);
          SDL_strlcat (user_dir, "/", l);
        }
      else
        {
          user_dir = (char*) SDL_malloc (SDL_strlen (p) + 1);
          if (!user_dir)
            goto error2;

          *user_dir = 0;
        }

      d = user_dir + SDL_strlen (user_dir);
      while (*p && *p != '"')
        {
          if ((*p == '\\') && (*(p+1) != 0))
            p++;
          *d++ = *p++;
        }
      *d = 0;
    }
error2:
  fclose (file);

  if (user_dir)
    return user_dir;

 error:
  if (fallback)
    return SDL_strdup (fallback);
  return NULL;
}

static char *xdg_user_dir_lookup (const char *type)
{
    const char *home_dir;
    char *dir, *user_dir;

    dir = xdg_user_dir_lookup_with_fallback(type, NULL);
    if (dir)
        return dir;

    home_dir = SDL_getenv("HOME");

    if (!home_dir)
        return NULL;

    // Special case desktop for historical compatibility
    if (SDL_strcmp(type, "DESKTOP") == 0) {
        size_t length = SDL_strlen(home_dir) + SDL_strlen("/Desktop") + 1;
        user_dir = (char*) SDL_malloc(length);
        if (!user_dir)
            return NULL;

        SDL_strlcpy(user_dir, home_dir, length);
        SDL_strlcat(user_dir, "/Desktop", length);
        return user_dir;
    }

    return NULL;
}

char *SDL_SYS_GetUserFolder(SDL_Folder folder)
{
    const char *param = NULL;
    char *result;
    char *newresult;

    /* According to `man xdg-user-dir`, the possible values are:
        DESKTOP
        DOWNLOAD
        TEMPLATES
        PUBLICSHARE
        DOCUMENTS
        MUSIC
        PICTURES
        VIDEOS
    */
    switch(folder) {
    case SDL_FOLDER_HOME:
        param = SDL_getenv("HOME");

        if (!param) {
            SDL_SetError("No $HOME environment variable available");
            return NULL;
        }

        result = SDL_strdup(param);
        goto append_slash;

    case SDL_FOLDER_DESKTOP:
        param = "DESKTOP";
        break;

    case SDL_FOLDER_DOCUMENTS:
        param = "DOCUMENTS";
        break;

    case SDL_FOLDER_DOWNLOADS:
        param = "DOWNLOAD";
        break;

    case SDL_FOLDER_MUSIC:
        param = "MUSIC";
        break;

    case SDL_FOLDER_PICTURES:
        param = "PICTURES";
        break;

    case SDL_FOLDER_PUBLICSHARE:
        param = "PUBLICSHARE";
        break;

    case SDL_FOLDER_SAVEDGAMES:
        SDL_SetError("Saved Games folder unavailable on XDG");
        return NULL;

    case SDL_FOLDER_SCREENSHOTS:
        SDL_SetError("Screenshots folder unavailable on XDG");
        return NULL;

    case SDL_FOLDER_TEMPLATES:
        param = "TEMPLATES";
        break;

    case SDL_FOLDER_VIDEOS:
        param = "VIDEOS";
        break;

    default:
        SDL_SetError("Invalid SDL_Folder: %d", (int) folder);
        return NULL;
    }

    /* param *should* to be set to something at this point, but just in case */
    if (!param) {
        SDL_SetError("No corresponding XDG user directory");
        return NULL;
    }

    result = xdg_user_dir_lookup(param);

    if (!result) {
        SDL_SetError("XDG directory not available");
        return NULL;
    }

append_slash:
    newresult = (char *) SDL_realloc(result, SDL_strlen(result) + 2);

    if (!newresult) {
        SDL_free(result);
        return NULL;
    }

    result = newresult;
    SDL_strlcat(result, "/", SDL_strlen(result) + 2);

    return result;
}

#endif // SDL_FILESYSTEM_UNIX
