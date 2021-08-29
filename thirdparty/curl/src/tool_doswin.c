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

#if defined(MSDOS) || defined(WIN32)

#if defined(HAVE_LIBGEN_H) && defined(HAVE_BASENAME)
#  include <libgen.h>
#endif

#ifdef WIN32
#  include <stdlib.h>
#  include <tlhelp32.h>
#  include "tool_cfgable.h"
#  include "tool_libinfo.h"
#endif

#include "tool_bname.h"
#include "tool_doswin.h"

#include "curlx.h"
#include "memdebug.h" /* keep this as LAST include */

#ifdef WIN32
#  undef  PATH_MAX
#  define PATH_MAX MAX_PATH
#endif

#ifndef S_ISCHR
#  ifdef S_IFCHR
#    define S_ISCHR(m) (((m) & S_IFMT) == S_IFCHR)
#  else
#    define S_ISCHR(m) (0) /* cannot tell if file is a device */
#  endif
#endif

#ifdef WIN32
#  define _use_lfn(f) (1)   /* long file names always available */
#elif !defined(__DJGPP__) || (__DJGPP__ < 2)  /* DJGPP 2.0 has _use_lfn() */
#  define _use_lfn(f) (0)  /* long file names never available */
#elif defined(__DJGPP__)
#  include <fcntl.h>                /* _use_lfn(f) prototype */
#endif

#ifndef UNITTESTS
static SANITIZEcode truncate_dryrun(const char *path,
                                    const size_t truncate_pos);
#ifdef MSDOS
static SANITIZEcode msdosify(char **const sanitized, const char *file_name,
                             int flags);
#endif
static SANITIZEcode rename_if_reserved_dos_device_name(char **const sanitized,
                                                       const char *file_name,
                                                       int flags);
#endif /* !UNITTESTS (static declarations used if no unit tests) */


/*
Sanitize a file or path name.

All banned characters are replaced by underscores, for example:
f?*foo => f__foo
f:foo::$DATA => f_foo__$DATA
f:\foo:bar => f__foo_bar
f:\foo:bar => f:\foo:bar   (flag SANITIZE_ALLOW_PATH)

This function was implemented according to the guidelines in 'Naming Files,
Paths, and Namespaces' section 'Naming Conventions'.
https://msdn.microsoft.com/en-us/library/windows/desktop/aa365247.aspx

Flags
-----
SANITIZE_ALLOW_COLONS:     Allow colons.
Without this flag colons are sanitized.

SANITIZE_ALLOW_PATH:       Allow path separators and colons.
Without this flag path separators and colons are sanitized.

SANITIZE_ALLOW_RESERVED:   Allow reserved device names.
Without this flag a reserved device name is renamed (COM1 => _COM1) unless it's
in a UNC prefixed path.

SANITIZE_ALLOW_TRUNCATE:   Allow truncating a long filename.
Without this flag if the sanitized filename or path will be too long an error
occurs. With this flag the filename --and not any other parts of the path-- may
be truncated to at least a single character. A filename followed by an
alternate data stream (ADS) cannot be truncated in any case.

Success: (SANITIZE_ERR_OK) *sanitized points to a sanitized copy of file_name.
Failure: (!= SANITIZE_ERR_OK) *sanitized is NULL.
*/
SANITIZEcode sanitize_file_name(char **const sanitized, const char *file_name,
                                int flags)
{
  char *p, *target;
  size_t len;
  SANITIZEcode sc;
  size_t max_sanitized_len;

  if(!sanitized)
    return SANITIZE_ERR_BAD_ARGUMENT;

  *sanitized = NULL;

  if(!file_name)
    return SANITIZE_ERR_BAD_ARGUMENT;

  if((flags & SANITIZE_ALLOW_PATH)) {
#ifndef MSDOS
    if(file_name[0] == '\\' && file_name[1] == '\\')
      /* UNC prefixed path \\ (eg \\?\C:\foo) */
      max_sanitized_len = 32767-1;
    else
#endif
      max_sanitized_len = PATH_MAX-1;
  }
  else
    /* The maximum length of a filename.
       FILENAME_MAX is often the same as PATH_MAX, in other words it is 260 and
       does not discount the path information therefore we shouldn't use it. */
    max_sanitized_len = (PATH_MAX-1 > 255) ? 255 : PATH_MAX-1;

  len = strlen(file_name);
  if(len > max_sanitized_len) {
    if(!(flags & SANITIZE_ALLOW_TRUNCATE) ||
       truncate_dryrun(file_name, max_sanitized_len))
      return SANITIZE_ERR_INVALID_PATH;

    len = max_sanitized_len;
  }

  target = malloc(len + 1);
  if(!target)
    return SANITIZE_ERR_OUT_OF_MEMORY;

  strncpy(target, file_name, len);
  target[len] = '\0';

#ifndef MSDOS
  if((flags & SANITIZE_ALLOW_PATH) && !strncmp(target, "\\\\?\\", 4))
    /* Skip the literal path prefix \\?\ */
    p = target + 4;
  else
#endif
    p = target;

  /* replace control characters and other banned characters */
  for(; *p; ++p) {
    const char *banned;

    if((1 <= *p && *p <= 31) ||
       (!(flags & (SANITIZE_ALLOW_COLONS|SANITIZE_ALLOW_PATH)) && *p == ':') ||
       (!(flags & SANITIZE_ALLOW_PATH) && (*p == '/' || *p == '\\'))) {
      *p = '_';
      continue;
    }

    for(banned = "|<>\"?*"; *banned; ++banned) {
      if(*p == *banned) {
        *p = '_';
        break;
      }
    }
  }

  /* remove trailing spaces and periods if not allowing paths */
  if(!(flags & SANITIZE_ALLOW_PATH) && len) {
    char *clip = NULL;

    p = &target[len];
    do {
      --p;
      if(*p != ' ' && *p != '.')
        break;
      clip = p;
    } while(p != target);

    if(clip) {
      *clip = '\0';
      len = clip - target;
    }
  }

#ifdef MSDOS
  sc = msdosify(&p, target, flags);
  free(target);
  if(sc)
    return sc;
  target = p;
  len = strlen(target);

  if(len > max_sanitized_len) {
    free(target);
    return SANITIZE_ERR_INVALID_PATH;
  }
#endif

  if(!(flags & SANITIZE_ALLOW_RESERVED)) {
    sc = rename_if_reserved_dos_device_name(&p, target, flags);
    free(target);
    if(sc)
      return sc;
    target = p;
    len = strlen(target);

    if(len > max_sanitized_len) {
      free(target);
      return SANITIZE_ERR_INVALID_PATH;
    }
  }

  *sanitized = target;
  return SANITIZE_ERR_OK;
}


/*
Test if truncating a path to a file will leave at least a single character in
the filename. Filenames suffixed by an alternate data stream can't be
truncated. This performs a dry run, nothing is modified.

Good truncate_pos 9:    C:\foo\bar  =>  C:\foo\ba
Good truncate_pos 6:    C:\foo      =>  C:\foo
Good truncate_pos 5:    C:\foo      =>  C:\fo
Bad* truncate_pos 5:    C:foo       =>  C:foo
Bad truncate_pos 5:     C:\foo:ads  =>  C:\fo
Bad truncate_pos 9:     C:\foo:ads  =>  C:\foo:ad
Bad truncate_pos 5:     C:\foo\bar  =>  C:\fo
Bad truncate_pos 5:     C:\foo\     =>  C:\fo
Bad truncate_pos 7:     C:\foo\     =>  C:\foo\
Error truncate_pos 7:   C:\foo      =>  (pos out of range)
Bad truncate_pos 1:     C:\foo\     =>  C

* C:foo is ambiguous, C could end up being a drive or file therefore something
  like C:superlongfilename can't be truncated.

Returns
SANITIZE_ERR_OK: Good -- 'path' can be truncated
SANITIZE_ERR_INVALID_PATH: Bad -- 'path' cannot be truncated
!= SANITIZE_ERR_OK && != SANITIZE_ERR_INVALID_PATH: Error
*/
SANITIZEcode truncate_dryrun(const char *path, const size_t truncate_pos)
{
  size_t len;

  if(!path)
    return SANITIZE_ERR_BAD_ARGUMENT;

  len = strlen(path);

  if(truncate_pos > len)
    return SANITIZE_ERR_BAD_ARGUMENT;

  if(!len || !truncate_pos)
    return SANITIZE_ERR_INVALID_PATH;

  if(strpbrk(&path[truncate_pos - 1], "\\/:"))
    return SANITIZE_ERR_INVALID_PATH;

  /* C:\foo can be truncated but C:\foo:ads can't */
  if(truncate_pos > 1) {
    const char *p = &path[truncate_pos - 1];
    do {
      --p;
      if(*p == ':')
        return SANITIZE_ERR_INVALID_PATH;
    } while(p != path && *p != '\\' && *p != '/');
  }

  return SANITIZE_ERR_OK;
}

/* The functions msdosify, rename_if_dos_device_name and __crt0_glob_function
 * were taken with modification from the DJGPP port of tar 1.12. They use
 * algorithms originally from DJTAR.
 */

/*
Extra sanitization MSDOS for file_name.

This is a supporting function for sanitize_file_name.

Warning: This is an MSDOS legacy function and was purposely written in a way
that some path information may pass through. For example drive letter names
(C:, D:, etc) are allowed to pass through. For sanitizing a filename use
sanitize_file_name.

Success: (SANITIZE_ERR_OK) *sanitized points to a sanitized copy of file_name.
Failure: (!= SANITIZE_ERR_OK) *sanitized is NULL.
*/
#if defined(MSDOS) || defined(UNITTESTS)
SANITIZEcode msdosify(char **const sanitized, const char *file_name,
                      int flags)
{
  char dos_name[PATH_MAX];
  static const char illegal_chars_dos[] = ".+, ;=[]" /* illegal in DOS */
    "|<>/\\\":?*"; /* illegal in DOS & W95 */
  static const char *illegal_chars_w95 = &illegal_chars_dos[8];
  int idx, dot_idx;
  const char *s = file_name;
  char *d = dos_name;
  const char *const dlimit = dos_name + sizeof(dos_name) - 1;
  const char *illegal_aliens = illegal_chars_dos;
  size_t len = sizeof(illegal_chars_dos) - 1;

  if(!sanitized)
    return SANITIZE_ERR_BAD_ARGUMENT;

  *sanitized = NULL;

  if(!file_name)
    return SANITIZE_ERR_BAD_ARGUMENT;

  if(strlen(file_name) > PATH_MAX-1 &&
     (!(flags & SANITIZE_ALLOW_TRUNCATE) ||
      truncate_dryrun(file_name, PATH_MAX-1)))
    return SANITIZE_ERR_INVALID_PATH;

  /* Support for Windows 9X VFAT systems, when available. */
  if(_use_lfn(file_name)) {
    illegal_aliens = illegal_chars_w95;
    len -= (illegal_chars_w95 - illegal_chars_dos);
  }

  /* Get past the drive letter, if any. */
  if(s[0] >= 'A' && s[0] <= 'z' && s[1] == ':') {
    *d++ = *s++;
    *d = ((flags & (SANITIZE_ALLOW_COLONS|SANITIZE_ALLOW_PATH))) ? ':' : '_';
    ++d, ++s;
  }

  for(idx = 0, dot_idx = -1; *s && d < dlimit; s++, d++) {
    if(memchr(illegal_aliens, *s, len)) {

      if((flags & (SANITIZE_ALLOW_COLONS|SANITIZE_ALLOW_PATH)) && *s == ':')
        *d = ':';
      else if((flags & SANITIZE_ALLOW_PATH) && (*s == '/' || *s == '\\'))
        *d = *s;
      /* Dots are special: DOS doesn't allow them as the leading character,
         and a file name cannot have more than a single dot.  We leave the
         first non-leading dot alone, unless it comes too close to the
         beginning of the name: we want sh.lex.c to become sh_lex.c, not
         sh.lex-c.  */
      else if(*s == '.') {
        if((flags & SANITIZE_ALLOW_PATH) && idx == 0 &&
           (s[1] == '/' || s[1] == '\\' ||
            (s[1] == '.' && (s[2] == '/' || s[2] == '\\')))) {
          /* Copy "./" and "../" verbatim.  */
          *d++ = *s++;
          if(d == dlimit)
            break;
          if(*s == '.') {
            *d++ = *s++;
            if(d == dlimit)
              break;
          }
          *d = *s;
        }
        else if(idx == 0)
          *d = '_';
        else if(dot_idx >= 0) {
          if(dot_idx < 5) { /* 5 is a heuristic ad-hoc'ery */
            d[dot_idx - idx] = '_'; /* replace previous dot */
            *d = '.';
          }
          else
            *d = '-';
        }
        else
          *d = '.';

        if(*s == '.')
          dot_idx = idx;
      }
      else if(*s == '+' && s[1] == '+') {
        if(idx - 2 == dot_idx) { /* .c++, .h++ etc. */
          *d++ = 'x';
          if(d == dlimit)
            break;
          *d   = 'x';
        }
        else {
          /* libg++ etc.  */
          if(dlimit - d < 4) {
            *d++ = 'x';
            if(d == dlimit)
              break;
            *d   = 'x';
          }
          else {
            memcpy(d, "plus", 4);
            d += 3;
          }
        }
        s++;
        idx++;
      }
      else
        *d = '_';
    }
    else
      *d = *s;
    if(*s == '/' || *s == '\\') {
      idx = 0;
      dot_idx = -1;
    }
    else
      idx++;
  }
  *d = '\0';

  if(*s) {
    /* dos_name is truncated, check that truncation requirements are met,
       specifically truncating a filename suffixed by an alternate data stream
       or truncating the entire filename is not allowed. */
    if(!(flags & SANITIZE_ALLOW_TRUNCATE) || strpbrk(s, "\\/:") ||
       truncate_dryrun(dos_name, d - dos_name))
      return SANITIZE_ERR_INVALID_PATH;
  }

  *sanitized = strdup(dos_name);
  return (*sanitized ? SANITIZE_ERR_OK : SANITIZE_ERR_OUT_OF_MEMORY);
}
#endif /* MSDOS || UNITTESTS */

/*
Rename file_name if it's a reserved dos device name.

This is a supporting function for sanitize_file_name.

Warning: This is an MSDOS legacy function and was purposely written in a way
that some path information may pass through. For example drive letter names
(C:, D:, etc) are allowed to pass through. For sanitizing a filename use
sanitize_file_name.

Success: (SANITIZE_ERR_OK) *sanitized points to a sanitized copy of file_name.
Failure: (!= SANITIZE_ERR_OK) *sanitized is NULL.
*/
SANITIZEcode rename_if_reserved_dos_device_name(char **const sanitized,
                                                const char *file_name,
                                                int flags)
{
  /* We could have a file whose name is a device on MS-DOS.  Trying to
   * retrieve such a file would fail at best and wedge us at worst.  We need
   * to rename such files. */
  char *p, *base;
  char fname[PATH_MAX];
#ifdef MSDOS
  struct_stat st_buf;
#endif

  if(!sanitized)
    return SANITIZE_ERR_BAD_ARGUMENT;

  *sanitized = NULL;

  if(!file_name)
    return SANITIZE_ERR_BAD_ARGUMENT;

  /* Ignore UNC prefixed paths, they are allowed to contain a reserved name. */
#ifndef MSDOS
  if((flags & SANITIZE_ALLOW_PATH) &&
     file_name[0] == '\\' && file_name[1] == '\\') {
    size_t len = strlen(file_name);
    *sanitized = malloc(len + 1);
    if(!*sanitized)
      return SANITIZE_ERR_OUT_OF_MEMORY;
    strncpy(*sanitized, file_name, len + 1);
    return SANITIZE_ERR_OK;
  }
#endif

  if(strlen(file_name) > PATH_MAX-1 &&
     (!(flags & SANITIZE_ALLOW_TRUNCATE) ||
      truncate_dryrun(file_name, PATH_MAX-1)))
    return SANITIZE_ERR_INVALID_PATH;

  strncpy(fname, file_name, PATH_MAX-1);
  fname[PATH_MAX-1] = '\0';
  base = basename(fname);

  /* Rename reserved device names that are known to be accessible without \\.\
     Examples: CON => _CON, CON.EXT => CON_EXT, CON:ADS => CON_ADS
     https://support.microsoft.com/en-us/kb/74496
     https://msdn.microsoft.com/en-us/library/windows/desktop/aa365247.aspx
     */
  for(p = fname; p; p = (p == fname && fname != base ? base : NULL)) {
    size_t p_len;
    int x = (curl_strnequal(p, "CON", 3) ||
             curl_strnequal(p, "PRN", 3) ||
             curl_strnequal(p, "AUX", 3) ||
             curl_strnequal(p, "NUL", 3)) ? 3 :
            (curl_strnequal(p, "CLOCK$", 6)) ? 6 :
            (curl_strnequal(p, "COM", 3) || curl_strnequal(p, "LPT", 3)) ?
              (('1' <= p[3] && p[3] <= '9') ? 4 : 3) : 0;

    if(!x)
      continue;

    /* the devices may be accessible with an extension or ADS, for
       example CON.AIR and 'CON . AIR' and CON:AIR access console */

    for(; p[x] == ' '; ++x)
      ;

    if(p[x] == '.') {
      p[x] = '_';
      continue;
    }
    else if(p[x] == ':') {
      if(!(flags & (SANITIZE_ALLOW_COLONS|SANITIZE_ALLOW_PATH))) {
        p[x] = '_';
        continue;
      }
      ++x;
    }
    else if(p[x]) /* no match */
      continue;

    /* p points to 'CON' or 'CON ' or 'CON:', etc */
    p_len = strlen(p);

    /* Prepend a '_' */
    if(strlen(fname) == PATH_MAX-1) {
      --p_len;
      if(!(flags & SANITIZE_ALLOW_TRUNCATE) || truncate_dryrun(p, p_len))
        return SANITIZE_ERR_INVALID_PATH;
      p[p_len] = '\0';
    }
    memmove(p + 1, p, p_len + 1);
    p[0] = '_';
    ++p_len;

    /* if fname was just modified then the basename pointer must be updated */
    if(p == fname)
      base = basename(fname);
  }

  /* This is the legacy portion from rename_if_dos_device_name that checks for
     reserved device names. It only works on MSDOS. On Windows XP the stat
     check errors with EINVAL if the device name is reserved. On Windows
     Vista/7/8 it sets mode S_IFREG (regular file or device). According to MSDN
     stat doc the latter behavior is correct, but that doesn't help us identify
     whether it's a reserved device name and not a regular file name. */
#ifdef MSDOS
  if(base && ((stat(base, &st_buf)) == 0) && (S_ISCHR(st_buf.st_mode))) {
    /* Prepend a '_' */
    size_t blen = strlen(base);
    if(blen) {
      if(strlen(fname) == PATH_MAX-1) {
        --blen;
        if(!(flags & SANITIZE_ALLOW_TRUNCATE) || truncate_dryrun(base, blen))
          return SANITIZE_ERR_INVALID_PATH;
        base[blen] = '\0';
      }
      memmove(base + 1, base, blen + 1);
      base[0] = '_';
    }
  }
#endif

  *sanitized = strdup(fname);
  return (*sanitized ? SANITIZE_ERR_OK : SANITIZE_ERR_OUT_OF_MEMORY);
}

#if defined(MSDOS) && (defined(__DJGPP__) || defined(__GO32__))

/*
 * Disable program default argument globbing. We do it on our own.
 */
char **__crt0_glob_function(char *arg)
{
  (void)arg;
  return (char **)0;
}

#endif /* MSDOS && (__DJGPP__ || __GO32__) */

#ifdef WIN32

/*
 * Function to find CACert bundle on a Win32 platform using SearchPath.
 * (SearchPath is already declared via inclusions done in setup header file)
 * (Use the ASCII version instead of the unicode one!)
 * The order of the directories it searches is:
 *  1. application's directory
 *  2. current working directory
 *  3. Windows System directory (e.g. C:\windows\system32)
 *  4. Windows Directory (e.g. C:\windows)
 *  5. all directories along %PATH%
 *
 * For WinXP and later search order actually depends on registry value:
 * HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\SafeProcessSearchMode
 */

CURLcode FindWin32CACert(struct OperationConfig *config,
                         curl_sslbackend backend,
                         const TCHAR *bundle_file)
{
  CURLcode result = CURLE_OK;

  /* Search and set cert file only if libcurl supports SSL.
   *
   * If Schannel is the selected SSL backend then these locations are
   * ignored. We allow setting CA location for schannel only when explicitly
   * specified by the user via CURLOPT_CAINFO / --cacert.
   */
  if((curlinfo->features & CURL_VERSION_SSL) &&
     backend != CURLSSLBACKEND_SCHANNEL) {

    DWORD res_len;
    TCHAR buf[PATH_MAX];
    TCHAR *ptr = NULL;

    buf[0] = TEXT('\0');

    res_len = SearchPath(NULL, bundle_file, NULL, PATH_MAX, buf, &ptr);
    if(res_len > 0) {
      Curl_safefree(config->cacert);
#ifdef UNICODE
      config->cacert = curlx_convert_wchar_to_UTF8(buf);
#else
      config->cacert = strdup(buf);
#endif
      if(!config->cacert)
        result = CURLE_OUT_OF_MEMORY;
    }
  }

  return result;
}


/* Get a list of all loaded modules with full paths.
 * Returns slist on success or NULL on error.
 */
struct curl_slist *GetLoadedModulePaths(void)
{
  HANDLE hnd = INVALID_HANDLE_VALUE;
  MODULEENTRY32 mod = {0};
  struct curl_slist *slist = NULL;

  mod.dwSize = sizeof(MODULEENTRY32);

  do {
    hnd = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
  } while(hnd == INVALID_HANDLE_VALUE && GetLastError() == ERROR_BAD_LENGTH);

  if(hnd == INVALID_HANDLE_VALUE)
    goto error;

  if(!Module32First(hnd, &mod))
    goto error;

  do {
    char *path; /* points to stack allocated buffer */
    struct curl_slist *temp;

#ifdef UNICODE
    /* sizeof(mod.szExePath) is the max total bytes of wchars. the max total
       bytes of multibyte chars won't be more than twice that. */
    char buffer[sizeof(mod.szExePath) * 2];
    if(!WideCharToMultiByte(CP_ACP, 0, mod.szExePath, -1,
                            buffer, sizeof(buffer), NULL, NULL))
      goto error;
    path = buffer;
#else
    path = mod.szExePath;
#endif
    temp = curl_slist_append(slist, path);
    if(!temp)
      goto error;
    slist = temp;
  } while(Module32Next(hnd, &mod));

  goto cleanup;

error:
  curl_slist_free_all(slist);
  slist = NULL;
cleanup:
  if(hnd != INVALID_HANDLE_VALUE)
    CloseHandle(hnd);
  return slist;
}

/* The terminal settings to restore on exit */
static struct TerminalSettings {
  HANDLE hStdOut;
  DWORD dwOutputMode;
  LONG valid;
} TerminalSettings;

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif

static void restore_terminal(void)
{
  if(InterlockedExchange(&TerminalSettings.valid, (LONG)FALSE))
    SetConsoleMode(TerminalSettings.hStdOut, TerminalSettings.dwOutputMode);
}

/* This is the console signal handler.
 * The system calls it in a separate thread.
 */
static BOOL WINAPI signal_handler(DWORD type)
{
  if(type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT)
    restore_terminal();
  return FALSE;
}

static void init_terminal(void)
{
  TerminalSettings.hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
  /*
   * Enable VT (Virtual Terminal) output.
   * Note: VT mode flag can be set on any version of Windows, but VT
   * processing only performed on Win10 >= Creators Update)
   */
  if((TerminalSettings.hStdOut != INVALID_HANDLE_VALUE) &&
     GetConsoleMode(TerminalSettings.hStdOut,
                    &TerminalSettings.dwOutputMode) &&
     !(TerminalSettings.dwOutputMode &
       ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
    /* The signal handler is set before attempting to change the console mode
       because otherwise a signal would not be caught after the change but
       before the handler was installed. */
    (void)InterlockedExchange(&TerminalSettings.valid, (LONG)TRUE);
    if(SetConsoleCtrlHandler(signal_handler, TRUE)) {
      if(SetConsoleMode(TerminalSettings.hStdOut,
                        (TerminalSettings.dwOutputMode |
                         ENABLE_VIRTUAL_TERMINAL_PROCESSING))) {
        atexit(restore_terminal);
      }
      else {
        SetConsoleCtrlHandler(signal_handler, FALSE);
        (void)InterlockedExchange(&TerminalSettings.valid, (LONG)FALSE);
      }
    }
  }
}

LARGE_INTEGER tool_freq;
bool tool_isVistaOrGreater;

CURLcode win32_init(void)
{
  /* curlx_verify_windows_version must be called during init at least once
     because it has its own initialization routine. */
  if(curlx_verify_windows_version(6, 0, 0, PLATFORM_WINNT,
                                  VERSION_GREATER_THAN_EQUAL))
    tool_isVistaOrGreater = true;
  else
    tool_isVistaOrGreater = false;

  QueryPerformanceFrequency(&tool_freq);

  init_terminal();

  return CURLE_OK;
}

#endif /* WIN32 */

#endif /* MSDOS || WIN32 */
