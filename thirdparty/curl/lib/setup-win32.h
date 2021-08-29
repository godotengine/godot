#ifndef HEADER_CURL_SETUP_WIN32_H
#define HEADER_CURL_SETUP_WIN32_H
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
 * Include header files for windows builds before redefining anything.
 * Use this preprocessor block only to include or exclude windows.h,
 * winsock2.h or ws2tcpip.h. Any other windows thing belongs
 * to any other further and independent block.  Under Cygwin things work
 * just as under linux (e.g. <sys/socket.h>) and the winsock headers should
 * never be included when __CYGWIN__ is defined.  configure script takes
 * care of this, not defining HAVE_WINDOWS_H, HAVE_WINSOCK2_H,
 * neither HAVE_WS2TCPIP_H when __CYGWIN__ is defined.
 */

#ifdef HAVE_WINDOWS_H
#  if defined(UNICODE) && !defined(_UNICODE)
#    define _UNICODE
#  endif
#  if defined(_UNICODE) && !defined(UNICODE)
#    define UNICODE
#  endif
#  include <winerror.h>
#  include <windows.h>
#  ifdef HAVE_WINSOCK2_H
#    include <winsock2.h>
#    ifdef HAVE_WS2TCPIP_H
#      include <ws2tcpip.h>
#    endif
#  endif
#  include <tchar.h>
#  ifdef UNICODE
     typedef wchar_t *(*curl_wcsdup_callback)(const wchar_t *str);
#  endif
#endif

/*
 * Define USE_WINSOCK to 2 if we have and use WINSOCK2 API, else
 * undefine USE_WINSOCK.
 */

#undef USE_WINSOCK

#ifdef HAVE_WINSOCK2_H
#  define USE_WINSOCK 2
#endif

/*
 * Define _WIN32_WINNT_[OS] symbols because not all Windows build systems have
 * those symbols to compare against, and even those that do may be missing
 * newer symbols.
 */

#ifndef _WIN32_WINNT_NT4
#define _WIN32_WINNT_NT4            0x0400   /* Windows NT 4.0 */
#endif
#ifndef _WIN32_WINNT_WIN2K
#define _WIN32_WINNT_WIN2K          0x0500   /* Windows 2000 */
#endif
#ifndef _WIN32_WINNT_WINXP
#define _WIN32_WINNT_WINXP          0x0501   /* Windows XP */
#endif
#ifndef _WIN32_WINNT_WS03
#define _WIN32_WINNT_WS03           0x0502   /* Windows Server 2003 */
#endif
#ifndef _WIN32_WINNT_WIN6
#define _WIN32_WINNT_WIN6           0x0600   /* Windows Vista */
#endif
#ifndef _WIN32_WINNT_VISTA
#define _WIN32_WINNT_VISTA          0x0600   /* Windows Vista */
#endif
#ifndef _WIN32_WINNT_WS08
#define _WIN32_WINNT_WS08           0x0600   /* Windows Server 2008 */
#endif
#ifndef _WIN32_WINNT_LONGHORN
#define _WIN32_WINNT_LONGHORN       0x0600   /* Windows Vista */
#endif
#ifndef _WIN32_WINNT_WIN7
#define _WIN32_WINNT_WIN7           0x0601   /* Windows 7 */
#endif
#ifndef _WIN32_WINNT_WIN8
#define _WIN32_WINNT_WIN8           0x0602   /* Windows 8 */
#endif
#ifndef _WIN32_WINNT_WINBLUE
#define _WIN32_WINNT_WINBLUE        0x0603   /* Windows 8.1 */
#endif
#ifndef _WIN32_WINNT_WINTHRESHOLD
#define _WIN32_WINNT_WINTHRESHOLD   0x0A00   /* Windows 10 */
#endif
#ifndef _WIN32_WINNT_WIN10
#define _WIN32_WINNT_WIN10          0x0A00   /* Windows 10 */
#endif

#endif /* HEADER_CURL_SETUP_WIN32_H */
