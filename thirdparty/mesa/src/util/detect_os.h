/* SPDX-License-Identifier: MIT */
/* Copyright 2008 VMware, Inc. */

/**
 * Auto-detect the operating system family.
 *
 * See also:
 * - http://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html
 * - echo | gcc -dM -E - | sort
 * - http://msdn.microsoft.com/en-us/library/b0084kay.aspx
 *
 * @author José Fonseca <jfonseca@vmware.com>
 */

#ifndef DETECT_OS_H
#define DETECT_OS_H

#if defined(__linux__)
#define DETECT_OS_LINUX 1
#define DETECT_OS_UNIX 1
#endif

/*
 * Android defines __linux__, so DETECT_OS_LINUX and DETECT_OS_UNIX will
 * also be defined.
 */
#if defined(ANDROID)
#define DETECT_OS_ANDROID 1
#endif

#if defined(__FreeBSD__) || defined(__FreeBSD_kernel__)
#define DETECT_OS_FREEBSD 1
#define DETECT_OS_BSD 1
#define DETECT_OS_UNIX 1
#endif

#if defined(__OpenBSD__)
#define DETECT_OS_OPENBSD 1
#define DETECT_OS_BSD 1
#define DETECT_OS_UNIX 1
#endif

#if defined(__NetBSD__)
#define DETECT_OS_NETBSD 1
#define DETECT_OS_BSD 1
#define DETECT_OS_UNIX 1
#endif

#if defined(__DragonFly__)
#define DETECT_OS_DRAGONFLY 1
#define DETECT_OS_BSD 1
#define DETECT_OS_UNIX 1
#endif

#if defined(__GNU__)
#define DETECT_OS_HURD 1
#define DETECT_OS_UNIX 1
#endif

#if defined(__sun)
#define DETECT_OS_SOLARIS 1
#define DETECT_OS_UNIX 1
#endif

#if defined(__APPLE__)
#define DETECT_OS_APPLE 1
#define DETECT_OS_UNIX 1
#endif

#if defined(_WIN32) || defined(WIN32)
#define DETECT_OS_WINDOWS 1
#endif

#if defined(__HAIKU__)
#define DETECT_OS_HAIKU 1
#define DETECT_OS_UNIX 1
#endif

#if defined(__CYGWIN__)
#define DETECT_OS_CYGWIN 1
#define DETECT_OS_UNIX 1
#endif


/*
 * Make sure DETECT_OS_* are always defined, so that they can be used with #if
 */
#ifndef DETECT_OS_ANDROID
#define DETECT_OS_ANDROID 0
#endif
#ifndef DETECT_OS_APPLE
#define DETECT_OS_APPLE 0
#endif
#ifndef DETECT_OS_BSD
#define DETECT_OS_BSD 0
#endif
#ifndef DETECT_OS_CYGWIN
#define DETECT_OS_CYGWIN 0
#endif
#ifndef DETECT_OS_DRAGONFLY
#define DETECT_OS_DRAGONFLY 0
#endif
#ifndef DETECT_OS_FREEBSD
#define DETECT_OS_FREEBSD 0
#endif
#ifndef DETECT_OS_HAIKU
#define DETECT_OS_HAIKU 0
#endif
#ifndef DETECT_OS_HURD
#define DETECT_OS_HURD 0
#endif
#ifndef DETECT_OS_LINUX
#define DETECT_OS_LINUX 0
#endif
#ifndef DETECT_OS_NETBSD
#define DETECT_OS_NETBSD 0
#endif
#ifndef DETECT_OS_OPENBSD
#define DETECT_OS_OPENBSD 0
#endif
#ifndef DETECT_OS_SOLARIS
#define DETECT_OS_SOLARIS 0
#endif
#ifndef DETECT_OS_UNIX
#define DETECT_OS_UNIX 0
#endif
#ifndef DETECT_OS_WINDOWS
#define DETECT_OS_WINDOWS 0
#endif

#endif /* DETECT_OS_H */
