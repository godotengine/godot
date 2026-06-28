/*
 * O/S-dependent (mis)feature macro definitions
 *
Copyright 1991, 1998  The Open Group

Permission to use, copy, modify, distribute, and sell this software and its
documentation for any purpose is hereby granted without fee, provided that
the above copyright notice appear in all copies and that both that
copyright notice and this permission notice appear in supporting
documentation.

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
OPEN GROUP BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name of The Open Group shall not be
used in advertising or otherwise to promote the sale, use or other dealings
in this Software without prior written authorization from The Open Group.
 */

#ifndef _XOSDEFS_H_
# define _XOSDEFS_H_

/*
 * X_NOT_POSIX means does not have POSIX header files.  Lack of this
 * symbol does NOT mean that the POSIX environment is the default.
 * You may still have to define _POSIX_SOURCE to get it.
 */


# ifdef _SCO_DS
#  ifndef __SCO__
#   define __SCO__
#  endif
# endif

# ifdef __i386__
#  ifdef SYSV
#   if !defined(__SCO__) && \
	!defined(__UNIXWARE__) && !defined(__sun)
#    if !defined(_POSIX_SOURCE)
#     define X_NOT_POSIX
#    endif
#   endif
#  endif
# endif

# ifdef __sun
/* Imake configs define SVR4 on Solaris, but cc & gcc only define __SVR4
 * This check allows non-Imake configured programs to build correctly.
 */
#  if defined(__SVR4) && !defined(SVR4)
#   define SVR4 1
#  endif
#  ifdef SVR4
/* define this to whatever it needs to be */
#   define X_POSIX_C_SOURCE 199300L
#  endif
# endif

# ifdef WIN32
#  ifndef _POSIX_
#   define X_NOT_POSIX
#  endif
# endif


# ifdef __APPLE__
#  define NULL_NOT_ZERO

/* Defining any of these will sanitize the namespace to JUST want is defined by
 * that particular standard.  If that happens, we don't get some expected
 * prototypes, typedefs, etc (like fd_mask).  We can define _DARWIN_C_SOURCE to
 * loosen our belts a tad.
 */
#  if defined(_XOPEN_SOURCE) || defined(_POSIX_SOURCE) || defined(_POSIX_C_SOURCE)
#   ifndef _DARWIN_C_SOURCE
#    define _DARWIN_C_SOURCE
#   endif
#  endif

# endif

# ifdef __GNU__
#  ifndef PATH_MAX
#   define PATH_MAX 4096
#  endif
#  ifndef MAXPATHLEN
#   define MAXPATHLEN 4096
#  endif
# endif

# if defined(__SCO__) || defined(__UNIXWARE__)
#  ifndef PATH_MAX
#   define PATH_MAX	1024
#  endif
#  ifndef MAXPATHLEN
#   define MAXPATHLEN	1024
#  endif
# endif

# if defined(__OpenBSD__) || defined(__NetBSD__) || defined(__FreeBSD__) \
	|| defined(__APPLE__) || defined(__DragonFly__)
#  ifndef CSRG_BASED
#   define CSRG_BASED
#  endif
# endif

#endif /* _XOSDEFS_H_ */

