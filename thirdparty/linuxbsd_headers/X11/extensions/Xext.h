/*
 *
Copyright 1989, 1998  The Open Group

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

#ifndef _XEXT_H_
#define _XEXT_H_

#include <X11/Xfuncproto.h>

_XFUNCPROTOBEGIN

typedef int (*XextErrorHandler) (
    Display *		/* dpy */,
    _Xconst char*	/* ext_name */,
    _Xconst char*	/* reason */
);

extern XextErrorHandler XSetExtensionErrorHandler(
    XextErrorHandler	/* handler */
);

extern int XMissingExtension(
    Display*		/* dpy */,
    _Xconst char*	/* ext_name */
);

_XFUNCPROTOEND

#define X_EXTENSION_UNKNOWN "unknown"
#define X_EXTENSION_MISSING "missing"

#endif /* _XEXT_H_ */
