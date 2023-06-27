/***********************************************************

Copyright (c) 1999  The XFree86 Project Inc.

All Rights Reserved.

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
OPEN GROUP BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name of The XFree86 Project
Inc. shall not be used in advertising or otherwise to promote the
sale, use or other dealings in this Software without prior written
authorization from The XFree86 Project Inc..

*/

/**
 ** Types definitions shared between server and clients
 **/

#ifndef _XDEFS_H
#define _XDEFS_H

#ifdef _XSERVER64
#include <X11/Xmd.h>
#endif

#ifndef _XTYPEDEF_ATOM
#  define _XTYPEDEF_ATOM
#  ifndef _XSERVER64
typedef unsigned long Atom;
#  else
typedef CARD32 Atom;
#  endif
#endif

#ifndef Bool
#  ifndef _XTYPEDEF_BOOL
#   define _XTYPEDEF_BOOL
typedef int Bool;
#  endif
#endif

#ifndef _XTYPEDEF_POINTER
#  define _XTYPEDEF_POINTER
typedef void *pointer;
#endif

#ifndef _XTYPEDEF_CLIENTPTR
typedef struct _Client *ClientPtr;
#  define _XTYPEDEF_CLIENTPTR
#endif

#ifndef _XTYPEDEF_XID
#  define _XTYPEDEF_XID
#  ifndef _XSERVER64
typedef unsigned long XID;
#  else
typedef CARD32 XID;
#  endif
#endif

#ifndef _XTYPEDEF_MASK
#  define _XTYPEDEF_MASK
#  ifndef _XSERVER64
typedef unsigned long Mask;
#  else
typedef CARD32 Mask;
#  endif
#endif

#ifndef _XTYPEDEF_FONTPTR
#  define _XTYPEDEF_FONTPTR
typedef struct _Font *FontPtr; /* also in fonts/include/font.h */
#endif

#ifndef _XTYPEDEF_FONT
#  define _XTYPEDEF_FONT
typedef XID	Font;
#endif

#ifndef _XTYPEDEF_FSID
#  ifndef _XSERVER64
typedef unsigned long FSID;
#  else
typedef CARD32 FSID;
#  endif
#endif

typedef FSID AccContext;

/* OS independent time value
   XXX Should probably go in Xos.h */
typedef struct timeval **OSTimePtr;


typedef void (* BlockHandlerProcPtr)(void * /* blockData */,
				     OSTimePtr /* pTimeout */,
				     void * /* pReadmask */);

#endif
