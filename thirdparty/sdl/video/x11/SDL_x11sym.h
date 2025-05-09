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

/* *INDENT-OFF* */ // clang-format off

#ifndef SDL_X11_MODULE
#define SDL_X11_MODULE(modname)
#endif

#ifndef SDL_X11_SYM
#define SDL_X11_SYM(rc, fn, params)
#endif

SDL_X11_MODULE(BASEXLIB)
SDL_X11_SYM(XSizeHints*,XAllocSizeHints,(void))
SDL_X11_SYM(XWMHints*,XAllocWMHints,(void))
SDL_X11_SYM(XClassHint*,XAllocClassHint,(void))
SDL_X11_SYM(int,XChangePointerControl,(Display* a,Bool b,Bool c,int d,int e,int f))
SDL_X11_SYM(int,XChangeProperty,(Display* a,Window b,Atom c,Atom d,int e,int f,_Xconst unsigned char* g,int h))
SDL_X11_SYM(Bool,XCheckIfEvent,(Display* a,XEvent *b,Bool (*c)(Display*,XEvent*,XPointer),XPointer d))
SDL_X11_SYM(int,XClearWindow,(Display* a,Window b))
SDL_X11_SYM(int,XCloseDisplay,(Display* a))
SDL_X11_SYM(int,XConvertSelection,(Display* a,Atom b,Atom c,Atom d,Window e,Time f))
SDL_X11_SYM(Pixmap,XCreateBitmapFromData,(Display *dpy,Drawable d,_Xconst char *data,unsigned int width,unsigned int height))
SDL_X11_SYM(Colormap,XCreateColormap,(Display* a,Window b,Visual* c,int d))
SDL_X11_SYM(Cursor,XCreatePixmapCursor,(Display* a,Pixmap b,Pixmap c,XColor* d,XColor* e,unsigned int f,unsigned int g))
SDL_X11_SYM(Cursor,XCreateFontCursor,(Display* a,unsigned int b))
SDL_X11_SYM(XFontSet,XCreateFontSet,(Display* a, _Xconst char* b, char*** c, int* d, char** e))
SDL_X11_SYM(GC,XCreateGC,(Display* a,Drawable b,unsigned long c,XGCValues* d))
SDL_X11_SYM(XImage*,XCreateImage,(Display* a,Visual* b,unsigned int c,int d,int e,char* f,unsigned int g,unsigned int h,int i,int j))
SDL_X11_SYM(Window,XCreateWindow,(Display* a,Window b,int c,int d,unsigned int e,unsigned int f,unsigned int g,int h,unsigned int i,Visual* j,unsigned long k,XSetWindowAttributes* l))
SDL_X11_SYM(int,XDefineCursor,(Display* a,Window b,Cursor c))
SDL_X11_SYM(int,XDeleteProperty,(Display* a,Window b,Atom c))
SDL_X11_SYM(int,XDestroyWindow,(Display* a,Window b))
SDL_X11_SYM(int,XDisplayKeycodes,(Display* a,int* b,int* c))
SDL_X11_SYM(int,XDrawRectangle,(Display* a,Drawable b,GC c,int d,int e,unsigned int f,unsigned int g))
SDL_X11_SYM(char*,XDisplayName,(_Xconst char* a))
SDL_X11_SYM(int,XDrawString,(Display* a,Drawable b,GC c,int d,int e,_Xconst char* f,int g))
SDL_X11_SYM(int,XEventsQueued,(Display* a,int b))
SDL_X11_SYM(int,XFillRectangle,(Display* a,Drawable b,GC c,int d,int e,unsigned int f,unsigned int g))
SDL_X11_SYM(Bool,XFilterEvent,(XEvent *event,Window w))
SDL_X11_SYM(int,XFlush,(Display* a))
SDL_X11_SYM(int,XFree,(void*a))
SDL_X11_SYM(int,XFreeCursor,(Display* a,Cursor b))
SDL_X11_SYM(void,XFreeFontSet,(Display* a, XFontSet b))
SDL_X11_SYM(int,XFreeGC,(Display* a,GC b))
SDL_X11_SYM(int,XFreeFont,(Display* a, XFontStruct* b))
SDL_X11_SYM(int,XFreeModifiermap,(XModifierKeymap* a))
SDL_X11_SYM(int,XFreePixmap,(Display* a,Pixmap b))
SDL_X11_SYM(void,XFreeStringList,(char** a))
SDL_X11_SYM(char*,XGetAtomName,(Display *a,Atom b))
SDL_X11_SYM(int,XGetInputFocus,(Display *a,Window *b,int *c))
SDL_X11_SYM(int,XGetErrorDatabaseText,(Display* a,_Xconst char* b,_Xconst char* c,_Xconst char* d,char* e,int f))
SDL_X11_SYM(XModifierKeymap*,XGetModifierMapping,(Display* a))
SDL_X11_SYM(int,XGetPointerControl,(Display* a,int* b,int* c,int* d))
SDL_X11_SYM(Window,XGetSelectionOwner,(Display* a,Atom b))
SDL_X11_SYM(XVisualInfo*,XGetVisualInfo,(Display* a,long b,XVisualInfo* c,int* d))
SDL_X11_SYM(Status,XGetWindowAttributes,(Display* a,Window b,XWindowAttributes* c))
SDL_X11_SYM(int,XGetWindowProperty,(Display* a,Window b,Atom c,long d,long e,Bool f,Atom g,Atom* h,int* i,unsigned long* j,unsigned long *k,unsigned char **l))
SDL_X11_SYM(XWMHints*,XGetWMHints,(Display* a,Window b))
SDL_X11_SYM(Status,XGetWMNormalHints,(Display *a,Window b, XSizeHints *c, long *d))
SDL_X11_SYM(int,XIfEvent,(Display* a,XEvent *b,Bool (*c)(Display*,XEvent*,XPointer),XPointer d))
SDL_X11_SYM(int,XGrabKeyboard,(Display* a,Window b,Bool c,int d,int e,Time f))
SDL_X11_SYM(int,XGrabPointer,(Display* a,Window b,Bool c,unsigned int d,int e,int f,Window g,Cursor h,Time i))
SDL_X11_SYM(int,XGrabServer,(Display* a))
SDL_X11_SYM(Status,XIconifyWindow,(Display* a,Window b,int c))
SDL_X11_SYM(KeyCode,XKeysymToKeycode,(Display* a,KeySym b))
SDL_X11_SYM(char*,XKeysymToString,(KeySym a))
SDL_X11_SYM(int,XInstallColormap,(Display* a,Colormap b))
SDL_X11_SYM(Atom,XInternAtom,(Display* a,_Xconst char* b,Bool c))
SDL_X11_SYM(XPixmapFormatValues*,XListPixmapFormats,(Display* a,int* b))
SDL_X11_SYM(XFontStruct*,XLoadQueryFont,(Display* a,_Xconst char* b))
SDL_X11_SYM(KeySym,XLookupKeysym,(XKeyEvent* a,int b))
SDL_X11_SYM(int,XLookupString,(XKeyEvent* a,char* b,int c,KeySym* d,XComposeStatus* e))
SDL_X11_SYM(int,XMapRaised,(Display* a,Window b))
SDL_X11_SYM(Status,XMatchVisualInfo,(Display* a,int b,int c,int d,XVisualInfo* e))
SDL_X11_SYM(int,XMissingExtension,(Display* a,_Xconst char* b))
SDL_X11_SYM(int,XMoveWindow,(Display* a,Window b,int c,int d))
SDL_X11_SYM(Display*,XOpenDisplay,(_Xconst char* a))
SDL_X11_SYM(Status,XInitThreads,(void))
SDL_X11_SYM(int,XPeekEvent,(Display* a,XEvent* b))
SDL_X11_SYM(int,XPending,(Display* a))
SDL_X11_SYM(int,XPutImage,(Display* a,Drawable b,GC c,XImage* d,int e,int f,int g,int h,unsigned int i,unsigned int j))
SDL_X11_SYM(int,XQueryKeymap,(Display* a,char b[32]))
SDL_X11_SYM(Bool,XQueryPointer,(Display* a,Window b,Window* c,Window* d,int* e,int* f,int* g,int* h,unsigned int* i))
SDL_X11_SYM(int,XRaiseWindow,(Display* a,Window b))
SDL_X11_SYM(int,XReparentWindow,(Display* a,Window b,Window c,int d,int e))
SDL_X11_SYM(int,XResetScreenSaver,(Display* a))
SDL_X11_SYM(int,XResizeWindow,(Display* a,Window b,unsigned int c,unsigned int d))
SDL_X11_SYM(int,XScreenNumberOfScreen,(Screen* a))
SDL_X11_SYM(int,XSelectInput,(Display* a,Window b,long c))
SDL_X11_SYM(Status,XSendEvent,(Display* a,Window b,Bool c,long d,XEvent* e))
SDL_X11_SYM(XErrorHandler,XSetErrorHandler,(XErrorHandler a))
SDL_X11_SYM(int,XSetForeground,(Display* a,GC b,unsigned long c))
SDL_X11_SYM(XIOErrorHandler,XSetIOErrorHandler,(XIOErrorHandler a))
SDL_X11_SYM(int,XSetInputFocus,(Display *a,Window b,int c,Time d))
SDL_X11_SYM(int,XSetSelectionOwner,(Display* a,Atom b,Window c,Time d))
SDL_X11_SYM(int,XSetTransientForHint,(Display* a,Window b,Window c))
SDL_X11_SYM(void,XSetTextProperty,(Display* a,Window b,XTextProperty* c,Atom d))
SDL_X11_SYM(int,XSetWindowBackground,(Display* a,Window b,unsigned long c))
SDL_X11_SYM(void,XSetWMHints,(Display* a,Window b,XWMHints* c))
SDL_X11_SYM(void,XSetWMNormalHints,(Display* a,Window b,XSizeHints* c))
SDL_X11_SYM(void,XSetWMProperties,(Display* a,Window b,XTextProperty* c,XTextProperty* d,char** e,int f,XSizeHints* g,XWMHints* h,XClassHint* i))
SDL_X11_SYM(Status,XSetWMProtocols,(Display* a,Window b,Atom* c,int d))
SDL_X11_SYM(int,XStoreColors,(Display* a,Colormap b,XColor* c,int d))
SDL_X11_SYM(int,XStoreName,(Display* a,Window b,_Xconst char* c))
SDL_X11_SYM(Status,XStringListToTextProperty,(char** a,int b,XTextProperty* c))
SDL_X11_SYM(int,XSync,(Display* a,Bool b))
SDL_X11_SYM(int,XTextExtents,(XFontStruct* a,_Xconst char* b,int c,int* d,int* e,int* f,XCharStruct* g))
SDL_X11_SYM(Bool,XTranslateCoordinates,(Display *a,Window b,Window c,int d,int e,int* f,int* g,Window* h))
SDL_X11_SYM(int,XUndefineCursor,(Display* a,Window b))
SDL_X11_SYM(int,XUngrabKeyboard,(Display* a,Time b))
SDL_X11_SYM(int,XUngrabPointer,(Display* a,Time b))
SDL_X11_SYM(int,XUngrabServer,(Display* a))
SDL_X11_SYM(int,XUninstallColormap,(Display* a,Colormap b))
SDL_X11_SYM(int,XUnloadFont,(Display* a,Font b))
SDL_X11_SYM(int,XWarpPointer,(Display* a,Window b,Window c,int d,int e,unsigned int f,unsigned int g,int h,int i))
SDL_X11_SYM(int,XWindowEvent,(Display* a,Window b,long c,XEvent* d))
SDL_X11_SYM(Status,XWithdrawWindow,(Display* a,Window b,int c))
SDL_X11_SYM(VisualID,XVisualIDFromVisual,(Visual* a))
SDL_X11_SYM(char*,XGetDefault,(Display* a,_Xconst char* b, _Xconst char* c))
SDL_X11_SYM(Bool,XQueryExtension,(Display* a,_Xconst char* b,int* c,int* d,int* e))
SDL_X11_SYM(char *,XDisplayString,(Display* a))
SDL_X11_SYM(int,XGetErrorText,(Display* a,int b,char* c,int d))
SDL_X11_SYM(void,_XEatData,(Display* a,unsigned long b))
SDL_X11_SYM(void,_XFlush,(Display* a))
SDL_X11_SYM(void,_XFlushGCCache,(Display* a,GC b))
SDL_X11_SYM(int,_XRead,(Display* a,char* b,long c))
SDL_X11_SYM(void,_XReadPad,(Display* a,char* b,long c))
SDL_X11_SYM(void,_XSend,(Display* a,_Xconst char* b,long c))
SDL_X11_SYM(Status,_XReply,(Display* a,xReply* b,int c,Bool d))
SDL_X11_SYM(unsigned long,_XSetLastRequestRead,(Display* a,xGenericReply* b))
SDL_X11_SYM(SDL_X11_XSynchronizeRetType,XSynchronize,(Display* a,Bool b))
SDL_X11_SYM(SDL_X11_XESetWireToEventRetType,XESetWireToEvent,(Display* a,int b,SDL_X11_XESetWireToEventRetType c))
SDL_X11_SYM(SDL_X11_XESetEventToWireRetType,XESetEventToWire,(Display* a,int b,SDL_X11_XESetEventToWireRetType c))
SDL_X11_SYM(void,XRefreshKeyboardMapping,(XMappingEvent *a))
SDL_X11_SYM(int,XQueryTree,(Display* a,Window b,Window* c,Window* d,Window** e,unsigned int* f))
SDL_X11_SYM(Bool,XSupportsLocale,(void))
SDL_X11_SYM(Status,XmbTextListToTextProperty,(Display* a,char** b,int c,XICCEncodingStyle d,XTextProperty* e))
SDL_X11_SYM(Region,XCreateRegion,(void))
SDL_X11_SYM(int,XUnionRectWithRegion,(XRectangle *a, Region b, Region c))
SDL_X11_SYM(void,XDestroyRegion,(Region))
SDL_X11_SYM(void,XrmInitialize,(void))
SDL_X11_SYM(char*,XResourceManagerString,(Display *display))
SDL_X11_SYM(XrmDatabase,XrmGetStringDatabase,(char *data))
SDL_X11_SYM(void,XrmDestroyDatabase,(XrmDatabase db))
SDL_X11_SYM(Bool,XrmGetResource,(XrmDatabase db, char* str_name, char* str_class, char **str_type_return, XrmValue *))

#ifdef SDL_VIDEO_DRIVER_X11_XFIXES
SDL_X11_MODULE(XFIXES)
SDL_X11_SYM(PointerBarrier, XFixesCreatePointerBarrier, (Display* a, Window b, int c, int d, int e, int f, int g, int h, int *i))
SDL_X11_SYM(void, XFixesDestroyPointerBarrier, (Display* a, PointerBarrier b))
SDL_X11_SYM(int, XIBarrierReleasePointer,(Display* a,  int b, PointerBarrier c, BarrierEventID d)) // this is actually Xinput2
SDL_X11_SYM(Status, XFixesQueryVersion,(Display* a, int* b, int* c))
SDL_X11_SYM(Status, XFixesSelectSelectionInput, (Display* a, Window b, Atom c, unsigned long d))
#endif

#ifdef SDL_VIDEO_DRIVER_X11_XSYNC
SDL_X11_MODULE(XSYNC)
SDL_X11_SYM(Status, XSyncQueryExtension, (Display* a, int* b, int* c))
SDL_X11_SYM(Status, XSyncInitialize, (Display* a, int* b, int* c))
SDL_X11_SYM(XSyncCounter, XSyncCreateCounter, (Display* a, XSyncValue b))
SDL_X11_SYM(Status, XSyncDestroyCounter, (Display* a, XSyncCounter b))
SDL_X11_SYM(Status, XSyncSetCounter, (Display* a, XSyncCounter b, XSyncValue c))
#endif

#ifdef SDL_VIDEO_DRIVER_X11_XTEST
SDL_X11_MODULE(XTEST)
SDL_X11_SYM(Status, XTestQueryExtension, (Display* a, int* b, int* c))
SDL_X11_SYM(int, XTestFakeMotionEvent, (Display* a, int b, int c, int d, unsigned long e))
#endif

#ifdef SDL_VIDEO_DRIVER_X11_SUPPORTS_GENERIC_EVENTS
SDL_X11_SYM(Bool,XGetEventData,(Display* a,XGenericEventCookie* b))
SDL_X11_SYM(void,XFreeEventData,(Display* a,XGenericEventCookie* b))
#endif

#ifdef SDL_VIDEO_DRIVER_X11_HAS_XKBLOOKUPKEYSYM
SDL_X11_SYM(Bool,XkbQueryExtension,(Display* a,int * b,int * c,int * d,int * e, int *f))
#if NeedWidePrototypes
SDL_X11_SYM(Bool,XkbLookupKeySym,(Display* a, unsigned int b, unsigned int c, unsigned int* d, KeySym* e))
#else
SDL_X11_SYM(Bool,XkbLookupKeySym,(Display* a, KeyCode b, unsigned int c, unsigned int* d, KeySym* e))
#endif
SDL_X11_SYM(Status,XkbGetState,(Display* a,unsigned int b,XkbStatePtr c))
SDL_X11_SYM(Status,XkbGetUpdatedMap,(Display* a,unsigned int b,XkbDescPtr c))
SDL_X11_SYM(XkbDescPtr,XkbGetMap,(Display* a,unsigned int b,unsigned int c))
SDL_X11_SYM(void,XkbFreeClientMap,(XkbDescPtr a,unsigned int b, Bool c))
SDL_X11_SYM(void,XkbFreeKeyboard,(XkbDescPtr a,unsigned int b, Bool c))
SDL_X11_SYM(Bool,XkbSetDetectableAutoRepeat,(Display* a, Bool b, Bool* c))
#endif

// XKeycodeToKeysym is a deprecated function
#ifdef HAVE_GCC_DIAGNOSTIC_PRAGMA
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#if NeedWidePrototypes
SDL_X11_SYM(KeySym,XKeycodeToKeysym,(Display* a,unsigned int b,int c))
#else
SDL_X11_SYM(KeySym,XKeycodeToKeysym,(Display* a,KeyCode b,int c))
#endif
#ifdef HAVE_GCC_DIAGNOSTIC_PRAGMA
#pragma GCC diagnostic pop
#endif

#ifdef X_HAVE_UTF8_STRING
SDL_X11_MODULE(UTF8)
SDL_X11_SYM(int,Xutf8TextListToTextProperty,(Display* a,char** b,int c,XICCEncodingStyle d,XTextProperty* e))
SDL_X11_SYM(int,Xutf8LookupString,(XIC a,XKeyPressedEvent* b,char* c,int d,KeySym* e,Status* f))
SDL_X11_SYM(XIC,XCreateIC,(XIM,...))
SDL_X11_SYM(void,XDestroyIC,(XIC a))
SDL_X11_SYM(char*,XGetICValues,(XIC,...))
SDL_X11_SYM(char*,XSetICValues,(XIC,...))
SDL_X11_SYM(XVaNestedList,XVaCreateNestedList,(int, ...))
SDL_X11_SYM(void,XSetICFocus,(XIC a))
SDL_X11_SYM(void,XUnsetICFocus,(XIC a))
SDL_X11_SYM(XIM,XOpenIM,(Display* a,struct _XrmHashBucketRec* b,char* c,char* d))
SDL_X11_SYM(Status,XCloseIM,(XIM a))
SDL_X11_SYM(void,Xutf8DrawString,(Display *a, Drawable b, XFontSet c, GC d, int e, int f, _Xconst char *g, int h))
SDL_X11_SYM(int,Xutf8TextExtents,(XFontSet a, _Xconst char* b, int c, XRectangle* d, XRectangle* e))
SDL_X11_SYM(char*,XSetLocaleModifiers,(const char *a))
SDL_X11_SYM(char*,Xutf8ResetIC,(XIC a))
#endif

#ifndef NO_SHARED_MEMORY
SDL_X11_MODULE(SHM)
SDL_X11_SYM(Status,XShmAttach,(Display* a,XShmSegmentInfo* b))
SDL_X11_SYM(Status,XShmDetach,(Display* a,XShmSegmentInfo* b))
SDL_X11_SYM(Status,XShmPutImage,(Display* a,Drawable b,GC c,XImage* d,int e,int f,int g,int h,unsigned int i,unsigned int j,Bool k))
SDL_X11_SYM(XImage*,XShmCreateImage,(Display* a,Visual* b,unsigned int c,int d,char* e,XShmSegmentInfo* f,unsigned int g,unsigned int h))
SDL_X11_SYM(Pixmap,XShmCreatePixmap,(Display *a,Drawable b,char* c,XShmSegmentInfo* d, unsigned int e, unsigned int f, unsigned int g))
SDL_X11_SYM(Bool,XShmQueryExtension,(Display* a))
#endif

/*
 * Not required...these only exist in code in headers on some 64-bit platforms,
 *  and are removed via macros elsewhere, so it's safe for them to be missing.
 */
#ifdef LONG64
SDL_X11_MODULE(IO_32BIT)
SDL_X11_SYM(int,_XData32,(Display *dpy,register _Xconst long *data,unsigned len))
SDL_X11_SYM(void,_XRead32,(Display *dpy,register long *data,long len))
#endif

/*
 * These only show up on some variants of Unix.
 */
#ifdef SDL_PLATFORM_OSF
SDL_X11_MODULE(OSF_ENTRY_POINTS)
SDL_X11_SYM(void,_SmtBufferOverflow,(Display *dpy,register smtDisplayPtr p))
SDL_X11_SYM(void,_SmtIpError,(Display *dpy,register smtDisplayPtr p,int i))
SDL_X11_SYM(int,ipAllocateData,(ChannelPtr a,IPCard b,IPDataPtr * c))
SDL_X11_SYM(int,ipUnallocateAndSendData,(ChannelPtr a,IPCard b))
#endif

// XCursor support
#ifdef SDL_VIDEO_DRIVER_X11_XCURSOR
SDL_X11_MODULE(XCURSOR)
SDL_X11_SYM(XcursorImage*,XcursorImageCreate,(int a,int b))
SDL_X11_SYM(void,XcursorImageDestroy,(XcursorImage *a))
SDL_X11_SYM(Cursor,XcursorImageLoadCursor,(Display *a,const XcursorImage *b))
SDL_X11_SYM(Cursor,XcursorLibraryLoadCursor,(Display *a, const char *b))
#endif

// Xdbe support
#ifdef SDL_VIDEO_DRIVER_X11_XDBE
SDL_X11_MODULE(XDBE)
SDL_X11_SYM(Status,XdbeQueryExtension,(Display *dpy,int *major_version_return,int *minor_version_return))
SDL_X11_SYM(XdbeBackBuffer,XdbeAllocateBackBufferName,(Display *dpy,Window window,XdbeSwapAction swap_action))
SDL_X11_SYM(Status,XdbeDeallocateBackBufferName,(Display *dpy,XdbeBackBuffer buffer))
SDL_X11_SYM(Status,XdbeSwapBuffers,(Display *dpy,XdbeSwapInfo *swap_info,int num_windows))
SDL_X11_SYM(Status,XdbeBeginIdiom,(Display *dpy))
SDL_X11_SYM(Status,XdbeEndIdiom,(Display *dpy))
SDL_X11_SYM(XdbeScreenVisualInfo*,XdbeGetVisualInfo,(Display *dpy,Drawable *screen_specifiers,int *num_screens))
SDL_X11_SYM(void,XdbeFreeVisualInfo,(XdbeScreenVisualInfo *visual_info))
SDL_X11_SYM(XdbeBackBufferAttributes*,XdbeGetBackBufferAttributes,(Display *dpy,XdbeBackBuffer buffer))
#endif

// XInput2 support for multiple mice, tablets, etc.
#ifdef SDL_VIDEO_DRIVER_X11_XINPUT2
SDL_X11_MODULE(XINPUT2)
SDL_X11_SYM(XIDeviceInfo*,XIQueryDevice,(Display *a,int b,int *c))
SDL_X11_SYM(void,XIFreeDeviceInfo,(XIDeviceInfo *a))
SDL_X11_SYM(int,XISelectEvents,(Display *a,Window b,XIEventMask *c,int d))
SDL_X11_SYM(int,XIGrabTouchBegin,(Display *a,int b,Window c,int d,XIEventMask *e,int f,XIGrabModifiers *g))
SDL_X11_SYM(int,XIUngrabTouchBegin, (Display *a,int b,Window c, int d,XIGrabModifiers *e))
SDL_X11_SYM(Status,XIQueryVersion,(Display *a,int *b,int *c))
SDL_X11_SYM(XIEventMask*,XIGetSelectedEvents,(Display *a,Window b,int *c))
SDL_X11_SYM(Bool,XIGetClientPointer,(Display *a,Window b,int *c))
SDL_X11_SYM(Bool,XIWarpPointer,(Display *a,int b,Window c,Window d,double e,double f,int g,int h,double i,double j))
SDL_X11_SYM(Status,XIGetProperty,(Display *a,int b,Atom c,long d,long e,Bool f, Atom g, Atom *h, int *i, unsigned long *j, unsigned long *k, unsigned char **l))
#endif

// XRandR support
#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
SDL_X11_MODULE(XRANDR)
SDL_X11_SYM(Status,XRRQueryVersion,(Display *dpy,int *major_versionp,int *minor_versionp))
SDL_X11_SYM(Bool,XRRQueryExtension,(Display *dpy,int *event_base_return,int *error_base_return))
SDL_X11_SYM(XRRScreenConfiguration *,XRRGetScreenInfo,(Display *dpy,Drawable draw))
SDL_X11_SYM(SizeID,XRRConfigCurrentConfiguration,(XRRScreenConfiguration *config,Rotation *rotation))
SDL_X11_SYM(short,XRRConfigCurrentRate,(XRRScreenConfiguration *config))
SDL_X11_SYM(short *,XRRConfigRates,(XRRScreenConfiguration *config,int sizeID,int *nrates))
SDL_X11_SYM(XRRScreenSize *,XRRConfigSizes,(XRRScreenConfiguration *config,int *nsizes))
SDL_X11_SYM(Status,XRRSetScreenConfigAndRate,(Display *dpy,XRRScreenConfiguration *config,Drawable draw,int size_index,Rotation rotation,short rate,Time timestamp))
SDL_X11_SYM(void,XRRFreeScreenConfigInfo,(XRRScreenConfiguration *config))
SDL_X11_SYM(void,XRRSetScreenSize,(Display *dpy, Window window,int width, int height,int mmWidth, int mmHeight))
SDL_X11_SYM(Status,XRRGetScreenSizeRange,(Display *dpy, Window window,int *minWidth, int *minHeight, int *maxWidth, int *maxHeight))
SDL_X11_SYM(XRRScreenResources *,XRRGetScreenResources,(Display *dpy, Window window))
SDL_X11_SYM(XRRScreenResources *,XRRGetScreenResourcesCurrent,(Display *dpy, Window window))
SDL_X11_SYM(void,XRRFreeScreenResources,(XRRScreenResources *resources))
SDL_X11_SYM(XRROutputInfo *,XRRGetOutputInfo,(Display *dpy, XRRScreenResources *resources, RROutput output))
SDL_X11_SYM(void,XRRFreeOutputInfo,(XRROutputInfo *outputInfo))
SDL_X11_SYM(XRRCrtcInfo *,XRRGetCrtcInfo,(Display *dpy, XRRScreenResources *resources, RRCrtc crtc))
SDL_X11_SYM(void,XRRFreeCrtcInfo,(XRRCrtcInfo *crtcInfo))
SDL_X11_SYM(Status,XRRSetCrtcConfig,(Display *dpy, XRRScreenResources *resources, RRCrtc crtc, Time timestamp, int x, int y, RRMode mode, Rotation rotation, RROutput *outputs, int noutputs))
SDL_X11_SYM(Atom*,XRRListOutputProperties,(Display *dpy, RROutput output, int *nprop))
SDL_X11_SYM(XRRPropertyInfo*,XRRQueryOutputProperty,(Display *dpy,RROutput output, Atom property))
SDL_X11_SYM(int,XRRGetOutputProperty,(Display *dpy,RROutput output, Atom property, long offset, long length, Bool _delete, Bool pending, Atom req_type, Atom *actual_type, int *actual_format, unsigned long *nitems, unsigned long *bytes_after, unsigned char **prop))
SDL_X11_SYM(RROutput,XRRGetOutputPrimary,(Display *dpy,Window window))
SDL_X11_SYM(void,XRRSelectInput,(Display *dpy, Window window, int mask))
SDL_X11_SYM(Status,XRRGetCrtcTransform,(Display *dpy,RRCrtc crtc,XRRCrtcTransformAttributes **attributes))
#endif

// MIT-SCREEN-SAVER support
#ifdef SDL_VIDEO_DRIVER_X11_XSCRNSAVER
SDL_X11_MODULE(XSS)
SDL_X11_SYM(Bool,XScreenSaverQueryExtension,(Display *dpy,int *event_base,int *error_base))
SDL_X11_SYM(Status,XScreenSaverQueryVersion,(Display *dpy,int *major_versionp,int *minor_versionp))
SDL_X11_SYM(void,XScreenSaverSuspend,(Display *dpy,Bool suspend))
#endif

#ifdef SDL_VIDEO_DRIVER_X11_XSHAPE
SDL_X11_MODULE(XSHAPE)
SDL_X11_SYM(void,XShapeCombineMask,(Display *dpy,Window dest,int dest_kind,int x_off,int y_off,Pixmap src,int op))
SDL_X11_SYM(void,XShapeCombineRegion,(Display *a,Window b,int c,int d,int e,Region f,int g))
#endif

#undef SDL_X11_MODULE
#undef SDL_X11_SYM

/* *INDENT-ON* */ // clang-format on
