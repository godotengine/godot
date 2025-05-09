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

#ifndef SDL_cocoawindow_h_
#define SDL_cocoawindow_h_

#import <Cocoa/Cocoa.h>

#ifdef SDL_VIDEO_OPENGL_EGL
#include "../SDL_egl_c.h"
#endif

#define SDL_METALVIEW_TAG 255

@class SDL_CocoaWindowData;

typedef enum
{
    PENDING_OPERATION_NONE = 0x00,
    PENDING_OPERATION_ENTER_FULLSCREEN = 0x01,
    PENDING_OPERATION_LEAVE_FULLSCREEN = 0x02,
    PENDING_OPERATION_MINIMIZE = 0x04,
    PENDING_OPERATION_ZOOM = 0x08
} PendingWindowOperation;

@interface SDL3Cocoa_WindowListener : NSResponder <NSWindowDelegate>
{
    /* SDL_CocoaWindowData owns this Listener and has a strong reference to it.
     * To avoid reference cycles, we could have either a weak or an
     * unretained ref to the WindowData. */
    __weak SDL_CocoaWindowData *_data;
    BOOL observingVisible;
    BOOL wasCtrlLeft;
    BOOL wasVisible;
    BOOL isFullscreenSpace;
    BOOL inFullscreenTransition;
    PendingWindowOperation pendingWindowOperation;
    BOOL isMoving;
    BOOL isMiniaturizing;
    NSInteger focusClickPending;
    float pendingWindowWarpX, pendingWindowWarpY;
    BOOL isDragAreaRunning;
    NSTimer *liveResizeTimer;
}

- (BOOL)isTouchFromTrackpad:(NSEvent *)theEvent;
- (void)listen:(SDL_CocoaWindowData *)data;
- (void)pauseVisibleObservation;
- (void)resumeVisibleObservation;
- (BOOL)setFullscreenSpace:(BOOL)state;
- (BOOL)isInFullscreenSpace;
- (BOOL)isInFullscreenSpaceTransition;
- (void)addPendingWindowOperation:(PendingWindowOperation)operation;
- (void)close;

- (BOOL)isMoving;
- (BOOL)isMovingOrFocusClickPending;
- (void)setFocusClickPending:(NSInteger)button;
- (void)clearFocusClickPending:(NSInteger)button;
- (void)updateIgnoreMouseState:(NSEvent *)theEvent;
- (void)setPendingMoveX:(float)x Y:(float)y;
- (void)windowDidFinishMoving;
- (void)onMovingOrFocusClickPendingStateCleared;

// Window delegate functionality
- (BOOL)windowShouldClose:(id)sender;
- (void)windowDidExpose:(NSNotification *)aNotification;
- (void)windowDidChangeOcclusionState:(NSNotification *)aNotification;
- (void)windowWillStartLiveResize:(NSNotification *)aNotification;
- (void)windowDidEndLiveResize:(NSNotification *)aNotification;
- (void)windowDidMove:(NSNotification *)aNotification;
- (void)windowDidResize:(NSNotification *)aNotification;
- (void)windowDidMiniaturize:(NSNotification *)aNotification;
- (void)windowDidDeminiaturize:(NSNotification *)aNotification;
- (void)windowDidBecomeKey:(NSNotification *)aNotification;
- (void)windowDidResignKey:(NSNotification *)aNotification;
- (void)windowDidChangeBackingProperties:(NSNotification *)aNotification;
- (void)windowDidChangeScreenProfile:(NSNotification *)aNotification;
- (void)windowDidChangeScreen:(NSNotification *)aNotification;
- (void)windowWillEnterFullScreen:(NSNotification *)aNotification;
- (void)windowDidEnterFullScreen:(NSNotification *)aNotification;
- (void)windowWillExitFullScreen:(NSNotification *)aNotification;
- (void)windowDidExitFullScreen:(NSNotification *)aNotification;
- (NSApplicationPresentationOptions)window:(NSWindow *)window willUseFullScreenPresentationOptions:(NSApplicationPresentationOptions)proposedOptions;

// See if event is in a drag area, toggle on window dragging.
- (void)updateHitTest;
- (BOOL)processHitTest:(NSEvent *)theEvent;

// Window event handling
- (void)mouseDown:(NSEvent *)theEvent;
- (void)rightMouseDown:(NSEvent *)theEvent;
- (void)otherMouseDown:(NSEvent *)theEvent;
- (void)mouseUp:(NSEvent *)theEvent;
- (void)rightMouseUp:(NSEvent *)theEvent;
- (void)otherMouseUp:(NSEvent *)theEvent;
- (void)mouseMoved:(NSEvent *)theEvent;
- (void)mouseDragged:(NSEvent *)theEvent;
- (void)rightMouseDragged:(NSEvent *)theEvent;
- (void)otherMouseDragged:(NSEvent *)theEvent;
- (void)scrollWheel:(NSEvent *)theEvent;
- (void)touchesBeganWithEvent:(NSEvent *)theEvent;
- (void)touchesMovedWithEvent:(NSEvent *)theEvent;
- (void)touchesEndedWithEvent:(NSEvent *)theEvent;
- (void)touchesCancelledWithEvent:(NSEvent *)theEvent;

// Touch event handling
- (void)handleTouches:(NSTouchPhase)phase withEvent:(NSEvent *)theEvent;

// Tablet event handling (but these also come through on mouse events sometimes!)
- (void)tabletProximity:(NSEvent *)theEvent;
- (void)tabletPoint:(NSEvent *)theEvent;

@end
/* *INDENT-ON* */

@class SDL3OpenGLContext;
@class SDL_CocoaVideoData;

@interface SDL_CocoaWindowData : NSObject
@property(nonatomic) SDL_Window *window;
@property(nonatomic) NSWindow *nswindow;
@property(nonatomic) NSView *sdlContentView;
@property(nonatomic) NSMutableArray *nscontexts;
@property(nonatomic) BOOL in_blocking_transition;
@property(nonatomic) BOOL fullscreen_space_requested;
@property(nonatomic) BOOL was_zoomed;
@property(nonatomic) NSInteger window_number;
@property(nonatomic) NSInteger flash_request;
@property(nonatomic) SDL3Cocoa_WindowListener *listener;
@property(nonatomic) NSModalSession modal_session;
@property(nonatomic) SDL_CocoaVideoData *videodata;
@property(nonatomic) bool pending_size;
@property(nonatomic) bool pending_position;
@property(nonatomic) bool border_toggled;

#ifdef SDL_VIDEO_OPENGL_EGL
@property(nonatomic) EGLSurface egl_surface;
#endif
@end

extern bool b_inModeTransition;

extern bool Cocoa_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props);
extern void Cocoa_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window);
extern bool Cocoa_SetWindowIcon(SDL_VideoDevice *_this, SDL_Window *window, SDL_Surface *icon);
extern bool Cocoa_SetWindowPosition(SDL_VideoDevice *_this, SDL_Window *window);
extern void Cocoa_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window);
extern void Cocoa_SetWindowMinimumSize(SDL_VideoDevice *_this, SDL_Window *window);
extern void Cocoa_SetWindowMaximumSize(SDL_VideoDevice *_this, SDL_Window *window);
extern void Cocoa_SetWindowAspectRatio(SDL_VideoDevice *_this, SDL_Window *window);
extern void Cocoa_GetWindowSizeInPixels(SDL_VideoDevice *_this, SDL_Window *window, int *w, int *h);
extern bool Cocoa_SetWindowOpacity(SDL_VideoDevice *_this, SDL_Window *window, float opacity);
extern void Cocoa_ShowWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void Cocoa_HideWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void Cocoa_RaiseWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void Cocoa_MaximizeWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void Cocoa_MinimizeWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void Cocoa_RestoreWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void Cocoa_SetWindowBordered(SDL_VideoDevice *_this, SDL_Window *window, bool bordered);
extern void Cocoa_SetWindowResizable(SDL_VideoDevice *_this, SDL_Window *window, bool resizable);
extern void Cocoa_SetWindowAlwaysOnTop(SDL_VideoDevice *_this, SDL_Window *window, bool on_top);
extern SDL_FullscreenResult Cocoa_SetWindowFullscreen(SDL_VideoDevice *_this, SDL_Window *window, SDL_VideoDisplay *display, SDL_FullscreenOp fullscreen);
extern void *Cocoa_GetWindowICCProfile(SDL_VideoDevice *_this, SDL_Window *window, size_t *size);
extern SDL_DisplayID Cocoa_GetDisplayForWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern bool Cocoa_SetWindowMouseRect(SDL_VideoDevice *_this, SDL_Window *window);
extern bool Cocoa_SetWindowMouseGrab(SDL_VideoDevice *_this, SDL_Window *window, bool grabbed);
extern void Cocoa_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern bool Cocoa_SetWindowHitTest(SDL_Window *window, bool enabled);
extern void Cocoa_AcceptDragAndDrop(SDL_Window *window, bool accept);
extern bool Cocoa_FlashWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_FlashOperation operation);
extern bool Cocoa_SetWindowFocusable(SDL_VideoDevice *_this, SDL_Window *window, bool focusable);
extern bool Cocoa_SetWindowModal(SDL_VideoDevice *_this, SDL_Window *window, bool modal);
extern bool Cocoa_SetWindowParent(SDL_VideoDevice *_this, SDL_Window *window, SDL_Window *parent);
extern bool Cocoa_SyncWindow(SDL_VideoDevice *_this, SDL_Window *window);

extern void Cocoa_MenuVisibilityCallback(void *userdata, const char *name, const char *oldValue, const char *newValue);

#endif // SDL_cocoawindow_h_
