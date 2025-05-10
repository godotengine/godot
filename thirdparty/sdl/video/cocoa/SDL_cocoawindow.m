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

#ifdef SDL_VIDEO_DRIVER_COCOA

#include <float.h> // For FLT_MAX

#include "../../events/SDL_dropevents_c.h"
#include "../../events/SDL_keyboard_c.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_touch_c.h"
#include "../../events/SDL_windowevents_c.h"
#include "../SDL_sysvideo.h"

#include "SDL_cocoamouse.h"
#include "SDL_cocoaopengl.h"
#include "SDL_cocoaopengles.h"
#include "SDL_cocoavideo.h"

#if 0
#define DEBUG_COCOAWINDOW
#endif

#ifdef DEBUG_COCOAWINDOW
#define DLog(fmt, ...) printf("%s: " fmt "\n", __func__, ##__VA_ARGS__)
#else
#define DLog(...) \
    do {          \
    } while (0)
#endif

#ifndef MAC_OS_X_VERSION_10_12
#define NSEventModifierFlagCapsLock NSAlphaShiftKeyMask
#endif
#ifndef NSAppKitVersionNumber10_13_2
#define NSAppKitVersionNumber10_13_2 1561.2
#endif
#ifndef NSAppKitVersionNumber10_14
#define NSAppKitVersionNumber10_14 1671
#endif

@implementation SDL_CocoaWindowData

@end

@interface NSScreen (SDL)
#if MAC_OS_X_VERSION_MAX_ALLOWED < 120000 // Added in the 12.0 SDK
@property(readonly) NSEdgeInsets safeAreaInsets;
#endif
@end

@interface NSWindow (SDL)
// This is available as of 10.13.2, but isn't in public headers
@property(nonatomic) NSRect mouseConfinementRect;
@end

@interface SDL3Window : NSWindow <NSDraggingDestination>
// These are needed for borderless/fullscreen windows
- (BOOL)canBecomeKeyWindow;
- (BOOL)canBecomeMainWindow;
- (void)sendEvent:(NSEvent *)event;
- (void)doCommandBySelector:(SEL)aSelector;

// Handle drag-and-drop of files onto the SDL window.
- (NSDragOperation)draggingEntered:(id<NSDraggingInfo>)sender;
- (void)draggingExited:(id<NSDraggingInfo>)sender;
- (NSDragOperation)draggingUpdated:(id<NSDraggingInfo>)sender;
- (BOOL)performDragOperation:(id<NSDraggingInfo>)sender;
- (BOOL)wantsPeriodicDraggingUpdates;
- (BOOL)validateMenuItem:(NSMenuItem *)menuItem;

- (SDL_Window *)findSDLWindow;
@end

@implementation SDL3Window

- (BOOL)validateMenuItem:(NSMenuItem *)menuItem
{
    /* Only allow using the macOS native fullscreen toggle menubar item if the
     * window is resizable and not in a SDL fullscreen mode.
     */
    if ([menuItem action] == @selector(toggleFullScreen:)) {
        SDL_Window *window = [self findSDLWindow];
        if (!window) {
            return NO;
        }

        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
        if ((window->flags & SDL_WINDOW_FULLSCREEN) && ![data.listener isInFullscreenSpace]) {
            return NO;
        } else if (!(window->flags & SDL_WINDOW_RESIZABLE)) {
            return NO;
        }
    }
    return [super validateMenuItem:menuItem];
}

- (BOOL)canBecomeKeyWindow
{
    SDL_Window *window = [self findSDLWindow];
    if (window && !(window->flags & (SDL_WINDOW_TOOLTIP | SDL_WINDOW_NOT_FOCUSABLE))) {
        return YES;
    } else {
        return NO;
    }
}

- (BOOL)canBecomeMainWindow
{
    SDL_Window *window = [self findSDLWindow];
    if (window && !(window->flags & (SDL_WINDOW_TOOLTIP | SDL_WINDOW_NOT_FOCUSABLE)) && !SDL_WINDOW_IS_POPUP(window)) {
        return YES;
    } else {
        return NO;
    }
}

- (void)sendEvent:(NSEvent *)event
{
    id delegate;
    [super sendEvent:event];

    if ([event type] != NSEventTypeLeftMouseUp) {
        return;
    }

    delegate = [self delegate];
    if (![delegate isKindOfClass:[SDL3Cocoa_WindowListener class]]) {
        return;
    }

    if ([delegate isMoving]) {
        [delegate windowDidFinishMoving];
    }
}

/* We'll respond to selectors by doing nothing so we don't beep.
 * The escape key gets converted to a "cancel" selector, etc.
 */
- (void)doCommandBySelector:(SEL)aSelector
{
    // NSLog(@"doCommandBySelector: %@\n", NSStringFromSelector(aSelector));
}

- (NSDragOperation)draggingEntered:(id<NSDraggingInfo>)sender
{
    if (([sender draggingSourceOperationMask] & NSDragOperationGeneric) == NSDragOperationGeneric) {
        return NSDragOperationGeneric;
    } else if (([sender draggingSourceOperationMask] & NSDragOperationCopy) == NSDragOperationCopy) {
        return NSDragOperationCopy;
    }

    return NSDragOperationNone; // no idea what to do with this, reject it.
}

- (void)draggingExited:(id<NSDraggingInfo>)sender
{
    SDL_Window *sdlwindow = [self findSDLWindow];
    SDL_SendDropComplete(sdlwindow);
}

- (NSDragOperation)draggingUpdated:(id<NSDraggingInfo>)sender
{
    if (([sender draggingSourceOperationMask] & NSDragOperationGeneric) == NSDragOperationGeneric) {
        SDL_Window *sdlwindow = [self findSDLWindow];
        NSPoint point = [sender draggingLocation];
        float x, y;
        x = point.x;
        y = (sdlwindow->h - point.y);
        SDL_SendDropPosition(sdlwindow, x, y);
        return NSDragOperationGeneric;
    } else if (([sender draggingSourceOperationMask] & NSDragOperationCopy) == NSDragOperationCopy) {
        SDL_Window *sdlwindow = [self findSDLWindow];
        NSPoint point = [sender draggingLocation];
        float x, y;
        x = point.x;
        y = (sdlwindow->h - point.y);
        SDL_SendDropPosition(sdlwindow, x, y);
        return NSDragOperationCopy;
    }

    return NSDragOperationNone; // no idea what to do with this, reject it.
}

- (BOOL)performDragOperation:(id<NSDraggingInfo>)sender
{
    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                 ". [SDL] In performDragOperation, draggingSourceOperationMask %lx, "
                 "expected Generic %lx, others Copy %lx, Link %lx, Private %lx, Move %lx, Delete %lx\n",
                 (unsigned long)[sender draggingSourceOperationMask],
                 (unsigned long)NSDragOperationGeneric,
                 (unsigned long)NSDragOperationCopy,
                 (unsigned long)NSDragOperationLink,
                 (unsigned long)NSDragOperationPrivate,
                 (unsigned long)NSDragOperationMove,
                 (unsigned long)NSDragOperationDelete);
    if ([sender draggingPasteboard]) {
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". [SDL] In performDragOperation, valid draggingPasteboard, "
                     "name [%s] '%s', changeCount %ld\n",
                     [[[[sender draggingPasteboard] name] className] UTF8String],
                     [[[[sender draggingPasteboard] name] description] UTF8String],
                     (long)[[sender draggingPasteboard] changeCount]);
    }
    @autoreleasepool {
        NSPasteboard *pasteboard = [sender draggingPasteboard];
        NSString *desiredType = [pasteboard availableTypeFromArray:@[ NSFilenamesPboardType, NSPasteboardTypeString ]];
        SDL_Window *sdlwindow = [self findSDLWindow];
        NSData *pboardData;
        id pboardPlist;
        NSString *pboardString;
        NSPoint point;
        float x, y;

        for (NSString *supportedType in [pasteboard types]) {
            NSString *typeString = [pasteboard stringForType:supportedType];
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". [SDL] In performDragOperation, Pasteboard type '%s', stringForType (%lu) '%s'\n",
                         [[supportedType description] UTF8String],
                         (unsigned long)[[typeString description] length],
                         [[typeString description] UTF8String]);
        }

        if (desiredType == nil) {
            return NO; // can't accept anything that's being dropped here.
        }
        pboardData = [pasteboard dataForType:desiredType];
        if (pboardData == nil) {
            return NO;
        }
        SDL_assert([desiredType isEqualToString:NSFilenamesPboardType] ||
                   [desiredType isEqualToString:NSPasteboardTypeString]);

        pboardString = [pasteboard stringForType:desiredType];
        pboardPlist = [pasteboard propertyListForType:desiredType];

        // Use SendDropPosition to update the mouse location
        point = [sender draggingLocation];
        x = point.x;
        y = (sdlwindow->h - point.y);
        if (x >= 0.0f && x < (float)sdlwindow->w && y >= 0.0f && y < (float)sdlwindow->h) {
            SDL_SendDropPosition(sdlwindow, x, y);
        }
        // Use SendDropPosition to update the mouse location

        if ([desiredType isEqualToString:NSFilenamesPboardType]) {
            for (NSString *path in (NSArray *)pboardPlist) {
                NSURL *fileURL = [NSURL fileURLWithPath:path];
                NSNumber *isAlias = nil;

                [fileURL getResourceValue:&isAlias forKey:NSURLIsAliasFileKey error:nil];

                // If the URL is an alias, resolve it.
                if ([isAlias boolValue]) {
                    NSURLBookmarkResolutionOptions opts = NSURLBookmarkResolutionWithoutMounting |
                                                          NSURLBookmarkResolutionWithoutUI;
                    NSData *bookmark = [NSURL bookmarkDataWithContentsOfURL:fileURL error:nil];
                    if (bookmark != nil) {
                        NSURL *resolvedURL = [NSURL URLByResolvingBookmarkData:bookmark
                                                                       options:opts
                                                                 relativeToURL:nil
                                                           bookmarkDataIsStale:nil
                                                                         error:nil];
                        if (resolvedURL != nil) {
                            fileURL = resolvedURL;
                        }
                    }
                }
                SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                             ". [SDL] In performDragOperation, desiredType '%s', "
                             "Submitting DropFile as (%lu) '%s'\n",
                             [[desiredType description] UTF8String],
                             (unsigned long)[[[fileURL path] description] length],
                             [[[fileURL path] description] UTF8String]);
                if (!SDL_SendDropFile(sdlwindow, NULL, [[[fileURL path] description] UTF8String])) {
                    return NO;
                }
            }
        } else if ([desiredType isEqualToString:NSPasteboardTypeString]) {
            char *buffer  = SDL_strdup([[pboardString description] UTF8String]);
            char *saveptr = NULL;
            char *token   = SDL_strtok_r(buffer, "\r\n", &saveptr);
            while (token) {
                SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                             ". [SDL] In performDragOperation, desiredType '%s', "
                             "Submitting DropText as (%lu) '%s'\n",
                             [[desiredType description] UTF8String],
                             SDL_strlen(token), token);
                if (!SDL_SendDropText(sdlwindow, token)) {
                    SDL_free(buffer);
                    return NO;
                }
                token = SDL_strtok_r(NULL, "\r\n", &saveptr);
            }
            SDL_free(buffer);
        }

        SDL_SendDropComplete(sdlwindow);
        return YES;
    }
}

- (BOOL)wantsPeriodicDraggingUpdates
{
    return NO;
}

- (SDL_Window *)findSDLWindow
{
    SDL_Window *sdlwindow = NULL;
    SDL_VideoDevice *_this = SDL_GetVideoDevice();

    // !!! FIXME: is there a better way to do this?
    if (_this) {
        for (sdlwindow = _this->windows; sdlwindow; sdlwindow = sdlwindow->next) {
            NSWindow *nswindow = ((__bridge SDL_CocoaWindowData *)sdlwindow->internal).nswindow;
            if (nswindow == self) {
                break;
            }
        }
    }

    return sdlwindow;
}

@end

bool b_inModeTransition;

static CGFloat SqDistanceToRect(const NSPoint *point, const NSRect *rect)
{
    NSPoint edge = *point;
    CGFloat left = NSMinX(*rect), right = NSMaxX(*rect);
    CGFloat bottom = NSMinX(*rect), top = NSMaxY(*rect);
    NSPoint delta;

    if (point->x < left) {
        edge.x = left;
    } else if (point->x > right) {
        edge.x = right;
    }

    if (point->y < bottom) {
        edge.y = bottom;
    } else if (point->y > top) {
        edge.y = top;
    }

    delta = NSMakePoint(edge.x - point->x, edge.y - point->y);
    return delta.x * delta.x + delta.y * delta.y;
}

static NSScreen *ScreenForPoint(const NSPoint *point)
{
    NSScreen *screen;

    // Do a quick check first to see if the point lies on a specific screen
    for (NSScreen *candidate in [NSScreen screens]) {
        if (NSPointInRect(*point, [candidate frame])) {
            screen = candidate;
            break;
        }
    }

    // Find the screen the point is closest to
    if (!screen) {
        CGFloat closest = MAXFLOAT;
        for (NSScreen *candidate in [NSScreen screens]) {
            NSRect screenRect = [candidate frame];

            CGFloat sqdist = SqDistanceToRect(point, &screenRect);
            if (sqdist < closest) {
                screen = candidate;
                closest = sqdist;
            }
        }
    }

    return screen;
}

bool Cocoa_IsWindowInFullscreenSpace(SDL_Window *window)
{
    @autoreleasepool {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;

        if ([data.listener isInFullscreenSpace]) {
            return true;
        } else {
            return false;
        }
    }
}

bool Cocoa_IsWindowZoomed(SDL_Window *window)
{
    SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
    NSWindow *nswindow = data.nswindow;
    bool zoomed = false;

    // isZoomed always returns true if the window is not resizable or the window is fullscreen
    if ((window->flags & SDL_WINDOW_RESIZABLE) && [nswindow isZoomed] &&
        !(window->flags & SDL_WINDOW_FULLSCREEN) && !Cocoa_IsWindowInFullscreenSpace(window)) {
        // If we are at our desired floating area, then we're not zoomed
        bool floating = (window->x == window->floating.x &&
                         window->y == window->floating.y &&
                         window->w == window->floating.w &&
                         window->h == window->floating.h);
        if (!floating) {
            zoomed = true;
        }
    }
    return zoomed;
}

typedef enum CocoaMenuVisibility
{
    COCOA_MENU_VISIBILITY_AUTO = 0,
    COCOA_MENU_VISIBILITY_NEVER,
    COCOA_MENU_VISIBILITY_ALWAYS
} CocoaMenuVisibility;

static CocoaMenuVisibility menu_visibility_hint = COCOA_MENU_VISIBILITY_AUTO;

static void Cocoa_ToggleFullscreenSpaceMenuVisibility(SDL_Window *window)
{
    if (window && Cocoa_IsWindowInFullscreenSpace(window)) {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;

        // 'Auto' sets the menu to visible if fullscreen wasn't explicitly entered via SDL_SetWindowFullscreen().
        if ((menu_visibility_hint == COCOA_MENU_VISIBILITY_AUTO && !data.fullscreen_space_requested) ||
            menu_visibility_hint == COCOA_MENU_VISIBILITY_ALWAYS) {
            [NSMenu setMenuBarVisible:YES];
        } else {
            [NSMenu setMenuBarVisible:NO];
        }
    }
}

void Cocoa_MenuVisibilityCallback(void *userdata, const char *name, const char *oldValue, const char *newValue)
{
    if (newValue) {
        if (*newValue == '0' || SDL_strcasecmp(newValue, "false") == 0) {
            menu_visibility_hint = COCOA_MENU_VISIBILITY_NEVER;
        } else if (*newValue == '1' || SDL_strcasecmp(newValue, "true") == 0) {
            menu_visibility_hint = COCOA_MENU_VISIBILITY_ALWAYS;
        } else {
            menu_visibility_hint = COCOA_MENU_VISIBILITY_AUTO;
        }
    } else {
        menu_visibility_hint = COCOA_MENU_VISIBILITY_AUTO;
    }

    // Update the current menu visibility.
    Cocoa_ToggleFullscreenSpaceMenuVisibility(SDL_GetKeyboardFocus());
}

static NSScreen *ScreenForRect(const NSRect *rect)
{
    NSPoint center = NSMakePoint(NSMidX(*rect), NSMidY(*rect));
    return ScreenForPoint(&center);
}

static void ConvertNSRect(NSRect *r)
{
    r->origin.y = CGDisplayPixelsHigh(kCGDirectMainDisplay) - r->origin.y - r->size.height;
}

static void ScheduleContextUpdates(SDL_CocoaWindowData *data)
{
// We still support OpenGL as long as Apple offers it, deprecated or not, so disable deprecation warnings about it.
#ifdef SDL_VIDEO_OPENGL

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

    NSOpenGLContext *currentContext;
    NSMutableArray *contexts;
    if (!data || !data.nscontexts) {
        return;
    }

    currentContext = [NSOpenGLContext currentContext];
    contexts = data.nscontexts;
    @synchronized(contexts) {
        for (SDL3OpenGLContext *context in contexts) {
            if (context == currentContext) {
                [context update];
            } else {
                [context scheduleUpdate];
            }
        }
    }

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif // SDL_VIDEO_OPENGL
}

// !!! FIXME: this should use a hint callback.
static bool GetHintCtrlClickEmulateRightClick(void)
{
    return SDL_GetHintBoolean(SDL_HINT_MAC_CTRL_CLICK_EMULATE_RIGHT_CLICK, false);
}

static NSUInteger GetWindowWindowedStyle(SDL_Window *window)
{
    /* IF YOU CHANGE ANY FLAGS IN HERE, PLEASE READ
       the NSWindowStyleMaskBorderless comments in SetupWindowData()! */

    /* always allow miniaturization, otherwise you can't programmatically
       minimize the window, whether there's a title bar or not */
    NSUInteger style = NSWindowStyleMaskMiniaturizable;

    if (!SDL_WINDOW_IS_POPUP(window)) {
        if (window->flags & SDL_WINDOW_BORDERLESS) {
            style |= NSWindowStyleMaskBorderless;
        } else {
            style |= (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable);
        }
        if (window->flags & SDL_WINDOW_RESIZABLE) {
            style |= NSWindowStyleMaskResizable;
        }
    } else {
        style |= NSWindowStyleMaskBorderless;
    }
    return style;
}

static NSUInteger GetWindowStyle(SDL_Window *window)
{
    NSUInteger style = 0;

    if (window->flags & SDL_WINDOW_FULLSCREEN) {
        style = NSWindowStyleMaskBorderless;
    } else {
        style = GetWindowWindowedStyle(window);
    }
    return style;
}

static bool SetWindowStyle(SDL_Window *window, NSUInteger style)
{
    SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
    NSWindow *nswindow = data.nswindow;

    // The view responder chain gets messed with during setStyleMask
    if ([data.sdlContentView nextResponder] == data.listener) {
        [data.sdlContentView setNextResponder:nil];
    }

    [nswindow setStyleMask:style];

    // The view responder chain gets messed with during setStyleMask
    if ([data.sdlContentView nextResponder] != data.listener) {
        [data.sdlContentView setNextResponder:data.listener];
    }

    return true;
}

static bool ShouldAdjustCoordinatesForGrab(SDL_Window *window)
{
    SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;

    if (!data || [data.listener isMovingOrFocusClickPending]) {
        return false;
    }

    if (!(window->flags & SDL_WINDOW_INPUT_FOCUS)) {
        return false;
    }

    if ((window->flags & SDL_WINDOW_MOUSE_GRABBED) || (window->mouse_rect.w > 0 && window->mouse_rect.h > 0)) {
        return true;
    }
    return false;
}

static bool AdjustCoordinatesForGrab(SDL_Window *window, float x, float y, CGPoint *adjusted)
{
    if (window->mouse_rect.w > 0 && window->mouse_rect.h > 0) {
        SDL_Rect window_rect;
        SDL_Rect mouse_rect;

        window_rect.x = 0;
        window_rect.y = 0;
        window_rect.w = window->w;
        window_rect.h = window->h;

        if (SDL_GetRectIntersection(&window->mouse_rect, &window_rect, &mouse_rect)) {
            float left = (float)window->x + mouse_rect.x;
            float right = left + mouse_rect.w - 1;
            float top = (float)window->y + mouse_rect.y;
            float bottom = top + mouse_rect.h - 1;
            if (x < left || x > right || y < top || y > bottom) {
                adjusted->x = SDL_clamp(x, left, right);
                adjusted->y = SDL_clamp(y, top, bottom);
                return true;
            }
            return false;
        }
    }

    if (window->flags & SDL_WINDOW_MOUSE_GRABBED) {
        float left = (float)window->x;
        float right = left + window->w - 1;
        float top = (float)window->y;
        float bottom = top + window->h - 1;
        if (x < left || x > right || y < top || y > bottom) {
            adjusted->x = SDL_clamp(x, left, right);
            adjusted->y = SDL_clamp(y, top, bottom);
            return true;
        }
    }
    return false;
}

static void Cocoa_UpdateClipCursor(SDL_Window *window)
{
    SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;

    if (NSAppKitVersionNumber >= NSAppKitVersionNumber10_13_2) {
        NSWindow *nswindow = data.nswindow;
        SDL_Rect mouse_rect;

        SDL_zero(mouse_rect);

        if (ShouldAdjustCoordinatesForGrab(window)) {
            SDL_Rect window_rect;

            window_rect.x = 0;
            window_rect.y = 0;
            window_rect.w = window->w;
            window_rect.h = window->h;

            if (window->mouse_rect.w > 0 && window->mouse_rect.h > 0) {
                SDL_GetRectIntersection(&window->mouse_rect, &window_rect, &mouse_rect);
            }

            if ((window->flags & SDL_WINDOW_MOUSE_GRABBED) != 0 &&
                SDL_RectEmpty(&mouse_rect)) {
                SDL_memcpy(&mouse_rect, &window_rect, sizeof(mouse_rect));
            }
        }

        if (SDL_RectEmpty(&mouse_rect)) {
            nswindow.mouseConfinementRect = NSZeroRect;
        } else {
            NSRect rect;
            rect.origin.x = mouse_rect.x;
            rect.origin.y = [nswindow contentLayoutRect].size.height - mouse_rect.y - mouse_rect.h;
            rect.size.width = mouse_rect.w;
            rect.size.height = mouse_rect.h;
            nswindow.mouseConfinementRect = rect;
        }
    } else {
        // Move the cursor to the nearest point in the window
        if (ShouldAdjustCoordinatesForGrab(window)) {
            float x, y;
            CGPoint cgpoint;

            SDL_GetGlobalMouseState(&x, &y);
            if (AdjustCoordinatesForGrab(window, x, y, &cgpoint)) {
                Cocoa_HandleMouseWarp(cgpoint.x, cgpoint.y);
                CGDisplayMoveCursorToPoint(kCGDirectMainDisplay, cgpoint);
            }
        }
    }
}

static SDL_Window *GetParentToplevelWindow(SDL_Window *window)
{
    SDL_Window *toplevel = window;

    // Find the topmost parent
    while (SDL_WINDOW_IS_POPUP(toplevel)) {
        toplevel = toplevel->parent;
    }

    return toplevel;
}

static void Cocoa_SetKeyboardFocus(SDL_Window *window, bool set_active_focus)
{
    SDL_Window *toplevel = GetParentToplevelWindow(window);
    toplevel->keyboard_focus = window;

    if (set_active_focus && !window->is_hiding && !window->is_destroying) {
        SDL_SetKeyboardFocus(window);
    }
}

static void Cocoa_SendExposedEventIfVisible(SDL_Window *window)
{
    NSWindow *nswindow = ((__bridge SDL_CocoaWindowData *)window->internal).nswindow;
    if ([nswindow occlusionState] & NSWindowOcclusionStateVisible) {
        SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_EXPOSED, 0, 0);
    }
}

static void Cocoa_WaitForMiniaturizable(SDL_Window *window)
{
    NSWindow *nswindow = ((__bridge SDL_CocoaWindowData *)window->internal).nswindow;
    NSButton *button = [nswindow standardWindowButton:NSWindowMiniaturizeButton];
    if (button) {
        int iterations = 0;
        while (![button isEnabled] && (iterations < 100)) {
            SDL_Delay(10);
            SDL_PumpEvents();
            iterations++;
        }
    }
}

static NSCursor *Cocoa_GetDesiredCursor(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse->cursor_visible && mouse->cur_cursor && !mouse->relative_mode) {
        return (__bridge NSCursor *)mouse->cur_cursor->internal;
    }

    return [NSCursor invisibleCursor];
}

@implementation SDL3Cocoa_WindowListener

- (void)listen:(SDL_CocoaWindowData *)data
{
    NSNotificationCenter *center;
    NSWindow *window = data.nswindow;
    NSView *view = data.sdlContentView;

    _data = data;
    observingVisible = YES;
    wasCtrlLeft = NO;
    wasVisible = [window isVisible];
    isFullscreenSpace = NO;
    inFullscreenTransition = NO;
    pendingWindowOperation = PENDING_OPERATION_NONE;
    isMoving = NO;
    isMiniaturizing = NO;
    isDragAreaRunning = NO;
    pendingWindowWarpX = pendingWindowWarpY = FLT_MAX;
    liveResizeTimer = nil;

    center = [NSNotificationCenter defaultCenter];

    if ([window delegate] != nil) {
        [center addObserver:self selector:@selector(windowDidExpose:) name:NSWindowDidExposeNotification object:window];
        [center addObserver:self selector:@selector(windowDidChangeOcclusionState:) name:NSWindowDidChangeOcclusionStateNotification object:window];
        [center addObserver:self selector:@selector(windowWillStartLiveResize:) name:NSWindowWillStartLiveResizeNotification object:window];
        [center addObserver:self selector:@selector(windowDidEndLiveResize:) name:NSWindowDidEndLiveResizeNotification object:window];
        [center addObserver:self selector:@selector(windowWillMove:) name:NSWindowWillMoveNotification object:window];
        [center addObserver:self selector:@selector(windowDidMove:) name:NSWindowDidMoveNotification object:window];
        [center addObserver:self selector:@selector(windowDidResize:) name:NSWindowDidResizeNotification object:window];
        [center addObserver:self selector:@selector(windowWillMiniaturize:) name:NSWindowWillMiniaturizeNotification object:window];
        [center addObserver:self selector:@selector(windowDidMiniaturize:) name:NSWindowDidMiniaturizeNotification object:window];
        [center addObserver:self selector:@selector(windowDidDeminiaturize:) name:NSWindowDidDeminiaturizeNotification object:window];
        [center addObserver:self selector:@selector(windowDidBecomeKey:) name:NSWindowDidBecomeKeyNotification object:window];
        [center addObserver:self selector:@selector(windowDidResignKey:) name:NSWindowDidResignKeyNotification object:window];
        [center addObserver:self selector:@selector(windowDidChangeBackingProperties:) name:NSWindowDidChangeBackingPropertiesNotification object:window];
        [center addObserver:self selector:@selector(windowDidChangeScreenProfile:) name:NSWindowDidChangeScreenProfileNotification object:window];
        [center addObserver:self selector:@selector(windowDidChangeScreen:) name:NSWindowDidChangeScreenNotification object:window];
        [center addObserver:self selector:@selector(windowWillEnterFullScreen:) name:NSWindowWillEnterFullScreenNotification object:window];
        [center addObserver:self selector:@selector(windowDidEnterFullScreen:) name:NSWindowDidEnterFullScreenNotification object:window];
        [center addObserver:self selector:@selector(windowWillExitFullScreen:) name:NSWindowWillExitFullScreenNotification object:window];
        [center addObserver:self selector:@selector(windowDidExitFullScreen:) name:NSWindowDidExitFullScreenNotification object:window];
        [center addObserver:self selector:@selector(windowDidFailToEnterFullScreen:) name:@"NSWindowDidFailToEnterFullScreenNotification" object:window];
        [center addObserver:self selector:@selector(windowDidFailToExitFullScreen:) name:@"NSWindowDidFailToExitFullScreenNotification" object:window];
    } else {
        [window setDelegate:self];
    }

    /* Haven't found a delegate / notification that triggers when the window is
     * ordered out (is not visible any more). You can be ordered out without
     * minimizing, so DidMiniaturize doesn't work. (e.g. -[NSWindow orderOut:])
     */
    [window addObserver:self
             forKeyPath:@"visible"
                options:NSKeyValueObservingOptionNew
                context:NULL];

    [window setNextResponder:self];
    [window setAcceptsMouseMovedEvents:YES];

    [view setNextResponder:self];

    [view setAcceptsTouchEvents:YES];
}

- (void)observeValueForKeyPath:(NSString *)keyPath
                      ofObject:(id)object
                        change:(NSDictionary *)change
                       context:(void *)context
{
    if (!observingVisible) {
        return;
    }

    if (object == _data.nswindow && [keyPath isEqualToString:@"visible"]) {
        int newVisibility = [[change objectForKey:@"new"] intValue];
        if (newVisibility) {
            SDL_SendWindowEvent(_data.window, SDL_EVENT_WINDOW_SHOWN, 0, 0);
        } else if (![_data.nswindow isMiniaturized]) {
            SDL_SendWindowEvent(_data.window, SDL_EVENT_WINDOW_HIDDEN, 0, 0);
        }
    }
}

- (void)pauseVisibleObservation
{
    observingVisible = NO;
    wasVisible = [_data.nswindow isVisible];
}

- (void)resumeVisibleObservation
{
    BOOL isVisible = [_data.nswindow isVisible];
    observingVisible = YES;
    if (wasVisible != isVisible) {
        if (isVisible) {
            SDL_SendWindowEvent(_data.window, SDL_EVENT_WINDOW_SHOWN, 0, 0);
        } else if (![_data.nswindow isMiniaturized]) {
            SDL_SendWindowEvent(_data.window, SDL_EVENT_WINDOW_HIDDEN, 0, 0);
        }

        wasVisible = isVisible;
    }
}

- (BOOL)setFullscreenSpace:(BOOL)state
{
    SDL_Window *window = _data.window;
    NSWindow *nswindow = _data.nswindow;
    SDL_CocoaVideoData *videodata = ((__bridge SDL_CocoaWindowData *)window->internal).videodata;

    if (!videodata.allow_spaces) {
        return NO; // Spaces are forcibly disabled.
    } else if (state && window->fullscreen_exclusive) {
        return NO; // we only allow you to make a Space on fullscreen desktop windows.
    } else if (!state && window->last_fullscreen_exclusive_display) {
        return NO; // we only handle leaving the Space on windows that were previously fullscreen desktop.
    } else if (state == isFullscreenSpace && !inFullscreenTransition) {
        return YES; // already there.
    }

    if (inFullscreenTransition) {
        if (state) {
            [self clearPendingWindowOperation:PENDING_OPERATION_LEAVE_FULLSCREEN];
            [self addPendingWindowOperation:PENDING_OPERATION_ENTER_FULLSCREEN];
        } else {
            [self clearPendingWindowOperation:PENDING_OPERATION_ENTER_FULLSCREEN];
            [self addPendingWindowOperation:PENDING_OPERATION_LEAVE_FULLSCREEN];
        }
        return YES;
    }
    inFullscreenTransition = YES;

    // you need to be FullScreenPrimary, or toggleFullScreen doesn't work. Unset it again in windowDidExitFullScreen.
    [nswindow setCollectionBehavior:NSWindowCollectionBehaviorFullScreenPrimary];
    [nswindow performSelectorOnMainThread:@selector(toggleFullScreen:) withObject:nswindow waitUntilDone:NO];
    return YES;
}

- (BOOL)isInFullscreenSpace
{
    return isFullscreenSpace;
}

- (BOOL)isInFullscreenSpaceTransition
{
    return inFullscreenTransition;
}

- (void)clearPendingWindowOperation:(PendingWindowOperation)operation
{
    pendingWindowOperation &= ~operation;
}

- (void)clearAllPendingWindowOperations
{
    pendingWindowOperation = PENDING_OPERATION_NONE;
}

- (void)addPendingWindowOperation:(PendingWindowOperation)operation
{
    pendingWindowOperation |= operation;
}

- (BOOL)windowOperationIsPending:(PendingWindowOperation)operation
{
    return !!(pendingWindowOperation & operation);
}

- (BOOL)hasPendingWindowOperation
{
    // A pending zoom may be deferred until leaving fullscreen, so don't block on it.
    return (pendingWindowOperation & ~PENDING_OPERATION_ZOOM) != PENDING_OPERATION_NONE ||
           isMiniaturizing || inFullscreenTransition;
}

- (void)close
{
    NSNotificationCenter *center;
    NSWindow *window = _data.nswindow;
    NSView *view = [window contentView];

    center = [NSNotificationCenter defaultCenter];

    if ([window delegate] != self) {
        [center removeObserver:self name:NSWindowDidExposeNotification object:window];
        [center removeObserver:self name:NSWindowDidChangeOcclusionStateNotification object:window];
        [center removeObserver:self name:NSWindowWillStartLiveResizeNotification object:window];
        [center removeObserver:self name:NSWindowDidEndLiveResizeNotification object:window];
        [center removeObserver:self name:NSWindowWillMoveNotification object:window];
        [center removeObserver:self name:NSWindowDidMoveNotification object:window];
        [center removeObserver:self name:NSWindowDidResizeNotification object:window];
        [center removeObserver:self name:NSWindowWillMiniaturizeNotification object:window];
        [center removeObserver:self name:NSWindowDidMiniaturizeNotification object:window];
        [center removeObserver:self name:NSWindowDidDeminiaturizeNotification object:window];
        [center removeObserver:self name:NSWindowDidBecomeKeyNotification object:window];
        [center removeObserver:self name:NSWindowDidResignKeyNotification object:window];
        [center removeObserver:self name:NSWindowDidChangeBackingPropertiesNotification object:window];
        [center removeObserver:self name:NSWindowDidChangeScreenProfileNotification object:window];
        [center removeObserver:self name:NSWindowDidChangeScreenNotification object:window];
        [center removeObserver:self name:NSWindowWillEnterFullScreenNotification object:window];
        [center removeObserver:self name:NSWindowDidEnterFullScreenNotification object:window];
        [center removeObserver:self name:NSWindowWillExitFullScreenNotification object:window];
        [center removeObserver:self name:NSWindowDidExitFullScreenNotification object:window];
        [center removeObserver:self name:@"NSWindowDidFailToEnterFullScreenNotification" object:window];
        [center removeObserver:self name:@"NSWindowDidFailToExitFullScreenNotification" object:window];
    } else {
        [window setDelegate:nil];
    }

    [window removeObserver:self forKeyPath:@"visible"];

    if ([window nextResponder] == self) {
        [window setNextResponder:nil];
    }
    if ([view nextResponder] == self) {
        [view setNextResponder:nil];
    }
}

- (BOOL)isMoving
{
    return isMoving;
}

- (BOOL)isMovingOrFocusClickPending
{
    return isMoving || (focusClickPending != 0);
}

- (void)setFocusClickPending:(NSInteger)button
{
    focusClickPending |= (1 << button);
}

- (void)clearFocusClickPending:(NSInteger)button
{
    if (focusClickPending & (1 << button)) {
        focusClickPending &= ~(1 << button);
        if (focusClickPending == 0) {
            [self onMovingOrFocusClickPendingStateCleared];
        }
    }
}

- (void)updateIgnoreMouseState:(NSEvent *)theEvent
{
    SDL_Window *window = _data.window;
    SDL_Surface *shape = (SDL_Surface *)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_SHAPE_POINTER, NULL);
    BOOL ignoresMouseEvents = NO;

    if (shape) {
        NSPoint point = [theEvent locationInWindow];
        NSRect windowRect = [[_data.nswindow contentView] frame];
        if (NSMouseInRect(point, windowRect, NO)) {
            int x = (int)SDL_roundf((point.x / (window->w - 1)) * (shape->w - 1));
            int y = (int)SDL_roundf(((window->h - point.y) / (window->h - 1)) * (shape->h - 1));
            Uint8 a;

            if (!SDL_ReadSurfacePixel(shape, x, y, NULL, NULL, NULL, &a) || a == SDL_ALPHA_TRANSPARENT) {
                ignoresMouseEvents = YES;
            }
        }
    }
    _data.nswindow.ignoresMouseEvents = ignoresMouseEvents;
}

- (void)setPendingMoveX:(float)x Y:(float)y
{
    pendingWindowWarpX = x;
    pendingWindowWarpY = y;
}

- (void)windowDidFinishMoving
{
    if (isMoving) {
        isMoving = NO;
        [self onMovingOrFocusClickPendingStateCleared];
    }
}

- (void)onMovingOrFocusClickPendingStateCleared
{
    if (![self isMovingOrFocusClickPending]) {
        SDL_Mouse *mouse = SDL_GetMouse();
        if (pendingWindowWarpX != FLT_MAX && pendingWindowWarpY != FLT_MAX) {
            mouse->WarpMouseGlobal(pendingWindowWarpX, pendingWindowWarpY);
            pendingWindowWarpX = pendingWindowWarpY = FLT_MAX;
        }
        if (mouse->relative_mode && mouse->focus == _data.window) {
            // Move the cursor to the nearest point in the window
            {
                float x, y;
                CGPoint cgpoint;

                SDL_GetMouseState(&x, &y);
                cgpoint.x = _data.window->x + x;
                cgpoint.y = _data.window->y + y;

                Cocoa_HandleMouseWarp(cgpoint.x, cgpoint.y);

                DLog("Returning cursor to (%g, %g)", cgpoint.x, cgpoint.y);
                CGDisplayMoveCursorToPoint(kCGDirectMainDisplay, cgpoint);
            }

            mouse->SetRelativeMouseMode(true);
        } else {
            Cocoa_UpdateClipCursor(_data.window);
        }
    }
}

- (BOOL)windowShouldClose:(id)sender
{
    SDL_SendWindowEvent(_data.window, SDL_EVENT_WINDOW_CLOSE_REQUESTED, 0, 0);
    return NO;
}

- (void)windowDidExpose:(NSNotification *)aNotification
{
    Cocoa_SendExposedEventIfVisible(_data.window);
}

- (void)windowDidChangeOcclusionState:(NSNotification *)aNotification
{
    if ([_data.nswindow occlusionState] & NSWindowOcclusionStateVisible) {
        SDL_SendWindowEvent(_data.window, SDL_EVENT_WINDOW_EXPOSED, 0, 0);
    } else {
        SDL_SendWindowEvent(_data.window, SDL_EVENT_WINDOW_OCCLUDED, 0, 0);
    }
}

- (void)windowWillStartLiveResize:(NSNotification *)aNotification
{
    // We'll try to maintain 60 FPS during live resizing
    const NSTimeInterval interval = 1.0 / 60.0;
    liveResizeTimer = [NSTimer scheduledTimerWithTimeInterval:interval
                                                      repeats:TRUE
                                                        block:^(NSTimer *unusedTimer)
    {
        SDL_OnWindowLiveResizeUpdate(_data.window);
    }];

    [[NSRunLoop currentRunLoop] addTimer:liveResizeTimer forMode:NSRunLoopCommonModes];
}

- (void)windowDidEndLiveResize:(NSNotification *)aNotification
{
    [liveResizeTimer invalidate];
    liveResizeTimer = nil;
}

- (void)windowWillMove:(NSNotification *)aNotification
{
    if ([_data.nswindow isKindOfClass:[SDL3Window class]]) {
        pendingWindowWarpX = pendingWindowWarpY = FLT_MAX;
        isMoving = YES;
    }
}

- (void)windowDidMove:(NSNotification *)aNotification
{
    int x, y;
    SDL_Window *window = _data.window;
    NSWindow *nswindow = _data.nswindow;
    NSRect rect = [nswindow contentRectForFrameRect:[nswindow frame]];
    ConvertNSRect(&rect);

    if (inFullscreenTransition || b_inModeTransition) {
        // We'll take care of this at the end of the transition
        return;
    }

    x = (int)rect.origin.x;
    y = (int)rect.origin.y;

    ScheduleContextUpdates(_data);

    // Get the parent-relative coordinates for child windows.
    SDL_GlobalToRelativeForWindow(window, x, y, &x, &y);
    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_MOVED, x, y);
}

- (NSSize)windowWillResize:(NSWindow *)sender toSize:(NSSize)frameSize
{
    SDL_Window *window = _data.window;

    if (window->min_aspect != window->max_aspect) {
        NSWindow *nswindow = _data.nswindow;
        NSRect newContentRect = [nswindow contentRectForFrameRect:NSMakeRect(0, 0, frameSize.width, frameSize.height)];
        NSSize newSize = newContentRect.size;
        CGFloat minAspectRatio = window->min_aspect;
        CGFloat maxAspectRatio = window->max_aspect;
        CGFloat aspectRatio;

        if (newSize.height > 0) {
            aspectRatio = newSize.width / newSize.height;

            if (maxAspectRatio > 0.0f && aspectRatio > maxAspectRatio) {
                newSize.width = SDL_roundf(newSize.height * maxAspectRatio);
            } else if (minAspectRatio > 0.0f && aspectRatio < minAspectRatio) {
                newSize.height = SDL_roundf(newSize.width / minAspectRatio);
            }

            NSRect newFrameRect = [nswindow frameRectForContentRect:NSMakeRect(0, 0, newSize.width, newSize.height)];
            frameSize = newFrameRect.size;
        }
    }
    return frameSize;
}

- (void)windowDidResize:(NSNotification *)aNotification
{
    SDL_Window *window;
    NSWindow *nswindow;
    NSRect rect;
    int x, y, w, h;
    BOOL zoomed;

    if (inFullscreenTransition || b_inModeTransition) {
        // We'll take care of this at the end of the transition
        return;
    }

    if (focusClickPending) {
        focusClickPending = 0;
        [self onMovingOrFocusClickPendingStateCleared];
    }
    window = _data.window;
    nswindow = _data.nswindow;
    rect = [nswindow contentRectForFrameRect:[nswindow frame]];
    ConvertNSRect(&rect);
    x = (int)rect.origin.x;
    y = (int)rect.origin.y;
    w = (int)rect.size.width;
    h = (int)rect.size.height;

    ScheduleContextUpdates(_data);

    /* isZoomed always returns true if the window is not resizable
     * and fullscreen windows are considered zoomed.
     */
    if ((window->flags & SDL_WINDOW_RESIZABLE) && [nswindow isZoomed] &&
        !(window->flags & SDL_WINDOW_FULLSCREEN) && ![self isInFullscreenSpace]) {
        zoomed = YES;
    } else {
        zoomed = NO;
    }
    if (!zoomed) {
        SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_RESTORED, 0, 0);
    } else {
        SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_MAXIMIZED, 0, 0);
        if ([self windowOperationIsPending:PENDING_OPERATION_MINIMIZE]) {
            [nswindow miniaturize:nil];
        }
    }

    /* The window can move during a resize event, such as when maximizing
       or resizing from a corner */
    SDL_GlobalToRelativeForWindow(window, x, y, &x, &y);
    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_MOVED, x, y);
    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_RESIZED, w, h);
}

- (void)windowWillMiniaturize:(NSNotification *)aNotification
{
    isMiniaturizing = YES;
    Cocoa_WaitForMiniaturizable(_data.window);
}

- (void)windowDidMiniaturize:(NSNotification *)aNotification
{
    if (focusClickPending) {
        focusClickPending = 0;
        [self onMovingOrFocusClickPendingStateCleared];
    }
    isMiniaturizing = NO;
    [self clearPendingWindowOperation:PENDING_OPERATION_MINIMIZE];
    SDL_SendWindowEvent(_data.window, SDL_EVENT_WINDOW_MINIMIZED, 0, 0);
}

- (void)windowDidDeminiaturize:(NSNotification *)aNotification
{
    // Always send restored before maximized.
    SDL_SendWindowEvent(_data.window, SDL_EVENT_WINDOW_RESTORED, 0, 0);

    if (Cocoa_IsWindowZoomed(_data.window)) {
        SDL_SendWindowEvent(_data.window, SDL_EVENT_WINDOW_MAXIMIZED, 0, 0);
    }

    if ([self windowOperationIsPending:PENDING_OPERATION_ENTER_FULLSCREEN]) {
        SDL_UpdateFullscreenMode(_data.window, true, true);
    }
}

- (void)windowDidBecomeKey:(NSNotification *)aNotification
{
    SDL_Window *window = _data.window;

    // We're going to get keyboard events, since we're key.
    // This needs to be done before restoring the relative mouse mode.
    Cocoa_SetKeyboardFocus(window->keyboard_focus ? window->keyboard_focus : window, true);

    // If we just gained focus we need the updated mouse position
    if (!(window->flags & SDL_WINDOW_MOUSE_RELATIVE_MODE)) {
        NSPoint point;
        float x, y;

        point = [_data.nswindow mouseLocationOutsideOfEventStream];
        x = point.x;
        y = (window->h - point.y);

        if (x >= 0.0f && x < (float)window->w && y >= 0.0f && y < (float)window->h) {
            SDL_SendMouseMotion(0, window, SDL_GLOBAL_MOUSE_ID, false, x, y);
        }
    }

    // Check to see if someone updated the clipboard
    Cocoa_CheckClipboardUpdate(_data.videodata);

    if (isFullscreenSpace && !window->fullscreen_exclusive) {
        Cocoa_ToggleFullscreenSpaceMenuVisibility(window);
    }
    {
        const unsigned int newflags = [NSEvent modifierFlags] & NSEventModifierFlagCapsLock;
        _data.videodata.modifierFlags = (_data.videodata.modifierFlags & ~NSEventModifierFlagCapsLock) | newflags;
        SDL_ToggleModState(SDL_KMOD_CAPS, newflags ? true : false);
    }

    /* Restore fullscreen mode unless the window is deminiaturizing.
     * If it is, fullscreen will be restored when deminiaturization is complete.
     */
    if (!(window->flags & SDL_WINDOW_MINIMIZED) &&
        [self windowOperationIsPending:PENDING_OPERATION_ENTER_FULLSCREEN]) {
        SDL_UpdateFullscreenMode(window, true, true);
    }
}

- (void)windowDidResignKey:(NSNotification *)aNotification
{
    // Some other window will get mouse events, since we're not key.
    if (SDL_GetMouseFocus() == _data.window) {
        SDL_SetMouseFocus(NULL);
    }

    // Some other window will get keyboard events, since we're not key.
    if (SDL_GetKeyboardFocus() == _data.window) {
        SDL_SetKeyboardFocus(NULL);
    }

    if (isFullscreenSpace) {
        [NSMenu setMenuBarVisible:YES];
    }
}

- (void)windowDidChangeBackingProperties:(NSNotification *)aNotification
{
    NSNumber *oldscale = [[aNotification userInfo] objectForKey:NSBackingPropertyOldScaleFactorKey];

    if (inFullscreenTransition) {
        return;
    }

    if ([oldscale doubleValue] != [_data.nswindow backingScaleFactor]) {
        // Send a resize event when the backing scale factor changes.
        [self windowDidResize:aNotification];
    }
}

- (void)windowDidChangeScreenProfile:(NSNotification *)aNotification
{
    SDL_SendWindowEvent(_data.window, SDL_EVENT_WINDOW_ICCPROF_CHANGED, 0, 0);
}

- (void)windowDidChangeScreen:(NSNotification *)aNotification
{
    // printf("WINDOWDIDCHANGESCREEN\n");

#ifdef SDL_VIDEO_OPENGL

    if (_data && _data.nscontexts) {
        for (SDL3OpenGLContext *context in _data.nscontexts) {
            [context movedToNewScreen];
        }
    }

#endif // SDL_VIDEO_OPENGL
}

- (void)windowWillEnterFullScreen:(NSNotification *)aNotification
{
    SDL_Window *window = _data.window;
    const NSUInteger flags = NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable | NSWindowStyleMaskTitled;

    /* For some reason, the fullscreen window won't get any mouse button events
     * without the NSWindowStyleMaskTitled flag being set when entering fullscreen,
     * so it's needed even if the window is borderless.
     */
    SetWindowStyle(window, flags);

    _data.was_zoomed = !!(window->flags & SDL_WINDOW_MAXIMIZED);

    isFullscreenSpace = YES;
    inFullscreenTransition = YES;
}

/* This is usually sent after an unexpected windowDidExitFullscreen if the window
 * failed to become fullscreen.
 *
 * Since something went wrong and the current state is unknown, dump any pending events.
 */
- (void)windowDidFailToEnterFullScreen:(NSNotification *)aNotification
{
    [self clearAllPendingWindowOperations];
}

- (void)windowDidEnterFullScreen:(NSNotification *)aNotification
{
    SDL_Window *window = _data.window;

    inFullscreenTransition = NO;
    isFullscreenSpace = YES;
    [self clearPendingWindowOperation:PENDING_OPERATION_ENTER_FULLSCREEN];

    if ([self windowOperationIsPending:PENDING_OPERATION_LEAVE_FULLSCREEN]) {
        [self setFullscreenSpace:NO];
    } else {
        Cocoa_ToggleFullscreenSpaceMenuVisibility(window);

        /* Don't recurse back into UpdateFullscreenMode() if this was hit in
         * a blocking transition, as the caller is already waiting in
         * UpdateFullscreenMode().
         */
        if (!_data.in_blocking_transition) {
            SDL_UpdateFullscreenMode(window, true, false);
        }
        SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_ENTER_FULLSCREEN, 0, 0);

        _data.pending_position = NO;
        _data.pending_size = NO;

        /* Force the size change event in case it was delivered earlier
           while the window was still animating into place.
         */
        window->w = 0;
        window->h = 0;
        [self windowDidMove:aNotification];
        [self windowDidResize:aNotification];

        Cocoa_UpdateClipCursor(window);
    }
}

- (void)windowWillExitFullScreen:(NSNotification *)aNotification
{
    SDL_Window *window = _data.window;

    /* If the windowed mode borders were toggled on while in a fullscreen space,
     * NSWindowStyleMaskTitled has to be cleared here, or the window can end up
     * in a weird, semi-decorated state upon returning to windowed mode.
     */
    if (_data.border_toggled && !(window->flags & SDL_WINDOW_BORDERLESS)) {
        const NSUInteger flags = NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;

        SetWindowStyle(window, flags);
        _data.border_toggled = false;
    }

    isFullscreenSpace = NO;
    inFullscreenTransition = YES;
}

/* This may be sent before windowDidExitFullscreen to signal that the window was
 * dumped out of fullscreen with no animation.
 *
 * Since something went wrong and the state is unknown, dump any pending events.
 */
- (void)windowDidFailToExitFullScreen:(NSNotification *)aNotification
{
    [self clearAllPendingWindowOperations];
}

- (void)windowDidExitFullScreen:(NSNotification *)aNotification
{
    SDL_Window *window = _data.window;
    NSWindow *nswindow = _data.nswindow;

    inFullscreenTransition = NO;
    isFullscreenSpace = NO;
    _data.fullscreen_space_requested = NO;

    /* As of macOS 10.15, the window decorations can go missing sometimes after
       certain fullscreen-desktop->exclusive-fullscreen->windowed mode flows
       sometimes. Making sure the style mask always uses the windowed mode style
       when returning to windowed mode from a space (instead of using a pending
       fullscreen mode style mask) seems to work around that issue.
     */
    SetWindowStyle(window, GetWindowWindowedStyle(window));

    /* This can happen if the window failed to enter fullscreen, as this
     * may be called *before* windowDidFailToEnterFullScreen in that case.
     */
    if (!(window->flags & SDL_WINDOW_FULLSCREEN)) {
        [self clearAllPendingWindowOperations];
    }

    /* Don't recurse back into UpdateFullscreenMode() if this was hit in
     * a blocking transition, as the caller is already waiting in
     * UpdateFullscreenMode().
     */
    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_LEAVE_FULLSCREEN, 0, 0);
    if (!_data.in_blocking_transition) {
        SDL_UpdateFullscreenMode(window, false, false);
    }

    if (window->flags & SDL_WINDOW_ALWAYS_ON_TOP) {
        [nswindow setLevel:NSFloatingWindowLevel];
    } else {
        [nswindow setLevel:kCGNormalWindowLevel];
    }

    [self clearPendingWindowOperation:PENDING_OPERATION_LEAVE_FULLSCREEN];

    if ([self windowOperationIsPending:PENDING_OPERATION_ENTER_FULLSCREEN]) {
        [self setFullscreenSpace:YES];
    } else if ([self windowOperationIsPending:PENDING_OPERATION_MINIMIZE]) {
        /* There's some state that isn't quite back to normal when
         * windowDidExitFullScreen triggers. For example, the minimize button on
         * the title bar doesn't actually enable for another 200 milliseconds or
         * so on this MacBook. Camp here and wait for that to happen before
         * going on, in case we're exiting fullscreen to minimize, which need
         * that window state to be normal before it will work.
         */
        Cocoa_WaitForMiniaturizable(_data.window);
        [self addPendingWindowOperation:PENDING_OPERATION_ENTER_FULLSCREEN];
        [nswindow miniaturize:nil];
    } else {
        // Adjust the fullscreen toggle button and readd menu now that we're here.
        if (window->flags & SDL_WINDOW_RESIZABLE) {
            // resizable windows are Spaces-friendly: they get the "go fullscreen" toggle button on their titlebar.
            [nswindow setCollectionBehavior:NSWindowCollectionBehaviorFullScreenPrimary];
        } else {
            [nswindow setCollectionBehavior:NSWindowCollectionBehaviorManaged];
        }
        [NSMenu setMenuBarVisible:YES];

        // Toggle zoom, if changed while fullscreen.
        if ([self windowOperationIsPending:PENDING_OPERATION_ZOOM]) {
            [self clearPendingWindowOperation:PENDING_OPERATION_ZOOM];
            [nswindow zoom:nil];
            _data.was_zoomed = !_data.was_zoomed;
        }

        if (!_data.was_zoomed) {
            // Apply a pending window size, if not zoomed.
            NSRect rect;
            rect.origin.x = _data.pending_position ? window->pending.x : window->floating.x;
            rect.origin.y = _data.pending_position ? window->pending.y : window->floating.y;
            rect.size.width = _data.pending_size ? window->pending.w : window->floating.w;
            rect.size.height = _data.pending_size ? window->pending.h : window->floating.h;
            ConvertNSRect(&rect);

            if (_data.pending_size) {
                [nswindow setContentSize:rect.size];
            }
            if (_data.pending_position) {
                [nswindow setFrameOrigin:rect.origin];
            }
        }

        _data.pending_size = NO;
        _data.pending_position = NO;
        _data.was_zoomed = NO;

        /* Force the size change event in case it was delivered earlier
         * while the window was still animating into place.
         */
        window->w = 0;
        window->h = 0;
        [self windowDidMove:aNotification];
        [self windowDidResize:aNotification];

        // FIXME: Why does the window get hidden?
        if (!(window->flags & SDL_WINDOW_HIDDEN)) {
            Cocoa_ShowWindow(SDL_GetVideoDevice(), window);
        }

        Cocoa_UpdateClipCursor(window);
    }
}

- (NSApplicationPresentationOptions)window:(NSWindow *)window willUseFullScreenPresentationOptions:(NSApplicationPresentationOptions)proposedOptions
{
    if (_data.window->fullscreen_exclusive) {
        return NSApplicationPresentationFullScreen | NSApplicationPresentationHideDock | NSApplicationPresentationHideMenuBar;
    } else {
        return proposedOptions;
    }
}

/* We'll respond to key events by mostly doing nothing so we don't beep.
 * We could handle key messages here, but we lose some in the NSApp dispatch,
 * where they get converted to action messages, etc.
 */
- (void)flagsChanged:(NSEvent *)theEvent
{
    // Cocoa_HandleKeyEvent(SDL_GetVideoDevice(), theEvent);

    /* Catch capslock in here as a special case:
       https://developer.apple.com/library/archive/qa/qa1519/_index.html
       Note that technote's check of keyCode doesn't work. At least on the
       10.15 beta, capslock comes through here as keycode 255, but it's safe
       to send duplicate key events; SDL filters them out quickly in
       SDL_SendKeyboardKey(). */

    /* Also note that SDL_SendKeyboardKey expects all capslock events to be
       keypresses; it won't toggle the mod state if you send a keyrelease.  */
    const bool osenabled = ([theEvent modifierFlags] & NSEventModifierFlagCapsLock) ? true : false;
    const bool sdlenabled = (SDL_GetModState() & SDL_KMOD_CAPS) ? true : false;
    if (osenabled ^ sdlenabled) {
        SDL_SendKeyboardKey(0, SDL_DEFAULT_KEYBOARD_ID, 0, SDL_SCANCODE_CAPSLOCK, true);
        SDL_SendKeyboardKey(0, SDL_DEFAULT_KEYBOARD_ID, 0, SDL_SCANCODE_CAPSLOCK, false);
    }
}
- (void)keyDown:(NSEvent *)theEvent
{
    // Cocoa_HandleKeyEvent(SDL_GetVideoDevice(), theEvent);
}
- (void)keyUp:(NSEvent *)theEvent
{
    // Cocoa_HandleKeyEvent(SDL_GetVideoDevice(), theEvent);
}

/* We'll respond to selectors by doing nothing so we don't beep.
 * The escape key gets converted to a "cancel" selector, etc.
 */
- (void)doCommandBySelector:(SEL)aSelector
{
    // NSLog(@"doCommandBySelector: %@\n", NSStringFromSelector(aSelector));
}

- (void)updateHitTest
{
    SDL_Window *window = _data.window;
    BOOL draggable = NO;

    if (window->hit_test) {
        float x, y;
        SDL_Point point;

        SDL_GetGlobalMouseState(&x, &y);
        point.x = (int)SDL_roundf(x - window->x);
        point.y = (int)SDL_roundf(y - window->y);
        if (point.x >= 0 && point.x < window->w && point.y >= 0 && point.y < window->h) {
            if (window->hit_test(window, &point, window->hit_test_data) == SDL_HITTEST_DRAGGABLE) {
                draggable = YES;
            }
        }
    }

    if (isDragAreaRunning != draggable) {
        isDragAreaRunning = draggable;
        [_data.nswindow setMovableByWindowBackground:draggable];
    }
}

- (BOOL)processHitTest:(NSEvent *)theEvent
{
    SDL_Window *window = _data.window;

    if (window->hit_test) { // if no hit-test, skip this.
        const NSPoint location = [theEvent locationInWindow];
        const SDL_Point point = { (int)location.x, window->h - (((int)location.y) - 1) };
        const SDL_HitTestResult rc = window->hit_test(window, &point, window->hit_test_data);
        if (rc == SDL_HITTEST_DRAGGABLE) {
            if (!isDragAreaRunning) {
                isDragAreaRunning = YES;
                [_data.nswindow setMovableByWindowBackground:YES];
            }
            return YES; // dragging!
        } else {
            if (isDragAreaRunning) {
                isDragAreaRunning = NO;
                [_data.nswindow setMovableByWindowBackground:NO];
                return YES; // was dragging, drop event.
            }
        }
    }

    return NO; // not a special area, carry on.
}

static void Cocoa_SendMouseButtonClicks(SDL_Mouse *mouse, NSEvent *theEvent, SDL_Window *window, Uint8 button, bool down)
{
    SDL_MouseID mouseID = SDL_DEFAULT_MOUSE_ID;
    //const int clicks = (int)[theEvent clickCount];
    SDL_Window *focus = SDL_GetKeyboardFocus();

    // macOS will send non-left clicks to background windows without raising them, so we need to
    //  temporarily adjust the mouse position when this happens, as `mouse` will be tracking
    //  the position in the currently-focused window. We don't (currently) send a mousemove
    //  event for the background window, this just makes sure the button is reported at the
    //  correct position in its own event.
    if (focus && ([theEvent window] == ((__bridge SDL_CocoaWindowData *)focus->internal).nswindow)) {
        //SDL_SendMouseButtonClicks(Cocoa_GetEventTimestamp([theEvent timestamp]), window, mouseID, button, down, clicks);
        SDL_SendMouseButton(Cocoa_GetEventTimestamp([theEvent timestamp]), window, mouseID, button, down);
    } else {
        const float orig_x = mouse->x;
        const float orig_y = mouse->y;
        const NSPoint point = [theEvent locationInWindow];
        mouse->x = (int)point.x;
        mouse->y = (int)(window->h - point.y);
        //SDL_SendMouseButtonClicks(Cocoa_GetEventTimestamp([theEvent timestamp]), window, mouseID, button, down, clicks);
        SDL_SendMouseButton(Cocoa_GetEventTimestamp([theEvent timestamp]), window, mouseID, button, down);
        mouse->x = orig_x;
        mouse->y = orig_y;
    }
}

- (void)mouseDown:(NSEvent *)theEvent
{
    if (Cocoa_HandlePenEvent(_data, theEvent)) {
        return;  // pen code handled it.
    }

    SDL_Mouse *mouse = SDL_GetMouse();
    int button;

    if (!mouse) {
        return;
    }

    // Ignore events that aren't inside the client area (i.e. title bar.)
    if ([theEvent window]) {
        NSRect windowRect = [[[theEvent window] contentView] frame];
        if (!NSMouseInRect([theEvent locationInWindow], windowRect, NO)) {
            return;
        }
    }

    switch ([theEvent buttonNumber]) {
    case 0:
        if (([theEvent modifierFlags] & NSEventModifierFlagControl) &&
            GetHintCtrlClickEmulateRightClick()) {
            wasCtrlLeft = YES;
            button = SDL_BUTTON_RIGHT;
        } else {
            wasCtrlLeft = NO;
            button = SDL_BUTTON_LEFT;
        }
        break;
    case 1:
        button = SDL_BUTTON_RIGHT;
        break;
    case 2:
        button = SDL_BUTTON_MIDDLE;
        break;
    default:
        button = (int)[theEvent buttonNumber] + 1;
        break;
    }

    if (button == SDL_BUTTON_LEFT && [self processHitTest:theEvent]) {
        SDL_SendWindowEvent(_data.window, SDL_EVENT_WINDOW_HIT_TEST, 0, 0);
        return; // dragging, drop event.
    }

    Cocoa_SendMouseButtonClicks(mouse, theEvent, _data.window, button, true);
}

- (void)rightMouseDown:(NSEvent *)theEvent
{
    [self mouseDown:theEvent];
}

- (void)otherMouseDown:(NSEvent *)theEvent
{
    [self mouseDown:theEvent];
}

- (void)mouseUp:(NSEvent *)theEvent
{
    if (Cocoa_HandlePenEvent(_data, theEvent)) {
        return;  // pen code handled it.
    }

    SDL_Mouse *mouse = SDL_GetMouse();
    int button;

    if (!mouse) {
        return;
    }

    switch ([theEvent buttonNumber]) {
    case 0:
        if (wasCtrlLeft) {
            button = SDL_BUTTON_RIGHT;
            wasCtrlLeft = NO;
        } else {
            button = SDL_BUTTON_LEFT;
        }
        break;
    case 1:
        button = SDL_BUTTON_RIGHT;
        break;
    case 2:
        button = SDL_BUTTON_MIDDLE;
        break;
    default:
        button = (int)[theEvent buttonNumber] + 1;
        break;
    }

    if (button == SDL_BUTTON_LEFT && [self processHitTest:theEvent]) {
        SDL_SendWindowEvent(_data.window, SDL_EVENT_WINDOW_HIT_TEST, 0, 0);
        return; // stopped dragging, drop event.
    }

    Cocoa_SendMouseButtonClicks(mouse, theEvent, _data.window, button, false);
}

- (void)rightMouseUp:(NSEvent *)theEvent
{
    [self mouseUp:theEvent];
}

- (void)otherMouseUp:(NSEvent *)theEvent
{
    [self mouseUp:theEvent];
}

- (void)mouseMoved:(NSEvent *)theEvent
{
    if (Cocoa_HandlePenEvent(_data, theEvent)) {
        return;  // pen code handled it.
    }

    SDL_MouseID mouseID = SDL_DEFAULT_MOUSE_ID;
    SDL_Mouse *mouse = SDL_GetMouse();
    NSPoint point;
    float x, y;
    SDL_Window *window;
    NSView *contentView;

    if (!mouse) {
        return;
    }

    if (!Cocoa_GetMouseFocus()) {
        // The mouse is no longer over any window in the application
        SDL_SetMouseFocus(NULL);
        return;
    }

    window = _data.window;
    contentView = _data.sdlContentView;
    point = [theEvent locationInWindow];

    if ([contentView mouse:[contentView convertPoint:point fromView:nil] inRect:[contentView bounds]] &&
        [NSCursor currentCursor] != Cocoa_GetDesiredCursor()) {
        // The wrong cursor is on screen, fix it. This fixes an macOS bug that is only known to
        // occur in fullscreen windows on the built-in displays of newer MacBooks with camera
        // notches. When the mouse is moved near the top of such a window (within about 44 units)
        // and then moved back down, the cursor rects aren't respected.
        [_data.nswindow invalidateCursorRectsForView:contentView];
    }

    if (window->flags & SDL_WINDOW_TRANSPARENT) {
        [self updateIgnoreMouseState:theEvent];
    }

    if ([self processHitTest:theEvent]) {
        SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_HIT_TEST, 0, 0);
        return; // dragging, drop event.
    }

    if (mouse->relative_mode) {
        return;
    }

    x = point.x;
    y = (window->h - point.y);

    if (NSAppKitVersionNumber >= NSAppKitVersionNumber10_13_2) {
        // Mouse grab is taken care of by the confinement rect
    } else {
        CGPoint cgpoint;
        if (ShouldAdjustCoordinatesForGrab(window) &&
            AdjustCoordinatesForGrab(window, window->x + x, window->y + y, &cgpoint)) {
            Cocoa_HandleMouseWarp(cgpoint.x, cgpoint.y);
            CGDisplayMoveCursorToPoint(kCGDirectMainDisplay, cgpoint);
            CGAssociateMouseAndMouseCursorPosition(YES);
        }
    }

    SDL_SendMouseMotion(Cocoa_GetEventTimestamp([theEvent timestamp]), window, mouseID, false, x, y);
}

- (void)mouseDragged:(NSEvent *)theEvent
{
    [self mouseMoved:theEvent];
}

- (void)rightMouseDragged:(NSEvent *)theEvent
{
    [self mouseMoved:theEvent];
}

- (void)otherMouseDragged:(NSEvent *)theEvent
{
    [self mouseMoved:theEvent];
}

- (void)scrollWheel:(NSEvent *)theEvent
{
    Cocoa_HandleMouseWheel(_data.window, theEvent);
}

- (BOOL)isTouchFromTrackpad:(NSEvent *)theEvent
{
    SDL_Window *window = _data.window;
    SDL_CocoaVideoData *videodata = ((__bridge SDL_CocoaWindowData *)window->internal).videodata;

    /* if this a MacBook trackpad, we'll make input look like a synthesized
       event. This is backwards from reality, but better matches user
       expectations. You can make it look like a generic touch device instead
       with the SDL_HINT_TRACKPAD_IS_TOUCH_ONLY hint. */
    BOOL istrackpad = NO;
    if (!videodata.trackpad_is_touch_only) {
        @try {
            istrackpad = ([theEvent subtype] == NSEventSubtypeMouseEvent);
        }
        @catch (NSException *e) {
            /* if NSEvent type doesn't have subtype, such as NSEventTypeBeginGesture on
             * macOS 10.5 to 10.10, then NSInternalInconsistencyException is thrown.
             * This still prints a message to terminal so catching it's not an ideal solution.
             *
             * *** Assertion failure in -[NSEvent subtype]
             */
        }
    }
    return istrackpad;
}

- (void)touchesBeganWithEvent:(NSEvent *)theEvent
{
    NSSet *touches;
    SDL_TouchID touchID;
    int existingTouchCount;
    const BOOL istrackpad = [self isTouchFromTrackpad:theEvent];

    touches = [theEvent touchesMatchingPhase:NSTouchPhaseAny inView:nil];
    touchID = istrackpad ? SDL_MOUSE_TOUCHID : (SDL_TouchID)(intptr_t)[[touches anyObject] device];
    existingTouchCount = 0;

    for (NSTouch *touch in touches) {
        if ([touch phase] != NSTouchPhaseBegan) {
            existingTouchCount++;
        }
    }
    if (existingTouchCount == 0) {
        int numFingers;
        SDL_Finger **fingers = SDL_GetTouchFingers(touchID, &numFingers);
        if (fingers) {
            DLog("Reset Lost Fingers: %d", numFingers);
            for (--numFingers; numFingers >= 0; --numFingers) {
                const SDL_Finger *finger = fingers[numFingers];
                /* trackpad touches have no window. If we really wanted one we could
                 * use the window that has mouse or keyboard focus.
                 * Sending a null window currently also prevents synthetic mouse
                 * events from being generated from touch events.
                 */
                SDL_Window *window = NULL;
                SDL_SendTouch(Cocoa_GetEventTimestamp([theEvent timestamp]), touchID, finger->id, window, SDL_EVENT_FINGER_CANCELED, 0, 0, 0);
            }
            SDL_free(fingers);
        }
    }

    DLog("Began Fingers: %lu .. existing: %d", (unsigned long)[touches count], existingTouchCount);
    [self handleTouches:NSTouchPhaseBegan withEvent:theEvent];
}

- (void)touchesMovedWithEvent:(NSEvent *)theEvent
{
    [self handleTouches:NSTouchPhaseMoved withEvent:theEvent];
}

- (void)touchesEndedWithEvent:(NSEvent *)theEvent
{
    [self handleTouches:NSTouchPhaseEnded withEvent:theEvent];
}

- (void)touchesCancelledWithEvent:(NSEvent *)theEvent
{
    [self handleTouches:NSTouchPhaseCancelled withEvent:theEvent];
}

- (void)handleTouches:(NSTouchPhase)phase withEvent:(NSEvent *)theEvent
{
    NSSet *touches = [theEvent touchesMatchingPhase:phase inView:nil];
    const BOOL istrackpad = [self isTouchFromTrackpad:theEvent];
    SDL_FingerID fingerId;
    float x, y;

    for (NSTouch *touch in touches) {
        const SDL_TouchID touchId = istrackpad ? SDL_MOUSE_TOUCHID : (SDL_TouchID)(uintptr_t)[touch device];
        SDL_TouchDeviceType devtype = SDL_TOUCH_DEVICE_INDIRECT_ABSOLUTE;

        /* trackpad touches have no window. If we really wanted one we could
         * use the window that has mouse or keyboard focus.
         * Sending a null window currently also prevents synthetic mouse events
         * from being generated from touch events.
         */
        SDL_Window *window = NULL;

        /* TODO: Before implementing direct touch support here, we need to
         * figure out whether the OS generates mouse events from them on its
         * own. If it does, we should prevent SendTouch from generating
         * synthetic mouse events for these touches itself (while also
         * sending a window.) It will also need to use normalized window-
         * relative coordinates via [touch locationInView:].
         */
        if ([touch type] == NSTouchTypeDirect) {
            continue;
        }

        if (SDL_AddTouch(touchId, devtype, "") < 0) {
            return;
        }

        fingerId = (SDL_FingerID)(uintptr_t)[touch identity];
        x = [touch normalizedPosition].x;
        y = [touch normalizedPosition].y;
        // Make the origin the upper left instead of the lower left
        y = 1.0f - y;

        switch (phase) {
        case NSTouchPhaseBegan:
            SDL_SendTouch(Cocoa_GetEventTimestamp([theEvent timestamp]), touchId, fingerId, window, SDL_EVENT_FINGER_DOWN, x, y, 1.0f);
            break;
        case NSTouchPhaseEnded:
            SDL_SendTouch(Cocoa_GetEventTimestamp([theEvent timestamp]), touchId, fingerId, window, SDL_EVENT_FINGER_UP, x, y, 1.0f);
            break;
        case NSTouchPhaseCancelled:
            SDL_SendTouch(Cocoa_GetEventTimestamp([theEvent timestamp]), touchId, fingerId, window, SDL_EVENT_FINGER_CANCELED, x, y, 1.0f);
            break;
        case NSTouchPhaseMoved:
            SDL_SendTouchMotion(Cocoa_GetEventTimestamp([theEvent timestamp]), touchId, fingerId, window, x, y, 1.0f);
            break;
        default:
            break;
        }
    }
}

- (void)tabletProximity:(NSEvent *)theEvent
{
    Cocoa_HandlePenEvent(_data, theEvent);
}

- (void)tabletPoint:(NSEvent *)theEvent
{
    Cocoa_HandlePenEvent(_data, theEvent);
}

@end

@interface SDL3View : NSView
{
    SDL_Window *_sdlWindow;
}

- (void)setSDLWindow:(SDL_Window *)window;

// The default implementation doesn't pass rightMouseDown to responder chain
- (void)rightMouseDown:(NSEvent *)theEvent;
- (BOOL)mouseDownCanMoveWindow;
- (void)drawRect:(NSRect)dirtyRect;
- (BOOL)acceptsFirstMouse:(NSEvent *)theEvent;
- (BOOL)wantsUpdateLayer;
- (void)updateLayer;
@end

@implementation SDL3View

- (void)setSDLWindow:(SDL_Window *)window
{
    _sdlWindow = window;
}

/* this is used on older macOS revisions, and newer ones which emulate old
   NSOpenGLContext behaviour while still using a layer under the hood. 10.8 and
   later use updateLayer, up until 10.14.2 or so, which uses drawRect without
   a GraphicsContext and with a layer active instead (for OpenGL contexts). */
- (void)drawRect:(NSRect)dirtyRect
{
    /* Force the graphics context to clear to black so we don't get a flash of
       white until the app is ready to draw. In practice on modern macOS, this
       only gets called for window creation and other extraordinary events. */
    BOOL transparent = (_sdlWindow->flags & SDL_WINDOW_TRANSPARENT) != 0;
    if ([NSGraphicsContext currentContext]) {
        NSColor *fillColor = transparent ? [NSColor clearColor] : [NSColor blackColor];
        [fillColor setFill];
        NSRectFill(dirtyRect);
    } else if (self.layer) {
        CFStringRef color = transparent ? kCGColorClear : kCGColorBlack;
        self.layer.backgroundColor = CGColorGetConstantColor(color);
    }

    Cocoa_SendExposedEventIfVisible(_sdlWindow);
}

- (BOOL)wantsUpdateLayer
{
    return YES;
}

// This is also called when a Metal layer is active.
- (void)updateLayer
{
    /* Force the graphics context to clear to black so we don't get a flash of
       white until the app is ready to draw. In practice on modern macOS, this
       only gets called for window creation and other extraordinary events. */
    BOOL transparent = (_sdlWindow->flags & SDL_WINDOW_TRANSPARENT) != 0;
    CFStringRef color = transparent ? kCGColorClear : kCGColorBlack;
    self.layer.backgroundColor = CGColorGetConstantColor(color);
    ScheduleContextUpdates((__bridge SDL_CocoaWindowData *)_sdlWindow->internal);
    Cocoa_SendExposedEventIfVisible(_sdlWindow);
}

- (void)rightMouseDown:(NSEvent *)theEvent
{
    [[self nextResponder] rightMouseDown:theEvent];
}

- (BOOL)mouseDownCanMoveWindow
{
    /* Always say YES, but this doesn't do anything until we call
       -[NSWindow setMovableByWindowBackground:YES], which we ninja-toggle
       during mouse events when we're using a drag area. */
    return YES;
}

- (void)resetCursorRects
{
    [super resetCursorRects];
    [self addCursorRect:[self bounds]
                 cursor:Cocoa_GetDesiredCursor()];
}

- (BOOL)acceptsFirstMouse:(NSEvent *)theEvent
{
    if (_sdlWindow->flags & SDL_WINDOW_POPUP_MENU) {
        return YES;
    } else if (SDL_GetHint(SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH)) {
        return SDL_GetHintBoolean(SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH, false);
    } else {
        return SDL_GetHintBoolean("SDL_MAC_MOUSE_FOCUS_CLICKTHROUGH", false);
    }
}

@end

static void Cocoa_UpdateMouseFocus()
{
    const NSPoint mouseLocation = [NSEvent mouseLocation];

    // Find the topmost window under the pointer and send a motion event if it is an SDL window.
    [NSApp enumerateWindowsWithOptions:NSWindowListOrderedFrontToBack
                            usingBlock:^(NSWindow *nswin, BOOL *stop) {
                              NSRect r = [nswin contentRectForFrameRect:[nswin frame]];
                              if (NSPointInRect(mouseLocation, r)) {
                                  SDL_VideoDevice *vid = SDL_GetVideoDevice();
                                  SDL_Window *sdlwindow;
                                  for (sdlwindow = vid->windows; sdlwindow; sdlwindow = sdlwindow->next) {
                                      if (nswin == ((__bridge SDL_CocoaWindowData *)sdlwindow->internal).nswindow) {
                                          break;
                                      }
                                  }
                                  *stop = YES;
                                  if (sdlwindow) {
                                      int wx, wy;
                                      SDL_RelativeToGlobalForWindow(sdlwindow, sdlwindow->x, sdlwindow->y, &wx, &wy);

                                      // Calculate the cursor coordinates relative to the window.
                                      const float dx = mouseLocation.x - wx;
                                      const float dy = (CGDisplayPixelsHigh(kCGDirectMainDisplay) - mouseLocation.y) - wy;
                                      SDL_SendMouseMotion(0, sdlwindow, SDL_GLOBAL_MOUSE_ID, false, dx, dy);
                                  }
                              }
                            }];
}

static bool SetupWindowData(SDL_VideoDevice *_this, SDL_Window *window, NSWindow *nswindow, NSView *nsview)
{
    @autoreleasepool {
        SDL_CocoaVideoData *videodata = (__bridge SDL_CocoaVideoData *)_this->internal;
        SDL_CocoaWindowData *data;

        // Allocate the window data
        data = [[SDL_CocoaWindowData alloc] init];
        if (!data) {
            return SDL_OutOfMemory();
        }
        window->internal = (SDL_WindowData *)CFBridgingRetain(data);
        data.window = window;
        data.nswindow = nswindow;
        data.videodata = videodata;
        data.window_number = nswindow.windowNumber;
        data.nscontexts = [[NSMutableArray alloc] init];
        data.sdlContentView = nsview;

        // Create an event listener for the window
        data.listener = [[SDL3Cocoa_WindowListener alloc] init];

        // Fill in the SDL window with the window data
        {
            int x, y;
            NSRect rect = [nswindow contentRectForFrameRect:[nswindow frame]];
            ConvertNSRect(&rect);
            SDL_GlobalToRelativeForWindow(window, (int)rect.origin.x, (int)rect.origin.y, &x, &y);
            window->x = x;
            window->y = y;
            window->w = (int)rect.size.width;
            window->h = (int)rect.size.height;
        }

        // Set up the listener after we create the view
        [data.listener listen:data];

        if ([nswindow isVisible]) {
            window->flags &= ~SDL_WINDOW_HIDDEN;
        } else {
            window->flags |= SDL_WINDOW_HIDDEN;
        }

        {
            unsigned long style = [nswindow styleMask];

            /* NSWindowStyleMaskBorderless is zero, and it's possible to be
                Resizeable _and_ borderless, so we can't do a simple bitwise AND
                of NSWindowStyleMaskBorderless here. */
            if ((style & ~(NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable)) == NSWindowStyleMaskBorderless) {
                window->flags |= SDL_WINDOW_BORDERLESS;
            } else {
                window->flags &= ~SDL_WINDOW_BORDERLESS;
            }
            if (style & NSWindowStyleMaskResizable) {
                window->flags |= SDL_WINDOW_RESIZABLE;
            } else {
                window->flags &= ~SDL_WINDOW_RESIZABLE;
            }
        }

        // isZoomed always returns true if the window is not resizable
        if ((window->flags & SDL_WINDOW_RESIZABLE) && [nswindow isZoomed]) {
            window->flags |= SDL_WINDOW_MAXIMIZED;
        } else {
            window->flags &= ~SDL_WINDOW_MAXIMIZED;
        }

        if ([nswindow isMiniaturized]) {
            window->flags |= SDL_WINDOW_MINIMIZED;
        } else {
            window->flags &= ~SDL_WINDOW_MINIMIZED;
        }

        if (window->parent) {
            NSWindow *nsparent = ((__bridge SDL_CocoaWindowData *)window->parent->internal).nswindow;
            [nsparent addChildWindow:nswindow ordered:NSWindowAbove];

            /* FIXME: Should not need to call addChildWindow then orderOut.
               Attaching a hidden child window to a hidden parent window will cause the child window
               to show when the parent does. We therefore shouldn't attach the child window here as we're
               going to do so when the child window is explicitly shown later but skipping the addChildWindow
               entirely causes the child window to not get key focus correctly the first time it's shown. Adding
               then immediately ordering out (removing) the window does work. */
            if (window->flags & SDL_WINDOW_HIDDEN) {
                [nswindow orderOut:nil];
            }
        }

        if (!SDL_WINDOW_IS_POPUP(window)) {
            if ([nswindow isKeyWindow]) {
                window->flags |= SDL_WINDOW_INPUT_FOCUS;
                Cocoa_SetKeyboardFocus(data.window, true);
            }
        } else {
            if (window->flags & SDL_WINDOW_TOOLTIP) {
                [nswindow setIgnoresMouseEvents:YES];
                [nswindow setAcceptsMouseMovedEvents:NO];
            } else if ((window->flags & SDL_WINDOW_POPUP_MENU) && !(window->flags & SDL_WINDOW_HIDDEN)) {
                if (!(window->flags & SDL_WINDOW_NOT_FOCUSABLE)) {
                	Cocoa_SetKeyboardFocus(window, true);
                }
                Cocoa_UpdateMouseFocus();
            }
        }

        if (nswindow.isOpaque) {
            window->flags &= ~SDL_WINDOW_TRANSPARENT;
        } else {
            window->flags |= SDL_WINDOW_TRANSPARENT;
        }

        /* SDL_CocoaWindowData will be holding a strong reference to the NSWindow, and
         * it will also call [NSWindow close] in DestroyWindow before releasing the
         * NSWindow, so the extra release provided by releasedWhenClosed isn't
         * necessary. */
        nswindow.releasedWhenClosed = NO;

        /* Prevents the window's "window device" from being destroyed when it is
         * hidden. See http://www.mikeash.com/pyblog/nsopenglcontext-and-one-shot.html
         */
        [nswindow setOneShot:NO];

        if (window->flags & SDL_WINDOW_EXTERNAL) {
            // Query the title from the existing window
            NSString *title = [nswindow title];
            if (title) {
                window->title = SDL_strdup([title UTF8String]);
            }
        }

        SDL_PropertiesID props = SDL_GetWindowProperties(window);
        SDL_SetPointerProperty(props, SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, (__bridge void *)data.nswindow);
        SDL_SetNumberProperty(props, SDL_PROP_WINDOW_COCOA_METAL_VIEW_TAG_NUMBER, SDL_METALVIEW_TAG);

        // All done!
        return true;
    }
}

bool Cocoa_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props)
{
    @autoreleasepool {
        SDL_CocoaVideoData *videodata = (__bridge SDL_CocoaVideoData *)_this->internal;
        const void *data = SDL_GetPointerProperty(create_props, "sdl2-compat.external_window", NULL);
        NSWindow *nswindow = nil;
        NSView *nsview = nil;

        if (data) {
            if ([(__bridge id)data isKindOfClass:[NSWindow class]]) {
                nswindow = (__bridge NSWindow *)data;
            } else if ([(__bridge id)data isKindOfClass:[NSView class]]) {
                nsview = (__bridge NSView *)data;
            } else {
                SDL_assert(false);
            }
        } else {
            nswindow = (__bridge NSWindow *)SDL_GetPointerProperty(create_props, SDL_PROP_WINDOW_CREATE_COCOA_WINDOW_POINTER, NULL);
            nsview = (__bridge NSView *)SDL_GetPointerProperty(create_props, SDL_PROP_WINDOW_CREATE_COCOA_VIEW_POINTER, NULL);
        }
        if (nswindow && !nsview) {
            nsview = [nswindow contentView];
        }
        if (nsview && !nswindow) {
            nswindow = [nsview window];
        }
        if (nswindow) {
            window->flags |= SDL_WINDOW_EXTERNAL;
        } else {
            int x, y;
            NSScreen *screen;
            NSRect rect, screenRect;
            NSUInteger style;
            SDL3View *contentView;

            SDL_RelativeToGlobalForWindow(window, window->x, window->y, &x, &y);
            rect.origin.x = x;
            rect.origin.y = y;
            rect.size.width = window->w;
            rect.size.height = window->h;
            ConvertNSRect(&rect);

            style = GetWindowStyle(window);

            // Figure out which screen to place this window
            screen = ScreenForRect(&rect);
            screenRect = [screen frame];
            rect.origin.x -= screenRect.origin.x;
            rect.origin.y -= screenRect.origin.y;

            // Constrain the popup
            if (SDL_WINDOW_IS_POPUP(window) && window->constrain_popup) {
                if (rect.origin.x + rect.size.width > screenRect.origin.x + screenRect.size.width) {
                    rect.origin.x -= (rect.origin.x + rect.size.width) - (screenRect.origin.x + screenRect.size.width);
                }
                if (rect.origin.y + rect.size.height > screenRect.origin.y + screenRect.size.height) {
                    rect.origin.y -= (rect.origin.y + rect.size.height) - (screenRect.origin.y + screenRect.size.height);
                }
                rect.origin.x = SDL_max(rect.origin.x, screenRect.origin.x);
                rect.origin.y = SDL_max(rect.origin.y, screenRect.origin.y);
            }

            @try {
                nswindow = [[SDL3Window alloc] initWithContentRect:rect styleMask:style backing:NSBackingStoreBuffered defer:NO screen:screen];
            }
            @catch (NSException *e) {
                return SDL_SetError("%s", [[e reason] UTF8String]);
            }

            [nswindow setColorSpace:[NSColorSpace sRGBColorSpace]];

            [nswindow setTabbingMode:NSWindowTabbingModeDisallowed];

            if (videodata.allow_spaces) {
                // we put fullscreen desktop windows in their own Space, without a toggle button or menubar, later
                if (window->flags & SDL_WINDOW_RESIZABLE) {
                    // resizable windows are Spaces-friendly: they get the "go fullscreen" toggle button on their titlebar.
                    [nswindow setCollectionBehavior:NSWindowCollectionBehaviorFullScreenPrimary];
                }
            }

            // Create a default view for this window
            rect = [nswindow contentRectForFrameRect:[nswindow frame]];
            contentView = [[SDL3View alloc] initWithFrame:rect];
            [contentView setSDLWindow:window];
            nsview = contentView;
        }

        if (window->flags & SDL_WINDOW_ALWAYS_ON_TOP) {
            [nswindow setLevel:NSFloatingWindowLevel];
        }

        if (window->flags & SDL_WINDOW_TRANSPARENT) {
            nswindow.opaque = NO;
            nswindow.hasShadow = NO;
            nswindow.backgroundColor = [NSColor clearColor];
        }

// We still support OpenGL as long as Apple offers it, deprecated or not, so disable deprecation warnings about it.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
        /* Note: as of the macOS 10.15 SDK, this defaults to YES instead of NO when
         * the NSHighResolutionCapable boolean is set in Info.plist. */
        BOOL highdpi = (window->flags & SDL_WINDOW_HIGH_PIXEL_DENSITY) ? YES : NO;
        [nsview setWantsBestResolutionOpenGLSurface:highdpi];
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef SDL_VIDEO_OPENGL_ES2
#ifdef SDL_VIDEO_OPENGL_EGL
        if ((window->flags & SDL_WINDOW_OPENGL) &&
            _this->gl_config.profile_mask == SDL_GL_CONTEXT_PROFILE_ES) {
            [nsview setWantsLayer:TRUE];
            if ((window->flags & SDL_WINDOW_HIGH_PIXEL_DENSITY)) {
                nsview.layer.contentsScale = nswindow.screen.backingScaleFactor;
            } else {
                nsview.layer.contentsScale = 1;
            }
        }
#endif // SDL_VIDEO_OPENGL_EGL
#endif // SDL_VIDEO_OPENGL_ES2
        [nswindow setContentView:nsview];

        if (!SetupWindowData(_this, window, nswindow, nsview)) {
            return false;
        }

        if (!(window->flags & SDL_WINDOW_OPENGL)) {
            return true;
        }

        // The rest of this macro mess is for OpenGL or OpenGL ES windows
#ifdef SDL_VIDEO_OPENGL_ES2
        if (_this->gl_config.profile_mask == SDL_GL_CONTEXT_PROFILE_ES) {
#ifdef SDL_VIDEO_OPENGL_EGL
            if (!Cocoa_GLES_SetupWindow(_this, window)) {
                Cocoa_DestroyWindow(_this, window);
                return false;
            }
            return true;
#else
            return SDL_SetError("Could not create GLES window surface (EGL support not configured)");
#endif // SDL_VIDEO_OPENGL_EGL
        }
#endif // SDL_VIDEO_OPENGL_ES2
        return true;
    }
}

void Cocoa_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        const char *title = window->title ? window->title : "";
        NSWindow *nswindow = ((__bridge SDL_CocoaWindowData *)window->internal).nswindow;
        NSString *string = [[NSString alloc] initWithUTF8String:title];
        [nswindow setTitle:string];
    }
}

bool Cocoa_SetWindowIcon(SDL_VideoDevice *_this, SDL_Window *window, SDL_Surface *icon)
{
    @autoreleasepool {
        NSImage *nsimage = Cocoa_CreateImage(icon);

        if (nsimage) {
            [NSApp setApplicationIconImage:nsimage];

            return true;
        }

        return SDL_SetError("Unable to set the window's icon");
    }
}

bool Cocoa_SetWindowPosition(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_CocoaWindowData *windata = (__bridge SDL_CocoaWindowData *)window->internal;
        NSWindow *nswindow = windata.nswindow;
        NSRect rect = [nswindow contentRectForFrameRect:[nswindow frame]];
        BOOL fullscreen = (window->flags & SDL_WINDOW_FULLSCREEN) ? YES : NO;
        int x, y;

        if ([windata.listener isInFullscreenSpaceTransition]) {
            windata.pending_position = YES;
            return true;
        }

        if (!(window->flags & SDL_WINDOW_MAXIMIZED)) {
            if (fullscreen) {
                SDL_VideoDisplay *display = SDL_GetVideoDisplayForFullscreenWindow(window);
                SDL_Rect r;
                SDL_GetDisplayBounds(display->id, &r);

                rect.origin.x = r.x;
                rect.origin.y = r.y;
            } else {
                SDL_RelativeToGlobalForWindow(window, window->pending.x, window->pending.y, &x, &y);
                rect.origin.x = x;
                rect.origin.y = y;
            }
            ConvertNSRect(&rect);

            // Position and constrain the popup
            if (SDL_WINDOW_IS_POPUP(window) && window->constrain_popup) {
                NSRect screenRect = [ScreenForRect(&rect) frame];

                if (rect.origin.x + rect.size.width > screenRect.origin.x + screenRect.size.width) {
                    rect.origin.x -= (rect.origin.x + rect.size.width) - (screenRect.origin.x + screenRect.size.width);
                }
                if (rect.origin.y + rect.size.height > screenRect.origin.y + screenRect.size.height) {
                    rect.origin.y -= (rect.origin.y + rect.size.height) - (screenRect.origin.y + screenRect.size.height);
                }
                rect.origin.x = SDL_max(rect.origin.x, screenRect.origin.x);
                rect.origin.y = SDL_max(rect.origin.y, screenRect.origin.y);
            }

            [nswindow setFrameOrigin:rect.origin];

            ScheduleContextUpdates(windata);
        }
    }
    return true;
}

void Cocoa_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_CocoaWindowData *windata = (__bridge SDL_CocoaWindowData *)window->internal;
        NSWindow *nswindow = windata.nswindow;

        if ([windata.listener isInFullscreenSpaceTransition]) {
            windata.pending_size = YES;
            return;
        }

        if (!Cocoa_IsWindowZoomed(window)) {
            int x, y;
            NSRect rect = [nswindow contentRectForFrameRect:[nswindow frame]];

            /* Cocoa will resize the window from the bottom-left rather than the
             * top-left when -[nswindow setContentSize:] is used, so we must set the
             * entire frame based on the new size, in order to preserve the position.
             */
            SDL_RelativeToGlobalForWindow(window, window->floating.x, window->floating.y, &x, &y);
            rect.origin.x = x;
            rect.origin.y = y;
            rect.size.width = window->pending.w;
            rect.size.height = window->pending.h;
            ConvertNSRect(&rect);

            [nswindow setFrame:[nswindow frameRectForContentRect:rect] display:YES];
            ScheduleContextUpdates(windata);
        } else {
            // Can't set the window size.
            window->last_size_pending = false;
        }
    }
}

void Cocoa_SetWindowMinimumSize(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_CocoaWindowData *windata = (__bridge SDL_CocoaWindowData *)window->internal;

        NSSize minSize;
        minSize.width = window->min_w;
        minSize.height = window->min_h;

        [windata.nswindow setContentMinSize:minSize];
    }
}

void Cocoa_SetWindowMaximumSize(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_CocoaWindowData *windata = (__bridge SDL_CocoaWindowData *)window->internal;

        NSSize maxSize;
        maxSize.width = window->max_w ? window->max_w : CGFLOAT_MAX;
        maxSize.height = window->max_h ? window->max_h : CGFLOAT_MAX;

        [windata.nswindow setContentMaxSize:maxSize];
    }
}

void Cocoa_SetWindowAspectRatio(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_CocoaWindowData *windata = (__bridge SDL_CocoaWindowData *)window->internal;

        if (window->min_aspect > 0.0f && window->min_aspect == window->max_aspect) {
            int numerator = 0, denominator = 1;
            SDL_CalculateFraction(window->max_aspect, &numerator, &denominator);
            [windata.nswindow setContentAspectRatio:NSMakeSize(numerator, denominator)];
        } else {
            [windata.nswindow setContentAspectRatio:NSMakeSize(0, 0)];
        }
    }
}

void Cocoa_GetWindowSizeInPixels(SDL_VideoDevice *_this, SDL_Window *window, int *w, int *h)
{
    @autoreleasepool {
        SDL_CocoaWindowData *windata = (__bridge SDL_CocoaWindowData *)window->internal;
        NSView *contentView = windata.sdlContentView;
        NSRect viewport = [contentView bounds];

        if (window->flags & SDL_WINDOW_HIGH_PIXEL_DENSITY) {
            // This gives us the correct viewport for a Retina-enabled view.
            viewport = [contentView convertRectToBacking:viewport];
        }

        *w = (int)viewport.size.width;
        *h = (int)viewport.size.height;
    }
}

void Cocoa_ShowWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_CocoaWindowData *windowData = ((__bridge SDL_CocoaWindowData *)window->internal);
        NSWindow *nswindow = windowData.nswindow;
        bool bActivate = SDL_GetHintBoolean(SDL_HINT_WINDOW_ACTIVATE_WHEN_SHOWN, true);

        if (![nswindow isMiniaturized]) {
            [windowData.listener pauseVisibleObservation];
            if (window->parent) {
                NSWindow *nsparent = ((__bridge SDL_CocoaWindowData *)window->parent->internal).nswindow;
                [nsparent addChildWindow:nswindow ordered:NSWindowAbove];

                if (window->flags & SDL_WINDOW_MODAL) {
                    Cocoa_SetWindowModal(_this, window, true);
                }
            }
            if (!SDL_WINDOW_IS_POPUP(window)) {
                if (bActivate) {
                    [nswindow makeKeyAndOrderFront:nil];
                } else {
                    // Order this window below the key window if we're not activating it
                    if ([NSApp keyWindow]) {
                        [nswindow orderWindow:NSWindowBelow relativeTo:[[NSApp keyWindow] windowNumber]];
                    }
                }
            } else if (window->flags & SDL_WINDOW_POPUP_MENU) {
                if (!(window->flags & SDL_WINDOW_NOT_FOCUSABLE)) {
                	Cocoa_SetKeyboardFocus(window, true);
                }
                Cocoa_UpdateMouseFocus();
            }
        }
        [nswindow setIsVisible:YES];
        [windowData.listener resumeVisibleObservation];
    }
}

void Cocoa_HideWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        NSWindow *nswindow = ((__bridge SDL_CocoaWindowData *)window->internal).nswindow;
        const BOOL waskey = [nswindow isKeyWindow];

        /* orderOut has no effect on miniaturized windows, so close must be used to remove
         * the window from the desktop and window list in this case.
         *
         * SDL holds a strong reference to the window (oneShot/releasedWhenClosed are 'NO'),
         * and calling 'close' doesn't send a 'windowShouldClose' message, so it's safe to
         * use for this purpose as nothing is implicitly released.
         */
        if (![nswindow isMiniaturized]) {
            [nswindow orderOut:nil];
        } else {
            [nswindow close];
        }

        /* If this window is the source of a modal session, end it when
         * hidden, or other windows will be prevented from closing.
         */
        Cocoa_SetWindowModal(_this, window, false);

        // Transfer keyboard focus back to the parent when closing a popup menu
        if ((window->flags & SDL_WINDOW_POPUP_MENU) && !(window->flags & SDL_WINDOW_NOT_FOCUSABLE)) {
            SDL_Window *new_focus;
            const bool set_focus = SDL_ShouldRelinquishPopupFocus(window, &new_focus);
            Cocoa_SetKeyboardFocus(new_focus, set_focus);
            Cocoa_UpdateMouseFocus();
        } else if (window->parent && waskey) {
            /* Key status is not automatically set on the parent when a child is hidden. Check if the
             * child window was key, and set the first visible parent to be key if so.
             */
            SDL_Window *new_focus = window->parent;

            while (new_focus->parent != NULL && (new_focus->is_hiding || new_focus->is_destroying)) {
                new_focus = new_focus->parent;
            }

            if (new_focus) {
                NSWindow *newkey = ((__bridge SDL_CocoaWindowData *)window->internal).nswindow;
                [newkey makeKeyAndOrderFront:nil];
            }
        }
    }
}

void Cocoa_RaiseWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_CocoaWindowData *windowData = ((__bridge SDL_CocoaWindowData *)window->internal);
        NSWindow *nswindow = windowData.nswindow;
        bool bActivate = SDL_GetHintBoolean(SDL_HINT_WINDOW_ACTIVATE_WHEN_RAISED, true);

        /* makeKeyAndOrderFront: has the side-effect of deminiaturizing and showing
         a minimized or hidden window, so check for that before showing it.
         */
        [windowData.listener pauseVisibleObservation];
        if (![nswindow isMiniaturized] && [nswindow isVisible]) {
            if (window->parent) {
                NSWindow *nsparent = ((__bridge SDL_CocoaWindowData *)window->parent->internal).nswindow;
                [nsparent addChildWindow:nswindow ordered:NSWindowAbove];
            }
            if (!SDL_WINDOW_IS_POPUP(window)) {
                if (bActivate) {
                    [NSApp activateIgnoringOtherApps:YES];
                    [nswindow makeKeyAndOrderFront:nil];
                } else {
                    [nswindow orderFront:nil];
                }
            } else {
                if (bActivate) {
                    [nswindow makeKeyWindow];
                }
            }
        }
        [windowData.listener resumeVisibleObservation];
    }
}

void Cocoa_MaximizeWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_CocoaWindowData *windata = (__bridge SDL_CocoaWindowData *)window->internal;
        NSWindow *nswindow = windata.nswindow;

        if ([windata.listener windowOperationIsPending:(PENDING_OPERATION_ENTER_FULLSCREEN | PENDING_OPERATION_LEAVE_FULLSCREEN)] ||
            [windata.listener isInFullscreenSpaceTransition]) {
            Cocoa_SyncWindow(_this, window);
        }

        if (!(window->flags & SDL_WINDOW_FULLSCREEN) &&
            ![windata.listener isInFullscreenSpaceTransition] &&
            ![windata.listener isInFullscreenSpace]) {
            [nswindow zoom:nil];
            ScheduleContextUpdates(windata);
        } else if (!windata.was_zoomed) {
            [windata.listener addPendingWindowOperation:PENDING_OPERATION_ZOOM];
        } else {
            [windata.listener clearPendingWindowOperation:PENDING_OPERATION_ZOOM];
        }
    }
}

void Cocoa_MinimizeWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
        NSWindow *nswindow = data.nswindow;

        [data.listener addPendingWindowOperation:PENDING_OPERATION_MINIMIZE];
        if ([data.listener isInFullscreenSpace] || (window->flags & SDL_WINDOW_FULLSCREEN)) {
            [data.listener addPendingWindowOperation:PENDING_OPERATION_LEAVE_FULLSCREEN];
            SDL_UpdateFullscreenMode(window, false, true);
        } else if ([data.listener isInFullscreenSpaceTransition]) {
            [data.listener addPendingWindowOperation:PENDING_OPERATION_LEAVE_FULLSCREEN];
        } else {
            [nswindow miniaturize:nil];
        }
    }
}

void Cocoa_RestoreWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
        NSWindow *nswindow = data.nswindow;

        if (([data.listener windowOperationIsPending:(PENDING_OPERATION_ENTER_FULLSCREEN | PENDING_OPERATION_LEAVE_FULLSCREEN)] &&
            ![data.nswindow isMiniaturized]) || [data.listener isInFullscreenSpaceTransition]) {
            Cocoa_SyncWindow(_this, window);
        }

        [data.listener clearPendingWindowOperation:(PENDING_OPERATION_MINIMIZE)];

        if (!(window->flags & SDL_WINDOW_FULLSCREEN) &&
            ![data.listener isInFullscreenSpaceTransition] &&
            ![data.listener isInFullscreenSpace]) {
            if ([nswindow isMiniaturized]) {
                [nswindow deminiaturize:nil];
            } else if (Cocoa_IsWindowZoomed(window)) {
                [nswindow zoom:nil];
            }
        } else if (data.was_zoomed) {
            [data.listener addPendingWindowOperation:PENDING_OPERATION_ZOOM];
        } else {
            [data.listener clearPendingWindowOperation:PENDING_OPERATION_ZOOM];
        }
    }
}

void Cocoa_SetWindowBordered(SDL_VideoDevice *_this, SDL_Window *window, bool bordered)
{
    @autoreleasepool {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;

        // If the window is in or transitioning to/from fullscreen, this will be set on leave.
        if (!(window->flags & SDL_WINDOW_FULLSCREEN) && ![data.listener isInFullscreenSpaceTransition]) {
            if (SetWindowStyle(window, GetWindowStyle(window))) {
                if (bordered) {
                    Cocoa_SetWindowTitle(_this, window); // this got blanked out.
                }
            }
        } else {
            data.border_toggled = true;
        }
        Cocoa_UpdateClipCursor(window);
    }
}

void Cocoa_SetWindowResizable(SDL_VideoDevice *_this, SDL_Window *window, bool resizable)
{
    @autoreleasepool {
        /* Don't set this if we're in or transitioning to/from a space!
         * The window will get permanently stuck if resizable is false.
         * -flibit
         */
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
        SDL3Cocoa_WindowListener *listener = data.listener;
        NSWindow *nswindow = data.nswindow;
        SDL_CocoaVideoData *videodata = data.videodata;
        if (![listener isInFullscreenSpace] && ![listener isInFullscreenSpaceTransition]) {
            SetWindowStyle(window, GetWindowStyle(window));
        }
        if (videodata.allow_spaces) {
            if (resizable) {
                // resizable windows are Spaces-friendly: they get the "go fullscreen" toggle button on their titlebar.
                [nswindow setCollectionBehavior:NSWindowCollectionBehaviorFullScreenPrimary];
            } else {
                [nswindow setCollectionBehavior:NSWindowCollectionBehaviorManaged];
            }
        }
    }
}

void Cocoa_SetWindowAlwaysOnTop(SDL_VideoDevice *_this, SDL_Window *window, bool on_top)
{
    @autoreleasepool {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
        NSWindow *nswindow = data.nswindow;

        // If the window is in or transitioning to/from fullscreen, this will be set on leave.
        if (!(window->flags & SDL_WINDOW_FULLSCREEN) && ![data.listener isInFullscreenSpaceTransition]) {
            if (on_top) {
                [nswindow setLevel:NSFloatingWindowLevel];
            } else {
                [nswindow setLevel:kCGNormalWindowLevel];
            }
        }
    }
}

SDL_FullscreenResult Cocoa_SetWindowFullscreen(SDL_VideoDevice *_this, SDL_Window *window, SDL_VideoDisplay *display, SDL_FullscreenOp fullscreen)
{
    @autoreleasepool {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
        NSWindow *nswindow = data.nswindow;
        NSRect rect;

        // This is a synchronous operation, so always clear the pending flags.
        [data.listener clearPendingWindowOperation:PENDING_OPERATION_ENTER_FULLSCREEN | PENDING_OPERATION_LEAVE_FULLSCREEN];

        // The view responder chain gets messed with during setStyleMask
        if ([data.sdlContentView nextResponder] == data.listener) {
            [data.sdlContentView setNextResponder:nil];
        }

        if (fullscreen) {
            SDL_Rect bounds;

            if (!(window->flags & SDL_WINDOW_FULLSCREEN)) {
                data.was_zoomed = !!(window->flags & SDL_WINDOW_MAXIMIZED);
            }

            SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_ENTER_FULLSCREEN, 0, 0);
            Cocoa_GetDisplayBounds(_this, display, &bounds);
            rect.origin.x = bounds.x;
            rect.origin.y = bounds.y;
            rect.size.width = bounds.w;
            rect.size.height = bounds.h;
            ConvertNSRect(&rect);

            /* Hack to fix origin on macOS 10.4
               This is no longer needed as of macOS 10.15, according to bug 4822.
             */
            if (SDL_floor(NSAppKitVersionNumber) <= NSAppKitVersionNumber10_14) {
                NSRect screenRect = [[nswindow screen] frame];
                if (screenRect.size.height >= 1.0f) {
                    rect.origin.y += (screenRect.size.height - rect.size.height);
                }
            }

            [nswindow setStyleMask:NSWindowStyleMaskBorderless];
        } else {
            NSRect frameRect;

            SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_LEAVE_FULLSCREEN, 0, 0);

            rect.origin.x = data.was_zoomed ? window->windowed.x : window->floating.x;
            rect.origin.y = data.was_zoomed ? window->windowed.y : window->floating.y;
            rect.size.width = data.was_zoomed ? window->windowed.w : window->floating.w;
            rect.size.height = data.was_zoomed ? window->windowed.h : window->floating.h;

            ConvertNSRect(&rect);

            /* The window is not meant to be fullscreen, but its flags might have a
             * fullscreen bit set if it's scheduled to go fullscreen immediately
             * after. Always using the windowed mode style here works around bugs in
             * macOS 10.15 where the window doesn't properly restore the windowed
             * mode decorations after exiting fullscreen-desktop, when the window
             * was created as fullscreen-desktop. */
            [nswindow setStyleMask:GetWindowWindowedStyle(window)];

            // Hack to restore window decorations on macOS 10.10
            frameRect = [nswindow frame];
            [nswindow setFrame:NSMakeRect(frameRect.origin.x, frameRect.origin.y, frameRect.size.width + 1, frameRect.size.height) display:NO];
            [nswindow setFrame:frameRect display:NO];
        }

        // The view responder chain gets messed with during setStyleMask
        if ([data.sdlContentView nextResponder] != data.listener) {
            [data.sdlContentView setNextResponder:data.listener];
        }

        [nswindow setContentSize:rect.size];
        [nswindow setFrameOrigin:rect.origin];

        // When the window style changes the title is cleared
        if (!fullscreen) {
            Cocoa_SetWindowTitle(_this, window);
            data.was_zoomed = NO;
            if ([data.listener windowOperationIsPending:PENDING_OPERATION_ZOOM]) {
                [data.listener clearPendingWindowOperation:PENDING_OPERATION_ZOOM];
                [nswindow zoom:nil];
            }
        }

        if (SDL_ShouldAllowTopmost() && fullscreen) {
            // OpenGL is rendering to the window, so make it visible!
            [nswindow setLevel:kCGMainMenuWindowLevel + 1];
        } else if (window->flags & SDL_WINDOW_ALWAYS_ON_TOP) {
            [nswindow setLevel:NSFloatingWindowLevel];
        } else {
            [nswindow setLevel:kCGNormalWindowLevel];
        }

        if ([nswindow isVisible] || fullscreen) {
            [data.listener pauseVisibleObservation];
            [nswindow makeKeyAndOrderFront:nil];
            [data.listener resumeVisibleObservation];
        }

        // Update the safe area insets
        // The view never seems to reflect the safe area, so we'll use the screen instead
        if (@available(macOS 12.0, *)) {
            if (fullscreen) {
                NSScreen *screen = [nswindow screen];

                SDL_SetWindowSafeAreaInsets(data.window,
                                            (int)SDL_ceilf(screen.safeAreaInsets.left),
                                            (int)SDL_ceilf(screen.safeAreaInsets.right),
                                            (int)SDL_ceilf(screen.safeAreaInsets.top),
                                            (int)SDL_ceilf(screen.safeAreaInsets.bottom));
            } else {
                SDL_SetWindowSafeAreaInsets(data.window, 0, 0, 0, 0);
            }
        }

        /* When coming out of fullscreen to minimize, this needs to happen after the window
         * is made key again, or it won't minimize on 15.0 (Sequoia).
         */
        if (!fullscreen && [data.listener windowOperationIsPending:PENDING_OPERATION_MINIMIZE]) {
            Cocoa_WaitForMiniaturizable(window);
            [data.listener addPendingWindowOperation:PENDING_OPERATION_ENTER_FULLSCREEN];
            [data.listener clearPendingWindowOperation:PENDING_OPERATION_MINIMIZE];
            [nswindow miniaturize:nil];
        }

        ScheduleContextUpdates(data);
        Cocoa_SyncWindow(_this, window);
        Cocoa_UpdateClipCursor(window);
    }

    return SDL_FULLSCREEN_SUCCEEDED;
}

void *Cocoa_GetWindowICCProfile(SDL_VideoDevice *_this, SDL_Window *window, size_t *size)
{
    @autoreleasepool {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
        NSWindow *nswindow = data.nswindow;
        NSScreen *screen = [nswindow screen];
        NSData *iccProfileData = nil;
        void *retIccProfileData = NULL;

        if (screen == nil) {
            SDL_SetError("Could not get screen of window.");
            return NULL;
        }

        if ([screen colorSpace] == nil) {
            SDL_SetError("Could not get colorspace information of screen.");
            return NULL;
        }

        iccProfileData = [[screen colorSpace] ICCProfileData];
        if (iccProfileData == nil) {
            SDL_SetError("Could not get ICC profile data.");
            return NULL;
        }

        retIccProfileData = SDL_malloc([iccProfileData length]);
        if (!retIccProfileData) {
            return NULL;
        }

        [iccProfileData getBytes:retIccProfileData length:[iccProfileData length]];
        *size = [iccProfileData length];
        return retIccProfileData;
    }
}

SDL_DisplayID Cocoa_GetDisplayForWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        NSScreen *screen;
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;

        // Not recognized via CHECK_WINDOW_MAGIC
        if (data == nil) {
            // Don't set the error here, it hides other errors and is ignored anyway
            // return SDL_SetError("Window data not set");
            return 0;
        }

        // NSWindow.screen may be nil when the window is off-screen.
        screen = data.nswindow.screen;

        if (screen != nil) {
            // https://developer.apple.com/documentation/appkit/nsscreen/1388360-devicedescription?language=objc
            CGDirectDisplayID displayid = [[screen.deviceDescription objectForKey:@"NSScreenNumber"] unsignedIntValue];
            SDL_VideoDisplay *display = Cocoa_FindSDLDisplayByCGDirectDisplayID(_this, displayid);
            if (display) {
                return display->id;
            }
        }

        // The higher level code will use other logic to find the display
        return 0;
    }
}

bool Cocoa_SetWindowMouseRect(SDL_VideoDevice *_this, SDL_Window *window)
{
    Cocoa_UpdateClipCursor(window);
    return true;
}

bool Cocoa_SetWindowMouseGrab(SDL_VideoDevice *_this, SDL_Window *window, bool grabbed)
{
    @autoreleasepool {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;

        Cocoa_UpdateClipCursor(window);

        if (data && (window->flags & SDL_WINDOW_FULLSCREEN) != 0) {
            if (SDL_ShouldAllowTopmost() && (window->flags & SDL_WINDOW_INPUT_FOCUS) && ![data.listener isInFullscreenSpace]) {
                // OpenGL is rendering to the window, so make it visible!
                // Doing this in 10.11 while in a Space breaks things (bug #3152)
                [data.nswindow setLevel:kCGMainMenuWindowLevel + 1];
            } else if (window->flags & SDL_WINDOW_ALWAYS_ON_TOP) {
                [data.nswindow setLevel:NSFloatingWindowLevel];
            } else {
                [data.nswindow setLevel:kCGNormalWindowLevel];
            }
        }
    }

    return true;
}

void Cocoa_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_CocoaWindowData *data = (SDL_CocoaWindowData *)CFBridgingRelease(window->internal);

        if (data) {
#ifdef SDL_VIDEO_OPENGL

            NSArray *contexts;

#endif // SDL_VIDEO_OPENGL
            SDL_Window *topmost = GetParentToplevelWindow(window);

            /* Reset the input focus of the root window if this window is still set as keyboard focus.
             * SDL_DestroyWindow will have already taken care of reassigning focus if this is the SDL
             * keyboard focus, this ensures that an inactive window with this window set as input focus
             * does not try to reference it the next time it gains focus.
             */
            if (topmost->keyboard_focus == window) {
                SDL_Window *new_focus = window;
                while (SDL_WINDOW_IS_POPUP(new_focus) && (new_focus->is_hiding || new_focus->is_destroying)) {
                    new_focus = new_focus->parent;
                }

                topmost->keyboard_focus = new_focus;
            }

            if ([data.listener isInFullscreenSpace]) {
                [NSMenu setMenuBarVisible:YES];
            }
            [data.listener close];
            data.listener = nil;

            if (!(window->flags & SDL_WINDOW_EXTERNAL)) {
                // Release the content view to avoid further updateLayer callbacks
                [data.nswindow setContentView:nil];
                [data.nswindow close];
            }

#ifdef SDL_VIDEO_OPENGL

            contexts = [data.nscontexts copy];
            for (SDL3OpenGLContext *context in contexts) {
                // Calling setWindow:NULL causes the context to remove itself from the context list.
                [context setWindow:NULL];
            }

#endif // SDL_VIDEO_OPENGL
        }
        window->internal = NULL;
    }
}

bool Cocoa_SetWindowFullscreenSpace(SDL_Window *window, bool state, bool blocking)
{
    @autoreleasepool {
        bool succeeded = false;
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;

        if (state) {
            data.fullscreen_space_requested = YES;
        }
        data.in_blocking_transition = blocking;
        if ([data.listener setFullscreenSpace:(state ? YES : NO)]) {
            if (blocking) {
                const int maxattempts = 3;
                int attempt = 0;
                while (++attempt <= maxattempts) {
                    /* Wait for the transition to complete, so application changes
                     take effect properly (e.g. setting the window size, etc.)
                     */
                    const int limit = 10000;
                    int count = 0;
                    while ([data.listener isInFullscreenSpaceTransition]) {
                        if (++count == limit) {
                            // Uh oh, transition isn't completing. Should we assert?
                            break;
                        }
                        SDL_Delay(1);
                        SDL_PumpEvents();
                    }
                    if ([data.listener isInFullscreenSpace] == (state ? YES : NO)) {
                        break;
                    }
                    // Try again, the last attempt was interrupted by user gestures
                    if (![data.listener setFullscreenSpace:(state ? YES : NO)]) {
                        break; // ???
                    }
                }
            }

            // Return TRUE to prevent non-space fullscreen logic from running
            succeeded = true;
        }

        data.in_blocking_transition = NO;
        return succeeded;
    }
}

bool Cocoa_SetWindowHitTest(SDL_Window *window, bool enabled)
{
    SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;

    [data.listener updateHitTest];
    return true;
}

void Cocoa_AcceptDragAndDrop(SDL_Window *window, bool accept)
{
    @autoreleasepool {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
        if (accept) {
            [data.nswindow registerForDraggedTypes:@[ (NSString *)kUTTypeFileURL,
                                                      (NSString *)kUTTypeUTF8PlainText ]];
        } else {
            [data.nswindow unregisterDraggedTypes];
        }
    }
}

bool Cocoa_SetWindowParent(SDL_VideoDevice *_this, SDL_Window *window, SDL_Window *parent)
{
    @autoreleasepool {
        SDL_CocoaWindowData *child_data = (__bridge SDL_CocoaWindowData *)window->internal;

        // Remove an existing parent.
        if (child_data.nswindow.parentWindow) {
            NSWindow *nsparent = ((__bridge SDL_CocoaWindowData *)window->parent->internal).nswindow;
            [nsparent removeChildWindow:child_data.nswindow];
        }

        if (parent) {
            SDL_CocoaWindowData *parent_data = (__bridge SDL_CocoaWindowData *)parent->internal;
            [parent_data.nswindow addChildWindow:child_data.nswindow ordered:NSWindowAbove];
        }
    }

    return true;
}

bool Cocoa_SetWindowModal(SDL_VideoDevice *_this, SDL_Window *window, bool modal)
{
    @autoreleasepool {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;

        if (data.modal_session) {
            [NSApp endModalSession:data.modal_session];
            data.modal_session = nil;
        }

        if (modal) {
            data.modal_session = [NSApp beginModalSessionForWindow:data.nswindow];
        }
    }

    return true;
}

bool Cocoa_FlashWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_FlashOperation operation)
{
    @autoreleasepool {
        // Note that this is app-wide and not window-specific!
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;

        if (data.flash_request) {
            [NSApp cancelUserAttentionRequest:data.flash_request];
            data.flash_request = 0;
        }

        switch (operation) {
        case SDL_FLASH_CANCEL:
            // Canceled above
            break;
        case SDL_FLASH_BRIEFLY:
            data.flash_request = [NSApp requestUserAttention:NSInformationalRequest];
            break;
        case SDL_FLASH_UNTIL_FOCUSED:
            data.flash_request = [NSApp requestUserAttention:NSCriticalRequest];
            break;
        default:
            return SDL_Unsupported();
        }
        return true;
    }
}

bool Cocoa_SetWindowFocusable(SDL_VideoDevice *_this, SDL_Window *window, bool focusable)
{
    if (window->flags & SDL_WINDOW_POPUP_MENU) {
        if (!(window->flags & SDL_WINDOW_HIDDEN)) {
            if (!focusable && (window->flags & SDL_WINDOW_INPUT_FOCUS)) {
                SDL_Window *new_focus;
            	const bool set_focus = SDL_ShouldRelinquishPopupFocus(window, &new_focus);
            	Cocoa_SetKeyboardFocus(new_focus, set_focus);
            } else if (focusable) {
                if (SDL_ShouldFocusPopup(window)) {
                    Cocoa_SetKeyboardFocus(window, true);
                }
            }
        }
    }

    return true; // just succeed, the real work is done elsewhere.
}

bool Cocoa_SetWindowOpacity(SDL_VideoDevice *_this, SDL_Window *window, float opacity)
{
    @autoreleasepool {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
        [data.nswindow setAlphaValue:opacity];
        return true;
    }
}

bool Cocoa_SyncWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    bool result = false;

    @autoreleasepool {
        const Uint64 timeout = SDL_GetTicksNS() + SDL_MS_TO_NS(2500);
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;

        for (;;) {
            SDL_PumpEvents();

            result = ![data.listener hasPendingWindowOperation];
            if (result || SDL_GetTicksNS() >= timeout) {
                break;
            }

            // Small delay before going again.
            SDL_Delay(10);
        }
    }

    return result;
}

#endif // SDL_VIDEO_DRIVER_COCOA
