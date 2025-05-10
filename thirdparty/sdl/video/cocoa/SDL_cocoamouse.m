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

#include "SDL_cocoamouse.h"
#include "SDL_cocoavideo.h"

#include "../../events/SDL_mouse_c.h"

#if 0
#define DEBUG_COCOAMOUSE
#endif

#ifdef DEBUG_COCOAMOUSE
#define DLog(fmt, ...) printf("%s: " fmt "\n", __func__, ##__VA_ARGS__)
#else
#define DLog(...) \
    do {          \
    } while (0)
#endif

@implementation NSCursor (InvisibleCursor)
+ (NSCursor *)invisibleCursor
{
    static NSCursor *invisibleCursor = NULL;
    if (!invisibleCursor) {
        // RAW 16x16 transparent GIF
        static unsigned char cursorBytes[] = {
            0x47, 0x49, 0x46, 0x38, 0x37, 0x61, 0x10, 0x00, 0x10, 0x00, 0x80,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0xF9, 0x04,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x2C, 0x00, 0x00, 0x00, 0x00, 0x10,
            0x00, 0x10, 0x00, 0x00, 0x02, 0x0E, 0x8C, 0x8F, 0xA9, 0xCB, 0xED,
            0x0F, 0xA3, 0x9C, 0xB4, 0xDA, 0x8B, 0xB3, 0x3E, 0x05, 0x00, 0x3B
        };

        NSData *cursorData = [NSData dataWithBytesNoCopy:&cursorBytes[0]
                                                  length:sizeof(cursorBytes)
                                            freeWhenDone:NO];
        NSImage *cursorImage = [[NSImage alloc] initWithData:cursorData];
        invisibleCursor = [[NSCursor alloc] initWithImage:cursorImage
                                                  hotSpot:NSZeroPoint];
    }

    return invisibleCursor;
}
@end

static SDL_Cursor *Cocoa_CreateCursor(SDL_Surface *surface, int hot_x, int hot_y)
{
    @autoreleasepool {
        NSImage *nsimage;
        NSCursor *nscursor = NULL;
        SDL_Cursor *cursor = NULL;

        nsimage = Cocoa_CreateImage(surface);
        if (nsimage) {
            nscursor = [[NSCursor alloc] initWithImage:nsimage hotSpot:NSMakePoint(hot_x, hot_y)];
        }

        if (nscursor) {
            cursor = SDL_calloc(1, sizeof(*cursor));
            if (cursor) {
                cursor->internal = (void *)CFBridgingRetain(nscursor);
            }
        }

        return cursor;
    }
}

/* there are .pdf files of some of the cursors we need, installed by default on macOS, but not available through NSCursor.
   If we can load them ourselves, use them, otherwise fallback to something standard but not super-great.
   Since these are under /System, they should be available even to sandboxed apps. */
static NSCursor *LoadHiddenSystemCursor(NSString *cursorName, SEL fallback)
{
    NSString *cursorPath = [@"/System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/HIServices.framework/Versions/A/Resources/cursors" stringByAppendingPathComponent:cursorName];
    NSDictionary *info = [NSDictionary dictionaryWithContentsOfFile:[cursorPath stringByAppendingPathComponent:@"info.plist"]];
    // we can't do animation atm.  :/
    const int frames = (int)[[info valueForKey:@"frames"] integerValue];
    NSCursor *cursor;
    NSImage *image = [[NSImage alloc] initWithContentsOfFile:[cursorPath stringByAppendingPathComponent:@"cursor.pdf"]];
    if ((image == nil) || (image.isValid == NO)) {
        return [NSCursor performSelector:fallback];
    }

    if (frames > 1) {
#ifdef MAC_OS_VERSION_12_0 // same value as deprecated symbol.
        const NSCompositingOperation operation = NSCompositingOperationCopy;
#else
        const NSCompositingOperation operation = NSCompositeCopy;
#endif
        const NSSize cropped_size = NSMakeSize(image.size.width, (int)(image.size.height / frames));
        NSImage *cropped = [[NSImage alloc] initWithSize:cropped_size];
        if (cropped == nil) {
            return [NSCursor performSelector:fallback];
        }

        [cropped lockFocus];
        {
            const NSRect cropped_rect = NSMakeRect(0, 0, cropped_size.width, cropped_size.height);
            [image drawInRect:cropped_rect fromRect:cropped_rect operation:operation fraction:1];
        }
        [cropped unlockFocus];
        image = cropped;
    }

    cursor = [[NSCursor alloc] initWithImage:image hotSpot:NSMakePoint([[info valueForKey:@"hotx"] doubleValue], [[info valueForKey:@"hoty"] doubleValue])];
    return cursor;
}

static SDL_Cursor *Cocoa_CreateSystemCursor(SDL_SystemCursor id)
{
    @autoreleasepool {
        NSCursor *nscursor = NULL;
        SDL_Cursor *cursor = NULL;

        switch (id) {
        case SDL_SYSTEM_CURSOR_DEFAULT:
            nscursor = [NSCursor arrowCursor];
            break;
        case SDL_SYSTEM_CURSOR_TEXT:
            nscursor = [NSCursor IBeamCursor];
            break;
        case SDL_SYSTEM_CURSOR_CROSSHAIR:
            nscursor = [NSCursor crosshairCursor];
            break;
        case SDL_SYSTEM_CURSOR_WAIT: // !!! FIXME: this is more like WAITARROW
            nscursor = LoadHiddenSystemCursor(@"busybutclickable", @selector(arrowCursor));
            break;
        case SDL_SYSTEM_CURSOR_PROGRESS: // !!! FIXME: this is meant to be animated
            nscursor = LoadHiddenSystemCursor(@"busybutclickable", @selector(arrowCursor));
            break;
        case SDL_SYSTEM_CURSOR_NWSE_RESIZE:
            nscursor = LoadHiddenSystemCursor(@"resizenorthwestsoutheast", @selector(closedHandCursor));
            break;
        case SDL_SYSTEM_CURSOR_NESW_RESIZE:
            nscursor = LoadHiddenSystemCursor(@"resizenortheastsouthwest", @selector(closedHandCursor));
            break;
        case SDL_SYSTEM_CURSOR_EW_RESIZE:
            nscursor = LoadHiddenSystemCursor(@"resizeeastwest", @selector(resizeLeftRightCursor));
            break;
        case SDL_SYSTEM_CURSOR_NS_RESIZE:
            nscursor = LoadHiddenSystemCursor(@"resizenorthsouth", @selector(resizeUpDownCursor));
            break;
        case SDL_SYSTEM_CURSOR_MOVE:
            nscursor = LoadHiddenSystemCursor(@"move", @selector(closedHandCursor));
            break;
        case SDL_SYSTEM_CURSOR_NOT_ALLOWED:
            nscursor = [NSCursor operationNotAllowedCursor];
            break;
        case SDL_SYSTEM_CURSOR_POINTER:
            nscursor = [NSCursor pointingHandCursor];
            break;
        case SDL_SYSTEM_CURSOR_NW_RESIZE:
            nscursor = LoadHiddenSystemCursor(@"resizenorthwestsoutheast", @selector(closedHandCursor));
            break;
        case SDL_SYSTEM_CURSOR_N_RESIZE:
            nscursor = LoadHiddenSystemCursor(@"resizenorthsouth", @selector(resizeUpDownCursor));
            break;
        case SDL_SYSTEM_CURSOR_NE_RESIZE:
            nscursor = LoadHiddenSystemCursor(@"resizenortheastsouthwest", @selector(closedHandCursor));
            break;
        case SDL_SYSTEM_CURSOR_E_RESIZE:
            nscursor = LoadHiddenSystemCursor(@"resizeeastwest", @selector(resizeLeftRightCursor));
            break;
        case SDL_SYSTEM_CURSOR_SE_RESIZE:
            nscursor = LoadHiddenSystemCursor(@"resizenorthwestsoutheast", @selector(closedHandCursor));
            break;
        case SDL_SYSTEM_CURSOR_S_RESIZE:
            nscursor = LoadHiddenSystemCursor(@"resizenorthsouth", @selector(resizeUpDownCursor));
            break;
        case SDL_SYSTEM_CURSOR_SW_RESIZE:
            nscursor = LoadHiddenSystemCursor(@"resizenortheastsouthwest", @selector(closedHandCursor));
            break;
        case SDL_SYSTEM_CURSOR_W_RESIZE:
            nscursor = LoadHiddenSystemCursor(@"resizeeastwest", @selector(resizeLeftRightCursor));
            break;
        default:
            SDL_assert(!"Unknown system cursor");
            return NULL;
        }

        if (nscursor) {
            cursor = SDL_calloc(1, sizeof(*cursor));
            if (cursor) {
                // We'll free it later, so retain it here
                cursor->internal = (void *)CFBridgingRetain(nscursor);
            }
        }

        return cursor;
    }
}

static SDL_Cursor *Cocoa_CreateDefaultCursor(void)
{
    SDL_SystemCursor id = SDL_GetDefaultSystemCursor();
    return Cocoa_CreateSystemCursor(id);
}

static void Cocoa_FreeCursor(SDL_Cursor *cursor)
{
    @autoreleasepool {
        CFBridgingRelease((void *)cursor->internal);
        SDL_free(cursor);
    }
}

static bool Cocoa_ShowCursor(SDL_Cursor *cursor)
{
    @autoreleasepool {
        SDL_VideoDevice *device = SDL_GetVideoDevice();
        SDL_Window *window = (device ? device->windows : NULL);
        for (; window != NULL; window = window->next) {
            SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
            if (data) {
                [data.nswindow performSelectorOnMainThread:@selector(invalidateCursorRectsForView:)
                                                withObject:[data.nswindow contentView]
                                             waitUntilDone:NO];
            }
        }
        return true;
    }
}

static SDL_Window *SDL_FindWindowAtPoint(const float x, const float y)
{
    const SDL_FPoint pt = { x, y };
    SDL_Window *i;
    for (i = SDL_GetVideoDevice()->windows; i; i = i->next) {
        const SDL_FRect r = { (float)i->x, (float)i->y, (float)i->w, (float)i->h };
        if (SDL_PointInRectFloat(&pt, &r)) {
            return i;
        }
    }

    return NULL;
}

static bool Cocoa_WarpMouseGlobal(float x, float y)
{
    CGPoint point;
    SDL_Mouse *mouse = SDL_GetMouse();
    if (mouse->focus) {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)mouse->focus->internal;
        if ([data.listener isMovingOrFocusClickPending]) {
            DLog("Postponing warp, window being moved or focused.");
            [data.listener setPendingMoveX:x Y:y];
            return true;
        }
    }
    point = CGPointMake(x, y);

    Cocoa_HandleMouseWarp(point.x, point.y);

    CGWarpMouseCursorPosition(point);

    /* CGWarpMouse causes a short delay by default, which is preventable by
     * Calling this directly after. CGSetLocalEventsSuppressionInterval can also
     * prevent it, but it's deprecated as macOS 10.6.
     */
    if (!mouse->relative_mode) {
        CGAssociateMouseAndMouseCursorPosition(YES);
    }

    /* CGWarpMouseCursorPosition doesn't generate a window event, unlike our
     * other implementations' APIs. Send what's appropriate.
     */
    if (!mouse->relative_mode) {
        SDL_Window *win = SDL_FindWindowAtPoint(x, y);
        SDL_SetMouseFocus(win);
        if (win) {
            SDL_assert(win == mouse->focus);
            SDL_SendMouseMotion(0, win, SDL_GLOBAL_MOUSE_ID, false, x - win->x, y - win->y);
        }
    }

    return true;
}

static bool Cocoa_WarpMouse(SDL_Window *window, float x, float y)
{
    return Cocoa_WarpMouseGlobal(window->x + x, window->y + y);
}

static bool Cocoa_SetRelativeMouseMode(bool enabled)
{
    CGError result;

    if (enabled) {
        SDL_Window *window = SDL_GetKeyboardFocus();
        if (window) {
            /* We will re-apply the relative mode when the window finishes being moved,
             * if it is being moved right now.
             */
            SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
            if ([data.listener isMovingOrFocusClickPending]) {
                return true;
            }

            // make sure the mouse isn't at the corner of the window, as this can confuse things if macOS thinks a window resize is happening on the first click.
            const CGPoint point = CGPointMake((float)(window->x + (window->w / 2)), (float)(window->y + (window->h / 2)));
            Cocoa_HandleMouseWarp(point.x, point.y);
            CGWarpMouseCursorPosition(point);
        }
        DLog("Turning on.");
        result = CGAssociateMouseAndMouseCursorPosition(NO);
    } else {
        DLog("Turning off.");
        result = CGAssociateMouseAndMouseCursorPosition(YES);
    }
    if (result != kCGErrorSuccess) {
        return SDL_SetError("CGAssociateMouseAndMouseCursorPosition() failed");
    }

    /* The hide/unhide calls are redundant most of the time, but they fix
     * https://bugzilla.libsdl.org/show_bug.cgi?id=2550
     */
    if (enabled) {
        [NSCursor hide];
    } else {
        [NSCursor unhide];
    }
    return true;
}

static bool Cocoa_CaptureMouse(SDL_Window *window)
{
    /* our Cocoa event code already tracks the mouse outside the window,
        so all we have to do here is say "okay" and do what we always do. */
    return true;
}

static SDL_MouseButtonFlags Cocoa_GetGlobalMouseState(float *x, float *y)
{
    const NSUInteger cocoaButtons = [NSEvent pressedMouseButtons];
    const NSPoint cocoaLocation = [NSEvent mouseLocation];
    SDL_MouseButtonFlags result = 0;

    *x = cocoaLocation.x;
    *y = (CGDisplayPixelsHigh(kCGDirectMainDisplay) - cocoaLocation.y);

    result |= (cocoaButtons & (1 << 0)) ? SDL_BUTTON_LMASK : 0;
    result |= (cocoaButtons & (1 << 1)) ? SDL_BUTTON_RMASK : 0;
    result |= (cocoaButtons & (1 << 2)) ? SDL_BUTTON_MMASK : 0;
    result |= (cocoaButtons & (1 << 3)) ? SDL_BUTTON_X1MASK : 0;
    result |= (cocoaButtons & (1 << 4)) ? SDL_BUTTON_X2MASK : 0;

    return result;
}

bool Cocoa_InitMouse(SDL_VideoDevice *_this)
{
    NSPoint location;
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_MouseData *internal = (SDL_MouseData *)SDL_calloc(1, sizeof(SDL_MouseData));
    if (internal == NULL) {
        return false;
    }

    mouse->internal = internal;
    mouse->CreateCursor = Cocoa_CreateCursor;
    mouse->CreateSystemCursor = Cocoa_CreateSystemCursor;
    mouse->ShowCursor = Cocoa_ShowCursor;
    mouse->FreeCursor = Cocoa_FreeCursor;
    mouse->WarpMouse = Cocoa_WarpMouse;
    mouse->WarpMouseGlobal = Cocoa_WarpMouseGlobal;
    mouse->SetRelativeMouseMode = Cocoa_SetRelativeMouseMode;
    mouse->CaptureMouse = Cocoa_CaptureMouse;
    mouse->GetGlobalMouseState = Cocoa_GetGlobalMouseState;

    SDL_SetDefaultCursor(Cocoa_CreateDefaultCursor());

    location = [NSEvent mouseLocation];
    internal->lastMoveX = location.x;
    internal->lastMoveY = location.y;
    return true;
}

static void Cocoa_HandleTitleButtonEvent(SDL_VideoDevice *_this, NSEvent *event)
{
    SDL_Window *window;
    NSWindow *nswindow = [event window];

    /* You might land in this function before SDL_Init if showing a message box.
       Don't dereference a NULL pointer if that happens. */
    if (_this == NULL) {
        return;
    }

    for (window = _this->windows; window; window = window->next) {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
        if (data && data.nswindow == nswindow) {
            switch ([event type]) {
            case NSEventTypeLeftMouseDown:
            case NSEventTypeRightMouseDown:
            case NSEventTypeOtherMouseDown:
                [data.listener setFocusClickPending:[event buttonNumber]];
                break;
            case NSEventTypeLeftMouseUp:
            case NSEventTypeRightMouseUp:
            case NSEventTypeOtherMouseUp:
                [data.listener clearFocusClickPending:[event buttonNumber]];
                break;
            default:
                break;
            }
            break;
        }
    }
}

static NSWindow *Cocoa_MouseFocus;

NSWindow *Cocoa_GetMouseFocus()
{
    return Cocoa_MouseFocus;
}

void Cocoa_HandleMouseEvent(SDL_VideoDevice *_this, NSEvent *event)
{
    SDL_MouseID mouseID = SDL_DEFAULT_MOUSE_ID;
    SDL_Mouse *mouse;
    SDL_MouseData *data;
    NSPoint location;
    CGFloat lastMoveX, lastMoveY;
    float deltaX, deltaY;
    bool seenWarp;

    // All events except NSEventTypeMouseExited can only happen if the window
    // has mouse focus, so we'll always set the focus even if we happen to miss
    // NSEventTypeMouseEntered, which apparently happens if the window is
    // created under the mouse on macOS 12.7
    NSEventType event_type = [event type];
    if (event_type == NSEventTypeMouseExited) {
        Cocoa_MouseFocus = NULL;
    } else {
        Cocoa_MouseFocus = [event window];
    }

    switch (event_type) {
    case NSEventTypeMouseEntered:
    case NSEventTypeMouseExited:
        // Focus is handled above
        return;

    case NSEventTypeMouseMoved:
    case NSEventTypeLeftMouseDragged:
    case NSEventTypeRightMouseDragged:
    case NSEventTypeOtherMouseDragged:
        break;

    case NSEventTypeLeftMouseDown:
    case NSEventTypeLeftMouseUp:
    case NSEventTypeRightMouseDown:
    case NSEventTypeRightMouseUp:
    case NSEventTypeOtherMouseDown:
    case NSEventTypeOtherMouseUp:
        if ([event window]) {
            NSRect windowRect = [[[event window] contentView] frame];
            if (!NSMouseInRect([event locationInWindow], windowRect, NO)) {
                Cocoa_HandleTitleButtonEvent(_this, event);
                return;
            }
        }
        return;

    default:
        // Ignore any other events.
        return;
    }

    mouse = SDL_GetMouse();
    data = (SDL_MouseData *)mouse->internal;
    if (!data) {
        return; // can happen when returning from fullscreen Space on shutdown
    }

    seenWarp = data->seenWarp;
    data->seenWarp = NO;

    location = [NSEvent mouseLocation];
    lastMoveX = data->lastMoveX;
    lastMoveY = data->lastMoveY;
    data->lastMoveX = location.x;
    data->lastMoveY = location.y;
    DLog("Last seen mouse: (%g, %g)", location.x, location.y);

    // Non-relative movement is handled in -[SDL3Cocoa_WindowListener mouseMoved:]
    if (!mouse->relative_mode) {
        return;
    }

    // Ignore events that aren't inside the client area (i.e. title bar.)
    if ([event window]) {
        NSRect windowRect = [[[event window] contentView] frame];
        if (!NSMouseInRect([event locationInWindow], windowRect, NO)) {
            return;
        }
    }

    deltaX = [event deltaX];
    deltaY = [event deltaY];

    if (seenWarp) {
        deltaX += (lastMoveX - data->lastWarpX);
        deltaY += ((CGDisplayPixelsHigh(kCGDirectMainDisplay) - lastMoveY) - data->lastWarpY);

        DLog("Motion was (%g, %g), offset to (%g, %g)", [event deltaX], [event deltaY], deltaX, deltaY);
    }

    SDL_SendMouseMotion(Cocoa_GetEventTimestamp([event timestamp]), mouse->focus, mouseID, true, deltaX, deltaY);
}

void Cocoa_HandleMouseWheel(SDL_Window *window, NSEvent *event)
{
    SDL_MouseID mouseID = SDL_DEFAULT_MOUSE_ID;
    SDL_MouseWheelDirection direction;
    CGFloat x, y;

    x = -[event deltaX];
    y = [event deltaY];
    direction = SDL_MOUSEWHEEL_NORMAL;

    if ([event isDirectionInvertedFromDevice] == YES) {
        direction = SDL_MOUSEWHEEL_FLIPPED;
    }

    /* For discrete scroll events from conventional mice, always send a full tick.
       For continuous scroll events from trackpads, send fractional deltas for smoother scrolling. */
    if (![event hasPreciseScrollingDeltas]) {
        if (x > 0) {
            x = SDL_ceil(x);
        } else if (x < 0) {
            x = SDL_floor(x);
        }
        if (y > 0) {
            y = SDL_ceil(y);
        } else if (y < 0) {
            y = SDL_floor(y);
        }
    }

    SDL_SendMouseWheel(Cocoa_GetEventTimestamp([event timestamp]), window, mouseID, x, y, direction);
}

void Cocoa_HandleMouseWarp(CGFloat x, CGFloat y)
{
    /* This makes Cocoa_HandleMouseEvent ignore the delta caused by the warp,
     * since it gets included in the next movement event.
     */
    SDL_MouseData *data = (SDL_MouseData *)SDL_GetMouse()->internal;
    data->lastWarpX = x;
    data->lastWarpY = y;
    data->seenWarp = true;

    DLog("(%g, %g)", x, y);
}

void Cocoa_QuitMouse(SDL_VideoDevice *_this)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    if (mouse) {
        if (mouse->internal) {
            SDL_free(mouse->internal);
            mouse->internal = NULL;
        }
    }
}

#endif // SDL_VIDEO_DRIVER_COCOA
