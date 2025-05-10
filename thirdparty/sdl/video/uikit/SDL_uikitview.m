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

#ifdef SDL_VIDEO_DRIVER_UIKIT

#include "SDL_uikitview.h"

#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_touch_c.h"
#include "../../events/SDL_events_c.h"

#include "SDL_uikitappdelegate.h"
#include "SDL_uikitevents.h"
#include "SDL_uikitmodes.h"
#include "SDL_uikitpen.h"
#include "SDL_uikitwindow.h"

// The maximum number of mouse buttons we support
#define MAX_MOUSE_BUTTONS 5

// This is defined in SDL_sysjoystick.m
#ifndef SDL_JOYSTICK_DISABLED
extern int SDL_AppleTVRemoteOpenedAsJoystick;
#endif

@implementation SDL_uikitview
{
    SDL_Window *sdlwindow;

    SDL_TouchID directTouchId;
    SDL_TouchID indirectTouchId;

#if !defined(SDL_PLATFORM_TVOS)
    UIPointerInteraction *indirectPointerInteraction API_AVAILABLE(ios(13.4));
#endif
}

- (instancetype)initWithFrame:(CGRect)frame
{
    if ((self = [super initWithFrame:frame])) {
#ifdef SDL_PLATFORM_TVOS
        // Apple TV Remote touchpad swipe gestures.
        UISwipeGestureRecognizer *swipeUp = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(swipeGesture:)];
        swipeUp.direction = UISwipeGestureRecognizerDirectionUp;
        [self addGestureRecognizer:swipeUp];

        UISwipeGestureRecognizer *swipeDown = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(swipeGesture:)];
        swipeDown.direction = UISwipeGestureRecognizerDirectionDown;
        [self addGestureRecognizer:swipeDown];

        UISwipeGestureRecognizer *swipeLeft = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(swipeGesture:)];
        swipeLeft.direction = UISwipeGestureRecognizerDirectionLeft;
        [self addGestureRecognizer:swipeLeft];

        UISwipeGestureRecognizer *swipeRight = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(swipeGesture:)];
        swipeRight.direction = UISwipeGestureRecognizerDirectionRight;
        [self addGestureRecognizer:swipeRight];
#endif

        self.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
        self.autoresizesSubviews = YES;

        directTouchId = 1;
        indirectTouchId = 2;

#ifndef SDL_PLATFORM_TVOS
        self.multipleTouchEnabled = YES;
        SDL_AddTouch(directTouchId, SDL_TOUCH_DEVICE_DIRECT, "");

        if (@available(iOS 13.0, *)) {
            UIHoverGestureRecognizer *pencilRecognizer = [[UIHoverGestureRecognizer alloc] initWithTarget:self action:@selector(pencilHovering:)];
            pencilRecognizer.allowedTouchTypes = @[@(UITouchTypePencil)];
            [self addGestureRecognizer:pencilRecognizer];
        }

        if (@available(iOS 13.4, *)) {
            indirectPointerInteraction = [[UIPointerInteraction alloc] initWithDelegate:self];
            [self addInteraction:indirectPointerInteraction];

            UIHoverGestureRecognizer *indirectPointerRecognizer = [[UIHoverGestureRecognizer alloc] initWithTarget:self action:@selector(indirectPointerHovering:)];
            indirectPointerRecognizer.allowedTouchTypes = @[@(UITouchTypeIndirectPointer)];
            [self addGestureRecognizer:indirectPointerRecognizer];
        }
#endif // !defined(SDL_PLATFORM_TVOS)
    }

    return self;
}

- (void)setSDLWindow:(SDL_Window *)window
{
    SDL_UIKitWindowData *data = nil;

    if (window == sdlwindow) {
        return;
    }

    // Remove ourself from the old window.
    if (sdlwindow) {
        SDL_uikitview *view = nil;
        data = (__bridge SDL_UIKitWindowData *)sdlwindow->internal;

        [data.views removeObject:self];

        [self removeFromSuperview];

        // Restore the next-oldest view in the old window.
        view = data.views.lastObject;

        data.viewcontroller.view = view;

        data.uiwindow.rootViewController = nil;
        data.uiwindow.rootViewController = data.viewcontroller;

        [data.uiwindow layoutIfNeeded];
    }

    sdlwindow = window;

    // Add ourself to the new window.
    if (window) {
        data = (__bridge SDL_UIKitWindowData *)window->internal;

        // Make sure the SDL window has a strong reference to this view.
        [data.views addObject:self];

        // Replace the view controller's old view with this one.
        [data.viewcontroller.view removeFromSuperview];
        data.viewcontroller.view = self;

        /* The root view controller handles rotation and the status bar.
         * Assigning it also adds the controller's view to the window. We
         * explicitly re-set it to make sure the view is properly attached to
         * the window. Just adding the sub-view if the root view controller is
         * already correct causes orientation issues on iOS 7 and below. */
        data.uiwindow.rootViewController = nil;
        data.uiwindow.rootViewController = data.viewcontroller;

        /* The view's bounds may not be correct until the next event cycle. That
         * might happen after the current dimensions are queried, so we force a
         * layout now to immediately update the bounds. */
        [data.uiwindow layoutIfNeeded];
    }
}

- (SDL_Window *)getSDLWindow
{
    return sdlwindow;
}

#if !defined(SDL_PLATFORM_TVOS)

- (UIPointerRegion *)pointerInteraction:(UIPointerInteraction *)interaction regionForRequest:(UIPointerRegionRequest *)request defaultRegion:(UIPointerRegion *)defaultRegion API_AVAILABLE(ios(13.4))
{
    return [UIPointerRegion regionWithRect:self.bounds identifier:nil];
}

- (UIPointerStyle *)pointerInteraction:(UIPointerInteraction *)interaction styleForRegion:(UIPointerRegion *)region API_AVAILABLE(ios(13.4))
{
    if (SDL_CursorVisible()) {
        return nil;
    } else {
        return [UIPointerStyle hiddenPointerStyle];
    }
}

- (void)indirectPointerHovering:(UIHoverGestureRecognizer *)recognizer API_AVAILABLE(ios(13.4))
{
    switch (recognizer.state) {
        case UIGestureRecognizerStateBegan:
        case UIGestureRecognizerStateChanged:
        {
            CGPoint point = [recognizer locationInView:self];
            SDL_SendMouseMotion(0, sdlwindow, SDL_GLOBAL_MOUSE_ID, false, point.x, point.y);
            break;
        }

        default:
            break;
    }
}

- (void)indirectPointerMoving:(UITouch *)touch API_AVAILABLE(ios(13.4))
{
    CGPoint locationInView = [self touchLocation:touch shouldNormalize:NO];
    SDL_SendMouseMotion(0, sdlwindow, SDL_GLOBAL_MOUSE_ID, false, locationInView.x, locationInView.y);
}

- (void)indirectPointerPressed:(UITouch *)touch fromEvent:(UIEvent *)event API_AVAILABLE(ios(13.4))
{
    if (!SDL_HasMouse()) {
        int i;

        for (i = 1; i <= MAX_MOUSE_BUTTONS; ++i) {
            if (event.buttonMask & SDL_BUTTON_MASK(i)) {
                Uint8 button;

                switch (i) {
                case 1:
                    button = SDL_BUTTON_LEFT;
                    break;
                case 2:
                    button = SDL_BUTTON_RIGHT;
                    break;
                case 3:
                    button = SDL_BUTTON_MIDDLE;
                    break;
                default:
                    button = (Uint8)i;
                    break;
                }
                SDL_SendMouseButton(UIKit_GetEventTimestamp([touch timestamp]), sdlwindow, SDL_GLOBAL_MOUSE_ID, button, true);
            }
        }
    }
}

- (void)indirectPointerReleased:(UITouch *)touch fromEvent:(UIEvent *)event API_AVAILABLE(ios(13.4))
{
    if (!SDL_HasMouse()) {
        int i;
        SDL_MouseButtonFlags buttons = SDL_GetMouseState(NULL, NULL);

        for (i = 1; i <= MAX_MOUSE_BUTTONS; ++i) {
            if (buttons & SDL_BUTTON_MASK(i)) {
                SDL_SendMouseButton(UIKit_GetEventTimestamp([touch timestamp]), sdlwindow, SDL_GLOBAL_MOUSE_ID, (Uint8)i, false);
            }
        }
    }
}

- (void)pencilHovering:(UIHoverGestureRecognizer *)recognizer API_AVAILABLE(ios(13.0))
{
    switch (recognizer.state) {
        case UIGestureRecognizerStateBegan:
        case UIGestureRecognizerStateChanged:
            UIKit_HandlePenHover(self, recognizer);
            break;

        case UIGestureRecognizerStateEnded:
        case UIGestureRecognizerStateCancelled:
            // we track touches elsewhere, so if a hover "ends" we'll deal with that there.
            break;

        default:
            break;
    }
}

- (void)pencilMoving:(UITouch *)touch
{
    UIKit_HandlePenMotion(self, touch);
}

- (void)pencilPressed:(UITouch *)touch
{
    UIKit_HandlePenPress(self, touch);
}

- (void)pencilReleased:(UITouch *)touch
{
    UIKit_HandlePenRelease(self, touch);
}

#endif // !defined(SDL_PLATFORM_TVOS)

- (SDL_TouchDeviceType)touchTypeForTouch:(UITouch *)touch
{
    if (touch.type == UITouchTypeIndirect) {
        return SDL_TOUCH_DEVICE_INDIRECT_RELATIVE;
    }
    return SDL_TOUCH_DEVICE_DIRECT;
}

- (SDL_TouchID)touchIdForType:(SDL_TouchDeviceType)type
{
    switch (type) {
    case SDL_TOUCH_DEVICE_DIRECT:
    default:
        return directTouchId;
    case SDL_TOUCH_DEVICE_INDIRECT_RELATIVE:
        return indirectTouchId;
    }
}

- (CGPoint)touchLocation:(UITouch *)touch shouldNormalize:(BOOL)normalize
{
    CGPoint point = [touch locationInView:self];

    if (normalize) {
        CGRect bounds = self.bounds;
        point.x /= bounds.size.width;
        point.y /= bounds.size.height;
    }

    return point;
}

- (float)pressureForTouch:(UITouch *)touch
{
    return (float)touch.force;
}

- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event
{
    for (UITouch *touch in touches) {
#if !defined(SDL_PLATFORM_TVOS)
        if (@available(iOS 13.0, *)) {
            if (touch.type == UITouchTypePencil) {
                [self pencilPressed:touch];
                continue;
            }
        }

        if (@available(iOS 13.4, *)) {
            if (touch.type == UITouchTypeIndirectPointer) {
                [self indirectPointerPressed:touch fromEvent:event];
                continue;
            }
        }
#endif // !defined(SDL_PLATFORM_TVOS)

        SDL_TouchDeviceType touchType = [self touchTypeForTouch:touch];
        SDL_TouchID touchId = [self touchIdForType:touchType];
        float pressure = [self pressureForTouch:touch];

        if (SDL_AddTouch(touchId, touchType, "") < 0) {
            continue;
        }

        // FIXME, need to send: int clicks = (int) touch.tapCount; ?

        CGPoint locationInView = [self touchLocation:touch shouldNormalize:YES];
        SDL_SendTouch(UIKit_GetEventTimestamp([event timestamp]),
                      touchId, (SDL_FingerID)(uintptr_t)touch, sdlwindow,
                      SDL_EVENT_FINGER_DOWN, locationInView.x, locationInView.y, pressure);
    }
}

- (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event
{
    for (UITouch *touch in touches) {
#if !defined(SDL_PLATFORM_TVOS)
        if (@available(iOS 13.0, *)) {
            if (touch.type == UITouchTypePencil) {
                [self pencilReleased:touch];
                continue;
            }
        }

        if (@available(iOS 13.4, *)) {
            if (touch.type == UITouchTypeIndirectPointer) {
                [self indirectPointerReleased:touch fromEvent:event];
                continue;
            }
        }
#endif // !defined(SDL_PLATFORM_TVOS)

        SDL_TouchDeviceType touchType = [self touchTypeForTouch:touch];
        SDL_TouchID touchId = [self touchIdForType:touchType];
        float pressure = [self pressureForTouch:touch];

        if (SDL_AddTouch(touchId, touchType, "") < 0) {
            continue;
        }

        // FIXME, need to send: int clicks = (int) touch.tapCount; ?

        CGPoint locationInView = [self touchLocation:touch shouldNormalize:YES];
        SDL_SendTouch(UIKit_GetEventTimestamp([event timestamp]),
                      touchId, (SDL_FingerID)(uintptr_t)touch, sdlwindow,
                      SDL_EVENT_FINGER_UP, locationInView.x, locationInView.y, pressure);
    }
}

- (void)touchesCancelled:(NSSet *)touches withEvent:(UIEvent *)event
{
    for (UITouch *touch in touches) {
#if !defined(SDL_PLATFORM_TVOS)
        if (@available(iOS 13.0, *)) {
            if (touch.type == UITouchTypePencil) {
                [self pencilReleased:touch];
                continue;
            }
        }

        if (@available(iOS 13.4, *)) {
            if (touch.type == UITouchTypeIndirectPointer) {
                [self indirectPointerReleased:touch fromEvent:event];
                continue;
            }
        }
#endif // !defined(SDL_PLATFORM_TVOS)

        SDL_TouchDeviceType touchType = [self touchTypeForTouch:touch];
        SDL_TouchID touchId = [self touchIdForType:touchType];
        float pressure = [self pressureForTouch:touch];

        if (SDL_AddTouch(touchId, touchType, "") < 0) {
            continue;
        }

        CGPoint locationInView = [self touchLocation:touch shouldNormalize:YES];
        SDL_SendTouch(UIKit_GetEventTimestamp([event timestamp]),
                      touchId, (SDL_FingerID)(uintptr_t)touch, sdlwindow,
                      SDL_EVENT_FINGER_CANCELED, locationInView.x, locationInView.y, pressure);
    }
}

- (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event
{
    for (UITouch *touch in touches) {
#if !defined(SDL_PLATFORM_TVOS)
        if (@available(iOS 13.0, *)) {
            if (touch.type == UITouchTypePencil) {
                [self pencilMoving:touch];
                continue;
            }
        }

        if (@available(iOS 13.4, *)) {
            if (touch.type == UITouchTypeIndirectPointer) {
                [self indirectPointerMoving:touch];
                continue;
            }
        }
#endif  // !defined(SDL_PLATFORM_TVOS)

        SDL_TouchDeviceType touchType = [self touchTypeForTouch:touch];
        SDL_TouchID touchId = [self touchIdForType:touchType];
        float pressure = [self pressureForTouch:touch];

        if (SDL_AddTouch(touchId, touchType, "") < 0) {
            continue;
        }

        CGPoint locationInView = [self touchLocation:touch shouldNormalize:YES];
        SDL_SendTouchMotion(UIKit_GetEventTimestamp([event timestamp]),
                            touchId, (SDL_FingerID)(uintptr_t)touch, sdlwindow,
                            locationInView.x, locationInView.y, pressure);
    }
}

- (void)safeAreaInsetsDidChange
{
    // Update the safe area insets
    SDL_SetWindowSafeAreaInsets(sdlwindow,
                                (int)SDL_ceilf(self.safeAreaInsets.left),
                                (int)SDL_ceilf(self.safeAreaInsets.right),
                                (int)SDL_ceilf(self.safeAreaInsets.top),
                                (int)SDL_ceilf(self.safeAreaInsets.bottom));
}

- (SDL_Scancode)scancodeFromPress:(UIPress *)press
{
    if (press.key != nil) {
        return (SDL_Scancode)press.key.keyCode;
    }

#ifndef SDL_JOYSTICK_DISABLED
    // Presses from Apple TV remote
    if (!SDL_AppleTVRemoteOpenedAsJoystick) {
        switch (press.type) {
        case UIPressTypeUpArrow:
            return SDL_SCANCODE_UP;
        case UIPressTypeDownArrow:
            return SDL_SCANCODE_DOWN;
        case UIPressTypeLeftArrow:
            return SDL_SCANCODE_LEFT;
        case UIPressTypeRightArrow:
            return SDL_SCANCODE_RIGHT;
        case UIPressTypeSelect:
            // HIG says: "primary button behavior"
            return SDL_SCANCODE_RETURN;
        case UIPressTypeMenu:
            // HIG says: "returns to previous screen"
            return SDL_SCANCODE_ESCAPE;
        case UIPressTypePlayPause:
            // HIG says: "secondary button behavior"
            return SDL_SCANCODE_PAUSE;
        default:
            break;
        }
    }
#endif // !SDL_JOYSTICK_DISABLED

    return SDL_SCANCODE_UNKNOWN;
}

- (void)pressesBegan:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event
{
    if (!SDL_HasKeyboard()) {
        for (UIPress *press in presses) {
            SDL_Scancode scancode = [self scancodeFromPress:press];
            SDL_SendKeyboardKey(UIKit_GetEventTimestamp([event timestamp]), SDL_GLOBAL_KEYBOARD_ID, 0, scancode, true);
        }
    }
    if (SDL_TextInputActive(sdlwindow)) {
        [super pressesBegan:presses withEvent:event];
    }
}

- (void)pressesEnded:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event
{
    if (!SDL_HasKeyboard()) {
        for (UIPress *press in presses) {
            SDL_Scancode scancode = [self scancodeFromPress:press];
            SDL_SendKeyboardKey(UIKit_GetEventTimestamp([event timestamp]), SDL_GLOBAL_KEYBOARD_ID, 0, scancode, false);
        }
    }
    if (SDL_TextInputActive(sdlwindow)) {
        [super pressesEnded:presses withEvent:event];
    }
}

- (void)pressesCancelled:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event
{
    if (!SDL_HasKeyboard()) {
        for (UIPress *press in presses) {
            SDL_Scancode scancode = [self scancodeFromPress:press];
            SDL_SendKeyboardKey(UIKit_GetEventTimestamp([event timestamp]), SDL_GLOBAL_KEYBOARD_ID, 0, scancode, false);
        }
    }
    if (SDL_TextInputActive(sdlwindow)) {
        [super pressesCancelled:presses withEvent:event];
    }
}

- (void)pressesChanged:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event
{
    // This is only called when the force of a press changes.
    if (SDL_TextInputActive(sdlwindow)) {
        [super pressesChanged:presses withEvent:event];
    }
}

#ifdef SDL_PLATFORM_TVOS
- (void)swipeGesture:(UISwipeGestureRecognizer *)gesture
{
    // Swipe gestures don't trigger begin states.
    if (gesture.state == UIGestureRecognizerStateEnded) {
#ifndef SDL_JOYSTICK_DISABLED
        if (!SDL_AppleTVRemoteOpenedAsJoystick) {
            /* Send arrow key presses for now, as we don't have an external API
             * which better maps to swipe gestures. */
            switch (gesture.direction) {
            case UISwipeGestureRecognizerDirectionUp:
                SDL_SendKeyboardKeyAutoRelease(0, SDL_SCANCODE_UP);
                break;
            case UISwipeGestureRecognizerDirectionDown:
                SDL_SendKeyboardKeyAutoRelease(0, SDL_SCANCODE_DOWN);
                break;
            case UISwipeGestureRecognizerDirectionLeft:
                SDL_SendKeyboardKeyAutoRelease(0, SDL_SCANCODE_LEFT);
                break;
            case UISwipeGestureRecognizerDirectionRight:
                SDL_SendKeyboardKeyAutoRelease(0, SDL_SCANCODE_RIGHT);
                break;
            }
        }
#endif // !SDL_JOYSTICK_DISABLED
    }
}
#endif // SDL_PLATFORM_TVOS

@end

#endif // SDL_VIDEO_DRIVER_UIKIT
