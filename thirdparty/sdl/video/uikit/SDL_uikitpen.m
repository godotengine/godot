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

#include "SDL_uikitevents.h"
#include "SDL_uikitpen.h"
#include "SDL_uikitwindow.h"

#include "../../events/SDL_pen_c.h"

// Fix build errors when using an older SDK by defining these selectors
#if !defined(SDL_PLATFORM_TVOS)

@interface UITouch (SDL)
#if !(__IPHONE_OS_VERSION_MAX_ALLOWED >= 170500)
@property (nonatomic, readonly) CGFloat rollAngle;
#endif
@end

@interface UIHoverGestureRecognizer (SDL)
#if !(__IPHONE_OS_VERSION_MAX_ALLOWED >= 160100)
@property (nonatomic, readonly) CGFloat zOffset;
#endif
#if !(__IPHONE_OS_VERSION_MAX_ALLOWED >= 160400)
- (CGFloat) azimuthAngleInView:(UIView *) view;

@property (nonatomic, readonly) CGFloat altitudeAngle;
#endif
#if !(__IPHONE_OS_VERSION_MAX_ALLOWED >= 170500)
@property (nonatomic, readonly) CGFloat rollAngle;
#endif
@end

#endif // !SDL_PLATFORM_TVOS

static SDL_PenID apple_pencil_id = 0;

bool UIKit_InitPen(SDL_VideoDevice *_this)
{
    return true;
}

// we only have one Apple Pencil at a time, and it must be paired to the iOS device.
// We only know about its existence when it first sends an event, so add an single SDL pen
// device here if we haven't already.
static SDL_PenID UIKit_AddPenIfNecesary()
{
    if (!apple_pencil_id) {
        SDL_PenInfo info;
        SDL_zero(info);
        info.capabilities = SDL_PEN_CAPABILITY_PRESSURE | SDL_PEN_CAPABILITY_XTILT | SDL_PEN_CAPABILITY_YTILT;
        info.max_tilt = 90.0f;
        info.num_buttons = 0;
        info.subtype = SDL_PEN_TYPE_PENCIL;

        if (@available(iOS 17.5, *)) {  // need rollAngle method.
            info.capabilities |= SDL_PEN_CAPABILITY_ROTATION;
        }

        if (@available(ios 16.1, *)) {  // need zOffset method.
            info.capabilities |= SDL_PEN_CAPABILITY_DISTANCE;
        }

        // Apple Pencil and iOS can report when the pencil is being "squeezed" but it's a boolean thing,
        // so we can't use it for tangential pressure.

        // There's only ever one Apple Pencil at most, so we just pass a non-zero value for the handle.
        apple_pencil_id = SDL_AddPenDevice(0, "Apple Pencil", &info, (void *) (size_t) 0x1);
    }

    return apple_pencil_id;
}

static void UIKit_HandlePenAxes(SDL_Window *window, NSTimeInterval nstimestamp, float zOffset, const CGPoint *point, float force,
                                float maximumPossibleForce, float azimuthAngleInView, float altitudeAngle, float rollAngle)
{
    const SDL_PenID penId = UIKit_AddPenIfNecesary();
    if (penId) {
        const Uint64 timestamp = UIKit_GetEventTimestamp(nstimestamp);
        const float radians_to_degrees = 180.0f / SDL_PI_F;

        // Normalize force to 0.0f ... 1.0f range.
        const float pressure = force / maximumPossibleForce;

        // azimuthAngleInView is in radians, with 0 being the pen's back end pointing due east on the screen when the
        // tip is touching the screen, and negative when heading north from there, positive to the south.
        // So convert to degrees, 0 being due east, etc.
        const float azimuth_angle = azimuthAngleInView * radians_to_degrees;

        // altitudeAngle is in radians, with 0 being the pen laying flat on (parallel to) the device
        //  screen and PI/2 being it pointing straight up from (perpendicular to) the device screen.
        // So convert to degrees, 0 being flat and 90 being straight up.
        const float altitude_angle = altitudeAngle * radians_to_degrees;

        // the azimuth_angle goes from -180 to 180 (with abs(angle) moving from 180 to 0, left to right), but SDL wants
        // it from -90 (back facing west) to 90 (back facing east).
        const float xtilt = (180.0f - SDL_fabsf(azimuth_angle)) - 90.0f;

        // the altitude_angle goes from 0 to 90 regardless of which direction the pen is lifting off the device, but SDL wants
        // it from -90 (flat facing north) to 90 (flat facing south).
        const float ytilt = (azimuth_angle < 0.0f) ? -(90.0f - altitude_angle) : (90.0f - altitude_angle);

        // rotation is in radians, and only available on a later iOS.
        const float rotation = rollAngle * radians_to_degrees;  // !!! FIXME: this might need adjustment, I don't have a pencil that supports it.

        SDL_SendPenMotion(timestamp, penId, window, point->x, point->y);
        SDL_SendPenAxis(timestamp, penId, window, SDL_PEN_AXIS_PRESSURE, pressure);
        SDL_SendPenAxis(timestamp, penId, window, SDL_PEN_AXIS_XTILT, xtilt);
        SDL_SendPenAxis(timestamp, penId, window, SDL_PEN_AXIS_YTILT, ytilt);
        SDL_SendPenAxis(timestamp, penId, window, SDL_PEN_AXIS_ROTATION, rotation);
        SDL_SendPenAxis(timestamp, penId, window, SDL_PEN_AXIS_DISTANCE, zOffset);
    }
}

#if !defined(SDL_PLATFORM_TVOS)
extern void UIKit_HandlePenHover(SDL_uikitview *view, UIHoverGestureRecognizer *recognizer)
{
    float zOffset = 0.0f;
    if (@available(iOS 16.1, *)) {
        zOffset = (float) [recognizer zOffset];
    }

    float azimuthAngleInView = 0.0f;
    if (@available(iOS 16.4, *)) {
        azimuthAngleInView = (float) [recognizer azimuthAngleInView:view];
    }

    float altitudeAngle = 0.0f;
    if (@available(iOS 16.4, *)) {
        altitudeAngle = (float) [recognizer altitudeAngle];
    }

    float rollAngle = 0.0f;
    if (@available(iOS 17.5, *)) {
        rollAngle = (float) [recognizer rollAngle];
    }

    SDL_Window *window = [view getSDLWindow];
    const CGPoint point = [recognizer locationInView:view];

    // force is zero here; if you're here, you're not touching.
    // !!! FIXME: no timestamp on these...?
    UIKit_HandlePenAxes(window, 0, zOffset, &point, 0.0f, 1.0f, azimuthAngleInView, altitudeAngle, rollAngle);
}
#endif

static void UIKit_HandlePenAxesFromUITouch(SDL_uikitview *view, UITouch *pencil)
{
    float rollAngle = 0.0f;
#if !defined(SDL_PLATFORM_TVOS)
    if (@available(iOS 17.5, *)) {
        rollAngle = (float) [pencil rollAngle];
    }
#endif

    SDL_Window *window = [view getSDLWindow];
    const CGPoint point = [pencil locationInView:view];

    // zOffset is zero here; if you're here, you're touching.
    UIKit_HandlePenAxes(window, [pencil timestamp], 0.0f, &point, [pencil force], [pencil maximumPossibleForce], [pencil azimuthAngleInView:view], [pencil altitudeAngle], rollAngle);
}

void UIKit_HandlePenMotion(SDL_uikitview *view, UITouch *pencil)
{
    UIKit_HandlePenAxesFromUITouch(view, pencil);
}

void UIKit_HandlePenPress(SDL_uikitview *view, UITouch *pencil)
{
    const SDL_PenID penId = UIKit_AddPenIfNecesary();
    if (penId) {
        UIKit_HandlePenAxesFromUITouch(view, pencil);
        SDL_SendPenTouch(UIKit_GetEventTimestamp([pencil timestamp]), penId, [view getSDLWindow], false, true);
    }
}

void UIKit_HandlePenRelease(SDL_uikitview *view, UITouch *pencil)
{
    const SDL_PenID penId = UIKit_AddPenIfNecesary();
    if (penId) {
        SDL_SendPenTouch(UIKit_GetEventTimestamp([pencil timestamp]), penId, [view getSDLWindow], false, false);
        UIKit_HandlePenAxesFromUITouch(view, pencil);
    }
}

void UIKit_QuitPen(SDL_VideoDevice *_this)
{
    if (apple_pencil_id) {
        SDL_RemovePenDevice(0, apple_pencil_id);
        apple_pencil_id = 0;
    }
}

#endif // SDL_VIDEO_DRIVER_UIKIT
