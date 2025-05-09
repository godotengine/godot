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

#include "SDL_cocoavideo.h"
#include "../../events/SDL_events_c.h"

// We need this for IODisplayCreateInfoDictionary and kIODisplayOnlyPreferredName
#include <IOKit/graphics/IOGraphicsLib.h>

// We need this for CVDisplayLinkGetNominalOutputVideoRefreshPeriod
#include <CoreVideo/CVBase.h>
#include <CoreVideo/CVDisplayLink.h>

#if (IOGRAPHICSTYPES_REV < 40)
#define kDisplayModeNativeFlag 0x02000000
#endif

static bool CG_SetError(const char *prefix, CGDisplayErr result)
{
    const char *error;

    switch (result) {
    case kCGErrorFailure:
        error = "kCGErrorFailure";
        break;
    case kCGErrorIllegalArgument:
        error = "kCGErrorIllegalArgument";
        break;
    case kCGErrorInvalidConnection:
        error = "kCGErrorInvalidConnection";
        break;
    case kCGErrorInvalidContext:
        error = "kCGErrorInvalidContext";
        break;
    case kCGErrorCannotComplete:
        error = "kCGErrorCannotComplete";
        break;
    case kCGErrorNotImplemented:
        error = "kCGErrorNotImplemented";
        break;
    case kCGErrorRangeCheck:
        error = "kCGErrorRangeCheck";
        break;
    case kCGErrorTypeCheck:
        error = "kCGErrorTypeCheck";
        break;
    case kCGErrorInvalidOperation:
        error = "kCGErrorInvalidOperation";
        break;
    case kCGErrorNoneAvailable:
        error = "kCGErrorNoneAvailable";
        break;
    default:
        error = "Unknown Error";
        break;
    }
    return SDL_SetError("%s: %s", prefix, error);
}

static NSScreen *GetNSScreenForDisplayID(CGDirectDisplayID displayID)
{
    NSArray *screens = [NSScreen screens];

    // !!! FIXME: maybe track the NSScreen in SDL_DisplayData?
    for (NSScreen *screen in screens) {
        const CGDirectDisplayID thisDisplay = (CGDirectDisplayID)[[[screen deviceDescription] objectForKey:@"NSScreenNumber"] unsignedIntValue];
        if (thisDisplay == displayID) {
            return screen;
        }
    }
    return nil;
}

SDL_VideoDisplay *Cocoa_FindSDLDisplayByCGDirectDisplayID(SDL_VideoDevice *_this, CGDirectDisplayID displayid)
{
    for (int i = 0; i < _this->num_displays; i++) {
        const SDL_DisplayData *displaydata = _this->displays[i]->internal;
        if (displaydata && (displaydata->display == displayid)) {
            return _this->displays[i];
        }
    }
    return NULL;
}

static float GetDisplayModeRefreshRate(CGDisplayModeRef vidmode, CVDisplayLinkRef link)
{
    float refreshRate = (float)CGDisplayModeGetRefreshRate(vidmode);

    // CGDisplayModeGetRefreshRate can return 0 (eg for built-in displays).
    if (refreshRate == 0 && link != NULL) {
        CVTime time = CVDisplayLinkGetNominalOutputVideoRefreshPeriod(link);
        if ((time.flags & kCVTimeIsIndefinite) == 0 && time.timeValue != 0) {
            refreshRate = (float)time.timeScale / time.timeValue;
        }
    }

    return refreshRate;
}

static bool HasValidDisplayModeFlags(CGDisplayModeRef vidmode)
{
    uint32_t ioflags = CGDisplayModeGetIOFlags(vidmode);

    // Filter out modes which have flags that we don't want.
    if (ioflags & (kDisplayModeNeverShowFlag | kDisplayModeNotGraphicsQualityFlag)) {
        return false;
    }

    // Filter out modes which don't have flags that we want.
    if (!(ioflags & kDisplayModeValidFlag) || !(ioflags & kDisplayModeSafeFlag)) {
        return false;
    }

    return true;
}

static Uint32 GetDisplayModePixelFormat(CGDisplayModeRef vidmode)
{
    // This API is deprecated in 10.11 with no good replacement (as of 10.15).
    CFStringRef fmt = CGDisplayModeCopyPixelEncoding(vidmode);
    Uint32 pixelformat = SDL_PIXELFORMAT_UNKNOWN;

    if (CFStringCompare(fmt, CFSTR(IO32BitDirectPixels),
                        kCFCompareCaseInsensitive) == kCFCompareEqualTo) {
        pixelformat = SDL_PIXELFORMAT_ARGB8888;
    } else if (CFStringCompare(fmt, CFSTR(IO16BitDirectPixels),
                               kCFCompareCaseInsensitive) == kCFCompareEqualTo) {
        pixelformat = SDL_PIXELFORMAT_ARGB1555;
    } else if (CFStringCompare(fmt, CFSTR(kIO30BitDirectPixels),
                               kCFCompareCaseInsensitive) == kCFCompareEqualTo) {
        pixelformat = SDL_PIXELFORMAT_ARGB2101010;
    } else {
        // ignore 8-bit and such for now.
    }

    CFRelease(fmt);

    return pixelformat;
}

static bool GetDisplayMode(CGDisplayModeRef vidmode, bool vidmodeCurrent, CFArrayRef modelist, CVDisplayLinkRef link, SDL_DisplayMode *mode)
{
    SDL_DisplayModeData *data;
    bool usableForGUI = CGDisplayModeIsUsableForDesktopGUI(vidmode);
    size_t width = CGDisplayModeGetWidth(vidmode);
    size_t height = CGDisplayModeGetHeight(vidmode);
    size_t pixelW = width;
    size_t pixelH = height;
    uint32_t ioflags = CGDisplayModeGetIOFlags(vidmode);
    float refreshrate = GetDisplayModeRefreshRate(vidmode, link);
    Uint32 format = GetDisplayModePixelFormat(vidmode);
    bool interlaced = (ioflags & kDisplayModeInterlacedFlag) != 0;
    CFMutableArrayRef modes;

    if (format == SDL_PIXELFORMAT_UNKNOWN) {
        return false;
    }

    /* Don't fail the current mode based on flags because this could prevent Cocoa_InitModes from
     * succeeding if the current mode lacks certain flags (esp kDisplayModeSafeFlag). */
    if (!vidmodeCurrent && !HasValidDisplayModeFlags(vidmode)) {
        return false;
    }

    modes = CFArrayCreateMutable(NULL, 0, &kCFTypeArrayCallBacks);
    CFArrayAppendValue(modes, vidmode);

    /* If a list of possible display modes is passed in, use it to filter out
     * modes that have duplicate sizes. We don't just rely on SDL's higher level
     * duplicate filtering because this code can choose what properties are
     * preferred, and it can add CGDisplayModes to the DisplayModeData's list of
     * modes to try (see comment below for why that's necessary). */
    pixelW = CGDisplayModeGetPixelWidth(vidmode);
    pixelH = CGDisplayModeGetPixelHeight(vidmode);

    if (modelist != NULL) {
        CFIndex modescount = CFArrayGetCount(modelist);
        int i;

        for (i = 0; i < modescount; i++) {
            size_t otherW, otherH, otherpixelW, otherpixelH;
            float otherrefresh;
            Uint32 otherformat;
            bool otherGUI;
            CGDisplayModeRef othermode = (CGDisplayModeRef)CFArrayGetValueAtIndex(modelist, i);
            uint32_t otherioflags = CGDisplayModeGetIOFlags(othermode);

            if (CFEqual(vidmode, othermode)) {
                continue;
            }

            if (!HasValidDisplayModeFlags(othermode)) {
                continue;
            }

            otherW = CGDisplayModeGetWidth(othermode);
            otherH = CGDisplayModeGetHeight(othermode);
            otherpixelW = CGDisplayModeGetPixelWidth(othermode);
            otherpixelH = CGDisplayModeGetPixelHeight(othermode);
            otherrefresh = GetDisplayModeRefreshRate(othermode, link);
            otherformat = GetDisplayModePixelFormat(othermode);
            otherGUI = CGDisplayModeIsUsableForDesktopGUI(othermode);

            /* Ignore this mode if it's interlaced and there's a non-interlaced
             * mode in the list with the same properties.
             */
            if (interlaced && ((otherioflags & kDisplayModeInterlacedFlag) == 0) && width == otherW && height == otherH && pixelW == otherpixelW && pixelH == otherpixelH && refreshrate == otherrefresh && format == otherformat && usableForGUI == otherGUI) {
                CFRelease(modes);
                return false;
            }

            /* Ignore this mode if it's not usable for desktop UI and its
             * properties are equal to another GUI-capable mode in the list.
             */
            if (width == otherW && height == otherH && pixelW == otherpixelW && pixelH == otherpixelH && !usableForGUI && otherGUI && refreshrate == otherrefresh && format == otherformat) {
                CFRelease(modes);
                return false;
            }

            /* If multiple modes have the exact same properties, they'll all
             * go in the list of modes to try when SetDisplayMode is called.
             * This is needed because kCGDisplayShowDuplicateLowResolutionModes
             * (which is used to expose highdpi display modes) can make the
             * list of modes contain duplicates (according to their properties
             * obtained via public APIs) which don't work with SetDisplayMode.
             * Those duplicate non-functional modes *do* have different pixel
             * formats according to their internal data structure viewed with
             * NSLog, but currently no public API can detect that.
             * https://bugzilla.libsdl.org/show_bug.cgi?id=4822
             *
             * As of macOS 10.15.0, those duplicates have the exact same
             * properties via public APIs in every way (even their IO flags and
             * CGDisplayModeGetIODisplayModeID is the same), so we could test
             * those for equality here too, but I'm intentionally not doing that
             * in case there are duplicate modes with different IO flags or IO
             * display mode IDs in the future. In that case I think it's better
             * to try them all in SetDisplayMode than to risk one of them being
             * correct but it being filtered out by SDL_AddFullscreenDisplayMode
             * as being a duplicate.
             */
            if (width == otherW && height == otherH && pixelW == otherpixelW && pixelH == otherpixelH && usableForGUI == otherGUI && refreshrate == otherrefresh && format == otherformat) {
                CFArrayAppendValue(modes, othermode);
            }
        }
    }

    SDL_zerop(mode);
    data = (SDL_DisplayModeData *)SDL_malloc(sizeof(*data));
    if (!data) {
        CFRelease(modes);
        return false;
    }
    data->modes = modes;
    mode->format = format;
    mode->w = (int)width;
    mode->h = (int)height;
    mode->pixel_density = (float)pixelW / width;
    mode->refresh_rate = refreshrate;
    mode->internal = data;
    return true;
}

static char *Cocoa_GetDisplayName(CGDirectDisplayID displayID)
{
    if (@available(macOS 10.15, *)) {
        NSScreen *screen = GetNSScreenForDisplayID(displayID);
        if (screen) {
            const char *name = [screen.localizedName UTF8String];
            if (name) {
                return SDL_strdup(name);
            }
        }
    }

    // This API is deprecated in 10.9 with no good replacement (as of 10.15).
    io_service_t servicePort = CGDisplayIOServicePort(displayID);
    CFDictionaryRef deviceInfo = IODisplayCreateInfoDictionary(servicePort, kIODisplayOnlyPreferredName);
    NSDictionary *localizedNames = [(__bridge NSDictionary *)deviceInfo objectForKey:[NSString stringWithUTF8String:kDisplayProductName]];
    char *displayName = NULL;

    if ([localizedNames count] > 0) {
        displayName = SDL_strdup([[localizedNames objectForKey:[[localizedNames allKeys] objectAtIndex:0]] UTF8String]);
    }
    CFRelease(deviceInfo);
    return displayName;
}

static void Cocoa_GetHDRProperties(CGDirectDisplayID displayID, SDL_HDROutputProperties *HDR)
{
    HDR->SDR_white_level = 1.0f;
    HDR->HDR_headroom = 1.0f;

    if (@available(macOS 10.15, *)) {
        NSScreen *screen = GetNSScreenForDisplayID(displayID);
        if (screen) {
            if (screen.maximumExtendedDynamicRangeColorComponentValue > 1.0f) {
                HDR->HDR_headroom = screen.maximumExtendedDynamicRangeColorComponentValue;
            } else {
                HDR->HDR_headroom = screen.maximumPotentialExtendedDynamicRangeColorComponentValue;
            }
        }
    }
}


bool Cocoa_AddDisplay(CGDirectDisplayID display, bool send_event)
{
    CGDisplayModeRef moderef = CGDisplayCopyDisplayMode(display);
    if (!moderef) {
        return false;
    }

    SDL_DisplayData *displaydata = (SDL_DisplayData *)SDL_malloc(sizeof(*displaydata));
    if (!displaydata) {
        CGDisplayModeRelease(moderef);
        return false;
    }
    displaydata->display = display;

    CVDisplayLinkRef link = NULL;
    CVDisplayLinkCreateWithCGDisplay(display, &link);

    SDL_VideoDisplay viddisplay;
    SDL_zero(viddisplay);
    viddisplay.name = Cocoa_GetDisplayName(display); // this returns a strdup'ed string

    SDL_DisplayMode mode;
    if (!GetDisplayMode(moderef, true, NULL, link, &mode)) {
        CVDisplayLinkRelease(link);
        CGDisplayModeRelease(moderef);
        SDL_free(viddisplay.name);
        SDL_free(displaydata);
        return false;
    }

    CVDisplayLinkRelease(link);
    CGDisplayModeRelease(moderef);

    Cocoa_GetHDRProperties(displaydata->display, &viddisplay.HDR);

    viddisplay.desktop_mode = mode;
    viddisplay.internal = displaydata;
    const bool retval = SDL_AddVideoDisplay(&viddisplay, send_event);
    SDL_free(viddisplay.name);
    return retval;
}

static void Cocoa_DisplayReconfigurationCallback(CGDirectDisplayID displayid, CGDisplayChangeSummaryFlags flags, void *userInfo)
{
    #if 0
    SDL_Log("COCOA DISPLAY RECONFIG CALLBACK! display=%u", (unsigned int) displayid);
    #define CHECK_DISPLAY_RECONFIG_FLAG(x) if (flags & x) { SDL_Log(" - " #x); }
    CHECK_DISPLAY_RECONFIG_FLAG(kCGDisplayBeginConfigurationFlag);
    CHECK_DISPLAY_RECONFIG_FLAG(kCGDisplayMovedFlag);
    CHECK_DISPLAY_RECONFIG_FLAG(kCGDisplaySetMainFlag);
    CHECK_DISPLAY_RECONFIG_FLAG(kCGDisplaySetModeFlag);
    CHECK_DISPLAY_RECONFIG_FLAG(kCGDisplayAddFlag);
    CHECK_DISPLAY_RECONFIG_FLAG(kCGDisplayRemoveFlag);
    CHECK_DISPLAY_RECONFIG_FLAG(kCGDisplayEnabledFlag);
    CHECK_DISPLAY_RECONFIG_FLAG(kCGDisplayDisabledFlag);
    CHECK_DISPLAY_RECONFIG_FLAG(kCGDisplayMirrorFlag);
    CHECK_DISPLAY_RECONFIG_FLAG(kCGDisplayUnMirrorFlag);
    CHECK_DISPLAY_RECONFIG_FLAG(kCGDisplayDesktopShapeChangedFlag);
    #undef CHECK_DISPLAY_RECONFIG_FLAG
    #endif

    SDL_VideoDevice *_this = (SDL_VideoDevice *) userInfo;
    SDL_VideoDisplay *display = Cocoa_FindSDLDisplayByCGDirectDisplayID(_this, displayid);  // will be NULL for newly-added (or newly-unmirrored) displays!

    if (flags & kCGDisplayDisabledFlag) {
        flags |= kCGDisplayRemoveFlag;  // treat this like a display leaving, even though it's still plugged in.
    }

    if (flags & kCGDisplayEnabledFlag) {
        flags |= kCGDisplayAddFlag;  // treat this like a display leaving, even though it's still plugged in.
    }

    if (flags & kCGDisplayMirrorFlag) {
        flags |= kCGDisplayRemoveFlag;  // treat this like a display leaving, even though it's still actually here.
    }

    if (flags & kCGDisplayUnMirrorFlag) {
        flags |= kCGDisplayAddFlag;  // treat this like a new display arriving, even though it was here all along.
    }

    if ((flags & kCGDisplayAddFlag) && (flags & kCGDisplayRemoveFlag)) {
        // sometimes you get a removed device that gets Add and Remove flags at the same time but the display dimensions are 0x0 or 1x1, hence the `> 1` test.
        // Mirrored things are always removed, since they don't represent a discrete display in this state.
        if (((flags & kCGDisplayMirrorFlag) == 0) && (CGDisplayPixelsWide(displayid) > 1)) {
            // Final state is connected
            flags &= ~kCGDisplayRemoveFlag;
        } else {
            // Final state is disconnected
            flags &= ~kCGDisplayAddFlag;
        }
    }

    if (flags & kCGDisplayAddFlag) {
        if (!display) {
            if (!Cocoa_AddDisplay(displayid, true)) {
                return;  // oh well.
            }
            display = Cocoa_FindSDLDisplayByCGDirectDisplayID(_this, displayid);
            SDL_assert(display != NULL);
        }
    }

    if (flags & kCGDisplayRemoveFlag) {
        if (display) {
            SDL_DelVideoDisplay(display->id, true);
            display = NULL;
        }
    }

    if (flags & kCGDisplaySetModeFlag) {
        if (display) {
            CGDisplayModeRef moderef = CGDisplayCopyDisplayMode(displayid);
            if (moderef) {
                CVDisplayLinkRef link = NULL;
                CVDisplayLinkCreateWithCGDisplay(displayid, &link);
                if (link) {
                    SDL_DisplayMode mode;
                    if (GetDisplayMode(moderef, true, NULL, link, &mode)) {
                        SDL_SetDesktopDisplayMode(display, &mode);
                    }
                    CVDisplayLinkRelease(link);
                }
                CGDisplayModeRelease(moderef);
            }
        }
    }

    if (flags & kCGDisplaySetMainFlag) {
        if (display) {
            for (int i = 0; i < _this->num_displays; i++) {
                if (_this->displays[i] == display) {
                    if (i > 0) {
                        // move this display to the front of _this->displays so it's treated as primary.
                        SDL_memmove(&_this->displays[1], &_this->displays[0], sizeof (*_this->displays) * i);
                        _this->displays[0] = display;
                    }
                    flags |= kCGDisplayMovedFlag;  // we don't have an SDL event atm for "this display became primary," so at least let everyone know it "moved".
                    break;
                }
            }
        }
    }

    if (flags & kCGDisplayMovedFlag) {
        if (display) {
            SDL_SendDisplayEvent(display, SDL_EVENT_DISPLAY_MOVED, 0, 0);
        }
    }

    if (flags & kCGDisplayDesktopShapeChangedFlag) {
        SDL_UpdateDesktopBounds();
    }
}

void Cocoa_InitModes(SDL_VideoDevice *_this)
{
    @autoreleasepool {
        CGDisplayErr result;
        CGDisplayCount numDisplays = 0;

        result = CGGetOnlineDisplayList(0, NULL, &numDisplays);
        if (result != kCGErrorSuccess) {
            CG_SetError("CGGetOnlineDisplayList()", result);
            return;
        }

        bool isstack;
        CGDirectDisplayID *displays = SDL_small_alloc(CGDirectDisplayID, numDisplays, &isstack);

        result = CGGetOnlineDisplayList(numDisplays, displays, &numDisplays);
        if (result != kCGErrorSuccess) {
            CG_SetError("CGGetOnlineDisplayList()", result);
            SDL_small_free(displays, isstack);
            return;
        }

        // future updates to the display graph will come through this callback.
        CGDisplayRegisterReconfigurationCallback(Cocoa_DisplayReconfigurationCallback, _this);

        // Pick up the primary display in the first pass, then get the rest
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < numDisplays; ++i) {
                if (pass == 0) {
                    if (!CGDisplayIsMain(displays[i])) {
                        continue;
                    }
                } else {
                    if (CGDisplayIsMain(displays[i])) {
                        continue;
                    }
                }

                if (CGDisplayMirrorsDisplay(displays[i]) != kCGNullDirectDisplay) {
                    continue;
                }

                Cocoa_AddDisplay(displays[i], false);
            }
        }
        SDL_small_free(displays, isstack);
    }
}

void Cocoa_UpdateDisplays(SDL_VideoDevice *_this)
{
    SDL_HDROutputProperties HDR;
    int i;

    for (i = 0; i < _this->num_displays; ++i) {
        SDL_VideoDisplay *display = _this->displays[i];
        SDL_DisplayData *displaydata = (SDL_DisplayData *)display->internal;

        Cocoa_GetHDRProperties(displaydata->display, &HDR);
        SDL_SetDisplayHDRProperties(display, &HDR);
    }
}

bool Cocoa_GetDisplayBounds(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_Rect *rect)
{
    SDL_DisplayData *displaydata = (SDL_DisplayData *)display->internal;
    CGRect cgrect;

    cgrect = CGDisplayBounds(displaydata->display);
    rect->x = (int)cgrect.origin.x;
    rect->y = (int)cgrect.origin.y;
    rect->w = (int)cgrect.size.width;
    rect->h = (int)cgrect.size.height;
    return true;
}

bool Cocoa_GetDisplayUsableBounds(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_Rect *rect)
{
    @autoreleasepool {
        SDL_DisplayData *displaydata = (SDL_DisplayData *)display->internal;
        NSScreen *screen = GetNSScreenForDisplayID(displaydata->display);

        if (screen == nil) {
            return SDL_SetError("Couldn't get NSScreen for display");
        }

        {
            const NSRect frame = [screen visibleFrame];
            rect->x = (int)frame.origin.x;
            rect->y = (int)(CGDisplayPixelsHigh(kCGDirectMainDisplay) - frame.origin.y - frame.size.height);
            rect->w = (int)frame.size.width;
            rect->h = (int)frame.size.height;
        }

        return true;
    }
}

bool Cocoa_GetDisplayModes(SDL_VideoDevice *_this, SDL_VideoDisplay *display)
{
    SDL_DisplayData *data = (SDL_DisplayData *)display->internal;
    CVDisplayLinkRef link = NULL;
    CFArrayRef modes;
    CFDictionaryRef dict = NULL;
    const CFStringRef dictkeys[] = { kCGDisplayShowDuplicateLowResolutionModes };
    const CFBooleanRef dictvalues[] = { kCFBooleanTrue };

    CVDisplayLinkCreateWithCGDisplay(data->display, &link);

    /* By default, CGDisplayCopyAllDisplayModes will only get a subset of the
     * system's available modes. For example on a 15" 2016 MBP, users can
     * choose 1920x1080@2x in System Preferences but it won't show up here,
     * unless we specify the option below.
     * The display modes returned by CGDisplayCopyAllDisplayModes are also not
     * high dpi-capable unless this option is set.
     * macOS 10.15 also seems to have a bug where entering, exiting, and
     * re-entering exclusive fullscreen with a low dpi display mode can cause
     * the content of the screen to move up, which this setting avoids:
     * https://bugzilla.libsdl.org/show_bug.cgi?id=4822
     */

    dict = CFDictionaryCreate(NULL,
                              (const void **)dictkeys,
                              (const void **)dictvalues,
                              1,
                              &kCFCopyStringDictionaryKeyCallBacks,
                              &kCFTypeDictionaryValueCallBacks);

    modes = CGDisplayCopyAllDisplayModes(data->display, dict);

    if (dict) {
        CFRelease(dict);
    }

    if (modes) {
        CFIndex i;
        const CFIndex count = CFArrayGetCount(modes);

        for (i = 0; i < count; i++) {
            CGDisplayModeRef moderef = (CGDisplayModeRef)CFArrayGetValueAtIndex(modes, i);
            SDL_DisplayMode mode;

            if (GetDisplayMode(moderef, false, modes, link, &mode)) {
                if (!SDL_AddFullscreenDisplayMode(display, &mode)) {
                    CFRelease(mode.internal->modes);
                    SDL_free(mode.internal);
                }
            }
        }

        CFRelease(modes);
    }

    CVDisplayLinkRelease(link);
    return true;
}

static CGError SetDisplayModeForDisplay(CGDirectDisplayID display, SDL_DisplayModeData *data)
{
    /* SDL_DisplayModeData can contain multiple CGDisplayModes to try (with
     * identical properties), some of which might not work. See GetDisplayMode.
     */
    CGError result = kCGErrorFailure;
    for (CFIndex i = 0; i < CFArrayGetCount(data->modes); i++) {
        CGDisplayModeRef moderef = (CGDisplayModeRef)CFArrayGetValueAtIndex(data->modes, i);
        result = CGDisplaySetDisplayMode(display, moderef, NULL);
        if (result == kCGErrorSuccess) {
            // If this mode works, try it first next time.
            if (i > 0) {
                CFArrayExchangeValuesAtIndices(data->modes, i, 0);
            }
            break;
        }
    }
    return result;
}

bool Cocoa_SetDisplayMode(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_DisplayMode *mode)
{
    SDL_DisplayData *displaydata = (SDL_DisplayData *)display->internal;
    SDL_DisplayModeData *data = mode->internal;
    CGDisplayFadeReservationToken fade_token = kCGDisplayFadeReservationInvalidToken;
    CGError result = kCGErrorSuccess;

    b_inModeTransition = true;

    // Fade to black to hide resolution-switching flicker
    if (CGAcquireDisplayFadeReservation(5, &fade_token) == kCGErrorSuccess) {
        CGDisplayFade(fade_token, 0.3, kCGDisplayBlendNormal, kCGDisplayBlendSolidColor, 0.0, 0.0, 0.0, TRUE);
    }

    if (data == display->desktop_mode.internal) {
        // Restoring desktop mode
        SetDisplayModeForDisplay(displaydata->display, data);
    } else {
        // Do the physical switch
        result = SetDisplayModeForDisplay(displaydata->display, data);
    }

    // Fade in again (asynchronously)
    if (fade_token != kCGDisplayFadeReservationInvalidToken) {
        CGDisplayFade(fade_token, 0.5, kCGDisplayBlendSolidColor, kCGDisplayBlendNormal, 0.0, 0.0, 0.0, FALSE);
        CGReleaseDisplayFadeReservation(fade_token);
    }

    b_inModeTransition = false;

    if (result != kCGErrorSuccess) {
        return CG_SetError("CGDisplaySwitchToMode()", result);
    }
    return true;
}

void Cocoa_QuitModes(SDL_VideoDevice *_this)
{
    int i, j;

    CGDisplayRemoveReconfigurationCallback(Cocoa_DisplayReconfigurationCallback, _this);

    for (i = 0; i < _this->num_displays; ++i) {
        SDL_VideoDisplay *display = _this->displays[i];
        SDL_DisplayModeData *mode;

        if (display->current_mode->internal != display->desktop_mode.internal) {
            Cocoa_SetDisplayMode(_this, display, &display->desktop_mode);
        }

        mode = display->desktop_mode.internal;
        CFRelease(mode->modes);

        for (j = 0; j < display->num_fullscreen_modes; j++) {
            mode = display->fullscreen_modes[j].internal;
            CFRelease(mode->modes);
        }
    }
}

#endif // SDL_VIDEO_DRIVER_COCOA
