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

#ifdef SDL_VIDEO_DRIVER_HAIKU

#include <AppKit.h>
#include <InterfaceKit.h>
#include "SDL_bmodes.h"
#include "SDL_BWin.h"

#ifdef SDL_VIDEO_OPENGL
#include "SDL_bopengl.h"
#endif

#include "../../core/haiku/SDL_BApp.h"

#ifdef __cplusplus
extern "C" {
#endif


#define WRAP_BMODE 1 // FIXME: Some debate as to whether this is necessary

#if WRAP_BMODE
/* This wrapper is here so that the internal can be freed without freeing
   the display_mode structure */
struct SDL_DisplayModeData {
    display_mode *bmode;
};
#endif

static SDL_INLINE SDL_BWin *_ToBeWin(SDL_Window *window)
{
    return (SDL_BWin *)(window->internal);
}

static SDL_INLINE SDL_BLooper *_GetBeLooper()
{
    return SDL_Looper;
}

static SDL_INLINE display_mode * _ExtractBMode(SDL_DisplayMode *mode)
{
#if WRAP_BMODE
    return mode->internal->bmode;
#else
    return (display_mode *)mode->internal;
#endif
}

// Copied from haiku/trunk/src/preferences/screen/ScreenMode.cpp
static void get_refresh_rate(display_mode &mode, int *numerator, int *denominator)
{
    *numerator = (mode.timing.pixel_clock * 1000);
    *denominator = (mode.timing.h_total * mode.timing.v_total);
}


#if 0
/* TODO:
 * This is a useful debugging tool.  Uncomment and insert into code as needed.
 */
void _SpoutModeData(display_mode *bmode)
{
    printf("BMode:\n");
    printf("\tw,h = (%i,%i)\n", bmode->virtual_width, bmode->virtual_height);
    printf("\th,v = (%i,%i)\n", bmode->h_display_start,
            bmode->v_display_start);
    if (bmode->flags) {
        printf("\tFlags:\n");
        if (bmode->flags & B_SCROLL) {
            printf("\t\tB_SCROLL\n");
        }
        if (bmode->flags & B_8_BIT_DAC) {
            printf("\t\tB_8_BIT_DAC\n");
        }
        if (bmode->flags & B_HARDWARE_CURSOR) {
            printf("\t\tB_HARDWARE_CURSOR\n");
        }
        if (bmode->flags & B_PARALLEL_ACCESS) {
            printf("\t\tB_PARALLEL_ACCESS\n");
        }
        if (bmode->flags & B_DPMS) {
            printf("\t\tB_DPMS\n");
        }
        if (bmode->flags & B_IO_FB_NA) {
            printf("\t\tB_IO_FB_NA\n");
        }
    }
    printf("\tTiming:\n");
    printf("\t\tpx clock: %i\n", bmode->timing.pixel_clock);
    printf("\t\th - display: %i sync start: %i sync end: %i total: %i\n",
        bmode->timing.h_display, bmode->timing.h_sync_start,
        bmode->timing.h_sync_end, bmode->timing.h_total);
    printf("\t\tv - display: %i sync start: %i sync end: %i total: %i\n",
        bmode->timing.v_display, bmode->timing.v_sync_start,
        bmode->timing.v_sync_end, bmode->timing.v_total);
    if (bmode->timing.flags) {
        printf("\t\tFlags:\n");
        if (bmode->timing.flags & B_BLANK_PEDESTAL) {
            printf("\t\t\tB_BLANK_PEDESTAL\n");
        }
        if (bmode->timing.flags & B_TIMING_INTERLACED) {
            printf("\t\t\tB_TIMING_INTERLACED\n");
        }
        if (bmode->timing.flags & B_POSITIVE_HSYNC) {
            printf("\t\t\tB_POSITIVE_HSYNC\n");
        }
        if (bmode->timing.flags & B_POSITIVE_VSYNC) {
            printf("\t\t\tB_POSITIVE_VSYNC\n");
        }
        if (bmode->timing.flags & B_SYNC_ON_GREEN) {
            printf("\t\t\tB_SYNC_ON_GREEN\n");
        }
    }
}
#endif



SDL_PixelFormat HAIKU_ColorSpaceToSDLPxFormat(uint32 colorspace)
{
    switch (colorspace) {
    case B_CMAP8:
        return SDL_PIXELFORMAT_INDEX8;
        break;
    case B_RGB15:
    case B_RGBA15:
    case B_RGB15_BIG:
    case B_RGBA15_BIG:
        return SDL_PIXELFORMAT_XRGB1555;
        break;
    case B_RGB16:
    case B_RGB16_BIG:
        return SDL_PIXELFORMAT_RGB565;
        break;
    case B_RGB24:
    case B_RGB24_BIG:
        return SDL_PIXELFORMAT_BGR24;
        break;
    case B_RGB32:
    case B_RGBA32:
    case B_RGB32_BIG:
    case B_RGBA32_BIG:
        return SDL_PIXELFORMAT_XRGB8888;
        break;
    }

    // May never get here, but safer and needed to shut up compiler
    SDL_SetError("Invalid color space");
    return SDL_PIXELFORMAT_UNKNOWN;
}

static void _BDisplayModeToSdlDisplayMode(display_mode *bmode, SDL_DisplayMode *mode)
{
    SDL_zerop(mode);
    mode->w = bmode->virtual_width;
    mode->h = bmode->virtual_height;
    get_refresh_rate(*bmode, &mode->refresh_rate_numerator, &mode->refresh_rate_denominator);

#if WRAP_BMODE
    SDL_DisplayModeData *data = (SDL_DisplayModeData*)SDL_calloc(1, sizeof(SDL_DisplayModeData));
    data->bmode = bmode;

    mode->internal = data;
#else
    mode->internal = bmode;
#endif

    // Set the format
    mode->format = HAIKU_ColorSpaceToSDLPxFormat(bmode->space);
}

// Later, there may be more than one monitor available
static void _AddDisplay(BScreen *screen)
{
    SDL_DisplayMode mode;
    display_mode bmode;
    screen->GetMode(&bmode);

    _BDisplayModeToSdlDisplayMode(&bmode, &mode);

    SDL_AddBasicVideoDisplay(&mode);
}

/*
 * Functions called by SDL
 */

bool HAIKU_InitModes(SDL_VideoDevice *_this)
{
    BScreen screen;

    /* TODO: When Haiku supports multiple display screens, call
       _AddDisplayScreen() for each of them. */
    _AddDisplay(&screen);
    return true;
}

void HAIKU_QuitModes(SDL_VideoDevice *_this)
{
    return;
}


bool HAIKU_GetDisplayBounds(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_Rect *rect)
{
    BScreen bscreen;
    BRect rc = bscreen.Frame();
    rect->x = (int)rc.left;
    rect->y = (int)rc.top;
    rect->w = (int)rc.Width() + 1;
    rect->h = (int)rc.Height() + 1;
    return true;
}

bool HAIKU_GetDisplayModes(SDL_VideoDevice *_this, SDL_VideoDisplay *display)
{
    // Get the current screen
    BScreen bscreen;

    // Iterate through all of the modes
    SDL_DisplayMode mode;
    display_mode this_bmode;
    display_mode *bmodes;
    uint32 count, i;

    // Get graphics-hardware supported modes
    bscreen.GetModeList(&bmodes, &count);
    bscreen.GetMode(&this_bmode);

    for (i = 0; i < count; ++i) {
        // FIXME: Apparently there are errors with colorspace changes
        if (bmodes[i].space == this_bmode.space) {
            _BDisplayModeToSdlDisplayMode(&bmodes[i], &mode);
            SDL_AddFullscreenDisplayMode(display, &mode);
        }
    }
    free(bmodes); // This should NOT be SDL_free()
    return true;
}


bool HAIKU_SetDisplayMode(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_DisplayMode *mode)
{
    // Get the current screen
    BScreen bscreen;
    if (!bscreen.IsValid()) {
        printf(__FILE__": %d - ERROR: BAD SCREEN\n", __LINE__);
    }

    // Set the mode using the driver data
    display_mode *bmode = _ExtractBMode(mode);


    // FIXME: Is the first option always going to be the right one?
    uint32 c = 0, i;
    display_mode *bmode_list;
    bscreen.GetModeList(&bmode_list, &c);
    for (i = 0; i < c; ++i) {
        if (    bmode_list[i].space == bmode->space &&
            bmode_list[i].virtual_width == bmode->virtual_width &&
            bmode_list[i].virtual_height == bmode->virtual_height ) {
                bmode = &bmode_list[i];
                break;
        }
    }

    if (bscreen.SetMode(bmode) != B_OK) {
        return SDL_SetError("Bad video mode");
    }

    free(bmode_list); // This should NOT be SDL_free()

#ifdef SDL_VIDEO_OPENGL
    /* FIXME: Is there some way to reboot the OpenGL context?  This doesn't
       help */
//    HAIKU_GL_RebootContexts(_this);
#endif

    return true;
}

#ifdef __cplusplus
}
#endif

#endif // SDL_VIDEO_DRIVER_HAIKU
