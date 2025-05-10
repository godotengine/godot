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

#include "../SDL_internal.h"

#ifndef SDL_pen_c_h_
#define SDL_pen_c_h_

#include "SDL_mouse_c.h"

typedef Uint32 SDL_PenCapabilityFlags;
#define SDL_PEN_CAPABILITY_PRESSURE  (1u << 0)  /**< Provides pressure information on SDL_PEN_AXIS_PRESSURE. */
#define SDL_PEN_CAPABILITY_XTILT     (1u << 1)  /**< Provides horizontal tilt information on SDL_PEN_AXIS_XTILT. */
#define SDL_PEN_CAPABILITY_YTILT     (1u << 2)  /**< Provides vertical tilt information on SDL_PEN_AXIS_YTILT. */
#define SDL_PEN_CAPABILITY_DISTANCE  (1u << 3)  /**< Provides distance to drawing tablet on SDL_PEN_AXIS_DISTANCE. */
#define SDL_PEN_CAPABILITY_ROTATION  (1u << 4)  /**< Provides barrel rotation info on SDL_PEN_AXIS_ROTATION. */
#define SDL_PEN_CAPABILITY_SLIDER    (1u << 5)  /**< Provides slider/finger wheel/etc on SDL_PEN_AXIS_SLIDER. */
#define SDL_PEN_CAPABILITY_TANGENTIAL_PRESSURE (1u << 6)  /**< Provides barrel pressure on SDL_PEN_AXIS_TANGENTIAL_PRESSURE. */
#define SDL_PEN_CAPABILITY_ERASER    (1u << 7)  /**< Pen also has an eraser tip. */

typedef enum SDL_PenSubtype
{
    SDL_PEN_TYPE_UNKNOWN,   /**< Unknown pen device */
    SDL_PEN_TYPE_ERASER,    /**< Eraser */
    SDL_PEN_TYPE_PEN,       /**< Generic pen; this is the default. */
    SDL_PEN_TYPE_PENCIL,    /**< Pencil */
    SDL_PEN_TYPE_BRUSH,     /**< Brush-like device */
    SDL_PEN_TYPE_AIRBRUSH   /**< Airbrush device that "sprays" ink */
} SDL_PenSubtype;

typedef struct SDL_PenInfo
{
    SDL_PenCapabilityFlags capabilities;  /**< bitflags of device capabilities */
    float max_tilt;    /**< Physical maximum tilt angle, for XTILT and YTILT, or -1.0f if unknown.  Pens cannot typically tilt all the way to 90 degrees, so this value is usually less than 90.0. */
    Uint32 wacom_id;   /**< For Wacom devices: wacom tool type ID, otherwise 0 (useful e.g. with libwacom) */
    int num_buttons; /**< Number of pen buttons (not counting the pen tip), or -1 if unknown. */
    SDL_PenSubtype subtype;  /**< type of pen device */
} SDL_PenInfo;

// Backend calls this when a new pen device is hotplugged, plus once for each pen already connected at startup.
// Note that name and info are copied but currently unused; this is placeholder for a potentially more robust API later.
// Both are allowed to be NULL.
extern SDL_PenID SDL_AddPenDevice(Uint64 timestamp, const char *name, const SDL_PenInfo *info, void *handle);

// Backend calls this when an existing pen device is disconnected during runtime. They must free their own stuff separately.
extern void SDL_RemovePenDevice(Uint64 timestamp, SDL_PenID instance_id);

// Backend can call this to remove all pens, probably during shutdown, with a callback to let them free their own handle.
extern void SDL_RemoveAllPenDevices(void (*callback)(SDL_PenID instance_id, void *handle, void *userdata), void *userdata);

// Backend calls this when a pen's button changes, to generate events and update state.
extern void SDL_SendPenTouch(Uint64 timestamp, SDL_PenID instance_id, SDL_Window *window, bool eraser, bool down);

// Backend calls this when a pen moves on the tablet, to generate events and update state.
extern void SDL_SendPenMotion(Uint64 timestamp, SDL_PenID instance_id, SDL_Window *window, float x, float y);

// Backend calls this when a pen's axis changes, to generate events and update state.
extern void SDL_SendPenAxis(Uint64 timestamp, SDL_PenID instance_id, SDL_Window *window, SDL_PenAxis axis, float value);

// Backend calls this when a pen's button changes, to generate events and update state.
extern void SDL_SendPenButton(Uint64 timestamp, SDL_PenID instance_id, SDL_Window *window, Uint8 button, bool down);

// Backend can optionally use this to find the SDL_PenID for the `handle` that was passed to SDL_AddPenDevice.
extern SDL_PenID SDL_FindPenByHandle(void *handle);

// Backend can optionally use this to find a SDL_PenID, selected by a callback examining all devices. Zero if not found.
extern SDL_PenID SDL_FindPenByCallback(bool (*callback)(void *handle, void *userdata), void *userdata);

// Backend can use this to query current pen status.
SDL_PenInputFlags SDL_GetPenStatus(SDL_PenID instance_id, float *axes, int num_axes);

// Backend can use this to map an axis to a capability bit.
SDL_PenCapabilityFlags SDL_GetPenCapabilityFromAxis(SDL_PenAxis axis);

// Higher-level SDL video subsystem code calls this when starting up. Backends shouldn't.
extern bool SDL_InitPen(void);

// Higher-level SDL video subsystem code calls this when shutting down. Backends shouldn't.
extern void SDL_QuitPen(void);

#endif // SDL_pen_c_h_
