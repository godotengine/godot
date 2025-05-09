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

#include "SDL_cocoapen.h"
#include "SDL_cocoavideo.h"

#include "../../events/SDL_pen_c.h"

bool Cocoa_InitPen(SDL_VideoDevice *_this)
{
    return true;
}

typedef struct Cocoa_PenHandle
{
    NSUInteger deviceid;
    NSUInteger toolid;
    SDL_PenID pen;
    bool is_eraser;
} Cocoa_PenHandle;

typedef struct FindPenByDeviceAndToolIDData
{
    NSUInteger deviceid;
    NSUInteger toolid;
    void *handle;
} FindPenByDeviceAndToolIDData;

static bool FindPenByDeviceAndToolID(void *handle, void *userdata)
{
    const Cocoa_PenHandle *cocoa_handle = (const Cocoa_PenHandle *) handle;
    FindPenByDeviceAndToolIDData *data = (FindPenByDeviceAndToolIDData *) userdata;

    if (cocoa_handle->deviceid != data->deviceid) {
        return false;
    } else if (cocoa_handle->toolid != data->toolid) {
        return false;
    }
    data->handle = handle;
    return true;
}

static Cocoa_PenHandle *Cocoa_FindPenByDeviceID(NSUInteger deviceid, NSUInteger toolid)
{
    FindPenByDeviceAndToolIDData data;
    data.deviceid = deviceid;
    data.toolid = toolid;
    data.handle = NULL;
    SDL_FindPenByCallback(FindPenByDeviceAndToolID, &data);
    return (Cocoa_PenHandle *) data.handle;
}

static void Cocoa_HandlePenProximityEvent(SDL_CocoaWindowData *_data, NSEvent *event)
{
    const NSUInteger devid = [event deviceID];
    const NSUInteger toolid = [event pointingDeviceID];

    if (event.enteringProximity) {  // new pen coming!
        const NSPointingDeviceType devtype = [event pointingDeviceType];
        const bool is_eraser = (devtype == NSPointingDeviceTypeEraser);
        const bool is_pen = (devtype == NSPointingDeviceTypePen);
        if (!is_eraser && !is_pen) {
            return;  // we ignore other things, which hopefully is right.
        }

        Cocoa_PenHandle *handle = (Cocoa_PenHandle *) SDL_calloc(1, sizeof (*handle));
        if (!handle) {
            return;  // oh well.
        }

        // Cocoa offers almost none of this information as specifics, but can without warning offer any of these specific things.
        SDL_PenInfo peninfo;
        SDL_zero(peninfo);
        peninfo.capabilities = SDL_PEN_CAPABILITY_PRESSURE | SDL_PEN_CAPABILITY_ROTATION | SDL_PEN_CAPABILITY_XTILT | SDL_PEN_CAPABILITY_YTILT | SDL_PEN_CAPABILITY_TANGENTIAL_PRESSURE | (is_eraser ? SDL_PEN_CAPABILITY_ERASER : 0);
        peninfo.max_tilt = 90.0f;
        peninfo.num_buttons = 2;
        peninfo.subtype = is_eraser ? SDL_PEN_TYPE_ERASER : SDL_PEN_TYPE_PEN;

        handle->deviceid = devid;
        handle->toolid = toolid;
        handle->is_eraser = is_eraser;
        handle->pen = SDL_AddPenDevice(Cocoa_GetEventTimestamp([event timestamp]), NULL, &peninfo, handle);
        if (!handle->pen) {
            SDL_free(handle);  // oh well.
        }
    } else {  // old pen leaving!
        Cocoa_PenHandle *handle = Cocoa_FindPenByDeviceID(devid, toolid);
        if (handle) {
            SDL_RemovePenDevice(Cocoa_GetEventTimestamp([event timestamp]), handle->pen);
            SDL_free(handle);
        }
    }
}

static void Cocoa_HandlePenPointEvent(SDL_CocoaWindowData *_data, NSEvent *event)
{
    const Uint64 timestamp = Cocoa_GetEventTimestamp([event timestamp]);
    Cocoa_PenHandle *handle = Cocoa_FindPenByDeviceID([event deviceID], [event pointingDeviceID]);
    if (!handle) {
        return;
    }

    const SDL_PenID pen = handle->pen;
    const NSEventButtonMask buttons = [event buttonMask];
    const NSPoint tilt = [event tilt];
    const NSPoint point = [event locationInWindow];
    const bool is_touching = (buttons & NSEventButtonMaskPenTip) != 0;
    SDL_Window *window = _data.window;

    SDL_SendPenTouch(timestamp, pen, window, handle->is_eraser, is_touching);
    SDL_SendPenMotion(timestamp, pen, window, (float) point.x, (float) (window->h - point.y));
    SDL_SendPenButton(timestamp, pen, window, 1, ((buttons & NSEventButtonMaskPenLowerSide) != 0));
    SDL_SendPenButton(timestamp, pen, window, 2, ((buttons & NSEventButtonMaskPenUpperSide) != 0));
    SDL_SendPenAxis(timestamp, pen, window, SDL_PEN_AXIS_PRESSURE, [event pressure]);
    SDL_SendPenAxis(timestamp, pen, window, SDL_PEN_AXIS_ROTATION, [event rotation]);
    SDL_SendPenAxis(timestamp, pen, window, SDL_PEN_AXIS_XTILT, ((float) tilt.x) * 90.0f);
    SDL_SendPenAxis(timestamp, pen, window, SDL_PEN_AXIS_YTILT, ((float) -tilt.y) * 90.0f);
    SDL_SendPenAxis(timestamp, pen, window, SDL_PEN_AXIS_TANGENTIAL_PRESSURE, event.tangentialPressure);
}

bool Cocoa_HandlePenEvent(SDL_CocoaWindowData *_data, NSEvent *event)
{
    NSEventType type = [event type];

    if ((type != NSEventTypeTabletPoint) && (type != NSEventTypeTabletProximity)) {
        const NSEventSubtype subtype = [event subtype];
        if (subtype == NSEventSubtypeTabletPoint) {
            type = NSEventTypeTabletPoint;
        } else if (subtype == NSEventSubtypeTabletProximity) {
            type = NSEventTypeTabletProximity;
        } else {
            return false;  // not a tablet event.
        }
    }

    if (type == NSEventTypeTabletPoint) {
        Cocoa_HandlePenPointEvent(_data, event);
    } else if (type == NSEventTypeTabletProximity) {
        Cocoa_HandlePenProximityEvent(_data, event);
    } else {
        return false;  // not a tablet event.
    }

    return true;
}

static void Cocoa_FreePenHandle(SDL_PenID instance_id, void *handle, void *userdata)
{
    SDL_free(handle);
}

void Cocoa_QuitPen(SDL_VideoDevice *_this)
{
    SDL_RemoveAllPenDevices(Cocoa_FreePenHandle, NULL);
}

#endif // SDL_VIDEO_DRIVER_COCOA
