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

// Pressure-sensitive pen handling code for SDL

#include "../SDL_hints_c.h"
#include "SDL_events_c.h"
#include "SDL_pen_c.h"

static SDL_PenID pen_touching = 0;  // used for synthetic mouse/touch events.

typedef struct SDL_Pen
{
    SDL_PenID instance_id;
    char *name;
    SDL_PenInfo info;
    float axes[SDL_PEN_AXIS_COUNT];
    float x;
    float y;
    SDL_PenInputFlags input_state;
    void *driverdata;
} SDL_Pen;

// we assume there's usually 0-1 pens in most cases and this list doesn't
// usually change after startup, so a simple array with a RWlock is fine for now.
static SDL_RWLock *pen_device_rwlock = NULL;
static SDL_Pen *pen_devices SDL_GUARDED_BY(pen_device_rwlock) = NULL;
static int pen_device_count SDL_GUARDED_BY(pen_device_rwlock) = 0;

// You must hold pen_device_rwlock before calling this, and result is only safe while lock is held!
// If SDL isn't initialized, grabbing the NULL lock is a no-op and there will be zero devices, so
// locking and calling this in that case will do the right thing.
static SDL_Pen *FindPenByInstanceId(SDL_PenID instance_id) SDL_REQUIRES_SHARED(pen_device_rwlock)
{
    if (instance_id) {
        for (int i = 0; i < pen_device_count; i++) {
            if (pen_devices[i].instance_id == instance_id) {
                return &pen_devices[i];
            }
        }
    }
    SDL_SetError("Invalid pen instance ID");
    return NULL;
}

SDL_PenID SDL_FindPenByHandle(void *handle)
{
    SDL_PenID result = 0;
    SDL_LockRWLockForReading(pen_device_rwlock);
    for (int i = 0; i < pen_device_count; i++) {
        if (pen_devices[i].driverdata == handle) {
            result = pen_devices[i].instance_id;
            break;
        }
    }
    SDL_UnlockRWLock(pen_device_rwlock);
    return result;
}

SDL_PenID SDL_FindPenByCallback(bool (*callback)(void *handle, void *userdata), void *userdata)
{
    SDL_PenID result = 0;
    SDL_LockRWLockForReading(pen_device_rwlock);
    for (int i = 0; i < pen_device_count; i++) {
        if (callback(pen_devices[i].driverdata, userdata)) {
            result = pen_devices[i].instance_id;
            break;
        }
    }
    SDL_UnlockRWLock(pen_device_rwlock);
    return result;
}



// public API ...

bool SDL_InitPen(void)
{
    SDL_assert(pen_device_rwlock == NULL);
    SDL_assert(pen_devices == NULL);
    SDL_assert(pen_device_count == 0);
    pen_device_rwlock = SDL_CreateRWLock();
    if (!pen_device_rwlock) {
        return false;
    }
    return true;
}

void SDL_QuitPen(void)
{
    SDL_DestroyRWLock(pen_device_rwlock);
    pen_device_rwlock = NULL;
    if (pen_devices) {
        for (int i = pen_device_count; i--; ) {
            SDL_free(pen_devices[i].name);
        }
        SDL_free(pen_devices);
        pen_devices = NULL;
    }
    pen_device_count = 0;
    pen_touching = 0;
}

#if 0 // not a public API at the moment.
SDL_PenID *SDL_GetPens(int *count)
{
    SDL_LockRWLockForReading(pen_device_rwlock);
    const int num_devices = pen_device_count;
    SDL_PenID *result = (SDL_PenID *) SDL_malloc((num_devices + 1) * sizeof (SDL_PenID));
    if (result) {
        for (int i = 0; i < num_devices; i++) {
            result[i] = pen_devices[i].instance_id;
        }
        result[num_devices] = 0;  // null-terminated.
    }
    SDL_UnlockRWLock(pen_device_rwlock);

    if (count) {
        *count = result ? num_devices : 0;
    }
    return result;
}

const char *SDL_GetPenName(SDL_PenID instance_id)
{
    SDL_LockRWLockForReading(pen_device_rwlock);
    const SDL_Pen *pen = FindPenByInstanceId(instance_id);
    const char *result = pen ? SDL_GetPersistentString(pen->name) : NULL;
    SDL_UnlockRWLock(pen_device_rwlock);
    return result;
}

bool SDL_GetPenInfo(SDL_PenID instance_id, SDL_PenInfo *info)
{
    SDL_LockRWLockForReading(pen_device_rwlock);
    const SDL_Pen *pen = FindPenByInstanceId(instance_id);
    const bool result = pen ? true : false;
    if (info) {
        if (result) {
            SDL_copyp(info, &pen->info);
        } else {
            SDL_zerop(info);
        }
    }
    SDL_UnlockRWLock(pen_device_rwlock);
    return result;
}

bool SDL_PenConnected(SDL_PenID instance_id)
{
    SDL_LockRWLockForReading(pen_device_rwlock);
    const SDL_Pen *pen = FindPenByInstanceId(instance_id);
    const bool result = (pen != NULL);
    SDL_UnlockRWLock(pen_device_rwlock);
    return result;
}
#endif

SDL_PenInputFlags SDL_GetPenStatus(SDL_PenID instance_id, float *axes, int num_axes)
{
    if (num_axes < 0) {
        num_axes = 0;
    }

    SDL_LockRWLockForReading(pen_device_rwlock);
    const SDL_Pen *pen = FindPenByInstanceId(instance_id);
    SDL_PenInputFlags result = 0;
    if (pen) {
        result = pen->input_state;
        if (axes && num_axes) {
            SDL_memcpy(axes, pen->axes, SDL_min(num_axes, SDL_PEN_AXIS_COUNT) * sizeof (*axes));
            // zero out axes we don't know about, in case the caller built with newer SDL headers that support more of them.
            if (num_axes > SDL_PEN_AXIS_COUNT) {
                SDL_memset(&axes[SDL_PEN_AXIS_COUNT], '\0', (num_axes - SDL_PEN_AXIS_COUNT) * sizeof (*axes));
            }
        }
    }
    SDL_UnlockRWLock(pen_device_rwlock);
    return result;
}

SDL_PenCapabilityFlags SDL_GetPenCapabilityFromAxis(SDL_PenAxis axis)
{
    // the initial capability bits happen to match up, but as
    // more features show up later, the bits may no longer be contiguous!
    if ((axis >= SDL_PEN_AXIS_PRESSURE) && (axis <= SDL_PEN_AXIS_SLIDER)) {
        return ((SDL_PenCapabilityFlags) 1u) << ((SDL_PenCapabilityFlags) axis);
    }
    return 0;  // oh well.
}

SDL_PenID SDL_AddPenDevice(Uint64 timestamp, const char *name, const SDL_PenInfo *info, void *handle)
{
    SDL_assert(handle != NULL);  // just allocate a Uint8 so you have a unique pointer if not needed!
    SDL_assert(SDL_FindPenByHandle(handle) == 0);  // Backends shouldn't double-add pens!
    SDL_assert(pen_device_rwlock != NULL);   // subsystem should be initialized by now!

    char *namecpy = SDL_strdup(name ? name : "Unnamed pen");
    if (!namecpy) {
        return 0;
    }

    SDL_PenID result = 0;

    SDL_LockRWLockForWriting(pen_device_rwlock);

    SDL_Pen *pen = NULL;
    void *ptr = SDL_realloc(pen_devices, (pen_device_count + 1) * sizeof (*pen));
    if (ptr) {
        result = (SDL_PenID) SDL_GetNextObjectID();
        pen_devices = (SDL_Pen *) ptr;
        pen = &pen_devices[pen_device_count];
        pen_device_count++;

        SDL_zerop(pen);
        pen->instance_id = result;
        pen->name = namecpy;
        if (info) {
            SDL_copyp(&pen->info, info);
        }
        pen->driverdata = handle;
        // axes and input state defaults to zero.
    }
    SDL_UnlockRWLock(pen_device_rwlock);

    if (!pen) {
        SDL_free(namecpy);
    }

    if (result && SDL_EventEnabled(SDL_EVENT_PEN_PROXIMITY_IN)) {
        SDL_Event event;
        SDL_zero(event);
        event.pproximity.type = SDL_EVENT_PEN_PROXIMITY_IN;
        event.pproximity.timestamp = timestamp;
        event.pproximity.which = result;
        SDL_PushEvent(&event);
    }

    return result;
}

void SDL_RemovePenDevice(Uint64 timestamp, SDL_PenID instance_id)
{
    if (!instance_id) {
        return;
    }

    SDL_LockRWLockForWriting(pen_device_rwlock);
    SDL_Pen *pen = FindPenByInstanceId(instance_id);
    if (pen) {
        SDL_free(pen->name);
        // we don't free `pen`, it's just part of simple array. Shuffle it out.
        const int idx = ((int) (pen - pen_devices));
        SDL_assert((idx >= 0) && (idx < pen_device_count));
        if ( idx < (pen_device_count - 1) ) {
            SDL_memmove(&pen_devices[idx], &pen_devices[idx + 1], sizeof (*pen) * ((pen_device_count - idx) - 1));
        }

        SDL_assert(pen_device_count > 0);
        pen_device_count--;

        if (pen_device_count) {
            void *ptr = SDL_realloc(pen_devices, sizeof (*pen) * pen_device_count);  // shrink it down.
            if (ptr) {
                pen_devices = (SDL_Pen *) ptr;
            }
        } else {
            SDL_free(pen_devices);
            pen_devices = NULL;
        }
    }
    SDL_UnlockRWLock(pen_device_rwlock);

    if (pen && SDL_EventEnabled(SDL_EVENT_PEN_PROXIMITY_OUT)) {
        SDL_Event event;
        SDL_zero(event);
        event.pproximity.type = SDL_EVENT_PEN_PROXIMITY_OUT;
        event.pproximity.timestamp = timestamp;
        event.pproximity.which = instance_id;
        SDL_PushEvent(&event);
    }
}

// This presumably is happening during video quit, so we don't send PROXIMITY_OUT events here.
void SDL_RemoveAllPenDevices(void (*callback)(SDL_PenID instance_id, void *handle, void *userdata), void *userdata)
{
    SDL_LockRWLockForWriting(pen_device_rwlock);
    if (pen_device_count > 0) {
        SDL_assert(pen_devices != NULL);
        for (int i = 0; i < pen_device_count; i++) {
            callback(pen_devices[i].instance_id, pen_devices[i].driverdata, userdata);
            SDL_free(pen_devices[i].name);
        }
    }
    SDL_free(pen_devices);
    pen_devices = NULL;
    SDL_UnlockRWLock(pen_device_rwlock);
}

void SDL_SendPenTouch(Uint64 timestamp, SDL_PenID instance_id, SDL_Window *window, bool eraser, bool down)
{
    bool send_event = false;
    SDL_PenInputFlags input_state = 0;
    float x = 0.0f;
    float y = 0.0f;

    // note that this locks for _reading_ because the lock protects the
    // pen_devices array from being reallocated from under us, not the data in it;
    // we assume only one thread (in the backend) is modifying an individual pen at
    // a time, so it can update input state cleanly here.
    SDL_LockRWLockForReading(pen_device_rwlock);
    SDL_Pen *pen = FindPenByInstanceId(instance_id);
    if (pen) {
        input_state = pen->input_state;
        x = pen->x;
        y = pen->y;

        if (down && ((input_state & SDL_PEN_INPUT_DOWN) == 0)) {
            input_state |= SDL_PEN_INPUT_DOWN;
            send_event = true;
        } else if (!down && (input_state & SDL_PEN_INPUT_DOWN)) {
            input_state &= ~SDL_PEN_INPUT_DOWN;
            send_event = true;
        }

        if (eraser && ((input_state & SDL_PEN_INPUT_ERASER_TIP) == 0)) {
            input_state |= SDL_PEN_INPUT_ERASER_TIP;
            send_event = true;
        } else if (!eraser && (input_state & SDL_PEN_INPUT_ERASER_TIP)) {
            input_state &= ~SDL_PEN_INPUT_ERASER_TIP;
            send_event = true;
        }

        pen->input_state = input_state;  // we could do an SDL_SetAtomicInt here if we run into trouble...
    }
    SDL_UnlockRWLock(pen_device_rwlock);

    if (send_event) {
        const SDL_EventType evtype = down ? SDL_EVENT_PEN_DOWN : SDL_EVENT_PEN_UP;
        if (SDL_EventEnabled(evtype)) {
            SDL_Event event;
            SDL_zero(event);
            event.ptouch.type = evtype;
            event.ptouch.timestamp = timestamp;
            event.ptouch.windowID = window ? window->id : 0;
            event.ptouch.which = instance_id;
            event.ptouch.pen_state = input_state;
            event.ptouch.x = x;
            event.ptouch.y = y;
            event.ptouch.eraser = eraser;
            event.ptouch.down = down;
            SDL_PushEvent(&event);
        }

        SDL_Mouse *mouse = SDL_GetMouse();
        if (mouse && window) {
            if (mouse->pen_mouse_events) {
                if (down) {
                    if (!pen_touching) {
                        SDL_SendMouseMotion(timestamp, window, SDL_PEN_MOUSEID, false, x, y);
                        SDL_SendMouseButton(timestamp, window, SDL_PEN_MOUSEID, SDL_BUTTON_LEFT, true);
                    }
                } else {
                    if (pen_touching == instance_id) {
                        SDL_SendMouseButton(timestamp, window, SDL_PEN_MOUSEID, SDL_BUTTON_LEFT, false);
                    }
                }
            }

            if (mouse->pen_touch_events) {
                const SDL_EventType touchtype = down ? SDL_EVENT_FINGER_DOWN : SDL_EVENT_FINGER_UP;
                const float normalized_x = x / (float)window->w;
                const float normalized_y = y / (float)window->h;
                if (!pen_touching || (pen_touching == instance_id)) {
                    SDL_SendTouch(timestamp, SDL_PEN_TOUCHID, SDL_BUTTON_LEFT, window, touchtype, normalized_x, normalized_y, pen->axes[SDL_PEN_AXIS_PRESSURE]);
                }
            }
        }

        if (down) {
            if (!pen_touching) {
                pen_touching = instance_id;
            }
        } else {
            if (pen_touching == instance_id) {
                pen_touching = 0;
            }
        }
    }
}

void SDL_SendPenAxis(Uint64 timestamp, SDL_PenID instance_id, SDL_Window *window, SDL_PenAxis axis, float value)
{
    SDL_assert((axis >= 0) && (axis < SDL_PEN_AXIS_COUNT));  // fix the backend if this triggers.

    bool send_event = false;
    SDL_PenInputFlags input_state = 0;
    float x = 0.0f;
    float y = 0.0f;

    // note that this locks for _reading_ because the lock protects the
    // pen_devices array from being reallocated from under us, not the data in it;
    // we assume only one thread (in the backend) is modifying an individual pen at
    // a time, so it can update input state cleanly here.
    SDL_LockRWLockForReading(pen_device_rwlock);
    SDL_Pen *pen = FindPenByInstanceId(instance_id);
    if (pen) {
        if (pen->axes[axis] != value) {
            pen->axes[axis] = value;  // we could do an SDL_SetAtomicInt here if we run into trouble...
            input_state = pen->input_state;
            x = pen->x;
            y = pen->y;
            send_event = true;
        }
    }
    SDL_UnlockRWLock(pen_device_rwlock);

    if (send_event && SDL_EventEnabled(SDL_EVENT_PEN_AXIS)) {
        SDL_Event event;
        SDL_zero(event);
        event.paxis.type = SDL_EVENT_PEN_AXIS;
        event.paxis.timestamp = timestamp;
        event.paxis.windowID = window ? window->id : 0;
        event.paxis.which = instance_id;
        event.paxis.pen_state = input_state;
        event.paxis.x = x;
        event.paxis.y = y;
        event.paxis.axis = axis;
        event.paxis.value = value;
        SDL_PushEvent(&event);

        if (window && (axis == SDL_PEN_AXIS_PRESSURE) && (pen_touching == instance_id)) {
            SDL_Mouse *mouse = SDL_GetMouse();
            if (mouse && mouse->pen_touch_events) {
                const float normalized_x = x / (float)window->w;
                const float normalized_y = y / (float)window->h;
                SDL_SendTouchMotion(timestamp, SDL_PEN_TOUCHID, SDL_BUTTON_LEFT, window, normalized_x, normalized_y, value);
            }
        }
    }
}

void SDL_SendPenMotion(Uint64 timestamp, SDL_PenID instance_id, SDL_Window *window, float x, float y)
{
    bool send_event = false;
    SDL_PenInputFlags input_state = 0;

    // note that this locks for _reading_ because the lock protects the
    // pen_devices array from being reallocated from under us, not the data in it;
    // we assume only one thread (in the backend) is modifying an individual pen at
    // a time, so it can update input state cleanly here.
    SDL_LockRWLockForReading(pen_device_rwlock);
    SDL_Pen *pen = FindPenByInstanceId(instance_id);
    if (pen) {
        if ((pen->x != x) || (pen->y != y)) {
            pen->x = x;  // we could do an SDL_SetAtomicInt here if we run into trouble...
            pen->y = y;  // we could do an SDL_SetAtomicInt here if we run into trouble...
            input_state = pen->input_state;
            send_event = true;
        }
    }
    SDL_UnlockRWLock(pen_device_rwlock);

    if (send_event && SDL_EventEnabled(SDL_EVENT_PEN_MOTION)) {
        SDL_Event event;
        SDL_zero(event);
        event.pmotion.type = SDL_EVENT_PEN_MOTION;
        event.pmotion.timestamp = timestamp;
        event.pmotion.windowID = window ? window->id : 0;
        event.pmotion.which = instance_id;
        event.pmotion.pen_state = input_state;
        event.pmotion.x = x;
        event.pmotion.y = y;
        SDL_PushEvent(&event);

        if (window) {
            SDL_Mouse *mouse = SDL_GetMouse();
            if (mouse) {
                if (pen_touching == instance_id) {
                    if (mouse->pen_mouse_events) {
                        SDL_SendMouseMotion(timestamp, window, SDL_PEN_MOUSEID, false, x, y);
                    }

                    if (mouse->pen_touch_events) {
                        const float normalized_x = x / (float)window->w;
                        const float normalized_y = y / (float)window->h;
                        SDL_SendTouchMotion(timestamp, SDL_PEN_TOUCHID, SDL_BUTTON_LEFT, window, normalized_x, normalized_y, pen->axes[SDL_PEN_AXIS_PRESSURE]);
                    }
                } else if (pen_touching == 0) {  // send mouse motion (without a pressed button) for pens that aren't touching.
                    // this might cause a little chaos if you have multiple pens hovering at the same time, but this seems unlikely in the real world, and also something you did to yourself.  :)
                    SDL_SendMouseMotion(timestamp, window, SDL_PEN_MOUSEID, false, x, y);
                }
            }
        }
    }
}

void SDL_SendPenButton(Uint64 timestamp, SDL_PenID instance_id, SDL_Window *window, Uint8 button, bool down)
{
    bool send_event = false;
    SDL_PenInputFlags input_state = 0;
    float x = 0.0f;
    float y = 0.0f;

    if ((button < 1) || (button > 5)) {
        return; // clamp for now.
    }

    // note that this locks for _reading_ because the lock protects the
    // pen_devices array from being reallocated from under us, not the data in it;
    // we assume only one thread (in the backend) is modifying an individual pen at
    // a time, so it can update input state cleanly here.
    SDL_LockRWLockForReading(pen_device_rwlock);
    SDL_Pen *pen = FindPenByInstanceId(instance_id);
    if (pen) {
        input_state = pen->input_state;
        const Uint32 flag = (Uint32) (1u << button);
        const bool current = ((input_state & flag) != 0);
        x = pen->x;
        y = pen->y;
        if (down && !current) {
            input_state |= flag;
            send_event = true;
        } else if (!down && current) {
            input_state &= ~flag;
            send_event = true;
        }
        pen->input_state = input_state;  // we could do an SDL_SetAtomicInt here if we run into trouble...
    }
    SDL_UnlockRWLock(pen_device_rwlock);

    if (send_event) {
        const SDL_EventType evtype = down ? SDL_EVENT_PEN_BUTTON_DOWN : SDL_EVENT_PEN_BUTTON_UP;
        if (SDL_EventEnabled(evtype)) {
            SDL_Event event;
            SDL_zero(event);
            event.pbutton.type = evtype;
            event.pbutton.timestamp = timestamp;
            event.pbutton.windowID = window ? window->id : 0;
            event.pbutton.which = instance_id;
            event.pbutton.pen_state = input_state;
            event.pbutton.x = x;
            event.pbutton.y = y;
            event.pbutton.button = button;
            event.pbutton.down = down;
            SDL_PushEvent(&event);

            if (window && (!pen_touching || (pen_touching == instance_id))) {
                SDL_Mouse *mouse = SDL_GetMouse();
                if (mouse && mouse->pen_mouse_events) {
                    SDL_SendMouseButton(timestamp, window, SDL_PEN_MOUSEID, button + 1, down);
                }
            }
        }
    }
}

