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

// General touch handling code for SDL

#include "SDL_events_c.h"
#include "../video/SDL_sysvideo.h"

static int SDL_num_touch = 0;
static SDL_Touch **SDL_touchDevices = NULL;

// for mapping touch events to mice
static bool finger_touching = false;
static SDL_FingerID track_fingerid;
static SDL_TouchID track_touchid;

// Public functions
bool SDL_InitTouch(void)
{
    return true;
}

bool SDL_TouchDevicesAvailable(void)
{
    return SDL_num_touch > 0;
}

SDL_TouchID *SDL_GetTouchDevices(int *count)
{
    if (count) {
        *count = 0;
    }

    const int total = SDL_num_touch;
    SDL_TouchID *result = (SDL_TouchID *) SDL_malloc(sizeof (SDL_TouchID) * (total + 1));
    if (result) {
        for (int i = 0; i < total; i++) {
            result[i] = SDL_touchDevices[i]->id;
        }
        result[total] = 0;
        if (count) {
            *count = SDL_num_touch;
        }
    }

    return result;
}

static int SDL_GetTouchIndex(SDL_TouchID id)
{
    int index;
    SDL_Touch *touch;

    for (index = 0; index < SDL_num_touch; ++index) {
        touch = SDL_touchDevices[index];
        if (touch->id == id) {
            return index;
        }
    }
    return -1;
}

SDL_Touch *SDL_GetTouch(SDL_TouchID id)
{
    int index = SDL_GetTouchIndex(id);
    if (index < 0 || index >= SDL_num_touch) {
        if ((id == SDL_MOUSE_TOUCHID) || (id == SDL_PEN_TOUCHID)) {
            // this is a virtual touch device, but for some reason they aren't added to the system. Just ignore it.
        } else if ( SDL_GetVideoDevice()->ResetTouch) {
            SDL_SetError("Unknown touch id %d, resetting", (int)id);
            SDL_GetVideoDevice()->ResetTouch(SDL_GetVideoDevice());
        } else {
            SDL_SetError("Unknown touch device id %d, cannot reset", (int)id);
        }
        return NULL;
    }
    return SDL_touchDevices[index];
}

const char *SDL_GetTouchDeviceName(SDL_TouchID id)
{
    SDL_Touch *touch = SDL_GetTouch(id);
    if (!touch) {
        return NULL;
    }
    return SDL_GetPersistentString(touch->name);
}

SDL_TouchDeviceType SDL_GetTouchDeviceType(SDL_TouchID id)
{
    SDL_Touch *touch = SDL_GetTouch(id);
    return touch ? touch->type : SDL_TOUCH_DEVICE_INVALID;
}

static int SDL_GetFingerIndex(const SDL_Touch *touch, SDL_FingerID fingerid)
{
    int index;
    for (index = 0; index < touch->num_fingers; ++index) {
        if (touch->fingers[index]->id == fingerid) {
            return index;
        }
    }
    return -1;
}

static SDL_Finger *SDL_GetFinger(const SDL_Touch *touch, SDL_FingerID id)
{
    int index = SDL_GetFingerIndex(touch, id);
    if (index < 0 || index >= touch->num_fingers) {
        return NULL;
    }
    return touch->fingers[index];
}

SDL_Finger **SDL_GetTouchFingers(SDL_TouchID touchID, int *count)
{
    SDL_Finger **fingers;
    SDL_Finger *finger_data;

    if (count) {
        *count = 0;
    }

    SDL_Touch *touch = SDL_GetTouch(touchID);
    if (!touch) {
        return NULL;
    }

    // Create a snapshot of the current finger state
    fingers = (SDL_Finger **)SDL_malloc((touch->num_fingers + 1) * sizeof(*fingers) + touch->num_fingers * sizeof(**fingers));
    if (!fingers) {
        return NULL;
    }
    finger_data = (SDL_Finger *)(fingers + (touch->num_fingers + 1));

    for (int i = 0; i < touch->num_fingers; ++i) {
        fingers[i] = &finger_data[i];
        SDL_copyp(fingers[i], touch->fingers[i]);
    }
    fingers[touch->num_fingers] = NULL;

    if (count) {
        *count = touch->num_fingers;
    }
    return fingers;
}

int SDL_AddTouch(SDL_TouchID touchID, SDL_TouchDeviceType type, const char *name)
{
    SDL_Touch **touchDevices;
    int index;

    SDL_assert(touchID != 0);

    index = SDL_GetTouchIndex(touchID);
    if (index >= 0) {
        return index;
    }

    // Add the touch to the list of touch
    touchDevices = (SDL_Touch **)SDL_realloc(SDL_touchDevices,
                                             (SDL_num_touch + 1) * sizeof(*touchDevices));
    if (!touchDevices) {
        return -1;
    }

    SDL_touchDevices = touchDevices;
    index = SDL_num_touch;

    SDL_touchDevices[index] = (SDL_Touch *)SDL_malloc(sizeof(*SDL_touchDevices[index]));
    if (!SDL_touchDevices[index]) {
        return -1;
    }

    // Added touch to list
    ++SDL_num_touch;

    // we're setting the touch properties
    SDL_touchDevices[index]->id = touchID;
    SDL_touchDevices[index]->type = type;
    SDL_touchDevices[index]->num_fingers = 0;
    SDL_touchDevices[index]->max_fingers = 0;
    SDL_touchDevices[index]->fingers = NULL;
    SDL_touchDevices[index]->name = SDL_strdup(name ? name : "");

    return index;
}

// Set or update the name of a touch.
void SDL_SetTouchName(SDL_TouchID id, const char *name)
{
    SDL_Touch *touch = SDL_GetTouch(id);
    if (touch) {
        SDL_free(touch->name);
        touch->name = SDL_strdup(name ? name : "");
    }
}

static bool SDL_AddFinger(SDL_Touch *touch, SDL_FingerID fingerid, float x, float y, float pressure)
{
    SDL_Finger *finger;

    SDL_assert(fingerid != 0);

    if (touch->num_fingers == touch->max_fingers) {
        SDL_Finger **new_fingers;
        new_fingers = (SDL_Finger **)SDL_realloc(touch->fingers, (touch->max_fingers + 1) * sizeof(*touch->fingers));
        if (!new_fingers) {
            return false;
        }
        touch->fingers = new_fingers;
        touch->fingers[touch->max_fingers] = (SDL_Finger *)SDL_malloc(sizeof(*finger));
        if (!touch->fingers[touch->max_fingers]) {
            return false;
        }
        touch->max_fingers++;
    }

    finger = touch->fingers[touch->num_fingers++];
    finger->id = fingerid;
    finger->x = x;
    finger->y = y;
    finger->pressure = pressure;
    return true;
}

static void SDL_DelFinger(SDL_Touch *touch, SDL_FingerID fingerid)
{
    int index = SDL_GetFingerIndex(touch, fingerid);
    if (index < 0) {
        return;
    }

    --touch->num_fingers;
    if (index < (touch->num_fingers)) {
        // Move the deleted finger to just past the end of the active fingers array and shift the active fingers by one.
        // This ensures that the descriptor for the now-deleted finger is located at `touch->fingers[touch->num_fingers]`
        // and is ready for use in SDL_AddFinger.
        SDL_Finger *deleted_finger = touch->fingers[index];
        SDL_memmove(&touch->fingers[index], &touch->fingers[index + 1], (touch->num_fingers - index) * sizeof(touch->fingers[index]));
        touch->fingers[touch->num_fingers] = deleted_finger;
    }
}

void SDL_SendTouch(Uint64 timestamp, SDL_TouchID id, SDL_FingerID fingerid, SDL_Window *window, SDL_EventType type, float x, float y, float pressure)
{
    SDL_Finger *finger;
    bool down = (type == SDL_EVENT_FINGER_DOWN);

    SDL_Touch *touch = SDL_GetTouch(id);
    if (!touch) {
        return;
    }

    SDL_Mouse *mouse = SDL_GetMouse();

    // SDL_HINT_TOUCH_MOUSE_EVENTS: controlling whether touch events should generate synthetic mouse events
    // SDL_HINT_VITA_TOUCH_MOUSE_DEVICE: controlling which touchpad should generate synthetic mouse events, PSVita-only
    {
        // FIXME: maybe we should only restrict to a few SDL_TouchDeviceType
        if ((id != SDL_MOUSE_TOUCHID) && (id != SDL_PEN_TOUCHID)) {
#ifdef SDL_PLATFORM_VITA
            if (mouse->touch_mouse_events && ((mouse->vita_touch_mouse_device == id) || (mouse->vita_touch_mouse_device == 3))) {
#else
            if (mouse->touch_mouse_events) {
#endif
                if (window) {
                    if (down) {
                        if (finger_touching == false) {
                            float pos_x = (x * (float)window->w);
                            float pos_y = (y * (float)window->h);
                            if (pos_x < 0) {
                                pos_x = 0;
                            }
                            if (pos_x > (float)(window->w - 1)) {
                                pos_x = (float)(window->w - 1);
                            }
                            if (pos_y < 0.0f) {
                                pos_y = 0.0f;
                            }
                            if (pos_y > (float)(window->h - 1)) {
                                pos_y = (float)(window->h - 1);
                            }
                            SDL_SendMouseMotion(timestamp, window, SDL_TOUCH_MOUSEID, false, pos_x, pos_y);
                            SDL_SendMouseButton(timestamp, window, SDL_TOUCH_MOUSEID, SDL_BUTTON_LEFT, true);
                        }
                    } else {
                        if (finger_touching == true && track_touchid == id && track_fingerid == fingerid) {
                            SDL_SendMouseButton(timestamp, window, SDL_TOUCH_MOUSEID, SDL_BUTTON_LEFT, false);
                        }
                    }
                }
                if (down) {
                    if (finger_touching == false) {
                        finger_touching = true;
                        track_touchid = id;
                        track_fingerid = fingerid;
                    }
                } else {
                    if (finger_touching == true && track_touchid == id && track_fingerid == fingerid) {
                        finger_touching = false;
                    }
                }
            }
        }
    }

    // SDL_HINT_MOUSE_TOUCH_EVENTS: if not set, discard synthetic touch events coming from platform layer
    if (!mouse->mouse_touch_events && (id == SDL_MOUSE_TOUCHID)) {
        return;
    } else if (!mouse->pen_touch_events && (id == SDL_PEN_TOUCHID)) {
        return;
    }

    finger = SDL_GetFinger(touch, fingerid);
    if (down) {
        if (finger) {
            /* This finger is already down.
               Assume the finger-up for the previous touch was lost, and send it. */
            SDL_SendTouch(timestamp, id, fingerid, window, SDL_EVENT_FINGER_CANCELED, x, y, pressure);
        }

        if (!SDL_AddFinger(touch, fingerid, x, y, pressure)) {
            return;
        }

        if (SDL_EventEnabled(type)) {
            SDL_Event event;
            event.type = type;
            event.common.timestamp = timestamp;
            event.tfinger.touchID = id;
            event.tfinger.fingerID = fingerid;
            event.tfinger.x = x;
            event.tfinger.y = y;
            event.tfinger.dx = 0;
            event.tfinger.dy = 0;
            event.tfinger.pressure = pressure;
            event.tfinger.windowID = window ? SDL_GetWindowID(window) : 0;
            SDL_PushEvent(&event);
        }
    } else {
        if (!finger) {
            // This finger is already up
            return;
        }

        if (SDL_EventEnabled(type)) {
            SDL_Event event;
            event.type = type;
            event.common.timestamp = timestamp;
            event.tfinger.touchID = id;
            event.tfinger.fingerID = fingerid;
            // I don't trust the coordinates passed on fingerUp
            event.tfinger.x = finger->x;
            event.tfinger.y = finger->y;
            event.tfinger.dx = 0;
            event.tfinger.dy = 0;
            event.tfinger.pressure = pressure;
            event.tfinger.windowID = window ? SDL_GetWindowID(window) : 0;
            SDL_PushEvent(&event);
        }

        SDL_DelFinger(touch, fingerid);
    }
}

void SDL_SendTouchMotion(Uint64 timestamp, SDL_TouchID id, SDL_FingerID fingerid, SDL_Window *window,
                        float x, float y, float pressure)
{
    SDL_Touch *touch;
    SDL_Finger *finger;
    float xrel, yrel, prel;

    touch = SDL_GetTouch(id);
    if (!touch) {
        return;
    }

    SDL_Mouse *mouse = SDL_GetMouse();

    // SDL_HINT_TOUCH_MOUSE_EVENTS: controlling whether touch events should generate synthetic mouse events
    {
        if ((id != SDL_MOUSE_TOUCHID) && (id != SDL_PEN_TOUCHID)) {
            if (mouse->touch_mouse_events) {
                if (window) {
                    if (finger_touching == true && track_touchid == id && track_fingerid == fingerid) {
                        float pos_x = (x * (float)window->w);
                        float pos_y = (y * (float)window->h);
                        if (pos_x < 0.0f) {
                            pos_x = 0.0f;
                        }
                        if (pos_x > (float)(window->w - 1)) {
                            pos_x = (float)(window->w - 1);
                        }
                        if (pos_y < 0.0f) {
                            pos_y = 0.0f;
                        }
                        if (pos_y > (float)(window->h - 1)) {
                            pos_y = (float)(window->h - 1);
                        }
                        SDL_SendMouseMotion(timestamp, window, SDL_TOUCH_MOUSEID, false, pos_x, pos_y);
                    }
                }
            }
        }
    }

    // SDL_HINT_MOUSE_TOUCH_EVENTS: if not set, discard synthetic touch events coming from platform layer
    if (mouse->mouse_touch_events == 0) {
        if (id == SDL_MOUSE_TOUCHID) {
            return;
        }
    }

    finger = SDL_GetFinger(touch, fingerid);
    if (!finger) {
        SDL_SendTouch(timestamp, id, fingerid, window, SDL_EVENT_FINGER_DOWN, x, y, pressure);
        return;
    }

    xrel = x - finger->x;
    yrel = y - finger->y;
    prel = pressure - finger->pressure;

    // Drop events that don't change state
    if (xrel == 0.0f && yrel == 0.0f && prel == 0.0f) {
#if 0
        printf("Touch event didn't change state - dropped!\n");
#endif
        return;
    }

    // Update internal touch coordinates
    finger->x = x;
    finger->y = y;
    finger->pressure = pressure;

    // Post the event, if desired
    if (SDL_EventEnabled(SDL_EVENT_FINGER_MOTION)) {
        SDL_Event event;
        event.type = SDL_EVENT_FINGER_MOTION;
        event.common.timestamp = timestamp;
        event.tfinger.touchID = id;
        event.tfinger.fingerID = fingerid;
        event.tfinger.x = x;
        event.tfinger.y = y;
        event.tfinger.dx = xrel;
        event.tfinger.dy = yrel;
        event.tfinger.pressure = pressure;
        event.tfinger.windowID = window ? SDL_GetWindowID(window) : 0;
        SDL_PushEvent(&event);
    }
}

void SDL_DelTouch(SDL_TouchID id)
{
    int i, index;
    SDL_Touch *touch;

    if (SDL_num_touch == 0) {
        // We've already cleaned up, we won't find this device
        return;
    }

    index = SDL_GetTouchIndex(id);
    touch = SDL_GetTouch(id);
    if (!touch) {
        return;
    }

    for (i = 0; i < touch->max_fingers; ++i) {
        SDL_free(touch->fingers[i]);
    }
    SDL_free(touch->fingers);
    SDL_free(touch->name);
    SDL_free(touch);

    SDL_num_touch--;
    SDL_touchDevices[index] = SDL_touchDevices[SDL_num_touch];
}

void SDL_QuitTouch(void)
{
    int i;

    for (i = SDL_num_touch; i--;) {
        SDL_DelTouch(SDL_touchDevices[i]->id);
    }
    SDL_assert(SDL_num_touch == 0);

    SDL_free(SDL_touchDevices);
    SDL_touchDevices = NULL;
}
