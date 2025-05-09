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
#include <sys/time.h>
#include <dev/wscons/wsconsio.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdlib.h>
#include <fcntl.h>

#include "../../events/SDL_mouse_c.h"

typedef struct SDL_WSCONS_mouse_input_data
{
    int fd;
    SDL_MouseID mouseID;
} SDL_WSCONS_mouse_input_data;

SDL_WSCONS_mouse_input_data *SDL_WSCONS_Init_Mouse(void)
{
#ifdef WSMOUSEIO_SETVERSION
    int version = WSMOUSE_EVENT_VERSION;
#endif
    SDL_WSCONS_mouse_input_data *input = SDL_calloc(1, sizeof(SDL_WSCONS_mouse_input_data));

    if (!input) {
        return NULL;
    }
    input->fd = open("/dev/wsmouse", O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (input->fd == -1) {
        SDL_free(input);
        return NULL;
    }

    input->mouseID = SDL_GetNextObjectID();
    SDL_AddMouse(input->mouseID, NULL, false);

#ifdef WSMOUSEIO_SETMODE
    ioctl(input->fd, WSMOUSEIO_SETMODE, WSMOUSE_COMPAT);
#endif
#ifdef WSMOUSEIO_SETVERSION
    ioctl(input->fd, WSMOUSEIO_SETVERSION, &version);
#endif
    return input;
}

static Uint64 GetEventTimestamp(struct timespec *time)
{
    // FIXME: Get the event time in the SDL tick time base
    return SDL_GetTicksNS();
}

void updateMouse(SDL_WSCONS_mouse_input_data *input)
{
    struct wscons_event events[64];
    int n;
    SDL_Mouse *mouse = SDL_GetMouse();

    if ((n = read(input->fd, events, sizeof(events))) > 0) {
        int i;
        n /= sizeof(struct wscons_event);
        for (i = 0; i < n; i++) {
            Uint64 timestamp = GetEventTimestamp(&events[i].time);
            int type = events[i].type;
            switch (type) {
            case WSCONS_EVENT_MOUSE_DOWN:
            case WSCONS_EVENT_MOUSE_UP:
            {
                Uint8 button = SDL_BUTTON_LEFT + events[i].value;
                bool down = (type == WSCONS_EVENT_MOUSE_DOWN);
                SDL_SendMouseButton(timestamp, mouse->focus, input->mouseID, button, down);
                break;
            }
            case WSCONS_EVENT_MOUSE_DELTA_X:
            {
                SDL_SendMouseMotion(timestamp, mouse->focus, input->mouseID, true, (float)events[i].value, 0.0f);
                break;
            }
            case WSCONS_EVENT_MOUSE_DELTA_Y:
            {
                SDL_SendMouseMotion(timestamp, mouse->focus, input->mouseID, true, 0.0f, -(float)events[i].value);
                break;
            }
            case WSCONS_EVENT_MOUSE_DELTA_W:
            {
                SDL_SendMouseWheel(timestamp, mouse->focus, input->mouseID, events[i].value, 0, SDL_MOUSEWHEEL_NORMAL);
                break;
            }
            case WSCONS_EVENT_MOUSE_DELTA_Z:
            {
                SDL_SendMouseWheel(timestamp, mouse->focus, input->mouseID, 0, -events[i].value, SDL_MOUSEWHEEL_NORMAL);
                break;
            }
            default:
                break;
            }
        }
    }
}

void SDL_WSCONS_Quit_Mouse(SDL_WSCONS_mouse_input_data *input)
{
    if (!input) {
        return;
    }
    close(input->fd);
    SDL_free(input);
}
