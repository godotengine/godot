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

#ifdef SDL_VIDEO_DRIVER_PSP

/* Being a null driver, there's no event stream. We just define stubs for
   most of the API. */

#include "../../events/SDL_events_c.h"
#include "../../events/SDL_keyboard_c.h"
#include "../SDL_sysvideo.h"
#include "SDL_pspvideo.h"
#include "SDL_pspevents_c.h"
#include "../../thread/SDL_systhread.h"
#include <psphprm.h>
#include <pspthreadman.h>

#ifdef PSPIRKEYB
#include <pspirkeyb.h>
#include <pspirkeyb_rawkeys.h>

#define IRKBD_CONFIG_FILE NULL // this will take ms0:/seplugins/pspirkeyb.ini

static int irkbd_ready = 0;
static SDL_Scancode keymap[256];
#endif

static enum PspHprmKeys hprm = 0;
static SDL_Semaphore *event_sem = NULL;
static SDL_Thread *thread = NULL;
static int running = 0;
static struct
{
    enum PspHprmKeys id;
    SDL_Scancode scancode;
} keymap_psp[] = {
    { PSP_HPRM_PLAYPAUSE, SDL_SCANCODE_F10 },
    { PSP_HPRM_FORWARD, SDL_SCANCODE_F11 },
    { PSP_HPRM_BACK, SDL_SCANCODE_F12 },
    { PSP_HPRM_VOL_UP, SDL_SCANCODE_F13 },
    { PSP_HPRM_VOL_DOWN, SDL_SCANCODE_F14 },
    { PSP_HPRM_HOLD, SDL_SCANCODE_F15 }
};

int EventUpdate(void *data)
{
    while (running) {
        SDL_WaitSemaphore(event_sem);
        sceHprmPeekCurrentKey((u32 *)&hprm);
        SDL_SignalSemaphore(event_sem);
        // Delay 1/60th of a second
        sceKernelDelayThread(1000000 / 60);
    }
    return 0;
}

void PSP_PumpEvents(SDL_VideoDevice *_this)
{
    int i;
    enum PspHprmKeys keys;
    enum PspHprmKeys changed;
    static enum PspHprmKeys old_keys = 0;

    SDL_WaitSemaphore(event_sem);
    keys = hprm;
    SDL_SignalSemaphore(event_sem);

    // HPRM Keyboard
    changed = old_keys ^ keys;
    old_keys = keys;
    if (changed) {
        for (i = 0; i < sizeof(keymap_psp) / sizeof(keymap_psp[0]); i++) {
            if (changed & keymap_psp[i].id) {
                bool down = ((keys & keymap_psp[i].id) != 0);
                SDL_SendKeyboardKey(0, SDL_GLOBAL_KEYBOARD_ID, keymap_psp[i].id, keymap_psp[i].scancode, down);
            }
        }
    }

#ifdef PSPIRKEYB
    if (irkbd_ready) {
        unsigned char buffer[255];
        int i, length, count;
        SIrKeybScanCodeData *scanData;

        if (pspIrKeybReadinput(buffer, &length) >= 0) {
            if ((length % sizeof(SIrKeybScanCodeData)) == 0) {
                count = length / sizeof(SIrKeybScanCodeData);
                for (i = 0; i < count; i++) {
                    unsigned char raw;
                    bool down;
                    scanData = (SIrKeybScanCodeData *)buffer + i;
                    raw = scanData->raw;
                    down = (scanData->pressed != 0);
                    SDL_SendKeyboardKey(0, SDL_GLOBAL_KEYBOARD_ID, raw, keymap[raw], down);
                }
            }
        }
    }
#endif
    sceKernelDelayThread(0);

    return;
}

void PSP_InitOSKeymap(SDL_VideoDevice *_this)
{
#ifdef PSPIRKEYB
    int i;
    for (i = 0; i < SDL_arraysize(keymap); ++i) {
        keymap[i] = SDL_SCANCODE_UNKNOWN;
    }

    keymap[KEY_ESC] = SDL_SCANCODE_ESCAPE;

    keymap[KEY_F1] = SDL_SCANCODE_F1;
    keymap[KEY_F2] = SDL_SCANCODE_F2;
    keymap[KEY_F3] = SDL_SCANCODE_F3;
    keymap[KEY_F4] = SDL_SCANCODE_F4;
    keymap[KEY_F5] = SDL_SCANCODE_F5;
    keymap[KEY_F6] = SDL_SCANCODE_F6;
    keymap[KEY_F7] = SDL_SCANCODE_F7;
    keymap[KEY_F8] = SDL_SCANCODE_F8;
    keymap[KEY_F9] = SDL_SCANCODE_F9;
    keymap[KEY_F10] = SDL_SCANCODE_F10;
    keymap[KEY_F11] = SDL_SCANCODE_F11;
    keymap[KEY_F12] = SDL_SCANCODE_F12;
    keymap[KEY_F13] = SDL_SCANCODE_PRINT;
    keymap[KEY_F14] = SDL_SCANCODE_PAUSE;

    keymap[KEY_GRAVE] = SDL_SCANCODE_GRAVE;
    keymap[KEY_1] = SDL_SCANCODE_1;
    keymap[KEY_2] = SDL_SCANCODE_2;
    keymap[KEY_3] = SDL_SCANCODE_3;
    keymap[KEY_4] = SDL_SCANCODE_4;
    keymap[KEY_5] = SDL_SCANCODE_5;
    keymap[KEY_6] = SDL_SCANCODE_6;
    keymap[KEY_7] = SDL_SCANCODE_7;
    keymap[KEY_8] = SDL_SCANCODE_8;
    keymap[KEY_9] = SDL_SCANCODE_9;
    keymap[KEY_0] = SDL_SCANCODE_0;
    keymap[KEY_MINUS] = SDL_SCANCODE_MINUS;
    keymap[KEY_EQUAL] = SDL_SCANCODE_EQUALS;
    keymap[KEY_BACKSPACE] = SDL_SCANCODE_BACKSPACE;

    keymap[KEY_TAB] = SDL_SCANCODE_TAB;
    keymap[KEY_Q] = SDL_SCANCODE_q;
    keymap[KEY_W] = SDL_SCANCODE_w;
    keymap[KEY_E] = SDL_SCANCODE_e;
    keymap[KEY_R] = SDL_SCANCODE_r;
    keymap[KEY_T] = SDL_SCANCODE_t;
    keymap[KEY_Y] = SDL_SCANCODE_y;
    keymap[KEY_U] = SDL_SCANCODE_u;
    keymap[KEY_I] = SDL_SCANCODE_i;
    keymap[KEY_O] = SDL_SCANCODE_o;
    keymap[KEY_P] = SDL_SCANCODE_p;
    keymap[KEY_LEFTBRACE] = SDL_SCANCODE_LEFTBRACKET;
    keymap[KEY_RIGHTBRACE] = SDL_SCANCODE_RIGHTBRACKET;
    keymap[KEY_ENTER] = SDL_SCANCODE_RETURN;

    keymap[KEY_CAPSLOCK] = SDL_SCANCODE_CAPSLOCK;
    keymap[KEY_A] = SDL_SCANCODE_a;
    keymap[KEY_S] = SDL_SCANCODE_s;
    keymap[KEY_D] = SDL_SCANCODE_d;
    keymap[KEY_F] = SDL_SCANCODE_f;
    keymap[KEY_G] = SDL_SCANCODE_g;
    keymap[KEY_H] = SDL_SCANCODE_h;
    keymap[KEY_J] = SDL_SCANCODE_j;
    keymap[KEY_K] = SDL_SCANCODE_k;
    keymap[KEY_L] = SDL_SCANCODE_l;
    keymap[KEY_SEMICOLON] = SDL_SCANCODE_SEMICOLON;
    keymap[KEY_APOSTROPHE] = SDL_SCANCODE_APOSTROPHE;
    keymap[KEY_BACKSLASH] = SDL_SCANCODE_BACKSLASH;

    keymap[KEY_Z] = SDL_SCANCODE_z;
    keymap[KEY_X] = SDL_SCANCODE_x;
    keymap[KEY_C] = SDL_SCANCODE_c;
    keymap[KEY_V] = SDL_SCANCODE_v;
    keymap[KEY_B] = SDL_SCANCODE_b;
    keymap[KEY_N] = SDL_SCANCODE_n;
    keymap[KEY_M] = SDL_SCANCODE_m;
    keymap[KEY_COMMA] = SDL_SCANCODE_COMMA;
    keymap[KEY_DOT] = SDL_SCANCODE_PERIOD;
    keymap[KEY_SLASH] = SDL_SCANCODE_SLASH;

    keymap[KEY_SPACE] = SDL_SCANCODE_SPACE;

    keymap[KEY_UP] = SDL_SCANCODE_UP;
    keymap[KEY_DOWN] = SDL_SCANCODE_DOWN;
    keymap[KEY_LEFT] = SDL_SCANCODE_LEFT;
    keymap[KEY_RIGHT] = SDL_SCANCODE_RIGHT;

    keymap[KEY_HOME] = SDL_SCANCODE_HOME;
    keymap[KEY_END] = SDL_SCANCODE_END;
    keymap[KEY_INSERT] = SDL_SCANCODE_INSERT;
    keymap[KEY_DELETE] = SDL_SCANCODE_DELETE;

    keymap[KEY_NUMLOCK] = SDL_SCANCODE_NUMLOCK;
    keymap[KEY_LEFTMETA] = SDL_SCANCODE_LSUPER;

    keymap[KEY_KPSLASH] = SDL_SCANCODE_KP_DIVIDE;
    keymap[KEY_KPASTERISK] = SDL_SCANCODE_KP_MULTIPLY;
    keymap[KEY_KPMINUS] = SDL_SCANCODE_KP_MINUS;
    keymap[KEY_KPPLUS] = SDL_SCANCODE_KP_PLUS;
    keymap[KEY_KPDOT] = SDL_SCANCODE_KP_PERIOD;
    keymap[KEY_KPEQUAL] = SDL_SCANCODE_KP_EQUALS;

    keymap[KEY_LEFTCTRL] = SDL_SCANCODE_LCTRL;
    keymap[KEY_RIGHTCTRL] = SDL_SCANCODE_RCTRL;
    keymap[KEY_LEFTALT] = SDL_SCANCODE_LALT;
    keymap[KEY_RIGHTALT] = SDL_SCANCODE_RALT;
    keymap[KEY_LEFTSHIFT] = SDL_SCANCODE_LSHIFT;
    keymap[KEY_RIGHTSHIFT] = SDL_SCANCODE_RSHIFT;
#endif
}

bool PSP_EventInit(SDL_VideoDevice *_this)
{
#ifdef PSPIRKEYB
    int outputmode = PSP_IRKBD_OUTPUT_MODE_SCANCODE;
    int ret = pspIrKeybInit(IRKBD_CONFIG_FILE, 0);
    if (ret == PSP_IRKBD_RESULT_OK) {
        pspIrKeybOutputMode(outputmode);
        irkbd_ready = 1;
    } else {
        irkbd_ready = 0;
    }
#endif
    // Start thread to read data
    if ((event_sem = SDL_CreateSemaphore(1)) == NULL) {
        return SDL_SetError("Can't create input semaphore");
    }
    running = 1;
    if ((thread = SDL_CreateThreadWithStackSize(EventUpdate, "PSPInputThread", 4096, NULL)) == NULL) {
        return SDL_SetError("Can't create input thread");
    }
    return true;
}

void PSP_EventQuit(SDL_VideoDevice *_this)
{
    running = 0;
    SDL_WaitThread(thread, NULL);
    SDL_DestroySemaphore(event_sem);
#ifdef PSPIRKEYB
    if (irkbd_ready) {
        pspIrKeybFinish();
        irkbd_ready = 0;
    }
#endif
}

// end of SDL_pspevents.c ...

#endif // SDL_VIDEO_DRIVER_PSP
