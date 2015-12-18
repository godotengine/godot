/*************************************************************************/
/*  joystick.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
//author: Andreas Haas <hondres,  liugam3@gmail.com>
#ifndef JOYSTICK_H
#define JOYSTICK_H

#include "os_windows.h"
#define DIRECTINPUT_VERSION 0x0800
#include <dinput.h>
#include <xinput.h> // on unix the file is called "xinput.h", on windows I'm sure it won't mind

#ifndef SAFE_RELEASE            // when Windows Media Device M? is not present
#define SAFE_RELEASE(x) \
if(x != NULL)           \
{                       \
    x->Release();       \
    x = NULL;           \
}
#endif


class joystick_windows
{
public:
    joystick_windows();
    joystick_windows(InputDefault* _input, HWND* hwnd);
    ~joystick_windows();

    void probe_joysticks();
    unsigned int process_joysticks(unsigned int p_last_id);
private:

    enum {
        JOYSTICKS_MAX = 16,
        JOY_AXIS_COUNT = 6,
        MIN_JOY_AXIS = 10,
        MAX_JOY_AXIS = 32768,
        MAX_JOY_BUTTONS = 128,
        KEY_EVENT_BUFFER_SIZE = 512,
        MAX_TRIGGER = 255
    };

    struct dinput_gamepad {

        int id;
        bool attached;
        bool confirmed;
        bool last_buttons[MAX_JOY_BUTTONS];
        DWORD last_pad;

        LPDIRECTINPUTDEVICE8 di_joy;
        List<DWORD> joy_axis;
        GUID guid;

        dinput_gamepad() {
            id = -1;
            last_pad = -1;
            attached = false;
            confirmed = false;

            for (int i = 0; i < MAX_JOY_BUTTONS; i++)
                last_buttons[i] = false;
        }
    };

    struct xinput_gamepad {

        int id;
        bool attached;
        DWORD last_packet;
        XINPUT_STATE state;

        xinput_gamepad() {
            attached = false;
            last_packet = 0;
        }
    };

    typedef DWORD (WINAPI *XInputGetState_t) (DWORD dwUserIndex, XINPUT_STATE* pState);

    HWND* hWnd;
    HANDLE xinput_dll;
    LPDIRECTINPUT8 dinput;
    InputDefault* input;

    int id_to_change;
    int joystick_count;
    bool attached_joysticks[JOYSTICKS_MAX];
    dinput_gamepad d_joysticks[JOYSTICKS_MAX];
    xinput_gamepad x_joysticks[XUSER_MAX_COUNT];

    static BOOL CALLBACK enumCallback(const DIDEVICEINSTANCE* p_instance, void* p_context);
    static BOOL CALLBACK objectsCallback(const DIDEVICEOBJECTINSTANCE* instance, void* context);

    void setup_joystick_object(const DIDEVICEOBJECTINSTANCE* ob, int p_joy_id);
    void close_joystick(int id = -1);
    void load_xinput();
    void unload_xinput();

    int check_free_joy_slot() const;
    unsigned int post_hat(unsigned int p_last_id, int p_device, DWORD p_dpad);

    bool have_device(const GUID &p_guid);
    bool is_xinput_device(const GUID* p_guid);
    bool setup_dinput_joystick(const DIDEVICEINSTANCE* instance);

    InputDefault::JoyAxis axis_correct(int p_val, bool p_xinput = false, bool p_trigger = false, bool p_negate = false) const;
    XInputGetState_t xinput_get_state;
};

#endif


