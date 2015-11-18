/*************************************************************************/
/*  joystick.cpp                                                         */
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
#include "joystick.h"
#include <iostream>
#include <wbemidl.h>
#include <oleauto.h>

joystick_windows::joystick_windows() {

}

joystick_windows::joystick_windows(InputDefault* _input, HWND* hwnd) {

    input = _input;
    hWnd = hwnd;
    joystick_count = 0;
    dinput = NULL;

    for (int i = 0; i < JOYSTICKS_MAX; i++)
        attached_joysticks[i] = false;


    HRESULT result;
    result = DirectInput8Create(GetModuleHandle(NULL), DIRECTINPUT_VERSION, IID_IDirectInput8, (void**)&dinput, NULL);
    if (FAILED(result)) {
        printf("failed init DINPUT: %ld\n", result);
    }

    probe_joysticks();
}

joystick_windows::~joystick_windows() {

    close_joystick();
    dinput->Release();
}


bool joystick_windows::have_device(const GUID &p_guid) {

    for (int i = 0; i < JOYSTICKS_MAX; i++) {

        if (d_joysticks[i].guid == p_guid) {

            d_joysticks[i].confirmed = true;
            return true;
        }
    }
    return false;
}

int joystick_windows::check_free_joy_slot() const {

    for (int i = 0; i < JOYSTICKS_MAX; i++) {

        if (!attached_joysticks[i])
            return i;
    }
    return -1;
}


// Taken from MSDN
bool joystick_windows::is_xinput_device(const GUID *p_guid) {

    IWbemLocator*           pIWbemLocator  = NULL;
    IEnumWbemClassObject*   pEnumDevices   = NULL;
    IWbemClassObject*       pDevices[20]   = {0};
    IWbemServices*          pIWbemServices = NULL;
    BSTR                    bstrNamespace  = NULL;
    BSTR                    bstrDeviceID   = NULL;
    BSTR                    bstrClassName  = NULL;
    DWORD                   uReturned      = 0;
    bool                    bIsXinputDevice= false;
    UINT                    iDevice        = 0;
    VARIANT                 var;
    HRESULT                 hr;

    // CoInit if needed
    hr = CoInitialize(NULL);
    bool bCleanupCOM = SUCCEEDED(hr);

    // Create WMI
    hr = CoCreateInstance( __uuidof(WbemLocator),
                           NULL,
                           CLSCTX_INPROC_SERVER,
                           __uuidof(IWbemLocator),
                           (LPVOID*) &pIWbemLocator);
    if( FAILED(hr) || pIWbemLocator == NULL )
        goto LCleanup;

    bstrNamespace = SysAllocString( L"\\\\.\\root\\cimv2" );if( bstrNamespace == NULL ) goto LCleanup;
    bstrClassName = SysAllocString( L"Win32_PNPEntity" );   if( bstrClassName == NULL ) goto LCleanup;
    bstrDeviceID  = SysAllocString( L"DeviceID" );          if( bstrDeviceID == NULL )  goto LCleanup;

    // Connect to WMI
    hr = pIWbemLocator->ConnectServer( bstrNamespace, NULL, NULL, 0L,
                                       0L, NULL, NULL, &pIWbemServices );
    if( FAILED(hr) || pIWbemServices == NULL )
        goto LCleanup;

    // Switch security level to IMPERSONATE.
    CoSetProxyBlanket( pIWbemServices, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, NULL,
                       RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE, NULL, EOAC_NONE );

    hr = pIWbemServices->CreateInstanceEnum( bstrClassName, 0, NULL, &pEnumDevices );
    if( FAILED(hr) || pEnumDevices == NULL )
        goto LCleanup;

    // Loop over all devices
    for( ;; )
    {
        // Get 20 at a time
        hr = pEnumDevices->Next( 10000, 20, pDevices, &uReturned );
        if( FAILED(hr) )
            goto LCleanup;
        if( uReturned == 0 )
            break;

        for( iDevice=0; iDevice<uReturned; iDevice++ )
        {
            // For each device, get its device ID
            hr = pDevices[iDevice]->Get( bstrDeviceID, 0L, &var, NULL, NULL );
            if( SUCCEEDED( hr ) && var.vt == VT_BSTR && var.bstrVal != NULL )
            {
                // Check if the device ID contains "IG_".  If it does, then it's an XInput device
                // This information can not be found from DirectInput
                if( wcsstr( var.bstrVal, L"IG_" ) )
                {
                    // If it does, then get the VID/PID from var.bstrVal
                    DWORD dwPid = 0, dwVid = 0;
                    WCHAR* strVid = wcsstr( var.bstrVal, L"VID_" );
                    if( strVid && swscanf( strVid, L"VID_%4X", &dwVid ) != 1 )
                        dwVid = 0;
                    WCHAR* strPid = wcsstr( var.bstrVal, L"PID_" );
                    if( strPid && swscanf( strPid, L"PID_%4X", &dwPid ) != 1 )
                        dwPid = 0;

                    // Compare the VID/PID to the DInput device
                    DWORD dwVidPid = MAKELONG( dwVid, dwPid );
                    if( dwVidPid == p_guid->Data1 )
                    {
                        bIsXinputDevice = true;
                        goto LCleanup;
                    }
                }
            }
            SAFE_RELEASE( pDevices[iDevice] );
        }
    }

LCleanup:
    if(bstrNamespace)
        SysFreeString(bstrNamespace);
    if(bstrDeviceID)
        SysFreeString(bstrDeviceID);
    if(bstrClassName)
        SysFreeString(bstrClassName);
    for( iDevice=0; iDevice<20; iDevice++ )
        SAFE_RELEASE( pDevices[iDevice] );
    SAFE_RELEASE( pEnumDevices );
    SAFE_RELEASE( pIWbemLocator );
    SAFE_RELEASE( pIWbemServices );

    if( bCleanupCOM )
        CoUninitialize();

    return bIsXinputDevice;
}


bool joystick_windows::setup_dinput_joystick(const DIDEVICEINSTANCE* instance) {

    HRESULT hr;
    int num = check_free_joy_slot();

    if (have_device(instance->guidInstance) || num == -1)
        return false;

    d_joysticks[joystick_count] = dinput_gamepad();
    dinput_gamepad* joy = &d_joysticks[num];

    const DWORD devtype = (instance->dwDevType & 0xFF);

    if ((devtype != DI8DEVTYPE_JOYSTICK) && (devtype != DI8DEVTYPE_GAMEPAD) && (devtype != DI8DEVTYPE_1STPERSON)) {
        //printf("ignore device %s, type %x\n", instance->tszProductName, devtype);
        return false;
    }

    hr = dinput->CreateDevice(instance->guidInstance, &joy->di_joy, NULL);

    if (FAILED(hr)) {

        //std::wcout << "failed to create device: " << instance->tszProductName << std::endl;
        return false;
    }

    const GUID &guid = instance->guidProduct;
    char uid[128];
    sprintf(uid, "%08lx%04hx%04hx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx",
            _byteswap_ulong(guid.Data1), guid.Data2, guid.Data3,
            guid.Data4[0], guid.Data4[1], guid.Data4[2], guid.Data4[3],
            guid.Data4[4], guid.Data4[5], guid.Data4[6], guid.Data4[7]);

    id_to_change = num;
    joy->di_joy->SetDataFormat(&c_dfDIJoystick2);
    joy->di_joy->SetCooperativeLevel(*hWnd, DISCL_FOREGROUND);
    joy->di_joy->EnumObjects(objectsCallback, this, NULL);
    joy->joy_axis.sort();

    joy->guid = instance->guidInstance;
    input->joy_connection_changed(num, true, instance->tszProductName, uid);
    joy->attached = true;
    joy->id = num;
    attached_joysticks[num] = true;
    joy->confirmed = true;
    joystick_count++;
    return true;
}

void joystick_windows::setup_joystick_object(const DIDEVICEOBJECTINSTANCE *ob, int p_joy_id) {

    if (ob->dwType & DIDFT_AXIS) {

        HRESULT res;
        DIPROPRANGE prop_range;
        DIPROPDWORD dilong;
        DWORD ofs;
        if (ob->guidType == GUID_XAxis)
            ofs = DIJOFS_X;
        else if (ob->guidType == GUID_YAxis)
            ofs = DIJOFS_Y;
        else if (ob->guidType == GUID_ZAxis)
            ofs = DIJOFS_Z;
        else if (ob->guidType == GUID_RxAxis)
            ofs = DIJOFS_RX;
        else if (ob->guidType == GUID_RyAxis)
            ofs = DIJOFS_RY;
        else if (ob->guidType == GUID_RzAxis)
            ofs = DIJOFS_RZ;
        else if (ob->guidType == GUID_Slider)
            ofs = DIJOFS_SLIDER(0);
        else
            return;
        prop_range.diph.dwSize = sizeof(DIPROPRANGE);
        prop_range.diph.dwHeaderSize = sizeof(DIPROPHEADER);
        prop_range.diph.dwObj = ob->dwType;
        prop_range.diph.dwHow = DIPH_BYID;
        prop_range.lMin = -MAX_JOY_AXIS;
        prop_range.lMax = +MAX_JOY_AXIS;

        dinput_gamepad &joy = d_joysticks[p_joy_id];


        res = joy.di_joy->SetProperty(DIPROP_RANGE, &prop_range.diph);
        if (FAILED(res))
            return;

        dilong.diph.dwSize = sizeof(dilong);
        dilong.diph.dwHeaderSize = sizeof(dilong.diph);
        dilong.diph.dwObj = ob->dwType;
        dilong.diph.dwHow = DIPH_BYID;
        dilong.dwData = 0;

        res = IDirectInputDevice8_SetProperty(joy.di_joy, DIPROP_DEADZONE, &dilong.diph);
        if (FAILED(res))
            return;

        joy.joy_axis.push_back(ofs);
    }
}

BOOL CALLBACK joystick_windows::enumCallback(const DIDEVICEINSTANCE* instance, void* pContext) {


    joystick_windows* self = (joystick_windows*)pContext;
    if (self->is_xinput_device(&instance->guidProduct)) {;
        return DIENUM_CONTINUE;
    }
    self->setup_dinput_joystick(instance);
    return DIENUM_CONTINUE;
}

BOOL CALLBACK joystick_windows::objectsCallback(const DIDEVICEOBJECTINSTANCE *instance, void *context) {

    joystick_windows* self = (joystick_windows*)context;
    self->setup_joystick_object(instance, self->id_to_change);

    return DIENUM_CONTINUE;
}

void joystick_windows::close_joystick(int id) {

    if (id == -1) {

        for (int i = 0; i < JOYSTICKS_MAX; i++) {

            close_joystick(i);
        }
        return;
    }

    if (!d_joysticks[id].attached) return;

    d_joysticks[id].di_joy->Unacquire();
    d_joysticks[id].di_joy->Release();
    d_joysticks[id].attached = false;
    attached_joysticks[d_joysticks[id].id] = false;
    d_joysticks[id].guid.Data1 = d_joysticks[id].guid.Data2 = d_joysticks[id].guid.Data3 = 0;
    input->joy_connection_changed(id, false, "");
    joystick_count--;
}

void joystick_windows::probe_joysticks() {

    DWORD dwResult;
    for (DWORD i = 0; i < XUSER_MAX_COUNT; i++) {

        ZeroMemory(&x_joysticks[i].state, sizeof(XINPUT_STATE));

        dwResult = XInputGetState(i, &x_joysticks[i].state);
        if ( dwResult == ERROR_SUCCESS) {

            int id = check_free_joy_slot();
            if (id != -1 && !x_joysticks[i].attached) {

                x_joysticks[i].attached = true;
                x_joysticks[i].id = id;
                attached_joysticks[id] = true;
                input->joy_connection_changed(id, true, "XInput Gamepad","__XINPUT_DEVICE__");
            }
        }
        else {

            if (x_joysticks[i].attached) {

                x_joysticks[i].attached = false;
                attached_joysticks[x_joysticks[i].id] = false;
                input->joy_connection_changed(x_joysticks[i].id, false, "");
            }
        }
    }

    for (int i = 0; i < joystick_count; i++) {

        d_joysticks[i].confirmed = false;
    }

    dinput->EnumDevices(DI8DEVCLASS_GAMECTRL, enumCallback, this, DIEDFL_ATTACHEDONLY);

    for (int i = 0; i < joystick_count; i++) {

        if (!d_joysticks[i].confirmed) {

            close_joystick(i);
        }
    }
}

unsigned int joystick_windows::process_joysticks(unsigned int p_last_id) {

    HRESULT hr;

    for (int i = 0; i < XUSER_MAX_COUNT; i++) {

        xinput_gamepad &joy = x_joysticks[i];
        if (!joy.attached) {
            continue;
        }
        ZeroMemory(&joy.state, sizeof(XINPUT_STATE));

        XInputGetState(i, &joy.state);
        if (joy.state.dwPacketNumber != joy.last_packet) {

            int button_mask = XINPUT_GAMEPAD_DPAD_UP;
            for (int i = 0; i <= 16; i++) {

                p_last_id = input->joy_button(p_last_id, joy.id, i, joy.state.Gamepad.wButtons & button_mask);
                button_mask = button_mask * 2;
            }

            int ly = joy.state.Gamepad.sThumbLY;
            int ry = joy.state.Gamepad.sThumbRY;
            p_last_id = input->joy_axis(p_last_id, joy.id, JOY_AXIS_0,  axis_correct(joy.state.Gamepad.sThumbLX, true));
            p_last_id = input->joy_axis(p_last_id, joy.id, JOY_AXIS_1,  (Math::abs(ly) < 20) ? 0.0f : -axis_correct(ly, true));
            p_last_id = input->joy_axis(p_last_id, joy.id, JOY_AXIS_2,  axis_correct(joy.state.Gamepad.sThumbRX, true));
            p_last_id = input->joy_axis(p_last_id, joy.id, JOY_AXIS_3,  (Math::abs(ry) < 20) ? 0.0f : -axis_correct(ry, true));
            p_last_id = input->joy_axis(p_last_id, joy.id, JOY_AXIS_4,  axis_correct(joy.state.Gamepad.bLeftTrigger, true, true));
            p_last_id = input->joy_axis(p_last_id, joy.id, JOY_AXIS_5,  axis_correct(joy.state.Gamepad.bRightTrigger, true, true));
            joy.last_packet = joy.state.dwPacketNumber;
        }
    }

    for (int i = 0; i < JOYSTICKS_MAX; i++) {

        dinput_gamepad* joy = &d_joysticks[i];

        if (!joy->attached)
            continue;

        DIJOYSTATE2 js;
        hr = joy->di_joy->Poll();
        if (hr == DIERR_INPUTLOST || hr == DIERR_NOTACQUIRED) {
            IDirectInputDevice8_Acquire(joy->di_joy);
            joy->di_joy->Poll();
        }
        if (FAILED(hr = d_joysticks[i].di_joy->GetDeviceState(sizeof(DIJOYSTATE2), &js))) {

            //printf("failed to read joy #%d\n", i);
            continue;
        }

        p_last_id = post_hat(p_last_id, i, js.rgdwPOV[0]);

        for (int j = 0; j < 128; j++) {

            if (js.rgbButtons[j] & 0x80) {

                if (!joy->last_buttons[j]) {

                    p_last_id = input->joy_button(p_last_id, i, j, true);
                    joy->last_buttons[j] = true;
                }
            }
            else {

                if (joy->last_buttons[j]) {

                    p_last_id = input->joy_button(p_last_id, i, j, false);
                    joy->last_buttons[j] = false;
                }
            }
        }

        for (int j = 0; j < joy->joy_axis.size(); j++) {

            switch (joy->joy_axis[j]) {

            case DIJOFS_X:
                p_last_id = input->joy_axis(p_last_id, i, j, axis_correct(js.lX));
                break;
            case DIJOFS_Y:
                p_last_id = input->joy_axis(p_last_id, i, j, axis_correct(js.lY));
                break;
            case DIJOFS_Z:
                p_last_id = input->joy_axis(p_last_id, i, j, axis_correct(js.lZ));
                break;
            case DIJOFS_RX:
                p_last_id = input->joy_axis(p_last_id, i, j, axis_correct(js.lRx));
                break;
            case DIJOFS_RY:
                p_last_id = input->joy_axis(p_last_id, i, j, axis_correct(js.lRy));
                break;
            case DIJOFS_RZ:
                p_last_id = input->joy_axis(p_last_id, i, j, axis_correct(js.lRz));
                break;
            }
        }

    }
    return p_last_id;
}

unsigned int joystick_windows::post_hat(unsigned int p_last_id, int p_device, DWORD p_dpad) {

    int dpad_val = 0;

    if (p_dpad == -1) {
        dpad_val = InputDefault::HAT_MASK_CENTER;
    }
    if (p_dpad == 0) {

        dpad_val = InputDefault::HAT_MASK_UP;

    }
    else if (p_dpad == 4500) {

        dpad_val = (InputDefault::HAT_MASK_UP | InputDefault::HAT_MASK_RIGHT);

    }
    else if (p_dpad == 9000) {

        dpad_val = InputDefault::HAT_MASK_RIGHT;

    }
    else if (p_dpad == 13500) {

        dpad_val = (InputDefault::HAT_MASK_RIGHT | InputDefault::HAT_MASK_DOWN);

    }
    else if (p_dpad == 18000) {

        dpad_val = InputDefault::HAT_MASK_DOWN;

    }
    else if (p_dpad == 22500) {

        dpad_val = (InputDefault::HAT_MASK_DOWN | InputDefault::HAT_MASK_LEFT);

    }
    else if (p_dpad == 27000) {

        dpad_val = InputDefault::HAT_MASK_LEFT;

    }
    else if (p_dpad == 31500) {

        dpad_val = (InputDefault::HAT_MASK_LEFT | InputDefault::HAT_MASK_UP);
    }
    return input->joy_hat(p_last_id, p_device, dpad_val);
};

float joystick_windows::axis_correct(int p_val, bool p_xinput, bool p_trigger) const {

    if (Math::abs(p_val) < MIN_JOY_AXIS) {

        return 0.0f;
    }

    if (p_xinput) {

        if (p_trigger) {
            return (float)p_val / MAX_TRIGGER;
        }

        if (p_val < 0) {
            return (float)p_val / MAX_JOY_AXIS;
        }
        else {
            return (float)p_val / (MAX_JOY_AXIS - 1);
        }
    }
    return (float)p_val / MAX_JOY_AXIS;
}
