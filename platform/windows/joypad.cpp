/*************************************************************************/
/*  joypad.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "joypad.h"
#include <oleauto.h>
#include <wbemidl.h>
#include <iostream>

#ifndef __GNUC__
#define __builtin_bswap32 _byteswap_ulong
#endif

DWORD WINAPI _xinput_get_state(DWORD dwUserIndex, XINPUT_STATE *pState) {
	return ERROR_DEVICE_NOT_CONNECTED;
}
DWORD WINAPI _xinput_set_state(DWORD dwUserIndex, XINPUT_VIBRATION *pVibration) {
	return ERROR_DEVICE_NOT_CONNECTED;
}

JoypadWindows::JoypadWindows() {
}

JoypadWindows::JoypadWindows(InputDefault *_input, HWND *hwnd) {

	input = _input;
	hWnd = hwnd;
	joypad_count = 0;
	dinput = NULL;
	xinput_dll = NULL;
	xinput_get_state = NULL;
	xinput_set_state = NULL;

	load_xinput();

	for (int i = 0; i < JOYPADS_MAX; i++)
		attached_joypads[i] = false;

	HRESULT result;
	result = DirectInput8Create(GetModuleHandle(NULL), DIRECTINPUT_VERSION, IID_IDirectInput8, (void **)&dinput, NULL);
	if (FAILED(result)) {
		printf("failed init DINPUT: %ld\n", result);
	}
	probe_joypads();
}

JoypadWindows::~JoypadWindows() {

	close_joypad();
	dinput->Release();
	unload_xinput();
}

bool JoypadWindows::have_device(const GUID &p_guid) {

	for (int i = 0; i < JOYPADS_MAX; i++) {

		if (d_joypads[i].guid == p_guid) {

			d_joypads[i].confirmed = true;
			return true;
		}
	}
	return false;
}

// adapted from SDL2, works a lot better than the MSDN version
bool JoypadWindows::is_xinput_device(const GUID *p_guid) {

	static GUID IID_ValveStreamingGamepad = { MAKELONG(0x28DE, 0x11FF), 0x0000, 0x0000, { 0x00, 0x00, 0x50, 0x49, 0x44, 0x56, 0x49, 0x44 } };
	static GUID IID_X360WiredGamepad = { MAKELONG(0x045E, 0x02A1), 0x0000, 0x0000, { 0x00, 0x00, 0x50, 0x49, 0x44, 0x56, 0x49, 0x44 } };
	static GUID IID_X360WirelessGamepad = { MAKELONG(0x045E, 0x028E), 0x0000, 0x0000, { 0x00, 0x00, 0x50, 0x49, 0x44, 0x56, 0x49, 0x44 } };

	if (p_guid == &IID_ValveStreamingGamepad || p_guid == &IID_X360WiredGamepad || p_guid == &IID_X360WirelessGamepad)
		return true;

	PRAWINPUTDEVICELIST dev_list = NULL;
	unsigned int dev_list_count = 0;

	if (GetRawInputDeviceList(NULL, &dev_list_count, sizeof(RAWINPUTDEVICELIST)) == -1) {
		return false;
	}
	dev_list = (PRAWINPUTDEVICELIST)malloc(sizeof(RAWINPUTDEVICELIST) * dev_list_count);
	if (!dev_list) return false;

	if (GetRawInputDeviceList(dev_list, &dev_list_count, sizeof(RAWINPUTDEVICELIST)) == -1) {
		free(dev_list);
		return false;
	}
	for (int i = 0; i < dev_list_count; i++) {

		RID_DEVICE_INFO rdi;
		char dev_name[128];
		UINT rdiSize = sizeof(rdi);
		UINT nameSize = sizeof(dev_name);

		rdi.cbSize = rdiSize;
		if ((dev_list[i].dwType == RIM_TYPEHID) &&
				(GetRawInputDeviceInfoA(dev_list[i].hDevice, RIDI_DEVICEINFO, &rdi, &rdiSize) != (UINT)-1) &&
				(MAKELONG(rdi.hid.dwVendorId, rdi.hid.dwProductId) == (LONG)p_guid->Data1) &&
				(GetRawInputDeviceInfoA(dev_list[i].hDevice, RIDI_DEVICENAME, &dev_name, &nameSize) != (UINT)-1) &&
				(strstr(dev_name, "IG_") != NULL)) {

			free(dev_list);
			return true;
		}
	}
	free(dev_list);
	return false;
}

bool JoypadWindows::setup_dinput_joypad(const DIDEVICEINSTANCE *instance) {

	HRESULT hr;
	int num = input->get_unused_joy_id();

	if (have_device(instance->guidInstance) || num == -1)
		return false;

	d_joypads[joypad_count] = dinput_gamepad();
	dinput_gamepad *joy = &d_joypads[joypad_count];

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
			__builtin_bswap32(guid.Data1), guid.Data2, guid.Data3,
			guid.Data4[0], guid.Data4[1], guid.Data4[2], guid.Data4[3],
			guid.Data4[4], guid.Data4[5], guid.Data4[6], guid.Data4[7]);

	id_to_change = joypad_count;

	joy->di_joy->SetDataFormat(&c_dfDIJoystick2);
	joy->di_joy->SetCooperativeLevel(*hWnd, DISCL_FOREGROUND);
	joy->di_joy->EnumObjects(objectsCallback, this, NULL);
	joy->joy_axis.sort();

	joy->guid = instance->guidInstance;
	input->joy_connection_changed(num, true, instance->tszProductName, uid);
	joy->attached = true;
	joy->id = num;
	attached_joypads[num] = true;
	joy->confirmed = true;
	joypad_count++;
	return true;
}

void JoypadWindows::setup_joypad_object(const DIDEVICEOBJECTINSTANCE *ob, int p_joy_id) {

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

		dinput_gamepad &joy = d_joypads[p_joy_id];

		res = IDirectInputDevice8_SetProperty(joy.di_joy, DIPROP_RANGE, &prop_range.diph);
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

BOOL CALLBACK JoypadWindows::enumCallback(const DIDEVICEINSTANCE *p_instance, void *p_context) {

	JoypadWindows *self = (JoypadWindows *)p_context;
	if (self->is_xinput_device(&p_instance->guidProduct)) {
		return DIENUM_CONTINUE;
	}
	self->setup_dinput_joypad(p_instance);
	return DIENUM_CONTINUE;
}

BOOL CALLBACK JoypadWindows::objectsCallback(const DIDEVICEOBJECTINSTANCE *instance, void *context) {

	JoypadWindows *self = (JoypadWindows *)context;
	self->setup_joypad_object(instance, self->id_to_change);

	return DIENUM_CONTINUE;
}

void JoypadWindows::close_joypad(int id) {

	if (id == -1) {

		for (int i = 0; i < JOYPADS_MAX; i++) {

			close_joypad(i);
		}
		return;
	}

	if (!d_joypads[id].attached) return;

	d_joypads[id].di_joy->Unacquire();
	d_joypads[id].di_joy->Release();
	d_joypads[id].attached = false;
	attached_joypads[d_joypads[id].id] = false;
	d_joypads[id].guid.Data1 = d_joypads[id].guid.Data2 = d_joypads[id].guid.Data3 = 0;
	input->joy_connection_changed(d_joypads[id].id, false, "");
	joypad_count--;
}

void JoypadWindows::probe_joypads() {

	DWORD dwResult;
	for (DWORD i = 0; i < XUSER_MAX_COUNT; i++) {

		ZeroMemory(&x_joypads[i].state, sizeof(XINPUT_STATE));

		dwResult = xinput_get_state(i, &x_joypads[i].state);
		if (dwResult == ERROR_SUCCESS) {

			int id = input->get_unused_joy_id();
			if (id != -1 && !x_joypads[i].attached) {

				x_joypads[i].attached = true;
				x_joypads[i].id = id;
				x_joypads[i].ff_timestamp = 0;
				x_joypads[i].ff_end_timestamp = 0;
				x_joypads[i].vibrating = false;
				attached_joypads[id] = true;
				input->joy_connection_changed(id, true, "XInput Gamepad", "__XINPUT_DEVICE__");
			}
		} else if (x_joypads[i].attached) {

			x_joypads[i].attached = false;
			attached_joypads[x_joypads[i].id] = false;
			input->joy_connection_changed(x_joypads[i].id, false, "");
		}
	}

	for (int i = 0; i < joypad_count; i++) {

		d_joypads[i].confirmed = false;
	}

	dinput->EnumDevices(DI8DEVCLASS_GAMECTRL, enumCallback, this, DIEDFL_ATTACHEDONLY);

	for (int i = 0; i < joypad_count; i++) {

		if (!d_joypads[i].confirmed) {

			close_joypad(i);
		}
	}
}

void JoypadWindows::process_joypads() {

	HRESULT hr;

	for (int i = 0; i < XUSER_MAX_COUNT; i++) {

		xinput_gamepad &joy = x_joypads[i];
		if (!joy.attached) {
			continue;
		}
		ZeroMemory(&joy.state, sizeof(XINPUT_STATE));

		xinput_get_state(i, &joy.state);
		if (joy.state.dwPacketNumber != joy.last_packet) {

			int button_mask = XINPUT_GAMEPAD_DPAD_UP;
			for (int i = 0; i <= 16; i++) {

				input->joy_button(joy.id, i, joy.state.Gamepad.wButtons & button_mask);
				button_mask = button_mask * 2;
			}

			input->joy_axis(joy.id, JOY_AXIS_0, axis_correct(joy.state.Gamepad.sThumbLX, true));
			input->joy_axis(joy.id, JOY_AXIS_1, axis_correct(joy.state.Gamepad.sThumbLY, true, false, true));
			input->joy_axis(joy.id, JOY_AXIS_2, axis_correct(joy.state.Gamepad.sThumbRX, true));
			input->joy_axis(joy.id, JOY_AXIS_3, axis_correct(joy.state.Gamepad.sThumbRY, true, false, true));
			input->joy_axis(joy.id, JOY_AXIS_4, axis_correct(joy.state.Gamepad.bLeftTrigger, true, true));
			input->joy_axis(joy.id, JOY_AXIS_5, axis_correct(joy.state.Gamepad.bRightTrigger, true, true));
			joy.last_packet = joy.state.dwPacketNumber;
		}
		uint64_t timestamp = input->get_joy_vibration_timestamp(joy.id);
		if (timestamp > joy.ff_timestamp) {
			Vector2 strength = input->get_joy_vibration_strength(joy.id);
			float duration = input->get_joy_vibration_duration(joy.id);
			if (strength.x == 0 && strength.y == 0) {
				joypad_vibration_stop_xinput(i, timestamp);
			} else {
				joypad_vibration_start_xinput(i, strength.x, strength.y, duration, timestamp);
			}
		} else if (joy.vibrating && joy.ff_end_timestamp != 0) {
			uint64_t current_time = OS::get_singleton()->get_ticks_usec();
			if (current_time >= joy.ff_end_timestamp)
				joypad_vibration_stop_xinput(i, current_time);
		}
	}

	for (int i = 0; i < JOYPADS_MAX; i++) {

		dinput_gamepad *joy = &d_joypads[i];

		if (!joy->attached)
			continue;

		DIJOYSTATE2 js;
		hr = joy->di_joy->Poll();
		if (hr == DIERR_INPUTLOST || hr == DIERR_NOTACQUIRED) {
			IDirectInputDevice8_Acquire(joy->di_joy);
			joy->di_joy->Poll();
		}

		hr = joy->di_joy->GetDeviceState(sizeof(DIJOYSTATE2), &js);
		if (FAILED(hr)) {

			//printf("failed to read joy #%d\n", i);
			continue;
		}

		post_hat(joy->id, js.rgdwPOV[0]);

		for (int j = 0; j < 128; j++) {

			if (js.rgbButtons[j] & 0x80) {

				if (!joy->last_buttons[j]) {

					input->joy_button(joy->id, j, true);
					joy->last_buttons[j] = true;
				}
			} else {

				if (joy->last_buttons[j]) {

					input->joy_button(joy->id, j, false);
					joy->last_buttons[j] = false;
				}
			}
		}

		// on mingw, these constants are not constants
		int count = 6;
		int axes[] = { DIJOFS_X, DIJOFS_Y, DIJOFS_Z, DIJOFS_RX, DIJOFS_RY, DIJOFS_RZ };
		int values[] = { js.lX, js.lY, js.lZ, js.lRx, js.lRy, js.lRz };

		for (int j = 0; j < joy->joy_axis.size(); j++) {

			for (int k = 0; k < count; k++) {
				if (joy->joy_axis[j] == axes[k]) {
					input->joy_axis(joy->id, j, axis_correct(values[k]));
					break;
				};
			};
		};
	}
	return;
}

void JoypadWindows::post_hat(int p_device, DWORD p_dpad) {

	int dpad_val = 0;

	if (p_dpad == -1) {
		dpad_val = InputDefault::HAT_MASK_CENTER;
	}
	if (p_dpad == 0) {

		dpad_val = InputDefault::HAT_MASK_UP;

	} else if (p_dpad == 4500) {

		dpad_val = (InputDefault::HAT_MASK_UP | InputDefault::HAT_MASK_RIGHT);

	} else if (p_dpad == 9000) {

		dpad_val = InputDefault::HAT_MASK_RIGHT;

	} else if (p_dpad == 13500) {

		dpad_val = (InputDefault::HAT_MASK_RIGHT | InputDefault::HAT_MASK_DOWN);

	} else if (p_dpad == 18000) {

		dpad_val = InputDefault::HAT_MASK_DOWN;

	} else if (p_dpad == 22500) {

		dpad_val = (InputDefault::HAT_MASK_DOWN | InputDefault::HAT_MASK_LEFT);

	} else if (p_dpad == 27000) {

		dpad_val = InputDefault::HAT_MASK_LEFT;

	} else if (p_dpad == 31500) {

		dpad_val = (InputDefault::HAT_MASK_LEFT | InputDefault::HAT_MASK_UP);
	}
	input->joy_hat(p_device, dpad_val);
};

InputDefault::JoyAxis JoypadWindows::axis_correct(int p_val, bool p_xinput, bool p_trigger, bool p_negate) const {

	InputDefault::JoyAxis jx;
	if (Math::abs(p_val) < MIN_JOY_AXIS) {
		jx.min = p_trigger ? 0 : -1;
		jx.value = 0.0f;
		return jx;
	}
	if (p_xinput) {

		if (p_trigger) {
			jx.min = 0;
			jx.value = (float)p_val / MAX_TRIGGER;
			return jx;
		}
		jx.min = -1;
		if (p_val < 0) {
			jx.value = (float)p_val / MAX_JOY_AXIS;
		} else {
			jx.value = (float)p_val / (MAX_JOY_AXIS - 1);
		}
		if (p_negate) {
			jx.value = -jx.value;
		}
		return jx;
	}
	jx.min = -1;
	jx.value = (float)p_val / MAX_JOY_AXIS;
	return jx;
}

void JoypadWindows::joypad_vibration_start_xinput(int p_device, float p_weak_magnitude, float p_strong_magnitude, float p_duration, uint64_t p_timestamp) {
	xinput_gamepad &joy = x_joypads[p_device];
	if (joy.attached) {
		XINPUT_VIBRATION effect;
		effect.wLeftMotorSpeed = (65535 * p_strong_magnitude);
		effect.wRightMotorSpeed = (65535 * p_weak_magnitude);
		if (xinput_set_state(p_device, &effect) == ERROR_SUCCESS) {
			joy.ff_timestamp = p_timestamp;
			joy.ff_end_timestamp = p_duration == 0 ? 0 : p_timestamp + (uint64_t)(p_duration * 1000000.0);
			joy.vibrating = true;
		}
	}
}

void JoypadWindows::joypad_vibration_stop_xinput(int p_device, uint64_t p_timestamp) {
	xinput_gamepad &joy = x_joypads[p_device];
	if (joy.attached) {
		XINPUT_VIBRATION effect;
		effect.wLeftMotorSpeed = 0;
		effect.wRightMotorSpeed = 0;
		if (xinput_set_state(p_device, &effect) == ERROR_SUCCESS) {
			joy.ff_timestamp = p_timestamp;
			joy.vibrating = false;
		}
	}
}

void JoypadWindows::load_xinput() {

	xinput_get_state = &_xinput_get_state;
	xinput_set_state = &_xinput_set_state;
	xinput_dll = LoadLibrary("XInput1_4.dll");
	if (!xinput_dll) {
		xinput_dll = LoadLibrary("XInput1_3.dll");
		if (!xinput_dll) {
			xinput_dll = LoadLibrary("XInput9_1_0.dll");
		}
	}

	if (!xinput_dll) {
		if (OS::get_singleton()->is_stdout_verbose()) {
			print_line("Could not find XInput, using DirectInput only");
		}
		return;
	}

	XInputGetState_t func = (XInputGetState_t)GetProcAddress((HMODULE)xinput_dll, "XInputGetState");
	XInputSetState_t set_func = (XInputSetState_t)GetProcAddress((HMODULE)xinput_dll, "XInputSetState");
	if (!func || !set_func) {
		unload_xinput();
		return;
	}
	xinput_get_state = func;
	xinput_set_state = set_func;
}

void JoypadWindows::unload_xinput() {

	if (xinput_dll) {

		FreeLibrary((HMODULE)xinput_dll);
	}
}
