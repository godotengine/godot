/*************************************************************************/
/*  joypad_osx.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef JOYPADOSX_H
#define JOYPADOSX_H

#ifdef MACOS_10_0_4
#include <IOKit/hidsystem/IOHIDUsageTables.h>
#else
#include <Kernel/IOKit/hidsystem/IOHIDUsageTables.h>
#endif
#include <ForceFeedback/ForceFeedback.h>
#include <ForceFeedback/ForceFeedbackConstants.h>
#include <IOKit/hid/IOHIDLib.h>

#include "core/input/input.h"

struct rec_element {
	IOHIDElementRef ref;
	IOHIDElementCookie cookie;

	uint32_t usage;

	int min;
	int max;

	struct Comparator {
		bool operator()(const rec_element p_a, const rec_element p_b) const { return p_a.usage < p_b.usage; }
	};
};

struct joypad {
	IOHIDDeviceRef device_ref;

	Vector<rec_element> axis_elements;
	Vector<rec_element> button_elements;
	Vector<rec_element> hat_elements;

	int id;

	io_service_t ffservice; /* Interface for force feedback, 0 = no ff */
	FFCONSTANTFORCE ff_constant_force;
	FFDeviceObjectReference ff_device;
	FFEffectObjectReference ff_object;
	uint64_t ff_timestamp;
	LONG *ff_directions;
	FFEFFECT ff_effect;
	DWORD *ff_axes;

	void add_hid_elements(CFArrayRef p_array);
	void add_hid_element(IOHIDElementRef p_element);

	bool has_element(IOHIDElementCookie p_cookie, Vector<rec_element> *p_list) const;
	bool config_force_feedback(io_service_t p_service);
	bool check_ff_features();

	int get_hid_element_state(rec_element *p_element) const;

	void free();
	joypad();
};

class JoypadOSX {
	enum {
		JOYPADS_MAX = 16,
	};

private:
	Input *input;
	IOHIDManagerRef hid_manager;

	Vector<joypad> device_list;

	bool have_device(IOHIDDeviceRef p_device) const;
	bool configure_joypad(IOHIDDeviceRef p_device_ref, joypad *p_joy);

	int get_joy_index(int p_id) const;
	int get_joy_ref(IOHIDDeviceRef p_device) const;

	void poll_joypads() const;
	void setup_joypad_objects();
	void config_hid_manager(CFArrayRef p_matching_array) const;

	void joypad_vibration_start(int p_id, float p_magnitude, float p_duration, uint64_t p_timestamp);
	void joypad_vibration_stop(int p_id, uint64_t p_timestamp);

public:
	void process_joypads();

	void _device_added(IOReturn p_res, IOHIDDeviceRef p_device);
	void _device_removed(IOReturn p_res, IOHIDDeviceRef p_device);

	JoypadOSX(Input *in);
	~JoypadOSX();
};

#endif // JOYPADOSX_H
