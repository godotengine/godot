/*************************************************************************/
/*  joypad_osx.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "joypad_osx.h"

#include <machine/endian.h>

#define GODOT_JOY_LOOP_RUN_MODE CFSTR("GodotJoypad")

static JoypadOSX *self = nullptr;

joypad::joypad() {
	ff_constant_force.lMagnitude = 10000;
	ff_effect.dwDuration = 0;
	ff_effect.dwSamplePeriod = 0;
	ff_effect.dwGain = 10000;
	ff_effect.dwFlags = FFEFF_OBJECTOFFSETS;
	ff_effect.dwTriggerButton = FFEB_NOTRIGGER;
	ff_effect.dwStartDelay = 0;
	ff_effect.dwTriggerRepeatInterval = 0;
	ff_effect.lpEnvelope = nullptr;
	ff_effect.cbTypeSpecificParams = sizeof(FFCONSTANTFORCE);
	ff_effect.lpvTypeSpecificParams = &ff_constant_force;
	ff_effect.dwSize = sizeof(ff_effect);
}

void joypad::free() {
	if (device_ref) {
		IOHIDDeviceUnscheduleFromRunLoop(device_ref, CFRunLoopGetCurrent(), GODOT_JOY_LOOP_RUN_MODE);
	}
	if (ff_device) {
		FFDeviceReleaseEffect(ff_device, ff_object);
		FFReleaseDevice(ff_device);
		ff_device = nullptr;
		memfree(ff_axes);
		memfree(ff_directions);
	}
}

bool joypad::has_element(IOHIDElementCookie p_cookie, Vector<rec_element> *p_list) const {
	for (int i = 0; i < p_list->size(); i++) {
		if (p_cookie == p_list->get(i).cookie) {
			return true;
		}
	}
	return false;
}

int joypad::get_hid_element_state(rec_element *p_element) const {
	int value = 0;
	if (p_element && p_element->ref) {
		IOHIDValueRef valueRef;
		if (IOHIDDeviceGetValue(device_ref, p_element->ref, &valueRef) == kIOReturnSuccess) {
			value = (SInt32)IOHIDValueGetIntegerValue(valueRef);

			/* record min and max for auto calibration */
			if (value < p_element->min) {
				p_element->min = value;
			}
			if (value > p_element->max) {
				p_element->max = value;
			}
		}
	}
	return value;
}

void joypad::add_hid_element(IOHIDElementRef p_element) {
	const CFTypeID elementTypeID = p_element ? CFGetTypeID(p_element) : 0;

	if (p_element && (elementTypeID == IOHIDElementGetTypeID())) {
		const IOHIDElementCookie cookie = IOHIDElementGetCookie(p_element);
		const uint32_t usagePage = IOHIDElementGetUsagePage(p_element);
		const uint32_t usage = IOHIDElementGetUsage(p_element);
		Vector<rec_element> *list = nullptr;

		switch (IOHIDElementGetType(p_element)) {
			case kIOHIDElementTypeInput_Misc:
			case kIOHIDElementTypeInput_Button:
			case kIOHIDElementTypeInput_Axis: {
				switch (usagePage) {
					case kHIDPage_GenericDesktop:
						switch (usage) {
							case kHIDUsage_GD_X:
							case kHIDUsage_GD_Y:
							case kHIDUsage_GD_Z:
							case kHIDUsage_GD_Rx:
							case kHIDUsage_GD_Ry:
							case kHIDUsage_GD_Rz:
							case kHIDUsage_GD_Slider:
							case kHIDUsage_GD_Dial:
							case kHIDUsage_GD_Wheel:
								if (!has_element(cookie, &axis_elements)) {
									list = &axis_elements;
								}
								break;

							case kHIDUsage_GD_Hatswitch:
								if (!has_element(cookie, &hat_elements)) {
									list = &hat_elements;
								}
								break;
							case kHIDUsage_GD_DPadUp:
							case kHIDUsage_GD_DPadDown:
							case kHIDUsage_GD_DPadRight:
							case kHIDUsage_GD_DPadLeft:
							case kHIDUsage_GD_Start:
							case kHIDUsage_GD_Select:
								if (!has_element(cookie, &button_elements)) {
									list = &button_elements;
								}
								break;
						}
						break;

					case kHIDPage_Simulation:
						switch (usage) {
							case kHIDUsage_Sim_Rudder:
							case kHIDUsage_Sim_Throttle:
							case kHIDUsage_Sim_Accelerator:
							case kHIDUsage_Sim_Brake:
								if (!has_element(cookie, &axis_elements)) {
									list = &axis_elements;
								}
								break;

							default:
								break;
						}
						break;

					case kHIDPage_Button:
					case kHIDPage_Consumer:
						if (!has_element(cookie, &button_elements)) {
							list = &button_elements;
						}
						break;

					default:
						break;
				}
			} break;

			case kIOHIDElementTypeCollection: {
				CFArrayRef array = IOHIDElementGetChildren(p_element);
				if (array) {
					add_hid_elements(array);
				}
			} break;

			default:
				break;
		}

		if (list) { /* add to list */
			rec_element element;

			element.ref = p_element;
			element.usage = usage;

			element.min = (SInt32)IOHIDElementGetLogicalMin(p_element);
			element.max = (SInt32)IOHIDElementGetLogicalMax(p_element);
			element.cookie = IOHIDElementGetCookie(p_element);
			list->push_back(element);
			list->sort_custom<rec_element::Comparator>();
		}
	}
}

static void hid_element_added(const void *p_value, void *p_parameter) {
	joypad *joy = (joypad *)p_parameter;
	joy->add_hid_element((IOHIDElementRef)p_value);
}

void joypad::add_hid_elements(CFArrayRef p_array) {
	CFRange range = { 0, CFArrayGetCount(p_array) };
	CFArrayApplyFunction(p_array, range, hid_element_added, this);
}

static void joypad_removed_callback(void *ctx, IOReturn res, void *sender, IOHIDDeviceRef ioHIDDeviceObject) {
	self->_device_removed(res, ioHIDDeviceObject);
}

static void joypad_added_callback(void *ctx, IOReturn res, void *sender, IOHIDDeviceRef ioHIDDeviceObject) {
	self->_device_added(res, ioHIDDeviceObject);
}

static bool is_joypad(IOHIDDeviceRef p_device_ref) {
	int usage_page = 0;
	int usage = 0;
	CFTypeRef refCF = IOHIDDeviceGetProperty(p_device_ref, CFSTR(kIOHIDPrimaryUsagePageKey));
	if (refCF) {
		CFNumberGetValue((CFNumberRef)refCF, kCFNumberSInt32Type, &usage_page);
	}
	if (usage_page != kHIDPage_GenericDesktop) {
		return false;
	}

	refCF = IOHIDDeviceGetProperty(p_device_ref, CFSTR(kIOHIDPrimaryUsageKey));
	if (refCF) {
		CFNumberGetValue((CFNumberRef)refCF, kCFNumberSInt32Type, &usage);
	}
	if ((usage != kHIDUsage_GD_Joystick &&
				usage != kHIDUsage_GD_GamePad &&
				usage != kHIDUsage_GD_MultiAxisController)) {
		return false;
	}
	return true;
}

void JoypadOSX::_device_added(IOReturn p_res, IOHIDDeviceRef p_device) {
	if (p_res != kIOReturnSuccess || have_device(p_device)) {
		return;
	}

	joypad new_joypad;
	if (is_joypad(p_device)) {
		configure_joypad(p_device, &new_joypad);
#if MAC_OS_X_VERSION_MIN_REQUIRED < 1060
		if (IOHIDDeviceGetService) {
#endif
			const io_service_t ioservice = IOHIDDeviceGetService(p_device);
			if ((ioservice) && (FFIsForceFeedback(ioservice) == FF_OK) && new_joypad.config_force_feedback(ioservice)) {
				new_joypad.ffservice = ioservice;
			}
#if MAC_OS_X_VERSION_MIN_REQUIRED < 1060
		}
#endif
		device_list.push_back(new_joypad);
	}
	IOHIDDeviceScheduleWithRunLoop(p_device, CFRunLoopGetCurrent(), GODOT_JOY_LOOP_RUN_MODE);
}

void JoypadOSX::_device_removed(IOReturn p_res, IOHIDDeviceRef p_device) {
	int device = get_joy_ref(p_device);
	ERR_FAIL_COND(device == -1);

	input->joy_connection_changed(device_list[device].id, false, "");
	device_list.write[device].free();
	device_list.remove_at(device);
}

static String _hex_str(uint8_t p_byte) {
	static const char *dict = "0123456789abcdef";
	char ret[3];
	ret[2] = 0;

	ret[0] = dict[p_byte >> 4];
	ret[1] = dict[p_byte & 0xF];

	return ret;
}

bool JoypadOSX::configure_joypad(IOHIDDeviceRef p_device_ref, joypad *p_joy) {
	p_joy->device_ref = p_device_ref;
	/* get device name */
	String name;
	char c_name[256];
	CFTypeRef refCF = IOHIDDeviceGetProperty(p_device_ref, CFSTR(kIOHIDProductKey));
	if (!refCF) {
		refCF = IOHIDDeviceGetProperty(p_device_ref, CFSTR(kIOHIDManufacturerKey));
	}
	if ((!refCF) || (!CFStringGetCString((CFStringRef)refCF, c_name, sizeof(c_name), kCFStringEncodingUTF8))) {
		name = "Unidentified Joypad";
	} else {
		name = c_name;
	}

	int id = input->get_unused_joy_id();
	ERR_FAIL_COND_V(id == -1, false);
	p_joy->id = id;
	int vendor = 0;
	refCF = IOHIDDeviceGetProperty(p_device_ref, CFSTR(kIOHIDVendorIDKey));
	if (refCF) {
		CFNumberGetValue((CFNumberRef)refCF, kCFNumberSInt32Type, &vendor);
	}

	int product_id = 0;
	refCF = IOHIDDeviceGetProperty(p_device_ref, CFSTR(kIOHIDProductIDKey));
	if (refCF) {
		CFNumberGetValue((CFNumberRef)refCF, kCFNumberSInt32Type, &product_id);
	}

	int version = 0;
	refCF = IOHIDDeviceGetProperty(p_device_ref, CFSTR(kIOHIDVersionNumberKey));
	if (refCF) {
		CFNumberGetValue((CFNumberRef)refCF, kCFNumberSInt32Type, &version);
	}

	if (vendor && product_id) {
		char uid[128];
		sprintf(uid, "%08x%08x%08x%08x", OSSwapHostToBigInt32(3), OSSwapHostToBigInt32(vendor), OSSwapHostToBigInt32(product_id), OSSwapHostToBigInt32(version));
		input->joy_connection_changed(id, true, name, uid);
	} else {
		//bluetooth device
		String guid = "05000000";
		for (int i = 0; i < 12; i++) {
			if (i < name.size())
				guid += _hex_str(name[i]);
			else
				guid += "00";
		}
		input->joy_connection_changed(id, true, name, guid);
	}

	CFArrayRef array = IOHIDDeviceCopyMatchingElements(p_device_ref, nullptr, kIOHIDOptionsTypeNone);
	if (array) {
		p_joy->add_hid_elements(array);
		CFRelease(array);
	}
	// Xbox controller hat values start at 1 rather than 0.
	p_joy->offset_hat = vendor == 0x45e &&
			(product_id == 0x0b05 ||
					product_id == 0x02e0 ||
					product_id == 0x02fd ||
					product_id == 0x0b13);

	return true;
}

#define FF_ERR()                        \
	{                                   \
		if (ret != FF_OK) {             \
			FFReleaseDevice(ff_device); \
			ff_device = nullptr;        \
			return false;               \
		}                               \
	}
bool joypad::config_force_feedback(io_service_t p_service) {
	HRESULT ret = FFCreateDevice(p_service, &ff_device);
	ERR_FAIL_COND_V(ret != FF_OK, false);

	ret = FFDeviceSendForceFeedbackCommand(ff_device, FFSFFC_RESET);
	FF_ERR();

	ret = FFDeviceSendForceFeedbackCommand(ff_device, FFSFFC_SETACTUATORSON);
	FF_ERR();

	if (check_ff_features()) {
		ret = FFDeviceCreateEffect(ff_device, kFFEffectType_ConstantForce_ID, &ff_effect, &ff_object);
		FF_ERR();
		return true;
	}
	FFReleaseDevice(ff_device);
	ff_device = nullptr;
	return false;
}
#undef FF_ERR

#define TEST_FF(ff) (features.supportedEffects & (ff))
bool joypad::check_ff_features() {
	FFCAPABILITIES features;
	HRESULT ret = FFDeviceGetForceFeedbackCapabilities(ff_device, &features);
	if (ret == FF_OK && (features.supportedEffects & FFCAP_ET_CONSTANTFORCE)) {
		uint32_t val;
		ret = FFDeviceGetForceFeedbackProperty(ff_device, FFPROP_FFGAIN, &val, sizeof(val));
		if (ret != FF_OK)
			return false;
		int num_axes = features.numFfAxes;
		ff_axes = (DWORD *)memalloc(sizeof(DWORD) * num_axes);
		ff_directions = (LONG *)memalloc(sizeof(LONG) * num_axes);

		for (int i = 0; i < num_axes; i++) {
			ff_axes[i] = features.ffAxes[i];
			ff_directions[i] = 0;
		}

		ff_effect.cAxes = num_axes;
		ff_effect.rgdwAxes = ff_axes;
		ff_effect.rglDirection = ff_directions;
		return true;
	}
	return false;
}

static HatMask process_hat_value(int p_min, int p_max, int p_value, bool p_offset_hat) {
	int range = (p_max - p_min + 1);
	int value = p_value - p_min;
	HatMask hat_value = HatMask::CENTER;
	if (range == 4) {
		value *= 2;
	}
	if (p_offset_hat) {
		value -= 1;
	}

	switch (value) {
		case 0:
			hat_value = HatMask::UP;
			break;
		case 1:
			hat_value = (HatMask::UP | HatMask::RIGHT);
			break;
		case 2:
			hat_value = HatMask::RIGHT;
			break;
		case 3:
			hat_value = (HatMask::DOWN | HatMask::RIGHT);
			break;
		case 4:
			hat_value = HatMask::DOWN;
			break;
		case 5:
			hat_value = (HatMask::DOWN | HatMask::LEFT);
			break;
		case 6:
			hat_value = HatMask::LEFT;
			break;
		case 7:
			hat_value = (HatMask::UP | HatMask::LEFT);
			break;
		default:
			hat_value = HatMask::CENTER;
			break;
	}
	return hat_value;
}

void JoypadOSX::poll_joypads() const {
	while (CFRunLoopRunInMode(GODOT_JOY_LOOP_RUN_MODE, 0, TRUE) == kCFRunLoopRunHandledSource) {
		/* no-op. Pending callbacks will fire. */
	}
}

static const Input::JoyAxisValue axis_correct(int p_value, int p_min, int p_max) {
	Input::JoyAxisValue jx;
	if (p_min < 0) {
		jx.min = -1;
		if (p_value < 0) {
			jx.value = (float)-p_value / p_min;
		} else
			jx.value = (float)p_value / p_max;
	}
	if (p_min == 0) {
		jx.min = 0;
		jx.value = 0.0f + (float)p_value / p_max;
	}
	return jx;
}

void JoypadOSX::process_joypads() {
	poll_joypads();

	for (int i = 0; i < device_list.size(); i++) {
		joypad &joy = device_list.write[i];

		for (int j = 0; j < joy.axis_elements.size(); j++) {
			rec_element &elem = joy.axis_elements.write[j];
			int value = joy.get_hid_element_state(&elem);
			input->joy_axis(joy.id, (JoyAxis)j, axis_correct(value, elem.min, elem.max));
		}
		for (int j = 0; j < joy.button_elements.size(); j++) {
			int value = joy.get_hid_element_state(&joy.button_elements.write[j]);
			input->joy_button(joy.id, (JoyButton)j, (value >= 1));
		}
		for (int j = 0; j < joy.hat_elements.size(); j++) {
			rec_element &elem = joy.hat_elements.write[j];
			int value = joy.get_hid_element_state(&elem);
			HatMask hat_value = process_hat_value(elem.min, elem.max, value, joy.offset_hat);
			input->joy_hat(joy.id, hat_value);
		}

		if (joy.ffservice) {
			uint64_t timestamp = input->get_joy_vibration_timestamp(joy.id);
			if (timestamp > joy.ff_timestamp) {
				Vector2 strength = input->get_joy_vibration_strength(joy.id);
				float duration = input->get_joy_vibration_duration(joy.id);
				if (strength.x == 0 && strength.y == 0) {
					joypad_vibration_stop(joy.id, timestamp);
				} else {
					float gain = MAX(strength.x, strength.y);
					joypad_vibration_start(joy.id, gain, duration, timestamp);
				}
			}
		}
	}
}

void JoypadOSX::joypad_vibration_start(int p_id, float p_magnitude, float p_duration, uint64_t p_timestamp) {
	joypad *joy = &device_list.write[get_joy_index(p_id)];
	joy->ff_timestamp = p_timestamp;
	joy->ff_effect.dwDuration = p_duration * FF_SECONDS;
	joy->ff_effect.dwGain = p_magnitude * FF_FFNOMINALMAX;
	FFEffectSetParameters(joy->ff_object, &joy->ff_effect, FFEP_DURATION | FFEP_GAIN);
	FFEffectStart(joy->ff_object, 1, 0);
}

void JoypadOSX::joypad_vibration_stop(int p_id, uint64_t p_timestamp) {
	joypad *joy = &device_list.write[get_joy_index(p_id)];
	joy->ff_timestamp = p_timestamp;
	FFEffectStop(joy->ff_object);
}

int JoypadOSX::get_joy_index(int p_id) const {
	for (int i = 0; i < device_list.size(); i++) {
		if (device_list[i].id == p_id)
			return i;
	}
	return -1;
}

int JoypadOSX::get_joy_ref(IOHIDDeviceRef p_device) const {
	for (int i = 0; i < device_list.size(); i++) {
		if (device_list[i].device_ref == p_device)
			return i;
	}
	return -1;
}

bool JoypadOSX::have_device(IOHIDDeviceRef p_device) const {
	for (int i = 0; i < device_list.size(); i++) {
		if (device_list[i].device_ref == p_device) {
			return true;
		}
	}
	return false;
}

static CFDictionaryRef create_match_dictionary(const UInt32 page, const UInt32 usage, int *okay) {
	CFDictionaryRef retval = nullptr;
	CFNumberRef pageNumRef = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &page);
	CFNumberRef usageNumRef = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &usage);
	const void *keys[2] = { (void *)CFSTR(kIOHIDDeviceUsagePageKey), (void *)CFSTR(kIOHIDDeviceUsageKey) };
	const void *vals[2] = { (void *)pageNumRef, (void *)usageNumRef };

	if (pageNumRef && usageNumRef) {
		retval = CFDictionaryCreate(kCFAllocatorDefault, keys, vals, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	}

	if (pageNumRef) {
		CFRelease(pageNumRef);
	}
	if (usageNumRef) {
		CFRelease(usageNumRef);
	}

	if (!retval) {
		*okay = 0;
	}

	return retval;
}

void JoypadOSX::config_hid_manager(CFArrayRef p_matching_array) const {
	CFRunLoopRef runloop = CFRunLoopGetCurrent();
	IOReturn ret = IOHIDManagerOpen(hid_manager, kIOHIDOptionsTypeNone);
	ERR_FAIL_COND(ret != kIOReturnSuccess);

	IOHIDManagerSetDeviceMatchingMultiple(hid_manager, p_matching_array);
	IOHIDManagerRegisterDeviceMatchingCallback(hid_manager, joypad_added_callback, nullptr);
	IOHIDManagerRegisterDeviceRemovalCallback(hid_manager, joypad_removed_callback, nullptr);
	IOHIDManagerScheduleWithRunLoop(hid_manager, runloop, GODOT_JOY_LOOP_RUN_MODE);

	while (CFRunLoopRunInMode(GODOT_JOY_LOOP_RUN_MODE, 0, TRUE) == kCFRunLoopRunHandledSource) {
		/* no-op. Callback fires once per existing device. */
	}
}

JoypadOSX::JoypadOSX(Input *in) {
	self = this;
	input = in;

	int okay = 1;
	const void *vals[] = {
		(void *)create_match_dictionary(kHIDPage_GenericDesktop, kHIDUsage_GD_Joystick, &okay),
		(void *)create_match_dictionary(kHIDPage_GenericDesktop, kHIDUsage_GD_GamePad, &okay),
		(void *)create_match_dictionary(kHIDPage_GenericDesktop, kHIDUsage_GD_MultiAxisController, &okay),
	};
	const size_t n_elements = sizeof(vals) / sizeof(vals[0]);
	CFArrayRef array = okay ? CFArrayCreate(kCFAllocatorDefault, vals, n_elements, &kCFTypeArrayCallBacks) : nullptr;

	for (size_t i = 0; i < n_elements; i++) {
		if (vals[i]) {
			CFRelease((CFTypeRef)vals[i]);
		}
	}

	if (array) {
		hid_manager = IOHIDManagerCreate(kCFAllocatorDefault, kIOHIDOptionsTypeNone);
		if (hid_manager) {
			config_hid_manager(array);
		}
		CFRelease(array);
	}
}

JoypadOSX::~JoypadOSX() {
	for (int i = 0; i < device_list.size(); i++) {
		device_list.write[i].free();
	}

	IOHIDManagerUnscheduleFromRunLoop(hid_manager, CFRunLoopGetCurrent(), GODOT_JOY_LOOP_RUN_MODE);
	IOHIDManagerClose(hid_manager, kIOHIDOptionsTypeNone);
	CFRelease(hid_manager);
	hid_manager = nullptr;
}
