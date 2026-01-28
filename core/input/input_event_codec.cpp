/**************************************************************************/
/*  input_event_codec.cpp                                                 */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "input_event_codec.h"

#include "core/input/input.h"
#include "core/io/marshalls.h"
#include "core/os/os.h"

enum class BoolShift : uint8_t {
	SHIFT = 0,
	CTRL,
	ALT,
	META,
	ECHO,
	PRESSED,
	DOUBLE_CLICK,
	PEN_INVERTED,
};

// cast operator for BoolShift to uint8_t
inline uint8_t operator<<(uint8_t a, BoolShift b) {
	return a << static_cast<uint8_t>(b);
}

uint8_t encode_key_modifier_state(Ref<InputEventWithModifiers> p_event) {
	uint8_t bools = 0;
	bools |= (uint8_t)p_event->is_shift_pressed() << BoolShift::SHIFT;
	bools |= (uint8_t)p_event->is_ctrl_pressed() << BoolShift::CTRL;
	bools |= (uint8_t)p_event->is_alt_pressed() << BoolShift::ALT;
	bools |= (uint8_t)p_event->is_meta_pressed() << BoolShift::META;
	return bools;
}

void decode_key_modifier_state(uint8_t bools, Ref<InputEventWithModifiers> p_event) {
	p_event->set_shift_pressed(bools & (1 << BoolShift::SHIFT));
	p_event->set_ctrl_pressed(bools & (1 << BoolShift::CTRL));
	p_event->set_alt_pressed(bools & (1 << BoolShift::ALT));
	p_event->set_meta_pressed(bools & (1 << BoolShift::META));
}

int encode_vector2(const Vector2 &p_vector, uint8_t *p_data) {
	p_data += encode_float(p_vector.x, p_data);
	encode_float(p_vector.y, p_data);
	return sizeof(float) * 2;
}

const uint8_t *decode_vector2(Vector2 &r_vector, const uint8_t *p_data) {
	r_vector.x = decode_float(p_data);
	p_data += sizeof(float);
	r_vector.y = decode_float(p_data);
	p_data += sizeof(float);
	return p_data;
}

void encode_input_event_key(const Ref<InputEventKey> &p_event, PackedByteArray &r_data) {
	r_data.resize(19);

	uint8_t *data = r_data.ptrw();
	*data = (uint8_t)InputEventType::KEY;
	data++;
	uint8_t bools = encode_key_modifier_state(p_event);
	bools |= (uint8_t)p_event->is_echo() << BoolShift::ECHO;
	bools |= (uint8_t)p_event->is_pressed() << BoolShift::PRESSED;
	*data = bools;
	data++;
	data += encode_uint32((uint32_t)p_event->get_keycode(), data);
	data += encode_uint32((uint32_t)p_event->get_physical_keycode(), data);
	data += encode_uint32((uint32_t)p_event->get_key_label(), data);
	data += encode_uint32(p_event->get_unicode(), data);
	*data = (uint8_t)p_event->get_location();
	data++;

	// Assert we had enough space.
	DEV_ASSERT(r_data.size() >= (data - r_data.ptrw()));
}

Error decode_input_event_key(const PackedByteArray &p_data, Ref<InputEventKey> &r_event) {
	const uint8_t *data = p_data.ptr();
	DEV_ASSERT(static_cast<InputEventType>(*data) == InputEventType::KEY);
	data++; // Skip event type.

	uint8_t bools = *data;
	data++;
	decode_key_modifier_state(bools, r_event);
	r_event->set_echo(bools & (1 << BoolShift::ECHO));
	r_event->set_pressed(bools & (1 << BoolShift::PRESSED));

	Key keycode = (Key)decode_uint32(data);
	data += sizeof(uint32_t);
	Key physical_keycode = (Key)decode_uint32(data);
	data += sizeof(uint32_t);
	Key key_label = (Key)decode_uint32(data);
	data += sizeof(uint32_t);
	char32_t unicode = (char32_t)decode_uint32(data);
	data += sizeof(uint32_t);
	KeyLocation location = (KeyLocation)*data;

	r_event->set_keycode(keycode);
	r_event->set_physical_keycode(physical_keycode);
	r_event->set_key_label(key_label);
	r_event->set_unicode(unicode);
	r_event->set_location(location);

	return OK;
}

void encode_input_event_mouse_button(const Ref<InputEventMouseButton> &p_event, PackedByteArray &r_data) {
	r_data.resize(12);

	uint8_t *data = r_data.ptrw();
	*data = (uint8_t)InputEventType::MOUSE_BUTTON;
	data++;

	uint8_t bools = encode_key_modifier_state(p_event);
	bools |= (uint8_t)p_event->is_pressed() << BoolShift::PRESSED;
	bools |= (uint8_t)p_event->is_double_click() << BoolShift::DOUBLE_CLICK;
	*data = bools;
	data++;

	*data = (uint8_t)p_event->get_button_index();
	data++;

	// Rather than use encode_variant, we explicitly encode the Vector2,
	// so decoding is easier. Specifically, we don't have to perform additional error
	// checking for decoding the variant and then checking the variant type.
	data += encode_vector2(p_event->get_position(), data);
	*data = (uint8_t)p_event->get_button_mask();
	data++;

	// Assert we had enough space.
	DEV_ASSERT(r_data.size() >= (data - r_data.ptrw()));
}

Error decode_input_event_mouse_button(const PackedByteArray &p_data, Ref<InputEventMouseButton> &r_event) {
	const uint8_t *data = p_data.ptr();
	DEV_ASSERT(static_cast<InputEventType>(*data) == InputEventType::MOUSE_BUTTON);
	data++; // Skip event type.

	uint8_t bools = *data;
	data++;
	decode_key_modifier_state(bools, r_event);
	r_event->set_pressed(bools & (1 << BoolShift::PRESSED));
	r_event->set_double_click(bools & (1 << BoolShift::DOUBLE_CLICK));

	r_event->set_button_index((MouseButton)*data);
	data++;

	Vector2 pos;
	data = decode_vector2(pos, data);
	r_event->set_position(pos);
	r_event->set_global_position(pos); // these are the same
	BitField<MouseButtonMask> button_mask = (MouseButtonMask)*data;
	r_event->set_button_mask(button_mask);

	return OK;
}

void encode_input_event_mouse_motion(const Ref<InputEventMouseMotion> &p_event, PackedByteArray &r_data) {
	r_data.resize(31);

	uint8_t *data = r_data.ptrw();
	*data = (uint8_t)InputEventType::MOUSE_MOTION;
	data++;

	uint8_t bools = encode_key_modifier_state(p_event);
	bools |= (uint8_t)p_event->get_pen_inverted() << BoolShift::PEN_INVERTED;
	*data = bools;
	data++;

	// Rather than use encode_variant, we explicitly encode the Vector2,
	// so decoding is easier. Specifically, we don't have to perform additional error
	// checking for decoding the variant and then checking the variant type.
	data += encode_vector2(p_event->get_position(), data);
	data += encode_float(p_event->get_pressure(), data);
	data += encode_vector2(p_event->get_tilt(), data);
	data += encode_vector2(p_event->get_relative(), data);
	*data = (uint8_t)p_event->get_button_mask();
	data++;

	// Assert we had enough space.
	DEV_ASSERT(r_data.size() >= (data - r_data.ptrw()));
}

void decode_input_event_mouse_motion(const PackedByteArray &p_data, Ref<InputEventMouseMotion> &r_event) {
	Input *input = Input::get_singleton();

	const uint8_t *data = p_data.ptr();
	DEV_ASSERT(static_cast<InputEventType>(*data) == InputEventType::MOUSE_MOTION);
	data++; // Skip event type.

	uint8_t bools = *data;
	data++;
	decode_key_modifier_state(bools, r_event);
	r_event->set_pen_inverted(bools & (1 << BoolShift::PEN_INVERTED));

	{
		Vector2 pos;
		data = decode_vector2(pos, data);
		r_event->set_position(pos);
		r_event->set_global_position(pos); // these are the same
	}
	r_event->set_pressure(decode_float(data));
	data += sizeof(float);
	{
		Vector2 tilt;
		data = decode_vector2(tilt, data);
		r_event->set_tilt(tilt);
	}
	r_event->set_velocity(input->get_last_mouse_velocity());
	r_event->set_screen_velocity(input->get_last_mouse_screen_velocity());
	{
		Vector2 relative;
		data = decode_vector2(relative, data);
		r_event->set_relative(relative);
		r_event->set_relative_screen_position(relative);
	}
	BitField<MouseButtonMask> button_mask = (MouseButtonMask)*data;
	r_event->set_button_mask(button_mask);
	data++;

	// Assert we had enough space.
	DEV_ASSERT(p_data.size() >= (data - p_data.ptr()));
}

void encode_input_event_joypad_button(const Ref<InputEventJoypadButton> &p_event, PackedByteArray &r_data) {
	r_data.resize(11);

	uint8_t *data = r_data.ptrw();
	*data = (uint8_t)InputEventType::JOY_BUTTON;
	data++;

	uint8_t bools = 0;
	bools |= (uint8_t)p_event->is_pressed() << BoolShift::PRESSED;
	*data = bools;
	data++;

	data += encode_uint64(p_event->get_device(), data);
	*data = (uint8_t)p_event->get_button_index();
	data++;

	// Assert we had enough space.
	DEV_ASSERT(r_data.size() >= (data - r_data.ptrw()));
}

void decode_input_event_joypad_button(const PackedByteArray &p_data, Ref<InputEventJoypadButton> &r_event) {
	const uint8_t *data = p_data.ptr();
	DEV_ASSERT(static_cast<InputEventType>(*data) == InputEventType::JOY_BUTTON);
	data++; // Skip event type.

	uint8_t bools = *data;
	data++;
	r_event->set_pressed(bools & (1 << BoolShift::PRESSED));
	r_event->set_device(decode_uint64(data));
	data += sizeof(uint64_t);
	r_event->set_button_index((JoyButton)*data);
	data++;

	// Assert we had enough space.
	DEV_ASSERT(p_data.size() >= (data - p_data.ptr()));
}

void encode_input_event_joypad_motion(const Ref<InputEventJoypadMotion> &p_event, PackedByteArray &r_data) {
	r_data.resize(14);

	uint8_t *data = r_data.ptrw();
	*data = (uint8_t)InputEventType::JOY_MOTION;
	data++;

	data += encode_uint64(p_event->get_device(), data);
	*data = (uint8_t)p_event->get_axis();
	data++;
	data += encode_float(p_event->get_axis_value(), data);

	// Assert we had enough space.
	DEV_ASSERT(r_data.size() >= (data - r_data.ptrw()));
}

void decode_input_event_joypad_motion(const PackedByteArray &p_data, Ref<InputEventJoypadMotion> &r_event) {
	const uint8_t *data = p_data.ptr();
	DEV_ASSERT(static_cast<InputEventType>(*data) == InputEventType::JOY_MOTION);
	data++; // Skip event type.

	r_event->set_device(decode_uint64(data));
	data += sizeof(uint64_t);
	r_event->set_axis((JoyAxis)*data);
	data++;
	r_event->set_axis_value(decode_float(data));
	data += sizeof(float);

	// Assert we had enough space.
	DEV_ASSERT(p_data.size() >= (data - p_data.ptr()));
}

void encode_input_event_gesture_pan(const Ref<InputEventPanGesture> &p_event, PackedByteArray &r_data) {
	r_data.resize(18);

	uint8_t *data = r_data.ptrw();
	*data = (uint8_t)InputEventType::PAN_GESTURE;
	data++;

	uint8_t bools = encode_key_modifier_state(p_event);
	*data = bools;
	data++;
	data += encode_vector2(p_event->get_position(), data);
	data += encode_vector2(p_event->get_delta(), data);

	// Assert we had enough space.
	DEV_ASSERT(r_data.size() >= (data - r_data.ptrw()));
}

void decode_input_event_gesture_pan(const PackedByteArray &p_data, Ref<InputEventPanGesture> &r_event) {
	const uint8_t *data = p_data.ptr();
	DEV_ASSERT(static_cast<InputEventType>(*data) == InputEventType::PAN_GESTURE);
	data++; // Skip event type.

	uint8_t bools = *data;
	data++;
	decode_key_modifier_state(bools, r_event);

	Vector2 pos;
	data = decode_vector2(pos, data);
	r_event->set_position(pos);
	Vector2 delta;
	data = decode_vector2(delta, data);
	r_event->set_delta(delta);

	// Assert we had enough space.
	DEV_ASSERT(p_data.size() >= (data - p_data.ptr()));
}

void encode_input_event_gesture_magnify(const Ref<InputEventMagnifyGesture> &p_event, PackedByteArray &r_data) {
	r_data.resize(14);

	uint8_t *data = r_data.ptrw();
	*data = (uint8_t)InputEventType::MAGNIFY_GESTURE;
	data++;

	uint8_t bools = encode_key_modifier_state(p_event);
	*data = bools;
	data++;
	data += encode_vector2(p_event->get_position(), data);
	data += encode_float(p_event->get_factor(), data);

	// Assert we had enough space.
	DEV_ASSERT(r_data.size() >= (data - r_data.ptrw()));
}

void decode_input_event_gesture_magnify(const PackedByteArray &p_data, Ref<InputEventMagnifyGesture> &r_event) {
	const uint8_t *data = p_data.ptr();
	DEV_ASSERT(static_cast<InputEventType>(*data) == InputEventType::MAGNIFY_GESTURE);
	data++; // Skip event type.

	uint8_t bools = *data;
	data++;
	decode_key_modifier_state(bools, r_event);

	Vector2 pos;
	data = decode_vector2(pos, data);
	r_event->set_position(pos);
	r_event->set_factor(decode_float(data));
	data += sizeof(float);

	// Assert we had enough space.
	DEV_ASSERT(p_data.size() >= (data - p_data.ptr()));
}

bool encode_input_event(const Ref<InputEvent> &p_event, PackedByteArray &r_data) {
	switch (p_event->get_type()) {
		case InputEventType::KEY:
			encode_input_event_key(p_event, r_data);
			break;
		case InputEventType::MOUSE_BUTTON:
			encode_input_event_mouse_button(p_event, r_data);
			break;
		case InputEventType::MOUSE_MOTION:
			encode_input_event_mouse_motion(p_event, r_data);
			break;
		case InputEventType::JOY_MOTION:
			encode_input_event_joypad_motion(p_event, r_data);
			break;
		case InputEventType::JOY_BUTTON:
			encode_input_event_joypad_button(p_event, r_data);
			break;
		case InputEventType::MAGNIFY_GESTURE:
			encode_input_event_gesture_magnify(p_event, r_data);
			break;
		case InputEventType::PAN_GESTURE:
			encode_input_event_gesture_pan(p_event, r_data);
			break;
		default:
			return false;
	}
	return true;
}

void decode_input_event(const PackedByteArray &p_data, Ref<InputEvent> &r_event) {
	const uint8_t *data = p_data.ptr();

	switch (static_cast<InputEventType>(*data)) {
		case InputEventType::KEY: {
			Ref<InputEventKey> event;
			event.instantiate();
			decode_input_event_key(p_data, event);
			r_event = event;
		} break;
		case InputEventType::MOUSE_BUTTON: {
			Ref<InputEventMouseButton> event;
			event.instantiate();
			decode_input_event_mouse_button(p_data, event);
			r_event = event;
		} break;
		case InputEventType::MOUSE_MOTION: {
			Ref<InputEventMouseMotion> event;
			event.instantiate();
			decode_input_event_mouse_motion(p_data, event);
			r_event = event;
		} break;
		case InputEventType::JOY_BUTTON: {
			Ref<InputEventJoypadButton> event;
			event.instantiate();
			decode_input_event_joypad_button(p_data, event);
			r_event = event;
		} break;
		case InputEventType::JOY_MOTION: {
			Ref<InputEventJoypadMotion> event;
			event.instantiate();
			decode_input_event_joypad_motion(p_data, event);
			r_event = event;
		} break;
		case InputEventType::PAN_GESTURE: {
			Ref<InputEventPanGesture> event;
			event.instantiate();
			decode_input_event_gesture_pan(p_data, event);
			r_event = event;
		} break;
		case InputEventType::MAGNIFY_GESTURE: {
			Ref<InputEventMagnifyGesture> event;
			event.instantiate();
			decode_input_event_gesture_magnify(p_data, event);
			r_event = event;
		} break;
		default: {
			WARN_PRINT(vformat("Unknown event type %d.", static_cast<int>(*data)));
		} break;
	}
}
