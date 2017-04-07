/*************************************************************************/
/*  event_queue.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "event_queue.h"

Error EventQueue::push_call(uint32_t p_instance_ID, const StringName &p_method, VARIANT_ARG_DECLARE) {

	uint8_t room_needed = sizeof(Event);
	int args = 0;
	if (p_arg5.get_type() != Variant::NIL)
		args = 5;
	else if (p_arg4.get_type() != Variant::NIL)
		args = 4;
	else if (p_arg3.get_type() != Variant::NIL)
		args = 3;
	else if (p_arg2.get_type() != Variant::NIL)
		args = 2;
	else if (p_arg1.get_type() != Variant::NIL)
		args = 1;
	else
		args = 0;

	room_needed += sizeof(Variant) * args;

	ERR_FAIL_COND_V((buffer_end + room_needed) >= buffer_size, ERR_OUT_OF_MEMORY);
	Event *ev = memnew_placement(&event_buffer[buffer_end], Event);
	ev->args = args;
	ev->instance_ID = p_instance_ID;
	ev->method = p_method;

	buffer_end += sizeof(Event);

	if (args >= 1) {

		Variant *v = memnew_placement(&event_buffer[buffer_end], Variant);
		buffer_end += sizeof(Variant);
		*v = p_arg1;
	}

	if (args >= 2) {

		Variant *v = memnew_placement(&event_buffer[buffer_end], Variant);
		buffer_end += sizeof(Variant);
		*v = p_arg2;
	}

	if (args >= 3) {

		Variant *v = memnew_placement(&event_buffer[buffer_end], Variant);
		buffer_end += sizeof(Variant);
		*v = p_arg3;
	}

	if (args >= 4) {

		Variant *v = memnew_placement(&event_buffer[buffer_end], Variant);
		buffer_end += sizeof(Variant);
		*v = p_arg4;
	}

	if (args >= 5) {

		Variant *v = memnew_placement(&event_buffer[buffer_end], Variant);
		buffer_end += sizeof(Variant);
		*v = p_arg5;
	}

	if (buffer_end > buffer_max_used)
		buffer_max_used = buffer_end;

	return OK;
}

void EventQueue::flush_events() {

	uint32_t read_pos = 0;

	while (read_pos < buffer_end) {

		Event *event = (Event *)&event_buffer[read_pos];
		Variant *args = (Variant *)(event + 1);
		Object *obj = ObjectDB::get_instance(event->instance_ID);

		if (obj) {
			// events don't expect a return value
			obj->call(event->method,
					(event->args >= 1) ? args[0] : Variant(),
					(event->args >= 2) ? args[1] : Variant(),
					(event->args >= 3) ? args[2] : Variant(),
					(event->args >= 4) ? args[3] : Variant(),
					(event->args >= 5) ? args[4] : Variant());
		}

		if (event->args >= 1) args[0].~Variant();
		if (event->args >= 2) args[1].~Variant();
		if (event->args >= 3) args[2].~Variant();
		if (event->args >= 4) args[3].~Variant();
		if (event->args >= 5) args[4].~Variant();
		event->~Event();

		read_pos += sizeof(Event) + sizeof(Variant) * event->args;
	}

	buffer_end = 0; // reset buffer
}

EventQueue::EventQueue(uint32_t p_buffer_size) {

	buffer_end = 0;
	buffer_max_used = 0;
	buffer_size = p_buffer_size;
	event_buffer = memnew_arr(uint8_t, buffer_size);
}
EventQueue::~EventQueue() {

	uint32_t read_pos = 0;

	while (read_pos < buffer_end) {

		Event *event = (Event *)&event_buffer[read_pos];
		Variant *args = (Variant *)(event + 1);
		for (int i = 0; i < event->args; i++)
			args[i].~Variant();
		event->~Event();

		read_pos += sizeof(Event) + sizeof(Variant) * event->args;
	}

	memdelete_arr(event_buffer);
	event_buffer = NULL;
}
