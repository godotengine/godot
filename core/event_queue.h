/*************************************************************************/
/*  event_queue.h                                                        */
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
#ifndef EVENT_QUEUE_H
#define EVENT_QUEUE_H

#include "object.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class EventQueue {

	enum {

		DEFAULT_EVENT_QUEUE_SIZE_KB = 256
	};

	struct Event {

		uint32_t instance_ID;
		StringName method;
		int args;
	};

	uint8_t *event_buffer;
	uint32_t buffer_end;
	uint32_t buffer_max_used;
	uint32_t buffer_size;

public:
	Error push_call(uint32_t p_instance_ID, const StringName &p_method, VARIANT_ARG_LIST);
	void flush_events();

	EventQueue(uint32_t p_buffer_size = DEFAULT_EVENT_QUEUE_SIZE_KB * 1024);
	~EventQueue();
};

#endif
