/*************************************************************************/
/*  message_queue.h                                                      */
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

#ifndef MESSAGE_QUEUE_H
#define MESSAGE_QUEUE_H

#include "core/class_db.h"
#include "core/os/thread_safe.h"

class MessageQueue {
	_THREAD_SAFE_CLASS_

	enum {
		DEFAULT_QUEUE_SIZE_KB = 4096
	};

	enum {
		TYPE_CALL,
		TYPE_NOTIFICATION,
		TYPE_SET,
		FLAG_SHOW_ERROR = 1 << 14,
		FLAG_MASK = FLAG_SHOW_ERROR - 1

	};

	struct Message {
		Callable callable;
		int16_t type;
		union {
			int16_t notification;
			int16_t args;
		};
	};

	uint8_t *buffer;
	uint32_t buffer_end = 0;
	uint32_t buffer_max_used = 0;
	uint32_t buffer_size;

	void _call_function(const Callable &p_callable, const Variant *p_args, int p_argcount, bool p_show_error);

	static MessageQueue *singleton;

	bool flushing = false;

public:
	static MessageQueue *get_singleton();

	Error push_call(ObjectID p_id, const StringName &p_method, const Variant **p_args, int p_argcount, bool p_show_error = false);
	Error push_call(ObjectID p_id, const StringName &p_method, VARIANT_ARG_LIST);
	Error push_notification(ObjectID p_id, int p_notification);
	Error push_set(ObjectID p_id, const StringName &p_prop, const Variant &p_value);
	Error push_callable(const Callable &p_callable, const Variant **p_args, int p_argcount, bool p_show_error = false);
	Error push_callable(const Callable &p_callable, VARIANT_ARG_LIST);

	Error push_call(Object *p_object, const StringName &p_method, VARIANT_ARG_LIST);
	Error push_notification(Object *p_object, int p_notification);
	Error push_set(Object *p_object, const StringName &p_prop, const Variant &p_value);

	void statistics();
	void flush();

	bool is_flushing() const;

	int get_max_buffer_usage() const;

	MessageQueue();
	~MessageQueue();
};

#endif // MESSAGE_QUEUE_H
