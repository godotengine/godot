/**************************************************************************/
/*  message_queue.cpp                                                     */
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

#include "message_queue.h"

#include "core/project_settings.h"
#include "core/script_language.h"

MessageQueue *MessageQueue::singleton = nullptr;

MessageQueue *MessageQueue::get_singleton() {
	return singleton;
}

Error MessageQueue::push_call(ObjectID p_id, const StringName &p_method, const Variant **p_args, int p_argcount, bool p_show_error) {
	_THREAD_SAFE_METHOD_

	int room_needed = sizeof(Message) + sizeof(Variant) * p_argcount;

	Buffer &buffer = buffers[write_buffer];

	if ((buffer.end + room_needed) > buffer.data.size()) {
		if ((buffer.end + room_needed) > max_allowed_buffer_size) {
			String type;
			if (ObjectDB::get_instance(p_id)) {
				type = ObjectDB::get_instance(p_id)->get_class();
			}
			print_line("Failed method: " + p_method);
			statistics();
			ERR_FAIL_V_MSG(ERR_OUT_OF_MEMORY, "Message queue out of memory. Try increasing 'memory/limits/message_queue/max_size_mb' in project settings.");
		} else {
			buffer.data.resize(buffer.end + room_needed);
		}
	}

	Message *msg = memnew_placement(&buffer.data[buffer.end], Message);

	msg->args = p_argcount;
	msg->instance_id = p_id;
	msg->target = p_method;
	msg->type = TYPE_CALL;
	if (p_show_error) {
		msg->type |= FLAG_SHOW_ERROR;
	}

	buffer.end += sizeof(Message);

	for (int i = 0; i < p_argcount; i++) {
		Variant *v = memnew_placement(&buffer.data[buffer.end], Variant);
		buffer.end += sizeof(Variant);
		*v = *p_args[i];
	}

	return OK;
}

Error MessageQueue::push_call(ObjectID p_id, const StringName &p_method, VARIANT_ARG_DECLARE) {
	VARIANT_ARGPTRS;

	int argc = 0;

	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		if (argptr[i]->get_type() == Variant::NIL) {
			break;
		}
		argc++;
	}

	return push_call(p_id, p_method, argptr, argc, false);
}

Error MessageQueue::push_set(ObjectID p_id, const StringName &p_prop, const Variant &p_value) {
	_THREAD_SAFE_METHOD_

	uint8_t room_needed = sizeof(Message) + sizeof(Variant);

	Buffer &buffer = buffers[write_buffer];

	if ((buffer.end + room_needed) > buffer.data.size()) {
		if ((buffer.end + room_needed) > max_allowed_buffer_size) {
			String type;
			if (ObjectDB::get_instance(p_id)) {
				type = ObjectDB::get_instance(p_id)->get_class();
			}
			print_line("Failed set: " + type + ":" + p_prop + " target ID: " + itos(p_id));
			statistics();
			ERR_FAIL_V_MSG(ERR_OUT_OF_MEMORY, "Message queue out of memory. Try increasing 'memory/limits/message_queue/max_size_mb' in project settings.");
		} else {
			buffer.data.resize(buffer.end + room_needed);
		}
	}

	Message *msg = memnew_placement(&buffer.data[buffer.end], Message);

	msg->args = 1;
	msg->instance_id = p_id;
	msg->target = p_prop;
	msg->type = TYPE_SET;

	buffer.end += sizeof(Message);

	Variant *v = memnew_placement(&buffer.data[buffer.end], Variant);
	buffer.end += sizeof(Variant);
	*v = p_value;

	return OK;
}

Error MessageQueue::push_notification(ObjectID p_id, int p_notification) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(p_notification < 0, ERR_INVALID_PARAMETER);

	uint8_t room_needed = sizeof(Message);

	Buffer &buffer = buffers[write_buffer];

	if ((buffer.end + room_needed) > buffer.data.size()) {
		if ((buffer.end + room_needed) > max_allowed_buffer_size) {
			String type;
			if (ObjectDB::get_instance(p_id)) {
				type = ObjectDB::get_instance(p_id)->get_class();
			}
			print_line("Failed notification: " + itos(p_notification) + " target ID: " + itos(p_id));
			statistics();
			ERR_FAIL_V_MSG(ERR_OUT_OF_MEMORY, "Message queue out of memory. Try increasing 'memory/limits/message_queue/max_size_mb' in project settings.");
		} else {
			buffer.data.resize(buffer.end + room_needed);
		}
	}

	Message *msg = memnew_placement(&buffer.data[buffer.end], Message);

	msg->type = TYPE_NOTIFICATION;
	msg->instance_id = p_id;
	//msg->target;
	msg->notification = p_notification;

	buffer.end += sizeof(Message);

	return OK;
}

Error MessageQueue::push_call(Object *p_object, const StringName &p_method, VARIANT_ARG_DECLARE) {
	return push_call(p_object->get_instance_id(), p_method, VARIANT_ARG_PASS);
}

Error MessageQueue::push_notification(Object *p_object, int p_notification) {
	return push_notification(p_object->get_instance_id(), p_notification);
}
Error MessageQueue::push_set(Object *p_object, const StringName &p_prop, const Variant &p_value) {
	return push_set(p_object->get_instance_id(), p_prop, p_value);
}

void MessageQueue::statistics() {
	Map<StringName, int> set_count;
	Map<int, int> notify_count;
	Map<StringName, int> call_count;
	int null_count = 0;

	Buffer &buffer = buffers[write_buffer];

	uint32_t read_pos = 0;
	while (read_pos < buffer.end) {
		Message *message = (Message *)&buffer.data[read_pos];

		Object *target = ObjectDB::get_instance(message->instance_id);

		if (target != nullptr) {
			switch (message->type & FLAG_MASK) {
				case TYPE_CALL: {
					if (!call_count.has(message->target)) {
						call_count[message->target] = 0;
					}

					call_count[message->target]++;

				} break;
				case TYPE_NOTIFICATION: {
					if (!notify_count.has(message->notification)) {
						notify_count[message->notification] = 0;
					}

					notify_count[message->notification]++;

				} break;
				case TYPE_SET: {
					if (!set_count.has(message->target)) {
						set_count[message->target] = 0;
					}

					set_count[message->target]++;

				} break;
			}

		} else {
			//object was deleted
			print_line("Object was deleted while awaiting a callback");

			null_count++;
		}

		read_pos += sizeof(Message);
		if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION) {
			read_pos += sizeof(Variant) * message->args;
		}
	}

	print_line("TOTAL BYTES: " + itos(buffer.end));
	print_line("NULL count: " + itos(null_count));

	for (Map<StringName, int>::Element *E = set_count.front(); E; E = E->next()) {
		print_line("SET " + E->key() + ": " + itos(E->get()));
	}

	for (Map<StringName, int>::Element *E = call_count.front(); E; E = E->next()) {
		print_line("CALL " + E->key() + ": " + itos(E->get()));
	}

	for (Map<int, int>::Element *E = notify_count.front(); E; E = E->next()) {
		print_line("NOTIFY " + itos(E->key()) + ": " + itos(E->get()));
	}
}

int MessageQueue::get_max_buffer_usage() const {
	return _buffer_size_monitor.max_size_overall;
}

void MessageQueue::_call_function(Object *p_target, const StringName &p_func, const Variant *p_args, int p_argcount, bool p_show_error) {
	const Variant **argptrs = nullptr;
	if (p_argcount) {
		argptrs = (const Variant **)alloca(sizeof(Variant *) * p_argcount);
		for (int i = 0; i < p_argcount; i++) {
			argptrs[i] = &p_args[i];
		}
	}

	Variant::CallError ce;
	p_target->call(p_func, argptrs, p_argcount, ce);
	if (p_show_error && ce.error != Variant::CallError::CALL_OK) {
		ERR_PRINT("Error calling deferred method: " + Variant::get_call_error_text(p_target, p_func, argptrs, p_argcount, ce) + ".");
	}
}

void MessageQueue::_update_buffer_monitor() {
	// The number of flushes is an approximate delay before
	// considering shrinking. This is somewhat of a magic number,
	// but only acts to prevent excessive oscillations.
	if (++_buffer_size_monitor.flush_count == 8192) {
		uint32_t max_size = _buffer_size_monitor.max_size;

		// Uncomment this define to log message queue sizes and
		// auto-shrinking behaviour.
		// #define GODOT_DEBUG_MESSAGE_QUEUE_SIZES
#ifdef GODOT_DEBUG_MESSAGE_QUEUE_SIZES
		print_line("MessageQueue buffer max size " + itos(max_size) + " bytes.");
#endif

		// reset for next time
		_buffer_size_monitor.flush_count = 0;
		_buffer_size_monitor.max_size = 0;

		for (uint32_t n = 0; n < 2; n++) {
			uint32_t cap = buffers[n].data.get_capacity();

			// Only worry about reducing memory if the capacity is high
			// (due to e.g. loading a level or something).
			// The shrinking will only take place below 256K, to prevent
			// excessive reallocating.
			if (cap > (256 * 1024)) {
				// Only shrink if we are routinely using a lot less than the capacity.
				if ((max_size * 4) < cap) {
					buffers[n].data.reserve(cap / 2, true);
#ifdef GODOT_DEBUG_MESSAGE_QUEUE_SIZES
					print_line("MessageQueue reducing buffer[" + itos(n) + "] capacity from " + itos(cap) + " bytes to " + itos(cap / 2) + " bytes.");
#endif
				}
			}
		}
	}
}

void MessageQueue::flush() {
	//using reverse locking strategy
	_THREAD_SAFE_LOCK_

	if (flushing) {
		_THREAD_SAFE_UNLOCK_
		ERR_FAIL_MSG("Already flushing"); //already flushing, you did something odd
	}

	// first flip buffers, in preparation
	SWAP(read_buffer, write_buffer);

	flushing = true;
	_update_buffer_monitor();
	_THREAD_SAFE_UNLOCK_

	// This loop works by having a read buffer and write buffer.
	// While we are reading from one buffer we can be filling another.
	// This enables them to be independent, and not require locks per message.
	// It also avoids pushing and resizing the write buffer corrupting the read buffer.
	// The trade off is that it requires more memory.
	// However the peak size of each can be lower, because they do not ADD
	// to each other during transit.

	while (buffers[read_buffer].data.size()) {
		uint32_t read_pos = 0;
		Buffer &buffer = buffers[read_buffer];

		while (read_pos < buffer.end) {
			Message *message = (Message *)&buffer.data[read_pos];

			uint32_t advance = sizeof(Message);
			if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION) {
				advance += sizeof(Variant) * message->args;
			}

			read_pos += advance;

			Object *target = ObjectDB::get_instance(message->instance_id);

			if (target != nullptr) {
				switch (message->type & FLAG_MASK) {
					case TYPE_CALL: {
						Variant *args = (Variant *)(message + 1);

						// messages don't expect a return value

						_call_function(target, message->target, args, message->args, message->type & FLAG_SHOW_ERROR);

					} break;
					case TYPE_NOTIFICATION: {
						// messages don't expect a return value
						target->notification(message->notification);

					} break;
					case TYPE_SET: {
						Variant *arg = (Variant *)(message + 1);
						// messages don't expect a return value
						target->set(message->target, *arg);

					} break;
				}
			}

			if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION) {
				Variant *args = (Variant *)(message + 1);
				for (int i = 0; i < message->args; i++) {
					args[i].~Variant();
				}
			}

			message->~Message();

		} // while going through buffer

		buffer.end = 0; // reset buffer

		uint32_t buffer_data_size = buffer.data.size();
		buffer.data.clear();

		_THREAD_SAFE_LOCK_
		// keep track of the maximum used size, so we can downsize buffers when appropriate
		_buffer_size_monitor.max_size = MAX(buffer_data_size, _buffer_size_monitor.max_size);
		_buffer_size_monitor.max_size_overall = MAX(buffer_data_size, _buffer_size_monitor.max_size_overall);

		// flip buffers, this is the only part that requires a lock
		SWAP(read_buffer, write_buffer);
		_THREAD_SAFE_UNLOCK_

	} // while read buffer not empty

	_THREAD_SAFE_LOCK_
	flushing = false;
	_THREAD_SAFE_UNLOCK_
}

bool MessageQueue::is_flushing() const {
	return flushing;
}

MessageQueue::MessageQueue() {
	ERR_FAIL_COND_MSG(singleton != nullptr, "A MessageQueue singleton already exists.");
	singleton = this;
	flushing = false;

	max_allowed_buffer_size = GLOBAL_DEF_RST("memory/limits/message_queue/max_size_mb", 32);
	ProjectSettings::get_singleton()->set_custom_property_info("memory/limits/message_queue/max_size_mb", PropertyInfo(Variant::INT, "memory/limits/message_queue/max_size_mb", PROPERTY_HINT_RANGE, "4,512,1,or_greater"));

	max_allowed_buffer_size *= 1024 * 1024;
}

MessageQueue::~MessageQueue() {
	for (int which = 0; which < 2; which++) {
		Buffer &buffer = buffers[which];
		uint32_t read_pos = 0;

		while (read_pos < buffer.end) {
			Message *message = (Message *)&buffer.data[read_pos];
			Variant *args = (Variant *)(message + 1);
			int argc = message->args;
			if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION) {
				for (int i = 0; i < argc; i++) {
					args[i].~Variant();
				}
			}
			message->~Message();

			read_pos += sizeof(Message);
			if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION) {
				read_pos += sizeof(Variant) * message->args;
			}
		}

	} // for which

	singleton = nullptr;
}
