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

#include "core/config/project_settings.h"
#include "core/core_string_names.h"
#include "core/object/class_db.h"
#include "core/object/script_language.h"

MessageQueue *MessageQueue::singleton = nullptr;

MessageQueue *MessageQueue::get_singleton() {
	return singleton;
}

Error MessageQueue::push_callp(ObjectID p_id, const StringName &p_method, const Variant **p_args, int p_argcount, bool p_show_error) {
	return push_callablep(Callable(p_id, p_method), p_args, p_argcount, p_show_error);
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
	msg->callable = Callable(p_id, p_prop);
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
	msg->callable = Callable(p_id, CoreStringNames::get_singleton()->notification); //name is meaningless but callable needs it
	//msg->target;
	msg->notification = p_notification;

	buffer.end += sizeof(Message);

	return OK;
}

Error MessageQueue::push_callp(Object *p_object, const StringName &p_method, const Variant **p_args, int p_argcount, bool p_show_error) {
	return push_callp(p_object->get_instance_id(), p_method, p_args, p_argcount, p_show_error);
}

Error MessageQueue::push_notification(Object *p_object, int p_notification) {
	return push_notification(p_object->get_instance_id(), p_notification);
}

Error MessageQueue::push_set(Object *p_object, const StringName &p_prop, const Variant &p_value) {
	return push_set(p_object->get_instance_id(), p_prop, p_value);
}

Error MessageQueue::push_callablep(const Callable &p_callable, const Variant **p_args, int p_argcount, bool p_show_error) {
	_THREAD_SAFE_METHOD_

	int room_needed = sizeof(Message) + sizeof(Variant) * p_argcount;
	Buffer &buffer = buffers[write_buffer];

	if ((buffer.end + room_needed) > buffer.data.size()) {
		if ((buffer.end + room_needed) > max_allowed_buffer_size) {
			String type;
			if (ObjectDB::get_instance(p_callable.get_object_id())) {
				type = ObjectDB::get_instance(p_callable.get_object_id())->get_class();
			}
			print_line("Failed method: " + p_callable);
			statistics();
			ERR_FAIL_V_MSG(ERR_OUT_OF_MEMORY, "Message queue out of memory. Try increasing 'memory/limits/message_queue/max_size_mb' in project settings.");
		} else {
			buffer.data.resize(buffer.end + room_needed);
		}
	}

	Message *msg = memnew_placement(&buffer.data[buffer.end], Message);
	msg->args = p_argcount;
	msg->callable = p_callable;
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

void MessageQueue::statistics() {
	HashMap<StringName, int> set_count;
	HashMap<int, int> notify_count;
	HashMap<Callable, int> call_count;
	int null_count = 0;

	Buffer &buffer = buffers[write_buffer];

	uint32_t read_pos = 0;
	while (read_pos < buffer.end) {
		Message *message = (Message *)&buffer.data[read_pos];

		Object *target = message->callable.get_object();

		if (target != nullptr) {
			switch (message->type & FLAG_MASK) {
				case TYPE_CALL: {
					if (!call_count.has(message->callable)) {
						call_count[message->callable] = 0;
					}

					call_count[message->callable]++;

				} break;
				case TYPE_NOTIFICATION: {
					if (!notify_count.has(message->notification)) {
						notify_count[message->notification] = 0;
					}

					notify_count[message->notification]++;

				} break;
				case TYPE_SET: {
					StringName t = message->callable.get_method();
					if (!set_count.has(t)) {
						set_count[t] = 0;
					}

					set_count[t]++;

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

	for (const KeyValue<StringName, int> &E : set_count) {
		print_line("SET " + E.key + ": " + itos(E.value));
	}

	for (const KeyValue<Callable, int> &E : call_count) {
		print_line("CALL " + E.key + ": " + itos(E.value));
	}

	for (const KeyValue<int, int> &E : notify_count) {
		print_line("NOTIFY " + itos(E.key) + ": " + itos(E.value));
	}
}

int MessageQueue::get_max_buffer_usage() const {
	// Note this may be better read_buffer, or a combination, depending when this is read.
	return buffers[write_buffer].data.size();
}

void MessageQueue::_call_function(const Callable &p_callable, const Variant *p_args, int p_argcount, bool p_show_error) {
	const Variant **argptrs = nullptr;
	if (p_argcount) {
		argptrs = (const Variant **)alloca(sizeof(Variant *) * p_argcount);
		for (int i = 0; i < p_argcount; i++) {
			argptrs[i] = &p_args[i];
		}
	}

	Callable::CallError ce;
	Variant ret;
	p_callable.callp(argptrs, p_argcount, ret, ce);
	if (p_show_error && ce.error != Callable::CallError::CALL_OK) {
		ERR_PRINT("Error calling deferred method: " + Variant::get_callable_error_text(p_callable, argptrs, p_argcount, ce) + ".");
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

			Object *target = message->callable.get_object();

			if (target != nullptr) {
				switch (message->type & FLAG_MASK) {
					case TYPE_CALL: {
						Variant *args = (Variant *)(message + 1);

						// messages don't expect a return value

						_call_function(message->callable, args, message->args, message->type & FLAG_SHOW_ERROR);

					} break;
					case TYPE_NOTIFICATION: {
						// messages don't expect a return value
						target->notification(message->notification);

					} break;
					case TYPE_SET: {
						Variant *arg = (Variant *)(message + 1);
						// messages don't expect a return value
						target->set(message->callable.get_method(), *arg);

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

	max_allowed_buffer_size = GLOBAL_DEF_RST(PropertyInfo(Variant::INT, "memory/limits/message_queue/max_size_mb", PROPERTY_HINT_RANGE, "4,512,1,or_greater"), 32);
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
