/*************************************************************************/
/*  message_queue.cpp                                                    */
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

#include "message_queue.h"

#include "core/config/project_settings.h"
#include "core/core_string_names.h"
#include "core/object/script_language.h"

MessageQueue *MessageQueue::singleton = nullptr;

MessageQueue *MessageQueue::get_singleton() {
	return singleton;
}

Error MessageQueue::push_call(ObjectID p_id, const StringName &p_method, const Variant **p_args, int p_argcount, bool p_show_error) {
	return push_callable(Callable(p_id, p_method), p_args, p_argcount, p_show_error);
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

	if ((buffer_end + room_needed) >= buffer_size) {
		String type;
		if (ObjectDB::get_instance(p_id)) {
			type = ObjectDB::get_instance(p_id)->get_class();
		}
		print_line("Failed set: " + type + ":" + p_prop + " target ID: " + itos(p_id));
		statistics();
		ERR_FAIL_V_MSG(ERR_OUT_OF_MEMORY, "Message queue out of memory. Try increasing 'memory/limits/message_queue/max_size_kb' in project settings.");
	}

	Message *msg = memnew_placement(&buffer[buffer_end], Message);
	msg->args = 1;
	msg->callable = Callable(p_id, p_prop);
	msg->type = TYPE_SET;

	buffer_end += sizeof(Message);

	Variant *v = memnew_placement(&buffer[buffer_end], Variant);
	buffer_end += sizeof(Variant);
	*v = p_value;

	return OK;
}

Error MessageQueue::push_notification(ObjectID p_id, int p_notification) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(p_notification < 0, ERR_INVALID_PARAMETER);

	uint8_t room_needed = sizeof(Message);

	if ((buffer_end + room_needed) >= buffer_size) {
		print_line("Failed notification: " + itos(p_notification) + " target ID: " + itos(p_id));
		statistics();
		ERR_FAIL_V_MSG(ERR_OUT_OF_MEMORY, "Message queue out of memory. Try increasing 'memory/limits/message_queue/max_size_kb' in project settings.");
	}

	Message *msg = memnew_placement(&buffer[buffer_end], Message);

	msg->type = TYPE_NOTIFICATION;
	msg->callable = Callable(p_id, CoreStringNames::get_singleton()->notification); //name is meaningless but callable needs it
	//msg->target;
	msg->notification = p_notification;

	buffer_end += sizeof(Message);

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

Error MessageQueue::push_callable(const Callable &p_callable, const Variant **p_args, int p_argcount, bool p_show_error) {
	_THREAD_SAFE_METHOD_

	int room_needed = sizeof(Message) + sizeof(Variant) * p_argcount;

	if ((buffer_end + room_needed) >= buffer_size) {
		print_line("Failed method: " + p_callable);
		statistics();
		ERR_FAIL_V_MSG(ERR_OUT_OF_MEMORY, "Message queue out of memory. Try increasing 'memory/limits/message_queue/max_size_kb' in project settings.");
	}

	Message *msg = memnew_placement(&buffer[buffer_end], Message);
	msg->args = p_argcount;
	msg->callable = p_callable;
	msg->type = TYPE_CALL;
	if (p_show_error) {
		msg->type |= FLAG_SHOW_ERROR;
	}

	buffer_end += sizeof(Message);

	for (int i = 0; i < p_argcount; i++) {
		Variant *v = memnew_placement(&buffer[buffer_end], Variant);
		buffer_end += sizeof(Variant);
		*v = *p_args[i];
	}

	return OK;
}

Error MessageQueue::push_callable(const Callable &p_callable, VARIANT_ARG_DECLARE) {
	VARIANT_ARGPTRS;

	int argc = 0;

	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		if (argptr[i]->get_type() == Variant::NIL) {
			break;
		}
		argc++;
	}

	return push_callable(p_callable, argptr, argc);
}

void MessageQueue::statistics() {
	Map<StringName, int> set_count;
	Map<int, int> notify_count;
	Map<Callable, int> call_count;
	int null_count = 0;

	uint32_t read_pos = 0;
	while (read_pos < buffer_end) {
		Message *message = (Message *)&buffer[read_pos];

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

	print_line("TOTAL BYTES: " + itos(buffer_end));
	print_line("NULL count: " + itos(null_count));

	for (Map<StringName, int>::Element *E = set_count.front(); E; E = E->next()) {
		print_line("SET " + E->key() + ": " + itos(E->get()));
	}

	for (Map<Callable, int>::Element *E = call_count.front(); E; E = E->next()) {
		print_line("CALL " + E->key() + ": " + itos(E->get()));
	}

	for (Map<int, int>::Element *E = notify_count.front(); E; E = E->next()) {
		print_line("NOTIFY " + itos(E->key()) + ": " + itos(E->get()));
	}
}

int MessageQueue::get_max_buffer_usage() const {
	return buffer_max_used;
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
	p_callable.call(argptrs, p_argcount, ret, ce);
	if (p_show_error && ce.error != Callable::CallError::CALL_OK) {
		ERR_PRINT("Error calling deferred method: " + Variant::get_callable_error_text(p_callable, argptrs, p_argcount, ce) + ".");
	}
}

void MessageQueue::flush() {
	if (buffer_end > buffer_max_used) {
		buffer_max_used = buffer_end;
	}

	uint32_t read_pos = 0;

	//using reverse locking strategy
	_THREAD_SAFE_LOCK_

	if (flushing) {
		_THREAD_SAFE_UNLOCK_
		ERR_FAIL_COND(flushing); //already flushing, you did something odd
	}
	flushing = true;

	while (read_pos < buffer_end) {
		//lock on each iteration, so a call can re-add itself to the message queue

		Message *message = (Message *)&buffer[read_pos];

		uint32_t advance = sizeof(Message);
		if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION) {
			advance += sizeof(Variant) * message->args;
		}

		//pre-advance so this function is reentrant
		read_pos += advance;

		_THREAD_SAFE_UNLOCK_

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

		_THREAD_SAFE_LOCK_
	}

	buffer_end = 0; // reset buffer
	flushing = false;
	_THREAD_SAFE_UNLOCK_
}

bool MessageQueue::is_flushing() const {
	return flushing;
}

MessageQueue::MessageQueue() {
	ERR_FAIL_COND_MSG(singleton != nullptr, "A MessageQueue singleton already exists.");
	singleton = this;

	buffer_size = GLOBAL_DEF_RST("memory/limits/message_queue/max_size_kb", DEFAULT_QUEUE_SIZE_KB);
	ProjectSettings::get_singleton()->set_custom_property_info("memory/limits/message_queue/max_size_kb", PropertyInfo(Variant::INT, "memory/limits/message_queue/max_size_kb", PROPERTY_HINT_RANGE, "1024,4096,1,or_greater"));
	buffer_size *= 1024;
	buffer = memnew_arr(uint8_t, buffer_size);
}

MessageQueue::~MessageQueue() {
	uint32_t read_pos = 0;

	while (read_pos < buffer_end) {
		Message *message = (Message *)&buffer[read_pos];
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

	singleton = nullptr;
	memdelete_arr(buffer);
}
