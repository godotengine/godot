/*************************************************************************/
/*  message_queue.cpp                                                    */
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
#include "message_queue.h"

#include "global_config.h"
#include "script_language.h"

MessageQueue *MessageQueue::singleton = NULL;

MessageQueue *MessageQueue::get_singleton() {

	return singleton;
}

Error MessageQueue::push_call(ObjectID p_id, const StringName &p_method, const Variant **p_args, int p_argcount, bool p_show_error) {

	_THREAD_SAFE_METHOD_

	int room_needed = sizeof(Message) + sizeof(Variant) * p_argcount;

	if ((buffer_end + room_needed) >= buffer_size) {
		String type;
		if (ObjectDB::get_instance(p_id))
			type = ObjectDB::get_instance(p_id)->get_class();
		print_line("failed method: " + type + ":" + p_method + " target ID: " + itos(p_id));
		statistics();
	}

	ERR_FAIL_COND_V((buffer_end + room_needed) >= buffer_size, ERR_OUT_OF_MEMORY);
	Message *msg = memnew_placement(&buffer[buffer_end], Message);
	msg->args = p_argcount;
	msg->instance_ID = p_id;
	msg->target = p_method;
	msg->type = TYPE_CALL;
	if (p_show_error)
		msg->type |= FLAG_SHOW_ERROR;

	buffer_end += sizeof(Message);

	for (int i = 0; i < p_argcount; i++) {

		Variant *v = memnew_placement(&buffer[buffer_end], Variant);
		buffer_end += sizeof(Variant);
		*v = *p_args[i];
	}

	return OK;
}

Error MessageQueue::push_call(ObjectID p_id, const StringName &p_method, VARIANT_ARG_DECLARE) {

	VARIANT_ARGPTRS;

	int argc = 0;

	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		if (argptr[i]->get_type() == Variant::NIL)
			break;
		argc++;
	}

	return push_call(p_id, p_method, argptr, argc, false);
}

Error MessageQueue::push_set(ObjectID p_id, const StringName &p_prop, const Variant &p_value) {

	_THREAD_SAFE_METHOD_

	uint8_t room_needed = sizeof(Message) + sizeof(Variant);

	if ((buffer_end + room_needed) >= buffer_size) {
		String type;
		if (ObjectDB::get_instance(p_id))
			type = ObjectDB::get_instance(p_id)->get_class();
		print_line("failed set: " + type + ":" + p_prop + " target ID: " + itos(p_id));
		statistics();
	}

	ERR_FAIL_COND_V((buffer_end + room_needed) >= buffer_size, ERR_OUT_OF_MEMORY);

	Message *msg = memnew_placement(&buffer[buffer_end], Message);
	msg->args = 1;
	msg->instance_ID = p_id;
	msg->target = p_prop;
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
		String type;
		if (ObjectDB::get_instance(p_id))
			type = ObjectDB::get_instance(p_id)->get_class();
		print_line("failed notification: " + itos(p_notification) + " target ID: " + itos(p_id));
		statistics();
	}

	ERR_FAIL_COND_V((buffer_end + room_needed) >= buffer_size, ERR_OUT_OF_MEMORY);
	Message *msg = memnew_placement(&buffer[buffer_end], Message);

	msg->type = TYPE_NOTIFICATION;
	msg->instance_ID = p_id;
	//msg->target;
	msg->notification = p_notification;

	buffer_end += sizeof(Message);

	return OK;
}

Error MessageQueue::push_call(Object *p_object, const StringName &p_method, VARIANT_ARG_DECLARE) {

	return push_call(p_object->get_instance_ID(), p_method, VARIANT_ARG_PASS);
}

Error MessageQueue::push_notification(Object *p_object, int p_notification) {

	return push_notification(p_object->get_instance_ID(), p_notification);
}
Error MessageQueue::push_set(Object *p_object, const StringName &p_prop, const Variant &p_value) {

	return push_set(p_object->get_instance_ID(), p_prop, p_value);
}

void MessageQueue::statistics() {

	Map<StringName, int> set_count;
	Map<int, int> notify_count;
	Map<StringName, int> call_count;
	int null_count = 0;

	uint32_t read_pos = 0;
	while (read_pos < buffer_end) {
		Message *message = (Message *)&buffer[read_pos];

		Object *target = ObjectDB::get_instance(message->instance_ID);

		if (target != NULL) {

			switch (message->type & FLAG_MASK) {

				case TYPE_CALL: {

					if (!call_count.has(message->target))
						call_count[message->target] = 0;

					call_count[message->target]++;

				} break;
				case TYPE_NOTIFICATION: {

					if (!notify_count.has(message->notification))
						notify_count[message->notification] = 0;

					notify_count[message->notification]++;

				} break;
				case TYPE_SET: {

					if (!set_count.has(message->target))
						set_count[message->target] = 0;

					set_count[message->target]++;

				} break;
			}

			//object was deleted
			//WARN_PRINT("Object was deleted while awaiting a callback")
			//should it print a warning?
		} else {

			null_count++;
		}

		read_pos += sizeof(Message);
		if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION)
			read_pos += sizeof(Variant) * message->args;
	}

	print_line("TOTAL BYTES: " + itos(buffer_end));
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

bool MessageQueue::print() {
#if 0
	uint32_t read_pos=0;
	while (read_pos < buffer_end ) {
		Message *message = (Message*)&buffer[ read_pos ];

		Object *target = ObjectDB::get_instance(message->instance_ID);
		String cname;
		String cfunc;

		if (target==NULL) {
			//object was deleted
			//WARN_PRINT("Object was deleted while awaiting a callback")
			//should it print a warning?
		} else if (message->notification>=0) {

			// messages don't expect a return value
			cfunc="notification # "+itos(message->notification);
			cname=target->get_type();

		} else if (!message->target.empty()) {

			cfunc="property:  "+message->target;
			cname=target->get_type();


		} else if (message->target) {

			cfunc=String(message->target)+"()";
			cname=target->get_type();
		}


		read_pos+=sizeof(Message);
		if (message->type!=TYPE_NOTIFICATION)
			read_pos+=sizeof(Variant)*message->args;
	}
#endif
	return false;
}

int MessageQueue::get_max_buffer_usage() const {

	return buffer_max_used;
}

void MessageQueue::_call_function(Object *p_target, const StringName &p_func, const Variant *p_args, int p_argcount, bool p_show_error) {

	const Variant **argptrs = NULL;
	if (p_argcount) {
		argptrs = (const Variant **)alloca(sizeof(Variant *) * p_argcount);
		for (int i = 0; i < p_argcount; i++) {
			argptrs[i] = &p_args[i];
		}
	}

	Variant::CallError ce;
	p_target->call(p_func, argptrs, p_argcount, ce);
	if (p_show_error && ce.error != Variant::CallError::CALL_OK) {

		ERR_PRINTS("Error calling deferred method: " + Variant::get_call_error_text(p_target, p_func, argptrs, p_argcount, ce));
	}
}

void MessageQueue::flush() {

	if (buffer_end > buffer_max_used) {
		buffer_max_used = buffer_end;
		//statistics();
	}

	uint32_t read_pos = 0;

	//using reverse locking strategy
	_THREAD_SAFE_LOCK_

	while (read_pos < buffer_end) {

		_THREAD_SAFE_UNLOCK_

		//lock on each interation, so a call can re-add itself to the message queue

		Message *message = (Message *)&buffer[read_pos];

		Object *target = ObjectDB::get_instance(message->instance_ID);

		if (target != NULL) {

			switch (message->type & FLAG_MASK) {
				case TYPE_CALL: {

					Variant *args = (Variant *)(message + 1);

					// messages don't expect a return value

					_call_function(target, message->target, args, message->args, message->type & FLAG_SHOW_ERROR);

					for (int i = 0; i < message->args; i++) {
						args[i].~Variant();
					}

				} break;
				case TYPE_NOTIFICATION: {

					// messages don't expect a return value
					target->notification(message->notification);

				} break;
				case TYPE_SET: {

					Variant *arg = (Variant *)(message + 1);
					// messages don't expect a return value
					target->set(message->target, *arg);

					arg->~Variant();
				} break;
			}
		}

		uint32_t advance = sizeof(Message);
		if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION)
			advance += sizeof(Variant) * message->args;
		message->~Message();

		_THREAD_SAFE_LOCK_
		read_pos += advance;
	}

	buffer_end = 0; // reset buffer
	_THREAD_SAFE_UNLOCK_
}

MessageQueue::MessageQueue() {

	ERR_FAIL_COND(singleton != NULL);
	singleton = this;

	buffer_end = 0;
	buffer_max_used = 0;
	buffer_size = GLOBAL_DEF("memory/buffers/message_queue_max_size_kb", DEFAULT_QUEUE_SIZE_KB);
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
			for (int i = 0; i < argc; i++)
				args[i].~Variant();
		}
		message->~Message();

		read_pos += sizeof(Message);
		if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION)
			read_pos += sizeof(Variant) * message->args;
	}

	singleton = NULL;
	memdelete_arr(buffer);
}
