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
#include "core/object/class_db.h"
#include "core/object/script_language.h"

#include <cstdio>

#ifdef DEV_ENABLED
// Includes safety checks to ensure that a queue set as a thread singleton override
// is only ever called from the thread it was set for.
#define LOCK_MUTEX                                \
	if (this != MessageQueue::thread_singleton) { \
		DEV_ASSERT(!is_current_thread_override);  \
		mutex.lock();                             \
	} else {                                      \
		DEV_ASSERT(is_current_thread_override);   \
	}
#else
#define LOCK_MUTEX                                \
	if (this != MessageQueue::thread_singleton) { \
		mutex.lock();                             \
	}
#endif

#define UNLOCK_MUTEX                              \
	if (this != MessageQueue::thread_singleton) { \
		mutex.unlock();                           \
	}

void CallQueue::_add_page() {
	if (pages_used == page_bytes.size()) {
		pages.push_back(allocator->alloc());
		page_bytes.push_back(0);
	}
	page_bytes[pages_used] = 0;
	pages_used++;
}

Error CallQueue::push_callp(ObjectID p_id, const StringName &p_method, const Variant **p_args, int p_argcount, bool p_show_error) {
	return push_callablep(Callable(p_id, p_method), p_args, p_argcount, p_show_error);
}

Error CallQueue::push_callp(Object *p_object, const StringName &p_method, const Variant **p_args, int p_argcount, bool p_show_error) {
	return push_callp(p_object->get_instance_id(), p_method, p_args, p_argcount, p_show_error);
}

Error CallQueue::push_notification(Object *p_object, int p_notification) {
	return push_notification(p_object->get_instance_id(), p_notification);
}

Error CallQueue::push_set(Object *p_object, const StringName &p_prop, const Variant &p_value) {
	return push_set(p_object->get_instance_id(), p_prop, p_value);
}

Error CallQueue::push_callablep(const Callable &p_callable, const Variant **p_args, int p_argcount, bool p_show_error) {
	uint32_t room_needed = sizeof(Message) + sizeof(Variant) * p_argcount;

	ERR_FAIL_COND_V_MSG(room_needed > uint32_t(PAGE_SIZE_BYTES), ERR_INVALID_PARAMETER, "Message is too large to fit on a page (" + itos(PAGE_SIZE_BYTES) + " bytes), consider passing less arguments.");

	LOCK_MUTEX;

	_ensure_first_page();

	if ((page_bytes[pages_used - 1] + room_needed) > uint32_t(PAGE_SIZE_BYTES)) {
		if (pages_used == max_pages) {
			fprintf(stderr, "Failed method: %s. Message queue out of memory. %s\n", String(p_callable).utf8().get_data(), error_text.utf8().get_data());
			statistics();
			UNLOCK_MUTEX;
			return ERR_OUT_OF_MEMORY;
		}
		_add_page();
	}

	Page *page = pages[pages_used - 1];

	uint8_t *buffer_end = &page->data[page_bytes[pages_used - 1]];

	Message *msg = memnew_placement(buffer_end, Message);
	msg->args = p_argcount;
	msg->callable = p_callable;
	msg->type = TYPE_CALL;
	if (p_show_error) {
		msg->type |= FLAG_SHOW_ERROR;
	}
	// Support callables of static methods.
	if (p_callable.get_object_id().is_null() && p_callable.is_valid()) {
		msg->type |= FLAG_NULL_IS_OK;
	}

	buffer_end += sizeof(Message);

	for (int i = 0; i < p_argcount; i++) {
		Variant *v = memnew_placement(buffer_end, Variant);
		buffer_end += sizeof(Variant);
		*v = *p_args[i];
	}

	page_bytes[pages_used - 1] += room_needed;

	UNLOCK_MUTEX;

	return OK;
}

Error CallQueue::push_set(ObjectID p_id, const StringName &p_prop, const Variant &p_value) {
	LOCK_MUTEX;
	uint32_t room_needed = sizeof(Message) + sizeof(Variant);

	_ensure_first_page();

	if ((page_bytes[pages_used - 1] + room_needed) > uint32_t(PAGE_SIZE_BYTES)) {
		if (pages_used == max_pages) {
			String type;
			if (ObjectDB::get_instance(p_id)) {
				type = ObjectDB::get_instance(p_id)->get_class();
			}
			fprintf(stderr, "Failed set: %s: %s target ID: %s. Message queue out of memory. %s\n", type.utf8().get_data(), String(p_prop).utf8().get_data(), itos(p_id).utf8().get_data(), error_text.utf8().get_data());
			statistics();

			UNLOCK_MUTEX;
			return ERR_OUT_OF_MEMORY;
		}
		_add_page();
	}

	Page *page = pages[pages_used - 1];
	uint8_t *buffer_end = &page->data[page_bytes[pages_used - 1]];

	Message *msg = memnew_placement(buffer_end, Message);
	msg->args = 1;
	msg->callable = Callable(p_id, p_prop);
	msg->type = TYPE_SET;

	buffer_end += sizeof(Message);

	Variant *v = memnew_placement(buffer_end, Variant);
	*v = p_value;

	page_bytes[pages_used - 1] += room_needed;
	UNLOCK_MUTEX;

	return OK;
}

Error CallQueue::push_notification(ObjectID p_id, int p_notification) {
	ERR_FAIL_COND_V(p_notification < 0, ERR_INVALID_PARAMETER);
	LOCK_MUTEX;
	uint32_t room_needed = sizeof(Message);

	_ensure_first_page();

	if ((page_bytes[pages_used - 1] + room_needed) > uint32_t(PAGE_SIZE_BYTES)) {
		if (pages_used == max_pages) {
			fprintf(stderr, "Failed notification: %d target ID: %s. Message queue out of memory. %s\n", p_notification, itos(p_id).utf8().get_data(), error_text.utf8().get_data());
			statistics();
			UNLOCK_MUTEX;
			return ERR_OUT_OF_MEMORY;
		}
		_add_page();
	}

	Page *page = pages[pages_used - 1];
	uint8_t *buffer_end = &page->data[page_bytes[pages_used - 1]];

	Message *msg = memnew_placement(buffer_end, Message);

	msg->type = TYPE_NOTIFICATION;
	msg->callable = Callable(p_id, CoreStringName(notification)); //name is meaningless but callable needs it
	//msg->target;
	msg->notification = p_notification;

	page_bytes[pages_used - 1] += room_needed;
	UNLOCK_MUTEX;

	return OK;
}

void CallQueue::_call_function(const Callable &p_callable, const Variant *p_args, int p_argcount, bool p_show_error) {
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

Error CallQueue::flush() {
	LOCK_MUTEX;

	if (pages.is_empty()) {
		// Never allocated
		UNLOCK_MUTEX;
		return OK; // Do nothing.
	}

	if (flushing) {
		UNLOCK_MUTEX;
		return ERR_BUSY;
	}

	flushing = true;

	uint32_t i = 0;
	uint32_t offset = 0;

	while (i < pages_used && offset < page_bytes[i]) {
		Page *page = pages[i];

		//lock on each iteration, so a call can re-add itself to the message queue

		Message *message = (Message *)&page->data[offset];

		uint32_t advance = sizeof(Message);
		if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION) {
			advance += sizeof(Variant) * message->args;
		}

		//pre-advance so this function is reentrant
		offset += advance;

		Object *target = message->callable.get_object();

		UNLOCK_MUTEX;

		switch (message->type & FLAG_MASK) {
			case TYPE_CALL: {
				if (target || (message->type & FLAG_NULL_IS_OK)) {
					Variant *args = (Variant *)(message + 1);
					_call_function(message->callable, args, message->args, message->type & FLAG_SHOW_ERROR);
				}
			} break;
			case TYPE_NOTIFICATION: {
				if (target) {
					target->notification(message->notification);
				}
			} break;
			case TYPE_SET: {
				if (target) {
					Variant *arg = (Variant *)(message + 1);
					target->set(message->callable.get_method(), *arg);
				}
			} break;
		}

		if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION) {
			Variant *args = (Variant *)(message + 1);
			for (int k = 0; k < message->args; k++) {
				args[k].~Variant();
			}
		}

		message->~Message();

		LOCK_MUTEX;
		if (offset == page_bytes[i]) {
			i++;
			offset = 0;
		}
	}

	page_bytes[0] = 0;
	pages_used = 1;

	flushing = false;
	UNLOCK_MUTEX;
	return OK;
}

void CallQueue::clear() {
	LOCK_MUTEX;

	if (pages.is_empty()) {
		UNLOCK_MUTEX;
		return; // Nothing to clear.
	}

	for (uint32_t i = 0; i < pages_used; i++) {
		uint32_t offset = 0;
		while (offset < page_bytes[i]) {
			Page *page = pages[i];

			//lock on each iteration, so a call can re-add itself to the message queue

			Message *message = (Message *)&page->data[offset];

			uint32_t advance = sizeof(Message);
			if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION) {
				advance += sizeof(Variant) * message->args;
			}

			offset += advance;

			if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION) {
				Variant *args = (Variant *)(message + 1);
				for (int k = 0; k < message->args; k++) {
					args[k].~Variant();
				}
			}

			message->~Message();
		}
	}

	pages_used = 1;
	page_bytes[0] = 0;

	UNLOCK_MUTEX;
}

void CallQueue::statistics() {
	LOCK_MUTEX;
	HashMap<StringName, int> set_count;
	HashMap<int, int> notify_count;
	HashMap<Callable, int> call_count;
	int null_count = 0;

	for (uint32_t i = 0; i < pages_used; i++) {
		uint32_t offset = 0;
		while (offset < page_bytes[i]) {
			Page *page = pages[i];

			//lock on each iteration, so a call can re-add itself to the message queue

			Message *message = (Message *)&page->data[offset];

			uint32_t advance = sizeof(Message);
			if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION) {
				advance += sizeof(Variant) * message->args;
			}

			Object *target = message->callable.get_object();

			bool null_target = true;
			switch (message->type & FLAG_MASK) {
				case TYPE_CALL: {
					if (target || (message->type & FLAG_NULL_IS_OK)) {
						if (!call_count.has(message->callable)) {
							call_count[message->callable] = 0;
						}

						call_count[message->callable]++;
						null_target = false;
					}
				} break;
				case TYPE_NOTIFICATION: {
					if (target) {
						if (!notify_count.has(message->notification)) {
							notify_count[message->notification] = 0;
						}

						notify_count[message->notification]++;
						null_target = false;
					}
				} break;
				case TYPE_SET: {
					if (target) {
						StringName t = message->callable.get_method();
						if (!set_count.has(t)) {
							set_count[t] = 0;
						}

						set_count[t]++;
						null_target = false;
					}
				} break;
			}
			if (null_target) {
				// Object was deleted.
				fprintf(stdout, "Object was deleted while awaiting a callback.\n");

				null_count++;
			}

			offset += advance;

			if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION) {
				Variant *args = (Variant *)(message + 1);
				for (int k = 0; k < message->args; k++) {
					args[k].~Variant();
				}
			}

			message->~Message();
		}
	}

	fprintf(stdout, "TOTAL PAGES: %d (%d bytes).\n", pages_used, pages_used * PAGE_SIZE_BYTES);
	fprintf(stdout, "NULL count: %d.\n", null_count);

	for (const KeyValue<StringName, int> &E : set_count) {
		fprintf(stdout, "SET %s: %d.\n", String(E.key).utf8().get_data(), E.value);
	}

	for (const KeyValue<Callable, int> &E : call_count) {
		fprintf(stdout, "CALL %s: %d.\n", String(E.key).utf8().get_data(), E.value);
	}

	for (const KeyValue<int, int> &E : notify_count) {
		fprintf(stdout, "NOTIFY %d: %d.\n", E.key, E.value);
	}

	UNLOCK_MUTEX;
}

bool CallQueue::is_flushing() const {
	return flushing;
}

bool CallQueue::has_messages() const {
	if (pages_used == 0) {
		return false;
	}
	if (pages_used == 1 && page_bytes[0] == 0) {
		return false;
	}

	return true;
}

int CallQueue::get_max_buffer_usage() const {
	return pages.size() * PAGE_SIZE_BYTES;
}

CallQueue::CallQueue(Allocator *p_custom_allocator, uint32_t p_max_pages, const String &p_error_text) {
	if (p_custom_allocator) {
		allocator = p_custom_allocator;
		allocator_is_custom = true;
	} else {
		allocator = memnew(Allocator(16)); // 16 elements per allocator page, 64kb per allocator page. Anything small will do, though.
		allocator_is_custom = false;
	}
	max_pages = p_max_pages;
	error_text = p_error_text;
}

CallQueue::~CallQueue() {
	clear();
	// Let go of pages.
	for (uint32_t i = 0; i < pages.size(); i++) {
		allocator->free(pages[i]);
	}
	if (!allocator_is_custom) {
		memdelete(allocator);
	}
	DEV_ASSERT(!is_current_thread_override);
}

//////////////////////

CallQueue *MessageQueue::main_singleton = nullptr;
thread_local CallQueue *MessageQueue::thread_singleton = nullptr;

void MessageQueue::set_thread_singleton_override(CallQueue *p_thread_singleton) {
#ifdef DEV_ENABLED
	if (thread_singleton) {
		thread_singleton->is_current_thread_override = false;
	}
#endif
	thread_singleton = p_thread_singleton;
#ifdef DEV_ENABLED
	if (thread_singleton) {
		thread_singleton->is_current_thread_override = true;
	}
#endif
}

MessageQueue::MessageQueue() :
		CallQueue(nullptr,
				int(GLOBAL_DEF_RST(PropertyInfo(Variant::INT, "memory/limits/message_queue/max_size_mb", PROPERTY_HINT_RANGE, "1,512,1,or_greater"), 32)) * 1024 * 1024 / PAGE_SIZE_BYTES,
				"Message queue out of memory. Try increasing 'memory/limits/message_queue/max_size_mb' in project settings.") {
	ERR_FAIL_COND_MSG(main_singleton != nullptr, "A MessageQueue singleton already exists.");
	main_singleton = this;
}

MessageQueue::~MessageQueue() {
	main_singleton = nullptr;
}
