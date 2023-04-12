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

void CallQueue::_add_page() {
	if (pages_used == page_messages.size()) {
		pages.push_back(allocator->alloc());
		page_messages.push_back(0);
	}
	page_messages[pages_used] = 0;
	pages_used++;
	page_offset = 0;
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
	mutex.lock();
	uint32_t room_needed = sizeof(Message) + sizeof(Variant) * p_argcount;

	ERR_FAIL_COND_V_MSG(room_needed > uint32_t(PAGE_SIZE_BYTES), ERR_INVALID_PARAMETER, "Message is too large to fit on a page (" + itos(PAGE_SIZE_BYTES) + " bytes), consider passing less arguments.");

	_ensure_first_page();

	if ((page_offset + room_needed) > uint32_t(PAGE_SIZE_BYTES)) {
		if (room_needed > uint32_t(PAGE_SIZE_BYTES) || pages_used == max_pages) {
			ERR_PRINT("Failed method: " + p_callable + ". Message queue out of memory. " + error_text);
			statistics();
			mutex.unlock();
			return ERR_OUT_OF_MEMORY;
		}
		_add_page();
	}

	Page *page = pages[pages_used - 1];

	uint8_t *buffer_end = &page->data[page_offset];

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

	page_messages[pages_used - 1]++;
	page_offset += room_needed;

	mutex.unlock();

	return OK;
}

Error CallQueue::push_set(ObjectID p_id, const StringName &p_prop, const Variant &p_value) {
	mutex.lock();
	uint32_t room_needed = sizeof(Message) + sizeof(Variant);

	_ensure_first_page();

	if ((page_offset + room_needed) > uint32_t(PAGE_SIZE_BYTES)) {
		if (pages_used == max_pages) {
			String type;
			if (ObjectDB::get_instance(p_id)) {
				type = ObjectDB::get_instance(p_id)->get_class();
			}
			ERR_PRINT("Failed set: " + type + ":" + p_prop + " target ID: " + itos(p_id) + ". Message queue out of memory. " + error_text);
			statistics();

			mutex.unlock();
			return ERR_OUT_OF_MEMORY;
		}
		_add_page();
	}

	Page *page = pages[pages_used - 1];
	uint8_t *buffer_end = &page->data[page_offset];

	Message *msg = memnew_placement(buffer_end, Message);
	msg->args = 1;
	msg->callable = Callable(p_id, p_prop);
	msg->type = TYPE_SET;

	buffer_end += sizeof(Message);

	Variant *v = memnew_placement(buffer_end, Variant);
	*v = p_value;

	page_messages[pages_used - 1]++;
	page_offset += room_needed;
	mutex.unlock();

	return OK;
}

Error CallQueue::push_notification(ObjectID p_id, int p_notification) {
	ERR_FAIL_COND_V(p_notification < 0, ERR_INVALID_PARAMETER);
	mutex.lock();
	uint32_t room_needed = sizeof(Message);

	_ensure_first_page();

	if ((page_offset + room_needed) > uint32_t(PAGE_SIZE_BYTES)) {
		if (pages_used == max_pages) {
			ERR_PRINT("Failed notification: " + itos(p_notification) + " target ID: " + itos(p_id) + ". Message queue out of memory. " + error_text);
			statistics();
			mutex.unlock();
			return ERR_OUT_OF_MEMORY;
		}
		_add_page();
	}

	Page *page = pages[pages_used - 1];
	uint8_t *buffer_end = &page->data[page_offset];

	Message *msg = memnew_placement(buffer_end, Message);

	msg->type = TYPE_NOTIFICATION;
	msg->callable = Callable(p_id, CoreStringNames::get_singleton()->notification); //name is meaningless but callable needs it
	//msg->target;
	msg->notification = p_notification;

	page_messages[pages_used - 1]++;
	page_offset += room_needed;
	mutex.unlock();

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
	mutex.lock();

	if (pages.size() == 0) {
		// Never allocated
		mutex.unlock();
		return OK; // Do nothing.
	}

	if (flushing) {
		mutex.unlock();
		return ERR_BUSY;
	}

	flushing = true;

	uint32_t i = 0;
	uint32_t j = 0;
	uint32_t offset = 0;

	while (i < pages_used && j < page_messages[i]) {
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

		mutex.unlock();

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

		mutex.lock();
		j++;
		if (j == page_messages[i]) {
			j = 0;
			i++;
			offset = 0;
		}
	}

	page_messages[0] = 0;
	page_offset = 0;
	pages_used = 1;

	flushing = false;
	mutex.unlock();
	return OK;
}

void CallQueue::clear() {
	mutex.lock();

	if (pages.size() == 0) {
		mutex.unlock();
		return; // Nothing to clear.
	}

	for (uint32_t i = 0; i < pages_used; i++) {
		uint32_t offset = 0;
		for (uint32_t j = 0; j < page_messages[i]; j++) {
			Page *page = pages[i];

			//lock on each iteration, so a call can re-add itself to the message queue

			Message *message = (Message *)&page->data[offset];

			uint32_t advance = sizeof(Message);
			if ((message->type & FLAG_MASK) != TYPE_NOTIFICATION) {
				advance += sizeof(Variant) * message->args;
			}

			//pre-advance so this function is reentrant
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
	page_offset = 0;
	page_messages[0] = 0;

	mutex.unlock();
}

void CallQueue::statistics() {
	mutex.lock();
	HashMap<StringName, int> set_count;
	HashMap<int, int> notify_count;
	HashMap<Callable, int> call_count;
	int null_count = 0;

	for (uint32_t i = 0; i < pages_used; i++) {
		uint32_t offset = 0;
		for (uint32_t j = 0; j < page_messages[i]; j++) {
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
				//object was deleted
				print_line("Object was deleted while awaiting a callback");

				null_count++;
			}

			//pre-advance so this function is reentrant
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

	print_line("TOTAL PAGES: " + itos(pages_used) + " (" + itos(pages_used * PAGE_SIZE_BYTES) + " bytes).");
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

	mutex.unlock();
}

bool CallQueue::is_flushing() const {
	return flushing;
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
}

//////////////////////

MessageQueue *MessageQueue::singleton = nullptr;

MessageQueue::MessageQueue() :
		CallQueue(nullptr,
				int(GLOBAL_DEF_RST(PropertyInfo(Variant::INT, "memory/limits/message_queue/max_size_mb", PROPERTY_HINT_RANGE, "1,512,1,or_greater"), 32)) * 1024 * 1024 / PAGE_SIZE_BYTES,
				"Message queue out of memory. Try increasing 'memory/limits/message_queue/max_size_mb' in project settings.") {
	ERR_FAIL_COND_MSG(singleton != nullptr, "A MessageQueue singleton already exists.");
	singleton = this;
}

MessageQueue::~MessageQueue() {
	singleton = nullptr;
}
