/**************************************************************************/
/*  visual_server_callbacks.h                                             */
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

#ifndef VISUAL_SERVER_CALLBACKS_H
#define VISUAL_SERVER_CALLBACKS_H

#include "core/local_vector.h"
#include "core/object_id.h"
#include "core/os/mutex.h"

class VisualServerCallbacks {
public:
	enum CallbackType {
		CALLBACK_NOTIFICATION_ENTER_GAMEPLAY,
		CALLBACK_NOTIFICATION_EXIT_GAMEPLAY,
		CALLBACK_SIGNAL_ENTER_GAMEPLAY,
		CALLBACK_SIGNAL_EXIT_GAMEPLAY,
	};

	struct Message {
		CallbackType type;
		ObjectID object_id;
	};

	void lock();
	void unlock();
	void flush();

	void push_message(const Message &p_message) { messages.push_back(p_message); }
	int32_t get_num_messages() const { return messages.size(); }
	const Message &get_message(int p_index) const { return messages[p_index]; }
	void clear() { messages.clear(); }

private:
	LocalVector<Message, int32_t> messages;
	Mutex mutex;
};

#endif // VISUAL_SERVER_CALLBACKS_H
