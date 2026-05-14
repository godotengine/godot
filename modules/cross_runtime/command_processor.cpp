/**************************************************************************/
/*  command_processor.cpp                                                 */
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

#include "core/object/callable_mp.h"
#include "core/object/message_queue.h"
#include "scene/main/scene_tree.h"

#include <cstdio>

#ifdef WEB_ENABLED
// Forward declaration
void process_api_commands();

class CommandProcessor : public Object {
	GDCLASS(CommandProcessor, Object);

	static CommandProcessor *singleton;

public:
	static CommandProcessor *get_singleton() { return singleton; }

	// This is the actual function that the engine will call every frame
	void _frame_update() {
		process_api_commands();
	}

	CommandProcessor() {
		singleton = this;
	}

	~CommandProcessor() {
		singleton = nullptr;
	}
};

// Initialize static member
CommandProcessor *CommandProcessor::singleton = nullptr;

// 2. The Registration Logic
void register_command_processing() {
	SceneTree *tree = SceneTree::get_singleton();

	// If the tree isn't ready, defer
	if (!tree) {
		MessageQueue::get_singleton()->push_callable(callable_mp_static(&register_command_processing));
		return;
	}

	// Ensure the Processor object exists
	if (!CommandProcessor::get_singleton()) {
		memnew(CommandProcessor);
	}

	// Connect the signal to the class method
	Callable update_callable = callable_mp(CommandProcessor::get_singleton(), &CommandProcessor::_frame_update);

	if (!tree->is_connected("process_frame", update_callable)) {
		Error err = tree->connect("process_frame", update_callable);

		if (err == OK) {
			printf("Godot ->  SUCCESS: Class-based frame callback connected!\n");
		} else {
			printf("Godot ->  ERROR: Connection failed with code: %d\n", (int)err);
		}
	} else {
		printf("Godot ->  Notice: Already connected.\n");
	}
}

// Cleanup
void unregister_command_processing() {
	CommandProcessor *cp = CommandProcessor::get_singleton();
	if (cp) {
		SceneTree *tree = SceneTree::get_singleton();
		if (tree) {
			tree->disconnect("process_frame", callable_mp(cp, &CommandProcessor::_frame_update));
		}
		memdelete(cp);
	}
}

#else

void register_command_processing() {}

#endif
