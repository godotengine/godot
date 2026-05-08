#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/object/message_queue.h"
#include "core/variant/callable_bind.h"
#include "scene/main/scene_tree.h"

#include <cstdio>

// Forward declaration
void process_api_commands();

class CommandProcessor : public Object {
	GDCLASS(CommandProcessor, Object);

	static CommandProcessor *singleton;

public:
	static CommandProcessor *get_singleton() { return singleton; }

	// This is the actual function the engine will call every frame
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
