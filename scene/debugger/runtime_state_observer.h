/**************************************************************************/
/*  runtime_state_observer.h                                              */
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

#pragma once

#ifdef DEBUG_ENABLED

#include "core/math/transform_2d.h"
#include "core/math/transform_3d.h"
#include "core/object/object.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"

class Node;

// Game side of the editor's Live Runtime view.
//
// While the editor asks for it, the running game sends the global transform and
// visibility of every VisualInstance3D and Node2D below the observed root, at a
// limited rate and only for the nodes that changed. New nodes are announced once
// with their path and with what the editor needs to show them when they are not
// part of the edited scene (the scene file of their instance, or their mesh or
// texture). Nothing is sent while the editor views are hidden.
//
// Messages (capture "runtime_observer"):
// - Editor to game: "runtime_observer:setup" [enabled, rate_hz, scene_path, observe_3d, observe_2d, max_nodes].
// - Game to editor: "runtime_observer:state" [reset, root_path, added, removed, ids_3d, xforms_3d, ids_2d, xforms_2d, shown, hidden].
class RuntimeStateObserver : public Object {
	GDCLASS(RuntimeStateObserver, Object);

public:
	enum Kind {
		KIND_3D,
		KIND_2D,
	};

	// Fields of one entry of the "added" array.
	enum AddedField {
		ADDED_ID,
		ADDED_KIND,
		ADDED_PATH,
		ADDED_CLASS,
		ADDED_OWNER_PATH,
		ADDED_OWNER_SCENE,
		ADDED_RESOURCE,
		ADDED_MAX,
	};

private:
	struct Tracked {
		uint32_t id = 0;
		Kind kind = KIND_3D;
		Transform3D xform_3d;
		Transform2D xform_2d;
		bool visible = true;
		uint64_t seen_tick = 0;
	};

	inline static RuntimeStateObserver *singleton = nullptr;

	bool enabled = false;
	bool processing = false;
	bool observe_3d = true;
	bool observe_2d = true;
	uint64_t interval_usec = 33333;
	uint64_t last_send_usec = 0;
	uint64_t tick = 0;
	int max_nodes = 20000;

	// Path of the scene edited in the editor. The observed root is the runtime node instanced from it,
	// or the current scene when no node matches.
	String scene_path;
	ObjectID root_id;
	bool reset_pending = true;
	uint32_t next_id = 1;
	HashMap<ObjectID, Tracked> tracked;
	LocalVector<Node *> walk_stack;

	static Error _capture(void *p_user, const String &p_msg, const Array &p_args, bool &r_captured);

	void _setup(const Array &p_args);
	void _set_processing(bool p_processing);
	void _reset();
	Node *_find_root() const;
	void _process_frame();

public:
	static RuntimeStateObserver *get_singleton() { return singleton; }

	RuntimeStateObserver();
	~RuntimeStateObserver();
};

#endif // DEBUG_ENABLED
