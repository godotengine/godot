/**************************************************************************/
/*  live_runtime_plugin.h                                                 */
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

#include "editor/debugger/editor_debugger_plugin.h"
#include "editor/plugins/editor_plugin.h"

class Button;
class Node2D;
class Node3D;
class PackedScene;

// Editor side of the Live Runtime view (see SPLIT_WORKSPACE.md).
//
// In Live Runtime mode, the 2D and 3D editor viewports show the running game's
// state: the game (RuntimeStateObserver) streams changed transforms and visibility,
// and this plugin applies them to the rendering server instances of the edited
// scene's nodes, without changing the nodes themselves. Nodes that only exist in
// the game (spawned at runtime) are shown with "ghost" copies that are not part of
// the edited scene. The editor camera stays independent. On Stop, or when the mode
// is turned off, every node is restored from its own properties.
class LiveRuntimeDebugger : public EditorDebuggerPlugin {
	GDCLASS(LiveRuntimeDebugger, EditorDebuggerPlugin);

public:
	enum Mode {
		MODE_STANDARD,
		MODE_LIVE,
	};

private:
	enum Kind {
		KIND_3D,
		KIND_2D,
	};

	struct Entry {
		ObjectID node;
		Kind kind = KIND_3D;
		bool ghost = false;
		String ghost_root; // Key in ghost_roots, for ghosts.
		bool visible = true;
		Transform3D xform_3d;
		Transform2D xform_2d;
	};

	Mode mode = MODE_STANDARD;
	bool observing = false;
	bool observe_3d = false;
	bool observe_2d = false;
	int observed_session = -1;

	HashMap<uint32_t, Entry> entries;
	HashMap<ObjectID, uint32_t> node_entries; // Editor node to entry, for 2D parent lookups.
	HashMap<String, ObjectID> ghost_roots; // Runtime path of the ghost's root to the ghost.
	HashMap<String, int> ghost_refs; // Live entries per ghost; a ghost is freed with its last entry.
	HashMap<String, Ref<PackedScene>> ghost_scenes;
	ObjectID ghost_parent_3d;
	ObjectID ghost_parent_2d;

	int unresolved_count = 0;
	String status;

	void _session_started(int p_session);
	void _session_stopped(int p_session);

	void _msg_state(const Array &p_data);
	Node *_resolve(const Array &p_added, bool &r_ghost, String &r_ghost_root);
	void _release_ghost(const String &p_root);
	Node *_get_ghost_parent(bool p_2d);
	Node *_make_ghost(const String &p_path, Node *p_node);
	void _hide_ghost_visuals(Node *p_node);
	void _apply_entry_visibility(Entry &p_entry);
	void _apply_entry_2d(const Entry &p_entry);
	void _restore_entry(const Entry &p_entry);
	void _clear();
	void _set_status(const String &p_status);

protected:
	static void _bind_methods();

public:
	virtual bool capture(const String &p_message, const Array &p_data, int p_session) override;
	virtual bool has_capture(const String &p_capture) const override;
	virtual void setup_session(int p_session_id) override;

	void set_mode(Mode p_mode);
	Mode get_mode() const { return mode; }
	String get_status() const { return status; }

	// Sends the setup to the game when the mode, the visible editors or the edited scene change.
	void update_observing();

	~LiveRuntimeDebugger();
};

class LiveRuntimePlugin : public EditorPlugin {
	GDCLASS(LiveRuntimePlugin, EditorPlugin);

	Ref<LiveRuntimeDebugger> debugger;
	Button *button_3d = nullptr;
	Button *button_2d = nullptr;

	Button *_make_button();
	void _button_toggled(bool p_pressed);
	void _status_changed();
	void _queue_update();
	bool update_queued = false;
	void _update_observing();

protected:
	void _notification(int p_what);

public:
	virtual String get_plugin_name() const override { return "LiveRuntime"; }

	LiveRuntimePlugin();
};
