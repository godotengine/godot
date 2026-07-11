/**************************************************************************/
/*  blueprint_player.h                                                    */
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

#include "blueprint.h"

#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "scene/main/node.h"

class BlueprintPlayer : public Node {
	GDCLASS(BlueprintPlayer, Node);

	struct SignalHookup {
		ObjectID object_id;
		StringName signal;
		Callable callable;
	};

	Ref<Blueprint> blueprint;
	bool active = true;
	double current_delta = 0.0;

	HashMap<int, Dictionary> node_map;
	HashMap<int, Variant> exec_results; // Last "result" of call_method nodes, keyed by node id.
	Vector<SignalHookup> signal_hookups;
	HashSet<String> warned_actions;

	void _build_node_map();
	Node *_resolve_target(const Dictionary &p_node) const;
	int _find_exec_target(int p_from_node, int p_from_port) const;
	bool _has_input_value(const Dictionary &p_node, int p_input_port) const;
	Variant _eval_input(const Dictionary &p_node, int p_input_port, int p_depth);
	Variant _eval_output(const Dictionary &p_node, int p_output_port, int p_depth);
	void _run_chain(const Dictionary &p_event_node, int p_start_port);
	void _poll_input_events();
	void _connect_signal_events();
	void _disconnect_signal_events();
	Variant _signal_fired(const Variant **p_args, int p_argcount, Callable::CallError &r_error);

	static Variant _parse_param(const Variant &p_value);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_blueprint(const Ref<Blueprint> &p_blueprint);
	Ref<Blueprint> get_blueprint() const;
	void set_active(bool p_active);
	bool is_active() const;

	void run_event(const String &p_event_type);
};
