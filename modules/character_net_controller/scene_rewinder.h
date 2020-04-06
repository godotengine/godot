/*************************************************************************/
/*  scene_rewinder.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "scene/main/node.h"

#include "core/hash_map.h"

#ifndef SCENE_REWINDER_H
#define SCENE_REWINDER_H

struct VarData;

class SceneRewinder : public Node {
	GDCLASS(SceneRewinder, Node);

	HashMap<ObjectID, Vector<VarData>> variables;

public:
	static void _bind_methods();

	virtual void _notification(int p_what);

public:
	SceneRewinder();

	void register_variable(Node *p_object, StringName p_variable, StringName p_on_change_notify_to = StringName());
	void unregister_variable(Node *p_object, StringName p_variable);

	String get_changed_event_name(StringName p_variable);

	void track_variable_changes(Node *p_object, StringName p_variable, StringName p_method);
	void untrack_variable_changes(Node *p_object, StringName p_variable, StringName p_method);

private:
	void process();
};

struct VarData {
	StringName name;
	Variant old_val;

	VarData();
	VarData(StringName p_name);
	VarData(StringName p_name, Variant p_val);

	bool operator==(const VarData &p_other) const;
};

#endif
