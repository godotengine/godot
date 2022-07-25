/*************************************************************************/
/*  physical_bone_plugin.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef PHYSICAL_BONE_PLUGIN_H
#define PHYSICAL_BONE_PLUGIN_H

#include "editor/editor_node.h"

class PhysicalBoneEditor : public Object {
	GDCLASS(PhysicalBoneEditor, Object);

	EditorNode *editor;
	HBoxContainer *spatial_editor_hb;
	ToolButton *button_transform_joint;

	PhysicalBone *selected;

protected:
	static void _bind_methods();

private:
	void _on_toggle_button_transform_joint(bool p_is_pressed);
	void _set_move_joint();

public:
	PhysicalBoneEditor(EditorNode *p_editor);
	~PhysicalBoneEditor();

	void set_selected(PhysicalBone *p_pb);

	void hide();
	void show();
};

class PhysicalBonePlugin : public EditorPlugin {
	GDCLASS(PhysicalBonePlugin, EditorPlugin);

	EditorNode *editor;
	PhysicalBone *selected;
	PhysicalBoneEditor physical_bone_editor;

public:
	virtual String get_name() const { return "PhysicalBone"; }
	virtual bool handles(Object *p_object) const { return p_object->is_class("PhysicalBone"); }
	virtual void make_visible(bool p_visible);
	virtual void edit(Object *p_node);

	PhysicalBonePlugin(EditorNode *p_editor);
};

#endif // PHYSICAL_BONE_PLUGIN_H
