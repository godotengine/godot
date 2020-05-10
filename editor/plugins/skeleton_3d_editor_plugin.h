/*************************************************************************/
/*  skeleton_3d_editor_plugin.h                                          */
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

#ifndef SKELETON_3D_EDITOR_PLUGIN_H
#define SKELETON_3D_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/3d/skeleton_3d.h"

class PhysicalBone3D;
class Joint3D;

class Skeleton3DEditor : public Node {
	GDCLASS(Skeleton3DEditor, Node);

	enum Menu {
		MENU_OPTION_CREATE_PHYSICAL_SKELETON
	};

	struct BoneInfo {
		PhysicalBone3D *physical_bone;
		Transform relative_rest; // Relative to skeleton node
		BoneInfo() :
				physical_bone(nullptr) {}
	};

	Skeleton3D *skeleton;

	MenuButton *options;

	void _on_click_option(int p_option);

	friend class Skeleton3DEditorPlugin;

protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();

	void create_physical_skeleton();
	PhysicalBone3D *create_physical_bone(int bone_id, int bone_child_id, const Vector<BoneInfo> &bones_infos);

public:
	void edit(Skeleton3D *p_node);

	Skeleton3DEditor();
	~Skeleton3DEditor();
};

class Skeleton3DEditorPlugin : public EditorPlugin {

	GDCLASS(Skeleton3DEditorPlugin, EditorPlugin);

	EditorNode *editor;
	Skeleton3DEditor *skeleton_editor;

public:
	virtual String get_name() const { return "Skeleton3D"; }
	virtual bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	Skeleton3DEditorPlugin(EditorNode *p_node);
	~Skeleton3DEditorPlugin();
};

#endif // SKELETON_3D_EDITOR_PLUGIN_H
