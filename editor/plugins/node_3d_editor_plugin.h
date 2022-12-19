/*************************************************************************/
/*  node_3d_editor_plugin.h                                              */
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

#ifndef NODE_3D_EDITOR_PLUGIN_H
#define NODE_3D_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"
#include "node_3d_editor.h"

class Node3DEditorPlugin : public EditorPlugin {
	GDCLASS(Node3DEditorPlugin, EditorPlugin);

	Node3DEditor *spatial_editor = nullptr;

public:
	Node3DEditor *get_spatial_editor() { return spatial_editor; }
	virtual String get_name() const override { return "3D"; }
	bool has_main_screen() const override { return true; }
	virtual void make_visible(bool p_visible) override;
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;

	virtual Dictionary get_state() const override;
	virtual void set_state(const Dictionary &p_state) override;
	virtual void clear() override { spatial_editor->clear(); }

	virtual void edited_scene_changed() override;

	Node3DEditorPlugin();
	~Node3DEditorPlugin();
};

#endif // NODE_3D_EDITOR_PLUGIN_H
