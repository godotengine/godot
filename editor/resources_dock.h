/*************************************************************************/
/*  resources_dock.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef RESOURCES_DOCK_H
#define RESOURCES_DOCK_H

#include "create_dialog.h"
#include "editor_file_dialog.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/control.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/label.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/tree.h"

class EditorNode;

class ResourcesDock : public VBoxContainer {

	GDCLASS(ResourcesDock, VBoxContainer);

	enum {
		TOOL_NEW,
		TOOL_OPEN,
		TOOL_SAVE,
		TOOL_SAVE_AS,
		TOOL_MAKE_LOCAL,
		TOOL_COPY,
		TOOL_PASTE,
		TOOL_MAX
	};

	EditorNode *editor;

	Button *button_new;
	Button *button_open;
	Button *button_save;
	Button *button_tools;

	CreateDialog *create_dialog;

	AcceptDialog *accept;
	EditorFileDialog *file;
	Tree *resources;
	bool block_add;
	int current_action;

	void _file_action(const String &p_action);

	void _delete(Object *p_item, int p_column, int p_id);
	void _resource_selected();
	void _update_name(TreeItem *item);
	void _tool_selected(int p_tool);
	void _create();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void add_resource(const Ref<Resource> &p_resource);
	void remove_resource(const Ref<Resource> &p_resource);
	void save_resource(const String &p_path, const Ref<Resource> &p_resource);
	void save_resource_as(const Ref<Resource> &p_resource);

	void cleanup();

	ResourcesDock(EditorNode *p_editor);
};

#endif // RESOURCES_DOCK_H
