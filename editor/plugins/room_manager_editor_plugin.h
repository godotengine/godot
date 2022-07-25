/*************************************************************************/
/*  room_manager_editor_plugin.h                                         */
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

#ifndef ROOM_MANAGER_EDITOR_PLUGIN_H
#define ROOM_MANAGER_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/3d/occluder.h"
#include "scene/3d/portal.h"
#include "scene/3d/room.h"
#include "scene/3d/room_manager.h"
#include "scene/resources/material.h"

class RoomManagerEditorPlugin : public EditorPlugin {
	GDCLASS(RoomManagerEditorPlugin, EditorPlugin);

	RoomManager *_room_manager;

	ToolButton *button_flip_portals;
	EditorNode *editor;

	void _flip_portals();

protected:
	static void _bind_methods();

public:
	virtual String get_name() const { return "RoomManager"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	RoomManagerEditorPlugin(EditorNode *p_node);
	~RoomManagerEditorPlugin();
};

///////////////////////

class RoomEditorPlugin : public EditorPlugin {
	GDCLASS(RoomEditorPlugin, EditorPlugin);

	Room *_room;
	ToolButton *button_generate;
	EditorNode *editor;
	UndoRedo *undo_redo;

	void _generate_points();

protected:
	static void _bind_methods();

public:
	virtual String get_name() const { return "Room"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	RoomEditorPlugin(EditorNode *p_node);
	~RoomEditorPlugin();
};

///////////////////////

class PortalEditorPlugin : public EditorPlugin {
	GDCLASS(PortalEditorPlugin, EditorPlugin);

	Portal *_portal;
	ToolButton *button_flip;
	EditorNode *editor;

	void _flip_portal();

protected:
	static void _bind_methods();

public:
	virtual String get_name() const { return "Portal"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	PortalEditorPlugin(EditorNode *p_node);
	~PortalEditorPlugin();
};

///////////////////////

class OccluderEditorPlugin : public EditorPlugin {
	GDCLASS(OccluderEditorPlugin, EditorPlugin);

	Occluder *_occluder;
	ToolButton *button_center;
	EditorNode *editor;
	UndoRedo *undo_redo;

	void _center();

protected:
	static void _bind_methods();

public:
	virtual String get_name() const { return "Occluder"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	OccluderEditorPlugin(EditorNode *p_node);
	~OccluderEditorPlugin();
};

#endif // ROOM_MANAGER_EDITOR_PLUGIN_H
