/*************************************************************************/
/*  particles_2d_editor_plugin.h                                         */
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
#ifndef PARTICLES_2D_EDITOR_PLUGIN_H
#define PARTICLES_2D_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/2d/collision_polygon_2d.h"

#include "scene/2d/particles_2d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/file_dialog.h"

class Particles2DEditorPlugin : public EditorPlugin {

	GDCLASS(Particles2DEditorPlugin, EditorPlugin);

	enum {

		MENU_LOAD_EMISSION_MASK,
		MENU_CLEAR_EMISSION_MASK
	};

	Particles2D *particles;

	EditorFileDialog *file;
	EditorNode *editor;

	HBoxContainer *toolbar;
	MenuButton *menu;

	SpinBox *epoints;

	UndoRedo *undo_redo;
	void _file_selected(const String &p_file);
	void _menu_callback(int p_idx);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual String get_name() const { return "Particles2D"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	Particles2DEditorPlugin(EditorNode *p_node);
	~Particles2DEditorPlugin();
};

#endif // PARTICLES_2D_EDITOR_PLUGIN_H
