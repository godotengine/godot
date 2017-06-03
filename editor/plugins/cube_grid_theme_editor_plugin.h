/*************************************************************************/
/*  cube_grid_theme_editor_plugin.h                                      */
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
#ifndef CUBE_GRID_THEME_EDITOR_PLUGIN_H
#define CUBE_GRID_THEME_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "scene/resources/mesh_library.h"

#if 0
class MeshLibraryEditor : public Control {

	GDCLASS( MeshLibraryEditor, Control );

	Ref<MeshLibrary> theme;

	EditorNode *editor;
	MenuButton *menu;
	ConfirmationDialog *cd;
	EditorFileDialog *file;
	int to_erase;

	enum {

		MENU_OPTION_ADD_ITEM,
		MENU_OPTION_REMOVE_ITEM,
		MENU_OPTION_UPDATE_FROM_SCENE,
		MENU_OPTION_IMPORT_FROM_SCENE
	};

	int option;
	void _import_scene_cbk(const String& p_str);
	void _menu_cbk(int p_option);
	void _menu_confirm();

	static void _import_scene(Node *p_scene, Ref<MeshLibrary> p_library, bool p_merge);

protected:
	static void _bind_methods();
public:

	void edit(const Ref<MeshLibrary>& p_theme);
	static Error update_library_file(Node *p_base_scene, Ref<MeshLibrary> ml,bool p_merge=true);

	MeshLibraryEditor(EditorNode *p_editor);
};



class MeshLibraryEditorPlugin : public EditorPlugin {

	GDCLASS( MeshLibraryEditorPlugin, EditorPlugin );

	MeshLibraryEditor *theme_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "MeshLibrary"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	MeshLibraryEditorPlugin(EditorNode *p_node);

};

#endif // CUBE_GRID_THEME_EDITOR_PLUGIN_H
#endif
