/*************************************************************************/
/*  sample_library_editor_plugin.h                                       */
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
#ifndef SAMPLE_LIBRARY_EDITOR_PLUGIN_H
#define SAMPLE_LIBRARY_EDITOR_PLUGIN_H

#if 0
#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/audio/sample_player.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/tree.h"
#include "scene/resources/sample.h"


class SampleLibraryEditor : public Panel {

	GDCLASS(SampleLibraryEditor, Panel );



	SamplePlayer *player;
	Ref<SampleLibrary> sample_library;
	Button *load;
	Tree *tree;
	bool is_playing;
	Object *last_sample_playing;

	EditorFileDialog *file;

	ConfirmationDialog *dialog;


	void _load_pressed();
	void _file_load_request(const PoolVector<String>& p_path);
	void _delete_pressed();
	void _update_library();
	void _item_edited();

	UndoRedo *undo_redo;

	void _button_pressed(Object *p_item,int p_column, int p_id);

	Variant get_drag_data_fw(const Point2& p_point,Control* p_from);
	bool can_drop_data_fw(const Point2& p_point,const Variant& p_data,Control* p_from) const;
	void drop_data_fw(const Point2& p_point,const Variant& p_data,Control* p_from);

protected:
	void _notification(int p_what);
	void _gui_input(InputEvent p_event);
	static void _bind_methods();
public:

	void set_undo_redo(UndoRedo *p_undo_redo) {undo_redo=p_undo_redo; }
	void edit(Ref<SampleLibrary> p_sample);
	SampleLibraryEditor();
};

class SampleLibraryEditorPlugin : public EditorPlugin {

	GDCLASS( SampleLibraryEditorPlugin, EditorPlugin );

	SampleLibraryEditor *sample_library_editor;
	EditorNode *editor;
	Button *button;

public:

	virtual String get_name() const { return "SampleLibrary"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	SampleLibraryEditorPlugin(EditorNode *p_node);
	~SampleLibraryEditorPlugin();

};

#endif
#endif // SAMPLE_LIBRARY_EDITOR_PLUGIN_H
