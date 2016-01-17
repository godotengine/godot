/*************************************************************************/
/*  sprite_frames_editor_plugin.h                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef SPRITE_FRAMES_EDITOR_PLUGIN_H
#define SPRITE_FRAMES_EDITOR_PLUGIN_H


#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/gui/tree.h"
#include "scene/2d/animated_sprite.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/dialogs.h"


class SpriteFramesEditor : public PanelContainer {

	OBJ_TYPE(SpriteFramesEditor, PanelContainer );

	Button *load;
	Button *_delete;
	Button *paste;
	Button *empty;
	Button *empty2;
	Button *move_up;
	Button *move_down;
	Tree *tree;
	bool loading_scene;
	int sel;


	EditorFileDialog *file;

	AcceptDialog *dialog;

	SpriteFrames *frames;


	void _load_pressed();
	void _load_scene_pressed();
	void _file_load_request(const DVector<String>& p_path);
	void _paste_pressed();
	void _empty_pressed();
	void _empty2_pressed();
	void _delete_pressed();
	void _delete_confirm_pressed();
	void _up_pressed();
	void _down_pressed();
	void _update_library();
	void _item_edited();

	UndoRedo *undo_redo;

protected:
	void _notification(int p_what);
	void _input_event(InputEvent p_event);
	static void _bind_methods();
public:

	void set_undo_redo(UndoRedo *p_undo_redo) {undo_redo=p_undo_redo; }

	void edit(SpriteFrames* p_frames);
	SpriteFramesEditor();
};

class SpriteFramesEditorPlugin : public EditorPlugin {

	OBJ_TYPE( SpriteFramesEditorPlugin, EditorPlugin );

	SpriteFramesEditor *frames_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "SpriteFrames"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	SpriteFramesEditorPlugin(EditorNode *p_node);
	~SpriteFramesEditorPlugin();

};

#endif // SPRITE_FRAMES_EDITOR_PLUGIN_H
