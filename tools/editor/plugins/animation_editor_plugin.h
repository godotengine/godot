/*************************************************************************/
/*  animation_editor_plugin.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef ANIMATION_EDITOR_PLUGIN_H
#define ANIMATION_EDITOR_PLUGIN_H

#include "scene/resources/animation.h"
#include "scene/gui/texture_frame.h"
#include "scene/gui/option_button.h"
#include "tools/editor/editor_node.h"
#include "tools/editor/property_editor.h"
#include "undo_redo.h"
class AnimationEditor_TrackEditor;

class AnimationEditor : public Control {

	OBJ_TYPE( AnimationEditor, Control );

	Panel *panel;
	Ref<Animation> animation;

	Button *add_track;
	Button *remove_track;
	Button *move_up_track;
	Button *move_down_track;

	Button *add_key;
	Button *remove_key;

	LineEdit *key_time;
	OptionButton *track_type;
	OptionButton *key_type;
	TextureFrame *time_icon;

	Tree *tracks;
	PropertyEditor *key_editor;
	AnimationEditor_TrackEditor *track_editor;
	int selected_track;

	void _track_selected();
	void _track_added();
	void _track_removed();
	void _track_moved_up();
	void _track_moved_down();
	void _track_path_changed();

	void _key_added();
	void _key_removed();


	void _update_track_keys();
	void update_anim();

	UndoRedo *undo_redo;

	void _internal_set_selected_track(int p_which,const Ref<Animation>& p_anim);
	void _internal_check_update(Ref<Animation> p_anim);

friend class AnimationEditor_TrackEditor;
	void _internal_set_key(int p_track, float p_time, float p_transition,const Variant& p_value);
	void _internal_set_interpolation_type(int p_track,Animation::InterpolationType p_type);

protected:
	void _notification(int p_what);
	static void _bind_methods();
public:

	void set_undo_redo(UndoRedo *p_undo_redo) { undo_redo=p_undo_redo; }
	void edit(const Ref<Animation>& p_animation);

	AnimationEditor();
	~AnimationEditor();
};



class AnimationEditorPlugin : public EditorPlugin {

	OBJ_TYPE( AnimationEditorPlugin, EditorPlugin );

	AnimationEditor *animation_editor;
	EditorNode *editor;

public:


	virtual String get_name() const { return "Animation"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	AnimationEditorPlugin(EditorNode *p_node);

};


#endif
