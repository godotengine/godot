/*************************************************************************/
/*  animation_motion_match_editor.h                                      */
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

#ifndef ANIMATION_MOTION_MATCH_EDITOR_H
#define ANIMATION_MOTION_MATCH_EDITOR_H

#ifdef TOOLS_ENABLED

#include "frame_model.h"

#include "core/object/reference.h"
#include "editor/plugins/animation_tree_editor_plugin.h"
#include "modules/motionmatch/animation_node_motion_match.h"

class AnimationNodeMotionMatchEditor : public AnimationTreeNodeEditorPlugin {
	GDCLASS(AnimationNodeMotionMatchEditor, AnimationTreeNodeEditorPlugin)

	Ref<AnimationNodeMotionMatch> motion_match;

	AcceptDialog *match_tracks_dialog;
	Tree *match_tracks;

	Button *edit_match_tracks;
	Button *update_tracks;
	Button *clear_tree;

	SpinBox *snap_x;

	Skeleton3D *skeleton;

	bool updating;
	Vector<frame_model *> *keys = new Vector<frame_model *>();

	void _match_tracks_edited();

	void _edit_match_tracks();
	void _update_match_tracks();
	void _update_tracks();
	void _clear_tree();
	int max_key_track;
	void _update_vel();

protected:
	static void _bind_methods();

public:
	virtual bool can_edit(const Ref<AnimationNode> &p_node) override;
	virtual void edit(const Ref<AnimationNode> &p_node) override;

	int fill_tracks(AnimationPlayer *player, Animation *anim, NodePath &root);

	AnimationNodeMotionMatchEditor();
};
#endif // ANIMATION_MOTION_MATCH_EDITOR_H
#endif
