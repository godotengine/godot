/**************************************************************************/
/*  animation_track_filter_editor_plugin.cpp                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "animation_track_filter_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_resource_picker.h"
#include "editor/editor_scale.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/animation_tree_editor_plugin.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tree.h"
#include "scene/resources/animation_track_filter.h"

// AnimationTrackFilterEditDialog::TrackItem
void AnimationTrackFilterEditDialog::TrackItem::_amount_changed(double p_value) {
	ERR_FAIL_COND(track_path.is_empty());
	emit_signal(SNAME("amount_changed"), get_track_path(), p_value);
}

void AnimationTrackFilterEditDialog::TrackItem::_bind_methods() {
	ADD_SIGNAL(MethodInfo("amount_changed", PropertyInfo(Variant::NODE_PATH, "track_path"), PropertyInfo(Variant::FLOAT, "amount")));
}

void AnimationTrackFilterEditDialog::TrackItem::set_editable(bool p_editable) {
	amount->set_editable(p_editable);
	amount->set_focus_mode(p_editable ? FOCUS_ALL : FOCUS_NONE);
	if (p_editable) {
		set_as_uneditable_track(false);
	}
}

void AnimationTrackFilterEditDialog::TrackItem::set_as_uneditable_track(bool p_editable) {
	amount->set_visible(!p_editable);
}

void AnimationTrackFilterEditDialog::TrackItem::set_icon(const Ref<Texture2D> &p_icon) {
	icon->set_texture(p_icon);
}

void AnimationTrackFilterEditDialog::TrackItem::set_text(const String &p_text) {
	text->set_text(p_text);
}

void AnimationTrackFilterEditDialog::TrackItem::set_amount(float p_amount) {
	amount->set_value_no_signal(p_amount);
}

float AnimationTrackFilterEditDialog::TrackItem::get_amount() const {
	return amount->get_value();
}

void AnimationTrackFilterEditDialog::TrackItem::set_track_path(const NodePath &p_track_path) {
	track_path = p_track_path;
}

NodePath AnimationTrackFilterEditDialog::TrackItem::get_track_path() const {
	return track_path;
}

AnimationTrackFilterEditDialog::TrackItem::TrackItem() {
	amount = memnew(SpinBox);
	add_child(amount);
	amount->set_step(0.01);
	amount->set_min(0.0);
	amount->set_max(1.0);
	amount->set_v_size_flags(SIZE_SHRINK_CENTER);
	amount->set_allow_lesser(false);
	amount->set_allow_greater(false);
	amount->set_focus_mode(FocusMode::FOCUS_NONE);
	amount->connect(SNAME("value_changed"), callable_mp(this, &TrackItem::_amount_changed));

	icon = memnew(TextureRect);
	add_child(icon);
	icon->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);

	text = memnew(Label);
	add_child(text);
}

// AnimationTrackFilterEditDialog
void AnimationTrackFilterEditDialog::_tree_draw() {
	hide_invisiable_track_items();

	RID ci = tree_focus_rect->get_canvas_item();

	if (filter_tree->has_focus()) {
		RS::get_singleton()->canvas_item_add_clip_ignore(ci, true);
		focus_style->draw(ci, filter_tree->get_rect());
		RS::get_singleton()->canvas_item_add_clip_ignore(ci, false);
	} else {
		RS::get_singleton()->canvas_item_clear(ci);
	}
}

void AnimationTrackFilterEditDialog::_tree_item_draw(TreeItem *p_item, const Rect2 &p_rect) const {
	ERR_FAIL_COND(!track_items.has(p_item));
	TrackItem *track_item = track_items[p_item];
	track_item->set_rect(p_rect);
	track_item->show();
}

void AnimationTrackFilterEditDialog::_track_item_amount_changed(const NodePath &p_track_path, double p_amount) {
	ERR_FAIL_NULL(filter);

	updating = true;

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Edit Filter"));

	undo_redo->add_do_method(filter.ptr(), "set_track", p_track_path, p_amount);
	undo_redo->add_undo_method(filter.ptr(), "set_track", p_track_path, filter->get_track_amount(p_track_path));

	undo_redo->commit_action();

	updating = false;
}

bool is_item_collapsed_recursively(TreeItem *p_item) {
	if (p_item->is_collapsed()) {
		return true;
	}

	if (p_item->get_parent()) {
		return is_item_collapsed_recursively(p_item->get_parent());
	}
	return false;
}

void AnimationTrackFilterEditDialog::hide_invisiable_track_items() {
	Rect2 tree_rect = Rect2(filter_tree->get_scroll(), filter_tree->get_size());
	for (const KeyValue<TreeItem *, TrackItem *> &E : track_items) {
		Rect2 item_rect = filter_tree->get_item_rect(E.key);
		if (!tree_rect.intersects(item_rect) || is_item_collapsed_recursively(E.key->get_parent())) {
			E.value->hide();
		}
	}
}

AnimationTrackFilterEditDialog::TrackItem *AnimationTrackFilterEditDialog::get_or_create_track_items_and_setup_tree_item(TreeItem *p_item) {
	if (track_items.has(p_item)) {
		return track_items[p_item];
	}
	p_item->set_cell_mode(0, TreeItem::CELL_MODE_CUSTOM);
	p_item->set_custom_draw(0, this, SNAME("_tree_item_draw"));

	TrackItem *ret = memnew(TrackItem);
	track_items.insert(p_item, ret);
	filter_tree->add_child(ret);
	ret->connect(SNAME("amount_changed"), callable_mp(this, &AnimationTrackFilterEditDialog::_track_item_amount_changed));
	return ret;
}

void AnimationTrackFilterEditDialog::_notification(int p_what) {
	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (!is_visible()) {
			anim_player = nullptr;
			filter.unref();
		}
	} else if (p_what == NOTIFICATION_THEME_CHANGED || p_what == NOTIFICATION_READY) {
		focus_style = filter_tree->get_theme_stylebox("focus");
	}
}

void AnimationTrackFilterEditDialog::set_read_only(bool p_read_only) {
	read_only = p_read_only;

	if (is_read_only()) {
		set_title(TTR("Inspect Filtered Tracks:"));
	} else {
		set_title(TTR("Edit Filtered Tracks:"));
	}
}

bool AnimationTrackFilterEditDialog::is_read_only() const {
	return read_only;
}

bool AnimationTrackFilterEditDialog::update_filters(class AnimationMixer *p_player, const Ref<AnimationTrackFilter> &p_filter) {
	anim_player = p_player;
	filter = p_filter;

	for (KeyValue<TreeItem *, TrackItem *> E : track_items) {
		filter_tree->remove_child(E.value);
		E.value->queue_free();
	}
	track_items.clear();
	filter_tree->clear();

	if (!p_player) {
		EditorNode::get_singleton()->show_warning(TTR("Player path set is invalid, so unable to retrieve track names."));
		return false;
	}

	Node *base = p_player->get_node(p_player->get_root_node());

	if (!base) {
		EditorNode::get_singleton()->show_warning(TTR("Animation player has no valid root node path, so unable to retrieve track names."));
		return false;
	}

	updating = true;

	HashSet<String> paths;
	HashMap<String, RBSet<String>> types;
	{
		List<StringName> animation_list;
		p_player->get_animation_list(&animation_list);

		for (const StringName &E : animation_list) {
			Ref<Animation> anim = p_player->get_animation(E);
			for (int i = 0; i < anim->get_track_count(); i++) {
				String track_path = anim->track_get_path(i);
				paths.insert(track_path);

				String track_type_name;
				Animation::TrackType track_type = anim->track_get_type(i);
				switch (track_type) {
					case Animation::TrackType::TYPE_ANIMATION: {
						track_type_name = TTR("Anim Clips");
					} break;
					case Animation::TrackType::TYPE_AUDIO: {
						track_type_name = TTR("Audio Clips");
					} break;
					case Animation::TrackType::TYPE_METHOD: {
						track_type_name = TTR("Functions");
					} break;
					default: {
					} break;
				}
				if (!track_type_name.is_empty()) {
					types[track_path].insert(track_type_name);
				}
			}
		}
	}

	TreeItem *root = filter_tree->create_item();

	HashMap<String, TreeItem *> parenthood;

	for (const String &E : paths) {
		NodePath path = E;
		TreeItem *ti = nullptr;
		String accum;
		for (int i = 0; i < path.get_name_count(); i++) {
			String name = path.get_name(i);
			if (!accum.is_empty()) {
				accum += "/";
			}
			accum += name;
			if (!parenthood.has(accum)) {
				if (ti) {
					ti = filter_tree->create_item(ti);
				} else {
					ti = filter_tree->create_item(root);
				}
				parenthood[accum] = ti;

				TrackItem *track_item = get_or_create_track_items_and_setup_tree_item(ti);
				track_item->set_text(name);
				track_item->set_editable(false);
				track_item->set_as_uneditable_track(true);

				if (base->has_node(accum)) {
					Node *node = base->get_node(accum);
					track_item->set_icon(EditorNode::get_singleton()->get_object_icon(node, "Node"));
				}

			} else {
				ti = parenthood[accum];
			}
		}

		Node *node = nullptr;
		if (base->has_node(accum)) {
			node = base->get_node(accum);
		}
		if (!node) {
			continue; //no node, can't edit
		}

		if (path.get_subname_count()) {
			String concat = path.get_concatenated_subnames();

			Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(node);
			if (skeleton && skeleton->find_bone(concat) != -1) {
				//path in skeleton
				const String &bone = concat;
				int idx = skeleton->find_bone(bone);
				List<String> bone_path;
				while (idx != -1) {
					bone_path.push_front(skeleton->get_bone_name(idx));
					idx = skeleton->get_bone_parent(idx);
				}

				accum += ":";
				for (List<String>::Element *F = bone_path.front(); F; F = F->next()) {
					if (F != bone_path.front()) {
						accum += "/";
					}

					accum += F->get();
					if (!parenthood.has(accum)) {
						ti = filter_tree->create_item(ti);
						parenthood[accum] = ti;

						TrackItem *track_item = get_or_create_track_items_and_setup_tree_item(ti);
						track_item->set_icon(get_theme_icon(SNAME("BoneAttachment3D"), SNAME("EditorIcons")));
						track_item->set_editable(false);
						track_item->set_as_uneditable_track(true);
						track_item->set_text(F->get());
					} else {
						ti = parenthood[accum];
					}
				}

				ti->set_metadata(0, path);

				TrackItem *track_item = get_or_create_track_items_and_setup_tree_item(ti);
				track_item->set_track_path(path);
				track_item->set_icon(get_theme_icon(SNAME("BoneAttachment3D"), SNAME("EditorIcons")));
				track_item->set_editable(!is_read_only());
				track_item->set_amount(p_filter.is_valid() && p_filter->has_track(path) ? p_filter->get_track_amount(path) : 0.0f);
				track_item->set_text(concat);

			} else {
				//just a property
				ti = filter_tree->create_item(ti);
				ti->set_metadata(0, path);

				TrackItem *track_item = get_or_create_track_items_and_setup_tree_item(ti);
				track_item->set_track_path(path);
				track_item->set_editable(!is_read_only());
				track_item->set_amount(p_filter.is_valid() && p_filter->has_track(path) ? p_filter->get_track_amount(path) : 0.0f);
				track_item->set_text(concat);
			}
		} else {
			if (ti) {
				//just a node, not a property track
				String types_text = "[";
				if (types.has(path)) {
					RBSet<String>::Iterator F = types[path].begin();
					types_text += *F;
					while (F) {
						types_text += " / " + *F;
						;
						++F;
					}
				}
				types_text += "]";
				ti = filter_tree->create_item(ti);
				ti->set_metadata(0, path);

				TrackItem *track_item = get_or_create_track_items_and_setup_tree_item(ti);
				track_item->set_track_path(path);
				track_item->set_editable(!is_read_only());
				track_item->set_amount(p_filter.is_valid() && p_filter->has_track(path) ? p_filter->get_track_amount(path) : 0.0f);
				track_item->set_text(types_text);
			}
		}
	}

	updating = false;
	return true;
}

HBoxContainer *AnimationTrackFilterEditDialog::get_tool_bar() const {
	return tool_bar;
}

void AnimationTrackFilterEditDialog::_bind_methods() {
	ClassDB::bind_method("_tree_item_draw", &AnimationTrackFilterEditDialog::_tree_item_draw);
	ClassDB::bind_method("update_filters", &AnimationTrackFilterEditDialog::update_filters);
}

AnimationTrackFilterEditDialog::AnimationTrackFilterEditDialog() {
	VBoxContainer *vbox = memnew(VBoxContainer);
	add_child(vbox);

	tool_bar = memnew(HBoxContainer);
	vbox->add_child(tool_bar);

	filter_tree = memnew(Tree);
	vbox->add_child(filter_tree);
	filter_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	filter_tree->set_hide_root(true);
	filter_tree->connect("draw", callable_mp(this, &AnimationTrackFilterEditDialog::_tree_draw));

	tree_focus_rect = memnew(Control);
	tree_focus_rect->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	add_child(tree_focus_rect);
	tree_focus_rect->set_anchors_preset(Control::PRESET_FULL_RECT);

	set_read_only(false);
}

// AnimationNodeFilterTracksEditor
void AnimationNodeFilterTracksEditor::_edit_requested() {
	AnimationTrackFilter *filter = cast_to<AnimationTrackFilter>(get_edited_object());
	ERR_FAIL_NULL(filter);

	Callable call_back = callable_mp(this, &AnimationNodeFilterTracksEditor::update_property);
	if (!filter->is_connected("changed", call_back)) {
		filter->connect("changed", call_back);
	}

	edit_dialog->set_read_only(is_read_only());

	if (!update_filters()) {
		return;
	}

	edit_dialog->popup_centered(Size2(500, 500) * EDSCALE);
}

bool AnimationNodeFilterTracksEditor::update_filters() {
	AnimationTree *tree = AnimationTreeEditor::get_singleton()->get_animation_tree();
	if (!tree) {
		EditorNode::get_singleton()->show_warning(TTR("No animation tree selected, so unable to retrieve track names."));
		return false;
	}

	Ref<AnimationTrackFilter> filter = cast_to<AnimationTrackFilter>(get_edited_object());

	return edit_dialog->update_filters(tree, filter);
}

void AnimationNodeFilterTracksEditor::update_property() {
	edit_btn->set_text(TTR("Count: ") + itos(cast_to<AnimationTrackFilter>(get_edited_object())->get_tracks().size()));
}

AnimationNodeFilterTracksEditor::AnimationNodeFilterTracksEditor() {
	edit_btn = memnew(Button);
	edit_btn->set_h_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	edit_btn->connect("pressed", callable_mp(this, &AnimationNodeFilterTracksEditor::_edit_requested));
	add_child(edit_btn);

	edit_dialog = memnew(AnimationTrackFilterEditDialog);
	add_child(edit_dialog);
}

// AnimationTrackFilterEditorInspectorPlugin
bool AnimationTrackFilterEditorInspectorPlugin::can_handle(Object *p_object) {
	return p_object && cast_to<AnimationTrackFilter>(p_object) && AnimationTreeEditor::get_singleton()->get_animation_tree();
}

bool AnimationTrackFilterEditorInspectorPlugin::parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide) {
	if (cast_to<AnimationTrackFilter>(p_object) && p_path == "tracks") {
		add_property_editor(p_path, memnew(AnimationNodeFilterTracksEditor));
		return true;
	}
	return false;
}

AnimationTrackFilterEditorPlugin::AnimationTrackFilterEditorPlugin() {
	Ref<AnimationTrackFilterEditorInspectorPlugin> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}