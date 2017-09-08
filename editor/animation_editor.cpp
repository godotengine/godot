/*************************************************************************/
/*  animation_editor.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "animation_editor.h"

#include "editor/plugins/animation_player_editor_plugin.h"
#include "editor_node.h"
#include "editor_settings.h"
#include "io/resource_saver.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "pair.h"
#include "scene/gui/separator.h"
#include "scene/main/viewport.h"

/* Missing to fix:

  *Set
  *Find better source for hint for edited value keys
  * + button on track to add a key
  * when clicked for first time, erase selection of not selected at first
  * automatically create discrete/continuous tracks!!
  *when create track do undo/redo
*/

class AnimationCurveEdit : public Control {
	GDCLASS(AnimationCurveEdit, Control);

public:
	enum Mode {
		MODE_DISABLED,
		MODE_SINGLE,
		MODE_MULTIPLE
	};

private:
	Set<float> multiples;
	float transition;
	Mode mode;

	void _notification(int p_what) {

		if (p_what == NOTIFICATION_DRAW) {

			RID ci = get_canvas_item();

			Size2 s = get_size();
			Rect2 r(Point2(), s);

			//r=r.grow(3);
			Ref<StyleBox> sb = get_stylebox("normal", "LineEdit");
			sb->draw(ci, r);
			r.size -= sb->get_minimum_size();
			r.position += sb->get_offset();
			//VisualServer::get_singleton()->canvas_item_add

			Ref<Font> f = get_font("font", "Label");
			r = r.grow(-2);
			Color color = get_color("font_color", "Label");

			int points = 48;
			if (mode == MODE_MULTIPLE) {

				Color mcolor = color;
				mcolor.a *= 0.3;

				Set<float>::Element *E = multiples.front();
				for (int j = 0; j < 16; j++) {

					if (!E)
						break;

					float prev = 1.0;
					float exp = E->get();
					bool flip = false; //hint_text=="attenuation";

					for (int i = 1; i <= points; i++) {

						float ifl = i / float(points);
						float iflp = (i - 1) / float(points);

						float h = 1.0 - Math::ease(ifl, exp);

						if (flip) {
							ifl = 1.0 - ifl;
							iflp = 1.0 - iflp;
						}

						VisualServer::get_singleton()->canvas_item_add_line(ci, r.position + Point2(iflp * r.size.width, prev * r.size.height), r.position + Point2(ifl * r.size.width, h * r.size.height), mcolor);
						prev = h;
					}

					E = E->next();
				}
			}

			float exp = transition;
			if (mode != MODE_DISABLED) {

				float prev = 1.0;

				bool flip = false; //hint_text=="attenuation";

				for (int i = 1; i <= points; i++) {

					float ifl = i / float(points);
					float iflp = (i - 1) / float(points);

					float h = 1.0 - Math::ease(ifl, exp);

					if (flip) {
						ifl = 1.0 - ifl;
						iflp = 1.0 - iflp;
					}

					VisualServer::get_singleton()->canvas_item_add_line(ci, r.position + Point2(iflp * r.size.width, prev * r.size.height), r.position + Point2(ifl * r.size.width, h * r.size.height), color);
					prev = h;
				}
			}

			String txt = String::num(exp, 2);
			if (mode == MODE_DISABLED) {
				txt = TTR("Disabled");
			} else if (mode == MODE_MULTIPLE) {
				txt += " - " + TTR("All Selection");
			}

			f->draw(ci, Point2(10, 10 + f->get_ascent()), txt, color);
		}
	}

	void _gui_input(const Ref<InputEvent> &p_ev) {

		Ref<InputEventMouseMotion> mm = p_ev;
		if (mm.is_valid() && mm->get_button_mask() & BUTTON_MASK_LEFT) {

			if (mode == MODE_DISABLED)
				return;

			float rel = mm->get_relative().x;
			if (rel == 0)
				return;

			bool flip = false;

			if (flip)
				rel = -rel;

			float val = transition;
			if (val == 0)
				return;
			bool sg = val < 0;
			val = Math::absf(val);

			val = Math::log(val) / Math::log((float)2.0);
			//logspace
			val += rel * 0.05;
			//

			val = Math::pow((float)2.0, val);
			if (sg)
				val = -val;

			transition = val;
			update();
			//emit_signal("variant_changed");
			emit_signal("transition_changed", transition);
		}
	}

public:
	static void _bind_methods() {

		//ClassDB::bind_method("_update_obj",&AnimationKeyEdit::_update_obj);
		ClassDB::bind_method("_gui_input", &AnimationCurveEdit::_gui_input);
		ADD_SIGNAL(MethodInfo("transition_changed"));
	}

	void set_mode(Mode p_mode) {

		mode = p_mode;
		update();
	}

	void clear_multiples() {
		multiples.clear();
		update();
	}
	void set_multiple(float p_transition) {

		multiples.insert(p_transition);
	}

	void set_transition(float p_transition) {
		transition = p_transition;
		update();
	}

	float get_transition() const {
		return transition;
	}

	void force_transition(float p_value) {
		if (mode == MODE_DISABLED)
			return;
		transition = p_value;
		emit_signal("transition_changed", p_value);
		update();
	}

	AnimationCurveEdit() {

		transition = 1.0;
		set_default_cursor_shape(CURSOR_HSPLIT);
		mode = MODE_DISABLED;
	}
};

class AnimationKeyEdit : public Object {

	GDCLASS(AnimationKeyEdit, Object);

public:
	bool setting;
	bool hidden;

	static void _bind_methods() {

		ClassDB::bind_method("_update_obj", &AnimationKeyEdit::_update_obj);
		ClassDB::bind_method("_key_ofs_changed", &AnimationKeyEdit::_key_ofs_changed);
	}

	//PopupDialog *ke_dialog;

	void _fix_node_path(Variant &value) {

		NodePath np = value;

		if (np == NodePath())
			return;

		Node *root = EditorNode::get_singleton()->get_tree()->get_root();

		Node *np_node = root->get_node(np);
		ERR_FAIL_COND(!np_node);

		Node *edited_node = root->get_node(base);
		ERR_FAIL_COND(!edited_node);

		value = edited_node->get_path_to(np_node);
	}

	void _update_obj(const Ref<Animation> &p_anim) {
		if (setting)
			return;
		if (hidden)
			return;
		if (!(animation == p_anim))
			return;
		notify_change();
	}

	void _key_ofs_changed(const Ref<Animation> &p_anim, float from, float to) {
		if (hidden)
			return;
		if (!(animation == p_anim))
			return;
		if (from != key_ofs)
			return;
		key_ofs = to;
		if (setting)
			return;
		notify_change();
	}

	bool _set(const StringName &p_name, const Variant &p_value) {

		int key = animation->track_find_key(track, key_ofs, true);
		ERR_FAIL_COND_V(key == -1, false);

		String name = p_name;
		if (name == "time") {

			float new_time = p_value;
			if (new_time == key_ofs)
				return true;

			int existing = animation->track_find_key(track, new_time, true);

			setting = true;
			undo_redo->create_action(TTR("Move Add Key"), UndoRedo::MERGE_ENDS);

			Variant val = animation->track_get_key_value(track, key);
			float trans = animation->track_get_key_transition(track, key);

			undo_redo->add_do_method(animation.ptr(), "track_remove_key", track, key);
			undo_redo->add_do_method(animation.ptr(), "track_insert_key", track, new_time, val, trans);
			undo_redo->add_do_method(this, "_key_ofs_changed", animation, key_ofs, new_time);
			undo_redo->add_undo_method(animation.ptr(), "track_remove_key_at_pos", track, new_time);
			undo_redo->add_undo_method(animation.ptr(), "track_insert_key", track, key_ofs, val, trans);
			undo_redo->add_undo_method(this, "_key_ofs_changed", animation, new_time, key_ofs);

			if (existing != -1) {
				Variant v = animation->track_get_key_value(track, existing);
				float trans = animation->track_get_key_transition(track, existing);
				undo_redo->add_undo_method(animation.ptr(), "track_insert_key", track, new_time, v, trans);
			}

			undo_redo->commit_action();
			setting = false;

			return true;
		} else if (name == "easing") {

			float val = p_value;
			float prev_val = animation->track_get_key_transition(track, key);
			setting = true;
			undo_redo->create_action(TTR("Anim Change Transition"), UndoRedo::MERGE_ENDS);
			undo_redo->add_do_method(animation.ptr(), "track_set_key_transition", track, key, val);
			undo_redo->add_undo_method(animation.ptr(), "track_set_key_transition", track, key, prev_val);
			undo_redo->add_do_method(this, "_update_obj", animation);
			undo_redo->add_undo_method(this, "_update_obj", animation);
			undo_redo->commit_action();
			setting = false;
			return true;
		}

		switch (animation->track_get_type(track)) {

			case Animation::TYPE_TRANSFORM: {

				Dictionary d_old = animation->track_get_key_value(track, key);
				Dictionary d_new = d_old;
				d_new[p_name] = p_value;
				setting = true;
				undo_redo->create_action(TTR("Anim Change Transform"));
				undo_redo->add_do_method(animation.ptr(), "track_set_key_value", track, key, d_new);
				undo_redo->add_undo_method(animation.ptr(), "track_set_key_value", track, key, d_old);
				undo_redo->add_do_method(this, "_update_obj", animation);
				undo_redo->add_undo_method(this, "_update_obj", animation);
				undo_redo->commit_action();
				setting = false;
				return true;

			} break;
			case Animation::TYPE_VALUE: {

				if (name == "value") {

					Variant value = p_value;

					if (value.get_type() == Variant::NODE_PATH) {

						_fix_node_path(value);
					}

					setting = true;
					undo_redo->create_action(TTR("Anim Change Value"), UndoRedo::MERGE_ENDS);
					Variant prev = animation->track_get_key_value(track, key);
					undo_redo->add_do_method(animation.ptr(), "track_set_key_value", track, key, value);
					undo_redo->add_undo_method(animation.ptr(), "track_set_key_value", track, key, prev);
					undo_redo->add_do_method(this, "_update_obj", animation);
					undo_redo->add_undo_method(this, "_update_obj", animation);
					undo_redo->commit_action();
					setting = false;
					return true;
				}

			} break;
			case Animation::TYPE_METHOD: {

				Dictionary d_old = animation->track_get_key_value(track, key);
				Dictionary d_new = d_old;

				bool change_notify_deserved = false;
				bool mergeable = false;

				if (name == "name") {

					d_new["method"] = p_value;
				}

				if (name == "arg_count") {

					Vector<Variant> args = d_old["args"];
					args.resize(p_value);
					d_new["args"] = args;
					change_notify_deserved = true;
				}

				if (name.begins_with("args/")) {

					Vector<Variant> args = d_old["args"];
					int idx = name.get_slice("/", 1).to_int();
					ERR_FAIL_INDEX_V(idx, args.size(), false);

					String what = name.get_slice("/", 2);
					if (what == "type") {
						Variant::Type t = Variant::Type(int(p_value));

						if (t != args[idx].get_type()) {
							Variant::CallError err;
							if (Variant::can_convert(args[idx].get_type(), t)) {
								Variant old = args[idx];
								Variant *ptrs[1] = { &old };
								args[idx] = Variant::construct(t, (const Variant **)ptrs, 1, err);
							} else {

								args[idx] = Variant::construct(t, NULL, 0, err);
							}
							change_notify_deserved = true;
							d_new["args"] = args;
						}
					}
					if (what == "value") {

						Variant value = p_value;
						if (value.get_type() == Variant::NODE_PATH) {

							_fix_node_path(value);
						}

						args[idx] = value;
						d_new["args"] = args;
						mergeable = true;
					}
				}

				if (mergeable)
					undo_redo->create_action(TTR("Anim Change Call"), UndoRedo::MERGE_ENDS);
				else
					undo_redo->create_action(TTR("Anim Change Call"));

				Variant prev = animation->track_get_key_value(track, key);
				setting = true;
				undo_redo->add_do_method(animation.ptr(), "track_set_key_value", track, key, d_new);
				undo_redo->add_undo_method(animation.ptr(), "track_set_key_value", track, key, d_old);
				undo_redo->add_do_method(this, "_update_obj", animation);
				undo_redo->add_undo_method(this, "_update_obj", animation);
				undo_redo->commit_action();
				setting = false;
				if (change_notify_deserved)
					notify_change();
				return true;
			} break;
		}

		return false;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {

		int key = animation->track_find_key(track, key_ofs, true);
		ERR_FAIL_COND_V(key == -1, false);

		String name = p_name;
		if (name == "time") {
			r_ret = key_ofs;
			return true;
		} else if (name == "easing") {
			r_ret = animation->track_get_key_transition(track, key);
			return true;
		}

		switch (animation->track_get_type(track)) {

			case Animation::TYPE_TRANSFORM: {

				Dictionary d = animation->track_get_key_value(track, key);
				ERR_FAIL_COND_V(!d.has(name), false);
				r_ret = d[p_name];
				return true;

			} break;
			case Animation::TYPE_VALUE: {

				if (name == "value") {
					r_ret = animation->track_get_key_value(track, key);
					return true;
				}

			} break;
			case Animation::TYPE_METHOD: {

				Dictionary d = animation->track_get_key_value(track, key);

				if (name == "name") {

					ERR_FAIL_COND_V(!d.has("method"), false);
					r_ret = d["method"];
					return true;
				}

				ERR_FAIL_COND_V(!d.has("args"), false);

				Vector<Variant> args = d["args"];

				if (name == "arg_count") {

					r_ret = args.size();
					return true;
				}

				if (name.begins_with("args/")) {

					int idx = name.get_slice("/", 1).to_int();
					ERR_FAIL_INDEX_V(idx, args.size(), false);

					String what = name.get_slice("/", 2);
					if (what == "type") {
						r_ret = args[idx].get_type();
						return true;
					}
					if (what == "value") {
						r_ret = args[idx];
						return true;
					}
				}

			} break;
		}

		return false;
	}
	void _get_property_list(List<PropertyInfo> *p_list) const {

		if (animation.is_null())
			return;

		ERR_FAIL_INDEX(track, animation->get_track_count());
		int key = animation->track_find_key(track, key_ofs, true);
		ERR_FAIL_COND(key == -1);

		p_list->push_back(PropertyInfo(Variant::REAL, "time", PROPERTY_HINT_RANGE, "0," + rtos(animation->get_length()) + ",0.01"));

		switch (animation->track_get_type(track)) {

			case Animation::TYPE_TRANSFORM: {

				p_list->push_back(PropertyInfo(Variant::VECTOR3, "loc"));
				p_list->push_back(PropertyInfo(Variant::QUAT, "rot"));
				p_list->push_back(PropertyInfo(Variant::VECTOR3, "scale"));

			} break;
			case Animation::TYPE_VALUE: {

				Variant v = animation->track_get_key_value(track, key);

				if (hint.type != Variant::NIL) {

					PropertyInfo pi = hint;
					pi.name = "value";
					p_list->push_back(pi);
				} else {

					PropertyHint hint = PROPERTY_HINT_NONE;
					String hint_string;

					if (v.get_type() == Variant::OBJECT) {
						//could actually check the object property if exists..? yes i will!
						Ref<Resource> res = v;
						if (res.is_valid()) {

							hint = PROPERTY_HINT_RESOURCE_TYPE;
							hint_string = res->get_class();
						}
					}

					if (v.get_type() != Variant::NIL)
						p_list->push_back(PropertyInfo(v.get_type(), "value", hint, hint_string));
				}

			} break;
			case Animation::TYPE_METHOD: {

				p_list->push_back(PropertyInfo(Variant::STRING, "name"));
				p_list->push_back(PropertyInfo(Variant::INT, "arg_count", PROPERTY_HINT_RANGE, "0,5,1"));

				Dictionary d = animation->track_get_key_value(track, key);
				ERR_FAIL_COND(!d.has("args"));
				Vector<Variant> args = d["args"];
				String vtypes;
				for (int i = 0; i < Variant::VARIANT_MAX; i++) {

					if (i > 0)
						vtypes += ",";
					vtypes += Variant::get_type_name(Variant::Type(i));
				}

				for (int i = 0; i < args.size(); i++) {

					p_list->push_back(PropertyInfo(Variant::INT, "args/" + itos(i) + "/type", PROPERTY_HINT_ENUM, vtypes));
					if (args[i].get_type() != Variant::NIL)
						p_list->push_back(PropertyInfo(args[i].get_type(), "args/" + itos(i) + "/value"));
				}

			} break;
		}

		/*
		if (animation->track_get_type(track)!=Animation::TYPE_METHOD)
			p_list->push_back( PropertyInfo( Variant::REAL, "easing", PROPERTY_HINT_EXP_EASING));
		*/
	}

	UndoRedo *undo_redo;
	Ref<Animation> animation;
	int track;
	float key_ofs;

	PropertyInfo hint;
	NodePath base;

	void notify_change() {

		_change_notify();
	}

	AnimationKeyEdit() {
		hidden = true;
		key_ofs = 0;
		track = -1;
		setting = false;
	}
};

void AnimationKeyEditor::_menu_add_track(int p_type) {

	ERR_FAIL_COND(!animation.is_valid());

	switch (p_type) {

		case ADD_TRACK_MENU_ADD_CALL_TRACK: {
			if (root) {
				call_select->popup_centered_ratio();
				break;
			}
		} break;
		case ADD_TRACK_MENU_ADD_VALUE_TRACK:
		case ADD_TRACK_MENU_ADD_TRANSFORM_TRACK: {

			undo_redo->create_action(TTR("Anim Add Track"));
			undo_redo->add_do_method(animation.ptr(), "add_track", p_type);
			undo_redo->add_do_method(animation.ptr(), "track_set_path", animation->get_track_count(), ".");
			undo_redo->add_undo_method(animation.ptr(), "remove_track", animation->get_track_count());
			undo_redo->commit_action();

		} break;
	}
}

void AnimationKeyEditor::_anim_duplicate_keys(bool transpose) {
	//duplicait!
	if (selection.size() && animation.is_valid() && selected_track >= 0 && selected_track < animation->get_track_count()) {

		int top_track = 0x7FFFFFFF;
		float top_time = 1e10;
		for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

			const SelectedKey &sk = E->key();

			float t = animation->track_get_key_time(sk.track, sk.key);
			if (t < top_time)
				top_time = t;
			if (sk.track < top_track)
				top_track = sk.track;
		}
		ERR_FAIL_COND(top_track == 0x7FFFFFFF || top_time == 1e10);

		//

		int start_track = transpose ? selected_track : top_track;

		undo_redo->create_action(TTR("Anim Duplicate Keys"));

		List<Pair<int, float> > new_selection_values;

		for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

			const SelectedKey &sk = E->key();

			float t = animation->track_get_key_time(sk.track, sk.key);

			float dst_time = t + (timeline_pos - top_time);
			int dst_track = sk.track + (start_track - top_track);

			if (dst_track < 0 || dst_track >= animation->get_track_count())
				continue;

			if (animation->track_get_type(dst_track) != animation->track_get_type(sk.track))
				continue;

			int existing_idx = animation->track_find_key(dst_track, dst_time, true);

			undo_redo->add_do_method(animation.ptr(), "track_insert_key", dst_track, dst_time, animation->track_get_key_value(E->key().track, E->key().key), animation->track_get_key_transition(E->key().track, E->key().key));
			undo_redo->add_undo_method(animation.ptr(), "track_remove_key_at_pos", dst_track, dst_time);

			Pair<int, float> p;
			p.first = dst_track;
			p.second = dst_time;
			new_selection_values.push_back(p);

			if (existing_idx != -1) {

				undo_redo->add_undo_method(animation.ptr(), "track_insert_key", dst_track, dst_time, animation->track_get_key_value(dst_track, existing_idx), animation->track_get_key_transition(dst_track, existing_idx));
			}
		}

		undo_redo->commit_action();

		//reselect duplicated

		Map<SelectedKey, KeyInfo> new_selection;
		for (List<Pair<int, float> >::Element *E = new_selection_values.front(); E; E = E->next()) {

			int track = E->get().first;
			float time = E->get().second;

			int existing_idx = animation->track_find_key(track, time, true);

			if (existing_idx == -1)
				continue;
			SelectedKey sk2;
			sk2.track = track;
			sk2.key = existing_idx;

			KeyInfo ki;
			ki.pos = time;

			new_selection[sk2] = ki;
		}

		selection = new_selection;
		track_editor->update();
		_edit_if_single_selection();
	}
}

void AnimationKeyEditor::_menu_track(int p_type) {

	ERR_FAIL_COND(!animation.is_valid());

	last_menu_track_opt = p_type;
	switch (p_type) {

		case TRACK_MENU_SCALE:
		case TRACK_MENU_SCALE_PIVOT: {

			scale_dialog->popup_centered(Size2(200, 100));
		} break;
		case TRACK_MENU_MOVE_UP: {

			int idx = selected_track;
			if (idx > 0 && idx < animation->get_track_count()) {
				undo_redo->create_action(TTR("Move Anim Track Up"));
				undo_redo->add_do_method(animation.ptr(), "track_move_down", idx);
				undo_redo->add_undo_method(animation.ptr(), "track_move_up", idx - 1);
				undo_redo->commit_action();
				selected_track = idx - 1;
			}

		} break;
		case TRACK_MENU_MOVE_DOWN: {

			int idx = selected_track;
			if (idx >= 0 && idx < animation->get_track_count() - 1) {
				undo_redo->create_action(TTR("Move Anim Track Down"));
				undo_redo->add_do_method(animation.ptr(), "track_move_up", idx);
				undo_redo->add_undo_method(animation.ptr(), "track_move_down", idx + 1);
				undo_redo->commit_action();
				selected_track = idx + 1;
			}

		} break;
		case TRACK_MENU_REMOVE: {

			int idx = selected_track;
			if (idx >= 0 && idx < animation->get_track_count()) {
				undo_redo->create_action(TTR("Remove Anim Track"));
				undo_redo->add_do_method(animation.ptr(), "remove_track", idx);
				undo_redo->add_undo_method(animation.ptr(), "add_track", animation->track_get_type(idx), idx);
				undo_redo->add_undo_method(animation.ptr(), "track_set_path", idx, animation->track_get_path(idx));
				//todo interpolation
				for (int i = 0; i < animation->track_get_key_count(idx); i++) {

					Variant v = animation->track_get_key_value(idx, i);
					float time = animation->track_get_key_time(idx, i);
					float trans = animation->track_get_key_transition(idx, i);

					undo_redo->add_undo_method(animation.ptr(), "track_insert_key", idx, time, v);
					undo_redo->add_undo_method(animation.ptr(), "track_set_key_transition", idx, i, trans);
				}

				undo_redo->add_undo_method(animation.ptr(), "track_set_interpolation_type", idx, animation->track_get_interpolation_type(idx));
				if (animation->track_get_type(idx) == Animation::TYPE_VALUE) {
					undo_redo->add_undo_method(animation.ptr(), "value_track_set_update_mode", idx, animation->value_track_get_update_mode(idx));
				}

				undo_redo->commit_action();
			}

		} break;
		case TRACK_MENU_DUPLICATE:
		case TRACK_MENU_DUPLICATE_TRANSPOSE: {

			_anim_duplicate_keys(p_type == TRACK_MENU_DUPLICATE_TRANSPOSE);
		} break;
		case TRACK_MENU_SET_ALL_TRANS_LINEAR:
		case TRACK_MENU_SET_ALL_TRANS_CONSTANT:
		case TRACK_MENU_SET_ALL_TRANS_OUT:
		case TRACK_MENU_SET_ALL_TRANS_IN:
		case TRACK_MENU_SET_ALL_TRANS_INOUT:
		case TRACK_MENU_SET_ALL_TRANS_OUTIN: {

			if (!selection.size() || !animation.is_valid())
				break;

			float t = 0;
			switch (p_type) {
				case TRACK_MENU_SET_ALL_TRANS_LINEAR: t = 1.0; break;
				case TRACK_MENU_SET_ALL_TRANS_CONSTANT: t = 0.0; break;
				case TRACK_MENU_SET_ALL_TRANS_OUT: t = 0.5; break;
				case TRACK_MENU_SET_ALL_TRANS_IN: t = 2.0; break;
				case TRACK_MENU_SET_ALL_TRANS_INOUT: t = -0.5; break;
				case TRACK_MENU_SET_ALL_TRANS_OUTIN: t = -2.0; break;
			}

			undo_redo->create_action(TTR("Set Transitions to:") + " " + rtos(t));

			for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

				const SelectedKey &sk = E->key();

				undo_redo->add_do_method(animation.ptr(), "track_set_key_transition", sk.track, sk.key, t);
				undo_redo->add_undo_method(animation.ptr(), "track_set_key_transition", sk.track, sk.key, animation->track_get_key_transition(sk.track, sk.key));
			}

			undo_redo->commit_action();

		} break;
		case TRACK_MENU_NEXT_STEP: {

			if (animation.is_null())
				break;
			float step = animation->get_step();
			if (step == 0)
				step = 1;

			float pos = timeline_pos;

			pos = Math::stepify(pos + step, step);
			if (pos > animation->get_length())
				pos = animation->get_length();
			timeline_pos = pos;
			track_pos->update();
			emit_signal("timeline_changed", pos, true);

		} break;
		case TRACK_MENU_PREV_STEP: {
			if (animation.is_null())
				break;
			float step = animation->get_step();
			if (step == 0)
				step = 1;

			float pos = timeline_pos;
			pos = Math::stepify(pos - step, step);
			if (pos < 0)
				pos = 0;
			timeline_pos = pos;
			track_pos->update();
			emit_signal("timeline_changed", pos, true);

		} break;

		case TRACK_MENU_OPTIMIZE: {

			optimize_dialog->popup_centered(Size2(250, 180));
		} break;
		case TRACK_MENU_CLEAN_UP: {

			cleanup_dialog->popup_centered_minsize(Size2(300, 0));
		} break;
		case TRACK_MENU_CLEAN_UP_CONFIRM: {

			if (cleanup_all->is_pressed()) {
				List<StringName> names;
				AnimationPlayerEditor::singleton->get_player()->get_animation_list(&names);
				for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
					_cleanup_animation(AnimationPlayerEditor::singleton->get_player()->get_animation(E->get()));
				}
			} else {
				_cleanup_animation(animation);
			}
		} break;
		case CURVE_SET_LINEAR: {
			curve_edit->force_transition(1.0);

		} break;
		case CURVE_SET_IN: {

			curve_edit->force_transition(4.0);

		} break;
		case CURVE_SET_OUT: {

			curve_edit->force_transition(0.25);
		} break;
		case CURVE_SET_INOUT: {
			curve_edit->force_transition(-4);

		} break;
		case CURVE_SET_OUTIN: {

			curve_edit->force_transition(-0.25);
		} break;
		case CURVE_SET_CONSTANT: {

			curve_edit->force_transition(0);
		} break;
	}
}

void AnimationKeyEditor::_cleanup_animation(Ref<Animation> p_animation) {

	for (int i = 0; i < p_animation->get_track_count(); i++) {

		bool prop_exists = false;
		Variant::Type valid_type = Variant::NIL;
		Object *obj = NULL;

		RES res;
		Node *node = root->get_node_and_resource(p_animation->track_get_path(i), res);

		if (res.is_valid()) {
			obj = res.ptr();
		} else if (node) {
			obj = node;
		}

		if (obj && p_animation->track_get_type(i) == Animation::TYPE_VALUE) {
			valid_type = obj->get_static_property_type(p_animation->track_get_path(i).get_property(), &prop_exists);
		}

		if (!obj && cleanup_tracks->is_pressed()) {

			p_animation->remove_track(i);
			i--;
			continue;
		}

		if (!prop_exists || p_animation->track_get_type(i) != Animation::TYPE_VALUE || cleanup_keys->is_pressed() == false)
			continue;

		for (int j = 0; j < p_animation->track_get_key_count(i); j++) {

			Variant v = p_animation->track_get_key_value(i, j);

			if (!Variant::can_convert(v.get_type(), valid_type)) {
				p_animation->track_remove_key(i, j);
				j--;
			}
		}

		if (p_animation->track_get_key_count(i) == 0 && cleanup_tracks->is_pressed()) {
			p_animation->remove_track(i);
			i--;
		}
	}

	undo_redo->clear_history();
	_update_paths();
}

void AnimationKeyEditor::_animation_optimize() {

	animation->optimize(optimize_linear_error->get_value(), optimize_angular_error->get_value(), optimize_max_angle->get_value());
	track_editor->update();
	undo_redo->clear_history();
}

float AnimationKeyEditor::_get_zoom_scale() const {

	float zv = zoom->get_value();
	if (zv < 1) {
		zv = 1.0 - zv;
		return Math::pow(1.0f + zv, 8.0f) * 100;
	} else {
		return 1.0 / Math::pow(zv, 8.0f) * 100;
	}
}

void AnimationKeyEditor::_track_pos_draw() {

	if (!animation.is_valid()) {
		return;
	}

	Ref<StyleBox> style = get_stylebox("normal", "TextEdit");
	Size2 size = track_editor->get_size() - style->get_minimum_size();
	Size2 ofs = style->get_offset();

	int settings_limit = size.width - right_data_size_cache;
	int name_limit = settings_limit * name_column_ratio;

	float keys_from = h_scroll->get_value();
	float zoom_scale = _get_zoom_scale();
	float keys_to = keys_from + (settings_limit - name_limit) / zoom_scale;

	//will move to separate control! (for speedup)
	if (timeline_pos >= keys_from && timeline_pos < keys_to) {
		//draw position
		int pixel = (timeline_pos - h_scroll->get_value()) * zoom_scale;
		pixel += name_limit;
		track_pos->draw_line(ofs + Point2(pixel, 0), ofs + Point2(pixel, size.height), get_color("accent_color", "Editor"));
	}
}

void AnimationKeyEditor::_track_editor_draw() {

	if (animation.is_valid() && animation->get_track_count()) {
		if (selected_track < 0)
			selected_track = 0;
		else if (selected_track >= animation->get_track_count())
			selected_track = animation->get_track_count() - 1;
	}

	track_pos->update();
	Control *te = track_editor;
	Ref<StyleBox> style = get_stylebox("normal", "TextEdit");
	te->draw_style_box(style, Rect2(Point2(), track_editor->get_size()));

	if (te->has_focus()) {
		te->draw_style_box(get_stylebox("bg_focus", "Tree"), Rect2(Point2(), track_editor->get_size()));
	}

	if (!animation.is_valid()) {
		v_scroll->hide();
		h_scroll->hide();
		menu_add_track->set_disabled(true);
		menu_track->set_disabled(true);
		edit_button->set_disabled(true);
		key_editor_tab->hide();
		move_up_button->set_disabled(true);
		move_down_button->set_disabled(true);
		remove_button->set_disabled(true);

		return;
	}

	menu_add_track->set_disabled(false);
	menu_track->set_disabled(false);
	edit_button->set_disabled(false);
	move_up_button->set_disabled(false);
	move_down_button->set_disabled(false);
	remove_button->set_disabled(false);
	if (edit_button->is_pressed())
		key_editor_tab->show();

	te_drawing = true;

	Size2 size = te->get_size() - style->get_minimum_size();
	Size2 ofs = style->get_offset();

	Ref<Font> font = te->get_font("font", "Tree");
	int sep = get_constant("vseparation", "Tree");
	int hsep = get_constant("hseparation", "Tree");
	Color color = get_color("font_color", "Tree");
	Color sepcolor = color;
	sepcolor.a = 0.2;
	Color timecolor = color;
	timecolor.a = 0.2;
	Color hover_color = color;
	hover_color.a = 0.05;
	Color select_color = color;
	select_color.a = 0.1;
	Color invalid_path_color = get_color("error_color", "Editor");
	Color track_select_color = get_color("accent", "Editor");

	Ref<Texture> remove_icon = get_icon("Remove", "EditorIcons");
	Ref<Texture> move_up_icon = get_icon("MoveUp", "EditorIcons");
	Ref<Texture> move_down_icon = get_icon("MoveDown", "EditorIcons");
	Ref<Texture> remove_icon_hl = get_icon("RemoveHl", "EditorIcons");
	Ref<Texture> move_up_icon_hl = get_icon("MoveUpHl", "EditorIcons");
	Ref<Texture> move_down_icon_hl = get_icon("MoveDownHl", "EditorIcons");
	Ref<Texture> add_key_icon = get_icon("TrackAddKey", "EditorIcons");
	Ref<Texture> add_key_icon_hl = get_icon("TrackAddKeyHl", "EditorIcons");
	Ref<Texture> down_icon = get_icon("select_arrow", "Tree");

	Ref<Texture> wrap_icon[2] = {
		get_icon("InterpWrapClamp", "EditorIcons"),
		get_icon("InterpWrapLoop", "EditorIcons"),
	};

	Ref<Texture> interp_icon[3] = {
		get_icon("InterpRaw", "EditorIcons"),
		get_icon("InterpLinear", "EditorIcons"),
		get_icon("InterpCubic", "EditorIcons")
	};
	Ref<Texture> cont_icon[3] = {
		get_icon("TrackContinuous", "EditorIcons"),
		get_icon("TrackDiscrete", "EditorIcons"),
		get_icon("TrackTrigger", "EditorIcons")
	};
	Ref<Texture> type_icon[3] = {
		get_icon("KeyValue", "EditorIcons"),
		get_icon("KeyXform", "EditorIcons"),
		get_icon("KeyCall", "EditorIcons")
	};

	Ref<Texture> invalid_icon = get_icon("KeyInvalid", "EditorIcons");
	Ref<Texture> invalid_icon_hover = get_icon("KeyInvalidHover", "EditorIcons");

	Ref<Texture> hsize_icon = get_icon("Hsize", "EditorIcons");

	Ref<Texture> type_hover = get_icon("KeyHover", "EditorIcons");
	Ref<Texture> type_selected = get_icon("KeySelected", "EditorIcons");

	int right_separator_ofs = down_icon->get_width() * 3 + add_key_icon->get_width() + interp_icon[0]->get_width() + wrap_icon[0]->get_width() + cont_icon[0]->get_width() + hsep * 9;

	int h = font->get_height() + sep;

	int fit = (size.height / h) - 1;
	int total = animation->get_track_count();
	if (total < fit) {
		v_scroll->hide();
		v_scroll->set_max(total);
		v_scroll->set_page(fit);
	} else {
		v_scroll->show();
		v_scroll->set_max(total);
		v_scroll->set_page(fit);
	}

	int settings_limit = size.width - right_separator_ofs;
	int name_limit = settings_limit * name_column_ratio;

	Color linecolor = color;
	linecolor.a = 0.2;
	te->draw_line(ofs + Point2(name_limit, 0), ofs + Point2(name_limit, size.height), linecolor);
	te->draw_line(ofs + Point2(settings_limit, 0), ofs + Point2(settings_limit, size.height), linecolor);
	te->draw_texture(hsize_icon, ofs + Point2(name_limit - hsize_icon->get_width() - hsep, (h - hsize_icon->get_height()) / 2));

	te->draw_line(ofs + Point2(0, h), ofs + Point2(size.width, h), linecolor);
	// draw time

	float keys_from;
	float keys_to;
	float zoom_scale;

	{

		int zoomw = settings_limit - name_limit;

		float scale = _get_zoom_scale();
		zoom_scale = scale;

		float l = animation->get_length();
		if (l <= 0)
			l = 0.001; //avoid crashor

		int end_px = (l - h_scroll->get_value()) * scale;
		int begin_px = -h_scroll->get_value() * scale;
		Color notimecol = get_color("dark_color_2", "Editor");

		{

			te->draw_rect(Rect2(ofs + Point2(name_limit, 0), Point2(zoomw - 1, h)), notimecol);

			if (begin_px < zoomw && end_px > 0) {

				if (begin_px < 0)
					begin_px = 0;
				if (end_px > zoomw)
					end_px = zoomw;

				te->draw_rect(Rect2(ofs + Point2(name_limit + begin_px, 0), Point2(end_px - begin_px - 1, h)), timecolor);
			}
		}

		keys_from = h_scroll->get_value();
		keys_to = keys_from + zoomw / scale;

		{
			float time_min = 0;
			float time_max = animation->get_length();
			for (int i = 0; i < animation->get_track_count(); i++) {

				if (animation->track_get_key_count(i) > 0) {

					float beg = animation->track_get_key_time(i, 0);
					if (beg < time_min)
						time_min = beg;
					float end = animation->track_get_key_time(i, animation->track_get_key_count(i) - 1);
					if (end > time_max)
						time_max = end;
				}
			}

			float extra = (zoomw / scale) * 0.5;

			if (time_min < -0.001)
				time_min -= extra;
			time_max += extra;
			h_scroll->set_min(time_min);
			h_scroll->set_max(time_max);

			if (zoomw / scale < (time_max - time_min)) {
				h_scroll->show();

			} else {

				h_scroll->hide();
			}
		}

		h_scroll->set_page(zoomw / scale);

		Color color_time_sec = color;
		Color color_time_dec = color;
		color_time_dec.a *= 0.5;
#define SC_ADJ 100
		int min = 30;
		int dec = 1;
		int step = 1;
		int decimals = 2;
		bool step_found = false;

		while (!step_found) {

			static const int _multp[3] = { 1, 2, 5 };
			for (int i = 0; i < 3; i++) {

				step = (_multp[i] * dec);
				if (step * scale / SC_ADJ > min) {
					step_found = true;
					break;
				}
			}
			if (step_found)
				break;
			dec *= 10;
			decimals--;
			if (decimals < 0)
				decimals = 0;
		}

		for (int i = 0; i < zoomw; i++) {

			float pos = h_scroll->get_value() + double(i) / scale;
			float prev = h_scroll->get_value() + (double(i) - 1.0) / scale;

			int sc = int(Math::floor(pos * SC_ADJ));
			int prev_sc = int(Math::floor(prev * SC_ADJ));
			bool sub = (sc % SC_ADJ);

			if ((sc / step) != (prev_sc / step) || (prev_sc < 0 && sc >= 0)) {

				int scd = sc < 0 ? prev_sc : sc;
				te->draw_line(ofs + Point2(name_limit + i, 0), ofs + Point2(name_limit + i, h), linecolor);
				te->draw_string(font, ofs + Point2(name_limit + i + 3, (h - font->get_height()) / 2 + font->get_ascent()).floor(), String::num((scd - (scd % step)) / double(SC_ADJ), decimals), sub ? color_time_dec : color_time_sec, zoomw - i);
			}
		}
	}

	color.a *= 0.5;

	for (int i = 0; i < fit; i++) {

		//this code sucks, i always forget how it works

		int idx = v_scroll->get_value() + i;
		if (idx >= animation->get_track_count())
			break;
		int y = h + i * h + sep;

		bool prop_exists = false;
		Variant::Type valid_type = Variant::NIL;
		Object *obj = NULL;

		RES res;
		Node *node = root ? root->get_node_and_resource(animation->track_get_path(idx), res) : (Node *)NULL;

		if (res.is_valid()) {
			obj = res.ptr();
		} else if (node) {
			obj = node;
		}

		if (obj && animation->track_get_type(idx) == Animation::TYPE_VALUE) {
			valid_type = obj->get_static_property_type(animation->track_get_path(idx).get_property(), &prop_exists);
		}

		if (/*mouse_over.over!=MouseOver::OVER_NONE &&*/ idx == mouse_over.track) {
			Color sepc = hover_color;
			te->draw_rect(Rect2(ofs + Point2(0, y), Size2(size.width, h - 1)), sepc);
		}

		if (selected_track == idx) {
			Color tc = select_color;
			//tc.a*=0.7;
			te->draw_rect(Rect2(ofs + Point2(0, y), Size2(size.width - 1, h - 1)), tc);
		}

		te->draw_texture(type_icon[animation->track_get_type(idx)], ofs + Point2(0, y + (h - type_icon[0]->get_height()) / 2).floor());
		NodePath np = animation->track_get_path(idx);
		Node *n = root ? root->get_node(np) : (Node *)NULL;
		Color ncol = color;
		if (n && editor_selection->is_selected(n))
			ncol = track_select_color;
		te->draw_string(font, Point2(ofs + Point2(type_icon[0]->get_width() + sep, y + font->get_ascent() + (sep / 2))).floor(), np, ncol, name_limit - (type_icon[0]->get_width() + sep) - 5);

		if (!obj)
			te->draw_line(ofs + Point2(0, y + h / 2), ofs + Point2(name_limit, y + h / 2), invalid_path_color);

		te->draw_line(ofs + Point2(0, y + h), ofs + Point2(size.width, y + h), sepcolor);

		Point2 icon_ofs = ofs + Point2(size.width, y + (h - remove_icon->get_height()) / 2).floor();
		icon_ofs.y += 4 * EDSCALE;

		/*		icon_ofs.x-=remove_icon->get_width();

		te->draw_texture((mouse_over.over==MouseOver::OVER_REMOVE && mouse_over.track==idx)?remove_icon_hl:remove_icon,icon_ofs);
		icon_ofs.x-=hsep;
		icon_ofs.x-=move_down_icon->get_width();
		te->draw_texture((mouse_over.over==MouseOver::OVER_DOWN && mouse_over.track==idx)?move_down_icon_hl:move_down_icon,icon_ofs);
		icon_ofs.x-=hsep;
		icon_ofs.x-=move_up_icon->get_width();
		te->draw_texture((mouse_over.over==MouseOver::OVER_UP && mouse_over.track==idx)?move_up_icon_hl:move_up_icon,icon_ofs);
		icon_ofs.x-=hsep;
		te->draw_line(Point2(icon_ofs.x,ofs.y+y),Point2(icon_ofs.x,ofs.y+y+h),sepcolor);

		icon_ofs.x-=hsep;
		*/
		track_ofs[0] = size.width - icon_ofs.x;
		icon_ofs.x -= down_icon->get_width();
		te->draw_texture(down_icon, icon_ofs - Size2(0, 4 * EDSCALE));

		int wrap_type = animation->track_get_interpolation_loop_wrap(idx) ? 1 : 0;
		icon_ofs.x -= hsep;
		icon_ofs.x -= wrap_icon[wrap_type]->get_width();
		te->draw_texture(wrap_icon[wrap_type], icon_ofs);

		icon_ofs.x -= hsep;
		te->draw_line(Point2(icon_ofs.x, ofs.y + y), Point2(icon_ofs.x, ofs.y + y + h), sepcolor);

		track_ofs[1] = size.width - icon_ofs.x;

		icon_ofs.x -= down_icon->get_width();
		te->draw_texture(down_icon, icon_ofs - Size2(0, 4 * EDSCALE));

		int interp_type = animation->track_get_interpolation_type(idx);
		ERR_CONTINUE(interp_type < 0 || interp_type >= 3);
		icon_ofs.x -= hsep;
		icon_ofs.x -= interp_icon[interp_type]->get_width();
		te->draw_texture(interp_icon[interp_type], icon_ofs);

		icon_ofs.x -= hsep;
		te->draw_line(Point2(icon_ofs.x, ofs.y + y), Point2(icon_ofs.x, ofs.y + y + h), sepcolor);

		track_ofs[2] = size.width - icon_ofs.x;

		if (animation->track_get_type(idx) == Animation::TYPE_VALUE) {

			int umode = animation->value_track_get_update_mode(idx);

			icon_ofs.x -= hsep;
			icon_ofs.x -= down_icon->get_width();
			te->draw_texture(down_icon, icon_ofs - Size2(0, 4 * EDSCALE));

			icon_ofs.x -= hsep;
			icon_ofs.x -= cont_icon[umode]->get_width();
			te->draw_texture(cont_icon[umode], icon_ofs);
		} else {

			icon_ofs.x -= hsep * 2 + cont_icon[0]->get_width() + down_icon->get_width();
		}

		icon_ofs.x -= hsep;
		te->draw_line(Point2(icon_ofs.x, ofs.y + y), Point2(icon_ofs.x, ofs.y + y + h), sepcolor);

		track_ofs[3] = size.width - icon_ofs.x;

		icon_ofs.x -= hsep;
		icon_ofs.x -= add_key_icon->get_width();
		te->draw_texture((mouse_over.over == MouseOver::OVER_ADD_KEY && mouse_over.track == idx) ? add_key_icon_hl : add_key_icon, icon_ofs);

		track_ofs[4] = size.width - icon_ofs.x;

		//draw the keys;
		int tt = animation->track_get_type(idx);
		float key_vofs = Math::floor((float)(h - type_icon[tt]->get_height()) / 2);
		float key_hofs = -Math::floor((float)type_icon[tt]->get_height() / 2);

		int kc = animation->track_get_key_count(idx);
		bool first = true;

		for (int i = 0; i < kc; i++) {

			float time = animation->track_get_key_time(idx, i);
			if (time < keys_from)
				continue;
			if (time > keys_to) {

				if (first && i > 0 && animation->track_get_key_value(idx, i) == animation->track_get_key_value(idx, i - 1)) {
					//draw whole line
					te->draw_line(ofs + Vector2(name_limit, y + h / 2), ofs + Point2(settings_limit, y + h / 2), color);
				}

				break;
			}

			float x = key_hofs + name_limit + (time - keys_from) * zoom_scale;

			Ref<Texture> tex = type_icon[tt];

			SelectedKey sk;
			sk.key = i;
			sk.track = idx;
			if (selection.has(sk)) {

				if (click.click == ClickOver::CLICK_MOVE_KEYS)
					continue;
				tex = type_selected;
			}

			if (mouse_over.over == MouseOver::OVER_KEY && mouse_over.track == idx && mouse_over.over_key == i)
				tex = type_hover;

			Variant value = animation->track_get_key_value(idx, i);
			if (first && i > 0 && value == animation->track_get_key_value(idx, i - 1)) {

				te->draw_line(ofs + Vector2(name_limit, y + h / 2), ofs + Point2(x, y + h / 2), color);
			}

			if (i < kc - 1 && value == animation->track_get_key_value(idx, i + 1)) {
				float x_n = key_hofs + name_limit + (animation->track_get_key_time(idx, i + 1) - keys_from) * zoom_scale;

				x_n = MIN(x_n, settings_limit);
				te->draw_line(ofs + Point2(x_n, y + h / 2), ofs + Point2(x, y + h / 2), color);
			}

			if (prop_exists && !Variant::can_convert(value.get_type(), valid_type)) {
				te->draw_texture(invalid_icon, ofs + Point2(x, y + key_vofs).floor());
			}

			if (prop_exists && !Variant::can_convert(value.get_type(), valid_type)) {
				if (tex == type_hover)
					te->draw_texture(invalid_icon_hover, ofs + Point2(x, y + key_vofs).floor());
				else
					te->draw_texture(invalid_icon, ofs + Point2(x, y + key_vofs).floor());
			} else {

				te->draw_texture(tex, ofs + Point2(x, y + key_vofs).floor());
			}

			first = false;
		}
	}

	switch (click.click) {
		case ClickOver::CLICK_SELECT_KEYS: {

			Color box_color = get_color("accent_color", "Editor");
			box_color.a = 0.35;
			te->draw_rect(Rect2(click.at, click.to - click.at), box_color);

		} break;
		case ClickOver::CLICK_MOVE_KEYS: {

			float from_t = 1e20;

			for (Map<SelectedKey, KeyInfo>::Element *E = selection.front(); E; E = E->next()) {
				float t = animation->track_get_key_time(E->key().track, E->key().key);
				if (t < from_t)
					from_t = t;
			}

			float motion = from_t + (click.to.x - click.at.x) / zoom_scale;
			if (step->get_value())
				motion = Math::stepify(motion, step->get_value());

			for (Map<SelectedKey, KeyInfo>::Element *E = selection.front(); E; E = E->next()) {

				int idx = E->key().track;
				int i = idx - v_scroll->get_value();
				if (i < 0 || i >= fit)
					continue;
				int y = h + i * h + sep;

				float key_vofs = Math::floor((float)(h - type_selected->get_height()) / 2);
				float key_hofs = -Math::floor((float)type_selected->get_height() / 2);

				float time = animation->track_get_key_time(idx, E->key().key);
				float diff = time - from_t;

				float t = motion + diff;

				float x = (t - keys_from) * zoom_scale;
				//x+=click.to.x - click.at.x;
				if (x < 0 || x >= (settings_limit - name_limit))
					continue;

				x += name_limit;

				te->draw_texture(type_selected, ofs + Point2(x + key_hofs, y + key_vofs).floor());
			}
		} break;
		default: {};
	}

	te_drawing = false;
}

void AnimationKeyEditor::_track_name_changed(const String &p_name) {

	ERR_FAIL_COND(!animation.is_valid());
	undo_redo->create_action(TTR("Anim Track Rename"));
	undo_redo->add_do_method(animation.ptr(), "track_set_path", track_name_editing, p_name);
	undo_redo->add_undo_method(animation.ptr(), "track_set_path", track_name_editing, animation->track_get_path(track_name_editing));
	undo_redo->commit_action();
	track_name->hide();
}

void AnimationKeyEditor::_track_menu_selected(int p_idx) {

	ERR_FAIL_COND(!animation.is_valid());

	if (interp_editing != -1) {

		ERR_FAIL_INDEX(interp_editing, animation->get_track_count());
		undo_redo->create_action(TTR("Anim Track Change Interpolation"));
		undo_redo->add_do_method(animation.ptr(), "track_set_interpolation_type", interp_editing, p_idx);
		undo_redo->add_undo_method(animation.ptr(), "track_set_interpolation_type", interp_editing, animation->track_get_interpolation_type(interp_editing));
		undo_redo->commit_action();
	} else if (cont_editing != -1) {

		ERR_FAIL_INDEX(cont_editing, animation->get_track_count());

		undo_redo->create_action(TTR("Anim Track Change Value Mode"));
		undo_redo->add_do_method(animation.ptr(), "value_track_set_update_mode", cont_editing, p_idx);
		undo_redo->add_undo_method(animation.ptr(), "value_track_set_update_mode", cont_editing, animation->value_track_get_update_mode(cont_editing));
		undo_redo->commit_action();
	} else if (wrap_editing != -1) {

		ERR_FAIL_INDEX(wrap_editing, animation->get_track_count());

		undo_redo->create_action(TTR("Anim Track Change Wrap Mode"));
		undo_redo->add_do_method(animation.ptr(), "track_set_interpolation_loop_wrap", wrap_editing, p_idx ? true : false);
		undo_redo->add_undo_method(animation.ptr(), "track_set_interpolation_loop_wrap", wrap_editing, animation->track_get_interpolation_loop_wrap(wrap_editing));
		undo_redo->commit_action();
	} else {
		switch (p_idx) {

			case RIGHT_MENU_DUPLICATE:
				_anim_duplicate_keys();
				break;
			case RIGHT_MENU_DUPLICATE_TRANSPOSE:
				_anim_duplicate_keys(true);
				break;
			case RIGHT_MENU_REMOVE:
				_anim_delete_keys();
				break;
		}
	}
}

struct _AnimMoveRestore {

	int track;
	float time;
	Variant key;
	float transition;
};

void AnimationKeyEditor::_clear_selection_for_anim(const Ref<Animation> &p_anim) {

	if (!(animation == p_anim))
		return;
	//selection.clear();
	_clear_selection();
}

void AnimationKeyEditor::_select_at_anim(const Ref<Animation> &p_anim, int p_track, float p_pos) {

	if (!(animation == p_anim))
		return;

	int idx = animation->track_find_key(p_track, p_pos, true);
	ERR_FAIL_COND(idx < 0);

	SelectedKey sk;
	sk.track = p_track;
	sk.key = idx;
	KeyInfo ki;
	ki.pos = p_pos;

	selection.insert(sk, ki);
}

PropertyInfo AnimationKeyEditor::_find_hint_for_track(int p_idx, NodePath &r_base_path) {

	r_base_path = NodePath();
	ERR_FAIL_COND_V(!animation.is_valid(), PropertyInfo());
	ERR_FAIL_INDEX_V(p_idx, animation->get_track_count(), PropertyInfo());

	if (!root)
		return PropertyInfo();

	NodePath path = animation->track_get_path(p_idx);

	if (!root->has_node_and_resource(path))
		return PropertyInfo();

	RES res;
	Node *node = root->get_node_and_resource(path, res);

	if (node) {
		r_base_path = node->get_path();
	}

	String property = path.get_property();
	if (property == "")
		return PropertyInfo();

	List<PropertyInfo> pinfo;
	if (res.is_valid())
		res->get_property_list(&pinfo);
	else if (node)
		node->get_property_list(&pinfo);

	for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {

		if (E->get().name == property)
			return E->get();
	}

	return PropertyInfo();
}

void AnimationKeyEditor::_curve_transition_changed(float p_what) {

	if (selection.size() == 0)
		return;
	if (selection.size() == 1)
		undo_redo->create_action(TTR("Edit Node Curve"), UndoRedo::MERGE_ENDS);
	else
		undo_redo->create_action(TTR("Edit Selection Curve"), UndoRedo::MERGE_ENDS);

	for (Map<SelectedKey, KeyInfo>::Element *E = selection.front(); E; E = E->next()) {

		int track = E->key().track;
		int key = E->key().key;
		float prev_val = animation->track_get_key_transition(track, key);
		undo_redo->add_do_method(animation.ptr(), "track_set_key_transition", track, key, p_what);
		undo_redo->add_undo_method(animation.ptr(), "track_set_key_transition", track, key, prev_val);
	}

	undo_redo->commit_action();
}

void AnimationKeyEditor::_toggle_edit_curves() {

	if (edit_button->is_pressed())
		key_editor_tab->show();
	else
		key_editor_tab->hide();
}

bool AnimationKeyEditor::_edit_if_single_selection() {

	if (selection.size() != 1) {

		if (selection.size() == 0) {
			curve_edit->set_mode(AnimationCurveEdit::MODE_DISABLED);
			//print_line("disable");
		} else {

			curve_edit->set_mode(AnimationCurveEdit::MODE_MULTIPLE);
			curve_edit->set_transition(1.0);
			curve_edit->clear_multiples();
			//add all
			for (Map<SelectedKey, KeyInfo>::Element *E = selection.front(); E; E = E->next()) {

				curve_edit->set_multiple(animation->track_get_key_transition(E->key().track, E->key().key));
			}
			//print_line("multiple");
		}
		return false;
	}
	curve_edit->set_mode(AnimationCurveEdit::MODE_SINGLE);
	//print_line("regular");

	int idx = selection.front()->key().track;
	int key = selection.front()->key().key;
	{

		key_edit->animation = animation;
		key_edit->track = idx;
		key_edit->key_ofs = animation->track_get_key_time(idx, key);
		key_edit->hint = _find_hint_for_track(idx, key_edit->base);
		key_edit->notify_change();

		curve_edit->set_transition(animation->track_get_key_transition(idx, key));

		/*key_edit_dialog->set_size( Size2( 200,200) );
		key_edit_dialog->set_position(  track_editor->get_global_position() + ofs + mpos +Point2(-100,20));
		key_edit_dialog->popup();*/
	}

	return true;
}

void AnimationKeyEditor::_anim_delete_keys() {
	if (selection.size()) {
		undo_redo->create_action(TTR("Anim Delete Keys"));

		for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

			undo_redo->add_do_method(animation.ptr(), "track_remove_key", E->key().track, E->key().key);
			undo_redo->add_undo_method(animation.ptr(), "track_insert_key", E->key().track, E->get().pos, animation->track_get_key_value(E->key().track, E->key().key), animation->track_get_key_transition(E->key().track, E->key().key));
		}
		undo_redo->add_do_method(this, "_clear_selection_for_anim", animation);
		undo_redo->add_undo_method(this, "_clear_selection_for_anim", animation);
		undo_redo->commit_action();
		//selection.clear();
		accept_event();
		_edit_if_single_selection();
	}
}

void AnimationKeyEditor::_track_editor_gui_input(const Ref<InputEvent> &p_input) {

	Control *te = track_editor;
	Ref<StyleBox> style = get_stylebox("normal", "TextEdit");

	if (!animation.is_valid()) {
		return;
	}

	Size2 size = te->get_size() - style->get_minimum_size();
	Size2 ofs = style->get_offset();

	Ref<Font> font = te->get_font("font", "Tree");
	int sep = get_constant("vseparation", "Tree");
	int hsep = get_constant("hseparation", "Tree");
	Ref<Texture> remove_icon = get_icon("Remove", "EditorIcons");
	Ref<Texture> move_up_icon = get_icon("MoveUp", "EditorIcons");
	Ref<Texture> move_down_icon = get_icon("MoveDown", "EditorIcons");
	Ref<Texture> down_icon = get_icon("select_arrow", "Tree");
	Ref<Texture> hsize_icon = get_icon("Hsize", "EditorIcons");
	Ref<Texture> add_key_icon = get_icon("TrackAddKey", "EditorIcons");

	Ref<Texture> wrap_icon[2] = {
		get_icon("InterpWrapClamp", "EditorIcons"),
		get_icon("InterpWrapLoop", "EditorIcons"),
	};
	Ref<Texture> interp_icon[3] = {
		get_icon("InterpRaw", "EditorIcons"),
		get_icon("InterpLinear", "EditorIcons"),
		get_icon("InterpCubic", "EditorIcons")
	};
	Ref<Texture> cont_icon[3] = {
		get_icon("TrackContinuous", "EditorIcons"),
		get_icon("TrackDiscrete", "EditorIcons"),
		get_icon("TrackTrigger", "EditorIcons")
	};
	Ref<Texture> type_icon[3] = {
		get_icon("KeyValue", "EditorIcons"),
		get_icon("KeyXform", "EditorIcons"),
		get_icon("KeyCall", "EditorIcons")
	};
	int right_separator_ofs = down_icon->get_width() * 3 + add_key_icon->get_width() + interp_icon[0]->get_width() + wrap_icon[0]->get_width() + cont_icon[0]->get_width() + hsep * 9;

	int h = font->get_height() + sep;

	int fit = (size.height / h) - 1;
	int total = animation->get_track_count();
	if (total < fit) {
		v_scroll->hide();
	} else {
		v_scroll->show();
		v_scroll->set_max(total);
		v_scroll->set_page(fit);
	}

	int settings_limit = size.width - right_separator_ofs;
	int name_limit = settings_limit * name_column_ratio;

	Ref<InputEventKey> key = p_input;
	if (key.is_valid()) {

		if (key->get_scancode() == KEY_D && key->is_pressed() && key->get_command()) {

			if (key->get_shift())
				_menu_track(TRACK_MENU_DUPLICATE_TRANSPOSE);
			else
				_menu_track(TRACK_MENU_DUPLICATE);

			accept_event();

		} else if (key->get_scancode() == KEY_DELETE && key->is_pressed() && click.click == ClickOver::CLICK_NONE) {

			_anim_delete_keys();
		} else if (animation.is_valid() && animation->get_track_count() > 0) {

			if (key->is_pressed() && (key->is_action("ui_up") || key->is_action("ui_page_up"))) {

				if (key->is_action("ui_up"))
					selected_track--;
				if (v_scroll->is_visible_in_tree() && key->is_action("ui_page_up"))
					selected_track--;

				if (selected_track < 0)
					selected_track = 0;

				if (v_scroll->is_visible_in_tree()) {
					if (v_scroll->get_value() > selected_track)
						v_scroll->set_value(selected_track);
				}

				track_editor->update();
				accept_event();
			}

			if (key->is_pressed() && (key->is_action("ui_down") || key->is_action("ui_page_down"))) {

				if (key->is_action("ui_down"))
					selected_track++;
				else if (v_scroll->is_visible_in_tree() && key->is_action("ui_page_down"))
					selected_track += v_scroll->get_page();

				if (selected_track >= animation->get_track_count())
					selected_track = animation->get_track_count() - 1;

				if (v_scroll->is_visible_in_tree() && v_scroll->get_page() + v_scroll->get_value() < selected_track + 1) {
					v_scroll->set_value(selected_track - v_scroll->get_page() + 1);
				}

				track_editor->update();
				accept_event();
			}
		}
	}

	Ref<InputEventMouseButton> mb = p_input;

	if (mb.is_valid()) {

		if (mb->get_button_index() == BUTTON_WHEEL_UP && mb->is_pressed()) {

			if (mb->get_command()) {

				zoom->set_value(zoom->get_value() + zoom->get_step());
			} else {

				v_scroll->set_value(v_scroll->get_value() - v_scroll->get_page() * mb->get_factor() / 8);
			}
		}

		if (mb->get_button_index() == BUTTON_WHEEL_DOWN && mb->is_pressed()) {

			if (mb->get_command()) {

				zoom->set_value(zoom->get_value() - zoom->get_step());
			} else {

				v_scroll->set_value(v_scroll->get_value() + v_scroll->get_page() * mb->get_factor() / 8);
			}
		}

		if (mb->get_button_index() == BUTTON_WHEEL_RIGHT && mb->is_pressed()) {

			h_scroll->set_value(h_scroll->get_value() - h_scroll->get_page() * mb->get_factor() / 8);
		}

		if (mb->get_button_index() == BUTTON_WHEEL_LEFT && mb->is_pressed()) {

			v_scroll->set_value(v_scroll->get_value() + v_scroll->get_page() * mb->get_factor() / 8);
		}

		if (mb->get_button_index() == BUTTON_RIGHT && mb->is_pressed()) {

			Point2 mpos = mb->get_position() - ofs;

			if (selection.size() == 0) {
				// Auto-select on right-click if nothing is selected
				// Note: This code is pretty much duplicated from the left click code,
				// both codes could be moved into a function to avoid the duplicated code.
				Point2 mpos = mb->get_position() - ofs;

				if (mpos.y < h) {
					return;
				}

				mpos.y -= h;

				int idx = mpos.y / h;
				idx += v_scroll->get_value();
				if (idx < 0 || idx >= animation->get_track_count())
					return;

				if (mpos.x < name_limit) {
				} else if (mpos.x < settings_limit) {
					float pos = mpos.x - name_limit;
					pos /= _get_zoom_scale();
					pos += h_scroll->get_value();
					float w_time = (type_icon[0]->get_width() / _get_zoom_scale()) / 2.0;

					int kidx = animation->track_find_key(idx, pos);
					int kidx_n = kidx + 1;
					int key = -1;

					if (kidx >= 0 && kidx < animation->track_get_key_count(idx)) {

						float kpos = animation->track_get_key_time(idx, kidx);
						if (ABS(pos - kpos) <= w_time) {

							key = kidx;
						}
					}

					if (key == -1 && kidx_n >= 0 && kidx_n < animation->track_get_key_count(idx)) {

						float kpos = animation->track_get_key_time(idx, kidx_n);
						if (ABS(pos - kpos) <= w_time) {

							key = kidx_n;
						}
					}

					if (key == -1) {

						click.click = ClickOver::CLICK_SELECT_KEYS;
						click.at = mb->get_position();
						click.to = click.at;
						click.shift = mb->get_shift();
						selected_track = idx;
						track_editor->update();
						//drag select region
						return;
					}

					SelectedKey sk;
					sk.track = idx;
					sk.key = key;
					KeyInfo ki;
					ki.pos = animation->track_get_key_time(idx, key);
					click.shift = mb->get_shift();
					click.selk = sk;

					if (!mb->get_shift() && !selection.has(sk))
						_clear_selection();

					selection.insert(sk, ki);

					click.click = ClickOver::CLICK_MOVE_KEYS;
					click.at = mb->get_position();
					click.to = click.at;
					update();
					selected_track = idx;
					track_editor->update();

					if (_edit_if_single_selection() && mb->get_command()) {
						edit_button->set_pressed(true);
						key_editor_tab->show();
					}
				}
			}

			if (selection.size()) {
				// User has right clicked and we have a selection, show a popup menu with options
				track_menu->clear();
				track_menu->set_size(Point2(1, 1));
				track_menu->add_item(TTR("Duplicate Selection"), RIGHT_MENU_DUPLICATE);
				track_menu->add_item(TTR("Duplicate Transposed"), RIGHT_MENU_DUPLICATE_TRANSPOSE);
				track_menu->add_item(TTR("Remove Selection"), RIGHT_MENU_REMOVE);

				track_menu->set_position(te->get_global_position() + mpos);

				interp_editing = -1;
				cont_editing = -1;
				wrap_editing = -1;

				track_menu->popup();
			}
		}

		if (mb->get_button_index() == BUTTON_LEFT && !(mb->get_button_mask() & ~BUTTON_MASK_LEFT)) {

			if (mb->is_pressed()) {

				Point2 mpos = mb->get_position() - ofs;

				if (mpos.y < h) {

					if (mpos.x < name_limit && mpos.x > (name_limit - hsep - hsize_icon->get_width())) {

						click.click = ClickOver::CLICK_RESIZE_NAMES;
						click.at = mb->get_position();
						click.to = click.at;
						click.at.y = name_limit;
					}

					if (mpos.x >= name_limit && mpos.x < settings_limit) {
						//seek
						//int zoomw = settings_limit-name_limit;
						float scale = _get_zoom_scale();
						float pos = h_scroll->get_value() + (mpos.x - name_limit) / scale;
						if (animation->get_step())
							pos = Math::stepify(pos, animation->get_step());

						if (pos < 0)
							pos = 0;
						if (pos >= animation->get_length())
							pos = animation->get_length();
						timeline_pos = pos;
						click.click = ClickOver::CLICK_DRAG_TIMELINE;
						click.at = mb->get_position();
						click.to = click.at;
						emit_signal("timeline_changed", pos, false);
					}

					return;
				}

				mpos.y -= h;

				int idx = mpos.y / h;
				idx += v_scroll->get_value();
				if (idx < 0)
					return;

				if (idx >= animation->get_track_count()) {

					if (mpos.x >= name_limit && mpos.x < settings_limit) {

						click.click = ClickOver::CLICK_SELECT_KEYS;
						click.at = mb->get_position();
						click.to = click.at;
						//drag select region
					}

					return;
				}

				if (mpos.x < name_limit) {
					//name column

					// area
					if (idx != selected_track) {

						selected_track = idx;
						track_editor->update();
						return;
					}

					Rect2 area(ofs.x, ofs.y + ((int(mpos.y) / h) + 1) * h, name_limit, h);
					track_name->set_text(animation->track_get_path(idx));
					track_name->set_position(te->get_global_position() + area.position);
					track_name->set_size(area.size);
					track_name->show_modal();
					track_name->grab_focus();
					track_name->select_all();
					track_name_editing = idx;

				} else if (mpos.x < settings_limit) {

					float pos = mpos.x - name_limit;
					pos /= _get_zoom_scale();
					pos += h_scroll->get_value();
					float w_time = (type_icon[0]->get_width() / _get_zoom_scale()) / 2.0;

					int kidx = animation->track_find_key(idx, pos);
					int kidx_n = kidx + 1;
					int key = -1;

					if (kidx >= 0 && kidx < animation->track_get_key_count(idx)) {

						float kpos = animation->track_get_key_time(idx, kidx);
						if (ABS(pos - kpos) <= w_time) {

							key = kidx;
						}
					}

					if (key == -1 && kidx_n >= 0 && kidx_n < animation->track_get_key_count(idx)) {

						float kpos = animation->track_get_key_time(idx, kidx_n);
						if (ABS(pos - kpos) <= w_time) {

							key = kidx_n;
						}
					}

					if (key == -1) {

						click.click = ClickOver::CLICK_SELECT_KEYS;
						click.at = mb->get_position();
						click.to = click.at;
						click.shift = mb->get_shift();
						selected_track = idx;
						track_editor->update();
						//drag select region
						return;
					}

					SelectedKey sk;
					sk.track = idx;
					sk.key = key;
					KeyInfo ki;
					ki.pos = animation->track_get_key_time(idx, key);
					click.shift = mb->get_shift();
					click.selk = sk;

					if (!mb->get_shift() && !selection.has(sk))
						_clear_selection();

					selection.insert(sk, ki);

					click.click = ClickOver::CLICK_MOVE_KEYS;
					click.at = mb->get_position();
					click.to = click.at;
					update();
					selected_track = idx;
					track_editor->update();

					if (_edit_if_single_selection() && mb->get_command()) {
						edit_button->set_pressed(true);
						key_editor_tab->show();
					}
				} else {
					//button column
					int ofsx = size.width - mpos.x;
					if (ofsx < 0)
						return;
					/*
					if (ofsx < remove_icon->get_width()) {

						undo_redo->create_action("Remove Anim Track");
						undo_redo->add_do_method(animation.ptr(),"remove_track",idx);
						undo_redo->add_undo_method(animation.ptr(),"add_track",animation->track_get_type(idx),idx);
						undo_redo->add_undo_method(animation.ptr(),"track_set_path",idx,animation->track_get_path(idx));
						//todo interpolation
						for(int i=0;i<animation->track_get_key_count(idx);i++) {

							Variant v = animation->track_get_key_value(idx,i);
							float time =  animation->track_get_key_time(idx,i);
							float trans =  animation->track_get_key_transition(idx,i);

							undo_redo->add_undo_method(animation.ptr(),"track_insert_key",idx,time,v);
							undo_redo->add_undo_method(animation.ptr(),"track_set_key_transition",idx,i,trans);

						}

						undo_redo->add_undo_method(animation.ptr(),"track_set_interpolation_type",idx,animation->track_get_interpolation_type(idx));
						if (animation->track_get_type(idx)==Animation::TYPE_VALUE) {
							undo_redo->add_undo_method(animation.ptr(),"value_track_set_continuous",idx,animation->value_track_is_continuous(idx));

						}

						undo_redo->commit_action();


						return;
					}

					ofsx-=hsep+remove_icon->get_width();

					if (ofsx < move_down_icon->get_width()) {

						if (idx < animation->get_track_count() -1) {
							undo_redo->create_action("Move Anim Track Down");
							undo_redo->add_do_method(animation.ptr(),"track_move_up",idx);
							undo_redo->add_undo_method(animation.ptr(),"track_move_down",idx+1);
							undo_redo->commit_action();
						}
						return;
					}

					ofsx-=hsep+move_down_icon->get_width();

					if (ofsx < move_up_icon->get_width()) {

						if (idx >0) {
							undo_redo->create_action("Move Anim Track Up");
							undo_redo->add_do_method(animation.ptr(),"track_move_down",idx);
							undo_redo->add_undo_method(animation.ptr(),"track_move_up",idx-1);
							undo_redo->commit_action();
						}
						return;
					}


					ofsx-=hsep*3+move_up_icon->get_width();
					*/

					if (ofsx < track_ofs[1]) {

						track_menu->clear();
						track_menu->set_size(Point2(1, 1));
						static const char *interp_name[2] = { "Clamp Loop Interp", "Wrap Loop Interp" };
						for (int i = 0; i < 2; i++) {
							track_menu->add_icon_item(wrap_icon[i], interp_name[i]);
						}

						int popup_y = ofs.y + ((int(mpos.y) / h) + 2) * h;
						int popup_x = size.width - track_ofs[1];

						track_menu->set_position(te->get_global_position() + Point2(popup_x, popup_y));

						wrap_editing = idx;
						interp_editing = -1;
						cont_editing = -1;

						track_menu->popup();

						return;
					}

					if (ofsx < track_ofs[2]) {

						track_menu->clear();
						track_menu->set_size(Point2(1, 1));
						static const char *interp_name[3] = { "Nearest", "Linear", "Cubic" };
						for (int i = 0; i < 3; i++) {
							track_menu->add_icon_item(interp_icon[i], interp_name[i]);
						}

						int popup_y = ofs.y + ((int(mpos.y) / h) + 2) * h;
						int popup_x = size.width - track_ofs[2];

						track_menu->set_position(te->get_global_position() + Point2(popup_x, popup_y));

						interp_editing = idx;
						cont_editing = -1;
						wrap_editing = -1;

						track_menu->popup();

						return;
					}

					if (ofsx < track_ofs[3]) {

						track_menu->clear();
						track_menu->set_size(Point2(1, 1));
						String cont_name[3] = { TTR("Continuous"), TTR("Discrete"), TTR("Trigger") };
						for (int i = 0; i < 3; i++) {
							track_menu->add_icon_item(cont_icon[i], cont_name[i]);
						}

						int popup_y = ofs.y + ((int(mpos.y) / h) + 2) * h;
						int popup_x = size.width - track_ofs[3];

						track_menu->set_position(te->get_global_position() + Point2(popup_x, popup_y));

						interp_editing = -1;
						wrap_editing = -1;
						cont_editing = idx;

						track_menu->popup();

						return;
					}

					if (ofsx < track_ofs[4]) {

						Animation::TrackType tt = animation->track_get_type(idx);

						float pos = timeline_pos;
						int existing = animation->track_find_key(idx, pos, true);

						Variant newval;

						if (tt == Animation::TYPE_TRANSFORM) {
							Dictionary d;
							d["loc"] = Vector3();
							d["rot"] = Quat();
							d["scale"] = Vector3();
							newval = d;

						} else if (tt == Animation::TYPE_METHOD) {

							Dictionary d;
							d["method"] = "";
							d["args"] = Vector<Variant>();

							newval = d;
						} else if (tt == Animation::TYPE_VALUE) {

							NodePath np;
							PropertyInfo inf = _find_hint_for_track(idx, np);
							if (inf.type != Variant::NIL) {

								Variant::CallError err;
								newval = Variant::construct(inf.type, NULL, 0, err);
							}

							if (newval.get_type() == Variant::NIL) {
								//popup a new type
								cvi_track = idx;
								cvi_pos = pos;

								type_menu->set_position(get_global_position() + mpos + ofs);
								type_menu->popup();
								return;
							}
						}

						undo_redo->create_action(TTR("Anim Add Key"));

						undo_redo->add_do_method(animation.ptr(), "track_insert_key", idx, pos, newval, 1);
						undo_redo->add_undo_method(animation.ptr(), "track_remove_key_at_pos", idx, pos);

						if (existing != -1) {
							Variant v = animation->track_get_key_value(idx, existing);
							float trans = animation->track_get_key_transition(idx, existing);
							undo_redo->add_undo_method(animation.ptr(), "track_insert_key", idx, pos, v, trans);
						}

						undo_redo->commit_action();

						return;
					}
				}

			} else {

				switch (click.click) {
					case ClickOver::CLICK_SELECT_KEYS: {

						float zoom_scale = _get_zoom_scale();
						float keys_from = h_scroll->get_value();
						float keys_to = keys_from + (settings_limit - name_limit) / zoom_scale;

						float from_time = keys_from + (click.at.x - (name_limit + ofs.x)) / zoom_scale;
						float to_time = keys_from + (click.to.x - (name_limit + ofs.x)) / zoom_scale;

						if (to_time < from_time)
							SWAP(from_time, to_time);

						if (from_time > keys_to || to_time < keys_from)
							break;

						if (from_time < keys_from)
							from_time = keys_from;

						if (to_time >= keys_to)
							to_time = keys_to;

						int from_track = int(click.at.y - ofs.y - h - sep) / h + v_scroll->get_value();
						int to_track = int(click.to.y - ofs.y - h - sep) / h + v_scroll->get_value();
						int from_mod = int(click.at.y - ofs.y - sep) % h;
						int to_mod = int(click.to.y - ofs.y - sep) % h;

						if (to_track < from_track) {

							SWAP(from_track, to_track);
							SWAP(from_mod, to_mod);
						}

						if ((from_mod > (h / 2)) && ((click.at.y - ofs.y) >= (h + sep))) {
							from_track++;
						}

						if (to_mod < h / 2) {
							to_track--;
						}

						if (from_track > to_track) {
							if (!click.shift)
								_clear_selection();
							_edit_if_single_selection();
							break;
						}

						int tracks_from = v_scroll->get_value();
						int tracks_to = v_scroll->get_value() + fit - 1;
						if (tracks_to >= animation->get_track_count())
							tracks_to = animation->get_track_count() - 1;

						tracks_from = 0;
						tracks_to = animation->get_track_count() - 1;
						if (to_track > tracks_to)
							to_track = tracks_to;
						if (from_track < tracks_from)
							from_track = tracks_from;

						if (from_track > tracks_to || to_track < tracks_from) {
							if (!click.shift)
								_clear_selection();
							_edit_if_single_selection();
							break;
						}

						if (!click.shift)
							_clear_selection();

						int higher_track = 0x7FFFFFFF;
						for (int i = from_track; i <= to_track; i++) {

							int kc = animation->track_get_key_count(i);
							for (int j = 0; j < kc; j++) {

								float t = animation->track_get_key_time(i, j);
								if (t < from_time)
									continue;
								if (t > to_time)
									break;

								if (i < higher_track)
									higher_track = i;

								SelectedKey sk;
								sk.track = i;
								sk.key = j;
								KeyInfo ki;
								ki.pos = t;
								selection[sk] = ki;
							}
						}

						if (higher_track != 0x7FFFFFFF) {
							selected_track = higher_track;
							track_editor->update();
						}

						_edit_if_single_selection();

					} break;
					case ClickOver::CLICK_MOVE_KEYS: {

						if (selection.empty())
							break;
						if (click.at == click.to) {

							if (!click.shift) {

								KeyInfo ki = selection[click.selk];
								_clear_selection();
								selection[click.selk] = ki;
								_edit_if_single_selection();
							}

							break;
						}

						float from_t = 1e20;

						for (Map<SelectedKey, KeyInfo>::Element *E = selection.front(); E; E = E->next()) {
							float t = animation->track_get_key_time(E->key().track, E->key().key);
							if (t < from_t)
								from_t = t;
						}

						float motion = from_t + (click.to.x - click.at.x) / _get_zoom_scale();
						if (step->get_value())
							motion = Math::stepify(motion, step->get_value());

						undo_redo->create_action(TTR("Anim Move Keys"));

						List<_AnimMoveRestore> to_restore;

						// 1-remove the keys
						for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

							undo_redo->add_do_method(animation.ptr(), "track_remove_key", E->key().track, E->key().key);
						}
						// 2- remove overlapped keys
						for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

							float newtime = E->get().pos - from_t + motion;
							int idx = animation->track_find_key(E->key().track, newtime, true);
							if (idx == -1)
								continue;
							SelectedKey sk;
							sk.key = idx;
							sk.track = E->key().track;
							if (selection.has(sk))
								continue; //already in selection, don't save

							undo_redo->add_do_method(animation.ptr(), "track_remove_key_at_pos", E->key().track, newtime);
							_AnimMoveRestore amr;

							amr.key = animation->track_get_key_value(E->key().track, idx);
							amr.track = E->key().track;
							amr.time = newtime;
							amr.transition = animation->track_get_key_transition(E->key().track, idx);

							to_restore.push_back(amr);
						}

						// 3-move the keys (re insert them)
						for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

							float newpos = E->get().pos - from_t + motion;
							/*
							if (newpos<0)
								continue; //no add at the beginning
							*/
							undo_redo->add_do_method(animation.ptr(), "track_insert_key", E->key().track, newpos, animation->track_get_key_value(E->key().track, E->key().key), animation->track_get_key_transition(E->key().track, E->key().key));
						}

						// 4-(undo) remove inserted keys
						for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

							float newpos = E->get().pos + -from_t + motion;
							/*
							if (newpos<0)
								continue; //no remove what no inserted
							*/
							undo_redo->add_undo_method(animation.ptr(), "track_remove_key_at_pos", E->key().track, newpos);
						}

						// 5-(undo) reinsert keys
						for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

							undo_redo->add_undo_method(animation.ptr(), "track_insert_key", E->key().track, E->get().pos, animation->track_get_key_value(E->key().track, E->key().key), animation->track_get_key_transition(E->key().track, E->key().key));
						}

						// 6-(undo) reinsert overlapped keys
						for (List<_AnimMoveRestore>::Element *E = to_restore.front(); E; E = E->next()) {

							_AnimMoveRestore &amr = E->get();
							undo_redo->add_undo_method(animation.ptr(), "track_insert_key", amr.track, amr.time, amr.key, amr.transition);
						}

						// 6-(undo) reinsert overlapped keys
						for (List<_AnimMoveRestore>::Element *E = to_restore.front(); E; E = E->next()) {

							_AnimMoveRestore &amr = E->get();
							undo_redo->add_undo_method(animation.ptr(), "track_insert_key", amr.track, amr.time, amr.key, amr.transition);
						}

						undo_redo->add_do_method(this, "_clear_selection_for_anim", animation);
						undo_redo->add_undo_method(this, "_clear_selection_for_anim", animation);

						// 7-reselect

						for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

							float oldpos = E->get().pos;
							float newpos = oldpos - from_t + motion;
							//if (newpos>=0)
							undo_redo->add_do_method(this, "_select_at_anim", animation, E->key().track, newpos);
							undo_redo->add_undo_method(this, "_select_at_anim", animation, E->key().track, oldpos);
						}

						undo_redo->commit_action();
						_edit_if_single_selection();

					} break;
					default: {}
				}

				//button released
				click.click = ClickOver::CLICK_NONE;
				track_editor->update();
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_input;

	if (mm.is_valid()) {

		mouse_over.over = MouseOver::OVER_NONE;
		mouse_over.track = -1;
		te->update();
		track_editor->set_tooltip("");

		if (!track_editor->has_focus() && (!get_focus_owner() || !get_focus_owner()->is_text_field()))
			track_editor->call_deferred("grab_focus");

		if (click.click != ClickOver::CLICK_NONE) {

			switch (click.click) {
				case ClickOver::CLICK_RESIZE_NAMES: {

					float base = click.at.y;
					float clickp = click.at.x - ofs.x;
					float dif = base - clickp;

					float target = mm->get_position().x + dif - ofs.x;

					float ratio = target / settings_limit;

					if (ratio > 0.9)
						ratio = 0.9;
					else if (ratio < 0.2)
						ratio = 0.2;

					name_column_ratio = ratio;

				} break;
				case ClickOver::CLICK_DRAG_TIMELINE: {

					Point2 mpos = mm->get_position() - ofs;
					/*
					if (mpos.x<name_limit)
						mpos.x=name_limit;
					if (mpos.x>settings_limit)
						mpos.x=settings_limit;
						*/

					//int zoomw = settings_limit-name_limit;
					float scale = _get_zoom_scale();
					float pos = h_scroll->get_value() + (mpos.x - name_limit) / scale;
					if (animation->get_step()) {
						pos = Math::stepify(pos, animation->get_step());
					}
					if (pos < 0)
						pos = 0;
					if (pos >= animation->get_length())
						pos = animation->get_length();

					if (pos < h_scroll->get_value()) {
						h_scroll->set_value(pos);
					} else if (pos > h_scroll->get_value() + (settings_limit - name_limit) / scale) {
						h_scroll->set_value(pos - (settings_limit - name_limit) / scale);
					}

					timeline_pos = pos;
					emit_signal("timeline_changed", pos, true);

				} break;
				case ClickOver::CLICK_SELECT_KEYS: {

					click.to = mm->get_position();
					if (click.to.y < h && click.at.y > h && mm->get_relative().y < 0) {

						float prev = v_scroll->get_value();
						v_scroll->set_value(v_scroll->get_value() - 1);
						if (prev != v_scroll->get_value())
							click.at.y += h;
					}
					if (click.to.y > size.height && click.at.y < size.height && mm->get_relative().y > 0) {

						float prev = v_scroll->get_value();
						v_scroll->set_value(v_scroll->get_value() + 1);
						if (prev != v_scroll->get_value())
							click.at.y -= h;
					}

				} break;
				case ClickOver::CLICK_MOVE_KEYS: {

					click.to = mm->get_position();
				} break;
				default: {}
			}

			return;
		} else if (mm->get_button_mask() & BUTTON_MASK_MIDDLE) {

			int rel = mm->get_relative().x;
			float relf = rel / _get_zoom_scale();
			h_scroll->set_value(h_scroll->get_value() - relf);
		}

		if (mm->get_button_mask() == 0) {

			Point2 mpos = mm->get_position() - ofs;

			if (mpos.y < h) {
				return;
			}

			mpos.y -= h;

			int idx = mpos.y / h;
			idx += v_scroll->get_value();
			if (idx < 0 || idx >= animation->get_track_count())
				return;

			mouse_over.track = idx;

			if (mpos.x < name_limit) {
				//name column

				mouse_over.over = MouseOver::OVER_NAME;

			} else if (mpos.x < settings_limit) {

				float pos = mpos.x - name_limit;
				pos /= _get_zoom_scale();
				pos += h_scroll->get_value();
				float w_time = (type_icon[0]->get_width() / _get_zoom_scale()) / 2.0;

				int kidx = animation->track_find_key(idx, pos);
				int kidx_n = kidx + 1;

				bool found = false;

				if (kidx >= 0 && kidx < animation->track_get_key_count(idx)) {

					float kpos = animation->track_get_key_time(idx, kidx);
					if (ABS(pos - kpos) <= w_time) {

						mouse_over.over = MouseOver::OVER_KEY;
						mouse_over.track = idx;
						mouse_over.over_key = kidx;
						found = true;
					}
				}

				if (!found && kidx_n >= 0 && kidx_n < animation->track_get_key_count(idx)) {

					float kpos = animation->track_get_key_time(idx, kidx_n);
					if (ABS(pos - kpos) <= w_time) {

						mouse_over.over = MouseOver::OVER_KEY;
						mouse_over.track = idx;
						mouse_over.over_key = kidx_n;
						found = true;
					}
				}

				if (found) {

					String text;
					text = "time: " + rtos(animation->track_get_key_time(idx, mouse_over.over_key)) + "\n";

					switch (animation->track_get_type(idx)) {

						case Animation::TYPE_TRANSFORM: {

							Dictionary d = animation->track_get_key_value(idx, mouse_over.over_key);
							if (d.has("loc"))
								text += "loc: " + String(d["loc"]) + "\n";
							if (d.has("rot"))
								text += "rot: " + String(d["rot"]) + "\n";
							if (d.has("scale"))
								text += "scale: " + String(d["scale"]) + "\n";
						} break;
						case Animation::TYPE_VALUE: {

							Variant v = animation->track_get_key_value(idx, mouse_over.over_key);
							//text+="value: "+String(v)+"\n";

							bool prop_exists = false;
							Variant::Type valid_type = Variant::NIL;
							Object *obj = NULL;

							RES res;
							Node *node = root->get_node_and_resource(animation->track_get_path(idx), res);

							if (res.is_valid()) {
								obj = res.ptr();
							} else if (node) {
								obj = node;
							}

							if (obj) {
								valid_type = obj->get_static_property_type(animation->track_get_path(idx).get_property(), &prop_exists);
							}

							text += "type: " + Variant::get_type_name(v.get_type()) + "\n";
							if (prop_exists && !Variant::can_convert(v.get_type(), valid_type)) {
								text += "value: " + String(v) + "  (Invalid, expected type: " + Variant::get_type_name(valid_type) + ")\n";
							} else {
								text += "value: " + String(v) + "\n";
							}

						} break;
						case Animation::TYPE_METHOD: {

							Dictionary d = animation->track_get_key_value(idx, mouse_over.over_key);
							if (d.has("method"))
								text += String(d["method"]);
							text += "(";
							Vector<Variant> args;
							if (d.has("args"))
								args = d["args"];
							for (int i = 0; i < args.size(); i++) {

								if (i > 0)
									text += ", ";
								text += String(args[i]);
							}
							text += ")\n";

						} break;
					}
					text += "easing: " + rtos(animation->track_get_key_transition(idx, mouse_over.over_key));

					track_editor->set_tooltip(text);
					return;
				}

			} else {
				//button column
				int ofsx = size.width - mpos.x;
				if (ofsx < 0)
					return;
				/*
				if (ofsx < remove_icon->get_width()) {

					mouse_over.over=MouseOver::OVER_REMOVE;

					return;
				}

				ofsx-=hsep+remove_icon->get_width();

				if (ofsx < move_down_icon->get_width()) {

					mouse_over.over=MouseOver::OVER_DOWN;
					return;
				}

				ofsx-=hsep+move_down_icon->get_width();

				if (ofsx < move_up_icon->get_width()) {

					mouse_over.over=MouseOver::OVER_UP;
					return;
				}

				ofsx-=hsep*3+move_up_icon->get_width();

*/

				if (ofsx < down_icon->get_width() + wrap_icon[0]->get_width() + hsep * 3) {

					mouse_over.over = MouseOver::OVER_WRAP;
					return;
				}

				ofsx -= hsep * 3 + wrap_icon[0]->get_width() + down_icon->get_width();

				if (ofsx < down_icon->get_width() + interp_icon[0]->get_width() + hsep * 3) {

					mouse_over.over = MouseOver::OVER_INTERP;
					return;
				}

				ofsx -= hsep * 2 + interp_icon[0]->get_width() + down_icon->get_width();

				if (ofsx < down_icon->get_width() + cont_icon[0]->get_width() + hsep * 3) {

					mouse_over.over = MouseOver::OVER_VALUE;
					return;
				}

				ofsx -= hsep * 3 + cont_icon[0]->get_width() + down_icon->get_width();

				if (ofsx < add_key_icon->get_width()) {

					mouse_over.over = MouseOver::OVER_ADD_KEY;
					return;
				}
			}
		}
	}
}

void AnimationKeyEditor::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {

			EditorNode::get_singleton()->update_keying();
			emit_signal("keying_changed");
		} break;

		case NOTIFICATION_ENTER_TREE: {

			key_editor->edit(key_edit);

			zoomicon->set_custom_minimum_size(Size2(24 * EDSCALE, 0));
			zoomicon->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);

			menu_add_track->get_popup()->add_icon_item(get_icon("KeyValue", "EditorIcons"), "Add Normal Track", ADD_TRACK_MENU_ADD_VALUE_TRACK);
			menu_add_track->get_popup()->add_icon_item(get_icon("KeyXform", "EditorIcons"), "Add Transform Track", ADD_TRACK_MENU_ADD_TRANSFORM_TRACK);
			menu_add_track->get_popup()->add_icon_item(get_icon("KeyCall", "EditorIcons"), "Add Call Func Track", ADD_TRACK_MENU_ADD_CALL_TRACK);

			menu_track->set_icon(get_icon("Tools", "EditorIcons"));
			menu_track->get_popup()->add_item(TTR("Scale Selection"), TRACK_MENU_SCALE);
			menu_track->get_popup()->add_item(TTR("Scale From Cursor"), TRACK_MENU_SCALE_PIVOT);
			menu_track->get_popup()->add_separator();
			menu_track->get_popup()->add_item(TTR("Duplicate Selection"), TRACK_MENU_DUPLICATE);
			menu_track->get_popup()->add_item(TTR("Duplicate Transposed"), TRACK_MENU_DUPLICATE_TRANSPOSE);
			menu_track->get_popup()->add_separator();
			menu_track->get_popup()->add_item(TTR("Goto Next Step"), TRACK_MENU_NEXT_STEP, KEY_MASK_CMD | KEY_RIGHT);
			menu_track->get_popup()->add_item(TTR("Goto Prev Step"), TRACK_MENU_PREV_STEP, KEY_MASK_CMD | KEY_LEFT);
			menu_track->get_popup()->add_separator();
			PopupMenu *tpp = memnew(PopupMenu);
			tpp->add_item(TTR("Linear"), TRACK_MENU_SET_ALL_TRANS_LINEAR);
			tpp->add_item(TTR("Constant"), TRACK_MENU_SET_ALL_TRANS_CONSTANT);
			tpp->add_item(TTR("In"), TRACK_MENU_SET_ALL_TRANS_IN);
			tpp->add_item(TTR("Out"), TRACK_MENU_SET_ALL_TRANS_OUT);
			tpp->add_item(TTR("In-Out"), TRACK_MENU_SET_ALL_TRANS_INOUT);
			tpp->add_item(TTR("Out-In"), TRACK_MENU_SET_ALL_TRANS_OUTIN);
			tpp->set_name(TTR("Transitions"));
			tpp->connect("id_pressed", this, "_menu_track");
			optimize_dialog->connect("confirmed", this, "_animation_optimize");

			menu_track->get_popup()->add_child(tpp);

			menu_track->get_popup()->add_item(TTR("Optimize Animation"), TRACK_MENU_OPTIMIZE);
			menu_track->get_popup()->add_item(TTR("Clean-Up Animation"), TRACK_MENU_CLEAN_UP);

			curve_linear->connect("pressed", this, "_menu_track", varray(CURVE_SET_LINEAR));
			curve_in->connect("pressed", this, "_menu_track", varray(CURVE_SET_IN));
			curve_out->connect("pressed", this, "_menu_track", varray(CURVE_SET_OUT));
			curve_inout->connect("pressed", this, "_menu_track", varray(CURVE_SET_INOUT));
			curve_outin->connect("pressed", this, "_menu_track", varray(CURVE_SET_OUTIN));
			curve_constant->connect("pressed", this, "_menu_track", varray(CURVE_SET_CONSTANT));

			edit_button->connect("pressed", this, "_toggle_edit_curves");

			curve_edit->connect("transition_changed", this, "_curve_transition_changed");
			call_select->connect("selected", this, "_add_call_track");

			_update_menu();

		} break;

		case NOTIFICATION_THEME_CHANGED: {
			zoomicon->set_texture(get_icon("Zoom", "EditorIcons"));

			menu_add_track->set_icon(get_icon("Add", "EditorIcons"));

			menu_track->set_icon(get_icon("Tools", "EditorIcons"));

			menu_add_track->get_popup()->set_item_icon(ADD_TRACK_MENU_ADD_VALUE_TRACK, get_icon("KeyValue", "EditorIcons"));
			menu_add_track->get_popup()->set_item_icon(ADD_TRACK_MENU_ADD_TRANSFORM_TRACK, get_icon("KeyXform", "EditorIcons"));
			menu_add_track->get_popup()->set_item_icon(ADD_TRACK_MENU_ADD_CALL_TRACK, get_icon("KeyCall", "EditorIcons"));

			curve_linear->set_icon(get_icon("CurveLinear", "EditorIcons"));
			curve_in->set_icon(get_icon("CurveIn", "EditorIcons"));
			curve_out->set_icon(get_icon("CurveOut", "EditorIcons"));
			curve_inout->set_icon(get_icon("CurveInOut", "EditorIcons"));
			curve_outin->set_icon(get_icon("CurveOutIn", "EditorIcons"));
			curve_constant->set_icon(get_icon("CurveConstant", "EditorIcons"));

			move_up_button->set_icon(get_icon("MoveUp", "EditorIcons"));
			move_down_button->set_icon(get_icon("MoveDown", "EditorIcons"));
			remove_button->set_icon(get_icon("Remove", "EditorIcons"));
			edit_button->set_icon(get_icon("EditKey", "EditorIcons"));

			loop->set_icon(get_icon("Loop", "EditorIcons"));

			{

				right_data_size_cache = 0;
				int hsep = get_constant("hseparation", "Tree");
				Ref<Texture> remove_icon = get_icon("Remove", "EditorIcons");
				Ref<Texture> move_up_icon = get_icon("MoveUp", "EditorIcons");
				Ref<Texture> move_down_icon = get_icon("MoveDown", "EditorIcons");
				Ref<Texture> down_icon = get_icon("select_arrow", "Tree");
				Ref<Texture> add_key_icon = get_icon("TrackAddKey", "EditorIcons");
				Ref<Texture> interp_icon[3] = {
					get_icon("InterpRaw", "EditorIcons"),
					get_icon("InterpLinear", "EditorIcons"),
					get_icon("InterpCubic", "EditorIcons")
				};
				Ref<Texture> cont_icon[3] = {
					get_icon("TrackContinuous", "EditorIcons"),
					get_icon("TrackDiscrete", "EditorIcons"),
					get_icon("TrackTrigger", "EditorIcons")
				};

				Ref<Texture> wrap_icon[2] = {
					get_icon("InterpWrapClamp", "EditorIcons"),
					get_icon("InterpWrapLoop", "EditorIcons"),
				};
				right_data_size_cache = down_icon->get_width() * 3 + add_key_icon->get_width() + interp_icon[0]->get_width() + cont_icon[0]->get_width() + wrap_icon[0]->get_width() + hsep * 8;
			}
		} break;
	}
}

void AnimationKeyEditor::_scroll_changed(double) {

	if (te_drawing)
		return;

	track_editor->update();
}

void AnimationKeyEditor::_update_paths() {

	if (animation.is_valid()) {
		//timeline->set_max(animation->get_length());
		//timeline->set_step(0.01);
		track_editor->update();
		length->set_value(animation->get_length());
		step->set_value(animation->get_step());
	}
}

void AnimationKeyEditor::_root_removed() {

	root = NULL;
}

void AnimationKeyEditor::_update_menu() {

	updating = true;

	if (animation.is_valid()) {

		length->set_value(animation->get_length());
		loop->set_pressed(animation->has_loop());
		step->set_value(animation->get_step());
	}

	track_editor->update();
	updating = false;
}
void AnimationKeyEditor::_clear_selection() {

	selection.clear();
	key_edit->animation = Ref<Animation>();
	key_edit->track = 0;
	key_edit->key_ofs = 0;
	key_edit->hint = PropertyInfo();
	key_edit->base = NodePath();
	key_edit->notify_change();
}

void AnimationKeyEditor::set_animation(const Ref<Animation> &p_anim) {

	if (animation.is_valid())
		animation->disconnect("changed", this, "_update_paths");
	animation = p_anim;
	if (animation.is_valid())
		animation->connect("changed", this, "_update_paths");

	timeline_pos = 0;
	_clear_selection();
	_update_paths();

	_update_menu();
	selected_track = -1;
	_edit_if_single_selection();

	EditorNode::get_singleton()->update_keying();
}

void AnimationKeyEditor::set_root(Node *p_root) {

	if (root)
		root->disconnect("tree_exited", this, "_root_removed");

	root = p_root;

	if (root)
		root->connect("tree_exited", this, "_root_removed", make_binds(), CONNECT_ONESHOT);
}

Node *AnimationKeyEditor::get_root() const {

	return root;
}

void AnimationKeyEditor::update_keying() {

	bool keying_enabled = is_visible_in_tree() && animation.is_valid();

	if (keying_enabled == keying)
		return;

	keying = keying_enabled;
	_update_menu();
	emit_signal("keying_changed");
}

bool AnimationKeyEditor::has_keying() const {

	return keying;
}

void AnimationKeyEditor::_query_insert(const InsertData &p_id) {

	if (insert_frame != Engine::get_singleton()->get_frames_drawn()) {
		//clear insert list for the frame if frame changed
		if (insert_confirm->is_visible_in_tree())
			return; //do nothing
		insert_data.clear();
		insert_query = false;
	}
	insert_frame = Engine::get_singleton()->get_frames_drawn();

	for (List<InsertData>::Element *E = insert_data.front(); E; E = E->next()) {
		//prevent insertion of multiple tracks
		if (E->get().path == p_id.path)
			return; //already inserted a track for this on this frame
	}

	insert_data.push_back(p_id);

	if (p_id.track_idx == -1) {
		if (bool(EDITOR_DEF("editors/animation/confirm_insert_track", true))) {
			//potential new key, does not exist
			if (insert_data.size() == 1)
				insert_confirm->set_text(vformat(TTR("Create NEW track for %s and insert key?"), p_id.query));
			else
				insert_confirm->set_text(vformat(TTR("Create %d NEW tracks and insert keys?"), insert_data.size()));

			insert_confirm->get_ok()->set_text(TTR("Create"));
			insert_confirm->popup_centered_minsize();
			insert_query = true;
		} else {
			call_deferred("_insert_delay");
			insert_queue = true;
		}

	} else {
		if (!insert_query && !insert_queue) {
			call_deferred("_insert_delay");
			insert_queue = true;
		}
	}
}

void AnimationKeyEditor::insert_transform_key(Spatial *p_node, const String &p_sub, const Transform &p_xform) {

	if (!keying)
		return;
	if (!animation.is_valid())
		return;

	ERR_FAIL_COND(!root);
	//let's build a node path
	String path = root->get_path_to(p_node);
	if (p_sub != "")
		path += ":" + p_sub;

	NodePath np = path;

	int track_idx = -1;

	for (int i = 0; i < animation->get_track_count(); i++) {

		if (animation->track_get_type(i) != Animation::TYPE_TRANSFORM)
			continue;
		if (animation->track_get_path(i) != np)
			continue;

		track_idx = i;
		break;
	}

	InsertData id;
	Dictionary val;

	id.path = np;
	id.track_idx = track_idx;
	id.value = p_xform;
	id.type = Animation::TYPE_TRANSFORM;
	id.query = "node '" + p_node->get_name() + "'";
	id.advance = false;

	//dialog insert

	_query_insert(id);
}

void AnimationKeyEditor::insert_node_value_key(Node *p_node, const String &p_property, const Variant &p_value, bool p_only_if_exists) {

	ERR_FAIL_COND(!root);
	//let's build a node path

	Node *node = p_node;

	String path = root->get_path_to(node);

	for (int i = 1; i < history->get_path_size(); i++) {

		String prop = history->get_path_property(i);
		ERR_FAIL_COND(prop == "");
		path += ":" + prop;
	}

	path += ":" + p_property;

	NodePath np = path;

	//locate track

	int track_idx = -1;

	for (int i = 0; i < animation->get_track_count(); i++) {

		if (animation->track_get_type(i) != Animation::TYPE_VALUE)
			continue;
		if (animation->track_get_path(i) != np)
			continue;

		track_idx = i;
		break;
	}

	if (p_only_if_exists && track_idx == -1)
		return;
	InsertData id;
	id.path = np;
	id.track_idx = track_idx;
	id.value = p_value;
	id.type = Animation::TYPE_VALUE;
	id.query = "property '" + p_property + "'";
	id.advance = false;
	//dialog insert
	_query_insert(id);
}

void AnimationKeyEditor::insert_value_key(const String &p_property, const Variant &p_value, bool p_advance) {

	ERR_FAIL_COND(!root);
	//let's build a node path
	ERR_FAIL_COND(history->get_path_size() == 0);
	Object *obj = ObjectDB::get_instance(history->get_path_object(0));
	ERR_FAIL_COND(!Object::cast_to<Node>(obj));

	Node *node = Object::cast_to<Node>(obj);

	String path = root->get_path_to(node);

	for (int i = 1; i < history->get_path_size(); i++) {

		String prop = history->get_path_property(i);
		ERR_FAIL_COND(prop == "");
		path += ":" + prop;
	}

	path += ":" + p_property;

	NodePath np = path;

	//locate track

	int track_idx = -1;

	for (int i = 0; i < animation->get_track_count(); i++) {

		if (animation->track_get_type(i) != Animation::TYPE_VALUE)
			continue;
		if (animation->track_get_path(i) != np)
			continue;

		track_idx = i;
		break;
	}

	InsertData id;
	id.path = np;
	id.track_idx = track_idx;
	id.value = p_value;
	id.type = Animation::TYPE_VALUE;
	id.query = "property '" + p_property + "'";
	id.advance = p_advance;
	//dialog insert
	_query_insert(id);
}

void AnimationKeyEditor::_confirm_insert_list() {

	undo_redo->create_action(TTR("Anim Create & Insert"));

	int last_track = animation->get_track_count();
	while (insert_data.size()) {

		last_track = _confirm_insert(insert_data.front()->get(), last_track);
		insert_data.pop_front();
	}

	undo_redo->commit_action();
}

int AnimationKeyEditor::_confirm_insert(InsertData p_id, int p_last_track) {

	if (p_last_track == -1)
		p_last_track = animation->get_track_count();

	bool created = false;
	if (p_id.track_idx < 0) {

		created = true;
		undo_redo->create_action(TTR("Anim Insert Track & Key"));
		Animation::UpdateMode update_mode = Animation::UPDATE_DISCRETE;

		if (p_id.type == Animation::TYPE_VALUE) {
			//wants a new tack

			{
				//shitty hack
				NodePath np;
				animation->add_track(p_id.type);
				animation->track_set_path(animation->get_track_count() - 1, p_id.path);
				PropertyInfo h = _find_hint_for_track(animation->get_track_count() - 1, np);
				animation->remove_track(animation->get_track_count() - 1); //hack

				if (h.type == Variant::REAL ||
						h.type == Variant::VECTOR2 ||
						h.type == Variant::RECT2 ||
						h.type == Variant::VECTOR3 ||
						h.type == Variant::RECT3 ||
						h.type == Variant::QUAT ||
						h.type == Variant::COLOR ||
						h.type == Variant::TRANSFORM) {

					update_mode = Animation::UPDATE_CONTINUOUS;
				}

				if (h.usage & PROPERTY_USAGE_ANIMATE_AS_TRIGGER) {
					update_mode = Animation::UPDATE_TRIGGER;
				}
			}
		}

		p_id.track_idx = p_last_track;

		undo_redo->add_do_method(animation.ptr(), "add_track", p_id.type);
		undo_redo->add_do_method(animation.ptr(), "track_set_path", p_id.track_idx, p_id.path);
		if (p_id.type == Animation::TYPE_VALUE)
			undo_redo->add_do_method(animation.ptr(), "value_track_set_update_mode", p_id.track_idx, update_mode);

	} else {
		undo_redo->create_action(TTR("Anim Insert Key"));
	}

	float time = timeline_pos;
	Variant value;

	switch (p_id.type) {

		case Animation::TYPE_VALUE: {

			value = p_id.value;

		} break;
		case Animation::TYPE_TRANSFORM: {

			Transform tr = p_id.value;
			Dictionary d;
			d["loc"] = tr.origin;
			d["scale"] = tr.basis.get_scale();
			d["rot"] = Quat(tr.basis); //.orthonormalized();
			value = d;
		} break;
		default: {}
	}

	undo_redo->add_do_method(animation.ptr(), "track_insert_key", p_id.track_idx, time, value);

	if (created) {

		//just remove the track
		undo_redo->add_undo_method(animation.ptr(), "remove_track", p_last_track);
		p_last_track++;
	} else {

		undo_redo->add_undo_method(animation.ptr(), "track_remove_key_at_pos", p_id.track_idx, time);
		int existing = animation->track_find_key(p_id.track_idx, time, true);
		if (existing != -1) {
			Variant v = animation->track_get_key_value(p_id.track_idx, existing);
			float trans = animation->track_get_key_transition(p_id.track_idx, existing);
			undo_redo->add_undo_method(animation.ptr(), "track_insert_key", p_id.track_idx, time, v, trans);
		}
	}

	undo_redo->add_do_method(this, "update");
	undo_redo->add_undo_method(this, "update");
	undo_redo->add_do_method(track_editor, "update");
	undo_redo->add_undo_method(track_editor, "update");
	undo_redo->add_do_method(track_pos, "update");
	undo_redo->add_undo_method(track_pos, "update");

	undo_redo->commit_action();

	return p_last_track;
}

Ref<Animation> AnimationKeyEditor::get_current_animation() const {

	return animation;
}

void AnimationKeyEditor::_animation_len_changed(float p_len) {

	if (updating)
		return;

	if (!animation.is_null()) {

		undo_redo->create_action(TTR("Change Anim Len"));
		undo_redo->add_do_method(animation.ptr(), "set_length", p_len);
		undo_redo->add_undo_method(animation.ptr(), "set_length", animation->get_length());
		undo_redo->add_do_method(this, "_animation_len_update");
		undo_redo->add_undo_method(this, "_animation_len_update");
		undo_redo->commit_action();
	}
}

void AnimationKeyEditor::_animation_len_update() {

	if (!animation.is_null())
		emit_signal(alc, animation->get_length());
}

void AnimationKeyEditor::_animation_changed() {
	if (updating)
		return;
	_update_menu();
}

void AnimationKeyEditor::_animation_loop_changed() {

	if (updating)
		return;

	if (!animation.is_null()) {

		undo_redo->create_action(TTR("Change Anim Loop"));
		undo_redo->add_do_method(animation.ptr(), "set_loop", loop->is_pressed());
		undo_redo->add_undo_method(animation.ptr(), "set_loop", !loop->is_pressed());
		undo_redo->commit_action();
	}
}

void AnimationKeyEditor::_create_value_item(int p_type) {

	undo_redo->create_action(TTR("Anim Create Typed Value Key"));

	Variant::CallError ce;
	Variant v = Variant::construct(Variant::Type(p_type), NULL, 0, ce);
	undo_redo->add_do_method(animation.ptr(), "track_insert_key", cvi_track, cvi_pos, v);
	undo_redo->add_undo_method(animation.ptr(), "track_remove_key_at_pos", cvi_track, cvi_pos);

	int existing = animation->track_find_key(cvi_track, cvi_pos, true);

	if (existing != -1) {
		Variant v = animation->track_get_key_value(cvi_track, existing);
		float trans = animation->track_get_key_transition(cvi_track, existing);
		undo_redo->add_undo_method(animation.ptr(), "track_insert_key", cvi_track, cvi_pos, v, trans);
	}

	undo_redo->commit_action();
}

void AnimationKeyEditor::set_anim_pos(float p_pos) {

	if (animation.is_null())
		return;
	timeline_pos = p_pos;
	update();
	track_pos->update();
	track_editor->update();
}

void AnimationKeyEditor::_pane_drag(const Point2 &p_delta) {

	Size2 ecs = ec->get_custom_minimum_size();
	ecs.y -= p_delta.y;
	if (ecs.y < 100)
		ecs.y = 100;
	ec->set_custom_minimum_size(ecs);
}

void AnimationKeyEditor::_insert_delay() {

	if (insert_query) {
		//discard since it's entered into query mode
		insert_queue = false;
		return;
	}

	undo_redo->create_action(TTR("Anim Insert"));

	int last_track = animation->get_track_count();
	bool advance = false;
	while (insert_data.size()) {

		if (insert_data.front()->get().advance)
			advance = true;
		last_track = _confirm_insert(insert_data.front()->get(), last_track);
		insert_data.pop_front();
	}

	undo_redo->commit_action();

	if (advance) {
		float step = animation->get_step();
		if (step == 0)
			step = 1;

		float pos = timeline_pos;

		pos = Math::stepify(pos + step, step);
		if (pos > animation->get_length())
			pos = animation->get_length();
		timeline_pos = pos;
		track_pos->update();
		emit_signal("timeline_changed", pos, true);
	}
	insert_queue = false;
}

void AnimationKeyEditor::_step_changed(float p_len) {

	updating = true;
	if (!animation.is_null()) {
		animation->set_step(p_len);
		emit_signal("animation_step_changed", animation->get_step());
	}
	updating = false;
}

void AnimationKeyEditor::_scale() {

	if (selection.empty())
		return;

	float from_t = 1e20;
	float to_t = -1e20;
	float len = -1e20;
	float pivot = 0;

	for (Map<SelectedKey, KeyInfo>::Element *E = selection.front(); E; E = E->next()) {
		float t = animation->track_get_key_time(E->key().track, E->key().key);
		if (t < from_t)
			from_t = t;
		if (t > to_t)
			to_t = t;
	}

	len = to_t - from_t;
	if (last_menu_track_opt == TRACK_MENU_SCALE_PIVOT) {
		pivot = timeline_pos;

	} else {

		pivot = from_t;
	}

	float s = scale->get_value();
	if (s == 0) {
		ERR_PRINT("Can't scale to 0");
	}

	undo_redo->create_action(TTR("Anim Scale Keys"));

	List<_AnimMoveRestore> to_restore;

	// 1-remove the keys
	for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

		undo_redo->add_do_method(animation.ptr(), "track_remove_key", E->key().track, E->key().key);
	}
	// 2- remove overlapped keys
	for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

		float newtime = (E->get().pos - from_t) * s + from_t;
		int idx = animation->track_find_key(E->key().track, newtime, true);
		if (idx == -1)
			continue;
		SelectedKey sk;
		sk.key = idx;
		sk.track = E->key().track;
		if (selection.has(sk))
			continue; //already in selection, don't save

		undo_redo->add_do_method(animation.ptr(), "track_remove_key_at_pos", E->key().track, newtime);
		_AnimMoveRestore amr;

		amr.key = animation->track_get_key_value(E->key().track, idx);
		amr.track = E->key().track;
		amr.time = newtime;
		amr.transition = animation->track_get_key_transition(E->key().track, idx);

		to_restore.push_back(amr);
	}

#define _NEW_POS(m_ofs) (((s > 0) ? m_ofs : from_t + (len - (m_ofs - from_t))) - pivot) * ABS(s) + from_t
	// 3-move the keys (re insert them)
	for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

		float newpos = _NEW_POS(E->get().pos);
		undo_redo->add_do_method(animation.ptr(), "track_insert_key", E->key().track, newpos, animation->track_get_key_value(E->key().track, E->key().key), animation->track_get_key_transition(E->key().track, E->key().key));
	}

	// 4-(undo) remove inserted keys
	for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

		float newpos = _NEW_POS(E->get().pos);
		undo_redo->add_undo_method(animation.ptr(), "track_remove_key_at_pos", E->key().track, newpos);
	}

	// 5-(undo) reinsert keys
	for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

		undo_redo->add_undo_method(animation.ptr(), "track_insert_key", E->key().track, E->get().pos, animation->track_get_key_value(E->key().track, E->key().key), animation->track_get_key_transition(E->key().track, E->key().key));
	}

	// 6-(undo) reinsert overlapped keys
	for (List<_AnimMoveRestore>::Element *E = to_restore.front(); E; E = E->next()) {

		_AnimMoveRestore &amr = E->get();
		undo_redo->add_undo_method(animation.ptr(), "track_insert_key", amr.track, amr.time, amr.key, amr.transition);
	}

	// 6-(undo) reinsert overlapped keys
	for (List<_AnimMoveRestore>::Element *E = to_restore.front(); E; E = E->next()) {

		_AnimMoveRestore &amr = E->get();
		undo_redo->add_undo_method(animation.ptr(), "track_insert_key", amr.track, amr.time, amr.key, amr.transition);
	}

	undo_redo->add_do_method(this, "_clear_selection_for_anim", animation);
	undo_redo->add_undo_method(this, "_clear_selection_for_anim", animation);

	// 7-reselect

	for (Map<SelectedKey, KeyInfo>::Element *E = selection.back(); E; E = E->prev()) {

		float oldpos = E->get().pos;
		float newpos = _NEW_POS(oldpos);
		if (newpos >= 0)
			undo_redo->add_do_method(this, "_select_at_anim", animation, E->key().track, newpos);
		undo_redo->add_undo_method(this, "_select_at_anim", animation, E->key().track, oldpos);
	}
#undef _NEW_POS
	undo_redo->commit_action();
}

void AnimationKeyEditor::_add_call_track(const NodePath &p_base) {

	Node *base = EditorNode::get_singleton()->get_edited_scene();
	if (!base)
		return;
	Node *from = base->get_node(p_base);
	if (!from || !root)
		return;

	NodePath path = root->get_path_to(from);

	//print_line("root: "+String(root->get_path()));
	//print_line("path: "+String(path));

	undo_redo->create_action(TTR("Anim Add Call Track"));
	undo_redo->add_do_method(animation.ptr(), "add_track", Animation::TYPE_METHOD);
	undo_redo->add_do_method(animation.ptr(), "track_set_path", animation->get_track_count(), path);
	undo_redo->add_undo_method(animation.ptr(), "remove_track", animation->get_track_count());
	undo_redo->commit_action();
}

void AnimationKeyEditor::cleanup() {

	set_animation(Ref<Animation>());
}

void AnimationKeyEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_root_removed"), &AnimationKeyEditor::_root_removed);
	ClassDB::bind_method(D_METHOD("_scale"), &AnimationKeyEditor::_scale);
	ClassDB::bind_method(D_METHOD("set_root"), &AnimationKeyEditor::set_root);

	//ClassDB::bind_method(D_METHOD("_confirm_insert"),&AnimationKeyEditor::_confirm_insert);
	ClassDB::bind_method(D_METHOD("_confirm_insert_list"), &AnimationKeyEditor::_confirm_insert_list);

	ClassDB::bind_method(D_METHOD("_update_paths"), &AnimationKeyEditor::_update_paths);
	ClassDB::bind_method(D_METHOD("_track_editor_draw"), &AnimationKeyEditor::_track_editor_draw);

	ClassDB::bind_method(D_METHOD("_animation_changed"), &AnimationKeyEditor::_animation_changed);
	ClassDB::bind_method(D_METHOD("_scroll_changed"), &AnimationKeyEditor::_scroll_changed);
	ClassDB::bind_method(D_METHOD("_track_editor_gui_input"), &AnimationKeyEditor::_track_editor_gui_input);
	ClassDB::bind_method(D_METHOD("_track_name_changed"), &AnimationKeyEditor::_track_name_changed);
	ClassDB::bind_method(D_METHOD("_track_menu_selected"), &AnimationKeyEditor::_track_menu_selected);
	ClassDB::bind_method(D_METHOD("_menu_add_track"), &AnimationKeyEditor::_menu_add_track);
	ClassDB::bind_method(D_METHOD("_menu_track"), &AnimationKeyEditor::_menu_track);
	ClassDB::bind_method(D_METHOD("_clear_selection_for_anim"), &AnimationKeyEditor::_clear_selection_for_anim);
	ClassDB::bind_method(D_METHOD("_select_at_anim"), &AnimationKeyEditor::_select_at_anim);
	ClassDB::bind_method(D_METHOD("_track_pos_draw"), &AnimationKeyEditor::_track_pos_draw);
	ClassDB::bind_method(D_METHOD("_insert_delay"), &AnimationKeyEditor::_insert_delay);
	ClassDB::bind_method(D_METHOD("_step_changed"), &AnimationKeyEditor::_step_changed);

	ClassDB::bind_method(D_METHOD("_animation_loop_changed"), &AnimationKeyEditor::_animation_loop_changed);
	ClassDB::bind_method(D_METHOD("_animation_len_changed"), &AnimationKeyEditor::_animation_len_changed);
	ClassDB::bind_method(D_METHOD("_create_value_item"), &AnimationKeyEditor::_create_value_item);
	ClassDB::bind_method(D_METHOD("_pane_drag"), &AnimationKeyEditor::_pane_drag);

	ClassDB::bind_method(D_METHOD("_animation_len_update"), &AnimationKeyEditor::_animation_len_update);

	ClassDB::bind_method(D_METHOD("set_animation"), &AnimationKeyEditor::set_animation);
	ClassDB::bind_method(D_METHOD("_animation_optimize"), &AnimationKeyEditor::_animation_optimize);
	ClassDB::bind_method(D_METHOD("_curve_transition_changed"), &AnimationKeyEditor::_curve_transition_changed);
	ClassDB::bind_method(D_METHOD("_toggle_edit_curves"), &AnimationKeyEditor::_toggle_edit_curves);
	ClassDB::bind_method(D_METHOD("_add_call_track"), &AnimationKeyEditor::_add_call_track);

	ADD_SIGNAL(MethodInfo("resource_selected", PropertyInfo(Variant::OBJECT, "res"), PropertyInfo(Variant::STRING, "prop")));
	ADD_SIGNAL(MethodInfo("keying_changed"));
	ADD_SIGNAL(MethodInfo("timeline_changed", PropertyInfo(Variant::REAL, "pos"), PropertyInfo(Variant::BOOL, "drag")));
	ADD_SIGNAL(MethodInfo("animation_len_changed", PropertyInfo(Variant::REAL, "len")));
	ADD_SIGNAL(MethodInfo("animation_step_changed", PropertyInfo(Variant::REAL, "step")));
	ADD_SIGNAL(MethodInfo("key_edited", PropertyInfo(Variant::INT, "track"), PropertyInfo(Variant::INT, "key")));
}

AnimationKeyEditor::AnimationKeyEditor() {

	alc = "animation_len_changed";
	editor_selection = EditorNode::get_singleton()->get_editor_selection();

	selected_track = -1;
	updating = false;
	te_drawing = false;
	undo_redo = EditorNode::get_singleton()->get_undo_redo();
	history = EditorNode::get_singleton()->get_editor_history();

	ec = memnew(Control);
	ec->set_custom_minimum_size(Size2(0, 150) * EDSCALE);
	add_child(ec);
	ec->set_v_size_flags(SIZE_EXPAND_FILL);

	h_scroll = memnew(HScrollBar);
	h_scroll->connect("value_changed", this, "_scroll_changed");
	add_child(h_scroll);
	h_scroll->set_value(0);

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);

	root = NULL;
	//menu = memnew( MenuButton );
	//menu->set_flat(true);
	//menu->set_position(Point2());
	//add_child(menu);

	zoomicon = memnew(TextureRect);
	hb->add_child(zoomicon);
	zoomicon->set_tooltip(TTR("Animation zoom."));

	zoom = memnew(HSlider);
	//hb->add_child(zoom);
	zoom->set_step(0.01);
	zoom->set_min(0.0);
	zoom->set_max(2.0);
	zoom->set_value(1.0);
	zoom->set_h_size_flags(SIZE_EXPAND_FILL);
	zoom->set_v_size_flags(SIZE_EXPAND_FILL);
	zoom->set_stretch_ratio(2);
	hb->add_child(zoom);
	zoom->connect("value_changed", this, "_scroll_changed");
	zoom->set_tooltip(TTR("Animation zoom."));

	hb->add_child(memnew(VSeparator));

	Label *l = memnew(Label);
	l->set_text(TTR("Length (s):"));
	hb->add_child(l);

	length = memnew(SpinBox);
	length->set_min(0.01);
	length->set_max(10000);
	length->set_step(0.01);
	length->set_h_size_flags(SIZE_EXPAND_FILL);
	length->set_stretch_ratio(1);
	length->set_tooltip(TTR("Animation length (in seconds)."));

	hb->add_child(length);
	length->connect("value_changed", this, "_animation_len_changed");

	l = memnew(Label);
	l->set_text(TTR("Step (s):"));
	hb->add_child(l);

	step = memnew(SpinBox);
	step->set_min(0.00);
	step->set_max(128);
	step->set_step(0.01);
	step->set_value(0.0);
	step->set_h_size_flags(SIZE_EXPAND_FILL);
	step->set_stretch_ratio(1);
	step->set_tooltip(TTR("Cursor step snap (in seconds)."));

	hb->add_child(step);
	step->connect("value_changed", this, "_step_changed");

	loop = memnew(ToolButton);
	loop->set_toggle_mode(true);
	loop->connect("pressed", this, "_animation_loop_changed");
	hb->add_child(loop);
	loop->set_tooltip(TTR("Enable/Disable looping in animation."));

	hb->add_child(memnew(VSeparator));

	menu_add_track = memnew(MenuButton);
	hb->add_child(menu_add_track);
	menu_add_track->get_popup()->connect("id_pressed", this, "_menu_add_track");
	menu_add_track->set_tooltip(TTR("Add new tracks."));

	move_up_button = memnew(ToolButton);
	hb->add_child(move_up_button);
	move_up_button->connect("pressed", this, "_menu_track", make_binds(TRACK_MENU_MOVE_UP));
	move_up_button->set_focus_mode(FOCUS_NONE);
	move_up_button->set_disabled(true);
	move_up_button->set_tooltip(TTR("Move current track up."));

	move_down_button = memnew(ToolButton);
	hb->add_child(move_down_button);
	move_down_button->connect("pressed", this, "_menu_track", make_binds(TRACK_MENU_MOVE_DOWN));
	move_down_button->set_focus_mode(FOCUS_NONE);
	move_down_button->set_disabled(true);
	move_down_button->set_tooltip(TTR("Move current track down."));

	remove_button = memnew(ToolButton);
	hb->add_child(remove_button);
	remove_button->connect("pressed", this, "_menu_track", make_binds(TRACK_MENU_REMOVE));
	remove_button->set_focus_mode(FOCUS_NONE);
	remove_button->set_disabled(true);
	remove_button->set_tooltip(TTR("Remove selected track."));

	hb->add_child(memnew(VSeparator));

	menu_track = memnew(MenuButton);
	hb->add_child(menu_track);
	menu_track->get_popup()->connect("id_pressed", this, "_menu_track");
	menu_track->set_tooltip(TTR("Track tools"));

	edit_button = memnew(ToolButton);
	edit_button->set_toggle_mode(true);
	edit_button->set_focus_mode(FOCUS_NONE);
	edit_button->set_disabled(true);

	hb->add_child(edit_button);
	edit_button->set_tooltip(TTR("Enable editing of individual keys by clicking them."));

	optimize_dialog = memnew(ConfirmationDialog);
	add_child(optimize_dialog);
	optimize_dialog->set_title(TTR("Anim. Optimizer"));
	VBoxContainer *optimize_vb = memnew(VBoxContainer);
	optimize_dialog->add_child(optimize_vb);

	optimize_linear_error = memnew(SpinBox);
	optimize_linear_error->set_max(1.0);
	optimize_linear_error->set_min(0.001);
	optimize_linear_error->set_step(0.001);
	optimize_linear_error->set_value(0.05);
	optimize_vb->add_margin_child(TTR("Max. Linear Error:"), optimize_linear_error);
	optimize_angular_error = memnew(SpinBox);
	optimize_angular_error->set_max(1.0);
	optimize_angular_error->set_min(0.001);
	optimize_angular_error->set_step(0.001);
	optimize_angular_error->set_value(0.01);

	optimize_vb->add_margin_child(TTR("Max. Angular Error:"), optimize_angular_error);
	optimize_max_angle = memnew(SpinBox);
	optimize_vb->add_margin_child(TTR("Max Optimizable Angle:"), optimize_max_angle);
	optimize_max_angle->set_max(360.0);
	optimize_max_angle->set_min(0.0);
	optimize_max_angle->set_step(0.1);
	optimize_max_angle->set_value(22);

	optimize_dialog->get_ok()->set_text(TTR("Optimize"));

	/*keying = memnew( Button );
	keying->set_toggle_mode(true);
	//keying->set_text("Keys");
	keying->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_END,60);
	keying->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,10);
	keying->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_BEGIN,55);
	keying->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,10);
	//add_child(keying);
	keying->connect("pressed",this,"_keying_toggled");
	*/

	/*	l = memnew( Label );
	l->set_text("Base: ");
	l->set_position(Point2(0,3));
	//dr_panel->add_child(l);*/

	//menu->get_popup()->connect("id_pressed",this,"_menu_callback");

	hb = memnew(HBoxContainer);
	hb->set_area_as_parent_rect();
	ec->add_child(hb);
	hb->set_v_size_flags(SIZE_EXPAND_FILL);

	track_editor = memnew(Control);
	track_editor->connect("draw", this, "_track_editor_draw");
	hb->add_child(track_editor);
	track_editor->connect("gui_input", this, "_track_editor_gui_input");
	track_editor->set_focus_mode(Control::FOCUS_ALL);
	track_editor->set_h_size_flags(SIZE_EXPAND_FILL);

	track_pos = memnew(Control);
	track_pos->set_area_as_parent_rect();
	track_pos->set_mouse_filter(MOUSE_FILTER_IGNORE);
	track_editor->add_child(track_pos);
	track_pos->connect("draw", this, "_track_pos_draw");

	select_anim_warning = memnew(Label);
	track_editor->add_child(select_anim_warning);
	select_anim_warning->set_area_as_parent_rect();
	select_anim_warning->set_text(TTR("Select an AnimationPlayer from the Scene Tree to edit animations."));
	select_anim_warning->set_autowrap(true);
	select_anim_warning->set_align(Label::ALIGN_CENTER);
	select_anim_warning->set_valign(Label::VALIGN_CENTER);

	v_scroll = memnew(VScrollBar);
	hb->add_child(v_scroll);
	v_scroll->connect("value_changed", this, "_scroll_changed");
	v_scroll->set_value(0);

	key_editor_tab = memnew(TabContainer);
	key_editor_tab->set_tab_align(TabContainer::ALIGN_LEFT);
	hb->add_child(key_editor_tab);
	key_editor_tab->set_custom_minimum_size(Size2(200, 0));

	key_editor = memnew(PropertyEditor);
	key_editor->set_area_as_parent_rect();
	key_editor->hide_top_label();
	key_editor->set_name(TTR("Key"));
	key_editor_tab->add_child(key_editor);

	key_edit = memnew(AnimationKeyEdit);
	key_edit->undo_redo = undo_redo;
	//key_edit->ke_dialog=key_edit_dialog;

	type_menu = memnew(PopupMenu);
	add_child(type_menu);
	for (int i = 0; i < Variant::VARIANT_MAX; i++)
		type_menu->add_item(Variant::get_type_name(Variant::Type(i)), i);
	type_menu->connect("id_pressed", this, "_create_value_item");

	VBoxContainer *curve_vb = memnew(VBoxContainer);
	curve_vb->set_name(TTR("Transition"));
	HBoxContainer *curve_hb = memnew(HBoxContainer);
	curve_vb->add_child(curve_hb);

	curve_linear = memnew(ToolButton);
	curve_linear->set_focus_mode(FOCUS_NONE);
	curve_hb->add_child(curve_linear);
	curve_in = memnew(ToolButton);
	curve_in->set_focus_mode(FOCUS_NONE);
	curve_hb->add_child(curve_in);
	curve_out = memnew(ToolButton);
	curve_out->set_focus_mode(FOCUS_NONE);
	curve_hb->add_child(curve_out);
	curve_inout = memnew(ToolButton);
	curve_inout->set_focus_mode(FOCUS_NONE);
	curve_hb->add_child(curve_inout);
	curve_outin = memnew(ToolButton);
	curve_outin->set_focus_mode(FOCUS_NONE);
	curve_hb->add_child(curve_outin);
	curve_constant = memnew(ToolButton);
	curve_constant->set_focus_mode(FOCUS_NONE);
	curve_hb->add_child(curve_constant);

	curve_edit = memnew(AnimationCurveEdit);
	curve_vb->add_child(curve_edit);
	curve_edit->set_v_size_flags(SIZE_EXPAND_FILL);
	key_editor_tab->add_child(curve_vb);

	track_name = memnew(LineEdit);
	track_name->set_as_toplevel(true);
	track_name->hide();
	add_child(track_name);
	track_name->connect("text_entered", this, "_track_name_changed");
	track_menu = memnew(PopupMenu);
	add_child(track_menu);
	track_menu->connect("id_pressed", this, "_track_menu_selected");

	key_editor_tab->hide();

	last_idx = 1;

	_update_menu();

	insert_confirm = memnew(ConfirmationDialog);
	add_child(insert_confirm);
	insert_confirm->connect("confirmed", this, "_confirm_insert_list");

	click.click = ClickOver::CLICK_NONE;

	name_column_ratio = 0.3;
	timeline_pos = 0;

	keying = false;
	insert_frame = 0;
	insert_query = false;
	insert_queue = false;

	editor_selection->connect("selection_changed", track_editor, "update");

	scale_dialog = memnew(ConfirmationDialog);
	VBoxContainer *vbc = memnew(VBoxContainer);
	scale_dialog->add_child(vbc);

	scale = memnew(SpinBox);
	scale->set_min(-99999);
	scale->set_max(99999);
	scale->set_step(0.001);
	vbc->add_margin_child(TTR("Scale Ratio:"), scale);
	scale_dialog->connect("confirmed", this, "_scale");
	add_child(scale_dialog);

	call_select = memnew(SceneTreeDialog);
	add_child(call_select);
	call_select->set_title(TTR("Call Functions in Which Node?"));

	cleanup_dialog = memnew(ConfirmationDialog);
	add_child(cleanup_dialog);
	VBoxContainer *cleanup_vb = memnew(VBoxContainer);
	cleanup_dialog->add_child(cleanup_vb);

	cleanup_keys = memnew(CheckButton);
	cleanup_keys->set_text(TTR("Remove invalid keys"));
	cleanup_keys->set_pressed(true);
	cleanup_vb->add_child(cleanup_keys);

	cleanup_tracks = memnew(CheckButton);
	cleanup_tracks->set_text(TTR("Remove unresolved and empty tracks"));
	cleanup_tracks->set_pressed(true);
	cleanup_vb->add_child(cleanup_tracks);

	cleanup_all = memnew(CheckButton);
	cleanup_all->set_text(TTR("Clean-up all animations"));
	cleanup_vb->add_child(cleanup_all);

	cleanup_dialog->set_title(TTR("Clean-Up Animation(s) (NO UNDO!)"));
	cleanup_dialog->get_ok()->set_text(TTR("Clean-Up"));

	cleanup_dialog->connect("confirmed", this, "_menu_track", varray(TRACK_MENU_CLEAN_UP_CONFIRM));

	add_constant_override("separation", get_constant("separation", "VBoxContainer"));

	track_editor->set_clip_contents(true);
}

AnimationKeyEditor::~AnimationKeyEditor() {

	memdelete(key_edit);
}
