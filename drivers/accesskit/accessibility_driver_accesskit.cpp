/**************************************************************************/
/*  accessibility_driver_accesskit.cpp                                    */
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

#ifdef ACCESSKIT_ENABLED

#include "accessibility_driver_accesskit.h"

#include "core/config/project_settings.h"
#include "core/version.h"

#include "servers/text_server.h"

AccessibilityDriverAccessKit *AccessibilityDriverAccessKit::singleton = nullptr;

_FORCE_INLINE_ accesskit_role AccessibilityDriverAccessKit::_accessibility_role(DisplayServer::AccessibilityRole p_role) const {
	if (role_map.has(p_role)) {
		return role_map[p_role];
	}
	return ACCESSKIT_ROLE_UNKNOWN;
}

_FORCE_INLINE_ accesskit_default_action_verb AccessibilityDriverAccessKit::_accessibility_def_action(DisplayServer::AccessibilityActionVerb p_action_verb) const {
	if (action_verb_map.has(p_action_verb)) {
		return action_verb_map[p_action_verb];
	}
	return ACCESSKIT_DEFAULT_ACTION_VERB_CLICK;
}

_FORCE_INLINE_ accesskit_action AccessibilityDriverAccessKit::_accessibility_action(DisplayServer::AccessibilityAction p_action) const {
	if (action_map.has(p_action)) {
		return action_map[p_action];
	}
	return ACCESSKIT_ACTION_DEFAULT;
}

bool AccessibilityDriverAccessKit::window_create(DisplayServer::WindowID p_window_id, void *p_handle) {
	ERR_FAIL_COND_V(windows.has(p_window_id), false);

	WindowData &wd = windows[p_window_id];

	AccessibilityElement *ae = memnew(AccessibilityElement);
	ae->role = ACCESSKIT_ROLE_WINDOW;
	ae->window_id = p_window_id;
	wd.root_id = rid_owner.make_rid(ae);

	accesskit_action_handler *accessibility_action_handler = accesskit_action_handler_new(&_accessibility_action_callback, (void *)(size_t)p_window_id);
#ifdef WINDOWS_ENABLED
	wd.adapter = accesskit_windows_subclassing_adapter_new(static_cast<HWND>(p_handle), &_accessibility_initial_tree_update_callback, (void *)(size_t)p_window_id, accessibility_action_handler);
#endif
#ifdef MACOS_ENABLED
	wd.adapter = accesskit_macos_subclassing_adapter_new(p_handle, &_accessibility_initial_tree_update_callback, (void *)(size_t)p_window_id, accessibility_action_handler);
#endif
#ifdef LINUXBSD_ENABLED
	wd.adapter = accesskit_unix_adapter_new(&_accessibility_initial_tree_update_callback, (void *)(size_t)p_window_id, true, accessibility_action_handler);
#endif

	if (wd.adapter == nullptr) {
		memdelete(ae);
		rid_owner.free(wd.root_id);
		windows.erase(p_window_id);

		return false;
	} else {
		return true;
	}
}

void AccessibilityDriverAccessKit::window_destroy(DisplayServer::WindowID p_window_id) {
	WindowData *wd = windows.getptr(p_window_id);
	ERR_FAIL_NULL(wd);

#ifdef WINDOWS_ENABLED
	accesskit_windows_subclassing_adapter_free(wd->adapter);
#endif
#ifdef MACOS_ENABLED
	accesskit_macos_subclassing_adapter_free(wd->adapter);
#endif
#ifdef LINUXBSD_ENABLED
	accesskit_unix_adapter_free(wd->adapter);
#endif
	accessibility_free_element(wd->root_id);

	windows.erase(p_window_id);
}

void AccessibilityDriverAccessKit::_accessibility_action_callback(const accesskit_action_request *p_request, void *p_user_data) {
	DisplayServer::WindowID window_id = (DisplayServer::WindowID)(size_t)p_user_data;
	ERR_FAIL_COND(!singleton->windows.has(window_id));

	RID rid = RID::from_uint64(p_request->target);
	AccessibilityElement *ae = singleton->rid_owner.get_or_null(rid);
	ERR_FAIL_NULL(ae);

	Variant rq_data;
	if (!ae->actions.has(p_request->action) && ae->role == ACCESSKIT_ROLE_INLINE_TEXT_BOX && p_request->action == ACCESSKIT_ACTION_SCROLL_INTO_VIEW) {
		AccessibilityElement *root_ae = singleton->rid_owner.get_or_null(ae->parent);
		ERR_FAIL_NULL(root_ae);
		ae = root_ae;
		rq_data = ae->run;
	}

	if (ae->actions.has(p_request->action)) {
		Callable &cb = ae->actions[p_request->action];
		if (cb.is_valid()) {
			if (p_request->data.has_value) {
				switch (p_request->data.value.tag) {
					case ACCESSKIT_ACTION_DATA_CUSTOM_ACTION: {
						rq_data = p_request->data.value.custom_action;
					} break;
					case ACCESSKIT_ACTION_DATA_VALUE: {
						rq_data = String::utf8(p_request->data.value.value);
					} break;
					case ACCESSKIT_ACTION_DATA_NUMERIC_VALUE: {
						rq_data = p_request->data.value.numeric_value;
					} break;
					case ACCESSKIT_ACTION_DATA_SCROLL_TARGET_RECT: {
						rq_data = Rect2(p_request->data.value.scroll_target_rect.x0, p_request->data.value.scroll_target_rect.y0, p_request->data.value.scroll_target_rect.x1 - p_request->data.value.scroll_target_rect.x0, p_request->data.value.scroll_target_rect.y1 - p_request->data.value.scroll_target_rect.y0);
					} break;
					case ACCESSKIT_ACTION_DATA_SCROLL_TO_POINT: {
						rq_data = Point2(p_request->data.value.scroll_to_point.x, p_request->data.value.scroll_to_point.y);
					} break;
					case ACCESSKIT_ACTION_DATA_SET_SCROLL_OFFSET: {
						rq_data = Point2(p_request->data.value.set_scroll_offset.x, p_request->data.value.set_scroll_offset.y);
					} break;
					case ACCESSKIT_ACTION_DATA_SET_TEXT_SELECTION: {
						Dictionary sel;

						RID start_rid = RID::from_uint64(p_request->data.value.set_text_selection.anchor.node);
						AccessibilityElement *start_ae = singleton->rid_owner.get_or_null(start_rid);
						ERR_FAIL_NULL(start_ae);

						RID end_rid = RID::from_uint64(p_request->data.value.set_text_selection.focus.node);
						AccessibilityElement *end_ae = singleton->rid_owner.get_or_null(end_rid);
						ERR_FAIL_NULL(end_ae);

						sel["start_element"] = start_ae->parent;
						sel["start_char"] = (int64_t)p_request->data.value.set_text_selection.anchor.character_index + start_ae->run.x;
						sel["end_element"] = end_ae->parent;
						sel["end_char"] = (int64_t)p_request->data.value.set_text_selection.focus.character_index + end_ae->run.x;
						rq_data = sel;
					} break;
				}
			}

			cb.call_deferred(rq_data);
		}
	}
}

accesskit_tree_update *AccessibilityDriverAccessKit::_accessibility_initial_tree_update_callback(void *p_user_data) {
	DisplayServer::WindowID window_id = (DisplayServer::WindowID)(size_t)p_user_data;
	WindowData *wd = singleton->windows.getptr(window_id);
	ERR_FAIL_NULL_V(wd, nullptr);

	accesskit_node_builder *win_bld = accesskit_node_builder_new(ACCESSKIT_ROLE_WINDOW);
	accesskit_node_builder_set_name(win_bld, "Godot Engine");
	accesskit_node_builder_set_busy(win_bld);

	accesskit_node *win_node = accesskit_node_builder_build(win_bld, singleton->node_classes);
	accesskit_node_id win_id = (accesskit_node_id)wd->root_id.get_id();

	accesskit_tree_update *tree_update = accesskit_tree_update_with_capacity_and_focus(1, win_id);

	accesskit_tree_update_set_tree(tree_update, accesskit_tree_new(win_id));
	accesskit_tree_update_push_node(tree_update, win_id, win_node);

	return tree_update;
}

RID AccessibilityDriverAccessKit::accessibility_create_element(DisplayServer::WindowID p_window_id, DisplayServer::AccessibilityRole p_role) {
	AccessibilityElement *ae = memnew(AccessibilityElement);
	ae->role = _accessibility_role(p_role);
	ae->window_id = p_window_id;
	RID rid = rid_owner.make_rid(ae);

	return rid;
}

RID AccessibilityDriverAccessKit::accessibility_create_sub_element(const RID &p_parent_rid, DisplayServer::AccessibilityRole p_role, int p_insert_pos) {
	AccessibilityElement *parent_ae = rid_owner.get_or_null(p_parent_rid);
	if (!parent_ae) {
		CRASH_NOW();
	}
	ERR_FAIL_NULL_V(parent_ae, RID());

	WindowData *wd = windows.getptr(parent_ae->window_id);
	ERR_FAIL_NULL_V(wd, RID());

	AccessibilityElement *ae = memnew(AccessibilityElement);
	ae->role = _accessibility_role(p_role);
	ae->window_id = parent_ae->window_id;
	ae->parent = p_parent_rid;
	ae->builder = accesskit_node_builder_new(ae->role);
	RID rid = rid_owner.make_rid(ae);
	if (p_insert_pos == -1) {
		parent_ae->children.push_back(rid);
	} else {
		parent_ae->children.insert(p_insert_pos, rid);
	}
	wd->update.insert(rid);

	return rid;
}

RID AccessibilityDriverAccessKit::accessibility_create_sub_text_edit_elements(const RID &p_parent_rid, const RID &p_shaped_text, float p_min_height, int p_insert_pos) {
	AccessibilityElement *parent_ae = rid_owner.get_or_null(p_parent_rid);
	ERR_FAIL_NULL_V(parent_ae, RID());

	WindowData *wd = windows.getptr(parent_ae->window_id);
	ERR_FAIL_NULL_V(wd, RID());

	AccessibilityElement *root_ae = memnew(AccessibilityElement);
	root_ae->role = ACCESSKIT_ROLE_GENERIC_CONTAINER;
	root_ae->window_id = parent_ae->window_id;
	root_ae->parent = p_parent_rid;
	root_ae->builder = accesskit_node_builder_new(root_ae->role);
	RID root_rid = rid_owner.make_rid(root_ae);
	if (p_insert_pos == -1) {
		parent_ae->children.push_back(root_rid);
	} else {
		parent_ae->children.insert(p_insert_pos, root_rid);
	}
	wd->update.insert(root_rid);

	float text_width = 0;
	float text_height = p_min_height;
	Vector<int32_t> words;
	int64_t run_count = 0; // Note: runs in visual order.
	const Glyph *gl = nullptr;
	int64_t gl_count = 0;
	int64_t gl_index = 0;
	float run_off_x = 0.0;
	Vector2i full_range;

	if (p_shaped_text.is_valid()) {
		text_width = TS->shaped_text_get_size(p_shaped_text).x;
		text_height = MAX(text_height, TS->shaped_text_get_size(p_shaped_text).y);
		words = TS->shaped_text_get_word_breaks(p_shaped_text);
		run_count = TS->shaped_get_run_count(p_shaped_text);
		gl = TS->shaped_text_get_glyphs(p_shaped_text);
		gl_count = TS->shaped_text_get_glyph_count(p_shaped_text);
		full_range = TS->shaped_text_get_range(p_shaped_text);
	}

	accesskit_rect root_rect;
	root_rect.x0 = 0;
	root_rect.y0 = 0;
	root_rect.x1 = text_width;
	root_rect.y1 = MAX(p_min_height, text_height);
	accesskit_node_builder_set_bounds(root_ae->builder, root_rect);

	// Create text element for each run.
	Vector<AccessibilityElement *> text_elements;
	for (int64_t i = 0; i < run_count; i++) {
		const Vector2i range = TS->shaped_get_run_range(p_shaped_text, i);
		String t = TS->shaped_get_run_text(p_shaped_text, i);

		if (t.is_empty()) {
			continue;
		}

		AccessibilityElement *ae = memnew(AccessibilityElement);
		ae->role = ACCESSKIT_ROLE_INLINE_TEXT_BOX;
		ae->window_id = parent_ae->window_id;
		ae->parent = root_rid;
		ae->run = Vector3i(range.x, range.y, i);
		ae->builder = accesskit_node_builder_new(ae->role);

		text_elements.push_back(ae);

		// UTF-8 text and char lengths.
		Vector<uint8_t> char_lengths;
		CharString text = t.utf8(&char_lengths);

		accesskit_node_builder_set_value(ae->builder, text.ptr());
		accesskit_node_builder_set_character_lengths(ae->builder, char_lengths.size(), char_lengths.ptr());

		// Word sizes.
		Vector<uint8_t> word_lengths;

		int32_t prev = ae->run.x;
		int32_t total = 0;
		for (int j = 0; j < words.size(); j += 2) {
			if (words[j] < ae->run.x) {
				continue;
			}
			if (words[j] >= ae->run.y) {
				break;
			}
			int32_t wlen = words[j] - prev;
			while (wlen > 255) {
				word_lengths.push_back(255);
				wlen -= 255;
				total += 255;
			}
			if (wlen > 0) {
				word_lengths.push_back(wlen);
				total += wlen;
			}
			prev = words[j];
		}
		if (total < t.length()) {
			word_lengths.push_back(t.length() - total);
		}
		accesskit_node_builder_set_word_lengths(ae->builder, word_lengths.size(), word_lengths.ptr());

		// Char widths and positions.
		Vector<float> char_positions;
		Vector<float> char_widths;

		char_positions.resize_zeroed(t.length());
		float *positions_ptr = char_positions.ptrw();

		char_widths.resize_zeroed(t.length());
		float *widths_ptr = char_widths.ptrw();

		float size_x = 0.0;
		for (int j = gl_index; j < gl_count; j += gl[j].count) {
			if (gl[j].start >= ae->run.y) {
				gl_index = j;
				break;
			}

			float advance = 0.0; // Graphame advance.
			for (int k = 0; k < gl[j].count; k++) {
				advance += gl[j + k].advance;
			}
			int chars = gl[j].end - gl[j].start;
			float adv_per_char = advance / (float)chars;

			for (int k = 0; k < chars; k++) {
				int index = gl[j].start + k - ae->run.x;
				ERR_CONTINUE(index < 0 || index >= t.length());
				positions_ptr[index] = size_x + adv_per_char * k;
				widths_ptr[index] = adv_per_char;
			}
			size_x += advance * gl[j].repeat;
		}
		positions_ptr[t.length() - 1] = size_x;
		widths_ptr[t.length() - 1] = 1.0;

		accesskit_node_builder_set_character_positions(ae->builder, char_positions.size(), char_positions.ptr());
		accesskit_node_builder_set_character_widths(ae->builder, char_widths.size(), char_widths.ptr());

		RID font_rid = TS->shaped_get_run_font_rid(p_shaped_text, i);
		if (font_rid != RID()) {
			CharString font_name = TS->font_get_name(font_rid).utf8();
			if (font_name.length() > 0) {
				accesskit_node_builder_set_font_family(ae->builder, font_name.ptr());
			}
			if (TS->font_get_style(font_rid).has_flag(TextServer::FONT_BOLD)) {
				accesskit_node_builder_set_bold(ae->builder);
			}
			if (TS->font_get_style(font_rid).has_flag(TextServer::FONT_ITALIC)) {
				accesskit_node_builder_set_italic(ae->builder);
			}
			accesskit_node_builder_set_font_weight(ae->builder, TS->font_get_weight(font_rid));
		}
		accesskit_node_builder_set_font_size(ae->builder, TS->shaped_get_run_font_size(p_shaped_text, i));
		CharString language = TS->shaped_get_run_language(p_shaped_text, i).utf8();
		if (language.length() > 0) {
			accesskit_node_builder_set_language(ae->builder, language.ptr());
		}
		accesskit_node_builder_set_text_direction(ae->builder, ACCESSKIT_TEXT_DIRECTION_LEFT_TO_RIGHT);

		accesskit_rect rect;
		rect.x0 = run_off_x;
		rect.y0 = 0;
		rect.x1 = run_off_x + size_x;
		rect.y1 = text_height;
		accesskit_node_builder_set_bounds(ae->builder, rect);
		accesskit_node_builder_add_action(ae->builder, ACCESSKIT_ACTION_SCROLL_INTO_VIEW);

		run_off_x += size_x;
	}
	{
		// Add "\n" at the end.
		AccessibilityElement *ae = memnew(AccessibilityElement);
		ae->role = ACCESSKIT_ROLE_INLINE_TEXT_BOX;
		ae->window_id = parent_ae->window_id;
		ae->parent = root_rid;
		ae->run = Vector3i(full_range.y, full_range.y, run_count);
		ae->builder = accesskit_node_builder_new(ae->role);

		text_elements.push_back(ae);

		Vector<uint8_t> char_lengths;
		char_lengths.push_back(1);
		accesskit_node_builder_set_value(ae->builder, "\n");
		accesskit_node_builder_set_character_lengths(ae->builder, char_lengths.size(), char_lengths.ptr());

		Vector<float> char_positions;
		Vector<float> char_widths;
		char_positions.push_back(0.0);
		char_widths.push_back(1.0);

		accesskit_node_builder_set_character_positions(ae->builder, char_positions.size(), char_positions.ptr());
		accesskit_node_builder_set_character_widths(ae->builder, char_widths.size(), char_widths.ptr());
		accesskit_node_builder_set_text_direction(ae->builder, ACCESSKIT_TEXT_DIRECTION_LEFT_TO_RIGHT);

		accesskit_rect rect;
		rect.x0 = run_off_x;
		rect.y0 = 0;
		rect.x1 = run_off_x + 1;
		rect.y1 = text_height;
		accesskit_node_builder_set_bounds(ae->builder, rect);
	}

	// Sort runs in logical order.
	struct RunCompare {
		_FORCE_INLINE_ bool operator()(const AccessibilityElement *l, const AccessibilityElement *r) const {
			return l->run.x < r->run.x;
		}
	};
	text_elements.sort_custom<RunCompare>();
	for (int64_t i = 0; i < text_elements.size(); i++) {
		RID rid = rid_owner.make_rid(text_elements[i]);
		root_ae->children.push_back(rid);
		wd->update.insert(rid);
	}

	return root_rid;
}

bool AccessibilityDriverAccessKit::accessibility_has_element(const RID &p_id) const {
	return rid_owner.owns(p_id);
}

void AccessibilityDriverAccessKit::_free_recursive(WindowData *p_wd, const RID &p_id) {
	if (p_wd && p_wd->update.has(p_id)) {
		p_wd->update.erase(p_id);
	}
	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	for (const RID &rid : ae->children) {
		_free_recursive(p_wd, rid);
	}
	if (ae->builder) {
		accesskit_node_builder_free(ae->builder);
	}
	memdelete(ae);
	rid_owner.free(p_id);
}

void AccessibilityDriverAccessKit::accessibility_free_element(const RID &p_id) {
	ERR_FAIL_COND_MSG(in_accessibility_update, "Element can't be removed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	if (ae) {
		WindowData *wd = windows.getptr(ae->window_id);
		AccessibilityElement *parent_ae = rid_owner.get_or_null(ae->parent);
		if (parent_ae) {
			parent_ae->children.erase(p_id);
		}
		_free_recursive(wd, p_id);
	}
}

void AccessibilityDriverAccessKit::accessibility_element_set_meta(const RID &p_id, const Variant &p_meta) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	ae->meta = p_meta;
}

Variant AccessibilityDriverAccessKit::accessibility_element_get_meta(const RID &p_id) const {
	const AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL_V(ae, Variant());
	return ae->meta;
}

void AccessibilityDriverAccessKit::accessibility_update_start() {
	ERR_FAIL_COND(in_accessibility_update);

	for (KeyValue<DisplayServer::WindowID, WindowData> &window : windows) {
		bool unfinished_update = false;
		for (const RID &rid : window.value.update) {
			AccessibilityElement *ae = rid_owner.get_or_null(rid);
			if (ae && ae->builder) {
				accesskit_node_builder_free(ae->builder);
				unfinished_update = true;
			}
		}
		window.value.update.clear();
		if (unfinished_update) {
			WARN_PRINT("Update restarted before previous update is finished!");
		}
	}
	focus_changed = false;

	in_accessibility_update = true;
}

void AccessibilityDriverAccessKit::accessibility_update_set_focus(const RID &p_id) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	if (p_id.is_valid() && rid_owner.owns(p_id)) {
		focus = p_id;
	} else {
		focus = RID();
	}

	focus_changed = true;
}

RID AccessibilityDriverAccessKit::accessibility_get_window_root(DisplayServer::WindowID p_window_id) const {
	const WindowData *wd = windows.getptr(p_window_id);
	ERR_FAIL_NULL_V(wd, RID());

	return wd->root_id;
}

void AccessibilityDriverAccessKit::accessibility_update_commit() {
	ERR_FAIL_COND(!in_accessibility_update);

	AccessibilityElement *focus_ae = rid_owner.get_or_null(focus);
	for (KeyValue<DisplayServer::WindowID, WindowData> &window : windows) {
		uint32_t update_size = window.value.update.size();
		if (update_size > 0 || focus_changed) {
			accesskit_node_id ac_focus = (accesskit_node_id)window.value.root_id.get_id();
			if (focus_ae && focus_ae->window_id == window.key) {
				ac_focus = (accesskit_node_id)focus.get_id();
			}

			accesskit_tree_update *tree_update = accesskit_tree_update_with_capacity_and_focus(update_size, ac_focus);
			for (const RID &rid : window.value.update) {
				AccessibilityElement *ae = rid_owner.get_or_null(rid);
				if (ae && ae->builder) {
					for (const RID &child_rid : ae->children) {
						accesskit_node_builder_push_child(ae->builder, (accesskit_node_id)child_rid.get_id());
					}

					accesskit_node *ac_node = accesskit_node_builder_build(ae->builder, node_classes);
					accesskit_node_id ac_id = (accesskit_node_id)rid.get_id();
					ae->builder = nullptr;

					accesskit_tree_update_push_node(tree_update, ac_id, ac_node);
				}
			}
			window.value.update.clear();

#ifdef WINDOWS_ENABLED
			accesskit_windows_queued_events *events = accesskit_windows_subclassing_adapter_update(window.value.adapter, tree_update);
			if (events) {
				accesskit_windows_queued_events_raise(events);
			}
#endif
#ifdef MACOS_ENABLED
			accesskit_macos_queued_events *events = accesskit_macos_subclassing_adapter_update(window.value.adapter, tree_update);
			if (events) {
				accesskit_macos_queued_events_raise(events);
			}
#endif
#ifdef LINUXBSD_ENABLED
			accesskit_unix_adapter_update(window.value.adapter, tree_update);
#endif
		}
	}
	focus_changed = false;
	in_accessibility_update = false;
}

_FORCE_INLINE_ void AccessibilityDriverAccessKit::_ensure_builder(const RID &p_id, AccessibilityElement *p_ae) {
	if (unlikely(!p_ae->builder)) {
		WindowData *wd = windows.getptr(p_ae->window_id);
		ERR_FAIL_NULL(wd);

		wd->update.insert(p_id);
		p_ae->builder = accesskit_node_builder_new(p_ae->role);
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_role(const RID &p_id, DisplayServer::AccessibilityRole p_role) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	if (ae->role == _accessibility_role(p_role)) {
		return;
	}
	ae->role = _accessibility_role(p_role);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_role(ae->builder, ae->role);
}

void AccessibilityDriverAccessKit::accessibility_update_set_name(const RID &p_id, const String &p_name) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (!p_name.is_empty()) {
		accesskit_node_builder_set_name(ae->builder, p_name.utf8().ptr());
	} else {
		accesskit_node_builder_set_name(ae->builder, "");
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_description(const RID &p_id, const String &p_description) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (!p_description.is_empty()) {
		accesskit_node_builder_set_description(ae->builder, p_description.utf8().ptr());
	} else {
		accesskit_node_builder_set_description(ae->builder, "");
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_value(const RID &p_id, const String &p_value) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (!p_value.is_empty()) {
		Vector<uint8_t> ch_length;
		accesskit_node_builder_set_value(ae->builder, p_value.utf8(&ch_length).ptr());
		accesskit_node_builder_set_character_lengths(ae->builder, ch_length.size(), ch_length.ptr());
	} else {
		accesskit_node_builder_set_value(ae->builder, "");
		accesskit_node_builder_set_character_lengths(ae->builder, 0, nullptr);
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_tooltip(const RID &p_id, const String &p_tooltip) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (!p_tooltip.is_empty()) {
		accesskit_node_builder_set_tooltip(ae->builder, p_tooltip.utf8().ptr());
	} else {
		accesskit_node_builder_set_tooltip(ae->builder, "");
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_bounds(const RID &p_id, const Rect2 &p_rect) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_rect rect;
	rect.x0 = p_rect.position.x;
	rect.y0 = p_rect.position.y;
	rect.x1 = p_rect.position.x + p_rect.size.x;
	rect.y1 = p_rect.position.y + p_rect.size.y;
	accesskit_node_builder_set_bounds(ae->builder, rect);
}

void AccessibilityDriverAccessKit::accessibility_update_set_transform(const RID &p_id, const Transform2D &p_transform) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_affine transform = { p_transform.columns[0][0], p_transform.columns[0][1], p_transform.columns[1][0], p_transform.columns[1][1], p_transform.columns[2][0], p_transform.columns[2][1] };
	accesskit_node_builder_set_transform(ae->builder, transform);
}

void AccessibilityDriverAccessKit::accessibility_update_add_child(const RID &p_id, const RID &p_child_id) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_child_id);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_push_child(ae->builder, (accesskit_node_id)p_child_id.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_add_related_controls(const RID &p_id, const RID &p_related_id) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_related_id);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_push_controlled(ae->builder, (accesskit_node_id)p_related_id.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_add_related_details(const RID &p_id, const RID &p_related_id) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_related_id);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_push_detail(ae->builder, (accesskit_node_id)p_related_id.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_add_related_described_by(const RID &p_id, const RID &p_related_id) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_related_id);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_push_described_by(ae->builder, (accesskit_node_id)p_related_id.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_add_related_flow_to(const RID &p_id, const RID &p_related_id) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_related_id);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_push_flow_to(ae->builder, (accesskit_node_id)p_related_id.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_add_related_labeled_by(const RID &p_id, const RID &p_related_id) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_related_id);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_push_labelled_by(ae->builder, (accesskit_node_id)p_related_id.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_add_related_radio_group(const RID &p_id, const RID &p_related_id) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_related_id);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_push_to_radio_group(ae->builder, (accesskit_node_id)p_related_id.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_set_active_descendant(const RID &p_id, const RID &p_other_id) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_other_id);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_active_descendant(ae->builder, (accesskit_node_id)p_other_id.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_set_next_on_line(const RID &p_id, const RID &p_other_id) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_other_id);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_next_on_line(ae->builder, (accesskit_node_id)p_other_id.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_set_previous_on_line(const RID &p_id, const RID &p_other_id) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_other_id);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_previous_on_line(ae->builder, (accesskit_node_id)p_other_id.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_set_member_of(const RID &p_id, const RID &p_other_id) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_other_id);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_member_of(ae->builder, (accesskit_node_id)p_other_id.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_set_in_page_link_target(const RID &p_id, const RID &p_other_id) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_other_id);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_in_page_link_target(ae->builder, (accesskit_node_id)p_other_id.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_set_error_message(const RID &p_id, const RID &p_other_id) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_other_id);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_error_message(ae->builder, (accesskit_node_id)p_other_id.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_set_live(const RID &p_id, DisplayServer::AccessibilityLiveMode p_live) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	switch (p_live) {
		case DisplayServer::AccessibilityLiveMode::LIVE_OFF: {
			accesskit_node_builder_set_live(ae->builder, ACCESSKIT_LIVE_OFF);
		} break;
		case DisplayServer::AccessibilityLiveMode::LIVE_POLITE: {
			accesskit_node_builder_set_live(ae->builder, ACCESSKIT_LIVE_POLITE);
		} break;
		case DisplayServer::AccessibilityLiveMode::LIVE_ASSERTIVE: {
			accesskit_node_builder_set_live(ae->builder, ACCESSKIT_LIVE_ASSERTIVE);
		} break;
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_default_action_verb(const RID &p_id, DisplayServer::AccessibilityActionVerb p_action) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_default_action_verb(ae->builder, _accessibility_def_action(p_action));
}

void AccessibilityDriverAccessKit::accessibility_update_add_action(const RID &p_id, DisplayServer::AccessibilityAction p_action, const Callable &p_callable) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	ae->actions[_accessibility_action(p_action)] = p_callable;

	accesskit_node_builder_add_action(ae->builder, _accessibility_action(p_action));
}

void AccessibilityDriverAccessKit::accessibility_update_add_custom_action(const RID &p_id, int p_action_id, const String &p_action_description) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (!p_action_description.is_empty()) {
		accesskit_custom_action ca = accesskit_custom_action_new(p_action_id, p_action_description.utf8().ptr());
		accesskit_node_builder_push_custom_action(ae->builder, ca);
	} else {
		String cs_name = vformat("Custom Action %d", p_action_id);
		accesskit_custom_action ca = accesskit_custom_action_new(p_action_id, cs_name.utf8().ptr());
		accesskit_node_builder_push_custom_action(ae->builder, ca);
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_table_header(const RID &p_id, const RID &p_header) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_header);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_table_header(ae->builder, (accesskit_node_id)p_header.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_set_table_row_header(const RID &p_id, const RID &p_header) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");
	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_header);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_table_row_header(ae->builder, (accesskit_node_id)p_header.get_id());
}

void AccessibilityDriverAccessKit::accessibility_update_set_table_column_header(const RID &p_id, const RID &p_header) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *other_ae = rid_owner.get_or_null(p_header);
	ERR_FAIL_NULL(other_ae);
	ERR_FAIL_COND(other_ae->window_id != ae->window_id);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_table_column_header(ae->builder, (accesskit_node_id)p_header.get_id());
}
void AccessibilityDriverAccessKit::accessibility_update_set_table_row_count(const RID &p_id, int p_count) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_table_row_count(ae->builder, p_count);
}

void AccessibilityDriverAccessKit::accessibility_update_set_table_column_count(const RID &p_id, int p_count) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_table_column_count(ae->builder, p_count);
}

void AccessibilityDriverAccessKit::accessibility_update_set_table_row_index(const RID &p_id, int p_index) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_table_row_index(ae->builder, p_index);
}

void AccessibilityDriverAccessKit::accessibility_update_set_table_column_index(const RID &p_id, int p_index) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_table_column_index(ae->builder, p_index);
}

void AccessibilityDriverAccessKit::accessibility_update_set_table_cell_position(const RID &p_id, int p_row_index, int p_column_index) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_table_cell_row_index(ae->builder, p_row_index);
	accesskit_node_builder_set_table_cell_column_index(ae->builder, p_column_index);
}

void AccessibilityDriverAccessKit::accessibility_update_set_table_cell_span(const RID &p_id, int p_row_span, int p_column_span) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_table_cell_row_span(ae->builder, p_row_span);
	accesskit_node_builder_set_table_cell_column_span(ae->builder, p_column_span);
}

void AccessibilityDriverAccessKit::accessibility_update_set_list_item_count(const RID &p_id, int p_size) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_size_of_set(ae->builder, p_size);
}

void AccessibilityDriverAccessKit::accessibility_update_set_list_item_index(const RID &p_id, int p_index) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_position_in_set(ae->builder, p_index);
}

void AccessibilityDriverAccessKit::accessibility_update_set_list_item_level(const RID &p_id, int p_level) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_hierarchical_level(ae->builder, p_level);
}

void AccessibilityDriverAccessKit::accessibility_update_set_list_item_selected(const RID &p_id, bool p_selected) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_selected(ae->builder, p_selected);
}

void AccessibilityDriverAccessKit::accessibility_update_set_list_item_expanded(const RID &p_id, bool p_expanded) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_expanded(ae->builder, p_expanded);
}

void AccessibilityDriverAccessKit::accessibility_update_set_popup_type(const RID &p_id, DisplayServer::AccessibilityPopupType p_popup) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	switch (p_popup) {
		case DisplayServer::AccessibilityPopupType::POPUP_UNKNOWN: {
			accesskit_node_builder_set_has_popup(ae->builder, ACCESSKIT_HAS_POPUP_TRUE);
		} break;
		case DisplayServer::AccessibilityPopupType::POPUP_MENU: {
			accesskit_node_builder_set_has_popup(ae->builder, ACCESSKIT_HAS_POPUP_MENU);
		} break;
		case DisplayServer::AccessibilityPopupType::POPUP_LIST: {
			accesskit_node_builder_set_has_popup(ae->builder, ACCESSKIT_HAS_POPUP_LISTBOX);
		} break;
		case DisplayServer::AccessibilityPopupType::POPUP_TREE: {
			accesskit_node_builder_set_has_popup(ae->builder, ACCESSKIT_HAS_POPUP_TREE);
		} break;
		case DisplayServer::AccessibilityPopupType::POPUP_DIALOG: {
			accesskit_node_builder_set_has_popup(ae->builder, ACCESSKIT_HAS_POPUP_DIALOG);
		} break;
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_checked(const RID &p_id, bool p_checekd) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (p_checekd) {
		accesskit_node_builder_set_checked(ae->builder, ACCESSKIT_CHECKED_TRUE);
	} else {
		accesskit_node_builder_set_checked(ae->builder, ACCESSKIT_CHECKED_FALSE);
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_num_value(const RID &p_id, double p_position) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_numeric_value(ae->builder, p_position);
}

void AccessibilityDriverAccessKit::accessibility_update_set_num_range(const RID &p_id, double p_min, double p_max) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_min_numeric_value(ae->builder, p_min);
	accesskit_node_builder_set_max_numeric_value(ae->builder, p_max);
}

void AccessibilityDriverAccessKit::accessibility_update_set_num_step(const RID &p_id, double p_step) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_numeric_value_step(ae->builder, p_step);
}

void AccessibilityDriverAccessKit::accessibility_update_set_num_jump(const RID &p_id, double p_jump) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_numeric_value_jump(ae->builder, p_jump);
}

void AccessibilityDriverAccessKit::accessibility_update_set_scroll_x(const RID &p_id, double p_position) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_scroll_x(ae->builder, p_position);
}

void AccessibilityDriverAccessKit::accessibility_update_set_scroll_x_range(const RID &p_id, double p_min, double p_max) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_scroll_x_min(ae->builder, p_min);
	accesskit_node_builder_set_scroll_x_max(ae->builder, p_max);
}

void AccessibilityDriverAccessKit::accessibility_update_set_scroll_y(const RID &p_id, double p_position) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_scroll_y(ae->builder, p_position);
}

void AccessibilityDriverAccessKit::accessibility_update_set_scroll_y_range(const RID &p_id, double p_min, double p_max) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_scroll_y_min(ae->builder, p_min);
	accesskit_node_builder_set_scroll_y_max(ae->builder, p_max);
}

void AccessibilityDriverAccessKit::accessibility_update_set_text_decorations(const RID &p_id, bool p_underline, bool p_strikethrough, bool p_overline) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (p_underline) {
		accesskit_node_builder_set_underline(ae->builder, ACCESSKIT_TEXT_DECORATION_SOLID);
	} else {
		accesskit_node_builder_clear_underline(ae->builder);
	}
	if (p_overline) {
		accesskit_node_builder_set_overline(ae->builder, ACCESSKIT_TEXT_DECORATION_SOLID);
	} else {
		accesskit_node_builder_clear_overline(ae->builder);
	}
	if (p_strikethrough) {
		accesskit_node_builder_set_strikethrough(ae->builder, ACCESSKIT_TEXT_DECORATION_SOLID);
	} else {
		accesskit_node_builder_clear_strikethrough(ae->builder);
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_text_align(const RID &p_id, HorizontalAlignment p_align) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	switch (p_align) {
		case HORIZONTAL_ALIGNMENT_LEFT: {
			accesskit_node_builder_set_text_align(ae->builder, ACCESSKIT_TEXT_ALIGN_LEFT);
		} break;
		case HORIZONTAL_ALIGNMENT_CENTER: {
			accesskit_node_builder_set_text_align(ae->builder, ACCESSKIT_TEXT_ALIGN_RIGHT);
		} break;
		case HORIZONTAL_ALIGNMENT_RIGHT: {
			accesskit_node_builder_set_text_align(ae->builder, ACCESSKIT_TEXT_ALIGN_CENTER);
		} break;
		case HORIZONTAL_ALIGNMENT_FILL: {
			accesskit_node_builder_set_text_align(ae->builder, ACCESSKIT_TEXT_ALIGN_JUSTIFY);
		} break;
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_text_selection(const RID &p_id, const RID &p_text_start_id, int p_start_char, const RID &p_text_end_id, int p_end_char) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	AccessibilityElement *start_ae = rid_owner.get_or_null(p_text_start_id);
	ERR_FAIL_NULL(start_ae);
	ERR_FAIL_COND(start_ae->window_id != ae->window_id);
	AccessibilityElement *end_ae = rid_owner.get_or_null(p_text_end_id);
	ERR_FAIL_NULL(end_ae);
	ERR_FAIL_COND(end_ae->window_id != ae->window_id);

	int start_pos = p_start_char;
	int end_pos = p_end_char;
	RID start_rid;
	RID end_rid;
	for (const RID &rid : start_ae->children) {
		const AccessibilityElement *child_ae = rid_owner.get_or_null(rid);
		if (child_ae && child_ae->role == ACCESSKIT_ROLE_INLINE_TEXT_BOX) {
			if (p_start_char >= child_ae->run.x && p_start_char <= child_ae->run.y) {
				start_rid = rid;
				start_pos = p_start_char - child_ae->run.x;
				break;
			}
		}
	}
	for (const RID &rid : end_ae->children) {
		const AccessibilityElement *child_ae = rid_owner.get_or_null(rid);
		if (child_ae && child_ae->role == ACCESSKIT_ROLE_INLINE_TEXT_BOX) {
			if (p_end_char >= child_ae->run.x && p_end_char <= child_ae->run.y) {
				end_rid = rid;
				end_pos = p_end_char - child_ae->run.x;
				break;
			}
		}
	}
	ERR_FAIL_COND(start_rid.is_null() && end_rid.is_null());
	_ensure_builder(p_id, ae);

	accesskit_text_selection sel;
	sel.anchor.node = (accesskit_node_id)start_rid.get_id();
	sel.anchor.character_index = start_pos;
	sel.focus.node = (accesskit_node_id)end_rid.get_id();
	sel.focus.character_index = end_pos;
	accesskit_builder_set_text_selection(ae->builder, sel);
}

void AccessibilityDriverAccessKit::accessibility_update_set_flag(const RID &p_id, DisplayServer::AccessibilityFlags p_flag, bool p_value) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	switch (p_flag) {
		case DisplayServer::AccessibilityFlags::FLAG_HOVERED: {
			if (p_value) {
				accesskit_node_builder_set_hovered(ae->builder);
			} else {
				accesskit_node_builder_clear_hovered(ae->builder);
			}
		} break;
		case DisplayServer::AccessibilityFlags::FLAG_HIDDEN: {
			if (p_value) {
				accesskit_node_builder_set_hidden(ae->builder);
			} else {
				accesskit_node_builder_clear_hidden(ae->builder);
			}
		} break;
		case DisplayServer::AccessibilityFlags::FLAG_LINKED: {
			if (p_value) {
				accesskit_node_builder_set_linked(ae->builder);
			} else {
				accesskit_node_builder_clear_linked(ae->builder);
			}
		} break;
		case DisplayServer::AccessibilityFlags::FLAG_MULTISELECTABLE: {
			if (p_value) {
				accesskit_node_builder_set_multiselectable(ae->builder);
			} else {
				accesskit_node_builder_clear_multiselectable(ae->builder);
			}
		} break;
		case DisplayServer::AccessibilityFlags::FLAG_REQUIRED: {
			if (p_value) {
				accesskit_node_builder_set_required(ae->builder);
			} else {
				accesskit_node_builder_clear_required(ae->builder);
			}
		} break;
		case DisplayServer::AccessibilityFlags::FLAG_VISITED: {
			if (p_value) {
				accesskit_node_builder_set_visited(ae->builder);
			} else {
				accesskit_node_builder_clear_visited(ae->builder);
			}
		} break;
		case DisplayServer::AccessibilityFlags::FLAG_BUSY: {
			if (p_value) {
				accesskit_node_builder_set_busy(ae->builder);
			} else {
				accesskit_node_builder_clear_busy(ae->builder);
			}
		} break;
		case DisplayServer::AccessibilityFlags::FLAG_MODAL: {
			if (p_value) {
				accesskit_node_builder_set_modal(ae->builder);
			} else {
				accesskit_node_builder_clear_modal(ae->builder);
			}
		} break;
		case DisplayServer::AccessibilityFlags::FLAG_TOUCH_PASSTHROUGH: {
			if (p_value) {
				accesskit_node_builder_set_touch_transparent(ae->builder);
			} else {
				accesskit_node_builder_clear_touch_transparent(ae->builder);
			}
		} break;
		case DisplayServer::AccessibilityFlags::FLAG_READONLY: {
			if (p_value) {
				accesskit_node_builder_set_read_only(ae->builder);
			} else {
				accesskit_node_builder_clear_read_only(ae->builder);
			}
		} break;
		case DisplayServer::AccessibilityFlags::FLAG_DISABLED: {
			if (p_value) {
				accesskit_node_builder_set_disabled(ae->builder);
			} else {
				accesskit_node_builder_clear_disabled(ae->builder);
			}
		} break;
		case DisplayServer::AccessibilityFlags::FLAG_CLIPS_CHILDREN: {
			if (p_value) {
				accesskit_node_builder_set_clips_children(ae->builder);
			} else {
				accesskit_node_builder_clear_clips_children(ae->builder);
			}
		} break;
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_classname(const RID &p_id, const String &p_classname) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (!p_classname.is_empty()) {
		accesskit_node_builder_set_class_name(ae->builder, p_classname.utf8().ptr());
	} else {
		accesskit_node_builder_set_class_name(ae->builder, "");
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_placeholder(const RID &p_id, const String &p_placeholder) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (!p_placeholder.is_empty()) {
		accesskit_node_builder_set_placeholder(ae->builder, p_placeholder.utf8().ptr());
	} else {
		accesskit_node_builder_clear_placeholder(ae->builder);
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_language(const RID &p_id, const String &p_language) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_language(ae->builder, p_language.utf8().ptr());
}

void AccessibilityDriverAccessKit::accessibility_update_set_text_orientation(const RID &p_id, bool p_vertical) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (p_vertical) {
		accesskit_node_builder_set_text_direction(ae->builder, ACCESSKIT_TEXT_DIRECTION_TOP_TO_BOTTOM);
	} else {
		accesskit_node_builder_set_text_direction(ae->builder, ACCESSKIT_TEXT_DIRECTION_LEFT_TO_RIGHT);
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_list_orientation(const RID &p_id, bool p_vertical) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (p_vertical) {
		accesskit_node_builder_set_orientation(ae->builder, ACCESSKIT_ORIENTATION_VERTICAL);
	} else {
		accesskit_node_builder_set_orientation(ae->builder, ACCESSKIT_ORIENTATION_HORIZONTAL);
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_shortcut(const RID &p_id, const String &p_shortcut) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (!p_shortcut.is_empty()) {
		accesskit_node_builder_set_keyboard_shortcut(ae->builder, p_shortcut.utf8().ptr());
	} else {
		accesskit_node_builder_clear_keyboard_shortcut(ae->builder);
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_url(const RID &p_id, const String &p_url) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (!p_url.is_empty()) {
		accesskit_node_builder_set_url(ae->builder, p_url.utf8().ptr());
	} else {
		accesskit_node_builder_set_url(ae->builder, "");
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_role_description(const RID &p_id, const String &p_description) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (!p_description.is_empty()) {
		accesskit_node_builder_set_role_description(ae->builder, p_description.utf8().ptr());
	} else {
		accesskit_node_builder_set_role_description(ae->builder, "");
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_state_description(const RID &p_id, const String &p_description) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	if (!p_description.is_empty()) {
		accesskit_node_builder_set_state_description(ae->builder, p_description.utf8().ptr());
	} else {
		accesskit_node_builder_set_state_description(ae->builder, "");
	}
}

void AccessibilityDriverAccessKit::accessibility_update_set_color_value(const RID &p_id, const Color &p_color) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_color_value(ae->builder, p_color.to_rgba32());
}

void AccessibilityDriverAccessKit::accessibility_update_set_background_color(const RID &p_id, const Color &p_color) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_background_color(ae->builder, p_color.to_rgba32());
}

void AccessibilityDriverAccessKit::accessibility_update_set_foreground_color(const RID &p_id, const Color &p_color) {
	ERR_FAIL_COND_MSG(!in_accessibility_update, "Accessiblinity update is only allowed inside NOTIFICATION_ACCESSIBILITY_UPDATE notification.");

	AccessibilityElement *ae = rid_owner.get_or_null(p_id);
	ERR_FAIL_NULL(ae);
	_ensure_builder(p_id, ae);

	accesskit_node_builder_set_foreground_color(ae->builder, p_color.to_rgba32());
}

Error AccessibilityDriverAccessKit::init() {
	node_classes = accesskit_node_class_set_new();
	return OK;
}

AccessibilityDriverAccessKit::AccessibilityDriverAccessKit() {
	singleton = this;

	role_map[DisplayServer::AccessibilityRole::ROLE_UNKNOWN] = ACCESSKIT_ROLE_UNKNOWN;
	role_map[DisplayServer::AccessibilityRole::ROLE_DEFAULT_BUTTON] = ACCESSKIT_ROLE_DEFAULT_BUTTON;
	role_map[DisplayServer::AccessibilityRole::ROLE_AUDIO] = ACCESSKIT_ROLE_AUDIO;
	role_map[DisplayServer::AccessibilityRole::ROLE_VIDEO] = ACCESSKIT_ROLE_VIDEO;
	role_map[DisplayServer::AccessibilityRole::ROLE_STATIC_TEXT] = ACCESSKIT_ROLE_STATIC_TEXT;
	role_map[DisplayServer::AccessibilityRole::ROLE_CONTAINER] = ACCESSKIT_ROLE_GENERIC_CONTAINER;
	role_map[DisplayServer::AccessibilityRole::ROLE_PANEL] = ACCESSKIT_ROLE_PANE;
	role_map[DisplayServer::AccessibilityRole::ROLE_BUTTON] = ACCESSKIT_ROLE_BUTTON;
	role_map[DisplayServer::AccessibilityRole::ROLE_LINK] = ACCESSKIT_ROLE_LINK;
	role_map[DisplayServer::AccessibilityRole::ROLE_CHECK_BOX] = ACCESSKIT_ROLE_CHECK_BOX;
	role_map[DisplayServer::AccessibilityRole::ROLE_RADIO_BUTTON] = ACCESSKIT_ROLE_RADIO_BUTTON;
	role_map[DisplayServer::AccessibilityRole::ROLE_CHECK_BUTTON] = ACCESSKIT_ROLE_SWITCH;
	role_map[DisplayServer::AccessibilityRole::ROLE_TOGGLE_BUTTON] = ACCESSKIT_ROLE_TOGGLE_BUTTON;
	role_map[DisplayServer::AccessibilityRole::ROLE_SCROLL_BAR] = ACCESSKIT_ROLE_SCROLL_BAR;
	role_map[DisplayServer::AccessibilityRole::ROLE_SCROLL_VIEW] = ACCESSKIT_ROLE_SCROLL_VIEW;
	role_map[DisplayServer::AccessibilityRole::ROLE_SPLITTER] = ACCESSKIT_ROLE_SPLITTER;
	role_map[DisplayServer::AccessibilityRole::ROLE_SLIDER] = ACCESSKIT_ROLE_SLIDER;
	role_map[DisplayServer::AccessibilityRole::ROLE_SPIN_BUTTON] = ACCESSKIT_ROLE_SPIN_BUTTON;
	role_map[DisplayServer::AccessibilityRole::ROLE_PROGRESS_INDICATOR] = ACCESSKIT_ROLE_PROGRESS_INDICATOR;
	role_map[DisplayServer::AccessibilityRole::ROLE_TEXT_FIELD] = ACCESSKIT_ROLE_TEXT_INPUT;
	role_map[DisplayServer::AccessibilityRole::ROLE_MULTILINE_TEXT_FIELD] = ACCESSKIT_ROLE_MULTILINE_TEXT_INPUT;
	role_map[DisplayServer::AccessibilityRole::ROLE_COLOR_PICKER] = ACCESSKIT_ROLE_COLOR_WELL;
	role_map[DisplayServer::AccessibilityRole::ROLE_TABLE] = ACCESSKIT_ROLE_TABLE;
	role_map[DisplayServer::AccessibilityRole::ROLE_CELL] = ACCESSKIT_ROLE_CELL;
	role_map[DisplayServer::AccessibilityRole::ROLE_ROW] = ACCESSKIT_ROLE_ROW;
	role_map[DisplayServer::AccessibilityRole::ROLE_ROW_GROUP] = ACCESSKIT_ROLE_ROW_GROUP;
	role_map[DisplayServer::AccessibilityRole::ROLE_ROW_HEADER] = ACCESSKIT_ROLE_ROW_HEADER;
	role_map[DisplayServer::AccessibilityRole::ROLE_COLUMN] = ACCESSKIT_ROLE_COLUMN;
	role_map[DisplayServer::AccessibilityRole::ROLE_COLUMN_HEADER] = ACCESSKIT_ROLE_COLUMN_HEADER;
	role_map[DisplayServer::AccessibilityRole::ROLE_TREE] = ACCESSKIT_ROLE_TREE;
	role_map[DisplayServer::AccessibilityRole::ROLE_TREE_ITEM] = ACCESSKIT_ROLE_TREE_ITEM;
	role_map[DisplayServer::AccessibilityRole::ROLE_LIST] = ACCESSKIT_ROLE_LIST;
	role_map[DisplayServer::AccessibilityRole::ROLE_LIST_ITEM] = ACCESSKIT_ROLE_LIST_ITEM;
	role_map[DisplayServer::AccessibilityRole::ROLE_TAB_BAR] = ACCESSKIT_ROLE_TAB_LIST;
	role_map[DisplayServer::AccessibilityRole::ROLE_TAB] = ACCESSKIT_ROLE_TAB;
	role_map[DisplayServer::AccessibilityRole::ROLE_MENU_BAR] = ACCESSKIT_ROLE_MENU_BAR;
	role_map[DisplayServer::AccessibilityRole::ROLE_MENU] = ACCESSKIT_ROLE_MENU;
	role_map[DisplayServer::AccessibilityRole::ROLE_MENU_ITEM] = ACCESSKIT_ROLE_MENU_ITEM;
	role_map[DisplayServer::AccessibilityRole::ROLE_MENU_ITEM_CHECK_BOX] = ACCESSKIT_ROLE_MENU_ITEM_CHECK_BOX;
	role_map[DisplayServer::AccessibilityRole::ROLE_MENU_ITEM_RADIO] = ACCESSKIT_ROLE_MENU_ITEM_RADIO;
	role_map[DisplayServer::AccessibilityRole::ROLE_IMAGE] = ACCESSKIT_ROLE_IMAGE;
	role_map[DisplayServer::AccessibilityRole::ROLE_WINDOW] = ACCESSKIT_ROLE_WINDOW;
	role_map[DisplayServer::AccessibilityRole::ROLE_TITLE_BAR] = ACCESSKIT_ROLE_TITLE_BAR;
	role_map[DisplayServer::AccessibilityRole::ROLE_DIALOG] = ACCESSKIT_ROLE_DIALOG;
	role_map[DisplayServer::AccessibilityRole::ROLE_TOOLTIP] = ACCESSKIT_ROLE_TOOLTIP;

	action_verb_map[DisplayServer::AccessibilityActionVerb::ACTION_VERB_CLICK] = ACCESSKIT_DEFAULT_ACTION_VERB_CLICK;
	action_verb_map[DisplayServer::AccessibilityActionVerb::ACTION_VERB_FOCUS] = ACCESSKIT_DEFAULT_ACTION_VERB_FOCUS;
	action_verb_map[DisplayServer::AccessibilityActionVerb::ACTION_VERB_CHECK] = ACCESSKIT_DEFAULT_ACTION_VERB_CHECK;
	action_verb_map[DisplayServer::AccessibilityActionVerb::ACTION_VERB_UNCHECK] = ACCESSKIT_DEFAULT_ACTION_VERB_UNCHECK;
	action_verb_map[DisplayServer::AccessibilityActionVerb::ACTION_VERB_CLICK_ANCESTOR] = ACCESSKIT_DEFAULT_ACTION_VERB_CLICK_ANCESTOR;
	action_verb_map[DisplayServer::AccessibilityActionVerb::ACTION_VERB_JUMP] = ACCESSKIT_DEFAULT_ACTION_VERB_JUMP;
	action_verb_map[DisplayServer::AccessibilityActionVerb::ACTION_VERB_OPEN] = ACCESSKIT_DEFAULT_ACTION_VERB_OPEN;
	action_verb_map[DisplayServer::AccessibilityActionVerb::ACTION_VERB_PRESS] = ACCESSKIT_DEFAULT_ACTION_VERB_PRESS;
	action_verb_map[DisplayServer::AccessibilityActionVerb::ACTION_VERB_SELECT] = ACCESSKIT_DEFAULT_ACTION_VERB_SELECT;

	action_map[DisplayServer::AccessibilityAction::ACTION_DEFAULT] = ACCESSKIT_ACTION_DEFAULT;
	action_map[DisplayServer::AccessibilityAction::ACTION_FOCUS] = ACCESSKIT_ACTION_FOCUS;
	action_map[DisplayServer::AccessibilityAction::ACTION_BLUR] = ACCESSKIT_ACTION_BLUR;
	action_map[DisplayServer::AccessibilityAction::ACTION_COLLAPSE] = ACCESSKIT_ACTION_COLLAPSE;
	action_map[DisplayServer::AccessibilityAction::ACTION_EXPAND] = ACCESSKIT_ACTION_EXPAND;
	action_map[DisplayServer::AccessibilityAction::ACTION_DECREMENT] = ACCESSKIT_ACTION_DECREMENT;
	action_map[DisplayServer::AccessibilityAction::ACTION_INCREMENT] = ACCESSKIT_ACTION_INCREMENT;
	action_map[DisplayServer::AccessibilityAction::ACTION_HIDE_TOOLTIP] = ACCESSKIT_ACTION_HIDE_TOOLTIP;
	action_map[DisplayServer::AccessibilityAction::ACTION_SHOW_TOOLTIP] = ACCESSKIT_ACTION_SHOW_TOOLTIP;
	//action_map[DisplayServer::AccessibilityAction::ACTION_INVALIDATE_TREE] = ACCESSKIT_ACTION_INVALIDATE_TREE;
	//action_map[DisplayServer::AccessibilityAction::ACTION_LOAD_INLINE_TEXT_BOXES] = ACCESSKIT_ACTION_LOAD_INLINE_TEXT_BOXES;
	action_map[DisplayServer::AccessibilityAction::ACTION_SET_TEXT_SELECTION] = ACCESSKIT_ACTION_SET_TEXT_SELECTION;
	action_map[DisplayServer::AccessibilityAction::ACTION_REPLACE_SELECTED_TEXT] = ACCESSKIT_ACTION_REPLACE_SELECTED_TEXT;
	action_map[DisplayServer::AccessibilityAction::ACTION_SCROLL_BACKWARD] = ACCESSKIT_ACTION_SCROLL_BACKWARD;
	action_map[DisplayServer::AccessibilityAction::ACTION_SCROLL_DOWN] = ACCESSKIT_ACTION_SCROLL_DOWN;
	action_map[DisplayServer::AccessibilityAction::ACTION_SCROLL_FORWARD] = ACCESSKIT_ACTION_SCROLL_FORWARD;
	action_map[DisplayServer::AccessibilityAction::ACTION_SCROLL_LEFT] = ACCESSKIT_ACTION_SCROLL_LEFT;
	action_map[DisplayServer::AccessibilityAction::ACTION_SCROLL_RIGHT] = ACCESSKIT_ACTION_SCROLL_RIGHT;
	action_map[DisplayServer::AccessibilityAction::ACTION_SCROLL_UP] = ACCESSKIT_ACTION_SCROLL_UP;
	action_map[DisplayServer::AccessibilityAction::ACTION_SCROLL_INTO_VIEW] = ACCESSKIT_ACTION_SCROLL_INTO_VIEW;
	action_map[DisplayServer::AccessibilityAction::ACTION_SCROLL_TO_POINT] = ACCESSKIT_ACTION_SCROLL_TO_POINT;
	action_map[DisplayServer::AccessibilityAction::ACTION_SET_SCROLL_OFFSET] = ACCESSKIT_ACTION_SET_SCROLL_OFFSET;
	//action_map[DisplayServer::AccessibilityAction::ACTION_SET_SEQUENTIAL_FOCUS_NAVIGATION_STARTING_POINT] = ACCESSKIT_ACTION_SET_SEQUENTIAL_FOCUS_NAVIGATION_STARTING_POINT;
	action_map[DisplayServer::AccessibilityAction::ACTION_SET_VALUE] = ACCESSKIT_ACTION_SET_VALUE;
	action_map[DisplayServer::AccessibilityAction::ACTION_SHOW_CONTEXT_MENU] = ACCESSKIT_ACTION_SHOW_CONTEXT_MENU;
	action_map[DisplayServer::AccessibilityAction::ACTION_CUSTOM] = ACCESSKIT_ACTION_CUSTOM_ACTION;
}

AccessibilityDriverAccessKit::~AccessibilityDriverAccessKit() {
	if (node_classes) {
		accesskit_node_class_set_free(node_classes);
	}
	singleton = nullptr;
}

#endif // ACCESSKIT_ENABLED
