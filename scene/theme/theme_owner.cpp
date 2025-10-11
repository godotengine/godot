/**************************************************************************/
/*  theme_owner.cpp                                                       */
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

#include "theme_owner.h"

#include "scene/gui/control.h"
#include "scene/main/window.h"
#include "scene/theme/theme_db.h"

// Theme owner node.

void ThemeOwner::set_owner_node(Node *p_node) {
	ERR_FAIL_COND(p_node && !Object::cast_to<Control>(p_node) && !Object::cast_to<Window>(p_node));
	owner_node = p_node;
}

void ThemeOwner::set_owner_context(ThemeContext *p_context, bool p_propagate) {
	ThemeContext *default_context = ThemeDB::get_singleton()->get_default_theme_context();

	Callable callback = callable_mp_static_with_subject(holder, &ThemeOwner::_owner_context_changed);

	if (owner_context && owner_context->is_connected(CoreStringName(changed), callback)) {
		owner_context->disconnect(CoreStringName(changed), callback);
	} else if (default_context->is_connected(CoreStringName(changed), callback)) {
		default_context->disconnect(CoreStringName(changed), callback);
	}

	owner_context = p_context;

	if (owner_context) {
		owner_context->connect(CoreStringName(changed), callback);
	} else {
		default_context->connect(CoreStringName(changed), callback);
	}

	if (p_propagate) {
		_owner_context_changed(holder);
	}
}

void ThemeOwner::_owner_context_changed(Node *p_node) {
	if (!p_node->is_inside_tree()) {
		// We ignore theme changes outside of tree, because NOTIFICATION_ENTER_TREE covers everything.
		return;
	}

	Control *c = Object::cast_to<Control>(p_node);
	Window *w = c == nullptr ? Object::cast_to<Window>(p_node) : nullptr;

	if (c) {
		c->notification(Control::NOTIFICATION_THEME_CHANGED);
	} else if (w) {
		w->notification(Window::NOTIFICATION_THEME_CHANGED);
	}
}

ThemeContext *ThemeOwner::_get_active_owner_context() const {
	if (owner_context) {
		return owner_context;
	}

	return ThemeDB::get_singleton()->get_default_theme_context();
}

// Theme propagation.

void ThemeOwner::assign_theme_on_parented(Node *p_for_node) {
	// We check if there are any themes affecting the parent. If that's the case
	// its children also need to be affected.
	// We don't notify here because `NOTIFICATION_THEME_CHANGED` will be handled
	// a bit later by `NOTIFICATION_ENTER_TREE`.

	Node *parent = p_for_node->get_parent();

	Control *parent_c = Object::cast_to<Control>(parent);
	if (parent_c && parent_c->has_theme_owner_node()) {
		propagate_theme_changed(p_for_node, parent_c->get_theme_owner_node(), false, true);
	} else {
		Window *parent_w = Object::cast_to<Window>(parent);
		if (parent_w && parent_w->has_theme_owner_node()) {
			propagate_theme_changed(p_for_node, parent_w->get_theme_owner_node(), false, true);
		}
	}
}

void ThemeOwner::clear_theme_on_unparented(Node *p_for_node) {
	// We check if there were any themes affecting the parent. If that's the case
	// its children need were also affected and need to be updated.
	// We don't notify because we're exiting the tree, and it's not important.

	Node *parent = p_for_node->get_parent();

	Control *parent_c = Object::cast_to<Control>(parent);
	if (parent_c && parent_c->has_theme_owner_node()) {
		propagate_theme_changed(p_for_node, nullptr, false, true);
	} else {
		Window *parent_w = Object::cast_to<Window>(parent);
		if (parent_w && parent_w->has_theme_owner_node()) {
			propagate_theme_changed(p_for_node, nullptr, false, true);
		}
	}
}

void ThemeOwner::propagate_theme_changed(Node *p_to_node, Node *p_owner_node, bool p_notify, bool p_assign) {
	Control *c = Object::cast_to<Control>(p_to_node);
	Window *w = c == nullptr ? Object::cast_to<Window>(p_to_node) : nullptr;

	if (!c && !w) {
		// Theme inheritance chains are broken by nodes that aren't Control or Window.
		return;
	}

	bool assign = p_assign;
	if (c) {
		if (c != p_owner_node && c->get_theme().is_valid()) {
			// Has a theme, so we don't want to change the theme owner,
			// but we still want to propagate in case this child has theme items
			// it inherits from the theme this node uses.
			// See https://github.com/godotengine/godot/issues/62844.
			assign = false;
		}

		if (assign) {
			c->set_theme_owner_node(p_owner_node);
		}

		if (p_notify) {
			c->notification(Control::NOTIFICATION_THEME_CHANGED);
		}
	} else if (w) {
		if (w != p_owner_node && w->get_theme().is_valid()) {
			// Same as above.
			assign = false;
		}

		if (assign) {
			w->set_theme_owner_node(p_owner_node);
		}

		if (p_notify) {
			w->notification(Window::NOTIFICATION_THEME_CHANGED);
		}
	}

	for (int i = 0; i < p_to_node->get_child_count(); i++) {
		propagate_theme_changed(p_to_node->get_child(i), p_owner_node, p_notify, assign);
	}
}

// Theme lookup.

void ThemeOwner::get_theme_type_dependencies(const Node *p_for_node, const StringName &p_theme_type, Vector<StringName> &r_result) const {
	const Control *for_c = Object::cast_to<Control>(p_for_node);
	const Window *for_w = Object::cast_to<Window>(p_for_node);
	ERR_FAIL_COND_MSG(!for_c && !for_w, "Only Control and Window nodes and derivatives can be polled for theming.");

	StringName type_name = p_for_node->get_class_name();
	StringName type_variation;
	if (for_c) {
		type_variation = for_c->get_theme_type_variation();
	} else if (for_w) {
		type_variation = for_w->get_theme_type_variation();
	}

	// If we are looking for dependencies of the current class (or a variation of it), check relevant themes.
	if (p_theme_type == StringName() || p_theme_type == type_name || p_theme_type == type_variation) {
		// We need one theme that can give us a valid dependency chain. It must be complete
		// (i.e. variations can depend on other variations, but only within the same theme,
		// and eventually the chain must lead to native types).

		// First, look through themes owned by nodes in the tree.
		Node *current_owner = owner_node;

		while (current_owner) {
			Ref<Theme> owner_theme = _get_owner_node_theme(current_owner);
			if (owner_theme.is_valid() && owner_theme->get_type_variation_base(type_variation) != StringName()) {
				owner_theme->get_type_dependencies(type_name, type_variation, r_result);
				return;
			}

			current_owner = _get_next_owner_node(current_owner);
		}

		// Second, check global contexts.
		ThemeContext *global_context = _get_active_owner_context();
		for (const Ref<Theme> &theme : global_context->get_themes()) {
			if (theme.is_valid() && theme->get_type_variation_base(type_variation) != StringName()) {
				theme->get_type_dependencies(type_name, type_variation, r_result);
				return;
			}
		}

		// If nothing was found, get the native dependencies for the current class.
		ThemeDB::get_singleton()->get_native_type_dependencies(type_name, r_result);
		return;
	}

	// Otherwise, get the native dependencies for the provided theme type.
	ThemeDB::get_singleton()->get_native_type_dependencies(p_theme_type, r_result);
}

Variant ThemeOwner::get_theme_item_in_types(Theme::DataType p_data_type, const StringName &p_name, const Vector<StringName> &p_theme_types) const {
	ERR_FAIL_COND_V_MSG(p_theme_types.is_empty(), Variant(), "At least one theme type must be specified.");

	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	Node *current_owner = owner_node;

	while (current_owner) {
		// For each theme resource check the theme types provided and see if p_name exists with any of them.
		for (const StringName &E : p_theme_types) {
			Ref<Theme> owner_theme = _get_owner_node_theme(current_owner);

			if (owner_theme.is_valid() && owner_theme->has_theme_item(p_data_type, p_name, E)) {
				return owner_theme->get_theme_item(p_data_type, p_name, E);
			}
		}

		current_owner = _get_next_owner_node(current_owner);
	}

	// Second, check global themes from the appropriate context.
	ThemeContext *global_context = _get_active_owner_context();
	for (const Ref<Theme> &theme : global_context->get_themes()) {
		if (theme.is_valid()) {
			for (const StringName &E : p_theme_types) {
				if (theme->has_theme_item(p_data_type, p_name, E)) {
					return theme->get_theme_item(p_data_type, p_name, E);
				}
			}
		}
	}

	// Finally, if no match exists, use any type to return the default/empty value.
	return global_context->get_fallback_theme()->get_theme_item(p_data_type, p_name, StringName());
}

bool ThemeOwner::has_theme_item_in_types(Theme::DataType p_data_type, const StringName &p_name, const Vector<StringName> &p_theme_types) const {
	ERR_FAIL_COND_V_MSG(p_theme_types.is_empty(), false, "At least one theme type must be specified.");

	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	Node *current_owner = owner_node;

	while (current_owner) {
		// For each theme resource check the theme types provided and see if p_name exists with any of them.
		for (const StringName &E : p_theme_types) {
			Ref<Theme> owner_theme = _get_owner_node_theme(current_owner);

			if (owner_theme.is_valid() && owner_theme->has_theme_item(p_data_type, p_name, E)) {
				return true;
			}
		}

		current_owner = _get_next_owner_node(current_owner);
	}

	// Second, check global themes from the appropriate context.
	ThemeContext *global_context = _get_active_owner_context();
	for (const Ref<Theme> &theme : global_context->get_themes()) {
		if (theme.is_valid()) {
			for (const StringName &E : p_theme_types) {
				if (theme->has_theme_item(p_data_type, p_name, E)) {
					return true;
				}
			}
		}
	}

	// Finally, if no match exists, return false.
	return false;
}

float ThemeOwner::get_theme_default_base_scale() const {
	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	// For each theme resource see if their assigned theme has the default value defined and valid.
	Node *current_owner = owner_node;

	while (current_owner) {
		Ref<Theme> owner_theme = _get_owner_node_theme(current_owner);

		if (owner_theme.is_valid() && owner_theme->has_default_base_scale()) {
			return owner_theme->get_default_base_scale();
		}

		current_owner = _get_next_owner_node(current_owner);
	}

	// Second, check global themes from the appropriate context.
	ThemeContext *global_context = _get_active_owner_context();
	for (const Ref<Theme> &theme : global_context->get_themes()) {
		if (theme.is_valid()) {
			if (theme->has_default_base_scale()) {
				return theme->get_default_base_scale();
			}
		}
	}

	// Finally, if no match exists, return the universal default.
	return ThemeDB::get_singleton()->get_fallback_base_scale();
}

Ref<Font> ThemeOwner::get_theme_default_font() const {
	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	// For each theme resource see if their assigned theme has the default value defined and valid.
	Node *current_owner = owner_node;

	while (current_owner) {
		Ref<Theme> owner_theme = _get_owner_node_theme(current_owner);

		if (owner_theme.is_valid() && owner_theme->has_default_font()) {
			return owner_theme->get_default_font();
		}

		current_owner = _get_next_owner_node(current_owner);
	}

	// Second, check global themes from the appropriate context.
	ThemeContext *global_context = _get_active_owner_context();
	for (const Ref<Theme> &theme : global_context->get_themes()) {
		if (theme.is_valid()) {
			if (theme->has_default_font()) {
				return theme->get_default_font();
			}
		}
	}

	// Finally, if no match exists, return the universal default.
	return ThemeDB::get_singleton()->get_fallback_font();
}

int ThemeOwner::get_theme_default_font_size() const {
	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	// For each theme resource see if their assigned theme has the default value defined and valid.
	Node *current_owner = owner_node;

	while (current_owner) {
		Ref<Theme> owner_theme = _get_owner_node_theme(current_owner);

		if (owner_theme.is_valid() && owner_theme->has_default_font_size()) {
			return owner_theme->get_default_font_size();
		}

		current_owner = _get_next_owner_node(current_owner);
	}

	// Second, check global themes from the appropriate context.
	ThemeContext *global_context = _get_active_owner_context();
	for (const Ref<Theme> &theme : global_context->get_themes()) {
		if (theme.is_valid()) {
			if (theme->has_default_font_size()) {
				return theme->get_default_font_size();
			}
		}
	}

	// Finally, if no match exists, return the universal default.
	return ThemeDB::get_singleton()->get_fallback_font_size();
}

Ref<Theme> ThemeOwner::_get_owner_node_theme(Node *p_owner_node) const {
	const Control *owner_c = Object::cast_to<Control>(p_owner_node);
	if (owner_c) {
		return owner_c->get_theme();
	}

	const Window *owner_w = Object::cast_to<Window>(p_owner_node);
	if (owner_w) {
		return owner_w->get_theme();
	}

	return Ref<Theme>();
}

Node *ThemeOwner::_get_next_owner_node(Node *p_from_node) const {
	Node *parent = p_from_node->get_parent();

	Control *parent_c = Object::cast_to<Control>(parent);
	if (parent_c) {
		return parent_c->get_theme_owner_node();
	} else {
		Window *parent_w = Object::cast_to<Window>(parent);
		if (parent_w) {
			return parent_w->get_theme_owner_node();
		}
	}

	return nullptr;
}
