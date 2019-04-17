/*************************************************************************/
/*  connections_dialog.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "connections_dialog.h"

#include "core/print_string.h"
#include "editor_node.h"
#include "editor_settings.h"
#include "plugins/script_editor_plugin.h"
#include "scene/gui/label.h"
#include "scene/gui/popup_menu.h"

static Node *_find_first_script(Node *p_root, Node *p_node) {
	if (p_node != p_root && p_node->get_owner() != p_root) {
		return NULL;
	}
	if (!p_node->get_script().is_null()) {
		return p_node;
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {

		Node *ret = _find_first_script(p_root, p_node->get_child(i));
		if (ret) {
			return ret;
		}
	}

	return NULL;
}

class ConnectDialogBinds : public Object {

	GDCLASS(ConnectDialogBinds, Object);

	Vector<PropertyInfo> locked_params;
	Vector<Variant> params;

public:

	void init(Vector<PropertyInfo> p_properties, Vector<Variant> p_params) {
		locked_params.clear();
		params.clear();
		locked_params = p_properties;
		params = p_params;

		notify_changed();
	}

	Vector<PropertyInfo> get_locked_params() {
		return locked_params;
	}

	Vector<Variant> *get_params() {
		return &params;
	}

	bool _set(const StringName &p_name, const Variant &p_value) {

		String name = p_name;

		if (name.begins_with("bind/")) {
			int which = name.get_slice("/", 1).to_int() - 1;
			ERR_FAIL_INDEX_V(which, params.size(), false);
			params.write[which] = p_value;
		} else
			return false;

		return true;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {

		String name = p_name;

		if (name.begins_with("bind/")) {
			int which = name.get_slice("/", 1).to_int() - 1;
			ERR_FAIL_INDEX_V(which, params.size(), false);
			r_ret = params[which];

			return true;
		} else {
			for (int i = 0; i < locked_params.size(); i++) {
				if (locked_params[i].name == p_name) {
					r_ret = locked_params[i].class_name != StringName() ? locked_params[i].class_name : Variant::get_type_name(locked_params[i].type);

					return true;
				}
			}
		}

		return false;
	}

	void _get_property_list(List<PropertyInfo> *p_list) const {

		for (int i = 0; i < locked_params.size(); i++) {
			PropertyInfo info = locked_params[i];

			p_list->push_back(PropertyInfo(Variant::STRING, info.name, PROPERTY_HINT_PROPERTY_OF_VARIANT_TYPE, "", PROPERTY_USAGE_EDITOR));
		}

		for (int i = 0; i < params.size(); i++) {
			p_list->push_back(PropertyInfo(params[i].get_type(), "bind/" + itos(i + 1)));
		}
	}

	void notify_changed() {

		_change_notify();
	}

	ConnectDialogBinds() {
	}
};

/*
Adds a new parameter bind to connection.
*/
void ArgumentsBindsDialog::_add_bind() {

	if (cdbinds->get_params()->size() >= VARIANT_ARG_MAX)
		return;
	Variant::Type vt = (Variant::Type)type_list->get_item_id(type_list->get_selected());

	Variant value;

	switch (vt) {
		case Variant::BOOL: value = false; break;
		case Variant::INT: value = 0; break;
		case Variant::REAL: value = 0.0; break;
		case Variant::STRING: value = ""; break;
		case Variant::VECTOR2: value = Vector2(); break;
		case Variant::RECT2: value = Rect2(); break;
		case Variant::VECTOR3: value = Vector3(); break;
		case Variant::TRANSFORM2D: value = Transform2D(); break;
		case Variant::PLANE: value = Plane(); break;
		case Variant::QUAT: value = Quat(); break;
		case Variant::AABB: value = AABB(); break;
		case Variant::BASIS: value = Basis(); break;
		case Variant::TRANSFORM: value = Transform(); break;
		case Variant::COLOR: value = Color(); break;
		case Variant::NODE_PATH: value = NodePath(); break;
		case Variant::_RID: value = RID(); break;
		case Variant::OBJECT: value = Ref<Reference>(memnew(Reference)); break;
		case Variant::DICTIONARY: value = Dictionary(); break;
		case Variant::ARRAY: value = Array(); break;
		case Variant::POOL_BYTE_ARRAY: value = PoolByteArray(); break;
		case Variant::POOL_INT_ARRAY: value = PoolIntArray(); break;
		case Variant::POOL_REAL_ARRAY: value = PoolRealArray(); break;
		case Variant::POOL_STRING_ARRAY: value = PoolStringArray(); break;
		case Variant::POOL_VECTOR2_ARRAY: value = PoolVector2Array(); break;
		case Variant::POOL_VECTOR3_ARRAY: value = PoolVector3Array(); break;
		case Variant::POOL_COLOR_ARRAY: value = PoolColorArray(); break;
		default: {
			ERR_FAIL();
		} break;
	}

	ERR_FAIL_COND(value.get_type() == Variant::NIL);

	cdbinds->get_params()->push_back(value);
	cdbinds->notify_changed();

	emit_signal("bindings_changed");
}

/*
Remove parameter bind from connection.
*/
void ArgumentsBindsDialog::_remove_bind() {

	String st = bind_editor->get_selected_path();
	if (st == "")
		return;
	int idx = st.get_slice("/", 1).to_int() - 1;

	ERR_FAIL_INDEX(idx, cdbinds->get_params()->size());
	cdbinds->get_params()->remove(idx);
	cdbinds->notify_changed();

	emit_signal("bindings_changed");
}

void ArgumentsBindsDialog::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {
		bind_editor->edit(cdbinds);
	}
}

void ArgumentsBindsDialog::_bind_methods() {

	ClassDB::bind_method("_add_bind", &ArgumentsBindsDialog::_add_bind);
	ClassDB::bind_method("_remove_bind", &ArgumentsBindsDialog::_remove_bind);

	ADD_SIGNAL(MethodInfo("bindings_changed"));
}

Vector<PropertyInfo> ArgumentsBindsDialog::get_params() const {

	return cdbinds->get_locked_params();
}

Vector<Variant> ArgumentsBindsDialog::get_binds() const {

	return *cdbinds->get_params();
}

void ArgumentsBindsDialog::init(Vector<PropertyInfo> p_arguments, Vector<Variant> p_binds) {

	cdbinds->init(p_arguments, p_binds);
}

ArgumentsBindsDialog::ArgumentsBindsDialog() {

	vb_container = memnew(VBoxContainer);
	add_child(vb_container);
	vb_container->set_h_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *add_bind_hb = memnew(HBoxContainer);

	type_list = memnew(OptionButton);
	type_list->set_h_size_flags(SIZE_EXPAND_FILL);
	add_bind_hb->add_child(type_list);
	type_list->add_item("bool", Variant::BOOL);
	type_list->add_item("int", Variant::INT);
	type_list->add_item("real", Variant::REAL);
	type_list->add_item("string", Variant::STRING);
	type_list->add_item("Vector2", Variant::VECTOR2);
	type_list->add_item("Rect2", Variant::RECT2);
	type_list->add_item("Vector3", Variant::VECTOR3);
	type_list->add_item("Transform2D", Variant::TRANSFORM2D);
	type_list->add_item("Plane", Variant::PLANE);
	type_list->add_item("Quat", Variant::QUAT);
	type_list->add_item("AABB", Variant::AABB);
	type_list->add_item("Basis", Variant::BASIS);
	type_list->add_item("Transform", Variant::TRANSFORM);
	type_list->add_item("Color", Variant::COLOR);
	type_list->add_item("NodePath", Variant::NODE_PATH);
	type_list->add_item("RID", Variant::_RID);
	type_list->add_item("Object", Variant::OBJECT);
	type_list->add_item("Dictionary", Variant::DICTIONARY);
	type_list->add_item("Array", Variant::ARRAY);
	type_list->add_item("PoolByteArray", Variant::POOL_BYTE_ARRAY);
	type_list->add_item("PoolIntArray", Variant::POOL_INT_ARRAY);
	type_list->add_item("PoolRealArray", Variant::POOL_REAL_ARRAY);
	type_list->add_item("PoolStringArray", Variant::POOL_STRING_ARRAY);
	type_list->add_item("PoolVector2Array", Variant::POOL_VECTOR2_ARRAY);
	type_list->add_item("PoolVector3Array", Variant::POOL_VECTOR3_ARRAY);
	type_list->add_item("PoolColorArray", Variant::POOL_COLOR_ARRAY);
	type_list->select(0);

	Button *add_bind = memnew(Button);
	add_bind->set_text(TTR("Add"));
	add_bind_hb->add_child(add_bind);
	add_bind->connect("pressed", this, "_add_bind");

	Button *del_bind = memnew(Button);
	del_bind->set_text(TTR("Remove"));
	add_bind_hb->add_child(del_bind);
	del_bind->connect("pressed", this, "_remove_bind");

	vb_container->add_margin_child(TTR("Add Extra Call Argument:"), add_bind_hb);

	bind_editor = memnew(EditorInspector);

	vb_container->add_margin_child(TTR("Call Arguments:"), bind_editor, true);

	set_as_toplevel(true);

	cdbinds = memnew(ConnectDialogBinds);
}

ArgumentsBindsDialog::~ArgumentsBindsDialog() {

	memdelete(cdbinds);
}

//ConnectDialog ==========================

/*
Signal automatically called by parent dialog.
*/
void ConnectDialog::ok_pressed() {

	if (dst_method->get_text() == "") {
		error->set_text(TTR("Method in target Node must be specified!"));
		error->popup_centered_minsize();
		return;
	}
	Node *target = tree->get_selected();
	if (target->get_script().is_null()) {
		if (!target->has_method(dst_method->get_text())) {
			error->set_text(TTR("Target method not found! Specify a valid method or attach a script to target Node."));
			error->popup_centered_minsize();
			return;
		}
	}
	emit_signal("connected");
	hide();
}

void ConnectDialog::_cancel_pressed() {

	hide();
}

/*
Called each time a target node is selected within the target node tree.
*/
void ConnectDialog::_tree_node_selected() {

	Node *current = tree->get_selected();

	if (!current)
		return;

	dst_path = source->get_path_to(current);

	if (mode_list->get_selected() != Mode::NEW_METHOD && current) {
		method_selector->select_method_from_instance(current);
	}

	_check_valid();
}

void ConnectDialog::_mode_changed(int p_mode) {

	bool show_methods = p_mode != 0;

	right_container->set_visible(show_methods);
	dst_method->set_editable(!show_methods);

	//Rect2 new_rect;

	if (show_methods) {
		//new_rect.size = Size2(900, 500) * EDSCALE;
		connect_to_label->set_text(TTR("Connect To Node:"));
		tree->set_connect_mode(SceneTreeEditor::ConnectMode::CONNECT_TO_NODE);

		if (tree->get_selected()) {
			method_selector->select_method_from_instance(tree->get_selected());
		}
	} else {
		//new_rect.size = Size2(700, 500) * EDSCALE;
		connect_to_label->set_text(TTR("Connect To Script:"));
		tree->set_connect_mode(SceneTreeEditor::ConnectMode::CONNECT_TO_SCRIPT);
	}

	_check_valid();
	//set_position(((get_viewport_rect().size - new_rect.size) / 2.0 + get_position() + get_size() / 2.0 - get_viewport_rect().size / 2.0).floor());
	//set_size(new_rect.size);
}

void ConnectDialog::_method_text_changed(String p_method_text) {

	_check_valid();
}

void ConnectDialog::_method_selected(String p_method) {

	set_dst_method(p_method);
}

void ConnectDialog::_settings_flags_changed(int p_setting_idx) {

	settings->get_popup()->set_item_checked(p_setting_idx, !settings->get_popup()->is_item_checked(p_setting_idx));

	_set_settings_flags(settings->get_popup()->is_item_checked(CallMode::DEFERRED), settings->get_popup()->is_item_checked(CallMode::ONESHOT));
}

void ConnectDialog::_set_settings_flags(bool p_deferred, bool p_oneshot) {

	String menu_button_text = TTR("Default");

	if (!p_deferred && !p_oneshot) {
		settings->set_text(menu_button_text);
		return;
	}

	menu_button_text = "";

	if (p_deferred) {
		menu_button_text += settings->get_popup()->get_item_text(CallMode::DEFERRED) + ", ";
	}
	if (p_oneshot) {
		menu_button_text += settings->get_popup()->get_item_text(CallMode::ONESHOT) + ", ";
	}
	menu_button_text = menu_button_text.substr(0, menu_button_text.length() - 2);
	settings->set_text(menu_button_text);
}

void ConnectDialog::_edit_arguments_pressed() {

	args_dialog->set_title(TTR("Edit Signal Arguments"));
	args_dialog->popup_centered_ratio(0.5);
}

void ConnectDialog::_bindings_changed() {

	args_info->set_text(vformat(TTR("Connection has %d argument(s)."), args_dialog->get_params().size() + args_dialog->get_binds().size()));
}

void ConnectDialog::_bind_arguments_pressed() {


}

void ConnectDialog::_check_valid() {

	bool show_error = false;

	if (mode_list->get_selected_id() == Mode::EXISTING_METHOD) {

		Node *target = tree->get_selected();
		if (!target->has_method(dst_method->get_text())) {
			error_label->set_text(TTR("Method not found in the selected node."));
			show_error = true;
		}
	} else if (mode_list->get_selected_id() == Mode::NEW_METHOD) {

		Node *target = tree->get_selected();
		if (!target && !_find_first_script(get_tree()->get_edited_scene_root(), get_tree()->get_edited_scene_root())) {
			error_label->set_text(TTR("Scene does not contain any script."));
			show_error = true;
		} else if (target->has_method(dst_method->get_text())) {
			error_label->set_text(TTR("Method already defined in the selected node."));
			show_error = true;
		}
	}

	// Todo: enable or disable bind arguments button.

	// Todo: arguments check.

	if (show_error) {
		error_label->show();
		get_ok()->set_disabled(true);
	} else {
		error_label->hide();
		get_ok()->set_disabled(false);
	}
}

void ConnectDialog::_notification(int p_what) {

	if (p_what == NOTIFICATION_READY || p_what == NOTIFICATION_THEME_CHANGED) {
		settings->set_icon(Control::get_icon("arrow", "OptionButton"));
		edit_args->set_icon(Control::get_icon("Edit", "EditorIcons"));
	}
}

void ConnectDialog::_bind_methods() {

	ClassDB::bind_method("_mode_changed", &ConnectDialog::_mode_changed);
	ClassDB::bind_method("_method_text_changed", &ConnectDialog::_method_text_changed);
	ClassDB::bind_method("_method_selected", &ConnectDialog::_method_selected);
	ClassDB::bind_method("_settings_flags_changed", &ConnectDialog::_settings_flags_changed);
	ClassDB::bind_method("_cancel", &ConnectDialog::_cancel_pressed);
	ClassDB::bind_method("_tree_node_selected", &ConnectDialog::_tree_node_selected);
	ClassDB::bind_method("_edit_arguments_pressed", &ConnectDialog::_edit_arguments_pressed);
	ClassDB::bind_method("_bindings_changed", &ConnectDialog::_bindings_changed);
	ClassDB::bind_method("_bind_arguments_pressed", &ConnectDialog::_bind_arguments_pressed);

	ADD_SIGNAL(MethodInfo("connected"));
}

Node *ConnectDialog::get_source() const {

	return source;
}

StringName ConnectDialog::get_signal_name() const {

	return signal;
}

NodePath ConnectDialog::get_dst_path() const {

	return dst_path;
}

void ConnectDialog::set_dst_node(Node *p_node) {

	tree->set_selected(p_node);
}

StringName ConnectDialog::get_dst_method_name() const {

	String txt = dst_method->get_text();
	if (txt.find("(") != -1)
		txt = txt.left(txt.find("(")).strip_edges();
	return txt;
}

void ConnectDialog::set_dst_method(const StringName &p_method) {

	dst_method->set_text(p_method);

	_check_valid();
}

Vector<Variant> ConnectDialog::get_binds() const {

	return args_dialog->get_binds();
}

bool ConnectDialog::get_deferred() const {

	return settings->get_popup()->is_item_checked(CallMode::DEFERRED);
}

bool ConnectDialog::get_oneshot() const {

	return settings->get_popup()->is_item_checked(CallMode::ONESHOT);
}

/*
Returns true if ConnectDialog is being used to edit an existing connection.
*/
bool ConnectDialog::is_editing() const {

	return bEditMode;
}

/*
Initialize ConnectDialog and populate fields with expected data.
If creating a connection from scratch, sensible defaults are used.
If editing an existing connection, previous data is retained.
*/
void ConnectDialog::init(const String &p_for_signal, Connection c, int p_mode, bool bEdit) {

	from_signal->set_text(p_for_signal);
	error_label->add_color_override("font_color", get_color("error_color", "Editor"));

	int active_mode = p_mode;
	mode_list->set_item_disabled(Mode::NEW_METHOD, false);

	Ref<EditorFeatureProfile> profile = EditorFeatureProfileManager::get_singleton()->get_current_profile();
	if (!profile.is_null()) {

		if (profile->is_feature_disabled(EditorFeatureProfile::FEATURE_SCRIPT)) {

			active_mode = Mode::EXISTING_METHOD;
			mode_list->set_item_disabled(Mode::NEW_METHOD, true);
		}
	}

	source = static_cast<Node *>(c.source);
	signal = c.signal;

	// Retrieve signal arguments and bindings.
	Vector<PropertyInfo> arguments;
	List<MethodInfo> signal_list;
	c.source->get_signal_list(&signal_list);

	for (List<MethodInfo>::Element *E = signal_list.front(); E; E = E->next()) {
		if (E->get().name == signal) {
			for (List<PropertyInfo>::Element *A = E->get().arguments.front(); A; A = A->next()) {
				arguments.push_back(A->get());
			}
			break;
		}
	}

	args_dialog->init(arguments, c.binds);
	_bindings_changed();

	if (!c.target) {
		// Creating a brand new connection, selecting initial target node and method.
		if (active_mode == Mode::NEW_METHOD) {
			// Select first node found with a script.
			Node *dst_node = static_cast<Node *>(c.source);
			dst_node = dst_node->get_owner() ? dst_node->get_owner() : dst_node;

			if (!dst_node || dst_node->get_script().is_null()) {
				dst_node = _find_first_script(get_tree()->get_edited_scene_root(), get_tree()->get_edited_scene_root());
			}
			c.target = dst_node;
		} else if (active_mode == Mode::EXISTING_METHOD) {
			// Select itself and the its first method found.
			c.target = c.source;
			method_selector->select_method_from_instance(c.source);

			// First method in the PropertySelector is the root's grandchild.
			TreeItem *item = method_selector->get_search_options()->get_root()->get_children()->get_children();
			if (item) {
				c.method = item->get_metadata(0);
			}
		}
	}
	set_dst_node(static_cast<Node *>(c.target));
	set_dst_method(c.method);

	// Index and IDs are the same in this case.
	mode_list->select(active_mode);
	_mode_changed(active_mode);

	//_check_valid();

	tree->set_marked(source, true);

	bool bDeferred = (c.flags & CONNECT_DEFERRED) == CONNECT_DEFERRED;
	bool bOneshot = (c.flags & CONNECT_ONESHOT) == CONNECT_ONESHOT;

	settings->get_popup()->set_item_checked(CallMode::DEFERRED, bDeferred);
	settings->get_popup()->set_item_checked(CallMode::ONESHOT, bOneshot);
	_set_settings_flags(bDeferred, bOneshot);

	bEditMode = bEdit;
}

ConnectDialog::ConnectDialog() {

	set_h_grow_direction(GrowDirection::GROW_DIRECTION_BOTH);

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	HBoxContainer *main_hb = memnew(HBoxContainer);
	vbc->add_child(main_hb);
	main_hb->set_v_size_flags(SIZE_EXPAND_FILL);

	VBoxContainer *vbc_left = memnew(VBoxContainer);
	main_hb->add_child(vbc_left);
	vbc_left->set_h_size_flags(SIZE_EXPAND_FILL);

	from_signal = memnew(LineEdit);
	from_signal->set_editable(false);
	vbc_left->add_margin_child(TTR("From Signal:"), from_signal);

	tree = memnew(SceneTreeEditor(false));
	tree->get_scene_tree()->connect("item_activated", this, "_ok");
	tree->connect("node_selected", this, "_tree_node_selected");
	tree->set_show_enabled_subscene(true);
	tree->set_connect_mode(SceneTreeEditor::ConnectMode::CONNECT_TO_SCRIPT);

	Node *mc = vbc_left->add_margin_child(TTR("Connect To Script:"), tree, true);
	connect_to_label = Object::cast_to<Label>(vbc_left->get_child(mc->get_index() - 1));

	error_label = memnew(Label);
	error_label->set_text(TTR("Scene does not contain any script."));
	vbc_left->add_child(error_label);
	error_label->hide();

	right_container = memnew(VBoxContainer);
	main_hb->add_child(right_container);
	right_container->set_h_size_flags(SIZE_EXPAND_FILL);
	right_container->hide();

	method_selector = memnew(PropertySelector);
	right_container->add_child(method_selector);
	method_selector->set_h_size_flags(SIZE_EXPAND_FILL);
	method_selector->set_v_size_flags(SIZE_EXPAND_FILL);
	method_selector->connect("item_selected", this, "_method_selected");
	method_selector->connect("request_hide", this, "_closed");

	HBoxContainer *rbottom_hb = memnew(HBoxContainer);
	right_container->add_child(rbottom_hb);

	Button *bind_arguments = memnew(Button);
	bind_arguments->set_text(TTR("Bind Arguments (Dummy)"));
	rbottom_hb->add_child(bind_arguments);
	bind_arguments->set_h_size_flags(SIZE_EXPAND | SIZE_SHRINK_CENTER);
	bind_arguments->connect("pressed", this, "_bind_arguments_pressed");

	/*
	bind_info = memnew(Label);
	bind_info->set_text(TTR("Bindings will be added."));
	bind_info->add_color_override("font_color", EditorNode::get_singleton()->get_gui_base()->get_color("disabled_font_color", "Editor"));
	bind_info->set_h_size_flags(SIZE_EXPAND);
	rbottom_hb->add_child(bind_info);
	*/

	HBoxContainer *dstm_hb = memnew(HBoxContainer);
	vbc_left->add_margin_child("Connect To:", dstm_hb);

	mode_list = memnew(OptionButton);
	mode_list->set_h_size_flags(SIZE_FILL);
	dstm_hb->add_child(mode_list);
	mode_list->add_item(TTR("New Method"), Mode::NEW_METHOD);
	mode_list->add_item(TTR("Existing Method"), Mode::EXISTING_METHOD);
	mode_list->select(0);
	mode_list->connect("item_selected", this, "_mode_changed");

	dst_method = memnew(LineEdit);
	dst_method->set_h_size_flags(SIZE_EXPAND_FILL);
	dst_method->connect("text_changed", this, "_method_text_changed");
	dstm_hb->add_child(dst_method);

	settings = memnew(MenuButton);
	settings->set_h_size_flags(SIZE_FILL);
	settings->get_popup()->set_hide_on_checkable_item_selection(false);
	settings->get_popup()->add_check_item(TTR("Deferred"), CallMode::DEFERRED);
	settings->get_popup()->add_check_item(TTR("Oneshot"), CallMode::ONESHOT);
	settings->get_popup()->connect("index_pressed", this, "_settings_flags_changed");
	dstm_hb->add_child(settings);

	HBoxContainer *args_hb = memnew(HBoxContainer);
	args_hb->set_alignment(BoxContainer::AlignMode::ALIGN_END);
	vbc_left->add_child(args_hb);

	args_info = memnew(Label);
	args_info->set_text("");
	args_info->add_color_override("font_color", EditorNode::get_singleton()->get_gui_base()->get_color("disabled_font_color", "Editor"));
	args_info->set_h_size_flags(SIZE_EXPAND);
	args_hb->add_child(args_info);

	edit_args = memnew(ToolButton);
	edit_args->set_text(TTR("Edit Arguments..."));
	edit_args->set_h_size_flags(SIZE_FILL);
	edit_args->connect("pressed", this, "_edit_arguments_pressed");
	args_hb->add_child(edit_args);

	args_dialog = memnew(ArgumentsBindsDialog);
	args_dialog->set_as_toplevel(true);
	args_dialog->connect("bindings_changed", this, "_bindings_changed");
	add_child(args_dialog);

	set_as_toplevel(true);

	error = memnew(ConfirmationDialog);
	add_child(error);
	error->get_ok()->set_text(TTR("Close"));
	get_ok()->set_text(TTR("Connect"));
}

ConnectDialog::~ConnectDialog() { }

//ConnectionsDock ==========================

struct _ConnectionsDockMethodInfoSort {

	_FORCE_INLINE_ bool operator()(const MethodInfo &a, const MethodInfo &b) const {
		return a.name < b.name;
	}
};

/*
Post-ConnectDialog callback for creating/editing connections.
Creates or edits connections based on state of the ConnectDialog when "Connect" is pressed.
*/
void ConnectionsDock::_make_or_edit_connection() {

	TreeItem *it = tree->get_selected();
	ERR_FAIL_COND(!it);

	NodePath dst_path = connect_dialog->get_dst_path();
	Node *target = selectedNode->get_node(dst_path);
	ERR_FAIL_COND(!target);

	Connection cToMake;
	cToMake.source = connect_dialog->get_source();
	cToMake.target = target;
	cToMake.signal = connect_dialog->get_signal_name();
	cToMake.method = connect_dialog->get_dst_method_name();
	cToMake.binds = connect_dialog->get_binds();
	bool defer = connect_dialog->get_deferred();
	bool oshot = connect_dialog->get_oneshot();
	cToMake.flags = CONNECT_PERSIST | (defer ? CONNECT_DEFERRED : 0) | (oshot ? CONNECT_ONESHOT : 0);

	//conditions to add function, must have a script and must have a method
	bool add_script_function = !target->get_script().is_null() && !ClassDB::has_method(target->get_class(), cToMake.method);
	PoolStringArray script_function_args;
	if (add_script_function) {
		// pick up args here before "it" is deleted by update_tree
		script_function_args = it->get_metadata(0).operator Dictionary()["args"];
		for (int i = 0; i < cToMake.binds.size(); i++) {
			script_function_args.append("extra_arg_" + itos(i));
		}
	}

	if (connect_dialog->is_editing()) {
		_disconnect(*it);
		_connect(cToMake);
	} else {
		_connect(cToMake);
	}

	// IMPORTANT NOTE: _disconnect and _connect cause an update_tree,
	// which will delete the object "it" is pointing to
	it = NULL;

	if (add_script_function) {
		editor->emit_signal("script_add_function_request", target, cToMake.method, script_function_args);
		hide();
	}

	update_tree();
}

/*
Creates single connection w/ undo-redo functionality.
*/
void ConnectionsDock::_connect(Connection cToMake) {

	Node *source = static_cast<Node *>(cToMake.source);
	Node *target = static_cast<Node *>(cToMake.target);

	if (!source || !target)
		return;

	undo_redo->create_action(vformat(TTR("Connect '%s' to '%s'"), String(cToMake.signal), String(cToMake.method)));

	undo_redo->add_do_method(source, "connect", cToMake.signal, target, cToMake.method, cToMake.binds, cToMake.flags);
	undo_redo->add_undo_method(source, "disconnect", cToMake.signal, target, cToMake.method);
	undo_redo->add_do_method(this, "update_tree");
	undo_redo->add_undo_method(this, "update_tree");
	undo_redo->add_do_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree"); //to force redraw of scene tree
	undo_redo->add_undo_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree");

	undo_redo->commit_action();
}

/*
Break single connection w/ undo-redo functionality.
*/
void ConnectionsDock::_disconnect(TreeItem &item) {

	Connection c = item.get_metadata(0);
	ERR_FAIL_COND(c.source != selectedNode); //shouldn't happen but...bugcheck

	undo_redo->create_action(vformat(TTR("Disconnect '%s' from '%s'"), c.signal, c.method));

	undo_redo->add_do_method(selectedNode, "disconnect", c.signal, c.target, c.method);
	undo_redo->add_undo_method(selectedNode, "connect", c.signal, c.target, c.method, c.binds, c.flags);
	undo_redo->add_do_method(this, "update_tree");
	undo_redo->add_undo_method(this, "update_tree");
	undo_redo->add_do_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree"); //to force redraw of scene tree
	undo_redo->add_undo_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree");

	undo_redo->commit_action();
}

/*
Break all connections of currently selected signal.
Can undo-redo as a single action.
*/
void ConnectionsDock::_disconnect_all() {

	TreeItem *item = tree->get_selected();

	if (!_is_item_signal(*item))
		return;

	TreeItem *child = item->get_children();
	String signalName = item->get_metadata(0).operator Dictionary()["name"];
	undo_redo->create_action(vformat(TTR("Disconnect all from signal: '%s'"), signalName));

	while (child) {
		Connection c = child->get_metadata(0);
		undo_redo->add_do_method(selectedNode, "disconnect", c.signal, c.target, c.method);
		undo_redo->add_undo_method(selectedNode, "connect", c.signal, c.target, c.method, c.binds, c.flags);
		child = child->get_next();
	}

	undo_redo->add_do_method(this, "update_tree");
	undo_redo->add_undo_method(this, "update_tree");
	undo_redo->add_do_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree");
	undo_redo->add_undo_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree");

	undo_redo->commit_action();
}

void ConnectionsDock::_tree_item_selected() {

	TreeItem *item = tree->get_selected();
	if (!item) { //Unlikely. Disable button just in case.
		connect_button->set_text(TTR("Connect..."));
		connect_button->set_disabled(true);
	} else if (_is_item_signal(*item)) {
		connect_button->set_text(TTR("Connect..."));
		connect_button->set_disabled(false);
	} else {
		connect_button->set_text(TTR("Disconnect"));
		connect_button->set_disabled(false);
	}
}

void ConnectionsDock::_tree_item_activated() { //"Activation" on double-click.

	TreeItem *item = tree->get_selected();

	if (!item)
		return;

	if (_is_item_signal(*item)) {
		_open_connection_dialog(*item);
	} else {
		_go_to_script(*item);
	}
}

bool ConnectionsDock::_is_item_signal(TreeItem &item) {

	return (item.get_parent() == tree->get_root() || item.get_parent()->get_parent() == tree->get_root());
}

/*
Open connection dialog with TreeItem data to CREATE a brand-new connection.
*/

void ConnectionsDock::_open_connection_dialog(TreeItem &item) {

	String signal = item.get_metadata(0).operator Dictionary()["name"];
	String signalname = signal;
	String midname = selectedNode->get_name();
	for (int i = 0; i < midname.length(); i++) { //TODO: Regex filter may be cleaner.
		CharType c = midname[i];
		if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) {
			if (c == ' ') {
				//Replace spaces with underlines.
				c = '_';
			} else {
				//Remove any other characters.
				midname.remove(i);
				i--;
				continue;
			}
		}
		midname[i] = c;
	}

	/*
	Node *dst_node = selectedNode->get_owner() ? selectedNode->get_owner() : selectedNode;
	if (!dst_node || dst_node->get_script().is_null()) {
		dst_node = _find_first_script(get_tree()->get_edited_scene_root(), get_tree()->get_edited_scene_root());
	}
	*/

	StringName dst_method = "_on_" + midname + "_" + signal;

	Connection c;
	c.source = selectedNode;
	c.signal = StringName(signalname);
	c.target = NULL;
	c.method = dst_method;
	
	connect_dialog->init(signalname, c, ConnectDialog::Mode::NEW_METHOD, false);
	connect_dialog->set_title(TTR("Connect a Signal to a Method"));

	connect_dialog->popup_centered_ratio();
}

/*
Open connection dialog with Connection data to EDIT an existing connection.
*/
void ConnectionsDock::_open_connection_dialog(Connection cToEdit) {

	Node *src = static_cast<Node *>(cToEdit.source);
	Node *dst = static_cast<Node *>(cToEdit.target);

	if (src && dst) {
		connect_dialog->init(cToEdit.signal, cToEdit, ConnectDialog::Mode::EXISTING_METHOD, true);
		connect_dialog->set_title(TTR("Edit Connection:") + cToEdit.signal);

		connect_dialog->popup_centered_ratio();
	}
}

/*
Open slot method location in script editor.
*/
void ConnectionsDock::_go_to_script(TreeItem &item) {

	if (_is_item_signal(item))
		return;

	Connection c = item.get_metadata(0);
	ERR_FAIL_COND(c.source != selectedNode); //shouldn't happen but...bugcheck

	if (!c.target)
		return;

	Ref<Script> script = c.target->get_script();

	if (script.is_null())
		return;

	if (script.is_valid() && ScriptEditor::get_singleton()->script_goto_method(script, c.method)) {
		editor->call("_editor_select", EditorNode::EDITOR_SCRIPT);
	}
}

void ConnectionsDock::_handle_signal_menu_option(int option) {

	TreeItem *item = tree->get_selected();

	if (!item)
		return;

	switch (option) {
		case CONNECT: {
			_open_connection_dialog(*item);
		} break;
		case DISCONNECT_ALL: {
			StringName signal_name = item->get_metadata(0).operator Dictionary()["name"];
			disconnect_all_dialog->set_text(vformat(TTR("Are you sure you want to remove all connections from the \"%s\" signal?"), signal_name));
			disconnect_all_dialog->popup_centered();
		} break;
	}
}

void ConnectionsDock::_handle_slot_menu_option(int option) {

	TreeItem *item = tree->get_selected();

	if (!item)
		return;

	switch (option) {
		case EDIT: {
			Connection c = item->get_metadata(0);
			_open_connection_dialog(c);
		} break;
		case GO_TO_SCRIPT: {
			_go_to_script(*item);
		} break;
		case DISCONNECT: {
			_disconnect(*item);
			update_tree();
		} break;
	}
}

void ConnectionsDock::_rmb_pressed(Vector2 position) {

	TreeItem *item = tree->get_selected();

	if (!item)
		return;

	Vector2 global_position = tree->get_global_position() + position;

	if (_is_item_signal(*item)) {
		signal_menu->set_position(global_position);
		signal_menu->popup();
	} else {
		slot_menu->set_position(global_position);
		slot_menu->popup();
	}
}

void ConnectionsDock::_close() {

	hide();
}

void ConnectionsDock::_connect_pressed() {

	TreeItem *item = tree->get_selected();
	if (!item) {
		connect_button->set_disabled(true);
		return;
	}

	if (_is_item_signal(*item)) {
		_open_connection_dialog(*item);
	} else {
		_disconnect(*item);
		update_tree();
	}
}

void ConnectionsDock::_notification(int p_what) {

	if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		update_tree();
	}
}

void ConnectionsDock::_bind_methods() {

	ClassDB::bind_method("_make_or_edit_connection", &ConnectionsDock::_make_or_edit_connection);
	ClassDB::bind_method("_disconnect_all", &ConnectionsDock::_disconnect_all);
	ClassDB::bind_method("_tree_item_selected", &ConnectionsDock::_tree_item_selected);
	ClassDB::bind_method("_tree_item_activated", &ConnectionsDock::_tree_item_activated);
	ClassDB::bind_method("_handle_signal_menu_option", &ConnectionsDock::_handle_signal_menu_option);
	ClassDB::bind_method("_handle_slot_menu_option", &ConnectionsDock::_handle_slot_menu_option);
	ClassDB::bind_method("_rmb_pressed", &ConnectionsDock::_rmb_pressed);
	ClassDB::bind_method("_close", &ConnectionsDock::_close);
	ClassDB::bind_method("_connect_pressed", &ConnectionsDock::_connect_pressed);
	ClassDB::bind_method("update_tree", &ConnectionsDock::update_tree);
}

void ConnectionsDock::set_node(Node *p_node) {

	selectedNode = p_node;
	update_tree();
}

void ConnectionsDock::update_tree() {

	tree->clear();

	if (!selectedNode)
		return;

	TreeItem *root = tree->create_item();

	List<MethodInfo> node_signals;

	selectedNode->get_signal_list(&node_signals);

	//node_signals.sort_custom<_ConnectionsDockMethodInfoSort>();
	bool did_script = false;
	StringName base = selectedNode->get_class();

	while (base) {

		List<MethodInfo> node_signals2;
		Ref<Texture> icon;
		String name;

		if (!did_script) {

			Ref<Script> scr = selectedNode->get_script();
			if (scr.is_valid()) {
				scr->get_script_signal_list(&node_signals2);
				if (scr->get_path().is_resource_file())
					name = scr->get_path().get_file();
				else
					name = scr->get_class();

				if (has_icon(scr->get_class(), "EditorIcons")) {
					icon = get_icon(scr->get_class(), "EditorIcons");
				}
			}

		} else {

			ClassDB::get_signal_list(base, &node_signals2, true);
			if (has_icon(base, "EditorIcons")) {
				icon = get_icon(base, "EditorIcons");
			}
			name = base;
		}

		TreeItem *pitem = NULL;

		if (node_signals2.size()) {
			pitem = tree->create_item(root);
			pitem->set_text(0, name);
			pitem->set_icon(0, icon);
			pitem->set_selectable(0, false);
			pitem->set_editable(0, false);
			pitem->set_custom_bg_color(0, get_color("prop_subsection", "Editor"));
			node_signals2.sort();
		}

		for (List<MethodInfo>::Element *E = node_signals2.front(); E; E = E->next()) {

			MethodInfo &mi = E->get();

			String signaldesc;
			signaldesc = mi.name + "(";
			PoolStringArray argnames;
			if (mi.arguments.size()) {
				signaldesc += " ";
				for (int i = 0; i < mi.arguments.size(); i++) {

					PropertyInfo &pi = mi.arguments[i];

					if (i > 0)
						signaldesc += ", ";
					String tname = "var";
					if (pi.type == Variant::OBJECT && pi.class_name != StringName()) {
						tname = pi.class_name.operator String();
					} else if (pi.type != Variant::NIL) {
						tname = Variant::get_type_name(pi.type);
					}
					signaldesc += tname + " " + (pi.name == "" ? String("arg " + itos(i)) : pi.name);
					argnames.push_back(pi.name + ":" + tname);
				}
				signaldesc += " ";
			}

			signaldesc += ")";

			TreeItem *item = tree->create_item(pitem);
			item->set_text(0, signaldesc);
			Dictionary sinfo;
			sinfo["name"] = mi.name;
			sinfo["args"] = argnames;
			item->set_metadata(0, sinfo);
			item->set_icon(0, get_icon("Signal", "EditorIcons"));

			List<Object::Connection> connections;
			selectedNode->get_signal_connection_list(mi.name, &connections);

			for (List<Object::Connection>::Element *F = connections.front(); F; F = F->next()) {

				Object::Connection &c = F->get();
				if (!(c.flags & CONNECT_PERSIST))
					continue;

				Node *target = Object::cast_to<Node>(c.target);
				if (!target)
					continue;

				String path = String(selectedNode->get_path_to(target)) + " :: " + c.method + "()";
				if (c.flags & CONNECT_DEFERRED)
					path += " (deferred)";
				if (c.flags & CONNECT_ONESHOT)
					path += " (oneshot)";
				if (c.binds.size()) {

					path += " binds( ";
					for (int i = 0; i < c.binds.size(); i++) {

						if (i > 0)
							path += ", ";
						path += c.binds[i].operator String();
					}
					path += " )";
				}

				TreeItem *item2 = tree->create_item(item);
				item2->set_text(0, path);
				item2->set_metadata(0, c);
				item2->set_icon(0, get_icon("Slot", "EditorIcons"));
			}
		}

		if (!did_script) {
			did_script = true;
		} else {
			base = ClassDB::get_parent_class(base);
		}
	}

	connect_button->set_text(TTR("Connect"));
	connect_button->set_disabled(true);
}

ConnectionsDock::ConnectionsDock(EditorNode *p_editor) {

	editor = p_editor;
	set_name(TTR("Signals"));

	VBoxContainer *vbc = this;

	tree = memnew(Tree);
	tree->set_columns(1);
	tree->set_select_mode(Tree::SELECT_ROW);
	tree->set_hide_root(true);
	vbc->add_child(tree);
	tree->set_v_size_flags(SIZE_EXPAND_FILL);
	tree->set_allow_rmb_select(true);

	connect_button = memnew(Button);
	connect_button->set_text(TTR("Connect"));
	HBoxContainer *hb = memnew(HBoxContainer);
	vbc->add_child(hb);
	hb->add_spacer();
	hb->add_child(connect_button);
	connect_button->connect("pressed", this, "_connect_pressed");

	connect_dialog = memnew(ConnectDialog);
	connect_dialog->set_as_toplevel(true);
	add_child(connect_dialog);

	disconnect_all_dialog = memnew(ConfirmationDialog);
	disconnect_all_dialog->set_as_toplevel(true);
	add_child(disconnect_all_dialog);
	disconnect_all_dialog->connect("confirmed", this, "_disconnect_all");
	disconnect_all_dialog->set_text(TTR("Are you sure you want to remove all connections from this signal?"));

	signal_menu = memnew(PopupMenu);
	add_child(signal_menu);
	signal_menu->connect("id_pressed", this, "_handle_signal_menu_option");
	signal_menu->add_item(TTR("Connect..."), CONNECT);
	signal_menu->add_item(TTR("Disconnect All"), DISCONNECT_ALL);

	slot_menu = memnew(PopupMenu);
	add_child(slot_menu);
	slot_menu->connect("id_pressed", this, "_handle_slot_menu_option");
	slot_menu->add_item(TTR("Edit..."), EDIT);
	slot_menu->add_item(TTR("Go To Method"), GO_TO_SCRIPT);
	slot_menu->add_item(TTR("Disconnect"), DISCONNECT);

	/*
	node_only->set_anchor( MARGIN_TOP, ANCHOR_END );
	node_only->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
	node_only->set_anchor( MARGIN_RIGHT, ANCHOR_END );

	node_only->set_begin( Point2( 20,51) );
	node_only->set_end( Point2( 10,44) );
	*/

	connect_dialog->connect("connected", this, "_make_or_edit_connection");
	tree->connect("item_selected", this, "_tree_item_selected");
	tree->connect("item_activated", this, "_tree_item_activated");
	tree->connect("item_rmb_selected", this, "_rmb_pressed");

	add_constant_override("separation", 3 * EDSCALE);
}

ConnectionsDock::~ConnectionsDock() {
}
