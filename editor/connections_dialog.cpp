/*************************************************************************/
/*  connections_dialog.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/string/print_string.h"
#include "editor/doc_tools.h"
#include "editor_node.h"
#include "editor_scale.h"
#include "editor_settings.h"
#include "plugins/script_editor_plugin.h"
#include "scene/gui/label.h"
#include "scene/gui/popup_menu.h"

static Node *_find_first_script(Node *p_root, Node *p_node) {
	if (p_node != p_root && p_node->get_owner() != p_root) {
		return nullptr;
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

	return nullptr;
}

class ConnectDialogBinds : public Object {
	GDCLASS(ConnectDialogBinds, Object);

public:
	Vector<Variant> params;

	bool _set(const StringName &p_name, const Variant &p_value) {
		String name = p_name;

		if (name.begins_with("bind/argument_")) {
			int which = name.get_slice("_", 1).to_int() - 1;
			ERR_FAIL_INDEX_V(which, params.size(), false);
			params.write[which] = p_value;
		} else {
			return false;
		}

		return true;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {
		String name = p_name;

		if (name.begins_with("bind/argument_")) {
			int which = name.get_slice("_", 1).to_int() - 1;
			ERR_FAIL_INDEX_V(which, params.size(), false);
			r_ret = params[which];
		} else {
			return false;
		}

		return true;
	}

	void _get_property_list(List<PropertyInfo> *p_list) const {
		for (int i = 0; i < params.size(); i++) {
			p_list->push_back(PropertyInfo(params[i].get_type(), "bind/argument_" + itos(i + 1)));
		}
	}

	void notify_changed() {
		notify_property_list_changed();
	}

	ConnectDialogBinds() {
	}
};

/*
 * Signal automatically called by parent dialog.
 */
void ConnectDialog::ok_pressed() {
	String method_name = dst_method->get_text();

	if (method_name.is_empty()) {
		error->set_text(TTR("Method in target node must be specified."));
		error->popup_centered();
		return;
	}

	if (!method_name.strip_edges().is_valid_identifier()) {
		error->set_text(TTR("Method name must be a valid identifier."));
		error->popup_centered();
		return;
	}

	Node *target = tree->get_selected();
	if (!target) {
		return; // Nothing selected in the tree, not an error.
	}
	if (target->get_script().is_null()) {
		if (!target->has_method(method_name)) {
			error->set_text(TTR("Target method not found. Specify a valid method or attach a script to the target node."));
			error->popup_centered();
			return;
		}
	}
	emit_signal(SNAME("connected"));
	hide();
}

void ConnectDialog::_cancel_pressed() {
	hide();
}

void ConnectDialog::_item_activated() {
	_ok_pressed(); // From AcceptDialog.
}

void ConnectDialog::_text_submitted(const String &p_text) {
	_ok_pressed(); // From AcceptDialog.
}

/*
 * Called each time a target node is selected within the target node tree.
 */
void ConnectDialog::_tree_node_selected() {
	Node *current = tree->get_selected();

	if (!current) {
		return;
	}

	dst_path = source->get_path_to(current);
	_update_ok_enabled();
}

/*
 * Adds a new parameter bind to connection.
 */
void ConnectDialog::_add_bind() {
	if (cdbinds->params.size() >= VARIANT_ARG_MAX) {
		return;
	}
	Variant::Type vt = (Variant::Type)type_list->get_item_id(type_list->get_selected());

	Variant value;

	switch (vt) {
		case Variant::BOOL:
			value = false;
			break;
		case Variant::INT:
			value = 0;
			break;
		case Variant::FLOAT:
			value = 0.0;
			break;
		case Variant::STRING:
			value = "";
			break;
		case Variant::STRING_NAME:
			value = "";
			break;
		case Variant::VECTOR2:
			value = Vector2();
			break;
		case Variant::RECT2:
			value = Rect2();
			break;
		case Variant::VECTOR3:
			value = Vector3();
			break;
		case Variant::PLANE:
			value = Plane();
			break;
		case Variant::QUATERNION:
			value = Quaternion();
			break;
		case Variant::AABB:
			value = AABB();
			break;
		case Variant::BASIS:
			value = Basis();
			break;
		case Variant::TRANSFORM3D:
			value = Transform3D();
			break;
		case Variant::COLOR:
			value = Color();
			break;
		default: {
			ERR_FAIL();
		} break;
	}

	ERR_FAIL_COND(value.get_type() == Variant::NIL);

	cdbinds->params.push_back(value);
	cdbinds->notify_changed();
}

/*
 * Remove parameter bind from connection.
 */
void ConnectDialog::_remove_bind() {
	String st = bind_editor->get_selected_path();
	if (st.is_empty()) {
		return;
	}
	int idx = st.get_slice("/", 1).to_int() - 1;

	ERR_FAIL_INDEX(idx, cdbinds->params.size());
	cdbinds->params.remove_at(idx);
	cdbinds->notify_changed();
}

/*
 * Enables or disables the connect button. The connect button is enabled if a
 * node is selected and valid in the selected mode.
 */
void ConnectDialog::_update_ok_enabled() {
	Node *target = tree->get_selected();

	if (target == nullptr) {
		get_ok_button()->set_disabled(true);
		return;
	}

	if (!advanced->is_pressed() && target->get_script().is_null()) {
		get_ok_button()->set_disabled(true);
		return;
	}

	get_ok_button()->set_disabled(false);
}

void ConnectDialog::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		bind_editor->edit(cdbinds);
	}
}

void ConnectDialog::_bind_methods() {
	ClassDB::bind_method("_cancel", &ConnectDialog::_cancel_pressed);
	ClassDB::bind_method("_update_ok_enabled", &ConnectDialog::_update_ok_enabled);

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
	if (txt.find("(") != -1) {
		txt = txt.left(txt.find("(")).strip_edges();
	}
	return txt;
}

void ConnectDialog::set_dst_method(const StringName &p_method) {
	dst_method->set_text(p_method);
}

Vector<Variant> ConnectDialog::get_binds() const {
	return cdbinds->params;
}

bool ConnectDialog::get_deferred() const {
	return deferred->is_pressed();
}

bool ConnectDialog::get_oneshot() const {
	return oneshot->is_pressed();
}

/*
 * Returns true if ConnectDialog is being used to edit an existing connection.
 */
bool ConnectDialog::is_editing() const {
	return bEditMode;
}

/*
 * Initialize ConnectDialog and populate fields with expected data.
 * If creating a connection from scratch, sensible defaults are used.
 * If editing an existing connection, previous data is retained.
 */
void ConnectDialog::init(ConnectionData c, bool bEdit) {
	set_hide_on_ok(false);

	source = static_cast<Node *>(c.source);
	signal = c.signal;

	tree->set_selected(nullptr);
	tree->set_marked(source, true);

	if (c.target) {
		set_dst_node(static_cast<Node *>(c.target));
		set_dst_method(c.method);
	}

	_update_ok_enabled();

	bool bDeferred = (c.flags & CONNECT_DEFERRED) == CONNECT_DEFERRED;
	bool bOneshot = (c.flags & CONNECT_ONESHOT) == CONNECT_ONESHOT;

	deferred->set_pressed(bDeferred);
	oneshot->set_pressed(bOneshot);

	cdbinds->params.clear();
	cdbinds->params = c.binds;
	cdbinds->notify_changed();

	bEditMode = bEdit;
}

void ConnectDialog::popup_dialog(const String &p_for_signal) {
	from_signal->set_text(p_for_signal);
	error_label->add_theme_color_override("font_color", error_label->get_theme_color(SNAME("error_color"), SNAME("Editor")));
	if (!advanced->is_pressed()) {
		error_label->set_visible(!_find_first_script(get_tree()->get_edited_scene_root(), get_tree()->get_edited_scene_root()));
	}

	popup_centered();
}

void ConnectDialog::_advanced_pressed() {
	if (advanced->is_pressed()) {
		set_min_size(Size2(900, 500) * EDSCALE);
		connect_to_label->set_text(TTR("Connect to Node:"));
		tree->set_connect_to_script_mode(false);

		vbc_right->show();
		error_label->hide();
	} else {
		set_min_size(Size2(600, 500) * EDSCALE);
		reset_size();
		connect_to_label->set_text(TTR("Connect to Script:"));
		tree->set_connect_to_script_mode(true);

		vbc_right->hide();
		error_label->set_visible(!_find_first_script(get_tree()->get_edited_scene_root(), get_tree()->get_edited_scene_root()));
	}

	_update_ok_enabled();

	popup_centered();
}

ConnectDialog::ConnectDialog() {
	set_min_size(Size2(600, 500) * EDSCALE);

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	HBoxContainer *main_hb = memnew(HBoxContainer);
	vbc->add_child(main_hb);
	main_hb->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	VBoxContainer *vbc_left = memnew(VBoxContainer);
	main_hb->add_child(vbc_left);
	vbc_left->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	from_signal = memnew(LineEdit);
	from_signal->set_editable(false);
	vbc_left->add_margin_child(TTR("From Signal:"), from_signal);

	tree = memnew(SceneTreeEditor(false));
	tree->set_connecting_signal(true);
	tree->set_show_enabled_subscene(true);
	tree->get_scene_tree()->connect("item_activated", callable_mp(this, &ConnectDialog::_item_activated));
	tree->connect("node_selected", callable_mp(this, &ConnectDialog::_tree_node_selected));
	tree->set_connect_to_script_mode(true);

	Node *mc = vbc_left->add_margin_child(TTR("Connect to Script:"), tree, true);
	connect_to_label = Object::cast_to<Label>(vbc_left->get_child(mc->get_index() - 1));

	error_label = memnew(Label);
	error_label->set_text(TTR("Scene does not contain any script."));
	vbc_left->add_child(error_label);
	error_label->hide();

	vbc_right = memnew(VBoxContainer);
	main_hb->add_child(vbc_right);
	vbc_right->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	vbc_right->hide();

	HBoxContainer *add_bind_hb = memnew(HBoxContainer);

	type_list = memnew(OptionButton);
	type_list->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_bind_hb->add_child(type_list);
	type_list->add_item("bool", Variant::BOOL);
	type_list->add_item("int", Variant::INT);
	type_list->add_item("real", Variant::FLOAT);
	type_list->add_item("String", Variant::STRING);
	type_list->add_item("StringName", Variant::STRING_NAME);
	type_list->add_item("Vector2", Variant::VECTOR2);
	type_list->add_item("Rect2", Variant::RECT2);
	type_list->add_item("Vector3", Variant::VECTOR3);
	type_list->add_item("Plane", Variant::PLANE);
	type_list->add_item("Quaternion", Variant::QUATERNION);
	type_list->add_item("AABB", Variant::AABB);
	type_list->add_item("Basis", Variant::BASIS);
	type_list->add_item("Transform3D", Variant::TRANSFORM3D);
	type_list->add_item("Color", Variant::COLOR);
	type_list->select(0);

	Button *add_bind = memnew(Button);
	add_bind->set_text(TTR("Add"));
	add_bind_hb->add_child(add_bind);
	add_bind->connect("pressed", callable_mp(this, &ConnectDialog::_add_bind));

	Button *del_bind = memnew(Button);
	del_bind->set_text(TTR("Remove"));
	add_bind_hb->add_child(del_bind);
	del_bind->connect("pressed", callable_mp(this, &ConnectDialog::_remove_bind));

	vbc_right->add_margin_child(TTR("Add Extra Call Argument:"), add_bind_hb);

	bind_editor = memnew(EditorInspector);

	vbc_right->add_margin_child(TTR("Extra Call Arguments:"), bind_editor, true);

	HBoxContainer *dstm_hb = memnew(HBoxContainer);
	vbc_left->add_margin_child(TTR("Receiver Method:"), dstm_hb);

	dst_method = memnew(LineEdit);
	dst_method->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	dst_method->connect("text_submitted", callable_mp(this, &ConnectDialog::_text_submitted));
	dstm_hb->add_child(dst_method);

	advanced = memnew(CheckButton);
	dstm_hb->add_child(advanced);
	advanced->set_text(TTR("Advanced"));
	advanced->connect("pressed", callable_mp(this, &ConnectDialog::_advanced_pressed));

	deferred = memnew(CheckBox);
	deferred->set_h_size_flags(0);
	deferred->set_text(TTR("Deferred"));
	deferred->set_tooltip(TTR("Defers the signal, storing it in a queue and only firing it at idle time."));
	vbc_right->add_child(deferred);

	oneshot = memnew(CheckBox);
	oneshot->set_h_size_flags(0);
	oneshot->set_text(TTR("Oneshot"));
	oneshot->set_tooltip(TTR("Disconnects the signal after its first emission."));
	vbc_right->add_child(oneshot);

	cdbinds = memnew(ConnectDialogBinds);

	error = memnew(AcceptDialog);
	add_child(error);
	error->set_title(TTR("Cannot connect signal"));
	error->get_ok_button()->set_text(TTR("Close"));
	get_ok_button()->set_text(TTR("Connect"));
}

ConnectDialog::~ConnectDialog() {
	memdelete(cdbinds);
}

//////////////////////////////////////////

// Originally copied and adapted from EditorProperty, try to keep style in sync.
Control *ConnectionsDockTree::make_custom_tooltip(const String &p_text) const {
	EditorHelpBit *help_bit = memnew(EditorHelpBit);
	help_bit->add_theme_style_override("panel", get_theme_stylebox(SNAME("panel"), SNAME("TooltipPanel")));
	help_bit->get_rich_text()->set_fixed_size_to_width(360 * EDSCALE);

	String text = TTR("Signal:") + " [u][b]" + p_text.get_slice("::", 0) + "[/b][/u]";
	text += p_text.get_slice("::", 1).strip_edges() + "\n";
	text += p_text.get_slice("::", 2).strip_edges();
	help_bit->call_deferred(SNAME("set_text"), text); //hack so it uses proper theme once inside scene
	return help_bit;
}

struct _ConnectionsDockMethodInfoSort {
	_FORCE_INLINE_ bool operator()(const MethodInfo &a, const MethodInfo &b) const {
		return a.name < b.name;
	}
};

void ConnectionsDock::_filter_changed(const String &p_text) {
	update_tree();
}

/*
 * Post-ConnectDialog callback for creating/editing connections.
 * Creates or edits connections based on state of the ConnectDialog when "Connect" is pressed.
 */
void ConnectionsDock::_make_or_edit_connection() {
	TreeItem *it = tree->get_selected();
	ERR_FAIL_COND(!it);

	NodePath dst_path = connect_dialog->get_dst_path();
	Node *target = selectedNode->get_node(dst_path);
	ERR_FAIL_COND(!target);

	ConnectDialog::ConnectionData cToMake;
	cToMake.source = connect_dialog->get_source();
	cToMake.target = target;
	cToMake.signal = connect_dialog->get_signal_name();
	cToMake.method = connect_dialog->get_dst_method_name();
	cToMake.binds = connect_dialog->get_binds();
	bool defer = connect_dialog->get_deferred();
	bool oshot = connect_dialog->get_oneshot();
	cToMake.flags = CONNECT_PERSIST | (defer ? CONNECT_DEFERRED : 0) | (oshot ? CONNECT_ONESHOT : 0);

	// Conditions to add function: must have a script and must not have the method already
	// (in the class, the script itself, or inherited).
	bool add_script_function = false;
	Ref<Script> script = target->get_script();
	if (!target->get_script().is_null() && !ClassDB::has_method(target->get_class(), cToMake.method)) {
		// There is a chance that the method is inherited from another script.
		bool found_inherited_function = false;
		Ref<Script> inherited_script = script->get_base_script();
		while (!inherited_script.is_null()) {
			int line = inherited_script->get_language()->find_function(cToMake.method, inherited_script->get_source_code());
			if (line != -1) {
				found_inherited_function = true;
				break;
			}

			inherited_script = inherited_script->get_base_script();
		}

		add_script_function = !found_inherited_function;
	}
	PackedStringArray script_function_args;
	if (add_script_function) {
		// Pick up args here before "it" is deleted by update_tree.
		script_function_args = it->get_metadata(0).operator Dictionary()["args"];
		for (int i = 0; i < cToMake.binds.size(); i++) {
			script_function_args.push_back("extra_arg_" + itos(i) + ":" + Variant::get_type_name(cToMake.binds[i].get_type()));
		}
	}

	if (connect_dialog->is_editing()) {
		_disconnect(*it);
		_connect(cToMake);
	} else {
		_connect(cToMake);
	}

	// IMPORTANT NOTE: _disconnect and _connect cause an update_tree, which will delete the object "it" is pointing to.
	it = nullptr;

	if (add_script_function) {
		editor->emit_signal(SNAME("script_add_function_request"), target, cToMake.method, script_function_args);
		hide();
	}

	update_tree();
}

/*
 * Creates single connection w/ undo-redo functionality.
 */
void ConnectionsDock::_connect(ConnectDialog::ConnectionData cToMake) {
	Node *source = static_cast<Node *>(cToMake.source);
	Node *target = static_cast<Node *>(cToMake.target);

	if (!source || !target) {
		return;
	}

	undo_redo->create_action(vformat(TTR("Connect '%s' to '%s'"), String(cToMake.signal), String(cToMake.method)));

	Callable c(target, cToMake.method);

	undo_redo->add_do_method(source, "connect", cToMake.signal, c, cToMake.binds, cToMake.flags);
	undo_redo->add_undo_method(source, "disconnect", cToMake.signal, c);
	undo_redo->add_do_method(this, "update_tree");
	undo_redo->add_undo_method(this, "update_tree");
	undo_redo->add_do_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree"); //to force redraw of scene tree
	undo_redo->add_undo_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree");

	undo_redo->commit_action();
}

/*
 * Break single connection w/ undo-redo functionality.
 */
void ConnectionsDock::_disconnect(TreeItem &item) {
	Connection cd = item.get_metadata(0);
	ConnectDialog::ConnectionData c = cd;

	ERR_FAIL_COND(c.source != selectedNode); // Shouldn't happen but... Bugcheck.

	undo_redo->create_action(vformat(TTR("Disconnect '%s' from '%s'"), c.signal, c.method));

	undo_redo->add_do_method(selectedNode, "disconnect", c.signal, Callable(c.target, c.method));
	undo_redo->add_undo_method(selectedNode, "connect", c.signal, Callable(c.target, c.method), c.binds, c.flags);
	undo_redo->add_do_method(this, "update_tree");
	undo_redo->add_undo_method(this, "update_tree");
	undo_redo->add_do_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree"); // To force redraw of scene tree.
	undo_redo->add_undo_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree");

	undo_redo->commit_action();
}

/*
 * Break all connections of currently selected signal.
 * Can undo-redo as a single action.
 */
void ConnectionsDock::_disconnect_all() {
	TreeItem *item = tree->get_selected();

	if (!_is_item_signal(*item)) {
		return;
	}

	TreeItem *child = item->get_first_child();
	String signalName = item->get_metadata(0).operator Dictionary()["name"];
	undo_redo->create_action(vformat(TTR("Disconnect all from signal: '%s'"), signalName));

	while (child) {
		Connection cd = child->get_metadata(0);
		ConnectDialog::ConnectionData c = cd;
		undo_redo->add_do_method(selectedNode, "disconnect", c.signal, Callable(c.target, c.method));
		undo_redo->add_undo_method(selectedNode, "connect", c.signal, Callable(c.target, c.method), c.binds, c.flags);
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
	if (!item) { // Unlikely. Disable button just in case.
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

void ConnectionsDock::_tree_item_activated() { // "Activation" on double-click.

	TreeItem *item = tree->get_selected();

	if (!item) {
		return;
	}

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
 * Open connection dialog with TreeItem data to CREATE a brand-new connection.
 */
void ConnectionsDock::_open_connection_dialog(TreeItem &item) {
	String signal = item.get_metadata(0).operator Dictionary()["name"];
	const String &signalname = signal;
	String midname = selectedNode->get_name();
	for (int i = 0; i < midname.length(); i++) { //TODO: Regex filter may be cleaner.
		char32_t c = midname[i];
		if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) {
			if (c == ' ') {
				// Replace spaces with underlines.
				c = '_';
			} else {
				// Remove any other characters.
				midname.remove_at(i);
				i--;
				continue;
			}
		}
		midname[i] = c;
	}

	Node *dst_node = selectedNode->get_owner() ? selectedNode->get_owner() : selectedNode;
	if (!dst_node || dst_node->get_script().is_null()) {
		dst_node = _find_first_script(get_tree()->get_edited_scene_root(), get_tree()->get_edited_scene_root());
	}

	StringName dst_method = "_on_" + midname + "_" + signal;

	ConnectDialog::ConnectionData c;
	c.source = selectedNode;
	c.signal = StringName(signalname);
	c.target = dst_node;
	c.method = dst_method;
	connect_dialog->popup_dialog(signalname);
	connect_dialog->init(c);
	connect_dialog->set_title(TTR("Connect a Signal to a Method"));
}

/*
 * Open connection dialog with Connection data to EDIT an existing connection.
 */
void ConnectionsDock::_open_connection_dialog(ConnectDialog::ConnectionData cToEdit) {
	Node *src = static_cast<Node *>(cToEdit.source);
	Node *dst = static_cast<Node *>(cToEdit.target);

	if (src && dst) {
		const String &signalname = cToEdit.signal;
		connect_dialog->set_title(TTR("Edit Connection:") + cToEdit.signal);
		connect_dialog->popup_dialog(signalname);
		connect_dialog->init(cToEdit, true);
	}
}

/*
 * Open slot method location in script editor.
 */
void ConnectionsDock::_go_to_script(TreeItem &item) {
	if (_is_item_signal(item)) {
		return;
	}

	Connection cd = item.get_metadata(0);
	ConnectDialog::ConnectionData c = cd;
	ERR_FAIL_COND(c.source != selectedNode); //shouldn't happen but...bugcheck

	if (!c.target) {
		return;
	}

	Ref<Script> script = c.target->get_script();

	if (script.is_null()) {
		return;
	}

	if (script.is_valid() && ScriptEditor::get_singleton()->script_goto_method(script, c.method)) {
		editor->call("_editor_select", EditorNode::EDITOR_SCRIPT);
	}
}

void ConnectionsDock::_handle_signal_menu_option(int option) {
	TreeItem *item = tree->get_selected();

	if (!item) {
		return;
	}

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

	if (!item) {
		return;
	}

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

	if (!item) {
		return;
	}

	Vector2 screen_position = tree->get_screen_position() + position;

	if (_is_item_signal(*item)) {
		signal_menu->set_position(screen_position);
		signal_menu->reset_size();
		signal_menu->popup();
	} else {
		slot_menu->set_position(screen_position);
		slot_menu->reset_size();
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
	ClassDB::bind_method("update_tree", &ConnectionsDock::update_tree);
}

void ConnectionsDock::set_node(Node *p_node) {
	selectedNode = p_node;
	update_tree();
}

void ConnectionsDock::update_tree() {
	tree->clear();

	if (!selectedNode) {
		return;
	}

	TreeItem *root = tree->create_item();

	List<MethodInfo> node_signals;

	selectedNode->get_signal_list(&node_signals);

	bool did_script = false;
	StringName base = selectedNode->get_class();

	while (base) {
		List<MethodInfo> node_signals2;
		Ref<Texture2D> icon;
		String name;

		if (!did_script) {
			// Get script signals (including signals from any base scripts).
			Ref<Script> scr = selectedNode->get_script();
			if (scr.is_valid()) {
				scr->get_script_signal_list(&node_signals2);
				if (scr->get_path().is_resource_file()) {
					name = scr->get_path().get_file();
				} else {
					name = scr->get_class();
				}

				if (has_theme_icon(scr->get_class(), "EditorIcons")) {
					icon = get_theme_icon(scr->get_class(), "EditorIcons");
				}
			}
		} else {
			ClassDB::get_signal_list(base, &node_signals2, true);
			if (has_theme_icon(base, SNAME("EditorIcons"))) {
				icon = get_theme_icon(base, SNAME("EditorIcons"));
			}
			name = base;
		}

		if (!icon.is_valid()) {
			icon = get_theme_icon(SNAME("Object"), SNAME("EditorIcons"));
		}

		TreeItem *section_item = nullptr;

		// Create subsections.
		if (node_signals2.size()) {
			section_item = tree->create_item(root);
			section_item->set_text(0, name);
			section_item->set_icon(0, icon);
			section_item->set_selectable(0, false);
			section_item->set_editable(0, false);
			section_item->set_custom_bg_color(0, get_theme_color(SNAME("prop_subsection"), SNAME("Editor")));
			node_signals2.sort();
		}

		for (MethodInfo &mi : node_signals2) {
			StringName signal_name = mi.name;
			String signaldesc = "(";
			PackedStringArray argnames;

			String filter_text = search_box->get_text();
			if (!filter_text.is_subsequence_ofi(signal_name)) {
				continue;
			}

			if (mi.arguments.size()) {
				for (int i = 0; i < mi.arguments.size(); i++) {
					PropertyInfo &pi = mi.arguments[i];

					if (i > 0) {
						signaldesc += ", ";
					}
					String tname = "var";
					if (pi.type == Variant::OBJECT && pi.class_name != StringName()) {
						tname = pi.class_name.operator String();
					} else if (pi.type != Variant::NIL) {
						tname = Variant::get_type_name(pi.type);
					}
					signaldesc += (pi.name.is_empty() ? String("arg " + itos(i)) : pi.name) + ": " + tname;
					argnames.push_back(pi.name + ":" + tname);
				}
			}
			signaldesc += ")";

			// Create the children of the subsection - the actual list of signals.
			TreeItem *signal_item = tree->create_item(section_item);
			signal_item->set_text(0, String(signal_name) + signaldesc);
			Dictionary sinfo;
			sinfo["name"] = signal_name;
			sinfo["args"] = argnames;
			signal_item->set_metadata(0, sinfo);
			signal_item->set_icon(0, get_theme_icon(SNAME("Signal"), SNAME("EditorIcons")));

			// Set tooltip with the signal's documentation.
			{
				String descr;
				bool found = false;

				Map<StringName, Map<StringName, String>>::Element *G = descr_cache.find(base);
				if (G) {
					Map<StringName, String>::Element *F = G->get().find(signal_name);
					if (F) {
						found = true;
						descr = F->get();
					}
				}

				if (!found) {
					DocTools *dd = EditorHelp::get_doc_data();
					Map<String, DocData::ClassDoc>::Element *F = dd->class_list.find(base);
					while (F && descr.is_empty()) {
						for (int i = 0; i < F->get().signals.size(); i++) {
							if (F->get().signals[i].name == signal_name.operator String()) {
								descr = DTR(F->get().signals[i].description);
								break;
							}
						}
						if (!F->get().inherits.is_empty()) {
							F = dd->class_list.find(F->get().inherits);
						} else {
							break;
						}
					}
					descr_cache[base][signal_name] = descr;
				}

				// "::" separators used in make_custom_tooltip for formatting.
				signal_item->set_tooltip(0, String(signal_name) + "::" + signaldesc + "::" + descr);
			}

			// List existing connections
			List<Object::Connection> connections;
			selectedNode->get_signal_connection_list(signal_name, &connections);

			for (const Object::Connection &F : connections) {
				Connection cn = F;
				if (!(cn.flags & CONNECT_PERSIST)) {
					continue;
				}
				ConnectDialog::ConnectionData c = cn;

				Node *target = Object::cast_to<Node>(c.target);
				if (!target) {
					continue;
				}

				String path = String(selectedNode->get_path_to(target)) + " :: " + c.method + "()";
				if (c.flags & CONNECT_DEFERRED) {
					path += " (deferred)";
				}
				if (c.flags & CONNECT_ONESHOT) {
					path += " (oneshot)";
				}
				if (c.binds.size()) {
					path += " binds(";
					for (int i = 0; i < c.binds.size(); i++) {
						if (i > 0) {
							path += ", ";
						}
						path += c.binds[i].operator String();
					}
					path += ")";
				}

				TreeItem *connection_item = tree->create_item(signal_item);
				connection_item->set_text(0, path);
				Connection cd = c;
				connection_item->set_metadata(0, cd);
				connection_item->set_icon(0, get_theme_icon(SNAME("Slot"), SNAME("EditorIcons")));
			}
		}

		if (!did_script) {
			did_script = true;
		} else {
			base = ClassDB::get_parent_class(base);
		}
	}

	connect_button->set_text(TTR("Connect..."));
	connect_button->set_disabled(true);
}

ConnectionsDock::ConnectionsDock(EditorNode *p_editor) {
	editor = p_editor;
	set_name(TTR("Signals"));

	VBoxContainer *vbc = this;

	search_box = memnew(LineEdit);
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_box->set_placeholder(TTR("Filter signals"));
	search_box->set_right_icon(get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
	search_box->set_clear_button_enabled(true);
	search_box->connect("text_changed", callable_mp(this, &ConnectionsDock::_filter_changed));
	vbc->add_child(search_box);

	tree = memnew(ConnectionsDockTree);
	tree->set_columns(1);
	tree->set_select_mode(Tree::SELECT_ROW);
	tree->set_hide_root(true);
	vbc->add_child(tree);
	tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tree->set_allow_rmb_select(true);

	connect_button = memnew(Button);
	HBoxContainer *hb = memnew(HBoxContainer);
	vbc->add_child(hb);
	hb->add_spacer();
	hb->add_child(connect_button);
	connect_button->connect("pressed", callable_mp(this, &ConnectionsDock::_connect_pressed));

	connect_dialog = memnew(ConnectDialog);
	add_child(connect_dialog);

	disconnect_all_dialog = memnew(ConfirmationDialog);
	add_child(disconnect_all_dialog);
	disconnect_all_dialog->connect("confirmed", callable_mp(this, &ConnectionsDock::_disconnect_all));
	disconnect_all_dialog->set_text(TTR("Are you sure you want to remove all connections from this signal?"));

	signal_menu = memnew(PopupMenu);
	add_child(signal_menu);
	signal_menu->connect("id_pressed", callable_mp(this, &ConnectionsDock::_handle_signal_menu_option));
	signal_menu->add_item(TTR("Connect..."), CONNECT);
	signal_menu->add_item(TTR("Disconnect All"), DISCONNECT_ALL);

	slot_menu = memnew(PopupMenu);
	add_child(slot_menu);
	slot_menu->connect("id_pressed", callable_mp(this, &ConnectionsDock::_handle_slot_menu_option));
	slot_menu->add_item(TTR("Edit..."), EDIT);
	slot_menu->add_item(TTR("Go to Method"), GO_TO_SCRIPT);
	slot_menu->add_item(TTR("Disconnect"), DISCONNECT);

	connect_dialog->connect("connected", callable_mp(this, &ConnectionsDock::_make_or_edit_connection));
	tree->connect("item_selected", callable_mp(this, &ConnectionsDock::_tree_item_selected));
	tree->connect("item_activated", callable_mp(this, &ConnectionsDock::_tree_item_activated));
	tree->connect("item_rmb_selected", callable_mp(this, &ConnectionsDock::_rmb_pressed));

	add_theme_constant_override("separation", 3 * EDSCALE);
}

ConnectionsDock::~ConnectionsDock() {
}
