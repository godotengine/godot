/*************************************************************************/
/*  connections_dialog.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor_node.h"
#include "editor_settings.h"
#include "plugins/script_editor_plugin.h"
#include "print_string.h"
#include "scene/gui/label.h"
#include "scene/gui/popup_menu.h"

class ConnectDialogBinds : public Object {

	GDCLASS(ConnectDialogBinds, Object);

public:
	Vector<Variant> params;

	bool _set(const StringName &p_name, const Variant &p_value) {

		String name = p_name;

		if (name.begins_with("bind/")) {
			int which = name.get_slice("/", 1).to_int() - 1;
			ERR_FAIL_INDEX_V(which, params.size(), false);
			params[which] = p_value;
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
		} else
			return false;

		return true;
	}

	void _get_property_list(List<PropertyInfo> *p_list) const {

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

void ConnectDialog::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {
		bind_editor->edit(cdbinds);
	}
}

void ConnectDialog::_tree_node_selected() {

	//dst_method_list->get_popup()->clear();
	Node *current = tree->get_selected();

	if (!current) {
		make_callback->hide();
		return;
	}

	if (current->get_script().is_null())
		make_callback->hide();
	else
		make_callback->show();

	dst_path->set_text(source->get_path_to(current));
}

void ConnectDialog::init(Connection c, bool bEdit) {
	source = static_cast<Node*>(c.source);
	signal = c.signal;

	tree->set_selected(NULL);
	tree->set_marked(source, true);

	set_dst_node( static_cast<Node*>(c.target) );
	set_dst_method( c.method );

	bool bDeferred = (c.flags & CONNECT_DEFERRED) == CONNECT_DEFERRED;
	bool bOneshot = (c.flags & CONNECT_ONESHOT) == CONNECT_ONESHOT;

	deferred->set_pressed(bDeferred);
	oneshot->set_pressed(bOneshot);

	cdbinds->params.clear();
	cdbinds->params = c.binds;
	cdbinds->notify_changed();

	bEditMode = bEdit;
}

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

NodePath ConnectDialog::get_dst_path() const {

	return dst_path->get_text();
}

bool ConnectDialog::get_deferred() const {

	return deferred->is_pressed();
}

bool ConnectDialog::get_oneshot() const {

	return oneshot->is_pressed();
}

bool ConnectDialog::is_editing() const
{
	return bEditMode;
}

StringName ConnectDialog::get_dst_method() const {

	String txt = dst_method->get_text();
	if (txt.find("(") != -1)
		txt = txt.left(txt.find("(")).strip_edges();
	return txt;
}

Node * ConnectDialog::get_source() const
{
	return source;
}

StringName ConnectDialog::get_signal() const
{
	return signal;
}

Vector<Variant> ConnectDialog::get_binds() const {

	return cdbinds->params;
}

void ConnectDialog::_add_bind() {

	if (cdbinds->params.size() >= VARIANT_ARG_MAX)
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
		case Variant::PLANE: value = Plane(); break;
		case Variant::QUAT: value = Quat(); break;
		case Variant::AABB: value = AABB(); break;
		case Variant::BASIS: value = Basis(); break;
		case Variant::TRANSFORM: value = Transform(); break;
		case Variant::COLOR: value = Color(); break;

		default: { ERR_FAIL(); } break;
	}

	ERR_FAIL_COND(value.get_type() == Variant::NIL);

	cdbinds->params.push_back(value);
	cdbinds->notify_changed();
}

void ConnectDialog::_remove_bind() {

	String st = bind_editor->get_selected_path();
	if (st == "")
		return;
	int idx = st.get_slice("/", 1).to_int() - 1;

	ERR_FAIL_INDEX(idx, cdbinds->params.size());
	cdbinds->params.remove(idx);
	cdbinds->notify_changed();
}

void ConnectDialog::set_dst_node(Node *p_node) {

	tree->set_selected(p_node);
}

void ConnectDialog::set_dst_method(const StringName &p_method) {

	dst_method->set_text(p_method);
}

void ConnectDialog::_bind_methods() {

	ClassDB::bind_method("_cancel", &ConnectDialog::_cancel_pressed);
	ClassDB::bind_method("_tree_node_selected", &ConnectDialog::_tree_node_selected);

	ClassDB::bind_method("_add_bind", &ConnectDialog::_add_bind);
	ClassDB::bind_method("_remove_bind", &ConnectDialog::_remove_bind);

	ADD_SIGNAL(MethodInfo("connected"));
}

ConnectDialog::ConnectDialog() {

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	HBoxContainer *main_hb = memnew(HBoxContainer);
	vbc->add_child(main_hb);
	main_hb->set_v_size_flags(SIZE_EXPAND_FILL);

	VBoxContainer *vbc_left = memnew(VBoxContainer);
	main_hb->add_child(vbc_left);
	vbc_left->set_h_size_flags(SIZE_EXPAND_FILL);

	tree = memnew(SceneTreeEditor(false));
	tree->get_scene_tree()->connect("item_activated", this, "_ok");
	vbc_left->add_margin_child(TTR("Connect To Node:"), tree, true);

	VBoxContainer *vbc_right = memnew(VBoxContainer);
	main_hb->add_child(vbc_right);
	vbc_right->set_h_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *add_bind_hb = memnew(HBoxContainer);

	type_list = memnew(OptionButton);
	type_list->set_h_size_flags(SIZE_EXPAND_FILL);
	add_bind_hb->add_child(type_list);

	type_list->add_item("bool", Variant::BOOL);
	type_list->add_item("int", Variant::INT);
	type_list->add_item("real", Variant::REAL);
	type_list->add_item("string", Variant::STRING);
	//type_list->add_separator();
	type_list->add_item("Vector2", Variant::VECTOR2);
	type_list->add_item("Rect2", Variant::RECT2);
	type_list->add_item("Vector3", Variant::VECTOR3);
	type_list->add_item("Plane", Variant::PLANE);
	type_list->add_item("Quat", Variant::QUAT);
	type_list->add_item("AABB", Variant::AABB);
	type_list->add_item("Basis", Variant::BASIS);
	type_list->add_item("Transform", Variant::TRANSFORM);
	//type_list->add_separator();
	type_list->add_item("Color", Variant::COLOR);
	type_list->select(0);

	Button *add_bind = memnew(Button);

	add_bind->set_text(TTR("Add"));
	add_bind_hb->add_child(add_bind);
	add_bind->connect("pressed", this, "_add_bind");

	Button *del_bind = memnew(Button);
	del_bind->set_text(TTR("Remove"));
	add_bind_hb->add_child(del_bind);
	del_bind->connect("pressed", this, "_remove_bind");

	vbc_right->add_margin_child(TTR("Add Extra Call Argument:"), add_bind_hb);

	bind_editor = memnew(PropertyEditor);
	bind_editor->hide_top_label();

	vbc_right->add_margin_child(TTR("Extra Call Arguments:"), bind_editor, true);

	dst_path = memnew(LineEdit);
	vbc->add_margin_child(TTR("Path to Node:"), dst_path);

	HBoxContainer *dstm_hb = memnew(HBoxContainer);
	vbc->add_margin_child("Method In Node:", dstm_hb);

	dst_method = memnew(LineEdit);
	dst_method->set_h_size_flags(SIZE_EXPAND_FILL);
	dstm_hb->add_child(dst_method);

	/*dst_method_list = memnew( MenuButton );
	dst_method_list->set_text("List...");
	dst_method_list->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	dst_method_list->set_anchor( MARGIN_LEFT, ANCHOR_END );
	dst_method_list->set_anchor( MARGIN_TOP, ANCHOR_END );
	dst_method_list->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
	dst_method_list->set_begin( Point2( 70,59) );
	dst_method_list->set_end( Point2( 15,39  ) );
	*/
	//add_child(dst_method_list);

	make_callback = memnew(CheckButton);
	make_callback->set_toggle_mode(true);
	make_callback->set_pressed(EDITOR_DEF("text_editor/tools/create_signal_callbacks", true));
	make_callback->set_text(TTR("Make Function"));
	dstm_hb->add_child(make_callback);

	deferred = memnew(CheckButton);
	deferred->set_text(TTR("Deferred"));
	dstm_hb->add_child(deferred);

	oneshot = memnew(CheckButton);
	oneshot->set_text(TTR("Oneshot"));
	dstm_hb->add_child(oneshot);

	tree->connect("node_selected", this, "_tree_node_selected");

	set_as_toplevel(true);

	cdbinds = memnew(ConnectDialogBinds);

	error = memnew(ConfirmationDialog);
	add_child(error);
	error->get_ok()->set_text(TTR("Close"));
	get_ok()->set_text(TTR("Connect"));
}

ConnectDialog::~ConnectDialog() {
	memdelete(cdbinds);
}

void ConnectionsDock::_notification(int p_what) {

	if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		update_tree();
	}
}

void ConnectionsDock::_close() {

	hide();
}

void ConnectionsDock::_make_or_edit_connection() {

	TreeItem *it = tree->get_selected();
	ERR_FAIL_COND(!it);

	NodePath dst_path = connect_dialog->get_dst_path();
	Node *target = node->get_node(dst_path);
	ERR_FAIL_COND(!target);

	Connection cToMake;
	cToMake.source = connect_dialog->get_source();
	cToMake.target = target;
	cToMake.signal = connect_dialog->get_signal(); //it->get_metadata(0).operator Dictionary()["name"]
	cToMake.method = connect_dialog->get_dst_method();
	cToMake.binds = connect_dialog->get_binds();
	bool defer = connect_dialog->get_deferred();
	bool oshot = connect_dialog->get_oneshot();
	cToMake.flags = CONNECT_PERSIST | (defer ? CONNECT_DEFERRED : 0) | (oshot ? CONNECT_ONESHOT : 0);
	
	if (connect_dialog->is_editing()) {
		_disconnect(it);
		_connect(cToMake);
	}
	else {
		_connect(cToMake);
	}

	if (connect_dialog->get_make_callback()) {
		PoolStringArray args = it->get_metadata(0).operator Dictionary()["args"];
		print_line("request connect");
		editor->emit_signal("script_add_function_request", target, cToMake.method, args);
		hide();
	}

	update_tree();
}

void ConnectionsDock::_connect(Connection cToMake)
{
	Node* source = static_cast<Node*>(cToMake.source);
	Node* target = static_cast<Node*>(cToMake.target);

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

void ConnectionsDock::_disconnect( TreeItem *item )
{
	Connection c = item->get_metadata(0);
	ERR_FAIL_COND(c.source != node); //shouldn't happen but...bugcheck

	undo_redo->create_action(vformat(TTR("Disconnect '%s' from '%s'"), c.signal, c.method));
	undo_redo->add_do_method(node, "disconnect", c.signal, c.target, c.method);
	undo_redo->add_undo_method(node, "connect", c.signal, c.target, c.method, Vector<Variant>(), c.flags); //Empty binds?
	undo_redo->add_do_method(this, "update_tree");
	undo_redo->add_undo_method(this, "update_tree");
	undo_redo->add_do_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree"); //to force redraw of scene tree
	undo_redo->add_undo_method(EditorNode::get_singleton()->get_scene_tree_dock()->get_tree_editor(), "update_tree");
	undo_redo->commit_action();

	update_tree();
}

void ConnectionsDock::_open_connection_dialog( TreeItem *item ) {

	String signal = item->get_metadata(0).operator Dictionary()["name"];
	String signalname = signal;
	String midname = node->get_name();
	for (int i = 0; i < midname.length(); i++) { //TODO: Regex filter is cleaner.
		CharType c = midname[i];
		if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
			//all good
		} else if (c == ' ') {
			c = '_';
		} else {
			midname.remove(i);
			i--;
			continue;
		}
		midname[i] = c;
	}

	Node *dst_node = node->get_owner() ? node->get_owner() : node;
	StringName dst_method = "_on_" + midname + "_" + signal;

	Connection c;
	c.source = node;
	c.signal = StringName(signalname);
	c.target = dst_node;
	c.method = dst_method;

	connect_dialog->init(c);
	connect_dialog->set_title(TTR("Connect Signal: ") + signalname);
	connect_dialog->popup_centered_ratio();
}

void ConnectionsDock::_open_connection_dialog( Connection cToEdit ) {

	Node *src = static_cast<Node*>(cToEdit.source);
	Node *dst = static_cast<Node*>(cToEdit.target);

	if(src && dst) {
		connect_dialog->init(cToEdit, true);
		connect_dialog->set_title(TTR("Edit Connection: ") + cToEdit.signal);
		connect_dialog->popup_centered_ratio();
	}
}

void ConnectionsDock::_connect_pressed() {
	TreeItem *item = tree->get_selected();
	if (!item) {
		connect_button->set_disabled(true);
		return;
	}

	if (_is_item_signal( item )) {
		_open_connection_dialog( item );
	} else {
		_disconnect( item );
	}
}

struct _ConnectionsDockMethodInfoSort {

	_FORCE_INLINE_ bool operator()(const MethodInfo &a, const MethodInfo &b) const {
		return a.name < b.name;
	}
};

void ConnectionsDock::update_tree() {

	tree->clear();

	if (!node)
		return;

	TreeItem *root = tree->create_item();

	List<MethodInfo> node_signals;

	node->get_signal_list(&node_signals);

	//node_signals.sort_custom<_ConnectionsDockMethodInfoSort>();
	bool did_script = false;
	StringName base = node->get_class();

	while (base) {

		List<MethodInfo> node_signals;
		Ref<Texture> icon;
		String name;

		if (!did_script) {

			Ref<Script> scr = node->get_script();
			if (scr.is_valid()) {
				scr->get_script_signal_list(&node_signals);
				if (scr->get_path().is_resource_file())
					name = scr->get_path().get_file();
				else
					name = scr->get_class();

				if (has_icon(scr->get_class(), "EditorIcons")) {
					icon = get_icon(scr->get_class(), "EditorIcons");
				}
			}

		} else {

			ClassDB::get_signal_list(base, &node_signals, true);
			if (has_icon(base, "EditorIcons")) {
				icon = get_icon(base, "EditorIcons");
			}
			name = base;
		}

		TreeItem *pitem = NULL;

		if (node_signals.size()) {
			pitem = tree->create_item(root);
			pitem->set_text(0, name);
			pitem->set_icon(0, icon);
			pitem->set_selectable(0, false);
			pitem->set_editable(0, false);
			pitem->set_custom_bg_color(0, get_color("prop_subsection", "Editor"));
			node_signals.sort();
		}

		for (List<MethodInfo>::Element *E = node_signals.front(); E; E = E->next()) {

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
					if (pi.type != Variant::NIL) {
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
			node->get_signal_connection_list(mi.name, &connections);

			for (List<Object::Connection>::Element *F = connections.front(); F; F = F->next()) {

				Object::Connection &c = F->get();
				if (!(c.flags & CONNECT_PERSIST))
					continue;

				Node *target = Object::cast_to<Node>(c.target);
				if (!target)
					continue;

				String path = String(node->get_path_to(target)) + " :: " + c.method + "()";
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

void ConnectionsDock::set_node(Node *p_node) {
	node = p_node;
	update_tree();
}

void ConnectionsDock::_something_selected() {

	TreeItem *item = tree->get_selected();
	if (!item) {
		//no idea how this happened, but disable
		connect_button->set_text(TTR("Connect..."));
		connect_button->set_disabled(true);

	} else if (_is_item_signal( item )) {
		connect_button->set_text(TTR("Connect..."));
		connect_button->set_disabled(false);
	} else {
		connect_button->set_text(TTR("Disconnect"));
		connect_button->set_disabled(false);
	}
}

void ConnectionsDock::_something_activated() {

	TreeItem *item = tree->get_selected();

	if (!item)
		return;

	if (_is_item_signal( item )) {
		_open_connection_dialog( item );
	} else {
		//Go to method within script editor, if avaiable.
		Connection c = item->get_metadata(0);
		ERR_FAIL_COND(c.source != node); //shouldn't happen but...bugcheck

		if (!c.target)
			return;

		Ref<Script> script = c.target->get_script();

		if (script.is_valid() && ScriptEditor::get_singleton()->script_goto_method(script, c.method)) {
			editor->call("_editor_select", EditorNode::EDITOR_SCRIPT);
		}
	}
}

bool ConnectionsDock::_is_item_signal( TreeItem *item ) {
	return (item->get_parent() == tree->get_root() || item->get_parent()->get_parent() == tree->get_root());
}

void ConnectionsDock::_handle_signal_option( int option ) {
	TreeItem *item = tree->get_selected();

	switch( option )
	{
			case SignalMenuOption::CONNECT:
			{
				_open_connection_dialog(item);
			} break;
	}
}

void ConnectionsDock::_handle_slot_option( int option ) {
	TreeItem *item = tree->get_selected();

	switch( option )
	{
		case SlotMenuOption::EDIT:
		{
			Connection c = item->get_metadata(0);
			_open_connection_dialog(c);
		} break;
		case SlotMenuOption::DISCONNECT:
		{
			_disconnect(item);
		} break;
	}
}

void ConnectionsDock::_rmb_pressed( Vector2 position ) {
	TreeItem *item = tree->get_selected();

	if( !item )
		return;

	Vector2 global_position = tree->get_global_position() + position;

	if( _is_item_signal(item) )
	{
		signal_menu->set_position( global_position );
		signal_menu->popup();
	}
	else
	{
		slot_menu->set_position( global_position );
		slot_menu->popup();
	}
}

void ConnectionsDock::_bind_methods() {
	ClassDB::bind_method("_connect", &ConnectionsDock::_make_or_edit_connection);
	ClassDB::bind_method("_something_selected", &ConnectionsDock::_something_selected);
	ClassDB::bind_method("_something_activated", &ConnectionsDock::_something_activated);
	ClassDB::bind_method("_handle_signal_option", &ConnectionsDock::_handle_signal_option);
	ClassDB::bind_method("_handle_slot_option", &ConnectionsDock::_handle_slot_option);
	ClassDB::bind_method("_rmb_pressed", &ConnectionsDock::_rmb_pressed);
	ClassDB::bind_method("_close", &ConnectionsDock::_close);
	ClassDB::bind_method("_connect_pressed", &ConnectionsDock::_connect_pressed);
	ClassDB::bind_method("update_tree", &ConnectionsDock::update_tree);
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
	//add_child(tree);

	connect_dialog = memnew(ConnectDialog);
	connect_dialog->set_as_toplevel(true);
	add_child(connect_dialog);

	remove_confirm = memnew(ConfirmationDialog);
	remove_confirm->set_as_toplevel(true);
	add_child(remove_confirm);

	signal_menu = memnew(PopupMenu);
	add_child(signal_menu);
	signal_menu->connect( "id_pressed", this, "_handle_signal_option" );
	signal_menu->add_item(TTR("Connect"), SignalMenuOption::CONNECT );

	slot_menu = memnew(PopupMenu);
	add_child(slot_menu);
	slot_menu->connect( "id_pressed", this, "_handle_slot_option" );
	slot_menu->add_item(TTR("Edit"), SlotMenuOption::EDIT);
	slot_menu->add_item(TTR("Disconnect"), SlotMenuOption::DISCONNECT);

	/*
	node_only->set_anchor( MARGIN_TOP, ANCHOR_END );
	node_only->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
	node_only->set_anchor( MARGIN_RIGHT, ANCHOR_END );

	node_only->set_begin( Point2( 20,51) );
	node_only->set_end( Point2( 10,44) );
	*/

	remove_confirm->connect("confirmed", this, "_remove_confirm");
	connect_dialog->connect("connected", this, "_connect");
	tree->connect("item_selected", this, "_something_selected");
	tree->connect("item_activated", this, "_something_activated");
	tree->connect("item_rmb_selected", this, "_rmb_pressed" );

	add_constant_override("separation", 3 * EDSCALE);
}

ConnectionsDock::~ConnectionsDock() {
}
