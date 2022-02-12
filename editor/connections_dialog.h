/*************************************************************************/
/*  connections_dialog.h                                                 */
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

#ifndef CONNECTIONS_DIALOG_H
#define CONNECTIONS_DIALOG_H

#include "core/object/undo_redo.h"
#include "editor/editor_inspector.h"
#include "editor/scene_tree_editor.h"
#include "scene/gui/button.h"
#include "scene/gui/check_button.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/popup.h"
#include "scene/gui/tree.h"

class PopupMenu;
class ConnectDialogBinds;
class SpinBox;

class ConnectDialog : public ConfirmationDialog {
	GDCLASS(ConnectDialog, ConfirmationDialog);

public:
	struct ConnectionData {
		Node *source = nullptr;
		Node *target = nullptr;
		StringName signal;
		StringName method;
		uint32_t flags = 0;
		int unbinds = 0;
		Vector<Variant> binds;

		ConnectionData() {}

		ConnectionData(const Connection &p_connection) {
			source = Object::cast_to<Node>(p_connection.signal.get_object());
			signal = p_connection.signal.get_name();
			target = Object::cast_to<Node>(p_connection.callable.get_object());
			flags = p_connection.flags;

			Callable base_callable;
			if (p_connection.callable.is_custom()) {
				CallableCustomBind *ccb = dynamic_cast<CallableCustomBind *>(p_connection.callable.get_custom());
				if (ccb) {
					binds = ccb->get_binds();
					base_callable = ccb->get_callable();
				}

				CallableCustomUnbind *ccu = dynamic_cast<CallableCustomUnbind *>(p_connection.callable.get_custom());
				if (ccu) {
					unbinds = ccu->get_unbinds();
					base_callable = ccu->get_callable();
				}
			} else {
				base_callable = p_connection.callable;
			}
			method = base_callable.get_method();
		}

		Callable get_callable() {
			if (unbinds > 0) {
				return Callable(target, method).unbind(unbinds);
			} else if (!binds.is_empty()) {
				const Variant *args = binds.ptr();
				return Callable(target, method).bind(&args, binds.size());
			} else {
				return Callable(target, method);
			}
		}
	};

private:
	Label *connect_to_label;
	LineEdit *from_signal;
	Node *source;
	StringName signal;
	LineEdit *dst_method;
	ConnectDialogBinds *cdbinds;
	bool edit_mode;
	NodePath dst_path;
	VBoxContainer *vbc_right;

	SceneTreeEditor *tree;
	AcceptDialog *error;
	SpinBox *unbind_count;
	EditorInspector *bind_editor;
	OptionButton *type_list;
	CheckBox *deferred;
	CheckBox *oneshot;
	CheckButton *advanced;
	Vector<Control *> bind_controls;

	Label *error_label;

	void ok_pressed() override;
	void _cancel_pressed();
	void _item_activated();
	void _text_submitted(const String &p_text);
	void _tree_node_selected();
	void _unbind_count_changed(double p_count);
	void _add_bind();
	void _remove_bind();
	void _advanced_pressed();
	void _update_ok_enabled();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	Node *get_source() const;
	StringName get_signal_name() const;
	NodePath get_dst_path() const;
	void set_dst_node(Node *p_node);
	StringName get_dst_method_name() const;
	void set_dst_method(const StringName &p_method);
	int get_unbinds() const;
	Vector<Variant> get_binds() const;

	bool get_deferred() const;
	bool get_oneshot() const;
	bool is_editing() const;

	void init(ConnectionData p_cd, bool p_edit = false);

	void popup_dialog(const String &p_for_signal);
	ConnectDialog();
	~ConnectDialog();
};

//////////////////////////////////////////

// Custom Tree needed to use a RichTextLabel as tooltip control
// when display signal documentation.
class ConnectionsDockTree : public Tree {
	virtual Control *make_custom_tooltip(const String &p_text) const;
};

class ConnectionsDock : public VBoxContainer {
	GDCLASS(ConnectionsDock, VBoxContainer);

	//Right-click Pop-up Menu Options.
	enum SignalMenuOption {
		CONNECT,
		DISCONNECT_ALL
	};

	enum SlotMenuOption {
		EDIT,
		GO_TO_SCRIPT,
		DISCONNECT
	};

	Node *selected_node;
	ConnectionsDockTree *tree;
	EditorNode *editor;

	ConfirmationDialog *disconnect_all_dialog;
	ConnectDialog *connect_dialog;
	Button *connect_button;
	PopupMenu *signal_menu;
	PopupMenu *slot_menu;
	UndoRedo *undo_redo;
	LineEdit *search_box;

	Map<StringName, Map<StringName, String>> descr_cache;

	void _filter_changed(const String &p_text);

	void _make_or_edit_connection();
	void _connect(ConnectDialog::ConnectionData p_cd);
	void _disconnect(TreeItem &p_item);
	void _disconnect_all();

	void _tree_item_selected();
	void _tree_item_activated();
	bool _is_item_signal(TreeItem &p_item);

	void _open_connection_dialog(TreeItem &p_item);
	void _open_connection_dialog(ConnectDialog::ConnectionData p_cd);
	void _go_to_script(TreeItem &p_item);

	void _handle_signal_menu_option(int p_option);
	void _handle_slot_menu_option(int p_option);
	void _rmb_pressed(Vector2 p_position);
	void _close();

protected:
	void _connect_pressed();
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_undoredo(UndoRedo *p_undo_redo) { undo_redo = p_undo_redo; }
	void set_node(Node *p_node);
	void update_tree();

	ConnectionsDock(EditorNode *p_editor = nullptr);
	~ConnectionsDock();
};

#endif // CONNECTIONS_DIALOG_H
