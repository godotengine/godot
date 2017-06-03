/*************************************************************************/
/*  editor_sub_scene.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "editor_sub_scene.h"

#include "scene/gui/margin_container.h"
#include "scene/resources/packed_scene.h"

void EditorSubScene::_path_selected(const String &p_path) {

	path->set_text(p_path);
	_path_changed(p_path);
}

void EditorSubScene::_path_changed(const String &p_path) {

	tree->clear();

	if (scene) {
		memdelete(scene);
		scene = NULL;
	}

	if (p_path == "")
		return;

	Ref<PackedScene> ps = ResourceLoader::load(p_path, "PackedScene");

	if (ps.is_null())
		return;

	scene = ps->instance();
	if (!scene)
		return;

	_fill_tree(scene, NULL);
}

void EditorSubScene::_path_browse() {

	file_dialog->popup_centered_ratio();
}

void EditorSubScene::_notification(int p_what) {

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {

		if (!is_visible_in_tree()) {
		}
	}
}

void EditorSubScene::_fill_tree(Node *p_node, TreeItem *p_parent) {

	TreeItem *it = tree->create_item(p_parent);
	it->set_metadata(0, p_node);
	it->set_text(0, p_node->get_name());
	it->set_editable(0, false);
	it->set_selectable(0, true);
	if (has_icon(p_node->get_class(), "EditorIcons")) {
		it->set_icon(0, get_icon(p_node->get_class(), "EditorIcons"));
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {

		Node *c = p_node->get_child(i);
		if (c->get_owner() != scene)
			continue;
		_fill_tree(c, it);
	}
}

void EditorSubScene::ok_pressed() {

	TreeItem *s = tree->get_selected();
	if (!s)
		return;
	Node *selnode = s->get_metadata(0);
	if (!selnode)
		return;
	emit_signal("subscene_selected");
	hide();
	clear();
}

void EditorSubScene::_reown(Node *p_node, List<Node *> *p_to_reown) {

	if (p_node == scene) {

		scene->set_filename("");
		p_to_reown->push_back(p_node);
	} else if (p_node->get_owner() == scene) {

		p_to_reown->push_back(p_node);
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *c = p_node->get_child(i);
		_reown(c, p_to_reown);
	}
}

void EditorSubScene::move(Node *p_new_parent, Node *p_new_owner) {

	if (!scene) {
		return;
	}
	TreeItem *s = tree->get_selected();
	if (!s) {
		return;
	}

	Node *selnode = s->get_metadata(0);
	if (!selnode) {
		return;
	}

	List<Node *> to_reown;
	_reown(selnode, &to_reown);

	if (selnode != scene) {
		selnode->get_parent()->remove_child(selnode);
	}

	p_new_parent->add_child(selnode);
	for (List<Node *>::Element *E = to_reown.front(); E; E = E->next()) {
		E->get()->set_owner(p_new_owner);
	}

	if (selnode != scene) {
		memdelete(scene);
	}
	scene = NULL;

	//return selnode;
}

void EditorSubScene::clear() {

	path->set_text("");
	_path_changed("");
}

void EditorSubScene::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_path_selected"), &EditorSubScene::_path_selected);
	ClassDB::bind_method(D_METHOD("_path_changed"), &EditorSubScene::_path_changed);
	ClassDB::bind_method(D_METHOD("_path_browse"), &EditorSubScene::_path_browse);
	ADD_SIGNAL(MethodInfo("subscene_selected"));
}

EditorSubScene::EditorSubScene() {

	scene = NULL;

	set_title(TTR("Select Node(s) to Import"));
	set_hide_on_ok(false);

	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);
	//set_child_rect(vb);

	HBoxContainer *hb = memnew(HBoxContainer);
	path = memnew(LineEdit);
	path->connect("text_entered", this, "_path_changed");
	hb->add_child(path);
	path->set_h_size_flags(SIZE_EXPAND_FILL);
	Button *b = memnew(Button);
	b->set_text(" .. ");
	hb->add_child(b);
	b->connect("pressed", this, "_path_browse");
	vb->add_margin_child(TTR("Scene Path:"), hb);

	tree = memnew(Tree);
	tree->set_v_size_flags(SIZE_EXPAND_FILL);
	vb->add_margin_child(TTR("Import From Node:"), tree, true);
	tree->connect("item_activated", this, "_ok", make_binds(), CONNECT_DEFERRED);

	file_dialog = memnew(EditorFileDialog);
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("PackedScene", &extensions);

	for (List<String>::Element *E = extensions.front(); E; E = E->next()) {

		file_dialog->add_filter("*." + E->get());
	}

	file_dialog->set_mode(EditorFileDialog::MODE_OPEN_FILE);
	add_child(file_dialog);
	file_dialog->connect("file_selected", this, "_path_selected");
}
