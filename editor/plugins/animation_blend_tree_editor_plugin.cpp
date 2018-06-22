#include "animation_blend_tree_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "core/project_settings.h"
#include "os/input.h"
#include "os/keyboard.h"
#include "scene/animation/animation_player.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "scene/main/viewport.h"

void AnimationNodeBlendTreeEditor::edit(AnimationNodeBlendTree *p_blend_tree) {

	if (blend_tree.is_valid()) {
		blend_tree->disconnect("removed_from_graph", this, "_removed_from_graph");
	}

	if (p_blend_tree) {
		blend_tree = Ref<AnimationNodeBlendTree>(p_blend_tree);
	} else {
		blend_tree.unref();
	}

	if (blend_tree.is_null()) {
		hide();
	} else {
		blend_tree->connect("removed_from_graph", this, "_removed_from_graph");

		_update_graph();
	}
}

void AnimationNodeBlendTreeEditor::add_custom_type(const String &p_name, const Ref<Script> &p_script) {

	for (int i = 0; i < add_options.size(); i++) {
		ERR_FAIL_COND(add_options[i].script == p_script);
	}

	AddOption ao;
	ao.name = p_name;
	ao.script = p_script;
	add_options.push_back(ao);

	_update_options_menu();
}

void AnimationNodeBlendTreeEditor::remove_custom_type(const Ref<Script> &p_script) {

	for (int i = 0; i < add_options.size(); i++) {
		if (add_options[i].script == p_script) {
			add_options.remove(i);
			return;
		}
	}

	_update_options_menu();
}

void AnimationNodeBlendTreeEditor::_update_options_menu() {

	add_node->get_popup()->clear();
	for (int i = 0; i < add_options.size(); i++) {
		add_node->get_popup()->add_item(add_options[i].name);
	}
}

Size2 AnimationNodeBlendTreeEditor::get_minimum_size() const {

	return Size2(10, 200);
}

void AnimationNodeBlendTreeEditor::_update_graph() {

	if (updating)
		return;

	graph->set_scroll_ofs(blend_tree->get_graph_offset() * EDSCALE);

	if (blend_tree->get_parent().is_valid()) {
		goto_parent->show();
	} else {
		goto_parent->hide();
	}
	graph->clear_connections();
	//erase all nodes
	for (int i = 0; i < graph->get_child_count(); i++) {

		if (Object::cast_to<GraphNode>(graph->get_child(i))) {
			memdelete(graph->get_child(i));
			i--;
		}
	}

	animations.clear();

	List<StringName> nodes;
	blend_tree->get_node_list(&nodes);

	for (List<StringName>::Element *E = nodes.front(); E; E = E->next()) {

		GraphNode *node = memnew(GraphNode);
		graph->add_child(node);

		Ref<AnimationNode> agnode = blend_tree->get_node(E->get());

		if (!agnode->is_connected("changed", this, "_node_changed")) {
			agnode->connect("changed", this, "_node_changed", varray(agnode->get_instance_id()), CONNECT_DEFERRED);
		}

		node->set_offset(agnode->get_position() * EDSCALE);

		node->set_title(agnode->get_caption());
		node->set_name(E->get());

		int base = 0;
		if (String(E->get()) != "output") {
			LineEdit *name = memnew(LineEdit);
			name->set_text(E->get());
			name->set_expand_to_text_length(true);
			node->add_child(name);
			node->set_slot(0, false, 0, Color(), true, 0, get_color("font_color", "Label"));
			name->connect("text_entered", this, "_node_renamed", varray(agnode));
			name->connect("focus_exited", this, "_node_renamed_focus_out", varray(name, agnode));
			base = 1;
			node->set_show_close_button(true);
			node->connect("close_request", this, "_delete_request", varray(E->get()), CONNECT_DEFERRED);
		}

		for (int i = 0; i < agnode->get_input_count(); i++) {
			Label *in_name = memnew(Label);
			node->add_child(in_name);
			in_name->set_text(agnode->get_input_name(i));
			node->set_slot(base + i, true, 0, get_color("font_color", "Label"), false, 0, Color());
		}

		node->connect("dragged", this, "_node_dragged", varray(agnode));

		if (EditorNode::get_singleton()->item_has_editor(agnode.ptr())) {
			node->add_child(memnew(HSeparator));
			Button *open_in_editor = memnew(Button);
			open_in_editor->set_text(TTR("Open Editor"));
			open_in_editor->set_icon(get_icon("Edit", "EditorIcons"));
			node->add_child(open_in_editor);
			open_in_editor->connect("pressed", this, "_open_in_editor", varray(E->get()), CONNECT_DEFERRED);
			open_in_editor->set_h_size_flags(SIZE_SHRINK_CENTER);
		}

		if (agnode->has_filter()) {

			node->add_child(memnew(HSeparator));
			Button *edit_filters = memnew(Button);
			edit_filters->set_text(TTR("Edit Filters"));
			edit_filters->set_icon(get_icon("AnimationFilter", "EditorIcons"));
			node->add_child(edit_filters);
			edit_filters->connect("pressed", this, "_edit_filters", varray(E->get()), CONNECT_DEFERRED);
			edit_filters->set_h_size_flags(SIZE_SHRINK_CENTER);
		}

		Ref<AnimationNodeAnimation> anim = agnode;
		if (anim.is_valid()) {

			MenuButton *mb = memnew(MenuButton);
			mb->set_text(anim->get_animation());
			mb->set_icon(get_icon("Animation", "EditorIcons"));
			Array options;

			node->add_child(memnew(HSeparator));
			node->add_child(mb);

			ProgressBar *pb = memnew(ProgressBar);

			AnimationGraphPlayer *player = anim->get_graph_player();
			if (player->has_node(player->get_animation_player())) {
				AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(player->get_node(player->get_animation_player()));
				if (ap) {
					List<StringName> anims;
					ap->get_animation_list(&anims);

					for (List<StringName>::Element *F = anims.front(); F; F = F->next()) {
						mb->get_popup()->add_item(F->get());
						options.push_back(F->get());
					}

					if (ap->has_animation(anim->get_animation())) {
						pb->set_max(ap->get_animation(anim->get_animation())->get_length());
					}
				}
			}

			pb->set_percent_visible(false);
			animations[E->get()] = pb;
			node->add_child(pb);

			mb->get_popup()->connect("index_pressed", this, "_anim_selected", varray(options, E->get()), CONNECT_DEFERRED);
		}

		Ref<AnimationNodeOneShot> oneshot = agnode;
		if (oneshot.is_valid()) {

			HBoxContainer *play_stop = memnew(HBoxContainer);
			play_stop->add_spacer();
			Button *play = memnew(Button);
			play->set_icon(get_icon("Play", "EditorIcons"));
			play->connect("pressed", this, "_oneshot_start", varray(E->get()), CONNECT_DEFERRED);
			play_stop->add_child(play);
			Button *stop = memnew(Button);
			stop->set_icon(get_icon("Stop", "EditorIcons"));
			stop->connect("pressed", this, "_oneshot_stop", varray(E->get()), CONNECT_DEFERRED);
			play_stop->add_child(stop);
			play_stop->add_spacer();
			node->add_child(play_stop);
		}
	}

	List<AnimationNodeBlendTree::NodeConnection> connections;
	blend_tree->get_node_connections(&connections);

	for (List<AnimationNodeBlendTree::NodeConnection>::Element *E = connections.front(); E; E = E->next()) {

		StringName from = E->get().output_node;
		StringName to = E->get().input_node;
		int to_idx = E->get().input_index;

		graph->connect_node(from, 0, to, to_idx);
	}
}

void AnimationNodeBlendTreeEditor::_add_node(int p_idx) {

	ERR_FAIL_INDEX(p_idx, add_options.size());

	Ref<AnimationNode> anode;

	if (add_options[p_idx].type != String()) {
		AnimationNode *an = Object::cast_to<AnimationNode>(ClassDB::instance(add_options[p_idx].type));
		ERR_FAIL_COND(!an);
		anode = Ref<AnimationNode>(an);
	} else {
		ERR_FAIL_COND(add_options[p_idx].script.is_null());
		String base_type = add_options[p_idx].script->get_instance_base_type();
		AnimationNode *an = Object::cast_to<AnimationNode>(ClassDB::instance(base_type));
		ERR_FAIL_COND(!an);
		anode = Ref<AnimationNode>(an);
		anode->set_script(add_options[p_idx].script.get_ref_ptr());
	}

	Point2 instance_pos = graph->get_scroll_ofs() + graph->get_size() * 0.5;

	anode->set_position(instance_pos);

	String base_name = add_options[p_idx].name;
	int base = 1;
	String name = base_name;
	while (blend_tree->has_node(name)) {
		base++;
		name = base_name + " " + itos(base);
	}

	undo_redo->create_action("Add Node to BlendTree");
	undo_redo->add_do_method(blend_tree.ptr(), "add_node", name, anode);
	undo_redo->add_undo_method(blend_tree.ptr(), "remove_node", name);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
}

void AnimationNodeBlendTreeEditor::_node_dragged(const Vector2 &p_from, const Vector2 &p_to, Ref<AnimationNode> p_node) {

	updating = true;
	undo_redo->create_action("Node Moved");
	undo_redo->add_do_method(p_node.ptr(), "set_position", p_to / EDSCALE);
	undo_redo->add_undo_method(p_node.ptr(), "set_position", p_from / EDSCALE);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
	updating = false;
}

void AnimationNodeBlendTreeEditor::_connection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index) {

	AnimationNodeBlendTree::ConnectionError err = blend_tree->can_connect_node(p_to, p_to_index, p_from);

	if (err != AnimationNodeBlendTree::CONNECTION_OK) {
		EditorNode::get_singleton()->show_warning(TTR("Unable to connect, port may be in use or connection may be invalid."));
		return;
	}

	undo_redo->create_action("Nodes Connected");
	undo_redo->add_do_method(blend_tree.ptr(), "connect_node", p_to, p_to_index, p_from);
	undo_redo->add_undo_method(blend_tree.ptr(), "disconnect_node", p_to, p_to_index, p_from);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
}

void AnimationNodeBlendTreeEditor::_disconnection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index) {

	graph->disconnect_node(p_from, p_from_index, p_to, p_to_index);

	updating = true;
	undo_redo->create_action("Nodes Disconnected");
	undo_redo->add_do_method(blend_tree.ptr(), "disconnect_node", p_to, p_to_index);
	undo_redo->add_undo_method(blend_tree.ptr(), "connect_node", p_to, p_to_index, p_from);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
	updating = false;
}

void AnimationNodeBlendTreeEditor::_anim_selected(int p_index, Array p_options, const String &p_node) {

	String option = p_options[p_index];

	Ref<AnimationNodeAnimation> anim = blend_tree->get_node(p_node);
	ERR_FAIL_COND(!anim.is_valid());

	undo_redo->create_action("Set Animation");
	undo_redo->add_do_method(anim.ptr(), "set_animation", option);
	undo_redo->add_undo_method(anim.ptr(), "set_animation", anim->get_animation());
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
}

void AnimationNodeBlendTreeEditor::_delete_request(const String &p_which) {

	undo_redo->create_action("Delete Node");
	undo_redo->add_do_method(blend_tree.ptr(), "remove_node", p_which);
	undo_redo->add_undo_method(blend_tree.ptr(), "add_node", p_which, blend_tree->get_node(p_which));

	List<AnimationNodeBlendTree::NodeConnection> conns;
	blend_tree->get_node_connections(&conns);

	for (List<AnimationNodeBlendTree::NodeConnection>::Element *E = conns.front(); E; E = E->next()) {
		if (E->get().output_node == p_which || E->get().input_node == p_which) {
			undo_redo->add_undo_method(blend_tree.ptr(), "connect_node", E->get().input_node, E->get().input_index, E->get().output_node);
		}
	}

	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
}

void AnimationNodeBlendTreeEditor::_oneshot_start(const StringName &p_name) {

	Ref<AnimationNodeOneShot> os = blend_tree->get_node(p_name);
	ERR_FAIL_COND(!os.is_valid());
	os->start();
}

void AnimationNodeBlendTreeEditor::_oneshot_stop(const StringName &p_name) {

	Ref<AnimationNodeOneShot> os = blend_tree->get_node(p_name);
	ERR_FAIL_COND(!os.is_valid());
	os->stop();
}

void AnimationNodeBlendTreeEditor::_node_selected(Object *p_node) {

	GraphNode *gn = Object::cast_to<GraphNode>(p_node);
	ERR_FAIL_COND(!gn);

	String name = gn->get_name();

	Ref<AnimationNode> anode = blend_tree->get_node(name);
	ERR_FAIL_COND(!anode.is_valid());

	EditorNode::get_singleton()->push_item(anode.ptr(), "", true);
}

void AnimationNodeBlendTreeEditor::_open_in_editor(const String &p_which) {

	Ref<AnimationNode> an = blend_tree->get_node(p_which);
	ERR_FAIL_COND(!an.is_valid())
	EditorNode::get_singleton()->edit_item(an.ptr());
}

void AnimationNodeBlendTreeEditor::_open_parent() {
	if (blend_tree->get_parent().is_valid()) {
		EditorNode::get_singleton()->edit_item(blend_tree->get_parent().ptr());
	}
}

void AnimationNodeBlendTreeEditor::_filter_toggled() {

	updating = true;
	undo_redo->create_action("Toggle filter on/off");
	undo_redo->add_do_method(_filter_edit.ptr(), "set_filter_enabled", filter_enabled->is_pressed());
	undo_redo->add_undo_method(_filter_edit.ptr(), "set_filter_enabled", _filter_edit->is_filter_enabled());
	undo_redo->add_do_method(this, "_update_filters", _filter_edit);
	undo_redo->add_undo_method(this, "_update_filters", _filter_edit);
	undo_redo->commit_action();
	updating = false;
}

void AnimationNodeBlendTreeEditor::_filter_edited() {

	TreeItem *edited = filters->get_edited();
	ERR_FAIL_COND(!edited);

	NodePath edited_path = edited->get_metadata(0);
	bool filtered = edited->is_checked(0);

	updating = true;
	undo_redo->create_action("Change filter");
	undo_redo->add_do_method(_filter_edit.ptr(), "set_filter_path", edited_path, filtered);
	undo_redo->add_undo_method(_filter_edit.ptr(), "set_filter_path", edited_path, _filter_edit->is_path_filtered(edited_path));
	undo_redo->add_do_method(this, "_update_filters", _filter_edit);
	undo_redo->add_undo_method(this, "_update_filters", _filter_edit);
	undo_redo->commit_action();
	updating = false;
}

bool AnimationNodeBlendTreeEditor::_update_filters(const Ref<AnimationNode> &anode) {

	if (updating || _filter_edit != anode)
		return false;

	NodePath player_path = anode->get_graph_player()->get_animation_player();

	if (!anode->get_graph_player()->has_node(player_path)) {
		EditorNode::get_singleton()->show_warning(TTR("No animation player set, so unable to retrieve track names."));
		return false;
	}

	AnimationPlayer *player = Object::cast_to<AnimationPlayer>(anode->get_graph_player()->get_node(player_path));
	if (!player) {
		EditorNode::get_singleton()->show_warning(TTR("Player path set is invalid, so unable to retrieve track names."));
		return false;
	}

	Node *base = player->get_node(player->get_root());

	if (!base) {
		EditorNode::get_singleton()->show_warning(TTR("Animation player has no valid root node path, so unable to retrieve track names."));
		return false;
	}

	updating = true;

	Set<String> paths;
	{
		List<StringName> animations;
		player->get_animation_list(&animations);

		for (List<StringName>::Element *E = animations.front(); E; E = E->next()) {

			Ref<Animation> anim = player->get_animation(E->get());
			for (int i = 0; i < anim->get_track_count(); i++) {
				paths.insert(anim->track_get_path(i));
			}
		}
	}

	filter_enabled->set_pressed(anode->is_filter_enabled());
	filters->clear();
	TreeItem *root = filters->create_item();

	Map<String, TreeItem *> parenthood;

	for (Set<String>::Element *E = paths.front(); E; E = E->next()) {

		NodePath path = E->get();
		TreeItem *ti = NULL;
		String accum;
		for (int i = 0; i < path.get_name_count(); i++) {
			String name = path.get_name(i);
			if (accum != String()) {
				accum += "/";
			}
			accum += name;
			if (!parenthood.has(accum)) {
				if (ti) {
					ti = filters->create_item(ti);
				} else {
					ti = filters->create_item(root);
				}
				parenthood[accum] = ti;
				ti->set_text(0, name);
				ti->set_selectable(0, false);
				ti->set_editable(0, false);

				if (base->has_node(accum)) {
					Node *node = base->get_node(accum);
					if (has_icon(node->get_class(), "EditorIcons")) {
						ti->set_icon(0, get_icon(node->get_class(), "EditorIcons"));
					} else {
						ti->set_icon(0, get_icon("Node", "EditorIcons"));
					}
				}

			} else {
				ti = parenthood[accum];
			}
		}

		Node *node = NULL;
		if (base->has_node(accum)) {
			node = base->get_node(accum);
		}
		if (!node)
			continue; //no node, cant edit

		if (path.get_subname_count()) {

			String concat = path.get_concatenated_subnames();

			Skeleton *skeleton = Object::cast_to<Skeleton>(node);
			if (skeleton && skeleton->find_bone(concat) != -1) {
				//path in skeleton
				String bone = concat;
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
						ti = filters->create_item(ti);
						parenthood[accum] = ti;
						ti->set_text(0, F->get());
						ti->set_selectable(0, false);
						ti->set_editable(0, false);
						ti->set_icon(0, get_icon("BoneAttachment", "EditorIcons"));
					} else {
						ti = parenthood[accum];
					}
				}

				ti->set_editable(0, true);
				ti->set_selectable(0, true);
				ti->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
				ti->set_text(0, concat);
				ti->set_checked(0, anode->is_path_filtered(path));
				ti->set_icon(0, get_icon("BoneAttachment", "EditorIcons"));
				ti->set_metadata(0, path);

			} else {
				//just a property
				ti = filters->create_item(ti);
				ti->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
				ti->set_text(0, concat);
				ti->set_editable(0, true);
				ti->set_selectable(0, true);
				ti->set_checked(0, anode->is_path_filtered(path));
				ti->set_metadata(0, path);
			}
		} else {
			if (ti) {
				//just a node, likely call or animation track
				ti->set_editable(0, true);
				ti->set_selectable(0, true);
				ti->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
				ti->set_checked(0, anode->is_path_filtered(path));
				ti->set_metadata(0, path);
			}
		}
	}

	updating = false;

	return true;
}

void AnimationNodeBlendTreeEditor::_edit_filters(const String &p_which) {

	Ref<AnimationNode> anode = blend_tree->get_node(p_which);
	ERR_FAIL_COND(!anode.is_valid());

	_filter_edit = anode;
	if (!_update_filters(anode))
		return;

	filter_dialog->popup_centered_minsize(Size2(500, 500) * EDSCALE);
}

void AnimationNodeBlendTreeEditor::_removed_from_graph() {
	if (is_visible()) {
		EditorNode::get_singleton()->edit_item(NULL);
	}
}

void AnimationNodeBlendTreeEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {

		goto_parent->set_icon(get_icon("MoveUp", "EditorIcons"));

		error_panel->add_style_override("panel", get_stylebox("bg", "Tree"));
		error_label->add_color_override("font_color", get_color("error_color", "Editor"));
	}

	if (p_what == NOTIFICATION_PROCESS) {

		String error;

		if (!blend_tree->get_graph_player()) {
			error = TTR("BlendTree does not belong to an AnimationGraphPlayer node.");
		} else if (!blend_tree->get_graph_player()->is_active()) {
			error = TTR("AnimationGraphPlayer is inactive.\nActivate to enable playback, check node warnings if activation fails.");
		} else if (blend_tree->get_graph_player()->is_state_invalid()) {
			error = blend_tree->get_graph_player()->get_invalid_state_reason();
		}

		if (error != error_label->get_text()) {
			error_label->set_text(error);
			if (error != String()) {
				error_panel->show();
			} else {
				error_panel->hide();
			}
		}

		List<AnimationNodeBlendTree::NodeConnection> conns;
		blend_tree->get_node_connections(&conns);
		for (List<AnimationNodeBlendTree::NodeConnection>::Element *E = conns.front(); E; E = E->next()) {
			float activity = 0;
			if (blend_tree->get_graph_player() && !blend_tree->get_graph_player()->is_state_invalid()) {
				activity = blend_tree->get_connection_activity(E->get().input_node, E->get().input_index);
			}
			graph->set_connection_activity(E->get().output_node, 0, E->get().input_node, E->get().input_index, activity);
		}

		AnimationGraphPlayer *graph_player = blend_tree->get_graph_player();
		AnimationPlayer *player = NULL;
		if (graph_player->has_node(graph_player->get_animation_player())) {
			player = Object::cast_to<AnimationPlayer>(graph_player->get_node(graph_player->get_animation_player()));
		}

		if (player) {
			for (Map<StringName, ProgressBar *>::Element *E = animations.front(); E; E = E->next()) {
				Ref<AnimationNodeAnimation> an = blend_tree->get_node(E->key());
				if (an.is_valid()) {
					if (player->has_animation(an->get_animation())) {
						Ref<Animation> anim = player->get_animation(an->get_animation());
						if (anim.is_valid()) {
							E->get()->set_max(anim->get_length());
							E->get()->set_value(an->get_playback_time());
						}
					}
				}
			}
		}
	}
}

void AnimationNodeBlendTreeEditor::_scroll_changed(const Vector2 &p_scroll) {
	if (updating)
		return;
	updating = true;
	blend_tree->set_graph_offset(p_scroll / EDSCALE);
	updating = false;
}

void AnimationNodeBlendTreeEditor::_node_changed(ObjectID p_node) {

	AnimationNode *an = Object::cast_to<AnimationNode>(ObjectDB::get_instance(p_node));
	if (an && an->get_parent() == blend_tree) {
		_update_graph();
	}
}

void AnimationNodeBlendTreeEditor::_bind_methods() {

	ClassDB::bind_method("_update_graph", &AnimationNodeBlendTreeEditor::_update_graph);
	ClassDB::bind_method("_add_node", &AnimationNodeBlendTreeEditor::_add_node);
	ClassDB::bind_method("_node_dragged", &AnimationNodeBlendTreeEditor::_node_dragged);
	ClassDB::bind_method("_node_renamed", &AnimationNodeBlendTreeEditor::_node_renamed);
	ClassDB::bind_method("_node_renamed_focus_out", &AnimationNodeBlendTreeEditor::_node_renamed_focus_out);
	ClassDB::bind_method("_connection_request", &AnimationNodeBlendTreeEditor::_connection_request);
	ClassDB::bind_method("_disconnection_request", &AnimationNodeBlendTreeEditor::_disconnection_request);
	ClassDB::bind_method("_node_selected", &AnimationNodeBlendTreeEditor::_node_selected);
	ClassDB::bind_method("_open_in_editor", &AnimationNodeBlendTreeEditor::_open_in_editor);
	ClassDB::bind_method("_open_parent", &AnimationNodeBlendTreeEditor::_open_parent);
	ClassDB::bind_method("_scroll_changed", &AnimationNodeBlendTreeEditor::_scroll_changed);
	ClassDB::bind_method("_delete_request", &AnimationNodeBlendTreeEditor::_delete_request);
	ClassDB::bind_method("_edit_filters", &AnimationNodeBlendTreeEditor::_edit_filters);
	ClassDB::bind_method("_update_filters", &AnimationNodeBlendTreeEditor::_update_filters);
	ClassDB::bind_method("_filter_edited", &AnimationNodeBlendTreeEditor::_filter_edited);
	ClassDB::bind_method("_filter_toggled", &AnimationNodeBlendTreeEditor::_filter_toggled);
	ClassDB::bind_method("_oneshot_start", &AnimationNodeBlendTreeEditor::_oneshot_start);
	ClassDB::bind_method("_oneshot_stop", &AnimationNodeBlendTreeEditor::_oneshot_stop);
	ClassDB::bind_method("_node_changed", &AnimationNodeBlendTreeEditor::_node_changed);
	ClassDB::bind_method("_removed_from_graph", &AnimationNodeBlendTreeEditor::_removed_from_graph);

	ClassDB::bind_method("_anim_selected", &AnimationNodeBlendTreeEditor::_anim_selected);
}

AnimationNodeBlendTreeEditor *AnimationNodeBlendTreeEditor::singleton = NULL;

void AnimationNodeBlendTreeEditor::_node_renamed(const String &p_text, Ref<AnimationNode> p_node) {

	String prev_name = blend_tree->get_node_name(p_node);
	ERR_FAIL_COND(prev_name == String());
	GraphNode *gn = Object::cast_to<GraphNode>(graph->get_node(prev_name));
	ERR_FAIL_COND(!gn);

	String new_name = p_text;

	ERR_FAIL_COND(new_name == "" || new_name.find(".") != -1 || new_name.find("/") != -1)

	ERR_FAIL_COND(new_name == prev_name);

	String base_name = new_name;
	int base = 1;
	String name = base_name;
	while (blend_tree->has_node(name)) {
		base++;
		name = base_name + " " + itos(base);
	}

	updating = true;
	undo_redo->create_action("Node Renamed");
	undo_redo->add_do_method(blend_tree.ptr(), "rename_node", prev_name, name);
	undo_redo->add_undo_method(blend_tree.ptr(), "rename_node", name, prev_name);
	undo_redo->add_do_method(this, "_update_graph");
	undo_redo->add_undo_method(this, "_update_graph");
	undo_redo->commit_action();
	updating = false;
	gn->set_name(new_name);
	gn->set_size(gn->get_minimum_size());
}

void AnimationNodeBlendTreeEditor::_node_renamed_focus_out(Node *le, Ref<AnimationNode> p_node) {
	_node_renamed(le->call("get_text"), p_node);
}

AnimationNodeBlendTreeEditor::AnimationNodeBlendTreeEditor() {

	singleton = this;
	updating = false;

	graph = memnew(GraphEdit);
	add_child(graph);
	graph->add_valid_right_disconnect_type(0);
	graph->add_valid_left_disconnect_type(0);
	graph->set_v_size_flags(SIZE_EXPAND_FILL);
	graph->connect("connection_request", this, "_connection_request", varray(), CONNECT_DEFERRED);
	graph->connect("disconnection_request", this, "_disconnection_request", varray(), CONNECT_DEFERRED);
	graph->connect("node_selected", this, "_node_selected");
	graph->connect("scroll_offset_changed", this, "_scroll_changed");

	VSeparator *vs = memnew(VSeparator);
	graph->get_zoom_hbox()->add_child(vs);
	graph->get_zoom_hbox()->move_child(vs, 0);

	add_node = memnew(MenuButton);
	graph->get_zoom_hbox()->add_child(add_node);
	add_node->set_text(TTR("Add Node.."));
	graph->get_zoom_hbox()->move_child(add_node, 0);
	add_node->get_popup()->connect("index_pressed", this, "_add_node");

	goto_parent = memnew(Button);
	graph->get_zoom_hbox()->add_child(goto_parent);
	graph->get_zoom_hbox()->move_child(goto_parent, 0);
	goto_parent->hide();
	goto_parent->connect("pressed", this, "_open_parent");

	add_options.push_back(AddOption("Animation", "AnimationNodeAnimation"));
	add_options.push_back(AddOption("OneShot", "AnimationNodeOneShot"));
	add_options.push_back(AddOption("Add", "AnimationNodeAdd"));
	add_options.push_back(AddOption("Blend2", "AnimationNodeBlend2"));
	add_options.push_back(AddOption("Blend3", "AnimationNodeBlend3"));
	add_options.push_back(AddOption("Seek", "AnimationNodeTimeSeek"));
	add_options.push_back(AddOption("TimeScale", "AnimationNodeTimeScale"));
	add_options.push_back(AddOption("Transition", "AnimationNodeTransition"));
	add_options.push_back(AddOption("BlendTree", "AnimationNodeBlendTree"));
	add_options.push_back(AddOption("BlendSpace1D", "AnimationNodeBlendSpace1D"));
	add_options.push_back(AddOption("BlendSpace2D", "AnimationNodeBlendSpace2D"));
	add_options.push_back(AddOption("StateMachine", "AnimationNodeStateMachine"));
	_update_options_menu();

	error_panel = memnew(PanelContainer);
	add_child(error_panel);
	error_label = memnew(Label);
	error_panel->add_child(error_label);
	error_label->set_text("eh");

	filter_dialog = memnew(AcceptDialog);
	add_child(filter_dialog);
	filter_dialog->set_title(TTR("Edit Filtered Tracks:"));

	VBoxContainer *filter_vbox = memnew(VBoxContainer);
	filter_dialog->add_child(filter_vbox);

	filter_enabled = memnew(CheckBox);
	filter_enabled->set_text(TTR("Enable filtering"));
	filter_enabled->connect("pressed", this, "_filter_toggled");
	filter_vbox->add_child(filter_enabled);

	filters = memnew(Tree);
	filter_vbox->add_child(filters);
	filters->set_v_size_flags(SIZE_EXPAND_FILL);
	filters->set_hide_root(true);
	filters->connect("item_edited", this, "_filter_edited");

	undo_redo = EditorNode::get_singleton()->get_undo_redo();
}

void AnimationNodeBlendTreeEditorPlugin::edit(Object *p_object) {

	anim_tree_editor->edit(Object::cast_to<AnimationNodeBlendTree>(p_object));
}

bool AnimationNodeBlendTreeEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("AnimationNodeBlendTree");
}

void AnimationNodeBlendTreeEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		//editor->hide_animation_player_editors();
		//editor->animation_panel_make_visible(true);
		button->show();
		editor->make_bottom_panel_item_visible(anim_tree_editor);
		anim_tree_editor->set_process(true);
	} else {

		if (anim_tree_editor->is_visible_in_tree())
			editor->hide_bottom_panel();
		button->hide();
		anim_tree_editor->set_process(false);
	}
}

AnimationNodeBlendTreeEditorPlugin::AnimationNodeBlendTreeEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	anim_tree_editor = memnew(AnimationNodeBlendTreeEditor);
	anim_tree_editor->set_custom_minimum_size(Size2(0, 300));

	button = editor->add_bottom_panel_item(TTR("BlendTree"), anim_tree_editor);
	button->hide();
}

AnimationNodeBlendTreeEditorPlugin::~AnimationNodeBlendTreeEditorPlugin() {
}
