/*************************************************************************/
/*  scene_tree_dock.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "scene_tree_dock.h"
#include "editor_node.h"
#include "globals.h"
#include "os/keyboard.h"
#include "scene/resources/packed_scene.h"
#include "editor_settings.h"
#include "tools/editor/plugins/canvas_item_editor_plugin.h"


void SceneTreeDock::_unhandled_key_input(InputEvent p_event) {


	uint32_t sc = p_event.key.get_scancode_with_modifiers();
	if (!p_event.key.pressed || p_event.key.echo)
		return;

	switch(sc) {
		case KEY_MASK_CMD|KEY_A: { _tool_selected(TOOL_NEW); } break;
		case KEY_MASK_CMD|KEY_D: { _tool_selected(TOOL_DUPLICATE); } break;
		case KEY_MASK_CMD|KEY_UP: { _tool_selected(TOOL_MOVE_UP); } break;
		case KEY_MASK_CMD|KEY_DOWN: { _tool_selected(TOOL_MOVE_DOWN); } break;
		case KEY_MASK_SHIFT|KEY_DELETE: { _tool_selected(TOOL_ERASE, true); } break;
		case KEY_DELETE: { _tool_selected(TOOL_ERASE); } break;
	}
}

Node* SceneTreeDock::instance(const String& p_file) {

	Node *parent = scene_tree->get_selected();
	if (!parent || !edited_scene) {

		current_option=-1;
		//accept->get_cancel()->hide();
		accept->get_ok()->set_text("Ok :( ");
		accept->set_text("No parent to instance a child at.");
		accept->popup_centered(Size2(300,70));
		return NULL;
	};

	ERR_FAIL_COND_V(!parent,NULL);

	Node*instanced_scene=NULL;
	Ref<PackedScene> sdata = ResourceLoader::load(p_file);
	if (sdata.is_valid())
		instanced_scene=sdata->instance();


	if (!instanced_scene) {

		current_option=-1;
		//accept->get_cancel()->hide();
		accept->get_ok()->set_text("Ugh");
		accept->set_text(String("Error loading scene from ")+p_file);
		accept->popup_centered(Size2(300,70));;
		return NULL;
	}

	instanced_scene->generate_instance_state();
	instanced_scene->set_filename( Globals::get_singleton()->localize_path(p_file) );

	editor_data->get_undo_redo().create_action("Instance Scene");
	editor_data->get_undo_redo().add_do_method(parent,"add_child",instanced_scene);
	editor_data->get_undo_redo().add_do_method(instanced_scene,"set_owner",edited_scene);
	editor_data->get_undo_redo().add_do_method(editor_selection,"clear");
	editor_data->get_undo_redo().add_do_method(editor_selection,"add_node",instanced_scene);
	editor_data->get_undo_redo().add_do_reference(instanced_scene);
	editor_data->get_undo_redo().add_undo_method(parent,"remove_child",instanced_scene);
	editor_data->get_undo_redo().commit_action();


	return instanced_scene;

}

static String _get_name_num_separator() {
	switch(EditorSettings::get_singleton()->get("scenetree_editor/duplicate_node_name_num_separator").operator int()) {
		case 0: return "";
		case 1: return " ";
		case 2: return "_";
		case 3: return "-";
	}
	return " ";
}

void SceneTreeDock::_tool_selected(int p_tool, bool p_confirm_override) {

	current_option=p_tool;

	switch(p_tool) {

		case TOOL_NEW: {


			if (!_validate_no_foreign())
				break;
			create_dialog->popup_centered_ratio();
		} break;
		case TOOL_INSTANCE: {

			Node *scene = edited_scene;

			if (!scene) {

				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done without a tree root.");
				accept->popup_centered(Size2(300,70));;
				break;
			}

			if (!_validate_no_foreign())
				break;

			file->set_mode(FileDialog::MODE_OPEN_FILE);
			List<String> extensions;
			ResourceLoader::get_recognized_extensions_for_type("PackedScene",&extensions);
			file->clear_filters();
			for(int i=0;i<extensions.size();i++) {

				file->add_filter("*."+extensions[i]+" ; "+extensions[i].to_upper());
			}

			//file->set_current_path(current_path);
			file->popup_centered_ratio();

		} break;
		case TOOL_REPLACE: {

			create_dialog->popup_centered_ratio();
		} break;
		case TOOL_CONNECT: {

			Node *current = scene_tree->get_selected();
			if (!current)
				break;

			if (!_validate_no_foreign())
				break;
			connect_dialog->popup_centered_ratio();
			connect_dialog->set_node(current);

		} break;
		case TOOL_GROUP: {

			Node *current = scene_tree->get_selected();
			if (!current)
				break;
			if (!_validate_no_foreign())
				break;
			groups_editor->set_current(current);
			groups_editor->popup_centered_ratio();
		} break;
		case TOOL_SCRIPT: {

			Node *selected = scene_tree->get_selected();
			if (!selected)
				break;

			if (!_validate_no_foreign())
				break;

			Ref<Script> existing = selected->get_script();
			if (existing.is_valid())
				editor->push_item(existing.ptr());
			else {
				String path = selected->get_filename();
				script_create_dialog->config(selected->get_type(),path);
				script_create_dialog->popup_centered(Size2(300,290));
				//script_create_dialog->popup_centered_minsize();

			}

		} break;
		case TOOL_MOVE_UP:
		case TOOL_MOVE_DOWN: {

			if (!scene_tree->get_selected())
				break;


			if (scene_tree->get_selected()==edited_scene) {


				current_option=-1;
				//accept->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done on the tree root.");
				accept->popup_centered(Size2(300,70));;
				break;
			}


			if (!_validate_no_foreign())
				break;

			bool MOVING_DOWN = (p_tool == TOOL_MOVE_DOWN);
			bool MOVING_UP = !MOVING_DOWN;

			Node *common_parent = scene_tree->get_selected()->get_parent();
			List<Node*> selection = editor_selection->get_selected_node_list();
			selection.sort_custom<Node::Comparator>();  // sort by index
			if (MOVING_DOWN)
				selection.invert();

			int lowest_id = common_parent->get_child_count() - 1;
			int highest_id = 0;
			for (List<Node*>::Element *E = selection.front(); E; E = E->next()) {
				int index = E->get()->get_index();

				if (index > highest_id) highest_id = index;
				if (index < lowest_id) lowest_id = index;

				if (E->get()->get_parent() != common_parent)
					common_parent = NULL;
			}

			if (!common_parent || (MOVING_DOWN && highest_id >= common_parent->get_child_count() - MOVING_DOWN) || (MOVING_UP && lowest_id == 0))
				break; // one or more nodes can not be moved

			if (selection.size() == 1) editor_data->get_undo_redo().create_action("Move Node In Parent");
			if (selection.size() > 1) editor_data->get_undo_redo().create_action("Move Nodes In Parent");

			for (int i = 0; i < selection.size(); i++) {
				Node *top_node = selection[i];
				Node *bottom_node = selection[selection.size() - 1 - i];
				 
				ERR_FAIL_COND(!top_node->get_parent());
				ERR_FAIL_COND(!bottom_node->get_parent());

				int top_node_pos = top_node->get_index();
				int bottom_node_pos = bottom_node->get_index();

				int top_node_pos_next = top_node_pos + (MOVING_DOWN ? 1 : -1);
				int bottom_node_pos_next = bottom_node_pos + (MOVING_DOWN ? 1 : -1);

				editor_data->get_undo_redo().add_do_method(top_node->get_parent(), "move_child", top_node, top_node_pos_next);
				editor_data->get_undo_redo().add_undo_method(bottom_node->get_parent(), "move_child", bottom_node, bottom_node_pos);
			}

			editor_data->get_undo_redo().commit_action();

		} break;
		case TOOL_DUPLICATE: {

			if (!edited_scene)
				break;


			if (editor_selection->is_selected(edited_scene)) {


				current_option=-1;
				//accept->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done on the tree root.");
				accept->popup_centered(Size2(300,70));;
				break;
			}

			if (!_validate_no_foreign())
				break;

			List<Node*> selection = editor_selection->get_selected_node_list();

			List<Node*> reselect;

			editor_data->get_undo_redo().create_action("Duplicate Node(s)");
			editor_data->get_undo_redo().add_do_method(editor_selection,"clear");

			Node *dupsingle=NULL;


			for (List<Node*>::Element *E=selection.front();E;E=E->next()) {

				Node *node = E->get();
				Node *parent = node->get_parent();

				List<Node*> owned;
				node->get_owned_by(node->get_owner(),&owned);

				Map<Node*,Node*> duplimap;
				Node * dup = _duplicate(node,duplimap);

				ERR_CONTINUE(!dup);

				if (selection.size()==1)
					dupsingle=dup;

				String name = node->get_name();

				String nums;
				for(int i=name.length()-1;i>=0;i--) {
					CharType n=name[i];
					if (n>='0' && n<='9') {
						nums=String::chr(name[i])+nums;
					} else {
						break;
					}
				}

				int num=nums.to_int();
				if (num<1)
					num=1;
				else
					num++;

				String nnsep = _get_name_num_separator();
				name = name.substr(0,name.length()-nums.length()).strip_edges();
				if ( name.substr(name.length()-nnsep.length(),nnsep.length()) == nnsep) {
					name = name.substr(0,name.length()-nnsep.length());
				}
				String attempt = (name + nnsep + itos(num)).strip_edges();

				while(parent->has_node(attempt)) {
					num++;
					attempt = (name + nnsep + itos(num)).strip_edges();
				}

				dup->set_name(attempt);

				editor_data->get_undo_redo().add_do_method(parent,"add_child",dup);
				for (List<Node*>::Element *F=owned.front();F;F=F->next()) {

					if (!duplimap.has(F->get())) {

						continue;
					}
					Node *d=duplimap[F->get()];
					editor_data->get_undo_redo().add_do_method(d,"set_owner",node->get_owner());
				}
				editor_data->get_undo_redo().add_do_method(editor_selection,"add_node",dup);
				editor_data->get_undo_redo().add_undo_method(parent,"remove_child",dup);
				editor_data->get_undo_redo().add_do_reference(dup);

				//parent->add_child(dup);
				//reselect.push_back(dup);
			}

			editor_data->get_undo_redo().commit_action();

			if (dupsingle)
				editor->push_item(dupsingle);





		} break;
		case TOOL_REPARENT: {


			if (!scene_tree->get_selected())
				break;


			if (editor_selection->is_selected(edited_scene)) {


				current_option=-1;
				//confirmation->get_cancel()->hide();
				accept->get_ok()->set_text("I see..");
				accept->set_text("This operation can't be done on the tree root.");
				accept->popup_centered(Size2(300,70));;
				break;
			}

			if (!_validate_no_foreign())
				break;

			List<Node*> nodes = editor_selection->get_selected_node_list();
			Set<Node*> nodeset;
			for(List<Node*>::Element *E=nodes.front();E;E=E->next()) {

				nodeset.insert(E->get());
			}
			reparent_dialog->popup_centered_ratio();
			reparent_dialog->set_current( nodeset );

		} break;
		case TOOL_ERASE: {

			List<Node*> remove_list = editor_selection->get_selected_node_list();

			if (remove_list.empty())
				return;

			if (!_validate_no_foreign())
				break;

			if (p_confirm_override) {
				_delete_confirm();

				// hack, force 2d editor viewport to refresh after deletion
				if (CanvasItemEditor *editor = CanvasItemEditor::get_singleton())
					editor->get_viewport_control()->update();

			} else {
				delete_dialog->set_text("Delete Node(s)?");
				delete_dialog->popup_centered(Size2(200,80));
			}



		} break;

	}

}

void SceneTreeDock::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {

			static const char* button_names[TOOL_BUTTON_MAX]={
				"New",
				"Add",
				"Replace",
				"Connect",
				"Groups",
				"Script",
				"MoveUp",
				"MoveDown",
				"Duplicate",
				"Reparent",
				"Del",
			};

			for(int i=0;i<TOOL_BUTTON_MAX;i++)
				tool_buttons[i]->set_icon(get_icon(button_names[i],"EditorIcons"));

		} break;
		case NOTIFICATION_READY: {

			CanvasItemEditorPlugin *canvas_item_plugin =  editor_data->get_editor("2D")->cast_to<CanvasItemEditorPlugin>();
			if (canvas_item_plugin) {
				canvas_item_plugin->get_canvas_item_editor()->connect("item_lock_status_changed", scene_tree, "_update_tree");
				canvas_item_plugin->get_canvas_item_editor()->connect("item_group_status_changed", scene_tree, "_update_tree");
				scene_tree->connect("node_changed", canvas_item_plugin->get_canvas_item_editor()->get_viewport_control(), "update");
			}
		} break;
	}
}


void SceneTreeDock::_load_request(const String& p_path) {

	editor->open_request(p_path);
}

void SceneTreeDock::_script_open_request(const Ref<Script>& p_script) {

	editor->edit_resource(p_script);
}

void SceneTreeDock::_node_selected() {


	Node *node=scene_tree->get_selected();

	if (!node) {

		editor->push_item(NULL);
		return;
	}

	editor->push_item(node);
}

void SceneTreeDock::_node_renamed() {

	_node_selected();
}

Node *SceneTreeDock::_duplicate(Node *p_node, Map<Node*,Node*> &duplimap) {

	Node *node=NULL;

	if (p_node->get_filename()!="") { //an instance

		Ref<PackedScene> sd = ResourceLoader::load( p_node->get_filename() );
		ERR_FAIL_COND_V(!sd.is_valid(),NULL);
		node = sd->instance();
		ERR_FAIL_COND_V(!node,NULL);
		node->generate_instance_state();
	} else {
		Object *obj = ObjectTypeDB::instance(p_node->get_type());
		ERR_FAIL_COND_V(!obj,NULL);
		node = obj->cast_to<Node>();
		if (!node)
			memdelete(obj);
		ERR_FAIL_COND_V(!node,NULL);

	}

	List<PropertyInfo> plist;

	p_node->get_property_list(&plist);

	for(List<PropertyInfo>::Element *E=plist.front();E;E=E->next()) {

		if (!(E->get().usage&PROPERTY_USAGE_STORAGE))
			continue;
		String name = E->get().name;
		node->set( name, p_node->get(name) );

	}


	List<Node::GroupInfo> group_info;
	p_node->get_groups(&group_info);
	for (List<Node::GroupInfo>::Element *E=group_info.front();E;E=E->next()) {

		if (E->get().persistent)
			node->add_to_group(E->get().name,true);
	}


	node->set_name(p_node->get_name());
	duplimap[p_node]=node;

	for(int i=0;i<p_node->get_child_count();i++) {

		Node *child = p_node->get_child(i);
		if (p_node->get_owner()!=child->get_owner())
			continue; //don't bother with not in-scene nodes.

		Node *dup = _duplicate(child,duplimap);
		if (!dup) {
			memdelete(node);
			return NULL;
		}

		node->add_child(dup);
	}

	return node;

}


void SceneTreeDock::_set_owners(Node *p_owner, const Array& p_nodes) {

	for(int i=0;i<p_nodes.size();i++) {

		Object *obj=p_nodes[i];
		if (!obj)
			continue;

		Node *n=obj->cast_to<Node>();
		if (!n)
			continue;
		n->set_owner(p_owner);
	}
}


void SceneTreeDock::_fill_path_renames(Vector<StringName> base_path,Vector<StringName> new_base_path,Node * p_node, List<Pair<NodePath,NodePath> > *p_renames) {

	base_path.push_back(p_node->get_name());
	if (new_base_path.size())
		new_base_path.push_back(p_node->get_name());

	NodePath from( base_path,true );
	NodePath to;
	if (new_base_path.size())
		to=NodePath( new_base_path,true );

	Pair<NodePath,NodePath> npp;
	npp.first=from;
	npp.second=to;

	p_renames->push_back(npp);

	for(int i=0;i<p_node->get_child_count();i++) {

		_fill_path_renames(base_path,new_base_path,p_node->get_child(i),p_renames);
	}


}

void SceneTreeDock::fill_path_renames(Node* p_node, Node *p_new_parent, List<Pair<NodePath,NodePath> > *p_renames) {

	if (!bool(EDITOR_DEF("animation/autorename_animation_tracks",true)))
		return;


	Vector<StringName> base_path;
	Node *n = p_node->get_parent();
	while(n) {
		base_path.push_back(n->get_name());
		n=n->get_parent();
	}
	base_path.invert();

	Vector<StringName> new_base_path;
	if (p_new_parent) {
		n = p_new_parent;
		while(n) {
			new_base_path.push_back(n->get_name());
			n=n->get_parent();
		}

		new_base_path.invert();
	}

	_fill_path_renames(base_path,new_base_path,p_node,p_renames);
}

void SceneTreeDock::perform_node_renames(Node* p_base,List<Pair<NodePath,NodePath> > *p_renames, Map<Ref<Animation>, Set<int> > *r_rem_anims) {

	Map<Ref<Animation>, Set<int> > rem_anims;

	if (!r_rem_anims)
		r_rem_anims=&rem_anims;

	if (!bool(EDITOR_DEF("animation/autorename_animation_tracks",true)))
		return;

	if (!p_base) {

		p_base=edited_scene;
	}

	if (!p_base)
		return;


	if (p_base->cast_to<AnimationPlayer>()) {

		AnimationPlayer *ap=p_base->cast_to<AnimationPlayer>();
		List<StringName> anims;
		ap->get_animation_list(&anims);
		Node *root = ap->get_node(ap->get_root());


		if (root) {


			NodePath root_path=root->get_path();
			NodePath new_root_path=root_path;


			for(List<Pair<NodePath,NodePath> >::Element* E=p_renames->front();E;E=E->next()) {

				if (E->get().first==root_path) {
					new_root_path=E->get().second;
					break;
				}
			}

			if (new_root_path!=NodePath()) {
				//will not be erased

				for(List<StringName>::Element *E=anims.front();E;E=E->next()) {

					Ref<Animation> anim=ap->get_animation(E->get());
					if (!r_rem_anims->has(anim)) {
						r_rem_anims->insert(anim,Set<int>());
						Set<int> &ran = r_rem_anims->find(anim)->get();
						for(int i=0;i<anim->get_track_count();i++)
							ran.insert(i);
					}

					Set<int> &ran = r_rem_anims->find(anim)->get();

					if (anim.is_null())
						continue;

					for(int i=0;i<anim->get_track_count();i++) {

						NodePath track_np=anim->track_get_path(i);
						Node *n = root->get_node(track_np);
						if (!n) {
							continue;
						}

						NodePath old_np = n->get_path();

						if (!ran.has(i))
							continue; //channel was removed

						for(List<Pair<NodePath,NodePath> >::Element* E=p_renames->front();E;E=E->next()) {

							if (E->get().first==old_np) {


								if (E->get().second==NodePath()) {
									//will be erased

									int idx=0;
									Set<int>::Element *EI=ran.front();
									ERR_FAIL_COND(!EI); //bug
									while(EI->get()!=i) {
										idx++;
										EI=EI->next();
										ERR_FAIL_COND(!EI); //another bug

									}

									editor_data->get_undo_redo().add_do_method(anim.ptr(),"remove_track",idx);
									editor_data->get_undo_redo().add_undo_method(anim.ptr(),"add_track",anim->track_get_type(i),idx);
									editor_data->get_undo_redo().add_undo_method(anim.ptr(),"track_set_path",idx,track_np);
									editor_data->get_undo_redo().add_undo_method(anim.ptr(),"track_set_interpolation_type",idx,anim->track_get_interpolation_type(i));
									for(int j=0;j<anim->track_get_key_count(i);j++) {

										editor_data->get_undo_redo().add_undo_method(anim.ptr(),"track_insert_key",idx,anim->track_get_key_time(i,j),anim->track_get_key_value(i,j),anim->track_get_key_transition(i,j));
									}

									ran.erase(i); //byebye channel

								} else {
									//will be renamed
									NodePath rel_path = new_root_path.rel_path_to(E->get().second);

									NodePath new_path = NodePath( rel_path.get_names(), track_np.get_subnames(), false, track_np.get_property() );
									if (new_path==track_np)
										continue; //bleh
									editor_data->get_undo_redo().add_do_method(anim.ptr(),"track_set_path",i,new_path);
									editor_data->get_undo_redo().add_undo_method(anim.ptr(),"track_set_path",i,track_np);
								}
							}
						}
					}
				}
			}
		}
	}


	for(int i=0;i<p_base->get_child_count();i++)
		perform_node_renames(p_base->get_child(i),p_renames,r_rem_anims);

}


void SceneTreeDock::_node_prerenamed(Node* p_node, const String& p_new_name) {


	List<Pair<NodePath,NodePath> > path_renames;

	Vector<StringName> base_path;
	Node *n = p_node->get_parent();
	while(n) {
		base_path.push_back(n->get_name());
		n=n->get_parent();
	}
	base_path.invert();


	Vector<StringName> new_base_path=base_path;
	base_path.push_back(p_node->get_name());

	new_base_path.push_back(p_new_name);

	Pair<NodePath,NodePath> npp;
	npp.first = NodePath(base_path,true);
	npp.second = NodePath(new_base_path,true);
	path_renames.push_back(npp);


	for(int i=0;i<p_node->get_child_count();i++)
		_fill_path_renames(base_path,new_base_path,p_node->get_child(i),&path_renames);

	perform_node_renames(NULL,&path_renames);

}

bool SceneTreeDock::_validate_no_foreign() {

	List<Node*> selection = editor_selection->get_selected_node_list();

	for (List<Node*>::Element *E=selection.front();E;E=E->next()) {

		if (E->get()!=edited_scene && E->get()->get_owner()!=edited_scene) {

			accept->get_ok()->set_text("Makes Sense!");
			accept->set_text("Can't operate on nodes from a foreign scene!");
			accept->popup_centered(Size2(300,70));;
			return false;

		}
	}

	return true;
}

void SceneTreeDock::_node_reparent(NodePath p_path,bool p_node_only) {


	Node *node = scene_tree->get_selected();
	ERR_FAIL_COND(!node);
	ERR_FAIL_COND(node==edited_scene);
	Node *new_parent = scene_root->get_node(p_path);
	ERR_FAIL_COND(!new_parent);

	Node *validate=new_parent;
	while(validate) {

		if (editor_selection->is_selected(validate)) {
			ERR_EXPLAIN("Selection changed at some point.. can't reparent");
			ERR_FAIL();
			return;
		}
		validate=validate->get_parent();
	}

	//ok all valid

	List<Node*> selection = editor_selection->get_selected_node_list();

	if (selection.empty())
		return; //nothing to reparent

	//sort by tree order, so re-adding is easy
	selection.sort_custom<Node::Comparator>();

	editor_data->get_undo_redo().create_action("Reparent Node");

	List<Pair<NodePath,NodePath> > path_renames;

	for(List<Node*>::Element *E=selection.front();E;E=E->next()) {

		//no undo for now, sorry
		Node *node = E->get();

		fill_path_renames(node,new_parent,&path_renames);

		List<Node*> owned;
		node->get_owned_by(node->get_owner(),&owned);
		Array owners;
		for(List<Node*>::Element *E=owned.front();E;E=E->next()) {

			owners.push_back(E->get());
		}



		editor_data->get_undo_redo().add_do_method(node->get_parent(),"remove_child",node);
		editor_data->get_undo_redo().add_do_method(new_parent,"add_child",node);
		editor_data->get_undo_redo().add_do_method(this,"_set_owners",edited_scene,owners);

		if (editor->get_animation_editor()->get_root()==node)
			editor_data->get_undo_redo().add_do_method(editor->get_animation_editor(),"set_root",node);

		editor_data->get_undo_redo().add_undo_method(new_parent,"remove_child",node);

	}

	//add and move in a second step.. (so old order is preserved)



	for(List<Node*>::Element *E=selection.front();E;E=E->next()) {

		Node *node = E->get();

		List<Node*> owned;
		node->get_owned_by(node->get_owner(),&owned);
		Array owners;
		for(List<Node*>::Element *E=owned.front();E;E=E->next()) {

			owners.push_back(E->get());
		}

		int child_pos = node->get_position_in_parent();

		editor_data->get_undo_redo().add_undo_method(node->get_parent(),"add_child",node);
		editor_data->get_undo_redo().add_undo_method(node->get_parent(),"move_child",node,child_pos);
		editor_data->get_undo_redo().add_undo_method(this,"_set_owners",edited_scene,owners);
		if (editor->get_animation_editor()->get_root()==node)
			editor_data->get_undo_redo().add_undo_method(editor->get_animation_editor(),"set_root",node);

	}

	perform_node_renames(NULL,&path_renames);

	editor_data->get_undo_redo().commit_action();
	//node->set_owner(owner);
}


void SceneTreeDock::_script_created(Ref<Script> p_script) {

	Node *selected = scene_tree->get_selected();
	if (!selected)
		return;
	selected->set_script(p_script.get_ref_ptr());
	editor->push_item(p_script.operator->());

}


void SceneTreeDock::_delete_confirm() {

	List<Node*> remove_list = editor_selection->get_selected_node_list();

	if (remove_list.empty())
		return;


	if (editor->get_editor_plugin_over())
		editor->get_editor_plugin_over()->make_visible(false);

	editor_data->get_undo_redo().create_action("Remove Node(s)");

	bool entire_scene=false;

	for(List<Node*>::Element *E=remove_list.front();E;E=E->next()) {

		if (E->get()==edited_scene) {
			entire_scene=true;
		}
	}

	if (entire_scene) {

		editor_data->get_undo_redo().add_do_method(editor,"set_edited_scene",(Object*)NULL);
		editor_data->get_undo_redo().add_undo_method(editor,"set_edited_scene",edited_scene);
		editor_data->get_undo_redo().add_undo_method(edited_scene,"set_owner",edited_scene->get_owner());
		editor_data->get_undo_redo().add_undo_reference(edited_scene);

	} else {

		remove_list.sort_custom<Node::Comparator>(); //sort nodes to keep positions
		List<Pair<NodePath,NodePath> > path_renames;


		//delete from animation
		for(List<Node*>::Element *E=remove_list.front();E;E=E->next()) {
			Node *n = E->get();
			if (!n->is_inside_tree() || !n->get_parent())
				continue;

			fill_path_renames(n,NULL,&path_renames);

		}

		perform_node_renames(NULL,&path_renames);
		//delete for read
		for(List<Node*>::Element *E=remove_list.front();E;E=E->next()) {
			Node *n = E->get();
			if (!n->is_inside_tree() || !n->get_parent())
				continue;

			List<Node*> owned;
			n->get_owned_by(n->get_owner(),&owned);
			Array owners;
			for(List<Node*>::Element *E=owned.front();E;E=E->next()) {

				owners.push_back(E->get());
			}


			editor_data->get_undo_redo().add_do_method(n->get_parent(),"remove_child",n);
			editor_data->get_undo_redo().add_undo_method(n->get_parent(),"add_child",n);
			editor_data->get_undo_redo().add_undo_method(n->get_parent(),"move_child",n,n->get_index());
			if (editor->get_animation_editor()->get_root()==n)
				editor_data->get_undo_redo().add_undo_method(editor->get_animation_editor(),"set_root",n);
			editor_data->get_undo_redo().add_undo_method(this,"_set_owners",edited_scene,owners);
			//editor_data->get_undo_redo().add_undo_method(n,"set_owner",n->get_owner());
			editor_data->get_undo_redo().add_undo_reference(n);
		}


	}
	editor_data->get_undo_redo().commit_action();
	_update_tool_buttons();

}

void SceneTreeDock::_update_tool_buttons() {

	Node *sel = scene_tree->get_selected();
	bool disable = !sel || (sel!=edited_scene && sel->get_owner()!=edited_scene);
	bool disable_root = disable || sel->get_parent()==scene_root;

	tool_buttons[TOOL_INSTANCE]->set_disabled(disable);
	tool_buttons[TOOL_REPLACE]->set_disabled(disable);
	tool_buttons[TOOL_CONNECT]->set_disabled(disable);
	tool_buttons[TOOL_GROUP]->set_disabled(disable);
	tool_buttons[TOOL_SCRIPT]->set_disabled(disable);
	tool_buttons[TOOL_MOVE_UP]->set_disabled(disable_root);
	tool_buttons[TOOL_MOVE_DOWN]->set_disabled(disable_root);
	tool_buttons[TOOL_DUPLICATE]->set_disabled(disable_root);
	tool_buttons[TOOL_REPARENT]->set_disabled(disable_root);
	tool_buttons[TOOL_ERASE]->set_disabled(disable);

}

void SceneTreeDock::_create() {


	if (current_option==TOOL_NEW) {

		Node *parent=NULL;


		if (edited_scene) {

			parent = scene_tree->get_selected();
			ERR_FAIL_COND(!parent);
		} else {

			parent = scene_root;
			ERR_FAIL_COND(!parent);

		}

		Object *c = create_dialog->instance_selected();

		ERR_FAIL_COND(!c);
		Node *child=c->cast_to<Node>();
		ERR_FAIL_COND(!child);

		editor_data->get_undo_redo().create_action("Create Node");

		if (edited_scene) {
			editor_data->get_undo_redo().add_do_method(parent,"add_child",child);
			editor_data->get_undo_redo().add_do_method(child,"set_owner",edited_scene);
			editor_data->get_undo_redo().add_do_method(editor_selection,"clear");
			editor_data->get_undo_redo().add_do_method(editor_selection,"add_node",child);
			editor_data->get_undo_redo().add_do_reference(child);
			editor_data->get_undo_redo().add_undo_method(parent,"remove_child",child);
		} else {

			editor_data->get_undo_redo().add_do_method(editor,"set_edited_scene",child);
			editor_data->get_undo_redo().add_do_reference(child);
			editor_data->get_undo_redo().add_undo_method(editor,"set_edited_scene",(Object*)NULL);

		}

		editor_data->get_undo_redo().commit_action();
		editor->push_item(c);

		if (c->cast_to<Control>()) {
			//make editor more comfortable, so some controls don't appear super shrunk
			Control *ct = c->cast_to<Control>();

			Size2 ms = ct->get_minimum_size();
			if (ms.width<4)
				ms.width=40;
			if (ms.height<4)
				ms.height=40;
			ct->set_size(ms);
		}


	} else if (current_option==TOOL_REPLACE) {
		Node * n = scene_tree->get_selected();
		ERR_FAIL_COND(!n);

		Object *c = create_dialog->instance_selected();

		ERR_FAIL_COND(!c);
		Node *newnode=c->cast_to<Node>();
		ERR_FAIL_COND(!newnode);

		List<PropertyInfo> pinfo;
		n->get_property_list(&pinfo);

		for(List<PropertyInfo>::Element *E=pinfo.front();E;E=E->next()) {
			if (!(E->get().usage&PROPERTY_USAGE_STORAGE))
				continue;
			newnode->set(E->get().name,n->get(E->get().name));
		}

		editor->push_item(NULL);

		//reconnect signals
		List<MethodInfo> sl;

		n->get_signal_list(&sl);
		for (List<MethodInfo>::Element *E=sl.front();E;E=E->next()) {

			List<Object::Connection> cl;
			n->get_signal_connection_list(E->get().name,&cl);

			for(List<Object::Connection>::Element *F=cl.front();F;F=F->next()) {

				Object::Connection &c=F->get();
				if (!(c.flags&Object::CONNECT_PERSIST))
					continue;
				newnode->connect(c.signal,c.target,c.method,varray(),Object::CONNECT_PERSIST);
			}

		}


		String newname=n->get_name();
		n->replace_by(newnode,true);


		if (n==edited_scene) {
			edited_scene=newnode;
			editor->set_edited_scene(newnode);
		}




		editor_data->get_undo_redo().clear_history();
		memdelete(n);
		newnode->set_name(newname);
		editor->push_item(newnode);

		_update_tool_buttons();

	}

}


void SceneTreeDock::set_edited_scene(Node* p_scene) {

	edited_scene=p_scene;
}

void SceneTreeDock::set_selected(Node *p_node, bool p_emit_selected ) {

	scene_tree->set_selected(p_node,p_emit_selected);
	_update_tool_buttons();
}

void SceneTreeDock::import_subscene() {

	import_subscene_dialog->popup_centered_ratio();
}

void SceneTreeDock::_import_subscene() {

	Node* parent = scene_tree->get_selected();
	ERR_FAIL_COND(!parent);

	import_subscene_dialog->move(parent,edited_scene);
	editor_data->get_undo_redo().clear_history(); //no undo for now..


/*
	editor_data->get_undo_redo().create_action("Import Subscene");
	editor_data->get_undo_redo().add_do_method(parent,"add_child",ss);
	//editor_data->get_undo_redo().add_do_method(editor_selection,"clear");
	//editor_data->get_undo_redo().add_do_method(editor_selection,"add_node",child);
	editor_data->get_undo_redo().add_do_reference(ss);
	editor_data->get_undo_redo().add_undo_method(parent,"remove_child",ss);
	editor_data->get_undo_redo().commit_action();
*/
}

void SceneTreeDock::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_tool_selected"),&SceneTreeDock::_tool_selected);
	ObjectTypeDB::bind_method(_MD("_create"),&SceneTreeDock::_create);
	//ObjectTypeDB::bind_method(_MD("_script_created"),&SceneTreeDock::_script_created);
	ObjectTypeDB::bind_method(_MD("_node_reparent"),&SceneTreeDock::_node_reparent);
	ObjectTypeDB::bind_method(_MD("_set_owners"),&SceneTreeDock::_set_owners);
	ObjectTypeDB::bind_method(_MD("_node_selected"),&SceneTreeDock::_node_selected);
	ObjectTypeDB::bind_method(_MD("_node_renamed"),&SceneTreeDock::_node_renamed);
	ObjectTypeDB::bind_method(_MD("_script_created"),&SceneTreeDock::_script_created);
	ObjectTypeDB::bind_method(_MD("_load_request"),&SceneTreeDock::_load_request);
	ObjectTypeDB::bind_method(_MD("_script_open_request"),&SceneTreeDock::_script_open_request);
	ObjectTypeDB::bind_method(_MD("_unhandled_key_input"),&SceneTreeDock::_unhandled_key_input);
	ObjectTypeDB::bind_method(_MD("_delete_confirm"),&SceneTreeDock::_delete_confirm);
	ObjectTypeDB::bind_method(_MD("_node_prerenamed"),&SceneTreeDock::_node_prerenamed);
	ObjectTypeDB::bind_method(_MD("_import_subscene"),&SceneTreeDock::_import_subscene);

	ObjectTypeDB::bind_method(_MD("instance"),&SceneTreeDock::instance);
}



SceneTreeDock::SceneTreeDock(EditorNode *p_editor,Node *p_scene_root,EditorSelection *p_editor_selection,EditorData &p_editor_data)  {

	editor=p_editor;
	edited_scene=NULL;
	editor_data=&p_editor_data;
	editor_selection=p_editor_selection;
	scene_root=p_scene_root;

	VBoxContainer *vbc = this;

	HBoxContainer *hbc_top = memnew( HBoxContainer );
	vbc->add_child(hbc_top);

	ToolButton *tb;

	tb = memnew( ToolButton );
	tb->connect("pressed",this,"_tool_selected",make_binds(TOOL_NEW, false));
	tb->set_tooltip("Add/Create a New Node\n("+keycode_get_string(KEY_MASK_CMD|KEY_A)+")");
	hbc_top->add_child(tb);
	tool_buttons[TOOL_NEW]=tb;

	tb = memnew( ToolButton );
	tb->connect("pressed",this,"_tool_selected",make_binds(TOOL_INSTANCE, false));
	tb->set_tooltip("Instance a Node from scene file.");
	hbc_top->add_child(tb);
	tool_buttons[TOOL_INSTANCE]=tb;

	tb = memnew( ToolButton );
	tb->connect("pressed",this,"_tool_selected",make_binds(TOOL_REPLACE, false));
	tb->set_tooltip("Replace a Node by Another Node Type");
	hbc_top->add_child(tb);
	tool_buttons[TOOL_REPLACE]=tb;

	hbc_top->add_spacer();

	tb = memnew( ToolButton );
	tb->connect("pressed",this,"_tool_selected",make_binds(TOOL_CONNECT, false));
	tb->set_tooltip("Edit the Node Connections");
	hbc_top->add_child(tb);
	tool_buttons[TOOL_CONNECT]=tb;

	tb = memnew( ToolButton );
	tb->connect("pressed",this,"_tool_selected",make_binds(TOOL_GROUP, false));
	tb->set_tooltip("Edit the Node Groups");
	hbc_top->add_child(tb);
	tool_buttons[TOOL_GROUP]=tb;

	tb = memnew( ToolButton );
	tb->connect("pressed",this,"_tool_selected",make_binds(TOOL_SCRIPT, false));
	tb->set_tooltip("Edit/Create the Node Script");
	hbc_top->add_child(tb);
	tool_buttons[TOOL_SCRIPT]=tb;


	scene_tree = memnew( SceneTreeEditor(false,true,true ));
	vbc->add_child(scene_tree);
	scene_tree->set_v_size_flags(SIZE_EXPAND|SIZE_FILL);

	scene_tree->connect("node_selected", this,"_node_selected",varray(),CONNECT_DEFERRED);
	scene_tree->connect("node_renamed", this,"_node_renamed",varray(),CONNECT_DEFERRED);
	scene_tree->connect("node_prerename", this,"_node_prerenamed");
	scene_tree->connect("open",this,"_load_request");
	scene_tree->connect("open_script",this,"_script_open_request");
	scene_tree->set_undo_redo(&editor_data->get_undo_redo());
	scene_tree->set_editor_selection(editor_selection);

	HBoxContainer *hbc_bottom = memnew( HBoxContainer );
	vbc->add_child(hbc_bottom);


	tb = memnew( ToolButton );
	tb->connect("pressed",this,"_tool_selected",make_binds(TOOL_MOVE_UP, false));
	tb->set_tooltip("Move Node Up\n("+keycode_get_string(KEY_MASK_CMD|KEY_UP)+")");
	hbc_bottom->add_child(tb);
	tool_buttons[TOOL_MOVE_UP]=tb;

	tb = memnew( ToolButton );
	tb->connect("pressed",this,"_tool_selected",make_binds(TOOL_MOVE_DOWN, false));
	tb->set_tooltip("Move Node Down\n("+keycode_get_string(KEY_MASK_CMD|KEY_DOWN)+")");
	hbc_bottom->add_child(tb);
	tool_buttons[TOOL_MOVE_DOWN]=tb;

	tb = memnew( ToolButton );
	tb->connect("pressed",this,"_tool_selected",make_binds(TOOL_DUPLICATE, false));
	tb->set_tooltip("Duplicate Selected Node(s)\n("+keycode_get_string(KEY_MASK_CMD|KEY_D)+")");
	hbc_bottom->add_child(tb);
	tool_buttons[TOOL_DUPLICATE]=tb;

	tb = memnew( ToolButton );
	tb->connect("pressed",this,"_tool_selected",make_binds(TOOL_REPARENT, false));
	tb->set_tooltip("Reparent Selected Node(s)");
	hbc_bottom->add_child(tb);
	tool_buttons[TOOL_REPARENT]=tb;

	hbc_bottom->add_spacer();

	tb = memnew( ToolButton );
	tb->connect("pressed",this,"_tool_selected",make_binds(TOOL_ERASE, false));
	tb->set_tooltip("Erase Selected Node(s)");
	hbc_bottom->add_child(tb);
	tool_buttons[TOOL_ERASE]=tb;

	create_dialog = memnew( CreateDialog );
	create_dialog->set_base_type("Node");
	add_child(create_dialog);
	create_dialog->connect("create",this,"_create");

	groups_editor = memnew( GroupsEditor );
	add_child(groups_editor);
	groups_editor->set_undo_redo(&editor_data->get_undo_redo());
	connect_dialog = memnew( ConnectionsDialog(p_editor) );
	add_child(connect_dialog);
	connect_dialog->set_undoredo(&editor_data->get_undo_redo());
	script_create_dialog = memnew( ScriptCreateDialog );
	add_child(script_create_dialog);
	script_create_dialog->connect("script_created",this,"_script_created");
	reparent_dialog = memnew( ReparentDialog );
	add_child(reparent_dialog);
	reparent_dialog->connect("reparent",this,"_node_reparent");

	accept = memnew( AcceptDialog );
	add_child(accept);

	file = memnew( FileDialog );
	add_child(file);
	file->connect("file_selected",this,"instance");
	set_process_unhandled_key_input(true);

	delete_dialog = memnew( ConfirmationDialog );
	add_child(delete_dialog);
	delete_dialog->connect("confirmed",this,"_delete_confirm");
	import_subscene_dialog = memnew( EditorSubScene );
	add_child(import_subscene_dialog);
	import_subscene_dialog->connect("subscene_selected",this,"_import_subscene");




}
