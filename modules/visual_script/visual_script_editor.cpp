#include "visual_script_editor.h"
#include "tools/editor/editor_node.h"
#include "visual_script_nodes.h"
#include "visual_script_flow_control.h"
#include "visual_script_func_nodes.h"

class VisualScriptEditorSignalEdit : public Object {

	OBJ_TYPE(VisualScriptEditorSignalEdit,Object)

	StringName sig;
public:
	UndoRedo *undo_redo;
	Ref<VisualScript> script;


protected:

	static void _bind_methods() {
		ObjectTypeDB::bind_method("_sig_changed",&VisualScriptEditorSignalEdit::_sig_changed);
	}

	void _sig_changed() {

		_change_notify();
	}

	bool _set(const StringName& p_name, const Variant& p_value) {

		if (sig==StringName())
			return false;

		if (p_name=="argument_count") {

			int new_argc=p_value;
			int argc = script->custom_signal_get_argument_count(sig);
			if (argc==new_argc)
				return true;

			undo_redo->create_action("Change Signal Arguments");



			if (new_argc < argc) {
				for(int i=new_argc;i<argc;i++) {
					undo_redo->add_do_method(script.ptr(),"custom_signal_remove_argument",sig,new_argc);
					undo_redo->add_undo_method(script.ptr(),"custom_signal_add_argument",sig,script->custom_signal_get_argument_name(sig,i),script->custom_signal_get_argument_type(sig,i),-1);
				}
			} else if (new_argc>argc) {

				for(int i=argc;i<new_argc;i++) {

					undo_redo->add_do_method(script.ptr(),"custom_signal_add_argument",sig,Variant::NIL,"arg"+itos(i+1),-1);
					undo_redo->add_undo_method(script.ptr(),"custom_signal_remove_argument",sig,argc);
				}
			}

			undo_redo->add_do_method(this,"_sig_changed");
			undo_redo->add_undo_method(this,"_sig_changed");

			undo_redo->commit_action();

			return true;
		}
		if (String(p_name).begins_with("argument/")) {
			int idx = String(p_name).get_slice("/",1).to_int()-1;
			ERR_FAIL_INDEX_V(idx,script->custom_signal_get_argument_count(sig),false);
			String what = String(p_name).get_slice("/",2);
			if (what=="type") {

				int old_type = script->custom_signal_get_argument_type(sig,idx);
				int new_type=p_value;
				undo_redo->create_action("Change Argument Type");
				undo_redo->add_do_method(script.ptr(),"custom_signal_set_argument_type",sig,idx,new_type);
				undo_redo->add_undo_method(script.ptr(),"custom_signal_set_argument_type",sig,idx,old_type);
				undo_redo->commit_action();

				return true;
			}

			if (what=="name") {

				String old_name = script->custom_signal_get_argument_name(sig,idx);
				String new_name=p_value;
				undo_redo->create_action("Change Argument name");
				undo_redo->add_do_method(script.ptr(),"custom_signal_set_argument_name",sig,idx,new_name);
				undo_redo->add_undo_method(script.ptr(),"custom_signal_set_argument_name",sig,idx,old_name);
				undo_redo->commit_action();
				return true;
			}


		}


		return false;
	}

	bool _get(const StringName& p_name,Variant &r_ret) const {

		if (sig==StringName())
			return false;

		if (p_name=="argument_count") {
			r_ret = script->custom_signal_get_argument_count(sig);
			return true;
		}
		if (String(p_name).begins_with("argument/")) {
			int idx = String(p_name).get_slice("/",1).to_int()-1;
			ERR_FAIL_INDEX_V(idx,script->custom_signal_get_argument_count(sig),false);
			String what = String(p_name).get_slice("/",2);
			if (what=="type") {
				r_ret = script->custom_signal_get_argument_type(sig,idx);
				return true;
			}
			if (what=="name") {
				r_ret = script->custom_signal_get_argument_name(sig,idx);
				return true;
			}



		}

		return false;
	}
	void _get_property_list( List<PropertyInfo> *p_list) const {

		if (sig==StringName())
			return;

		p_list->push_back(PropertyInfo(Variant::INT,"argument_count",PROPERTY_HINT_RANGE,"0,256"));
		String argt="Variant";
		for(int i=1;i<Variant::VARIANT_MAX;i++) {
			argt+=","+Variant::get_type_name(Variant::Type(i));
		}

		for(int i=0;i<script->custom_signal_get_argument_count(sig);i++) {
			p_list->push_back(PropertyInfo(Variant::INT,"argument/"+itos(i+1)+"/type",PROPERTY_HINT_ENUM,argt));
			p_list->push_back(PropertyInfo(Variant::STRING,"argument/"+itos(i+1)+"/name"));
		}
	}

public:


	void edit(const StringName& p_sig) {

		sig=p_sig;
		_change_notify();
	}

	VisualScriptEditorSignalEdit() { undo_redo=NULL; }
};

class VisualScriptEditorVariableEdit : public Object {

	OBJ_TYPE(VisualScriptEditorVariableEdit,Object)

	StringName var;
public:
	UndoRedo *undo_redo;
	Ref<VisualScript> script;


protected:

	static void _bind_methods() {
		ObjectTypeDB::bind_method("_var_changed",&VisualScriptEditorVariableEdit::_var_changed);
		ObjectTypeDB::bind_method("_var_value_changed",&VisualScriptEditorVariableEdit::_var_value_changed);
	}

	void _var_changed() {

		_change_notify();
	}
	void _var_value_changed() {

		_change_notify("value"); //so the whole tree is not redrawn, makes editing smoother in general
	}

	bool _set(const StringName& p_name, const Variant& p_value) {

		if (var==StringName())
			return false;



		if (String(p_name)=="value") {
			undo_redo->create_action("Set Variable Default Value");
			Variant current=script->get_variable_default_value(var);
			undo_redo->add_do_method(script.ptr(),"set_variable_default_value",var,p_value);
			undo_redo->add_undo_method(script.ptr(),"set_variable_default_value",var,current);
			undo_redo->add_do_method(this,"_var_value_changed");
			undo_redo->add_undo_method(this,"_var_value_changed");
			undo_redo->commit_action();
			return true;
		}

		Dictionary d = script->call("get_variable_info",var);

		if (String(p_name)=="type") {

			Dictionary dc=d.copy();
			dc["type"]=p_value;
			undo_redo->create_action("Set Variable Type");
			undo_redo->add_do_method(script.ptr(),"set_variable_info",var,dc);
			undo_redo->add_undo_method(script.ptr(),"set_variable_info",var,d);
			undo_redo->add_do_method(this,"_var_changed");
			undo_redo->add_undo_method(this,"_var_changed");
			undo_redo->commit_action();
			return true;
		}

		if (String(p_name)=="hint") {

			Dictionary dc=d.copy();
			dc["hint"]=p_value;
			undo_redo->create_action("Set Variable Type");
			undo_redo->add_do_method(script.ptr(),"set_variable_info",var,dc);
			undo_redo->add_undo_method(script.ptr(),"set_variable_info",var,d);
			undo_redo->add_do_method(this,"_var_changed");
			undo_redo->add_undo_method(this,"_var_changed");
			undo_redo->commit_action();
			return true;
		}

		if (String(p_name)=="hint_string") {

			Dictionary dc=d.copy();
			dc["hint_string"]=p_value;
			undo_redo->create_action("Set Variable Type");
			undo_redo->add_do_method(script.ptr(),"set_variable_info",var,dc);
			undo_redo->add_undo_method(script.ptr(),"set_variable_info",var,d);
			undo_redo->add_do_method(this,"_var_changed");
			undo_redo->add_undo_method(this,"_var_changed");
			undo_redo->commit_action();
			return true;
		}


		return false;
	}

	bool _get(const StringName& p_name,Variant &r_ret) const {

		if (var==StringName())
			return false;

		if (String(p_name)=="value") {
			r_ret=script->get_variable_default_value(var);
			return true;
		}

		PropertyInfo pinfo = script->get_variable_info(var);

		if (String(p_name)=="type") {
			r_ret=pinfo.type;
			return true;
		}
		if (String(p_name)=="hint") {
			r_ret=pinfo.hint;
			return true;
		}
		if (String(p_name)=="hint_string") {
			r_ret=pinfo.hint_string;
			return true;
		}

		return false;
	}
	void _get_property_list( List<PropertyInfo> *p_list) const {

		if (var==StringName())
			return;

		String argt="Variant";
		for(int i=1;i<Variant::VARIANT_MAX;i++) {
			argt+=","+Variant::get_type_name(Variant::Type(i));
		}
		p_list->push_back(PropertyInfo(Variant::INT,"type",PROPERTY_HINT_ENUM,argt));
		p_list->push_back(PropertyInfo(script->get_variable_info(var).type,"value",script->get_variable_info(var).hint,script->get_variable_info(var).hint_string,PROPERTY_USAGE_DEFAULT));
		p_list->push_back(PropertyInfo(Variant::INT,"hint",PROPERTY_HINT_ENUM,"None,Range,ExpRange,Enum,ExpEasing,Length,SpriteFrame,KeyAccel,BitFlags,AllFlags,File,Dir,GlobalFile,GlobalDir,ResourceType,MultilineText"));
		p_list->push_back(PropertyInfo(Variant::STRING,"hint_string"));

	}

public:


	void edit(const StringName& p_var) {

		var=p_var;
		_change_notify();
	}

	VisualScriptEditorVariableEdit() { undo_redo=NULL; }
};

static Color _color_from_type(Variant::Type p_type) {

	Color color;
	color.set_hsv(p_type/float(Variant::VARIANT_MAX),0.7,0.7);
	return color;
}



void VisualScriptEditor::_update_graph_connections() {

	graph->clear_connections();

	List<VisualScript::SequenceConnection> sequence_conns;
	script->get_sequence_connection_list(edited_func,&sequence_conns);


	for (List<VisualScript::SequenceConnection>::Element *E=sequence_conns.front();E;E=E->next()) {

		graph->connect_node(itos(E->get().from_node),E->get().from_output,itos(E->get().to_node),0);
	}

	List<VisualScript::DataConnection> data_conns;
	script->get_data_connection_list(edited_func,&data_conns);

	for (List<VisualScript::DataConnection>::Element *E=data_conns.front();E;E=E->next()) {

		VisualScript::DataConnection dc=E->get();


		Ref<VisualScriptNode> from_node = script->get_node(edited_func,E->get().from_node);
		Ref<VisualScriptNode> to_node = script->get_node(edited_func,E->get().to_node);

		if (to_node->has_input_sequence_port()) {
			dc.to_port++;
		}

		dc.from_port+=from_node->get_output_sequence_port_count();

		graph->connect_node(itos(E->get().from_node),dc.from_port,itos(E->get().to_node),dc.to_port);
	}

}


void VisualScriptEditor::_update_graph(int p_only_id) {


	//byebye all nodes
	if (p_only_id>=0) {
		if (graph->has_node(itos(p_only_id))) {
			Node* gid = graph->get_node(itos(p_only_id));
			if (gid)
				memdelete(gid);
		}
	} else {

		for(int i=0;i<graph->get_child_count();i++) {

			if (graph->get_child(i)->cast_to<GraphNode>()) {
				memdelete(graph->get_child(i));
				i--;
			}
		}
	}

	if (!script->has_function(edited_func))
		return;

	Ref<Texture> type_icons[Variant::VARIANT_MAX]={
		Control::get_icon("MiniVariant","EditorIcons"),
		Control::get_icon("MiniBoolean","EditorIcons"),
		Control::get_icon("MiniInteger","EditorIcons"),
		Control::get_icon("MiniFloat","EditorIcons"),
		Control::get_icon("MiniString","EditorIcons"),
		Control::get_icon("MiniVector2","EditorIcons"),
		Control::get_icon("MiniRect2","EditorIcons"),
		Control::get_icon("MiniVector3","EditorIcons"),
		Control::get_icon("MiniMatrix2","EditorIcons"),
		Control::get_icon("MiniPlane","EditorIcons"),
		Control::get_icon("MiniQuat","EditorIcons"),
		Control::get_icon("MiniAabb","EditorIcons"),
		Control::get_icon("MiniMatrix3","EditorIcons"),
		Control::get_icon("MiniTransform","EditorIcons"),
		Control::get_icon("MiniColor","EditorIcons"),
		Control::get_icon("MiniImage","EditorIcons"),
		Control::get_icon("MiniPath","EditorIcons"),
		Control::get_icon("MiniRid","EditorIcons"),
		Control::get_icon("MiniObject","EditorIcons"),
		Control::get_icon("MiniInput","EditorIcons"),
		Control::get_icon("MiniDictionary","EditorIcons"),
		Control::get_icon("MiniArray","EditorIcons"),
		Control::get_icon("MiniRawArray","EditorIcons"),
		Control::get_icon("MiniIntArray","EditorIcons"),
		Control::get_icon("MiniFloatArray","EditorIcons"),
		Control::get_icon("MiniStringArray","EditorIcons"),
		Control::get_icon("MiniVector2Array","EditorIcons"),
		Control::get_icon("MiniVector3Array","EditorIcons"),
		Control::get_icon("MiniColorArray","EditorIcons")
	};



	Ref<Texture> seq_port = Control::get_icon("VisualShaderPort","EditorIcons");

	List<int> ids;
	script->get_node_list(edited_func,&ids);
	StringName editor_icons="EditorIcons";

	for(List<int>::Element *E=ids.front();E;E=E->next()) {

		if (p_only_id>=0 && p_only_id!=E->get())
			continue;

		Ref<VisualScriptNode> node = script->get_node(edited_func,E->get());
		Vector2 pos = script->get_node_pos(edited_func,E->get());

		GraphNode *gnode = memnew( GraphNode );
		gnode->set_title(node->get_caption());

		gnode->set_meta("__vnode",node);
		gnode->set_name(itos(E->get()));
		gnode->connect("dragged",this,"_node_moved",varray(E->get()));
		gnode->connect("close_request",this,"_remove_node",varray(E->get()),CONNECT_DEFERRED);


		if (E->get()!=script->get_function_node_id(edited_func)) {
			//function can't be erased
			gnode->set_show_close_button(true);
		}

		Label *text = memnew( Label );
		text->set_text(node->get_text());
		gnode->add_child(text);

		int slot_idx=0;

		bool single_seq_output = node->get_output_sequence_port_count()==1 && node->get_output_sequence_port_text(0)==String();
		gnode->set_slot(0,node->has_input_sequence_port(),TYPE_SEQUENCE,Color(1,1,1,1),single_seq_output,TYPE_SEQUENCE,Color(1,1,1,1),seq_port,seq_port);
		gnode->set_offset(pos*EDSCALE);
		slot_idx++;

		if (!single_seq_output) {
			for(int i=0;i<node->get_output_sequence_port_count();i++) {

				Label *text2 = memnew( Label );
				text2->set_text(node->get_output_sequence_port_text(i));
				text2->set_align(Label::ALIGN_RIGHT);
				gnode->add_child(text2);
				gnode->set_slot(slot_idx,false,0,Color(),true,TYPE_SEQUENCE,Color(1,1,1,1),seq_port,seq_port);
				slot_idx++;
			}
		}

		for(int i=0;i<MAX(node->get_output_value_port_count(),node->get_input_value_port_count());i++) {

			bool left_ok=false;
			Variant::Type left_type=Variant::NIL;
			String left_name;

			if (i<node->get_input_value_port_count()) {
				PropertyInfo pi = node->get_input_value_port_info(i);
				left_ok=true;
				left_type=pi.type;
				left_name=pi.name;
			}

			bool right_ok=false;
			Variant::Type right_type=Variant::NIL;
			String right_name;

			if (i<node->get_output_value_port_count()) {
				PropertyInfo pi = node->get_output_value_port_info(i);
				right_ok=true;
				right_type=pi.type;
				right_name=pi.name;
			}

			HBoxContainer *hbc = memnew( HBoxContainer);

			if (left_ok) {

				Ref<Texture> t;
				if (left_type>=0 && left_type<Variant::VARIANT_MAX) {
					t=type_icons[left_type];
				}
				if (t.is_valid()) {
					TextureFrame *tf = memnew(TextureFrame);
					tf->set_texture(t);
					tf->set_stretch_mode(TextureFrame::STRETCH_KEEP_CENTERED);
					hbc->add_child(tf);
				}

				hbc->add_child(memnew(Label(left_name)));
			} else {
				Control *c = memnew(Control);
				c->set_custom_minimum_size(Size2(10,0)*EDSCALE);
				hbc->add_child(c);
			}

			hbc->add_spacer();

			if (right_ok) {

				hbc->add_child(memnew(Label(right_name)));

				Ref<Texture> t;
				if (right_type>=0 && right_type<Variant::VARIANT_MAX) {
					t=type_icons[right_type];
				}
				if (t.is_valid()) {
					TextureFrame *tf = memnew(TextureFrame);
					tf->set_texture(t);
					tf->set_stretch_mode(TextureFrame::STRETCH_KEEP_CENTERED);
					hbc->add_child(tf);
				}

			}

			gnode->add_child(hbc);

			gnode->set_slot(slot_idx,left_ok,left_type,_color_from_type(left_type),right_ok,right_type,_color_from_type(right_type));

			slot_idx++;
		}

		graph->add_child(gnode);
	}

	_update_graph_connections();

}

void VisualScriptEditor::_update_members() {


	updating_members=true;

	members->clear();
	TreeItem *root = members->create_item();

	TreeItem *functions = members->create_item(root);
	functions->set_selectable(0,false);
	functions->set_text(0,TTR("Functions:"));
	functions->add_button(0,Control::get_icon("Override","EditorIcons"),1);
	functions->add_button(0,Control::get_icon("Add","EditorIcons"),0);
	functions->set_custom_bg_color(0,Control::get_color("prop_section","Editor"));

	List<StringName> func_names;
	script->get_function_list(&func_names);
	for (List<StringName>::Element *E=func_names.front();E;E=E->next()) {
		TreeItem *ti = members->create_item(functions)		;
		ti->set_text(0,E->get());
		ti->set_selectable(0,true);
		ti->set_editable(0,true);
		//ti->add_button(0,Control::get_icon("Edit","EditorIcons"),0); function arguments are in the node now
		ti->add_button(0,Control::get_icon("Del","EditorIcons"),1);
		ti->set_metadata(0,E->get());
		if (selected==E->get())
			ti->select(0);
	}

	TreeItem *variables = members->create_item(root);
	variables->set_selectable(0,false);
	variables->set_text(0,TTR("Variables:"));
	variables->add_button(0,Control::get_icon("Add","EditorIcons"));
	variables->set_custom_bg_color(0,Control::get_color("prop_section","Editor"));


	List<StringName> var_names;
	script->get_variable_list(&var_names);
	for (List<StringName>::Element *E=var_names.front();E;E=E->next()) {
		TreeItem *ti = members->create_item(variables);
		ti->set_text(0,E->get());
		ti->set_selectable(0,true);
		ti->set_editable(0,true);
		ti->add_button(0,Control::get_icon("Edit","EditorIcons"),0);
		ti->add_button(0,Control::get_icon("Del","EditorIcons"),1);
		ti->set_metadata(0,E->get());
		if (selected==E->get())
			ti->select(0);
	}

	TreeItem *_signals = members->create_item(root);
	_signals->set_selectable(0,false);
	_signals->set_text(0,TTR("Signals:"));
	_signals->add_button(0,Control::get_icon("Add","EditorIcons"));
	_signals->set_custom_bg_color(0,Control::get_color("prop_section","Editor"));

	List<StringName> signal_names;
	script->get_custom_signal_list(&signal_names);
	for (List<StringName>::Element *E=signal_names.front();E;E=E->next()) {
		TreeItem *ti = members->create_item(_signals);
		ti->set_text(0,E->get());
		ti->set_selectable(0,true);
		ti->set_editable(0,true);
		ti->add_button(0,Control::get_icon("Edit","EditorIcons"),0);
		ti->add_button(0,Control::get_icon("Del","EditorIcons"),1);
		ti->set_metadata(0,E->get());
		if (selected==E->get())
			ti->select(0);
	}

	String base_type=script->get_instance_base_type();
	String icon_type=base_type;
	if (!Control::has_icon(base_type,"EditorIcons")) {
		icon_type="Object";
	}

	base_type_select->set_text(base_type);
	base_type_select->set_icon(Control::get_icon(icon_type,"EditorIcons"));

	updating_members=false;

}

void VisualScriptEditor::_member_selected() {

	if (updating_members)
		return;

	TreeItem *ti=members->get_selected();
	ERR_FAIL_COND(!ti);


	selected=ti->get_metadata(0);
//	print_line("selected: "+String(selected));


	if (ti->get_parent()==members->get_root()->get_children()) {

		if (edited_func!=selected) {

			revert_on_drag=edited_func;
			edited_func=selected;
			_update_graph();
		}

		return; //or crash because it will become invalid

	}



}

void VisualScriptEditor::_member_edited() {

	if (updating_members)
		return;

	TreeItem *ti=members->get_edited();
	ERR_FAIL_COND(!ti);

	String name = ti->get_metadata(0);
	String new_name = ti->get_text(0);

	if (name==new_name)
		return;

	if (!new_name.is_valid_identifier()) {

		EditorNode::get_singleton()->show_warning(TTR("Name is not a valid identifier:")+" "+new_name);
		updating_members=true;
		ti->set_text(0,name);
		updating_members=false;
		return;

	}

	if (script->has_function(new_name) || script->has_variable(new_name) || script->has_custom_signal(new_name)) {

		EditorNode::get_singleton()->show_warning(TTR("Name already in use by another func/var/signal:")+" "+new_name);
		updating_members=true;
		ti->set_text(0,name);
		updating_members=false;
		return;
	}

	TreeItem *root=members->get_root();

	if (ti->get_parent()==root->get_children()) {

		if (edited_func==selected) {
			edited_func=new_name;
		}
		selected=new_name;


		_update_graph();

		undo_redo->create_action(TTR("Rename Function"));
		undo_redo->add_do_method(script.ptr(),"rename_function",name,new_name);
		undo_redo->add_undo_method(script.ptr(),"rename_function",new_name,name);
		undo_redo->add_do_method(this,"_update_members");
		undo_redo->add_undo_method(this,"_update_members");
		undo_redo->commit_action();

		return; //or crash because it will become invalid

	}

	if (ti->get_parent()==root->get_children()->get_next()) {

		selected=new_name;
		undo_redo->create_action(TTR("Rename Variable"));
		undo_redo->add_do_method(script.ptr(),"rename_variable",name,new_name);
		undo_redo->add_undo_method(script.ptr(),"rename_variable",new_name,name);
		undo_redo->add_do_method(this,"_update_members");
		undo_redo->add_undo_method(this,"_update_members");
		undo_redo->commit_action();

		return; //or crash because it will become invalid
	}

	if (ti->get_parent()==root->get_children()->get_next()->get_next()) {

		selected=new_name;
		undo_redo->create_action(TTR("Rename Signal"));
		undo_redo->add_do_method(script.ptr(),"rename_custom_signal",name,new_name);
		undo_redo->add_undo_method(script.ptr(),"rename_custom_signal",new_name,name);
		undo_redo->add_do_method(this,"_update_members");
		undo_redo->add_undo_method(this,"_update_members");
		undo_redo->commit_action();

		return; //or crash because it will become invalid
	}
}

void VisualScriptEditor::_override_pressed(int p_id) {

	//override a virtual function or method from base type

	ERR_FAIL_COND(!virtuals_in_menu.has(p_id));

	VirtualInMenu vim=virtuals_in_menu[p_id];

	String name = _validate_name(vim.name);
	selected=name;
	edited_func=selected;
	Ref<VisualScriptFunction> func_node;
	func_node.instance();
	func_node->set_name(vim.name);

	undo_redo->create_action(TTR("Add Function"));
	undo_redo->add_do_method(script.ptr(),"add_function",name);
	for(int i=0;i<vim.args.size();i++) {
		func_node->add_argument(vim.args[i].first,vim.args[i].second);
	}


	undo_redo->add_do_method(script.ptr(),"add_node",name,script->get_available_id(),func_node);
	if (vim.ret!=Variant::NIL || vim.ret_variant) {
		Ref<VisualScriptReturn> ret_node;
		ret_node.instance();
		ret_node->set_return_type(vim.ret);
		ret_node->set_enable_return_value(true);
		ret_node->set_name(vim.name);
		undo_redo->add_do_method(script.ptr(),"add_node",name,script->get_available_id()+1,ret_node,Vector2(500,0));

	}

	undo_redo->add_undo_method(script.ptr(),"remove_function",name);
	undo_redo->add_do_method(this,"_update_members");
	undo_redo->add_undo_method(this,"_update_members");
	undo_redo->commit_action();


	_update_graph();
}

void VisualScriptEditor::_member_button(Object *p_item, int p_column, int p_button) {

	TreeItem *ti=p_item->cast_to<TreeItem>();

	TreeItem *root=members->get_root();

	if (ti->get_parent()==root) {
		//main buttons
		if (ti==root->get_children()) {
			//add function, this one uses menu

			if (p_button==1) {
				new_function_menu->clear();
				new_function_menu->set_size(Size2(0,0));
				int idx=0;

				virtuals_in_menu.clear();

				List<MethodInfo> mi;
				ObjectTypeDB::get_method_list(script->get_instance_base_type(),&mi);
				for (List<MethodInfo>::Element *E=mi.front();E;E=E->next()) {
					MethodInfo mi=E->get();
					if (mi.flags&METHOD_FLAG_VIRTUAL) {

						VirtualInMenu vim;
						vim.name=mi.name;
						vim.ret=mi.return_val.type;
						if (mi.return_val.name!=String())
							vim.ret_variant=true;
						else
							vim.ret_variant=false;

						String desc;

						if (mi.return_val.type==Variant::NIL)
							desc="var";
						else
							desc=Variant::get_type_name(mi.return_val.type);
						desc+=" "+mi.name+" ( ";


						for(int i=0;i<mi.arguments.size();i++) {

							if (i>0)
								desc+=", ";

							if (mi.arguments[i].type==Variant::NIL)
								desc+="var ";
							else
								desc+=Variant::get_type_name(mi.arguments[i].type)+" ";

							desc+=mi.arguments[i].name;

							Pair<Variant::Type,String> p;
							p.first=mi.arguments[i].type;
							p.second=mi.arguments[i].name;
							vim.args.push_back( p );

						}

						desc+=" )";

						virtuals_in_menu[idx]=vim;

						new_function_menu->add_item(desc,idx);
						idx++;
					}
				}

				Rect2 pos = members->get_item_rect(ti);
				new_function_menu->set_pos(members->get_global_pos()+pos.pos+Vector2(0,pos.size.y));
				new_function_menu->popup();
				return;
			} else if (p_button==0) {


				String name = _validate_name("new_function");
				selected=name;
				edited_func=selected;

				Ref<VisualScriptFunction> func_node;
				func_node.instance();
				func_node->set_name(name);

				undo_redo->create_action(TTR("Add Function"));
				undo_redo->add_do_method(script.ptr(),"add_function",name);
				undo_redo->add_do_method(script.ptr(),"add_node",name,script->get_available_id(),func_node);
				undo_redo->add_undo_method(script.ptr(),"remove_function",name);
				undo_redo->add_do_method(this,"_update_members");
				undo_redo->add_undo_method(this,"_update_members");
				undo_redo->commit_action();

				_update_graph();
			}

			return; //or crash because it will become invalid

		}

		if (ti==root->get_children()->get_next()) {
			//add variable
			String name = _validate_name("new_variable");
			selected=name;

			undo_redo->create_action(TTR("Add Variable"));
			undo_redo->add_do_method(script.ptr(),"add_variable",name);
			undo_redo->add_undo_method(script.ptr(),"remove_variable",name);
			undo_redo->add_do_method(this,"_update_members");
			undo_redo->add_undo_method(this,"_update_members");
			undo_redo->commit_action();
			return; //or crash because it will become invalid

		}

		if (ti==root->get_children()->get_next()->get_next()) {
			//add variable
			String name = _validate_name("new_signal");
			selected=name;

			undo_redo->create_action(TTR("Add Signal"));
			undo_redo->add_do_method(script.ptr(),"add_custom_signal",name);
			undo_redo->add_undo_method(script.ptr(),"remove_custom_signal",name);
			undo_redo->add_do_method(this,"_update_members");
			undo_redo->add_undo_method(this,"_update_members");
			undo_redo->commit_action();
			return; //or crash because it will become invalid

		}

	} else {

		if (ti->get_parent()==root->get_children()) {
			//edit/remove function
			String name = ti->get_metadata(0);

			if (p_button==1) {
				//delete the function
				undo_redo->create_action(TTR("Remove Function"));
				undo_redo->add_do_method(script.ptr(),"remove_function",name);
				undo_redo->add_undo_method(script.ptr(),"add_function",name);
				List<int> nodes;
				script->get_node_list(name,&nodes);
				for (List<int>::Element *E=nodes.front();E;E=E->next()) {
					undo_redo->add_undo_method(script.ptr(),"add_node",name,E->get(),script->get_node(name,E->get()),script->get_node_pos(name,E->get()));
				}

				List<VisualScript::SequenceConnection> seq_connections;

				script->get_sequence_connection_list(name,&seq_connections);

				for (List<VisualScript::SequenceConnection>::Element *E=seq_connections.front();E;E=E->next()) {
					undo_redo->add_undo_method(script.ptr(),"sequence_connect",name,E->get().from_node,E->get().from_output,E->get().to_node);
				}

				List<VisualScript::DataConnection> data_connections;

				script->get_data_connection_list(name,&data_connections);

				for (List<VisualScript::DataConnection>::Element *E=data_connections.front();E;E=E->next()) {
					undo_redo->add_undo_method(script.ptr(),"data_connect",name,E->get().from_node,E->get().from_port,E->get().to_node,E->get().to_port);
				}

				//for(int i=0;i<script->function_get_argument_count(name);i++) {
				////	undo_redo->add_undo_method(script.ptr(),"function_add_argument",name,script->function_get_argument_name(name,i),script->function_get_argument_type(name,i));
				//}
				undo_redo->add_do_method(this,"_update_members");
				undo_redo->add_undo_method(this,"_update_members");
				undo_redo->commit_action();

			} else if (p_button==0) {

			}
			return; //or crash because it will become invalid

		}

		if (ti->get_parent()==root->get_children()->get_next()) {
			//edit/remove variable

			String name = ti->get_metadata(0);

			if (p_button==1) {


				undo_redo->create_action(TTR("Remove Variable"));
				undo_redo->add_do_method(script.ptr(),"remove_variable",name);
				undo_redo->add_undo_method(script.ptr(),"add_variable",name,script->get_variable_default_value(name));
				undo_redo->add_undo_method(script.ptr(),"set_variable_info",name,script->call("get_variable_info",name)); //return as dict
				undo_redo->add_do_method(this,"_update_members");
				undo_redo->add_undo_method(this,"_update_members");
				undo_redo->commit_action();
				return; //or crash because it will become invalid
			} else if (p_button==0) {

				variable_editor->edit(name);
				edit_variable_dialog->set_title(TTR("Editing Variable:")+" "+name);
				edit_variable_dialog->popup_centered_minsize(Size2(400,200)*EDSCALE);
			}

		}

		if (ti->get_parent()==root->get_children()->get_next()->get_next()) {
			//edit/remove variable
			String name = ti->get_metadata(0);

			if (p_button==1) {

				undo_redo->create_action(TTR("Remove Signal"));
				undo_redo->add_do_method(script.ptr(),"remove_custom_signal",name);
				undo_redo->add_undo_method(script.ptr(),"add_custom_signal",name);

				for(int i=0;i<script->custom_signal_get_argument_count(name);i++) {
					undo_redo->add_undo_method(script.ptr(),"custom_signal_add_argument",name,script->custom_signal_get_argument_name(name,i),script->custom_signal_get_argument_type(name,i));
				}

				undo_redo->add_do_method(this,"_update_members");
				undo_redo->add_undo_method(this,"_update_members");
				undo_redo->commit_action();
			} else if (p_button==0) {

				signal_editor->edit(name);
				edit_signal_dialog->set_title(TTR("Editing Signal:")+" "+name);
				edit_signal_dialog->popup_centered_minsize(Size2(400,300)*EDSCALE);
			}

			return; //or crash because it will become invalid

		}


	}
}

void VisualScriptEditor::_available_node_doubleclicked() {

	TreeItem *item = nodes->get_selected();

	if (!item)
		return;

	String which = item->get_metadata(0);
	if (which==String())
		return;

	Vector2 ofs = graph->get_scroll_ofs() + graph->get_size() * 0.5;
	ofs/=EDSCALE;

	while(true) {
		bool exists=false;
		List<int> existing;
		script->get_node_list(edited_func,&existing);
		for (List<int>::Element *E=existing.front();E;E=E->next()) {
			Point2 pos = script->get_node_pos(edited_func,E->get());
			if (pos.distance_to(ofs)<15) {
				ofs+=Vector2(25,25);
				exists=true;
				break;
			}
		}

		if (exists)
			continue;
		break;

	}


	Ref<VisualScriptNode> vnode = VisualScriptLanguage::singleton->create_node_from_name(which);
	int new_id = script->get_available_id();

	undo_redo->create_action(TTR("Add Node"));
	undo_redo->add_do_method(script.ptr(),"add_node",edited_func,new_id,vnode,ofs);
	undo_redo->add_undo_method(script.ptr(),"remove_node",edited_func,new_id);
	undo_redo->add_do_method(this,"_update_graph");
	undo_redo->add_undo_method(this,"_update_graph");
	undo_redo->commit_action();

	Node* node = graph->get_node(itos(new_id));
	if (node) {
		graph->set_selected(node);
		_node_selected(node);
	}

}

void VisualScriptEditor::_update_available_nodes() {

	nodes->clear();

	TreeItem *root = nodes->create_item();

	Map<String,TreeItem*> path_cache;

	List<String> fnodes;
	VisualScriptLanguage::singleton->get_registered_node_names(&fnodes);

	for (List<String>::Element *E=fnodes.front();E;E=E->next()) {

		Vector<String> path = E->get().split("/");
		String sp;
		TreeItem* parent=root;

		for(int i=0;i<path.size()-1;i++) {

			if (i>0)
				sp+=",";
			sp+=path[i];
			if (!path_cache.has(sp)) {
				TreeItem* pathn = nodes->create_item(parent);
				pathn->set_selectable(0,false);
				pathn->set_text(0,path[i].capitalize());
				path_cache[sp]=pathn;
				parent=pathn;
				pathn->set_collapsed(true); //should remember state
			} else {
				parent=path_cache[sp];
			}
		}

		TreeItem *item = nodes->create_item(parent);
		item->set_text(0,path[path.size()-1].capitalize());
		item->set_selectable(0,true);
		item->set_metadata(0,E->get());
	}

}

String VisualScriptEditor::_validate_name(const String& p_name) const {

	String valid=p_name;

	int counter=1;
	while(true) {

		bool exists = script->has_function(valid) || script->has_variable(valid) || script->has_custom_signal(valid);

		if (exists) {
			counter++;
			valid=p_name+"_"+itos(counter);
			continue;
		}

		break;
	}

	return valid;
}

void VisualScriptEditor::_on_nodes_delete() {


	List<int> to_erase;

	for(int i=0;i<graph->get_child_count();i++) {
		GraphNode *gn = graph->get_child(i)->cast_to<GraphNode>();
		if (gn) {
			if (gn->is_selected() && gn->is_close_button_visible()) {
				to_erase.push_back(gn->get_name().operator String().to_int());
			}
		}
	}

	if (to_erase.empty())
		return;

	undo_redo->create_action("Remove VisualScript Nodes");

	for(List<int>::Element*F=to_erase.front();F;F=F->next()) {


		undo_redo->add_do_method(script.ptr(),"remove_node",edited_func,F->get());
		undo_redo->add_undo_method(script.ptr(),"add_node",edited_func,F->get(),script->get_node(edited_func,F->get()),script->get_node_pos(edited_func,F->get()));


		List<VisualScript::SequenceConnection> sequence_conns;
		script->get_sequence_connection_list(edited_func,&sequence_conns);


		for (List<VisualScript::SequenceConnection>::Element *E=sequence_conns.front();E;E=E->next()) {

			if (E->get().from_node==F->get() || E->get().to_node==F->get()) {
				undo_redo->add_undo_method(script.ptr(),"sequence_connect",edited_func,E->get().from_node,E->get().from_output,E->get().to_node);
			}
		}

		List<VisualScript::DataConnection> data_conns;
		script->get_data_connection_list(edited_func,&data_conns);

		for (List<VisualScript::DataConnection>::Element *E=data_conns.front();E;E=E->next()) {

			if (E->get().from_node==F->get() || E->get().to_node==F->get()) {
				undo_redo->add_undo_method(script.ptr(),"data_connect",edited_func,E->get().from_node,E->get().from_port,E->get().to_node,E->get().to_port);
			}
		}

	}
	undo_redo->add_do_method(this,"_update_graph");
	undo_redo->add_undo_method(this,"_update_graph");

	undo_redo->commit_action();
}


void VisualScriptEditor::_on_nodes_duplicate() {


	List<int> to_duplicate;

	for(int i=0;i<graph->get_child_count();i++) {
		GraphNode *gn = graph->get_child(i)->cast_to<GraphNode>();
		if (gn) {
			if (gn->is_selected() && gn->is_close_button_visible()) {
				to_duplicate.push_back(gn->get_name().operator String().to_int());
			}
		}
	}

	if (to_duplicate.empty())
		return;

	undo_redo->create_action("Duplicate VisualScript Nodes");
	int idc=script->get_available_id()+1;

	Set<int> to_select;

	for(List<int>::Element*F=to_duplicate.front();F;F=F->next()) {

		Ref<VisualScriptNode> node = script->get_node(edited_func,F->get());

		Ref<VisualScriptNode> dupe = node->duplicate();

		int new_id = idc++;
		to_select.insert(new_id);
		undo_redo->add_do_method(script.ptr(),"add_node",edited_func,new_id,dupe,script->get_node_pos(edited_func,F->get())+Vector2(20,20));
		undo_redo->add_undo_method(script.ptr(),"remove_node",edited_func,new_id);

	}
	undo_redo->add_do_method(this,"_update_graph");
	undo_redo->add_undo_method(this,"_update_graph");

	undo_redo->commit_action();

	for(int i=0;i<graph->get_child_count();i++) {
		GraphNode *gn = graph->get_child(i)->cast_to<GraphNode>();
		if (gn) {
			int id = gn->get_name().operator String().to_int();
			gn->set_selected(to_select.has(id));

		}
	}

	if (to_select.size()) {
		EditorNode::get_singleton()->push_item(script->get_node(edited_func,to_select.front()->get()).ptr());
	}

}

void VisualScriptEditor::_input(const InputEvent& p_event) {

	if (p_event.type==InputEvent::MOUSE_BUTTON && !p_event.mouse_button.pressed && p_event.mouse_button.button_index==BUTTON_LEFT) {
		revert_on_drag=String(); //so we can still drag functions
	}
}

Variant VisualScriptEditor::get_drag_data_fw(const Point2& p_point,Control* p_from) {


	if (p_from==nodes) {

		TreeItem *it = nodes->get_item_at_pos(p_point);
		if (!it)
			return Variant();
		String type=it->get_metadata(0);
		if (type==String())
			return Variant();

		Dictionary dd;
		dd["type"]="visual_script_node_drag";
		dd["node_type"]=type;

		Label *label = memnew(Label);
		label->set_text(it->get_text(0));
		set_drag_preview(label);
		return dd;
	}

	if (p_from==members) {


		TreeItem *it = members->get_item_at_pos(p_point);
		if (!it)
			return Variant();

		String type=it->get_metadata(0);

		if (type==String())
			return Variant();


		Dictionary dd;
		TreeItem *root=members->get_root();

		if (it->get_parent()==root->get_children()) {

			dd["type"]="visual_script_function_drag";
			dd["function"]=type;
			if (revert_on_drag!=String()) {
				edited_func=revert_on_drag; //revert so function does not change
				revert_on_drag=String();
				_update_graph();
			}
		} else if (it->get_parent()==root->get_children()->get_next()) {

			dd["type"]="visual_script_variable_drag";
			dd["variable"]=type;
		} else if (it->get_parent()==root->get_children()->get_next()->get_next()) {

			dd["type"]="visual_script_signal_drag";
			dd["signal"]=type;

		} else {
			return Variant();
		}






		Label *label = memnew(Label);
		label->set_text(it->get_text(0));
		set_drag_preview(label);
		return dd;
	}
	return Variant();
}

bool VisualScriptEditor::can_drop_data_fw(const Point2& p_point,const Variant& p_data,Control* p_from) const{

	if (p_from==graph) {

		Dictionary d = p_data;
		if (d.has("type") &&
				(
					String(d["type"])=="visual_script_node_drag" ||
					String(d["type"])=="visual_script_function_drag" ||
					String(d["type"])=="visual_script_variable_drag" ||
					String(d["type"])=="visual_script_signal_drag"
				) )
				return true;
	}


	return false;
}
void VisualScriptEditor::drop_data_fw(const Point2& p_point,const Variant& p_data,Control* p_from){

	if (p_from==graph) {

		Dictionary d = p_data;
		if (d.has("type") && String(d["type"])=="visual_script_node_drag") {

			Vector2 ofs = graph->get_scroll_ofs() + p_point;

			ofs/=EDSCALE;

			Ref<VisualScriptNode> vnode = VisualScriptLanguage::singleton->create_node_from_name(d["node_type"]);
			int new_id = script->get_available_id();

			undo_redo->create_action(TTR("Add Node"));
			undo_redo->add_do_method(script.ptr(),"add_node",edited_func,new_id,vnode,ofs);
			undo_redo->add_undo_method(script.ptr(),"remove_node",edited_func,new_id);
			undo_redo->add_do_method(this,"_update_graph");
			undo_redo->add_undo_method(this,"_update_graph");
			undo_redo->commit_action();

			Node* node = graph->get_node(itos(new_id));
			if (node) {
				graph->set_selected(node);
				_node_selected(node);
			}
		}

		if (d.has("type") && String(d["type"])=="visual_script_variable_drag") {

			Vector2 ofs = graph->get_scroll_ofs() + p_point;

			ofs/=EDSCALE;

			Ref<VisualScriptVariable> vnode;
			vnode.instance();
			vnode->set_variable(d["variable"]);

			int new_id = script->get_available_id();

			undo_redo->create_action(TTR("Add Node"));
			undo_redo->add_do_method(script.ptr(),"add_node",edited_func,new_id,vnode,ofs);
			undo_redo->add_undo_method(script.ptr(),"remove_node",edited_func,new_id);
			undo_redo->add_do_method(this,"_update_graph");
			undo_redo->add_undo_method(this,"_update_graph");
			undo_redo->commit_action();

			Node* node = graph->get_node(itos(new_id));
			if (node) {
				graph->set_selected(node);
				_node_selected(node);
			}
		}

		if (d.has("type") && String(d["type"])=="visual_script_function_drag") {

			Vector2 ofs = graph->get_scroll_ofs() + p_point;

			ofs/=EDSCALE;

			Ref<VisualScriptScriptCall> vnode;
			vnode.instance();
			vnode->set_call_mode(VisualScriptScriptCall::CALL_MODE_SELF);
			vnode->set_function(d["function"]);

			int new_id = script->get_available_id();

			undo_redo->create_action(TTR("Add Node"));
			undo_redo->add_do_method(script.ptr(),"add_node",edited_func,new_id,vnode,ofs);
			undo_redo->add_undo_method(script.ptr(),"remove_node",edited_func,new_id);
			undo_redo->add_do_method(this,"_update_graph");
			undo_redo->add_undo_method(this,"_update_graph");
			undo_redo->commit_action();

			Node* node = graph->get_node(itos(new_id));
			if (node) {
				graph->set_selected(node);
				_node_selected(node);
			}
		}


		if (d.has("type") && String(d["type"])=="visual_script_signal_drag") {

			Vector2 ofs = graph->get_scroll_ofs() + p_point;

			ofs/=EDSCALE;

			Ref<VisualScriptEmitSignal> vnode;
			vnode.instance();
			vnode->set_signal(d["signal"]);

			int new_id = script->get_available_id();

			undo_redo->create_action(TTR("Add Node"));
			undo_redo->add_do_method(script.ptr(),"add_node",edited_func,new_id,vnode,ofs);
			undo_redo->add_undo_method(script.ptr(),"remove_node",edited_func,new_id);
			undo_redo->add_do_method(this,"_update_graph");
			undo_redo->add_undo_method(this,"_update_graph");
			undo_redo->commit_action();

			Node* node = graph->get_node(itos(new_id));
			if (node) {
				graph->set_selected(node);
				_node_selected(node);
			}
		}
	}


}


/////////////////////////



void VisualScriptEditor::apply_code() {


}

Ref<Script> VisualScriptEditor::get_edited_script() const{

	return script;
}

Vector<String> VisualScriptEditor::get_functions(){

	return Vector<String>();
}

void VisualScriptEditor::set_edited_script(const Ref<Script>& p_script){

	script=p_script;
	signal_editor->script=p_script;
	signal_editor->undo_redo=undo_redo;
	variable_editor->script=p_script;
	variable_editor->undo_redo=undo_redo;


	script->connect("node_ports_changed",this,"_node_ports_changed");

	_update_members();
	_update_available_nodes();
}

void VisualScriptEditor::reload_text(){


}

String VisualScriptEditor::get_name(){

	String name;

	if (script->get_path().find("local://")==-1 && script->get_path().find("::")==-1) {
		name=script->get_path().get_file();
		if (is_unsaved()) {
			name+="(*)";
		}
	} else if (script->get_name()!="")
		name=script->get_name();
	else
		name=script->get_type()+"("+itos(script->get_instance_ID())+")";

	return name;

}

Ref<Texture> VisualScriptEditor::get_icon(){

	return Control::get_icon("VisualScript","EditorIcons");
}

bool VisualScriptEditor::is_unsaved(){

	return false;
}

Variant VisualScriptEditor::get_edit_state(){

	return Variant();
}

void VisualScriptEditor::set_edit_state(const Variant& p_state){


}

void VisualScriptEditor::goto_line(int p_line){


}

void VisualScriptEditor::trim_trailing_whitespace(){


}

void VisualScriptEditor::ensure_focus(){


}

void VisualScriptEditor::tag_saved_version(){


}

void VisualScriptEditor::reload(bool p_soft){


}

void VisualScriptEditor::get_breakpoints(List<int> *p_breakpoints){


}

bool VisualScriptEditor::goto_method(const String& p_method){

	return false;
}

void VisualScriptEditor::add_callback(const String& p_function,StringArray p_args){


}

void VisualScriptEditor::update_settings(){


}


void VisualScriptEditor::set_tooltip_request_func(String p_method,Object* p_obj){


}

Control *VisualScriptEditor::get_edit_menu(){

	return NULL;
}

void VisualScriptEditor::_change_base_type() {

	select_base_type->popup(true);
}

void VisualScriptEditor::_change_base_type_callback() {

	String bt = select_base_type->get_selected_type();

	ERR_FAIL_COND(bt==String());
	undo_redo->create_action("Change Base Type");
	undo_redo->add_do_method(script.ptr(),"set_instance_base_type",bt);
	undo_redo->add_undo_method(script.ptr(),"set_instance_base_type",script->get_instance_base_type());
	undo_redo->add_do_method(this,"_update_members");
	undo_redo->add_undo_method(this,"_update_members");
	undo_redo->commit_action();

}

void VisualScriptEditor::_node_selected(Node* p_node) {

	Ref<VisualScriptNode> vnode = p_node->get_meta("__vnode");
	if (vnode.is_null())
		return;

	EditorNode::get_singleton()->push_item(vnode.ptr());	//edit node in inspector
}

static bool _get_out_slot(const Ref<VisualScriptNode>& p_node,int p_slot,int& r_real_slot,bool& r_sequence) {

	if (p_slot<p_node->get_output_sequence_port_count()) {
		r_sequence=true;
		r_real_slot=p_slot;

		return true;
	}

	r_real_slot=p_slot-p_node->get_output_sequence_port_count();
	r_sequence=false;

	return (r_real_slot<p_node->get_output_value_port_count());

}

static bool _get_in_slot(const Ref<VisualScriptNode>& p_node,int p_slot,int& r_real_slot,bool& r_sequence) {

	if (p_slot==0 && p_node->has_input_sequence_port()) {
		r_sequence=true;
		r_real_slot=0;
		return true;
	}


	r_real_slot=p_slot-(p_node->has_input_sequence_port()?1:0);
	r_sequence=false;

	return r_real_slot<p_node->get_input_value_port_count();

}


void VisualScriptEditor::_begin_node_move() {

	undo_redo->create_action("Move Node(s)");
}

void VisualScriptEditor::_end_node_move() {

	undo_redo->commit_action();
}

void VisualScriptEditor::_move_node(String func,int p_id,const Vector2& p_to) {



	if (func==String(edited_func)) {
		Node* node = graph->get_node(itos(p_id));
		if (node && node->cast_to<GraphNode>())
			node->cast_to<GraphNode>()->set_offset(p_to);
	}
	script->set_node_pos(edited_func,p_id,p_to/EDSCALE);
}

void VisualScriptEditor::_node_moved(Vector2 p_from,Vector2 p_to, int p_id) {

	undo_redo->add_do_method(this,"_move_node",String(edited_func),p_id,p_to);
	undo_redo->add_undo_method(this,"_move_node",String(edited_func),p_id,p_from);
}

void VisualScriptEditor::_remove_node(int p_id) {


	undo_redo->create_action("Remove VisualScript Node");

	undo_redo->add_do_method(script.ptr(),"remove_node",edited_func,p_id);
	undo_redo->add_undo_method(script.ptr(),"add_node",edited_func,p_id,script->get_node(edited_func,p_id),script->get_node_pos(edited_func,p_id));


	List<VisualScript::SequenceConnection> sequence_conns;
	script->get_sequence_connection_list(edited_func,&sequence_conns);


	for (List<VisualScript::SequenceConnection>::Element *E=sequence_conns.front();E;E=E->next()) {

		if (E->get().from_node==p_id || E->get().to_node==p_id) {
			undo_redo->add_undo_method(script.ptr(),"sequence_connect",edited_func,E->get().from_node,E->get().from_output,E->get().to_node);
		}
	}

	List<VisualScript::DataConnection> data_conns;
	script->get_data_connection_list(edited_func,&data_conns);

	for (List<VisualScript::DataConnection>::Element *E=data_conns.front();E;E=E->next()) {

		if (E->get().from_node==p_id || E->get().to_node==p_id) {
			undo_redo->add_undo_method(script.ptr(),"data_connect",edited_func,E->get().from_node,E->get().from_port,E->get().to_node,E->get().to_port);
		}
	}

	undo_redo->add_do_method(this,"_update_graph");
	undo_redo->add_undo_method(this,"_update_graph");

	undo_redo->commit_action();
}


void VisualScriptEditor::_node_ports_changed(const String& p_func,int p_id) {

	if (p_func!=String(edited_func))
		return;

	_update_graph(p_id);
}

void VisualScriptEditor::_graph_connected(const String& p_from,int p_from_slot,const String& p_to,int p_to_slot) {

	Ref<VisualScriptNode> from_node = script->get_node(edited_func,p_from.to_int());
	ERR_FAIL_COND(!from_node.is_valid());

	bool from_seq;
	int from_port;

	if (!_get_out_slot(from_node,p_from_slot,from_port,from_seq))
		return; //can't  connect this, it' s invalid

	Ref<VisualScriptNode> to_node = script->get_node(edited_func,p_to.to_int());
	ERR_FAIL_COND(!to_node.is_valid());

	bool to_seq;
	int to_port;

	if (!_get_in_slot(to_node,p_to_slot,to_port,to_seq))
		return; //can't  connect this, it' s invalid


	ERR_FAIL_COND(from_seq!=to_seq);


	undo_redo->create_action("Connect Nodes");

	if (from_seq) {
		undo_redo->add_do_method(script.ptr(),"sequence_connect",edited_func,p_from.to_int(),from_port,p_to.to_int());
		undo_redo->add_undo_method(script.ptr(),"sequence_disconnect",edited_func,p_from.to_int(),from_port,p_to.to_int());
	} else {
		undo_redo->add_do_method(script.ptr(),"data_connect",edited_func,p_from.to_int(),from_port,p_to.to_int(),to_port);
		undo_redo->add_undo_method(script.ptr(),"data_disconnect",edited_func,p_from.to_int(),from_port,p_to.to_int(),to_port);
	}

	undo_redo->add_do_method(this,"_update_graph_connections");
	undo_redo->add_undo_method(this,"_update_graph_connections");

	undo_redo->commit_action();

}

void VisualScriptEditor::_graph_disconnected(const String& p_from,int p_from_slot,const String& p_to,int p_to_slot){

	Ref<VisualScriptNode> from_node = script->get_node(edited_func,p_from.to_int());
	ERR_FAIL_COND(!from_node.is_valid());

	bool from_seq;
	int from_port;

	if (!_get_out_slot(from_node,p_from_slot,from_port,from_seq))
		return; //can't  connect this, it' s invalid

	Ref<VisualScriptNode> to_node = script->get_node(edited_func,p_to.to_int());
	ERR_FAIL_COND(!to_node.is_valid());

	bool to_seq;
	int to_port;

	if (!_get_in_slot(to_node,p_to_slot,to_port,to_seq))
		return; //can't  connect this, it' s invalid


	ERR_FAIL_COND(from_seq!=to_seq);


	undo_redo->create_action("Connect Nodes");

	if (from_seq) {
		undo_redo->add_do_method(script.ptr(),"sequence_disconnect",edited_func,p_from.to_int(),from_port,p_to.to_int());
		undo_redo->add_undo_method(script.ptr(),"sequence_connect",edited_func,p_from.to_int(),from_port,p_to.to_int());
	} else {
		undo_redo->add_do_method(script.ptr(),"data_disconnect",edited_func,p_from.to_int(),from_port,p_to.to_int(),to_port);
		undo_redo->add_undo_method(script.ptr(),"data_connect",edited_func,p_from.to_int(),from_port,p_to.to_int(),to_port);
	}

	undo_redo->add_do_method(this,"_update_graph_connections");
	undo_redo->add_undo_method(this,"_update_graph_connections");

	undo_redo->commit_action();
}


void VisualScriptEditor::_bind_methods() {

	ObjectTypeDB::bind_method("_member_button",&VisualScriptEditor::_member_button);
	ObjectTypeDB::bind_method("_member_edited",&VisualScriptEditor::_member_edited);
	ObjectTypeDB::bind_method("_member_selected",&VisualScriptEditor::_member_selected);
	ObjectTypeDB::bind_method("_update_members",&VisualScriptEditor::_update_members);
	ObjectTypeDB::bind_method("_change_base_type",&VisualScriptEditor::_change_base_type);
	ObjectTypeDB::bind_method("_change_base_type_callback",&VisualScriptEditor::_change_base_type_callback);
	ObjectTypeDB::bind_method("_override_pressed",&VisualScriptEditor::_override_pressed);
	ObjectTypeDB::bind_method("_node_selected",&VisualScriptEditor::_node_selected);
	ObjectTypeDB::bind_method("_node_moved",&VisualScriptEditor::_node_moved);
	ObjectTypeDB::bind_method("_move_node",&VisualScriptEditor::_move_node);
	ObjectTypeDB::bind_method("_begin_node_move",&VisualScriptEditor::_begin_node_move);
	ObjectTypeDB::bind_method("_end_node_move",&VisualScriptEditor::_end_node_move);
	ObjectTypeDB::bind_method("_remove_node",&VisualScriptEditor::_remove_node);
	ObjectTypeDB::bind_method("_update_graph",&VisualScriptEditor::_update_graph,DEFVAL(-1));
	ObjectTypeDB::bind_method("_node_ports_changed",&VisualScriptEditor::_node_ports_changed);
	ObjectTypeDB::bind_method("_available_node_doubleclicked",&VisualScriptEditor::_available_node_doubleclicked);

	ObjectTypeDB::bind_method("get_drag_data_fw",&VisualScriptEditor::get_drag_data_fw);
	ObjectTypeDB::bind_method("can_drop_data_fw",&VisualScriptEditor::can_drop_data_fw);
	ObjectTypeDB::bind_method("drop_data_fw",&VisualScriptEditor::drop_data_fw);

	ObjectTypeDB::bind_method("_input",&VisualScriptEditor::_input);
	ObjectTypeDB::bind_method("_on_nodes_delete",&VisualScriptEditor::_on_nodes_delete);
	ObjectTypeDB::bind_method("_on_nodes_duplicate",&VisualScriptEditor::_on_nodes_duplicate);


	ObjectTypeDB::bind_method("_graph_connected",&VisualScriptEditor::_graph_connected);
	ObjectTypeDB::bind_method("_graph_disconnected",&VisualScriptEditor::_graph_disconnected);
	ObjectTypeDB::bind_method("_update_graph_connections",&VisualScriptEditor::_update_graph_connections);


}



VisualScriptEditor::VisualScriptEditor() {

	main_hsplit = memnew( HSplitContainer );
	add_child(main_hsplit);
	main_hsplit->set_area_as_parent_rect();

	left_vsplit = memnew( VSplitContainer );
	main_hsplit->add_child(left_vsplit);

	VBoxContainer *left_vb = memnew( VBoxContainer );
	left_vsplit->add_child(left_vb);
	left_vb->set_v_size_flags(SIZE_EXPAND_FILL);
	left_vb->set_custom_minimum_size(Size2(180,1)*EDSCALE);

	base_type_select = memnew( Button );	
	left_vb->add_margin_child(TTR("Base Type:"),base_type_select);
	base_type_select->connect("pressed",this,"_change_base_type");

	members = memnew( Tree );
	left_vb->add_margin_child(TTR("Members:"),members,true);
	members->set_hide_root(true);
	members->connect("button_pressed",this,"_member_button");
	members->connect("item_edited",this,"_member_edited");
	members->connect("cell_selected",this,"_member_selected");
	members->set_single_select_cell_editing_only_when_already_selected(true);
	members->set_hide_folding(true);
	members->set_drag_forwarding(this);


	VBoxContainer *left_vb2 = memnew( VBoxContainer );
	left_vsplit->add_child(left_vb2);
	left_vb2->set_v_size_flags(SIZE_EXPAND_FILL);

	nodes = memnew( Tree );
	left_vb2->add_margin_child(TTR("Available Nodes:"),nodes,true);
	nodes->set_hide_root(true);
	nodes->connect("item_activated",this,"_available_node_doubleclicked");
	nodes->set_drag_forwarding(this);

	graph = memnew( GraphEdit );
	main_hsplit->add_child(graph);
	graph->set_h_size_flags(SIZE_EXPAND_FILL);
	graph->connect("node_selected",this,"_node_selected");
	graph->connect("_begin_node_move",this,"_begin_node_move");
	graph->connect("_end_node_move",this,"_end_node_move");
	graph->connect("delete_nodes_request",this,"_on_nodes_delete");
	graph->connect("duplicate_nodes_request",this,"_on_nodes_duplicate");
	graph->set_drag_forwarding(this);



	//allowed casts (connections)
	for(int i=0;i<Variant::VARIANT_MAX;i++) {
		graph->add_valid_connection_type(Variant::NIL,i);
		graph->add_valid_connection_type(i,Variant::NIL);
		for(int j=0;j<Variant::VARIANT_MAX;j++) {
			if (Variant::can_convert(Variant::Type(i),Variant::Type(j))) {
				graph->add_valid_connection_type(i,j);
			}
		}

		graph->add_valid_right_disconnect_type(i);
	}

	graph->add_valid_left_disconnect_type(TYPE_SEQUENCE);

	graph->connect("connection_request",this,"_graph_connected");
	graph->connect("disconnection_request",this,"_graph_disconnected");

	edit_signal_dialog = memnew( AcceptDialog );
	edit_signal_dialog->get_ok()->set_text(TTR("Close"));
	add_child(edit_signal_dialog);
	edit_signal_dialog->set_title(TTR("Edit Signal Arguments:"));

	signal_editor = memnew( VisualScriptEditorSignalEdit );
	edit_signal_edit = memnew( PropertyEditor );
	edit_signal_edit->hide_top_label();
	edit_signal_dialog->add_child(edit_signal_edit);
	edit_signal_dialog->set_child_rect(edit_signal_edit);
	edit_signal_edit->edit(signal_editor);

	edit_variable_dialog = memnew( AcceptDialog );
	edit_variable_dialog->get_ok()->set_text(TTR("Close"));
	add_child(edit_variable_dialog);
	edit_variable_dialog->set_title(TTR("Edit Variable:"));

	variable_editor = memnew( VisualScriptEditorVariableEdit );
	edit_variable_edit = memnew( PropertyEditor );
	edit_variable_edit->hide_top_label();
	edit_variable_dialog->add_child(edit_variable_edit);
	edit_variable_dialog->set_child_rect(edit_variable_edit);
	edit_variable_edit->edit(variable_editor);

	select_base_type=memnew(CreateDialog);
	select_base_type->set_base_type("Object"); //anything goes
	select_base_type->connect("create",this,"_change_base_type_callback");
	select_base_type->get_ok()->set_text(TTR("Change"));
	add_child(select_base_type);

	undo_redo = EditorNode::get_singleton()->get_undo_redo();

	new_function_menu = memnew( PopupMenu );
	new_function_menu->connect("item_pressed",this,"_override_pressed");
	add_child(new_function_menu);
	updating_members=false;

	set_process_input(true); //for revert on drag
	set_process_unhandled_input(true); //for revert on drag
}

VisualScriptEditor::~VisualScriptEditor() {

	undo_redo->clear_history(); //avoid crashes
	memdelete(signal_editor);
	memdelete(variable_editor);
}

static ScriptEditorBase * create_editor(const Ref<Script>& p_script) {

	if (p_script->cast_to<VisualScript>()) {
		return memnew( VisualScriptEditor );
	}

	return NULL;
}


void VisualScriptEditor::register_editor() {

	ScriptEditor::register_create_script_editor_function(create_editor);
}

