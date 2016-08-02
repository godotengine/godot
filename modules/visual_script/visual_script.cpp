#include "visual_script.h"
#include "visual_script_nodes.h"

void VisualScriptNode::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("get_visual_script:VisualScript"),&VisualScriptNode::get_visual_script);
	ADD_SIGNAL(MethodInfo("ports_changed"));
}


Ref<VisualScript> VisualScriptNode::get_visual_script() const {

	if (scripts_used.size())
		return Ref<VisualScript>(scripts_used.front()->get());

	return Ref<VisualScript>();

}

////////////////

/////////////////////

VisualScriptNodeInstance::~VisualScriptNodeInstance() {

}

void VisualScript::add_function(const StringName& p_name) {

	ERR_FAIL_COND(!String(p_name).is_valid_identifier());
	ERR_FAIL_COND(functions.has(p_name));

	functions[p_name]=Function();
}

bool VisualScript::has_function(const StringName& p_name) const {

	return functions.has(p_name);

}
void VisualScript::remove_function(const StringName& p_name) {

	ERR_FAIL_COND(!functions.has(p_name));

	for (Map<int,Function::NodeData>::Element *E=functions[p_name].nodes.front();E;E=E->next()) {

		E->get().node->disconnect("ports_changed",this,"_node_ports_changed");
		E->get().node->scripts_used.erase(this);
	}

	functions.erase(p_name);

}

void VisualScript::rename_function(const StringName& p_name,const StringName& p_new_name) {

	ERR_FAIL_COND(!functions.has(p_name));
	if (p_new_name==p_name)
		return;

	ERR_FAIL_COND(!String(p_new_name).is_valid_identifier());

	ERR_FAIL_COND(functions.has(p_new_name));
	ERR_FAIL_COND(variables.has(p_new_name));
	ERR_FAIL_COND(custom_signals.has(p_new_name));

	functions[p_new_name]=functions[p_name];
	functions.erase(p_name);

}

void VisualScript::get_function_list(List<StringName> *r_functions) const {

	for (const Map<StringName,Function>::Element *E=functions.front();E;E=E->next()) {
		r_functions->push_back(E->key());
	}

	r_functions->sort_custom<StringName::AlphCompare>();

}

int VisualScript::get_function_node_id(const StringName& p_name) const {

	ERR_FAIL_COND_V(!functions.has(p_name),-1);

	return functions[p_name].function_id;

}


void VisualScript::_node_ports_changed(int p_id) {


	StringName function;

	for (Map<StringName,Function>::Element *E=functions.front();E;E=E->next()) {

		if (E->get().nodes.has(p_id)) {
			function=E->key();
			break;
		}
	}

	ERR_FAIL_COND(function==StringName());

	Function &func = functions[function];
	Ref<VisualScriptNode> vsn = func.nodes[p_id].node;

	//must revalidate all the functions

	{
		List<SequenceConnection> to_remove;

		for (Set<SequenceConnection>::Element *E=func.sequence_connections.front();E;E=E->next()) {
			if (E->get().from_node==p_id && E->get().from_output>=vsn->get_output_sequence_port_count()) {

				to_remove.push_back(E->get());
			}
			if (E->get().to_node==p_id && !vsn->has_input_sequence_port()) {

				to_remove.push_back(E->get());
			}
		}

		while(to_remove.size()) {
			func.sequence_connections.erase(to_remove.front()->get());
			to_remove.pop_front();
		}
	}

	{

		List<DataConnection> to_remove;


		for (Set<DataConnection>::Element *E=func.data_connections.front();E;E=E->next()) {
			if (E->get().from_node==p_id && E->get().from_port>=vsn->get_output_value_port_count()) {
				to_remove.push_back(E->get());
			}
			if (E->get().to_node==p_id && E->get().to_port>=vsn->get_input_value_port_count()) {
				to_remove.push_back(E->get());
			}
		}

		while(to_remove.size()) {
			func.data_connections.erase(to_remove.front()->get());
			to_remove.pop_front();
		}
	}

	emit_signal("node_ports_changed",function,p_id);
}

void VisualScript::add_node(const StringName& p_func,int p_id, const Ref<VisualScriptNode>& p_node, const Point2 &p_pos) {

	ERR_FAIL_COND(!functions.has(p_func));


	for (Map<StringName,Function>::Element *E=functions.front();E;E=E->next()) {

		ERR_FAIL_COND(E->get().nodes.has(p_id)); //id can exist only one in script, even for different functions
	}

	Function &func = functions[p_func];


	if (p_node->cast_to<VisualScriptFunction>()) {
		//the function indeed
		ERR_EXPLAIN("A function node already has been set here.");
		ERR_FAIL_COND(func.function_id>=0);

		func.function_id=p_id;
	}

	Function::NodeData nd;
	nd.node=p_node;
	nd.pos=p_pos;

	Ref<VisualScriptNode> vsn = p_node;
	vsn->connect("ports_changed",this,"_node_ports_changed",varray(p_id));
	vsn->scripts_used.insert(this);



	func.nodes[p_id]=nd;
}

void VisualScript::remove_node(const StringName& p_func,int p_id){

	ERR_FAIL_COND(!functions.has(p_func));
	Function &func = functions[p_func];

	ERR_FAIL_COND(!func.nodes.has(p_id));
	{
		List<SequenceConnection> to_remove;

		for (Set<SequenceConnection>::Element *E=func.sequence_connections.front();E;E=E->next()) {
			if (E->get().from_node==p_id || E->get().to_node==p_id) {
				to_remove.push_back(E->get());
			}
		}

		while(to_remove.size()) {
			func.sequence_connections.erase(to_remove.front()->get());
			to_remove.pop_front();
		}
	}

	{

		List<DataConnection> to_remove;


		for (Set<DataConnection>::Element *E=func.data_connections.front();E;E=E->next()) {
			if (E->get().from_node==p_id || E->get().to_node==p_id) {
				to_remove.push_back(E->get());
			}
		}

		while(to_remove.size()) {
			func.data_connections.erase(to_remove.front()->get());
			to_remove.pop_front();
		}
	}

	if (func.nodes[p_id].node->cast_to<VisualScriptFunction>()) {
		func.function_id=-1; //revert to invalid
	}

	func.nodes[p_id].node->disconnect("ports_changed",this,"_node_ports_changed");
	func.nodes[p_id].node->scripts_used.erase(this);

	func.nodes.erase(p_id);


}



Ref<VisualScriptNode> VisualScript::get_node(const StringName& p_func,int p_id) const{

	ERR_FAIL_COND_V(!functions.has(p_func),Ref<VisualScriptNode>());
	const Function &func = functions[p_func];

	ERR_FAIL_COND_V(!func.nodes.has(p_id),Ref<VisualScriptNode>());

	return func.nodes[p_id].node;
}

void VisualScript::set_node_pos(const StringName& p_func,int p_id,const Point2& p_pos) {

	ERR_FAIL_COND(!functions.has(p_func));
	Function &func = functions[p_func];

	ERR_FAIL_COND(!func.nodes.has(p_id));
	func.nodes[p_id].pos=p_pos;
}

Point2 VisualScript::get_node_pos(const StringName& p_func,int p_id) const{

	ERR_FAIL_COND_V(!functions.has(p_func),Point2());
	const Function &func = functions[p_func];

	ERR_FAIL_COND_V(!func.nodes.has(p_id),Point2());
	return func.nodes[p_id].pos;
}


void VisualScript::get_node_list(const StringName& p_func,List<int> *r_nodes) const{

	ERR_FAIL_COND(!functions.has(p_func));
	const Function &func = functions[p_func];

	for (const Map<int,Function::NodeData>::Element *E=func.nodes.front();E;E=E->next()) {
		r_nodes->push_back(E->key());
	}

}


void VisualScript::sequence_connect(const StringName& p_func,int p_from_node,int p_from_output,int p_to_node){

	ERR_FAIL_COND(!functions.has(p_func));
	Function &func = functions[p_func];


	SequenceConnection sc;
	sc.from_node=p_from_node;
	sc.from_output=p_from_output;
	sc.to_node=p_to_node;
	ERR_FAIL_COND(func.sequence_connections.has(sc));

	func.sequence_connections.insert(sc);

}

void VisualScript::sequence_disconnect(const StringName& p_func,int p_from_node,int p_from_output,int p_to_node){

	ERR_FAIL_COND(!functions.has(p_func));
	Function &func = functions[p_func];

	SequenceConnection sc;
	sc.from_node=p_from_node;
	sc.from_output=p_from_output;
	sc.to_node=p_to_node;
	ERR_FAIL_COND(!func.sequence_connections.has(sc));

	func.sequence_connections.erase(sc);

}

bool VisualScript::has_sequence_connection(const StringName& p_func,int p_from_node,int p_from_output,int p_to_node) const{

	ERR_FAIL_COND_V(!functions.has(p_func),false);
	const Function &func = functions[p_func];

	SequenceConnection sc;
	sc.from_node=p_from_node;
	sc.from_output=p_from_output;
	sc.to_node=p_to_node;

	return func.sequence_connections.has(sc);
}

void VisualScript::get_sequence_connection_list(const StringName& p_func,List<SequenceConnection> *r_connection) const {

	ERR_FAIL_COND(!functions.has(p_func));
	const Function &func = functions[p_func];

	for (const Set<SequenceConnection>::Element *E=func.sequence_connections.front();E;E=E->next()) {
		r_connection->push_back(E->get());
	}
}


void VisualScript::data_connect(const StringName& p_func,int p_from_node,int p_from_port,int p_to_node,int p_to_port) {

	ERR_FAIL_COND(!functions.has(p_func));
	Function &func = functions[p_func];

	DataConnection dc;
	dc.from_node=p_from_node;
	dc.from_port=p_from_port;
	dc.to_node=p_to_node;
	dc.to_port=p_to_port;

	ERR_FAIL_COND( func.data_connections.has(dc));

	func.data_connections.insert(dc);
}

void VisualScript::data_disconnect(const StringName& p_func,int p_from_node,int p_from_port,int p_to_node,int p_to_port) {

	ERR_FAIL_COND(!functions.has(p_func));
	Function &func = functions[p_func];

	DataConnection dc;
	dc.from_node=p_from_node;
	dc.from_port=p_from_port;
	dc.to_node=p_to_node;
	dc.to_port=p_to_port;

	ERR_FAIL_COND( !func.data_connections.has(dc));

	func.data_connections.erase(dc);

}

bool VisualScript::has_data_connection(const StringName& p_func,int p_from_node,int p_from_port,int p_to_node,int p_to_port) const {

	ERR_FAIL_COND_V(!functions.has(p_func),false);
	const Function &func = functions[p_func];

	DataConnection dc;
	dc.from_node=p_from_node;
	dc.from_port=p_from_port;
	dc.to_node=p_to_node;
	dc.to_port=p_to_port;

	return func.data_connections.has(dc);

}

void VisualScript::get_data_connection_list(const StringName& p_func,List<DataConnection> *r_connection) const {

	ERR_FAIL_COND(!functions.has(p_func));
	const Function &func = functions[p_func];

	for (const Set<DataConnection>::Element *E=func.data_connections.front();E;E=E->next()) {
		r_connection->push_back(E->get());
	}
}

void VisualScript::add_variable(const StringName& p_name,const Variant& p_default_value) {

	ERR_FAIL_COND(!String(p_name).is_valid_identifier());
	ERR_FAIL_COND(variables.has(p_name));

	Variable v;
	v.default_value=p_default_value;
	v.info.type=p_default_value.get_type();
	v.info.name=p_name;
	v.info.hint=PROPERTY_HINT_NONE;

	variables[p_name]=v;

}

bool VisualScript::has_variable(const StringName& p_name) const {

	return variables.has(p_name);
}

void VisualScript::remove_variable(const StringName& p_name) {

	ERR_FAIL_COND(!variables.has(p_name));
	variables.erase(p_name);
}

void VisualScript::set_variable_default_value(const StringName& p_name,const Variant& p_value){

	ERR_FAIL_COND(!variables.has(p_name));

	variables[p_name].default_value=p_value;

}
Variant VisualScript::get_variable_default_value(const StringName& p_name) const{

	ERR_FAIL_COND_V(!variables.has(p_name),Variant());
	return variables[p_name].default_value;

}
void VisualScript::set_variable_info(const StringName& p_name,const PropertyInfo& p_info){

	ERR_FAIL_COND(!variables.has(p_name));
	variables[p_name].info=p_info;
	variables[p_name].info.name=p_name;


}
PropertyInfo VisualScript::get_variable_info(const StringName& p_name) const{

	ERR_FAIL_COND_V(!variables.has(p_name),PropertyInfo());
	return variables[p_name].info;
}

void VisualScript::_set_variable_info(const StringName& p_name,const Dictionary& p_info) {

	PropertyInfo pinfo;
	if (p_info.has("type"))
		pinfo.type=Variant::Type(int(p_info["type"]));
	if (p_info.has("name"))
		pinfo.name=p_info["name"];
	if (p_info.has("hint"))
		pinfo.hint=PropertyHint(int(p_info["hint"]));
	if (p_info.has("hint_string"))
		pinfo.hint_string=p_info["hint_string"];
	if (p_info.has("usage"))
		pinfo.usage=p_info["usage"];

	set_variable_info(p_name,pinfo);
}

Dictionary VisualScript::_get_variable_info(const StringName& p_name) const{

	PropertyInfo pinfo=get_variable_info(p_name);
	Dictionary d;
	d["type"]=pinfo.type;
	d["name"]=pinfo.name;
	d["hint"]=pinfo.hint;
	d["hint_string"]=pinfo.hint_string;
	d["usage"]=pinfo.usage;

	return d;
}

void VisualScript::get_variable_list(List<StringName> *r_variables){


	for (Map<StringName,Variable>::Element *E=variables.front();E;E=E->next()) {
		r_variables->push_back(E->key());
	}

	r_variables->sort_custom<StringName::AlphCompare>();
}


void VisualScript::set_instance_base_type(const StringName& p_type) {

	base_type=p_type;
}


void VisualScript::rename_variable(const StringName& p_name,const StringName& p_new_name) {

	ERR_FAIL_COND(!variables.has(p_name));
	if (p_new_name==p_name)
		return;

	ERR_FAIL_COND(!String(p_new_name).is_valid_identifier());

	ERR_FAIL_COND(functions.has(p_new_name));
	ERR_FAIL_COND(variables.has(p_new_name));
	ERR_FAIL_COND(custom_signals.has(p_new_name));

	variables[p_new_name]=variables[p_name];
	variables.erase(p_name);

}

void VisualScript::add_custom_signal(const StringName& p_name) {

	ERR_FAIL_COND(!String(p_name).is_valid_identifier());
	ERR_FAIL_COND(custom_signals.has(p_name));

	custom_signals[p_name]=Vector<Argument>();
}

bool VisualScript::has_custom_signal(const StringName& p_name) const {

	return custom_signals.has(p_name);

}
void VisualScript::custom_signal_add_argument(const StringName& p_func,Variant::Type p_type,const String& p_name,int p_index) {

	ERR_FAIL_COND(!custom_signals.has(p_func));
	Argument arg;
	arg.type=p_type;
	arg.name=p_name;
	if (p_index<0)
		custom_signals[p_func].push_back(arg);
	else
		custom_signals[p_func].insert(0,arg);

}
void VisualScript::custom_signal_set_argument_type(const StringName& p_func,int p_argidx,Variant::Type p_type) {

	ERR_FAIL_COND(!custom_signals.has(p_func));
	ERR_FAIL_INDEX(p_argidx,custom_signals[p_func].size());
	custom_signals[p_func][p_argidx].type=p_type;
}
Variant::Type VisualScript::custom_signal_get_argument_type(const StringName& p_func,int p_argidx) const  {

	ERR_FAIL_COND_V(!custom_signals.has(p_func),Variant::NIL);
	ERR_FAIL_INDEX_V(p_argidx,custom_signals[p_func].size(),Variant::NIL);
	return custom_signals[p_func][p_argidx].type;
}
void VisualScript::custom_signal_set_argument_name(const StringName& p_func,int p_argidx,const String& p_name) {
	ERR_FAIL_COND(!custom_signals.has(p_func));
	ERR_FAIL_INDEX(p_argidx,custom_signals[p_func].size());
	custom_signals[p_func][p_argidx].name=p_name;

}
String VisualScript::custom_signal_get_argument_name(const StringName& p_func,int p_argidx) const {

	ERR_FAIL_COND_V(!custom_signals.has(p_func),String());
	ERR_FAIL_INDEX_V(p_argidx,custom_signals[p_func].size(),String());
	return custom_signals[p_func][p_argidx].name;

}
void VisualScript::custom_signal_remove_argument(const StringName& p_func,int p_argidx) {

	ERR_FAIL_COND(!custom_signals.has(p_func));
	ERR_FAIL_INDEX(p_argidx,custom_signals[p_func].size());
	custom_signals[p_func].remove(p_argidx);

}

int VisualScript::custom_signal_get_argument_count(const StringName& p_func) const {

	ERR_FAIL_COND_V(!custom_signals.has(p_func),0);
	return custom_signals[p_func].size();

}
void VisualScript::custom_signal_swap_argument(const StringName& p_func,int p_argidx,int p_with_argidx) {

	ERR_FAIL_COND(!custom_signals.has(p_func));
	ERR_FAIL_INDEX(p_argidx,custom_signals[p_func].size());
	ERR_FAIL_INDEX(p_with_argidx,custom_signals[p_func].size());

	SWAP( custom_signals[p_func][p_argidx], custom_signals[p_func][p_with_argidx] );

}
void VisualScript::remove_custom_signal(const StringName& p_name) {

	ERR_FAIL_COND(!custom_signals.has(p_name));
	custom_signals.erase(p_name);

}

void VisualScript::rename_custom_signal(const StringName& p_name,const StringName& p_new_name) {

	ERR_FAIL_COND(!custom_signals.has(p_name));
	if (p_new_name==p_name)
		return;

	ERR_FAIL_COND(!String(p_new_name).is_valid_identifier());

	ERR_FAIL_COND(functions.has(p_new_name));
	ERR_FAIL_COND(variables.has(p_new_name));
	ERR_FAIL_COND(custom_signals.has(p_new_name));

	custom_signals[p_new_name]=custom_signals[p_name];
	custom_signals.erase(p_name);

}

void VisualScript::get_custom_signal_list(List<StringName> *r_custom_signals) const {

	for (const Map<StringName,Vector<Argument> >::Element *E=custom_signals.front();E;E=E->next()) {
		r_custom_signals->push_back(E->key());
	}

	r_custom_signals->sort_custom<StringName::AlphCompare>();

}

int VisualScript::get_available_id() const {

	int max_id=0;
	for (Map<StringName,Function>::Element *E=functions.front();E;E=E->next()) {
		if (E->get().nodes.empty())
			continue;

		int last_id = E->get().nodes.back()->key();
		max_id=MAX(max_id,last_id+1);
	}

	return max_id;
}

/////////////////////////////////


bool VisualScript::can_instance() const {

	return ScriptServer::is_scripting_enabled();

}


StringName VisualScript::get_instance_base_type() const {

	return base_type;
}

ScriptInstance* VisualScript::instance_create(Object *p_this) {

	return NULL;
}

bool VisualScript::instance_has(const Object *p_this) const {

	return false;
}

bool VisualScript::has_source_code() const {

	return false;
}

String VisualScript::get_source_code() const {

	return String();
}

void VisualScript::set_source_code(const String& p_code) {

}

Error VisualScript::reload(bool p_keep_state) {

	return OK;
}


bool VisualScript::is_tool() const {

	return false;
}


String VisualScript::get_node_type() const {

	return String();
}


ScriptLanguage *VisualScript::get_language() const {

	return VisualScriptLanguage::singleton;
}


bool VisualScript::has_script_signal(const StringName& p_signal) const {

	return false;
}

void VisualScript::get_script_signal_list(List<MethodInfo> *r_signals) const {

}


bool VisualScript::get_property_default_value(const StringName& p_property,Variant& r_value) const {

	return false;
}
void VisualScript::get_method_list(List<MethodInfo> *p_list) const {

	for (Map<StringName,Function>::Element *E=functions.front();E;E=E->next()) {

		MethodInfo mi;
		mi.name=E->key();
		if (E->get().function_id>=0) {

			Ref<VisualScriptFunction> func=E->get().nodes[E->get().function_id].node;
			if (func.is_valid()) {

				for(int i=0;i<func->get_argument_count();i++) {
					PropertyInfo arg;
					arg.name=func->get_argument_name(i);
					arg.type=func->get_argument_type(i);
					mi.arguments.push_back(arg);
				}
			}
		}

		p_list->push_back(mi);
	}
}

void VisualScript::_set_data(const Dictionary& p_data) {

	Dictionary d = p_data;
	if (d.has("base_type"))
		base_type=d["base_type"];

	variables.clear();
	Array vars=d["variables"];
	for (int i=0;i<vars.size();i++) {

		Dictionary v=vars[i];
		add_variable(v["name"],v["default_value"]);
		_set_variable_info(v["name"],v);
	}


	custom_signals.clear();
	Array sigs=d["signals"];
	for (int i=0;i<sigs.size();i++) {

		Dictionary cs=sigs[i];
		add_custom_signal(cs["name"]);

		Array args=cs["arguments"];
		for(int j=0;j<args.size();j+=2) {
			custom_signal_add_argument(cs["name"],Variant::Type(int(args[j+1])),args[j]);
		}
	}

	Array funcs=d["functions"];
	functions.clear();

	for (int i=0;i<funcs.size();i++) {

		Dictionary func=funcs[i];

		StringName name=func["name"];
		//int id=func["function_id"];
		add_function(name);

		Array nodes = func["nodes"];

		for(int i=0;i<nodes.size();i+=3) {

			add_node(name,nodes[i],nodes[i+2],nodes[i+1]);
		}


		Array sequence_connections=func["sequence_connections"];

		for (int j=0;j<sequence_connections.size();j+=3) {

			sequence_connect(name,sequence_connections[j+0],sequence_connections[j+1],sequence_connections[j+2]);
		}


		Array data_connections=func["data_connections"];

		for (int j=0;j<data_connections.size();j+=4) {

			data_connect(name,data_connections[j+0],data_connections[j+1],data_connections[j+2],data_connections[j+3]);

		}


	}

}

Dictionary VisualScript::_get_data() const{

	Dictionary d;
	d["base_type"]=base_type;
	Array vars;
	for (const Map<StringName,Variable>::Element *E=variables.front();E;E=E->next()) {

		Dictionary var = _get_variable_info(E->key());
		var["name"]=E->key(); //make sure it's the right one
		var["default_value"]=E->get().default_value;
		vars.push_back(var);
	}
	d["variables"]=vars;

	Array sigs;
	for (const Map<StringName,Vector<Argument> >::Element *E=custom_signals.front();E;E=E->next()) {

		Dictionary cs;
		cs["name"]=E->key();
		Array args;
		for(int i=0;i<E->get().size();i++) {
			args.push_back(E->get()[i].name);
			args.push_back(E->get()[i].type);
		}
		cs["arguments"]=args;

		sigs.push_back(cs);
	}

	d["signals"]=sigs;

	Array funcs;

	for (const Map<StringName,Function>::Element *E=functions.front();E;E=E->next()) {

		Dictionary func;
		func["name"]=E->key();
		func["function_id"]=E->get().function_id;

		Array nodes;

		for (const Map<int,Function::NodeData>::Element *F=E->get().nodes.front();F;F=F->next()) {

			nodes.push_back(F->key());
			nodes.push_back(F->get().pos);
			nodes.push_back(F->get().node);

		}

		func["nodes"]=nodes;

		Array sequence_connections;

		for (const Set<SequenceConnection>::Element *F=E->get().sequence_connections.front();F;F=F->next()) {

			sequence_connections.push_back(F->get().from_node);
			sequence_connections.push_back(F->get().from_output);
			sequence_connections.push_back(F->get().to_node);

		}


		func["sequence_connections"]=sequence_connections;

		Array data_connections;

		for (const Set<DataConnection>::Element *F=E->get().data_connections.front();F;F=F->next()) {

			data_connections.push_back(F->get().from_node);
			data_connections.push_back(F->get().from_port);
			data_connections.push_back(F->get().to_node);
			data_connections.push_back(F->get().to_port);

		}


		func["data_connections"]=data_connections;

		funcs.push_back(func);

	}

	d["functions"]=funcs;


	return d;

}

void VisualScript::_bind_methods() {



	ObjectTypeDB::bind_method(_MD("_node_ports_changed"),&VisualScript::_node_ports_changed);

	ObjectTypeDB::bind_method(_MD("add_function","name"),&VisualScript::add_function);
	ObjectTypeDB::bind_method(_MD("has_function","name"),&VisualScript::has_function);	
	ObjectTypeDB::bind_method(_MD("remove_function","name"),&VisualScript::remove_function);
	ObjectTypeDB::bind_method(_MD("rename_function","name","new_name"),&VisualScript::rename_function);

	ObjectTypeDB::bind_method(_MD("add_node","func","id","node","pos"),&VisualScript::add_node,DEFVAL(Point2()));
	ObjectTypeDB::bind_method(_MD("remove_node","func","id"),&VisualScript::remove_node);
	ObjectTypeDB::bind_method(_MD("get_function_node_id","name"),&VisualScript::get_function_node_id);

	ObjectTypeDB::bind_method(_MD("get_node","func","id"),&VisualScript::get_node);
	ObjectTypeDB::bind_method(_MD("set_node_pos","func","id","pos"),&VisualScript::set_node_pos);
	ObjectTypeDB::bind_method(_MD("get_node_pos","func","id"),&VisualScript::get_node_pos);

	ObjectTypeDB::bind_method(_MD("sequence_connect","func","from_node","from_output","to_node"),&VisualScript::sequence_connect);
	ObjectTypeDB::bind_method(_MD("sequence_disconnect","func","from_node","from_output","to_node"),&VisualScript::sequence_disconnect);
	ObjectTypeDB::bind_method(_MD("has_sequence_connection","func","from_node","from_output","to_node"),&VisualScript::has_sequence_connection);

	ObjectTypeDB::bind_method(_MD("data_connect","func","from_node","from_port","to_node","to_port"),&VisualScript::data_connect);
	ObjectTypeDB::bind_method(_MD("data_disconnect","func","from_node","from_port","to_node","to_port"),&VisualScript::data_disconnect);
	ObjectTypeDB::bind_method(_MD("has_data_connection","func","from_node","from_port","to_node","to_port"),&VisualScript::has_data_connection);

	ObjectTypeDB::bind_method(_MD("add_variable","name","default_value"),&VisualScript::add_variable,DEFVAL(Variant()));
	ObjectTypeDB::bind_method(_MD("has_variable","name"),&VisualScript::has_variable);
	ObjectTypeDB::bind_method(_MD("remove_variable","name"),&VisualScript::remove_variable);
	ObjectTypeDB::bind_method(_MD("set_variable_default_value","name","value"),&VisualScript::set_variable_default_value);
	ObjectTypeDB::bind_method(_MD("get_variable_default_value","name"),&VisualScript::get_variable_default_value);
	ObjectTypeDB::bind_method(_MD("set_variable_info","name","value"),&VisualScript::_set_variable_info);
	ObjectTypeDB::bind_method(_MD("get_variable_info","name"),&VisualScript::_get_variable_info);
	ObjectTypeDB::bind_method(_MD("rename_variable","name","new_name"),&VisualScript::rename_variable);

	ObjectTypeDB::bind_method(_MD("add_custom_signal","name"),&VisualScript::add_custom_signal);
	ObjectTypeDB::bind_method(_MD("has_custom_signal","name"),&VisualScript::has_custom_signal);
	ObjectTypeDB::bind_method(_MD("custom_signal_add_argument","name","type","argname","index"),&VisualScript::custom_signal_add_argument,DEFVAL(-1));
	ObjectTypeDB::bind_method(_MD("custom_signal_set_argument_type","name","argidx","type"),&VisualScript::custom_signal_set_argument_type);
	ObjectTypeDB::bind_method(_MD("custom_signal_get_argument_type","name","argidx"),&VisualScript::custom_signal_get_argument_type);
	ObjectTypeDB::bind_method(_MD("custom_signal_set_argument_name","name","argidx","argname"),&VisualScript::custom_signal_set_argument_name);
	ObjectTypeDB::bind_method(_MD("custom_signal_get_argument_name","name","argidx"),&VisualScript::custom_signal_get_argument_name);
	ObjectTypeDB::bind_method(_MD("custom_signal_remove_argument","argidx"),&VisualScript::custom_signal_remove_argument);
	ObjectTypeDB::bind_method(_MD("custom_signal_get_argument_count","name"),&VisualScript::custom_signal_get_argument_count);
	ObjectTypeDB::bind_method(_MD("custom_signal_swap_argument","name","argidx","withidx"),&VisualScript::custom_signal_swap_argument);
	ObjectTypeDB::bind_method(_MD("remove_custom_signal","name"),&VisualScript::remove_custom_signal);
	ObjectTypeDB::bind_method(_MD("rename_custom_signal","name","new_name"),&VisualScript::rename_custom_signal);

	//ObjectTypeDB::bind_method(_MD("set_variable_info","name","info"),&VScript::set_variable_info);
	//ObjectTypeDB::bind_method(_MD("get_variable_info","name"),&VScript::set_variable_info);

	ObjectTypeDB::bind_method(_MD("set_instance_base_type","type"),&VisualScript::set_instance_base_type);

	ObjectTypeDB::bind_method(_MD("_set_data","data"),&VisualScript::_set_data);
	ObjectTypeDB::bind_method(_MD("_get_data"),&VisualScript::_get_data);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY,"data",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR),_SCS("_set_data"),_SCS("_get_data"));

	ADD_SIGNAL(MethodInfo("node_ports_changed",PropertyInfo(Variant::STRING,"function"),PropertyInfo(Variant::INT,"id")));
}

VisualScript::VisualScript() {

	base_type="Object";

}

VisualScript::~VisualScript() {

	while(!functions.empty()) {
		remove_function(functions.front()->key());
	}

}

////////////////////////////////////////////


String VisualScriptLanguage::get_name() const {

	return "VisualScript";
}

/* LANGUAGE FUNCTIONS */
void VisualScriptLanguage::init() {


}
String VisualScriptLanguage::get_type() const {

	return "VisualScript";
}
String VisualScriptLanguage::get_extension() const {

	return "vs";
}
Error VisualScriptLanguage::execute_file(const String& p_path) {

	return OK;
}
void VisualScriptLanguage::finish() {


}

/* EDITOR FUNCTIONS */
void VisualScriptLanguage::get_reserved_words(List<String> *p_words) const {


}
void VisualScriptLanguage::get_comment_delimiters(List<String> *p_delimiters) const {


}
void VisualScriptLanguage::get_string_delimiters(List<String> *p_delimiters) const {


}
String VisualScriptLanguage::get_template(const String& p_class_name, const String& p_base_class_name) const {

	return String();
}
bool VisualScriptLanguage::validate(const String& p_script, int &r_line_error,int &r_col_error,String& r_test_error, const String& p_path,List<String> *r_functions) const {

	return false;
}
Script *VisualScriptLanguage::create_script() const {

	return memnew( VisualScript );
}
bool VisualScriptLanguage::has_named_classes() const {

	return false;
}
int VisualScriptLanguage::find_function(const String& p_function,const String& p_code) const {

	return -1;
}
String VisualScriptLanguage::make_function(const String& p_class,const String& p_name,const StringArray& p_args) const {

	return String();
}

void VisualScriptLanguage::auto_indent_code(String& p_code,int p_from_line,int p_to_line) const {


}
void VisualScriptLanguage::add_global_constant(const StringName& p_variable,const Variant& p_value) {


}


/* DEBUGGER FUNCTIONS */

String VisualScriptLanguage::debug_get_error() const {

	return String();
}
int VisualScriptLanguage::debug_get_stack_level_count() const {

	return 0;
}
int VisualScriptLanguage::debug_get_stack_level_line(int p_level) const {

	return 0;
}
String VisualScriptLanguage::debug_get_stack_level_function(int p_level) const {

	return String();
}
String VisualScriptLanguage::debug_get_stack_level_source(int p_level) const {

	return String();
}
void VisualScriptLanguage::debug_get_stack_level_locals(int p_level,List<String> *p_locals, List<Variant> *p_values, int p_max_subitems,int p_max_depth) {


}
void VisualScriptLanguage::debug_get_stack_level_members(int p_level,List<String> *p_members, List<Variant> *p_values, int p_max_subitems,int p_max_depth) {


}
void VisualScriptLanguage::debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems,int p_max_depth) {


}
String VisualScriptLanguage::debug_parse_stack_level_expression(int p_level,const String& p_expression,int p_max_subitems,int p_max_depth) {

	return String();
}


void VisualScriptLanguage::reload_all_scripts() {


}
void VisualScriptLanguage::reload_tool_script(const Ref<Script>& p_script,bool p_soft_reload) {


}
/* LOADER FUNCTIONS */

void VisualScriptLanguage::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("vs");

}
void VisualScriptLanguage::get_public_functions(List<MethodInfo> *p_functions) const {


}
void VisualScriptLanguage::get_public_constants(List<Pair<String,Variant> > *p_constants) const {


}

void VisualScriptLanguage::profiling_start() {


}
void VisualScriptLanguage::profiling_stop() {


}

int VisualScriptLanguage::profiling_get_accumulated_data(ProfilingInfo *p_info_arr,int p_info_max) {

	return 0;
}

int VisualScriptLanguage::profiling_get_frame_data(ProfilingInfo *p_info_arr,int p_info_max) {

	return 0;
}


VisualScriptLanguage* VisualScriptLanguage::singleton=NULL;


void VisualScriptLanguage::add_register_func(const String& p_name,VisualScriptNodeRegisterFunc p_func) {

	ERR_FAIL_COND(register_funcs.has(p_name));
	register_funcs[p_name]=p_func;
}

Ref<VisualScriptNode> VisualScriptLanguage::create_node_from_name(const String& p_name) {

	ERR_FAIL_COND_V(!register_funcs.has(p_name),Ref<VisualScriptNode>());

	return register_funcs[p_name](p_name);
}

void VisualScriptLanguage::get_registered_node_names(List<String> *r_names) {

	for (Map<String,VisualScriptNodeRegisterFunc>::Element *E=register_funcs.front();E;E=E->next()) {
		r_names->push_back(E->key());
	}
}


VisualScriptLanguage::VisualScriptLanguage() {

	singleton=this;
}
