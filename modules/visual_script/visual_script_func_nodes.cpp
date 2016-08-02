#include "visual_script_func_nodes.h"
#include "scene/main/scene_main_loop.h"
#include "os/os.h"
#include "scene/main/node.h"
#include "visual_script_nodes.h"

//////////////////////////////////////////
////////////////CALL//////////////////////
//////////////////////////////////////////

int VisualScriptFunctionCall::get_output_sequence_port_count() const {

	return 1;
}

bool VisualScriptFunctionCall::has_input_sequence_port() const{

	return true;
}
#ifdef TOOLS_ENABLED

static Node* _find_script_node(Node* p_edited_scene,Node* p_current_node,const Ref<Script> &script) {

	if (p_edited_scene!=p_current_node && p_current_node->get_owner()!=p_edited_scene)
		return NULL;

	Ref<Script> scr = p_current_node->get_script();

	if (scr.is_valid() && scr==script)
		return p_current_node;

	for(int i=0;i<p_current_node->get_child_count();i++) {
		Node *n = _find_script_node(p_edited_scene,p_current_node->get_child(i),script);
		if (n)
			return n;
	}

	return NULL;
}

#endif
Node *VisualScriptFunctionCall::_get_base_node() const {

#ifdef TOOLS_ENABLED
	Ref<Script> script = get_visual_script();
	if (!script.is_valid())
		return NULL;

	MainLoop * main_loop = OS::get_singleton()->get_main_loop();
	if (!main_loop)
		return NULL;

	SceneTree *scene_tree = main_loop->cast_to<SceneTree>();

	if (!scene_tree)
		return NULL;

	Node *edited_scene = scene_tree->get_edited_scene_root();

	if (!edited_scene)
		return NULL;

	Node* script_node = _find_script_node(edited_scene,edited_scene,script);

	if (!script_node)
		return NULL;

	if (!script_node->has_node(base_path))
		return NULL;

	Node *path_to = script_node->get_node(base_path);

	return path_to;
#else

	return NULL;
#endif
}

StringName VisualScriptFunctionCall::_get_base_type() const {

	if (call_mode==CALL_MODE_SELF && get_visual_script().is_valid())
		return get_visual_script()->get_instance_base_type();
	else if (call_mode==CALL_MODE_NODE_PATH && get_visual_script().is_valid()) {
		Node *path = _get_base_node();
		if (path)
			return path->get_type();

	}

	return base_type;
}

int VisualScriptFunctionCall::get_input_value_port_count() const{

	MethodBind *mb = ObjectTypeDB::get_method(_get_base_type(),function);
	if (!mb)
		return 0;

	return mb->get_argument_count() + (call_mode==CALL_MODE_INSTANCE?1:0) - use_default_args;

}
int VisualScriptFunctionCall::get_output_value_port_count() const{

	MethodBind *mb = ObjectTypeDB::get_method(_get_base_type(),function);
	if (!mb)
		return 0;

	return mb->has_return() ? 1 : 0;
}

String VisualScriptFunctionCall::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptFunctionCall::get_input_value_port_info(int p_idx) const{

	if (call_mode==CALL_MODE_INSTANCE) {
		if (p_idx==0) {
			PropertyInfo pi;
			pi.type=Variant::OBJECT;
			pi.name="instance";
			return pi;
		} else {
			p_idx--;
		}
	}

#ifdef DEBUG_METHODS_ENABLED



	MethodBind *mb = ObjectTypeDB::get_method(_get_base_type(),function);
	if (!mb)
		return PropertyInfo();

	return mb->get_argument_info(p_idx);
#else
	return PropertyInfo();
#endif

}

PropertyInfo VisualScriptFunctionCall::get_output_value_port_info(int p_idx) const{


#ifdef DEBUG_METHODS_ENABLED

	MethodBind *mb = ObjectTypeDB::get_method(_get_base_type(),function);
	if (!mb)
		return PropertyInfo();

	PropertyInfo pi = mb->get_argument_info(-1);
	pi.name="";
	return pi;
#else
	return PropertyInfo();
#endif
}


String VisualScriptFunctionCall::get_caption() const {

	return "Call";
}

String VisualScriptFunctionCall::get_text() const {

	if (call_mode==CALL_MODE_SELF)
		return "  "+String(function)+"()";
	else
		return "  "+base_type+"."+String(function)+"()";

}

void VisualScriptFunctionCall::_update_defargs() {

	if (!get_visual_script().is_valid())
		return; //do not change if not valid yet

	MethodBind *mb = ObjectTypeDB::get_method(_get_base_type(),function);
	if (!mb)
		return;

	use_default_args=mb->get_default_argument_count();

}

void VisualScriptFunctionCall::set_base_type(const StringName& p_type) {

	if (base_type==p_type)
		return;

	base_type=p_type;
	_update_defargs();
	_change_notify();
	emit_signal("ports_changed");
}

StringName VisualScriptFunctionCall::get_base_type() const{

	return base_type;
}

void VisualScriptFunctionCall::set_function(const StringName& p_type){

	if (function==p_type)
		return;

	function=p_type;
	_update_defargs();
	_change_notify();
	emit_signal("ports_changed");
}
StringName VisualScriptFunctionCall::get_function() const {


	return function;
}

void VisualScriptFunctionCall::set_base_path(const NodePath& p_type) {

	if (base_path==p_type)
		return;

	base_path=p_type;
	_update_defargs();
	_change_notify();
	emit_signal("ports_changed");
}

NodePath VisualScriptFunctionCall::get_base_path() const {

	return base_path;
}


void VisualScriptFunctionCall::set_call_mode(CallMode p_mode) {

	if (call_mode==p_mode)
		return;

	call_mode=p_mode;
	_update_defargs();
	_change_notify();
	emit_signal("ports_changed");

}
VisualScriptFunctionCall::CallMode VisualScriptFunctionCall::get_call_mode() const {

	return call_mode;
}

void VisualScriptFunctionCall::set_use_default_args(int p_amount) {

	if (use_default_args==p_amount)
		return;

	use_default_args=p_amount;
	emit_signal("ports_changed");


}

int VisualScriptFunctionCall::get_use_default_args() const{

	return use_default_args;
}
void VisualScriptFunctionCall::_validate_property(PropertyInfo& property) const {

	if (property.name=="function/base_type") {
		if (call_mode!=CALL_MODE_INSTANCE) {
			property.usage=0;
		}
	}


	if (property.name=="function/node_path") {
		if (call_mode!=CALL_MODE_NODE_PATH) {
			property.usage=0;
		} else {

			Node *bnode = _get_base_node();
			if (bnode) {
				property.hint_string=bnode->get_path(); //convert to loong string
			} else {

			}
		}
	}

	if (property.name=="function/function") {
		property.hint=PROPERTY_HINT_ENUM;


		List<MethodInfo> methods;

		StringName base = _get_base_type();
		ObjectTypeDB::get_method_list(base,&methods);


		List<String> mstring;
		for (List<MethodInfo>::Element *E=methods.front();E;E=E->next()) {
			if (E->get().name.begins_with("_"))
				continue;
			mstring.push_back(E->get().name.get_slice(":",0));
		}

		mstring.sort();

		String ml;
		for (List<String>::Element *E=mstring.front();E;E=E->next()) {

			if (ml!=String())
				ml+=",";
			ml+=E->get();
		}

		property.hint_string=ml;
	}

	if (property.name=="function/use_default_args") {
		property.hint=PROPERTY_HINT_RANGE;

		int mc=0;

		MethodBind *mb = ObjectTypeDB::get_method(_get_base_type(),function);
		if (mb) {

			mc=mb->get_default_argument_count();
		}

		property.hint_string="0,"+itos(mc)+",1";
	}
}


void VisualScriptFunctionCall::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_base_type","base_type"),&VisualScriptFunctionCall::set_base_type);
	ObjectTypeDB::bind_method(_MD("get_base_type"),&VisualScriptFunctionCall::get_base_type);

	ObjectTypeDB::bind_method(_MD("set_function","function"),&VisualScriptFunctionCall::set_function);
	ObjectTypeDB::bind_method(_MD("get_function"),&VisualScriptFunctionCall::get_function);

	ObjectTypeDB::bind_method(_MD("set_call_mode","mode"),&VisualScriptFunctionCall::set_call_mode);
	ObjectTypeDB::bind_method(_MD("get_call_mode"),&VisualScriptFunctionCall::get_call_mode);

	ObjectTypeDB::bind_method(_MD("set_base_path","base_path"),&VisualScriptFunctionCall::set_base_path);
	ObjectTypeDB::bind_method(_MD("get_base_path"),&VisualScriptFunctionCall::get_base_path);

	ObjectTypeDB::bind_method(_MD("set_use_default_args","amount"),&VisualScriptFunctionCall::set_use_default_args);
	ObjectTypeDB::bind_method(_MD("get_use_default_args"),&VisualScriptFunctionCall::get_use_default_args);


	ADD_PROPERTY(PropertyInfo(Variant::INT,"function/call_mode",PROPERTY_HINT_ENUM,"Self,Node Path,Instance"),_SCS("set_call_mode"),_SCS("get_call_mode"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"function/base_type",PROPERTY_HINT_TYPE_STRING,"Object"),_SCS("set_base_type"),_SCS("get_base_type"));
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,"function/node_path",PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE),_SCS("set_base_path"),_SCS("get_base_path"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"function/function"),_SCS("set_function"),_SCS("get_function"));
	ADD_PROPERTY(PropertyInfo(Variant::INT,"function/use_default_args"),_SCS("set_use_default_args"),_SCS("get_use_default_args"));

	BIND_CONSTANT( CALL_MODE_SELF );
	BIND_CONSTANT( CALL_MODE_NODE_PATH);
	BIND_CONSTANT( CALL_MODE_INSTANCE);
}

VisualScriptNodeInstance* VisualScriptFunctionCall::instance(VScriptInstance* p_instance) {

	return NULL;
}

VisualScriptFunctionCall::VisualScriptFunctionCall() {

	call_mode=CALL_MODE_INSTANCE;
	use_default_args=0;
	base_type="Object";

}

template<VisualScriptFunctionCall::CallMode cmode>
static Ref<VisualScriptNode> create_function_call_node(const String& p_name) {

	Ref<VisualScriptFunctionCall> node;
	node.instance();
	node->set_call_mode(cmode);
	return node;
}


//////////////////////////////////////////
////////////////SET//////////////////////
//////////////////////////////////////////

int VisualScriptPropertySet::get_output_sequence_port_count() const {

	return 1;
}

bool VisualScriptPropertySet::has_input_sequence_port() const{

	return true;
}

Node *VisualScriptPropertySet::_get_base_node() const {

#ifdef TOOLS_ENABLED
	Ref<Script> script = get_visual_script();
	if (!script.is_valid())
		return NULL;

	MainLoop * main_loop = OS::get_singleton()->get_main_loop();
	if (!main_loop)
		return NULL;

	SceneTree *scene_tree = main_loop->cast_to<SceneTree>();

	if (!scene_tree)
		return NULL;

	Node *edited_scene = scene_tree->get_edited_scene_root();

	if (!edited_scene)
		return NULL;

	Node* script_node = _find_script_node(edited_scene,edited_scene,script);

	if (!script_node)
		return NULL;

	if (!script_node->has_node(base_path))
		return NULL;

	Node *path_to = script_node->get_node(base_path);

	return path_to;
#else

	return NULL;
#endif
}

StringName VisualScriptPropertySet::_get_base_type() const {

	if (call_mode==CALL_MODE_SELF && get_visual_script().is_valid())
		return get_visual_script()->get_instance_base_type();
	else if (call_mode==CALL_MODE_NODE_PATH && get_visual_script().is_valid()) {
		Node *path = _get_base_node();
		if (path)
			return path->get_type();

	}

	return base_type;
}

int VisualScriptPropertySet::get_input_value_port_count() const{

	if (use_builtin_value)
		return 0;
	else
		return 1;


}
int VisualScriptPropertySet::get_output_value_port_count() const{

	return 0;
}

String VisualScriptPropertySet::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptPropertySet::get_input_value_port_info(int p_idx) const{

	if (call_mode==CALL_MODE_INSTANCE) {
		if (p_idx==0) {
			PropertyInfo pi;
			pi.type=Variant::OBJECT;
			pi.name="instance";
			return pi;
		} else {
			p_idx--;
		}
	}

#ifdef DEBUG_METHODS_ENABLED

	//not very efficient but..


	List<PropertyInfo> pinfo;

	if (call_mode==CALL_MODE_NODE_PATH) {

		Node *n = _get_base_node();
		if (n) {
			n->get_property_list(&pinfo);
		} else {
			ObjectTypeDB::get_property_list(_get_base_type(),&pinfo);
		}
	} else {
		ObjectTypeDB::get_property_list(_get_base_type(),&pinfo);
	}

	for (List<PropertyInfo>::Element *E=pinfo.front();E;E=E->next()) {

		if (E->get().name==property) {

			PropertyInfo info=E->get();
			info.name="value";
			return info;
		}
	}


#endif

	return PropertyInfo(Variant::NIL,"value");

}

PropertyInfo VisualScriptPropertySet::get_output_value_port_info(int p_idx) const{

	return PropertyInfo();

}


String VisualScriptPropertySet::get_caption() const {

	return "Set";
}

String VisualScriptPropertySet::get_text() const {

	return property;

}

void VisualScriptPropertySet::set_base_type(const StringName& p_type) {

	if (base_type==p_type)
		return;

	base_type=p_type;
	_change_notify();
	emit_signal("ports_changed");
}

StringName VisualScriptPropertySet::get_base_type() const{

	return base_type;
}

void VisualScriptPropertySet::set_property(const StringName& p_type){

	if (property==p_type)
		return;

	property=p_type;
	_change_notify();
	emit_signal("ports_changed");
}
StringName VisualScriptPropertySet::get_property() const {


	return property;
}

void VisualScriptPropertySet::set_base_path(const NodePath& p_type) {

	if (base_path==p_type)
		return;

	base_path=p_type;
	_change_notify();
	emit_signal("ports_changed");
}

NodePath VisualScriptPropertySet::get_base_path() const {

	return base_path;
}


void VisualScriptPropertySet::set_call_mode(CallMode p_mode) {

	if (call_mode==p_mode)
		return;

	call_mode=p_mode;
	_change_notify();
	emit_signal("ports_changed");

}
VisualScriptPropertySet::CallMode VisualScriptPropertySet::get_call_mode() const {

	return call_mode;
}


void VisualScriptPropertySet::set_use_builtin_value(bool p_use) {

	if (use_builtin_value==p_use)
		return;

	use_builtin_value=p_use;
	_change_notify();
	emit_signal("ports_changed");

}

bool VisualScriptPropertySet::is_using_builtin_value() const{

	return use_builtin_value;
}

void VisualScriptPropertySet::set_builtin_value(const Variant& p_value){

	if (builtin_value==p_value)
		return;

	builtin_value=p_value;

}
Variant VisualScriptPropertySet::get_builtin_value() const{

	return builtin_value;
}
void VisualScriptPropertySet::_validate_property(PropertyInfo& property) const {

	if (property.name=="property/base_type") {
		if (call_mode!=CALL_MODE_INSTANCE) {
			property.usage=0;
		}
	}


	if (property.name=="property/node_path") {
		if (call_mode!=CALL_MODE_NODE_PATH) {
			property.usage=0;
		} else {

			Node *bnode = _get_base_node();
			if (bnode) {
				property.hint_string=bnode->get_path(); //convert to loong string
			} else {

			}
		}
	}

	if (property.name=="property/property") {
		property.hint=PROPERTY_HINT_ENUM;


		List<PropertyInfo> pinfo;

		if (call_mode==CALL_MODE_NODE_PATH) {

			Node *n = _get_base_node();
			if (n) {
				n->get_property_list(&pinfo);
			} else {
				ObjectTypeDB::get_property_list(_get_base_type(),&pinfo);
			}
		} else {
			ObjectTypeDB::get_property_list(_get_base_type(),&pinfo);
		}

		List<String> mstring;

		for (List<PropertyInfo>::Element *E=pinfo.front();E;E=E->next()) {

			if (E->get().usage&PROPERTY_USAGE_EDITOR)
				mstring.push_back(E->get().name);
		}

		String ml;
		for (List<String>::Element *E=mstring.front();E;E=E->next()) {

			if (ml!=String())
				ml+=",";
			ml+=E->get();
		}

		property.hint_string=ml;
	}

	if (property.name=="value/builtin") {

		if (!use_builtin_value) {
			property.usage=0;
		} else {
			List<PropertyInfo> pinfo;


			if (call_mode==CALL_MODE_NODE_PATH) {

				Node *n = _get_base_node();
				if (n) {
					n->get_property_list(&pinfo);
				} else {
					ObjectTypeDB::get_property_list(_get_base_type(),&pinfo);
				}
			} else {
				ObjectTypeDB::get_property_list(_get_base_type(),&pinfo);
			}

			for (List<PropertyInfo>::Element *E=pinfo.front();E;E=E->next()) {

				if (E->get().name==this->property) {

					property.hint=E->get().hint;
					property.type=E->get().type;
					property.hint_string=E->get().hint_string;
				}
			}
		}

	}
}

void VisualScriptPropertySet::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_base_type","base_type"),&VisualScriptPropertySet::set_base_type);
	ObjectTypeDB::bind_method(_MD("get_base_type"),&VisualScriptPropertySet::get_base_type);

	ObjectTypeDB::bind_method(_MD("set_property","property"),&VisualScriptPropertySet::set_property);
	ObjectTypeDB::bind_method(_MD("get_property"),&VisualScriptPropertySet::get_property);

	ObjectTypeDB::bind_method(_MD("set_call_mode","mode"),&VisualScriptPropertySet::set_call_mode);
	ObjectTypeDB::bind_method(_MD("get_call_mode"),&VisualScriptPropertySet::get_call_mode);

	ObjectTypeDB::bind_method(_MD("set_base_path","base_path"),&VisualScriptPropertySet::set_base_path);
	ObjectTypeDB::bind_method(_MD("get_base_path"),&VisualScriptPropertySet::get_base_path);

	ObjectTypeDB::bind_method(_MD("set_builtin_value","value"),&VisualScriptPropertySet::set_builtin_value);
	ObjectTypeDB::bind_method(_MD("get_builtin_value"),&VisualScriptPropertySet::get_builtin_value);

	ObjectTypeDB::bind_method(_MD("set_use_builtin_value","enable"),&VisualScriptPropertySet::set_use_builtin_value);
	ObjectTypeDB::bind_method(_MD("is_using_builtin_value"),&VisualScriptPropertySet::is_using_builtin_value);

	ADD_PROPERTY(PropertyInfo(Variant::INT,"property/set_mode",PROPERTY_HINT_ENUM,"Self,Node Path,Instance"),_SCS("set_call_mode"),_SCS("get_call_mode"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"property/base_type",PROPERTY_HINT_TYPE_STRING,"Object"),_SCS("set_base_type"),_SCS("get_base_type"));
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,"property/node_path",PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE),_SCS("set_base_path"),_SCS("get_base_path"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"property/property"),_SCS("set_property"),_SCS("get_property"));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"value/use_builtin"),_SCS("set_use_builtin_value"),_SCS("is_using_builtin_value"));
	ADD_PROPERTY(PropertyInfo(Variant::NIL,"value/builtin"),_SCS("set_builtin_value"),_SCS("get_builtin_value"));

	BIND_CONSTANT( CALL_MODE_SELF );
	BIND_CONSTANT( CALL_MODE_NODE_PATH);
	BIND_CONSTANT( CALL_MODE_INSTANCE);
}

VisualScriptNodeInstance* VisualScriptPropertySet::instance(VScriptInstance* p_instance) {

	return NULL;
}

VisualScriptPropertySet::VisualScriptPropertySet() {

	call_mode=CALL_MODE_INSTANCE;
	base_type="Object";

}

template<VisualScriptPropertySet::CallMode cmode>
static Ref<VisualScriptNode> create_property_set_node(const String& p_name) {

	Ref<VisualScriptPropertySet> node;
	node.instance();
	node->set_call_mode(cmode);
	return node;
}


//////////////////////////////////////////
////////////////SET//////////////////////
//////////////////////////////////////////

int VisualScriptPropertyGet::get_output_sequence_port_count() const {

	return 1;
}

bool VisualScriptPropertyGet::has_input_sequence_port() const{

	return true;
}

Node *VisualScriptPropertyGet::_get_base_node() const {

#ifdef TOOLS_ENABLED
	Ref<Script> script = get_visual_script();
	if (!script.is_valid())
		return NULL;

	MainLoop * main_loop = OS::get_singleton()->get_main_loop();
	if (!main_loop)
		return NULL;

	SceneTree *scene_tree = main_loop->cast_to<SceneTree>();

	if (!scene_tree)
		return NULL;

	Node *edited_scene = scene_tree->get_edited_scene_root();

	if (!edited_scene)
		return NULL;

	Node* script_node = _find_script_node(edited_scene,edited_scene,script);

	if (!script_node)
		return NULL;

	if (!script_node->has_node(base_path))
		return NULL;

	Node *path_to = script_node->get_node(base_path);

	return path_to;
#else

	return NULL;
#endif
}

StringName VisualScriptPropertyGet::_get_base_type() const {

	if (call_mode==CALL_MODE_SELF && get_visual_script().is_valid())
		return get_visual_script()->get_instance_base_type();
	else if (call_mode==CALL_MODE_NODE_PATH && get_visual_script().is_valid()) {
		Node *path = _get_base_node();
		if (path)
			return path->get_type();

	}

	return base_type;
}

int VisualScriptPropertyGet::get_input_value_port_count() const{

	return 0;

}
int VisualScriptPropertyGet::get_output_value_port_count() const{

	return 1;
}

String VisualScriptPropertyGet::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptPropertyGet::get_input_value_port_info(int p_idx) const{

	if (call_mode==CALL_MODE_INSTANCE) {
		if (p_idx==0) {
			PropertyInfo pi;
			pi.type=Variant::OBJECT;
			pi.name="instance";
			return pi;
		} else {
			p_idx--;
		}
	}
	return PropertyInfo();

}

PropertyInfo VisualScriptPropertyGet::get_output_value_port_info(int p_idx) const{



#ifdef DEBUG_METHODS_ENABLED

	//not very efficient but..


	List<PropertyInfo> pinfo;

	if (call_mode==CALL_MODE_NODE_PATH) {

		Node *n = _get_base_node();
		if (n) {
			n->get_property_list(&pinfo);
		} else {
			ObjectTypeDB::get_property_list(_get_base_type(),&pinfo);
		}
	} else {
		ObjectTypeDB::get_property_list(_get_base_type(),&pinfo);
	}

	for (List<PropertyInfo>::Element *E=pinfo.front();E;E=E->next()) {

		if (E->get().name==property) {

			PropertyInfo info=E->get();
			info.name="";
			return info;
		}
	}


#endif

	return PropertyInfo(Variant::NIL,"");
}


String VisualScriptPropertyGet::get_caption() const {

	return "Get";
}

String VisualScriptPropertyGet::get_text() const {

	return property;

}

void VisualScriptPropertyGet::set_base_type(const StringName& p_type) {

	if (base_type==p_type)
		return;

	base_type=p_type;
	_change_notify();
	emit_signal("ports_changed");
}

StringName VisualScriptPropertyGet::get_base_type() const{

	return base_type;
}

void VisualScriptPropertyGet::set_property(const StringName& p_type){

	if (property==p_type)
		return;

	property=p_type;
	_change_notify();
	emit_signal("ports_changed");
}
StringName VisualScriptPropertyGet::get_property() const {


	return property;
}

void VisualScriptPropertyGet::set_base_path(const NodePath& p_type) {

	if (base_path==p_type)
		return;

	base_path=p_type;
	_change_notify();
	emit_signal("ports_changed");
}

NodePath VisualScriptPropertyGet::get_base_path() const {

	return base_path;
}


void VisualScriptPropertyGet::set_call_mode(CallMode p_mode) {

	if (call_mode==p_mode)
		return;

	call_mode=p_mode;
	_change_notify();
	emit_signal("ports_changed");

}
VisualScriptPropertyGet::CallMode VisualScriptPropertyGet::get_call_mode() const {

	return call_mode;
}

void VisualScriptPropertyGet::_validate_property(PropertyInfo& property) const {

	if (property.name=="property/base_type") {
		if (call_mode!=CALL_MODE_INSTANCE) {
			property.usage=0;
		}
	}


	if (property.name=="property/node_path") {
		if (call_mode!=CALL_MODE_NODE_PATH) {
			property.usage=0;
		} else {

			Node *bnode = _get_base_node();
			if (bnode) {
				property.hint_string=bnode->get_path(); //convert to loong string
			} else {

			}
		}
	}

	if (property.name=="property/property") {
		property.hint=PROPERTY_HINT_ENUM;


		List<PropertyInfo> pinfo;

		if (call_mode==CALL_MODE_NODE_PATH) {

			Node *n = _get_base_node();
			if (n) {
				n->get_property_list(&pinfo);
			} else {
				ObjectTypeDB::get_property_list(_get_base_type(),&pinfo);
			}
		} else {
			ObjectTypeDB::get_property_list(_get_base_type(),&pinfo);
		}

		List<String> mstring;

		for (List<PropertyInfo>::Element *E=pinfo.front();E;E=E->next()) {

			if (E->get().usage&PROPERTY_USAGE_EDITOR)
				mstring.push_back(E->get().name);
		}

		String ml;
		for (List<String>::Element *E=mstring.front();E;E=E->next()) {

			if (ml!=String())
				ml+=",";
			ml+=E->get();
		}

		property.hint_string=ml;
	}

}

void VisualScriptPropertyGet::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_base_type","base_type"),&VisualScriptPropertyGet::set_base_type);
	ObjectTypeDB::bind_method(_MD("get_base_type"),&VisualScriptPropertyGet::get_base_type);

	ObjectTypeDB::bind_method(_MD("set_property","property"),&VisualScriptPropertyGet::set_property);
	ObjectTypeDB::bind_method(_MD("get_property"),&VisualScriptPropertyGet::get_property);

	ObjectTypeDB::bind_method(_MD("set_call_mode","mode"),&VisualScriptPropertyGet::set_call_mode);
	ObjectTypeDB::bind_method(_MD("get_call_mode"),&VisualScriptPropertyGet::get_call_mode);

	ObjectTypeDB::bind_method(_MD("set_base_path","base_path"),&VisualScriptPropertyGet::set_base_path);
	ObjectTypeDB::bind_method(_MD("get_base_path"),&VisualScriptPropertyGet::get_base_path);



	ADD_PROPERTY(PropertyInfo(Variant::INT,"property/set_mode",PROPERTY_HINT_ENUM,"Self,Node Path,Instance"),_SCS("set_call_mode"),_SCS("get_call_mode"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"property/base_type",PROPERTY_HINT_TYPE_STRING,"Object"),_SCS("set_base_type"),_SCS("get_base_type"));
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,"property/node_path",PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE),_SCS("set_base_path"),_SCS("get_base_path"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"property/property"),_SCS("set_property"),_SCS("get_property"));

	BIND_CONSTANT( CALL_MODE_SELF );
	BIND_CONSTANT( CALL_MODE_NODE_PATH);
	BIND_CONSTANT( CALL_MODE_INSTANCE);
}

VisualScriptNodeInstance* VisualScriptPropertyGet::instance(VScriptInstance* p_instance) {

	return NULL;
}

VisualScriptPropertyGet::VisualScriptPropertyGet() {

	call_mode=CALL_MODE_INSTANCE;
	base_type="Object";

}

template<VisualScriptPropertyGet::CallMode cmode>
static Ref<VisualScriptNode> create_property_get_node(const String& p_name) {

	Ref<VisualScriptPropertyGet> node;
	node.instance();
	node->set_call_mode(cmode);
	return node;
}


//////////////////////////////////////////
////////////////SCRIPT CALL//////////////////////
//////////////////////////////////////////

int VisualScriptScriptCall::get_output_sequence_port_count() const {

	return 1;
}

bool VisualScriptScriptCall::has_input_sequence_port() const{

	return true;
}

Node *VisualScriptScriptCall::_get_base_node() const {

#ifdef TOOLS_ENABLED
	Ref<Script> script = get_visual_script();
	if (!script.is_valid())
		return NULL;

	MainLoop * main_loop = OS::get_singleton()->get_main_loop();
	if (!main_loop)
		return NULL;

	SceneTree *scene_tree = main_loop->cast_to<SceneTree>();

	if (!scene_tree)
		return NULL;

	Node *edited_scene = scene_tree->get_edited_scene_root();

	if (!edited_scene)
		return NULL;

	Node* script_node = _find_script_node(edited_scene,edited_scene,script);

	if (!script_node)
		return NULL;

	if (!script_node->has_node(base_path))
		return NULL;

	Node *path_to = script_node->get_node(base_path);

	return path_to;
#else

	return NULL;
#endif
}


int VisualScriptScriptCall::get_input_value_port_count() const{


	if (call_mode==CALL_MODE_SELF) {

		Ref<VisualScript> vs = get_visual_script();
		if (vs.is_valid()) {

			if (!vs->has_function(function))
				return 0;

			int id = vs->get_function_node_id(function);
			if (id<0)
				return 0;

			Ref<VisualScriptFunction> func = vs->get_node(function,id);

			return func->get_argument_count();
		}
	} else {

		Node*base = _get_base_node();
		if (!base)
			return 0;
		Ref<Script> script = base->get_script();
		if (!script.is_valid())
			return 0;

		List<MethodInfo> functions;
		script->get_method_list(&functions);
		for (List<MethodInfo>::Element *E=functions.front();E;E=E->next()) {
			if (E->get().name==function) {
				return E->get().arguments.size();
			}
		}

	}


	return 0;

}
int VisualScriptScriptCall::get_output_value_port_count() const{
	return 1;
}

String VisualScriptScriptCall::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptScriptCall::get_input_value_port_info(int p_idx) const{

	if (call_mode==CALL_MODE_SELF) {

		Ref<VisualScript> vs = get_visual_script();
		if (vs.is_valid()) {

			if (!vs->has_function(function))
				return PropertyInfo();

			int id = vs->get_function_node_id(function);
			if (id<0)
				return PropertyInfo();

			Ref<VisualScriptFunction> func = vs->get_node(function,id);

			if (p_idx>=func->get_argument_count())
				return PropertyInfo();
			return PropertyInfo(func->get_argument_type(p_idx),func->get_argument_name(p_idx));
		}
	} else {

		Node*base = _get_base_node();
		if (!base)
			return PropertyInfo();
		Ref<Script> script = base->get_script();
		if (!script.is_valid())
			return PropertyInfo();

		List<MethodInfo> functions;
		script->get_method_list(&functions);
		for (List<MethodInfo>::Element *E=functions.front();E;E=E->next()) {
			if (E->get().name==function) {
				if (p_idx<0 || p_idx>=E->get().arguments.size())
					return PropertyInfo();
				return E->get().arguments[p_idx];
			}
		}

	}

	return PropertyInfo();

}

PropertyInfo VisualScriptScriptCall::get_output_value_port_info(int p_idx) const{

	return PropertyInfo();
}


String VisualScriptScriptCall::get_caption() const {

	return "ScriptCall";
}

String VisualScriptScriptCall::get_text() const {

	return "  "+String(function)+"()";
}



void VisualScriptScriptCall::set_function(const StringName& p_type){

	if (function==p_type)
		return;

	function=p_type;

	_change_notify();
	emit_signal("ports_changed");
}
StringName VisualScriptScriptCall::get_function() const {


	return function;
}

void VisualScriptScriptCall::set_base_path(const NodePath& p_type) {

	if (base_path==p_type)
		return;

	base_path=p_type;

	_change_notify();
	emit_signal("ports_changed");
}

NodePath VisualScriptScriptCall::get_base_path() const {

	return base_path;
}


void VisualScriptScriptCall::set_call_mode(CallMode p_mode) {

	if (call_mode==p_mode)
		return;

	call_mode=p_mode;

	_change_notify();
	emit_signal("ports_changed");

}
VisualScriptScriptCall::CallMode VisualScriptScriptCall::get_call_mode() const {

	return call_mode;
}

void VisualScriptScriptCall::_validate_property(PropertyInfo& property) const {



	if (property.name=="function/node_path") {
		if (call_mode!=CALL_MODE_NODE_PATH) {
			property.usage=0;
		} else {

			Node *bnode = _get_base_node();
			if (bnode) {
				property.hint_string=bnode->get_path(); //convert to loong string
			} else {

			}
		}
	}

	if (property.name=="function/function") {
		property.hint=PROPERTY_HINT_ENUM;


		List<MethodInfo> methods;

		if (call_mode==CALL_MODE_SELF) {

			Ref<VisualScript> vs = get_visual_script();
			if (vs.is_valid()) {

				vs->get_method_list(&methods);

			}
		} else {

			Node*base = _get_base_node();
			if (!base)
				return;
			Ref<Script> script = base->get_script();
			if (!script.is_valid())
				return;

			script->get_method_list(&methods);

		}

		List<String> mstring;
		for (List<MethodInfo>::Element *E=methods.front();E;E=E->next()) {
			if (E->get().name.begins_with("_"))
				continue;
			mstring.push_back(E->get().name.get_slice(":",0));
		}

		mstring.sort();

		String ml;
		for (List<String>::Element *E=mstring.front();E;E=E->next()) {

			if (ml!=String())
				ml+=",";
			ml+=E->get();
		}

		property.hint_string=ml;
	}

}


void VisualScriptScriptCall::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_function","function"),&VisualScriptScriptCall::set_function);
	ObjectTypeDB::bind_method(_MD("get_function"),&VisualScriptScriptCall::get_function);

	ObjectTypeDB::bind_method(_MD("set_call_mode","mode"),&VisualScriptScriptCall::set_call_mode);
	ObjectTypeDB::bind_method(_MD("get_call_mode"),&VisualScriptScriptCall::get_call_mode);

	ObjectTypeDB::bind_method(_MD("set_base_path","base_path"),&VisualScriptScriptCall::set_base_path);
	ObjectTypeDB::bind_method(_MD("get_base_path"),&VisualScriptScriptCall::get_base_path);


	ADD_PROPERTY(PropertyInfo(Variant::INT,"function/call_mode",PROPERTY_HINT_ENUM,"Self,Node Path"),_SCS("set_call_mode"),_SCS("get_call_mode"));
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,"function/node_path",PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE),_SCS("set_base_path"),_SCS("get_base_path"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"function/function"),_SCS("set_function"),_SCS("get_function"));

	BIND_CONSTANT( CALL_MODE_SELF );
	BIND_CONSTANT( CALL_MODE_NODE_PATH);

}

VisualScriptNodeInstance* VisualScriptScriptCall::instance(VScriptInstance* p_instance) {

	return NULL;
}

VisualScriptScriptCall::VisualScriptScriptCall() {

	call_mode=CALL_MODE_SELF;


}

template<VisualScriptScriptCall::CallMode cmode>
static Ref<VisualScriptNode> create_script_call_node(const String& p_name) {

	Ref<VisualScriptScriptCall> node;
	node.instance();
	node->set_call_mode(cmode);
	return node;
}


//////////////////////////////////////////
////////////////SCRIPT CALL//////////////////////
//////////////////////////////////////////

int VisualScriptEmitSignal::get_output_sequence_port_count() const {

	return 1;
}

bool VisualScriptEmitSignal::has_input_sequence_port() const{

	return true;
}


int VisualScriptEmitSignal::get_input_value_port_count() const{

	Ref<VisualScript> vs = get_visual_script();
	if (vs.is_valid()) {

		if (!vs->has_custom_signal(name))
			return 0;

		return vs->custom_signal_get_argument_count(name);
	}

	return 0;

}
int VisualScriptEmitSignal::get_output_value_port_count() const{
	return 0;
}

String VisualScriptEmitSignal::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptEmitSignal::get_input_value_port_info(int p_idx) const{

	Ref<VisualScript> vs = get_visual_script();
	if (vs.is_valid()) {

		if (!vs->has_custom_signal(name))
			return PropertyInfo();

		return PropertyInfo(vs->custom_signal_get_argument_type(name,p_idx),vs->custom_signal_get_argument_name(name,p_idx));
	}

	return PropertyInfo();

}

PropertyInfo VisualScriptEmitSignal::get_output_value_port_info(int p_idx) const{

	return PropertyInfo();
}


String VisualScriptEmitSignal::get_caption() const {

	return "EmitSignal";
}

String VisualScriptEmitSignal::get_text() const {

	return "emit "+String(name);
}



void VisualScriptEmitSignal::set_signal(const StringName& p_type){

	if (name==p_type)
		return;

	name=p_type;

	_change_notify();
	emit_signal("ports_changed");
}
StringName VisualScriptEmitSignal::get_signal() const {


	return name;
}


void VisualScriptEmitSignal::_validate_property(PropertyInfo& property) const {



	if (property.name=="signal/signal") {
		property.hint=PROPERTY_HINT_ENUM;


		List<StringName> sigs;

		Ref<VisualScript> vs = get_visual_script();
		if (vs.is_valid()) {

			vs->get_custom_signal_list(&sigs);

		}

		String ml;
		for (List<StringName>::Element *E=sigs.front();E;E=E->next()) {

			if (ml!=String())
				ml+=",";
			ml+=E->get();
		}

		property.hint_string=ml;
	}

}


void VisualScriptEmitSignal::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_signal","name"),&VisualScriptEmitSignal::set_signal);
	ObjectTypeDB::bind_method(_MD("get_signal"),&VisualScriptEmitSignal::get_signal);


	ADD_PROPERTY(PropertyInfo(Variant::STRING,"signal/signal"),_SCS("set_signal"),_SCS("get_signal"));


}

VisualScriptNodeInstance* VisualScriptEmitSignal::instance(VScriptInstance* p_instance) {

	return NULL;
}

VisualScriptEmitSignal::VisualScriptEmitSignal() {
}

void register_visual_script_func_nodes() {

	VisualScriptLanguage::singleton->add_register_func("functions/call_method/call",create_function_call_node<VisualScriptFunctionCall::CALL_MODE_INSTANCE>);
	VisualScriptLanguage::singleton->add_register_func("functions/call_method/call_in_self",create_function_call_node<VisualScriptFunctionCall::CALL_MODE_SELF>);
	VisualScriptLanguage::singleton->add_register_func("functions/call_method/call_in_node",create_function_call_node<VisualScriptFunctionCall::CALL_MODE_NODE_PATH>);

	VisualScriptLanguage::singleton->add_register_func("functions/set_property/set",create_property_set_node<VisualScriptPropertySet::CALL_MODE_INSTANCE>);
	VisualScriptLanguage::singleton->add_register_func("functions/set_property/set_in_self",create_property_set_node<VisualScriptPropertySet::CALL_MODE_SELF>);
	VisualScriptLanguage::singleton->add_register_func("functions/set_property/set_in_node",create_property_set_node<VisualScriptPropertySet::CALL_MODE_NODE_PATH>);

	VisualScriptLanguage::singleton->add_register_func("functions/get_property/get",create_property_get_node<VisualScriptPropertyGet::CALL_MODE_INSTANCE>);
	VisualScriptLanguage::singleton->add_register_func("functions/get_property/get_from_self",create_property_get_node<VisualScriptPropertyGet::CALL_MODE_SELF>);
	VisualScriptLanguage::singleton->add_register_func("functions/get_property/get_from_node",create_property_get_node<VisualScriptPropertyGet::CALL_MODE_NODE_PATH>);

	VisualScriptLanguage::singleton->add_register_func("functions/script/script_call",create_script_call_node<VisualScriptScriptCall::CALL_MODE_SELF>);
	VisualScriptLanguage::singleton->add_register_func("functions/script/script_call_in_node",create_script_call_node<VisualScriptScriptCall::CALL_MODE_NODE_PATH>);
	VisualScriptLanguage::singleton->add_register_func("functions/script/emit_signal",create_node_generic<VisualScriptEmitSignal>);


}
