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

	if (call_mode==CALL_MODE_BASIC_TYPE) {


		Vector<StringName> names = Variant::get_method_argument_names(basic_type,function);
		return names.size()+1;

	} else {
		MethodBind *mb = ObjectTypeDB::get_method(_get_base_type(),function);
		if (!mb)
			return 0;

		return mb->get_argument_count() + (call_mode==CALL_MODE_INSTANCE?1:0) - use_default_args;
	}

}
int VisualScriptFunctionCall::get_output_value_port_count() const{

	if (call_mode==CALL_MODE_BASIC_TYPE) {

		bool returns=false;
		Variant::get_method_return_type(basic_type,function,&returns);
		return returns?1:0;

	} else {
		MethodBind *mb = ObjectTypeDB::get_method(_get_base_type(),function);
		if (!mb)
			return 0;

		return mb->has_return() ? 1 : 0;
	}
}

String VisualScriptFunctionCall::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptFunctionCall::get_input_value_port_info(int p_idx) const{

	if (call_mode==CALL_MODE_INSTANCE || call_mode==CALL_MODE_BASIC_TYPE) {
		if (p_idx==0) {
			PropertyInfo pi;
			pi.type=(call_mode==CALL_MODE_INSTANCE?Variant::OBJECT:basic_type);
			pi.name=(call_mode==CALL_MODE_INSTANCE?String("instance"):Variant::get_type_name(basic_type).to_lower());
			return pi;
		} else {
			p_idx--;
		}
	}

#ifdef DEBUG_METHODS_ENABLED

	if (call_mode==CALL_MODE_BASIC_TYPE) {


		Vector<StringName> names = Variant::get_method_argument_names(basic_type,function);
		Vector<Variant::Type> types = Variant::get_method_argument_types(basic_type,function);
		return PropertyInfo(types[p_idx],names[p_idx]);

	} else {

		MethodBind *mb = ObjectTypeDB::get_method(_get_base_type(),function);
		if (!mb)
			return PropertyInfo();

		return mb->get_argument_info(p_idx);
	}
#else
	return PropertyInfo();
#endif

}

PropertyInfo VisualScriptFunctionCall::get_output_value_port_info(int p_idx) const{


#ifdef DEBUG_METHODS_ENABLED

	if (call_mode==CALL_MODE_BASIC_TYPE) {


		return PropertyInfo(Variant::get_method_return_type(basic_type,function),"");
	} else {

		MethodBind *mb = ObjectTypeDB::get_method(_get_base_type(),function);
		if (!mb)
			return PropertyInfo();

		PropertyInfo pi = mb->get_argument_info(-1);
		pi.name="";
		return pi;
	}
#else
	return PropertyInfo();
#endif
}


String VisualScriptFunctionCall::get_caption() const {

	static const char*cname[4]= {
		"CallSelf",
		"CallNode",
		"CallInstance",
		"CallBasic"
	};

	return cname[call_mode];
}

String VisualScriptFunctionCall::get_text() const {

	if (call_mode==CALL_MODE_SELF)
		return "  "+String(function)+"()";
	else if (call_mode==CALL_MODE_BASIC_TYPE)
		return Variant::get_type_name(basic_type)+"."+String(function)+"()";
	else
		return "  "+base_type+"."+String(function)+"()";

}

void VisualScriptFunctionCall::_update_defargs() {

	//save base type if accessible

	if (call_mode==CALL_MODE_NODE_PATH) {

		Node* node=_get_base_node();
		if (node) {
			base_type=node->get_type();
		}
	} else if (call_mode==CALL_MODE_SELF) {

		if (get_visual_script().is_valid()) {
			base_type=get_visual_script()->get_instance_base_type();
		}
	}


	if (call_mode==CALL_MODE_BASIC_TYPE) {
		use_default_args = Variant::get_method_default_arguments(basic_type,function).size();
	} else {
		if (!get_visual_script().is_valid())
			return; //do not change if not valid yet

		MethodBind *mb = ObjectTypeDB::get_method(_get_base_type(),function);
		if (!mb)
			return;

		use_default_args=mb->get_default_argument_count();
	}

}

void VisualScriptFunctionCall::set_basic_type(Variant::Type p_type) {

	if (basic_type==p_type)
		return;
	basic_type=p_type;

	_update_defargs();
	_change_notify();
	ports_changed_notify();
}

Variant::Type VisualScriptFunctionCall::get_basic_type() const{

	return basic_type;
}

void VisualScriptFunctionCall::set_base_type(const StringName& p_type) {

	if (base_type==p_type)
		return;

	base_type=p_type;
	_update_defargs();
	_change_notify();
	ports_changed_notify();
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
	ports_changed_notify();
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
	ports_changed_notify();
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
	ports_changed_notify();

}
VisualScriptFunctionCall::CallMode VisualScriptFunctionCall::get_call_mode() const {

	return call_mode;
}

void VisualScriptFunctionCall::set_use_default_args(int p_amount) {

	if (use_default_args==p_amount)
		return;

	use_default_args=p_amount;
	ports_changed_notify();


}

int VisualScriptFunctionCall::get_use_default_args() const{

	return use_default_args;
}
void VisualScriptFunctionCall::_validate_property(PropertyInfo& property) const {

	if (property.name=="function/base_type") {
		if (call_mode!=CALL_MODE_INSTANCE) {
			property.usage=PROPERTY_USAGE_NOEDITOR;
		}
	}

	if (property.name=="function/basic_type") {
		if (call_mode!=CALL_MODE_BASIC_TYPE) {
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

		if (call_mode==CALL_MODE_BASIC_TYPE) {

			if (basic_type==Variant::NIL) {
				property.usage=0;
				return; //nothing for nil
			}
			Variant::CallError ce;
			Variant v = Variant::construct(basic_type,NULL,0,ce);
			v.get_method_list(&methods);


		} else {

			StringName base = _get_base_type();
			ObjectTypeDB::get_method_list(base,&methods);


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

	if (property.name=="function/use_default_args") {

		property.hint=PROPERTY_HINT_RANGE;

		int mc=0;

		if (call_mode==CALL_MODE_BASIC_TYPE) {

			mc = Variant::get_method_default_arguments(basic_type,function).size();
		} else {
			MethodBind *mb = ObjectTypeDB::get_method(_get_base_type(),function);
			if (mb) {

				mc=mb->get_default_argument_count();
			}
		}

		if (mc==0) {
			property.usage=0; //do not show
		} else {

			property.hint_string="0,"+itos(mc)+",1";
		}
	}
}


void VisualScriptFunctionCall::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_base_type","base_type"),&VisualScriptFunctionCall::set_base_type);
	ObjectTypeDB::bind_method(_MD("get_base_type"),&VisualScriptFunctionCall::get_base_type);

	ObjectTypeDB::bind_method(_MD("set_basic_type","basic_type"),&VisualScriptFunctionCall::set_basic_type);
	ObjectTypeDB::bind_method(_MD("get_basic_type"),&VisualScriptFunctionCall::get_basic_type);

	ObjectTypeDB::bind_method(_MD("set_function","function"),&VisualScriptFunctionCall::set_function);
	ObjectTypeDB::bind_method(_MD("get_function"),&VisualScriptFunctionCall::get_function);

	ObjectTypeDB::bind_method(_MD("set_call_mode","mode"),&VisualScriptFunctionCall::set_call_mode);
	ObjectTypeDB::bind_method(_MD("get_call_mode"),&VisualScriptFunctionCall::get_call_mode);

	ObjectTypeDB::bind_method(_MD("set_base_path","base_path"),&VisualScriptFunctionCall::set_base_path);
	ObjectTypeDB::bind_method(_MD("get_base_path"),&VisualScriptFunctionCall::get_base_path);

	ObjectTypeDB::bind_method(_MD("set_use_default_args","amount"),&VisualScriptFunctionCall::set_use_default_args);
	ObjectTypeDB::bind_method(_MD("get_use_default_args"),&VisualScriptFunctionCall::get_use_default_args);


	String bt;
	for(int i=0;i<Variant::VARIANT_MAX;i++) {
		if (i>0)
			bt+=",";

		bt+=Variant::get_type_name(Variant::Type(i));
	}

	ADD_PROPERTY(PropertyInfo(Variant::INT,"function/call_mode",PROPERTY_HINT_ENUM,"Self,Node Path,Instance,Basic Type",PROPERTY_USAGE_NOEDITOR),_SCS("set_call_mode"),_SCS("get_call_mode"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"function/base_type",PROPERTY_HINT_TYPE_STRING,"Object"),_SCS("set_base_type"),_SCS("get_base_type"));
	ADD_PROPERTY(PropertyInfo(Variant::INT,"function/basic_type",PROPERTY_HINT_ENUM,bt),_SCS("set_basic_type"),_SCS("get_basic_type"));
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,"function/node_path",PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE),_SCS("set_base_path"),_SCS("get_base_path"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"function/function"),_SCS("set_function"),_SCS("get_function"));
	ADD_PROPERTY(PropertyInfo(Variant::INT,"function/use_default_args"),_SCS("set_use_default_args"),_SCS("get_use_default_args"));

	BIND_CONSTANT( CALL_MODE_SELF );
	BIND_CONSTANT( CALL_MODE_NODE_PATH);
	BIND_CONSTANT( CALL_MODE_INSTANCE);
	BIND_CONSTANT( CALL_MODE_BASIC_TYPE );
}

class VisualScriptNodeInstanceFunctionCall : public VisualScriptNodeInstance {
public:


	VisualScriptFunctionCall::CallMode call_mode;
	NodePath node_path;
	int input_args;
	bool returns;
	StringName function;

	VisualScriptFunctionCall *node;
	VisualScriptInstance *instance;



	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {


		switch(call_mode) {

			case VisualScriptFunctionCall::CALL_MODE_SELF: {

				Object *object=instance->get_owner_ptr();

				if (returns) {
					*p_outputs[0] = object->call(function,p_inputs,input_args,r_error);
				} else {
					object->call(function,p_inputs,input_args,r_error);
				}
			} break;
			case VisualScriptFunctionCall::CALL_MODE_NODE_PATH: {

				Node* node = instance->get_owner_ptr()->cast_to<Node>();
				if (!node) {
					r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str="Base object is not a Node!";
					return 0;
				}

				Node* another = node->get_node(node_path);
				if (!node) {
					r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str="Path does not lead Node!";
					return 0;
				}

				if (returns) {
					*p_outputs[0] = another->call(function,p_inputs,input_args,r_error);
				} else {
					another->call(function,p_inputs,input_args,r_error);
				}

			} break;
			case VisualScriptFunctionCall::CALL_MODE_INSTANCE:
			case VisualScriptFunctionCall::CALL_MODE_BASIC_TYPE: {

				Variant v = *p_inputs[0];

				if (returns) {
					*p_outputs[0] = v.call(function,p_inputs+1,input_args,r_error);
				} else {
					v.call(function,p_inputs+1,input_args,r_error);
				}

			} break;

		}
		return 0;

	}


};

VisualScriptNodeInstance* VisualScriptFunctionCall::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceFunctionCall * instance = memnew(VisualScriptNodeInstanceFunctionCall );
	instance->node=this;
	instance->instance=p_instance;
	instance->function=function;
	instance->call_mode=call_mode;
	instance->returns=get_output_value_port_count();
	instance->node_path=base_path;
	instance->input_args = get_input_value_port_count() - ( (call_mode==CALL_MODE_BASIC_TYPE || call_mode==CALL_MODE_INSTANCE) ? 1: 0 );
	return instance;
}
VisualScriptFunctionCall::VisualScriptFunctionCall() {

	call_mode=CALL_MODE_INSTANCE;
	basic_type=Variant::NIL;
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

static const char* event_type_names[InputEvent::TYPE_MAX]={
	"None",
	"Key",
	"MouseMotion",
	"MouseButton",
	"JoystickMotion",
	"JoystickButton",
	"ScreenTouch",
	"ScreenDrag",
	"Action"
};


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

	int pc = (call_mode==CALL_MODE_BASIC_TYPE || call_mode==CALL_MODE_INSTANCE)?1:0;

	if (!use_builtin_value)
		pc++;

	return pc;
}
int VisualScriptPropertySet::get_output_value_port_count() const{

	return call_mode==CALL_MODE_BASIC_TYPE? 1 : 0;
}

String VisualScriptPropertySet::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptPropertySet::get_input_value_port_info(int p_idx) const{

	if (call_mode==CALL_MODE_INSTANCE || call_mode==CALL_MODE_BASIC_TYPE) {
		if (p_idx==0) {
			PropertyInfo pi;
			pi.type=(call_mode==CALL_MODE_INSTANCE?Variant::OBJECT:basic_type);
			pi.name=(call_mode==CALL_MODE_INSTANCE?String("instance"):Variant::get_type_name(basic_type).to_lower());
			return pi;
		} else {
			p_idx--;
		}
	}

#ifdef DEBUG_METHODS_ENABLED

	//not very efficient but..


	List<PropertyInfo> pinfo;

	if (call_mode==CALL_MODE_BASIC_TYPE) {


		Variant v;
		if (basic_type==Variant::INPUT_EVENT) {
			InputEvent ev;
			ev.type=event_type;
			v=ev;
		} else {
			Variant::CallError ce;
			v = Variant::construct(basic_type,NULL,0,ce);
		}
		v.get_property_list(&pinfo);

	} else if (call_mode==CALL_MODE_NODE_PATH) {

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
	if (call_mode==CALL_MODE_BASIC_TYPE) {
		return PropertyInfo(basic_type,"out");
	} else {
		return PropertyInfo();
	}

}


String VisualScriptPropertySet::get_caption() const {

	static const char*cname[4]= {
		"SelfSet",
		"NodeSet",
		"InstanceSet",
		"BasicSet"
	};

	return cname[call_mode];
}

String VisualScriptPropertySet::get_text() const {

	String prop;

	if (call_mode==CALL_MODE_BASIC_TYPE)
		prop=Variant::get_type_name(basic_type)+"."+property;
	else
		prop=property;

	if (use_builtin_value) {
		String bit = builtin_value.get_construct_string();
		if (bit.length()>40) {
			bit=bit.substr(0,40);
			bit+="...";
		}

		prop+="\n  "+bit;
	}

	return prop;

}

void VisualScriptPropertySet::_update_base_type() {
	//cache it because this information may not be available on load
	if (call_mode==CALL_MODE_NODE_PATH) {

		Node* node=_get_base_node();
		if (node) {
			base_type=node->get_type();
		}
	} else if (call_mode==CALL_MODE_SELF) {

		if (get_visual_script().is_valid()) {
			base_type=get_visual_script()->get_instance_base_type();
		}
	}

}
void VisualScriptPropertySet::set_basic_type(Variant::Type p_type) {

	if (basic_type==p_type)
		return;
	basic_type=p_type;


	_change_notify();
	_update_base_type();
	ports_changed_notify();
}

Variant::Type VisualScriptPropertySet::get_basic_type() const{

	return basic_type;
}

void VisualScriptPropertySet::set_event_type(InputEvent::Type p_type) {

	if (event_type==p_type)
		return;
	event_type=p_type;
	_change_notify();
	_update_base_type();
	ports_changed_notify();
}

InputEvent::Type VisualScriptPropertySet::get_event_type() const{

	return event_type;
}


void VisualScriptPropertySet::set_base_type(const StringName& p_type) {

	if (base_type==p_type)
		return;

	base_type=p_type;
	_change_notify();	
	ports_changed_notify();
}

StringName VisualScriptPropertySet::get_base_type() const{

	return base_type;
}

void VisualScriptPropertySet::set_property(const StringName& p_type){

	if (property==p_type)
		return;

	property=p_type;
	_change_notify();	
	ports_changed_notify();
}
StringName VisualScriptPropertySet::get_property() const {


	return property;
}

void VisualScriptPropertySet::set_base_path(const NodePath& p_type) {

	if (base_path==p_type)
		return;

	base_path=p_type;
	_update_base_type();
	_change_notify();
	ports_changed_notify();
}

NodePath VisualScriptPropertySet::get_base_path() const {

	return base_path;
}


void VisualScriptPropertySet::set_call_mode(CallMode p_mode) {

	if (call_mode==p_mode)
		return;

	call_mode=p_mode;
	_update_base_type();
	_change_notify();
	ports_changed_notify();

}
VisualScriptPropertySet::CallMode VisualScriptPropertySet::get_call_mode() const {

	return call_mode;
}


void VisualScriptPropertySet::set_use_builtin_value(bool p_use) {

	if (use_builtin_value==p_use)
		return;

	use_builtin_value=p_use;
	_change_notify();
	ports_changed_notify();

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
			property.usage=PROPERTY_USAGE_NOEDITOR;
		}
	}


	if (property.name=="property/basic_type") {
		if (call_mode!=CALL_MODE_BASIC_TYPE) {
			property.usage=0;
		}
	}

	if (property.name=="property/event_type") {
		if (call_mode!=CALL_MODE_BASIC_TYPE || basic_type!=Variant::INPUT_EVENT) {
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


		if (call_mode==CALL_MODE_BASIC_TYPE) {
			Variant::CallError ce;
			Variant v;
			if (basic_type==Variant::INPUT_EVENT) {
				InputEvent ev;
				ev.type=event_type;
				v=ev;
			} else {
				v = Variant::construct(basic_type,NULL,0,ce);
			}
			v.get_property_list(&pinfo);

		} else if (call_mode==CALL_MODE_NODE_PATH) {

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

			if (E->get().usage&PROPERTY_USAGE_EDITOR) {
				mstring.push_back(E->get().name);
			}
		}

		String ml;
		for (List<String>::Element *E=mstring.front();E;E=E->next()) {

			if (ml!=String())
				ml+=",";
			ml+=E->get();
		}

		if (ml==String()) {
			property.usage=PROPERTY_USAGE_NOEDITOR; //do not show for editing if empty
		} else {
			property.hint_string=ml;
		}
	}

	if (property.name=="value/builtin") {

		if (!use_builtin_value) {
			property.usage=0;
		} else {
			List<PropertyInfo> pinfo;

			if (call_mode==CALL_MODE_BASIC_TYPE) {
				Variant::CallError ce;
				Variant v = Variant::construct(basic_type,NULL,0,ce);
				v.get_property_list(&pinfo);

			} else if (call_mode==CALL_MODE_NODE_PATH) {

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


	ObjectTypeDB::bind_method(_MD("set_basic_type","basic_type"),&VisualScriptPropertySet::set_basic_type);
	ObjectTypeDB::bind_method(_MD("get_basic_type"),&VisualScriptPropertySet::get_basic_type);

	ObjectTypeDB::bind_method(_MD("set_event_type","event_type"),&VisualScriptPropertySet::set_event_type);
	ObjectTypeDB::bind_method(_MD("get_event_type"),&VisualScriptPropertySet::get_event_type);

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

	String bt;
	for(int i=0;i<Variant::VARIANT_MAX;i++) {
		if (i>0)
			bt+=",";

		bt+=Variant::get_type_name(Variant::Type(i));
	}

	String et;
	for(int i=0;i<InputEvent::TYPE_MAX;i++) {
		if (i>0)
			et+=",";

		et+=event_type_names[i];
	}


	ADD_PROPERTY(PropertyInfo(Variant::INT,"property/set_mode",PROPERTY_HINT_ENUM,"Self,Node Path,Instance,Basic Type",PROPERTY_USAGE_NOEDITOR),_SCS("set_call_mode"),_SCS("get_call_mode"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"property/base_type",PROPERTY_HINT_TYPE_STRING,"Object"),_SCS("set_base_type"),_SCS("get_base_type"));
	ADD_PROPERTY(PropertyInfo(Variant::INT,"property/basic_type",PROPERTY_HINT_ENUM,bt),_SCS("set_basic_type"),_SCS("get_basic_type"));
	ADD_PROPERTY(PropertyInfo(Variant::INT,"property/event_type",PROPERTY_HINT_ENUM,et),_SCS("set_event_type"),_SCS("get_event_type"));
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,"property/node_path",PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE),_SCS("set_base_path"),_SCS("get_base_path"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"property/property"),_SCS("set_property"),_SCS("get_property"));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"value/use_builtin"),_SCS("set_use_builtin_value"),_SCS("is_using_builtin_value"));
	ADD_PROPERTY(PropertyInfo(Variant::NIL,"value/builtin"),_SCS("set_builtin_value"),_SCS("get_builtin_value"));

	BIND_CONSTANT( CALL_MODE_SELF );
	BIND_CONSTANT( CALL_MODE_NODE_PATH);
	BIND_CONSTANT( CALL_MODE_INSTANCE);

}

class VisualScriptNodeInstancePropertySet : public VisualScriptNodeInstance {
public:


	VisualScriptPropertySet::CallMode call_mode;
	NodePath node_path;
	StringName property;
	bool use_builtin;
	Variant builtin_val;

	VisualScriptPropertySet *node;
	VisualScriptInstance *instance;



	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {


		switch(call_mode) {

			case VisualScriptPropertySet::CALL_MODE_SELF: {

				Object *object=instance->get_owner_ptr();

				bool valid;

				if (use_builtin) {
					object->set(property,builtin_val,&valid);
				} else {
					object->set(property,*p_inputs[0],&valid);
				}

				if (!valid) {
					r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str="Invalid index property name.";
				}
			} break;
			case VisualScriptPropertySet::CALL_MODE_NODE_PATH: {

				Node* node = instance->get_owner_ptr()->cast_to<Node>();
				if (!node) {
					r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str="Base object is not a Node!";
					return 0;
				}

				Node* another = node->get_node(node_path);
				if (!node) {
					r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str="Path does not lead Node!";
					return 0;
				}

				bool valid;

				if (use_builtin) {
					another->set(property,builtin_val,&valid);
				} else {
					another->set(property,*p_inputs[0],&valid);
				}

				if (!valid) {
					r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str="Invalid index property name.";
				}

			} break;
			case VisualScriptPropertySet::CALL_MODE_INSTANCE:
			case VisualScriptPropertySet::CALL_MODE_BASIC_TYPE: {

				Variant v = *p_inputs[0];

				bool valid;

				if (use_builtin) {
					v.set(property,builtin_val,&valid);
				} else {
					v.set(property,p_inputs[1],&valid);
				}

				if (!valid) {
					r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str="Invalid index property name.";
				}

				if (call_mode==VisualScriptPropertySet::CALL_MODE_BASIC_TYPE) {
					*p_outputs[0]=v;
				}

			} break;

		}
		return 0;

	}


};

VisualScriptNodeInstance* VisualScriptPropertySet::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstancePropertySet * instance = memnew(VisualScriptNodeInstancePropertySet );
	instance->node=this;
	instance->instance=p_instance;
	instance->property=property;
	instance->call_mode=call_mode;
	instance->node_path=base_path;
	instance->use_builtin=use_builtin_value;
	instance->builtin_val=builtin_value;
	return instance;
}

VisualScriptPropertySet::VisualScriptPropertySet() {

	call_mode=CALL_MODE_INSTANCE;
	base_type="Object";
	basic_type=Variant::NIL;
	event_type=InputEvent::NONE;

}

template<VisualScriptPropertySet::CallMode cmode>
static Ref<VisualScriptNode> create_property_set_node(const String& p_name) {

	Ref<VisualScriptPropertySet> node;
	node.instance();
	node->set_call_mode(cmode);
	return node;
}


//////////////////////////////////////////
////////////////GET//////////////////////
//////////////////////////////////////////

int VisualScriptPropertyGet::get_output_sequence_port_count() const {

	return (call_mode==CALL_MODE_SELF || call_mode==CALL_MODE_NODE_PATH)?0:1;
}

bool VisualScriptPropertyGet::has_input_sequence_port() const{

	return (call_mode==CALL_MODE_SELF || call_mode==CALL_MODE_NODE_PATH)?false:true;
}
void VisualScriptPropertyGet::_update_base_type() {
	//cache it because this information may not be available on load
	if (call_mode==CALL_MODE_NODE_PATH) {

		Node* node=_get_base_node();
		if (node) {
			base_type=node->get_type();
		}
	} else if (call_mode==CALL_MODE_SELF) {

		if (get_visual_script().is_valid()) {
			base_type=get_visual_script()->get_instance_base_type();
		}
	}

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

	return (call_mode==CALL_MODE_BASIC_TYPE || call_mode==CALL_MODE_INSTANCE)?1:0;

}
int VisualScriptPropertyGet::get_output_value_port_count() const{

	return 1;
}

String VisualScriptPropertyGet::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptPropertyGet::get_input_value_port_info(int p_idx) const{

	if (call_mode==CALL_MODE_INSTANCE || call_mode==CALL_MODE_BASIC_TYPE) {
		if (p_idx==0) {
			PropertyInfo pi;
			pi.type=(call_mode==CALL_MODE_INSTANCE?Variant::OBJECT:basic_type);
			pi.name=(call_mode==CALL_MODE_INSTANCE?String("instance"):Variant::get_type_name(basic_type).to_lower());
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

	if (call_mode==CALL_MODE_BASIC_TYPE) {


		Variant v;
		if (basic_type==Variant::INPUT_EVENT) {
			InputEvent ev;
			ev.type=event_type;
			v=ev;
		} else {
			Variant::CallError ce;
			v = Variant::construct(basic_type,NULL,0,ce);
		}
		v.get_property_list(&pinfo);
	} else if (call_mode==CALL_MODE_NODE_PATH) {

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

	static const char*cname[4]= {
		"SelfGet",
		"NodeGet",
		"InstanceGet",
		"BasicGet"
	};

	return cname[call_mode];
}

String VisualScriptPropertyGet::get_text() const {


	if (call_mode==CALL_MODE_BASIC_TYPE)
		return Variant::get_type_name(basic_type)+"."+property;
	else
		return property;

}

void VisualScriptPropertyGet::set_base_type(const StringName& p_type) {

	if (base_type==p_type)
		return;

	base_type=p_type;	
	_change_notify();
	ports_changed_notify();
}

StringName VisualScriptPropertyGet::get_base_type() const{

	return base_type;
}

void VisualScriptPropertyGet::set_property(const StringName& p_type){

	if (property==p_type)
		return;

	property=p_type;
	_change_notify();
	ports_changed_notify();
}
StringName VisualScriptPropertyGet::get_property() const {


	return property;
}

void VisualScriptPropertyGet::set_base_path(const NodePath& p_type) {

	if (base_path==p_type)
		return;

	base_path=p_type;
	_change_notify();
	_update_base_type();
	ports_changed_notify();
}

NodePath VisualScriptPropertyGet::get_base_path() const {

	return base_path;
}


void VisualScriptPropertyGet::set_call_mode(CallMode p_mode) {

	if (call_mode==p_mode)
		return;

	call_mode=p_mode;
	_change_notify();
	_update_base_type();
	ports_changed_notify();

}
VisualScriptPropertyGet::CallMode VisualScriptPropertyGet::get_call_mode() const {

	return call_mode;
}



void VisualScriptPropertyGet::set_basic_type(Variant::Type p_type) {

	if (basic_type==p_type)
		return;
	basic_type=p_type;


	_change_notify();
	ports_changed_notify();
}

Variant::Type VisualScriptPropertyGet::get_basic_type() const{

	return basic_type;
}


void VisualScriptPropertyGet::set_event_type(InputEvent::Type p_type) {

	if (event_type==p_type)
		return;
	event_type=p_type;
	_change_notify();
	_update_base_type();
	ports_changed_notify();
}

InputEvent::Type VisualScriptPropertyGet::get_event_type() const{

	return event_type;
}

void VisualScriptPropertyGet::_validate_property(PropertyInfo& property) const {

	if (property.name=="property/base_type") {
		if (call_mode!=CALL_MODE_INSTANCE) {
			property.usage=PROPERTY_USAGE_NOEDITOR;
		}
	}


	if (property.name=="property/basic_type") {
		if (call_mode!=CALL_MODE_BASIC_TYPE) {
			property.usage=0;
		}
	}
	if (property.name=="property/event_type") {
		if (call_mode!=CALL_MODE_BASIC_TYPE || basic_type!=Variant::INPUT_EVENT) {
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

		if (call_mode==CALL_MODE_BASIC_TYPE) {
			Variant::CallError ce;
			Variant v;
			if (basic_type==Variant::INPUT_EVENT) {
				InputEvent ev;
				ev.type=event_type;
				v=ev;
			} else {
				v = Variant::construct(basic_type,NULL,0,ce);
			}
			v.get_property_list(&pinfo);

		} else if (call_mode==CALL_MODE_NODE_PATH) {

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

		if (ml==String()) {
			property.usage=PROPERTY_USAGE_NOEDITOR; //do not show for editing if empty
		} else {
			property.hint_string=ml;
		}

	}

}

void VisualScriptPropertyGet::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_base_type","base_type"),&VisualScriptPropertyGet::set_base_type);
	ObjectTypeDB::bind_method(_MD("get_base_type"),&VisualScriptPropertyGet::get_base_type);


	ObjectTypeDB::bind_method(_MD("set_basic_type","basic_type"),&VisualScriptPropertyGet::set_basic_type);
	ObjectTypeDB::bind_method(_MD("get_basic_type"),&VisualScriptPropertyGet::get_basic_type);

	ObjectTypeDB::bind_method(_MD("set_event_type","event_type"),&VisualScriptPropertyGet::set_event_type);
	ObjectTypeDB::bind_method(_MD("get_event_type"),&VisualScriptPropertyGet::get_event_type);


	ObjectTypeDB::bind_method(_MD("set_property","property"),&VisualScriptPropertyGet::set_property);
	ObjectTypeDB::bind_method(_MD("get_property"),&VisualScriptPropertyGet::get_property);

	ObjectTypeDB::bind_method(_MD("set_call_mode","mode"),&VisualScriptPropertyGet::set_call_mode);
	ObjectTypeDB::bind_method(_MD("get_call_mode"),&VisualScriptPropertyGet::get_call_mode);

	ObjectTypeDB::bind_method(_MD("set_base_path","base_path"),&VisualScriptPropertyGet::set_base_path);
	ObjectTypeDB::bind_method(_MD("get_base_path"),&VisualScriptPropertyGet::get_base_path);

	String bt;
	for(int i=0;i<Variant::VARIANT_MAX;i++) {
		if (i>0)
			bt+=",";

		bt+=Variant::get_type_name(Variant::Type(i));
	}

	String et;
	for(int i=0;i<InputEvent::TYPE_MAX;i++) {
		if (i>0)
			et+=",";

		et+=event_type_names[i];
	}


	ADD_PROPERTY(PropertyInfo(Variant::INT,"property/set_mode",PROPERTY_HINT_ENUM,"Self,Node Path,Instance",PROPERTY_USAGE_NOEDITOR),_SCS("set_call_mode"),_SCS("get_call_mode"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"property/base_type",PROPERTY_HINT_TYPE_STRING,"Object"),_SCS("set_base_type"),_SCS("get_base_type"));
	ADD_PROPERTY(PropertyInfo(Variant::INT,"property/basic_type",PROPERTY_HINT_ENUM,bt),_SCS("set_basic_type"),_SCS("get_basic_type"));
	ADD_PROPERTY(PropertyInfo(Variant::INT,"property/event_type",PROPERTY_HINT_ENUM,et),_SCS("set_event_type"),_SCS("get_event_type"));
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,"property/node_path",PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE),_SCS("set_base_path"),_SCS("get_base_path"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"property/property"),_SCS("set_property"),_SCS("get_property"));

	BIND_CONSTANT( CALL_MODE_SELF );
	BIND_CONSTANT( CALL_MODE_NODE_PATH);
	BIND_CONSTANT( CALL_MODE_INSTANCE);
}

class VisualScriptNodeInstancePropertyGet : public VisualScriptNodeInstance {
public:


	VisualScriptPropertyGet::CallMode call_mode;
	NodePath node_path;
	StringName property;

	VisualScriptPropertyGet *node;
	VisualScriptInstance *instance;



	//virtual int get_working_memory_size() const { return 0; }
	virtual bool is_output_port_unsequenced(int p_idx) const { return (call_mode==VisualScriptPropertyGet::CALL_MODE_SELF || call_mode==VisualScriptPropertyGet::CALL_MODE_NODE_PATH); }
	virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const {

		//these two modes can be get directly, so they use unsequenced mode
		switch(call_mode) {

			case VisualScriptPropertyGet::CALL_MODE_SELF: {

				Object *object=instance->get_owner_ptr();

				bool valid;

				*r_value = object->get(property,&valid);

				if (!valid) {
					//r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error=RTR("Invalid index property name.");
					return false;
				}
			} break;
			case VisualScriptPropertyGet::CALL_MODE_NODE_PATH: {

				Node* node = instance->get_owner_ptr()->cast_to<Node>();
				if (!node) {
					//r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error=RTR("Base object is not a Node!");
					return false;
				}

				Node* another = node->get_node(node_path);
				if (!node) {
					//r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error=RTR("Path does not lead Node!");
					return false;
				}

				bool valid;


				*r_value = another->get(property,&valid);

				if (!valid) {
					//r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error=vformat(RTR("Invalid index property name '%s' in node %s."),String(property),another->get_name());
					return false;
				}

			} break;
			default: {};
		}
		return true;

	}

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {


		bool valid;
		Variant v = *p_inputs[0];

		*p_outputs[0] = v.get(property,&valid);

		if (!valid) {
			r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
			r_error_str=RTR("Invalid index property name.");

		}


		return 0;
	}




};

VisualScriptNodeInstance* VisualScriptPropertyGet::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstancePropertyGet * instance = memnew(VisualScriptNodeInstancePropertyGet );
	instance->node=this;
	instance->instance=p_instance;
	instance->property=property;
	instance->call_mode=call_mode;
	instance->node_path=base_path;

	return instance;
}

VisualScriptPropertyGet::VisualScriptPropertyGet() {

	call_mode=CALL_MODE_INSTANCE;
	base_type="Object";
	basic_type=Variant::NIL;
	event_type=InputEvent::NONE;

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

#if 1
	return argument_count;
#else
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
#endif

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

void VisualScriptScriptCall::_update_argument_count() {

	//try to remember the amount of arguments in the function, because if loaded from scratch
	//this information will not be available

	if (call_mode==CALL_MODE_SELF) {

		Ref<VisualScript> vs = get_visual_script();
		if (vs.is_valid()) {

			if (!vs->has_function(function))
				return ;

			int id = vs->get_function_node_id(function);
			if (id<0)
				return;

			Ref<VisualScriptFunction> func = vs->get_node(function,id);

			argument_count=func->get_argument_count();
		}
	} else {

		Node*base = _get_base_node();
		if (!base)
			return;

		Ref<Script> script = base->get_script();
		if (!script.is_valid())
			return ;

		List<MethodInfo> functions;
		script->get_method_list(&functions);
		for (List<MethodInfo>::Element *E=functions.front();E;E=E->next()) {
			if (E->get().name==function) {
				argument_count=E->get().arguments.size();
				return;
			}
		}

	}
}


void VisualScriptScriptCall::set_function(const StringName& p_type){

	if (function==p_type)
		return;

	function=p_type;
	_update_argument_count();
	_change_notify();
	ports_changed_notify();
}
StringName VisualScriptScriptCall::get_function() const {


	return function;
}

void VisualScriptScriptCall::set_base_path(const NodePath& p_type) {

	if (base_path==p_type)
		return;

	base_path=p_type;
	_update_argument_count();
	_change_notify();
	ports_changed_notify();
}

NodePath VisualScriptScriptCall::get_base_path() const {

	return base_path;
}


void VisualScriptScriptCall::set_call_mode(CallMode p_mode) {

	if (call_mode==p_mode)
		return;

	call_mode=p_mode;
	_update_argument_count();
	_change_notify();
	ports_changed_notify();

}

void VisualScriptScriptCall::set_argument_count(int p_count) {

	argument_count=p_count;
	_change_notify();
	ports_changed_notify();

}

int VisualScriptScriptCall::get_argument_count() const {

	return argument_count;
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

	ObjectTypeDB::bind_method(_MD("set_argument_count","argument_count"),&VisualScriptScriptCall::set_argument_count);
	ObjectTypeDB::bind_method(_MD("get_argument_count"),&VisualScriptScriptCall::get_argument_count);

	ADD_PROPERTY(PropertyInfo(Variant::INT,"function/call_mode",PROPERTY_HINT_ENUM,"Self,Node Path"),_SCS("set_call_mode"),_SCS("get_call_mode"));
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,"function/node_path",PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE),_SCS("set_base_path"),_SCS("get_base_path"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"function/function"),_SCS("set_function"),_SCS("get_function"));
	ADD_PROPERTY(PropertyInfo(Variant::STRING,"function/argument_count"),_SCS("set_argument_count"),_SCS("get_argument_count"));

	BIND_CONSTANT( CALL_MODE_SELF );
	BIND_CONSTANT( CALL_MODE_NODE_PATH);

}

class VisualScriptNodeInstanceScriptCall : public VisualScriptNodeInstance {
public:


	VisualScriptScriptCall::CallMode call_mode;
	NodePath node_path;
	int input_args;
	bool returns;
	StringName function;

	VisualScriptScriptCall *node;
	VisualScriptInstance *instance;



	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {


		switch(call_mode) {

			case VisualScriptScriptCall::CALL_MODE_SELF: {

				Object *object=instance->get_owner_ptr();

				*p_outputs[0] = object->call(function,p_inputs,input_args,r_error);

			} break;
			case VisualScriptScriptCall::CALL_MODE_NODE_PATH: {

				Node* node = instance->get_owner_ptr()->cast_to<Node>();
				if (!node) {
					r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str="Base object is not a Node!";
					return 0;
				}

				Node* another = node->get_node(node_path);
				if (!node) {
					r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str="Path does not lead Node!";
					return 0;
				}


				*p_outputs[0] = another->call(function,p_inputs,input_args,r_error);

			} break;

		}
		return 0;

	}


};

VisualScriptNodeInstance* VisualScriptScriptCall::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceScriptCall * instance = memnew(VisualScriptNodeInstanceScriptCall );
	instance->node=this;
	instance->instance=p_instance;
	instance->function=function;
	instance->call_mode=call_mode;
	instance->node_path=base_path;
	instance->input_args = argument_count;
	return instance;
}

VisualScriptScriptCall::VisualScriptScriptCall() {

	call_mode=CALL_MODE_SELF;
	argument_count=0;


}

template<VisualScriptScriptCall::CallMode cmode>
static Ref<VisualScriptNode> create_script_call_node(const String& p_name) {

	Ref<VisualScriptScriptCall> node;
	node.instance();
	node->set_call_mode(cmode);
	return node;
}


//////////////////////////////////////////
////////////////EMIT//////////////////////
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
	ports_changed_notify();
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

class VisualScriptNodeInstanceEmitSignal : public VisualScriptNodeInstance {
public:

	VisualScriptEmitSignal *node;
	VisualScriptInstance *instance;
	int argcount;
	StringName name;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {


		Object *obj = instance->get_owner_ptr();

		obj->emit_signal(name,p_inputs,argcount);


		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptEmitSignal::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceEmitSignal * instance = memnew(VisualScriptNodeInstanceEmitSignal );
	instance->node=this;
	instance->instance=p_instance;
	instance->name=name;
	instance->argcount=get_input_value_port_count();
	return instance;
}

VisualScriptEmitSignal::VisualScriptEmitSignal() {
}



static Ref<VisualScriptNode> create_basic_type_call_node(const String& p_name) {

	Vector<String> path = p_name.split("/");
	ERR_FAIL_COND_V(path.size()<4,Ref<VisualScriptNode>());
	String base_type = path[2];
	String method = path[3];

	Ref<VisualScriptFunctionCall> node;
	node.instance();

	Variant::Type type=Variant::VARIANT_MAX;

	for(int i=0;i<Variant::VARIANT_MAX;i++) {

		if (Variant::get_type_name(Variant::Type(i))==base_type) {
			type=Variant::Type(i);
			break;
		}
	}

	ERR_FAIL_COND_V(type==Variant::VARIANT_MAX,Ref<VisualScriptNode>());


	node->set_call_mode(VisualScriptFunctionCall::CALL_MODE_BASIC_TYPE);
	node->set_basic_type(type);
	node->set_function(method);

	return node;
}


void register_visual_script_func_nodes() {

	VisualScriptLanguage::singleton->add_register_func("functions/call_method/call_instance",create_function_call_node<VisualScriptFunctionCall::CALL_MODE_INSTANCE>);
	VisualScriptLanguage::singleton->add_register_func("functions/call_method/call_basic_type",create_function_call_node<VisualScriptFunctionCall::CALL_MODE_BASIC_TYPE>);
	VisualScriptLanguage::singleton->add_register_func("functions/call_method/call_self",create_function_call_node<VisualScriptFunctionCall::CALL_MODE_SELF>);
	VisualScriptLanguage::singleton->add_register_func("functions/call_method/call_node",create_function_call_node<VisualScriptFunctionCall::CALL_MODE_NODE_PATH>);

	VisualScriptLanguage::singleton->add_register_func("functions/property_set/instace_set",create_property_set_node<VisualScriptPropertySet::CALL_MODE_INSTANCE>);
	VisualScriptLanguage::singleton->add_register_func("functions/property_set/basic_type_set",create_property_set_node<VisualScriptPropertySet::CALL_MODE_BASIC_TYPE>);
	VisualScriptLanguage::singleton->add_register_func("functions/property_set/self_set",create_property_set_node<VisualScriptPropertySet::CALL_MODE_SELF>);
	VisualScriptLanguage::singleton->add_register_func("functions/property_set/node_set",create_property_set_node<VisualScriptPropertySet::CALL_MODE_NODE_PATH>);

	VisualScriptLanguage::singleton->add_register_func("functions/property_get/instance_get",create_property_get_node<VisualScriptPropertyGet::CALL_MODE_INSTANCE>);
	VisualScriptLanguage::singleton->add_register_func("functions/property_get/basic_type_get",create_property_get_node<VisualScriptPropertyGet::CALL_MODE_BASIC_TYPE>);
	VisualScriptLanguage::singleton->add_register_func("functions/property_get/self_get",create_property_get_node<VisualScriptPropertyGet::CALL_MODE_SELF>);
	VisualScriptLanguage::singleton->add_register_func("functions/property_get/node_get",create_property_get_node<VisualScriptPropertyGet::CALL_MODE_NODE_PATH>);

	VisualScriptLanguage::singleton->add_register_func("functions/call_script/call_self",create_script_call_node<VisualScriptScriptCall::CALL_MODE_SELF>);
	VisualScriptLanguage::singleton->add_register_func("functions/call_script/call_node",create_script_call_node<VisualScriptScriptCall::CALL_MODE_NODE_PATH>);
	VisualScriptLanguage::singleton->add_register_func("functions/call_script/emit_signal",create_node_generic<VisualScriptEmitSignal>);


	for(int i=0;i<Variant::VARIANT_MAX;i++) {

		Variant::Type t = Variant::Type(i);
		String type_name = Variant::get_type_name(t);
		Variant::CallError ce;
		Variant vt = Variant::construct(t,NULL,0,ce);
		List<MethodInfo> ml;
		vt.get_method_list(&ml);

		for (List<MethodInfo>::Element *E=ml.front();E;E=E->next()) {
			VisualScriptLanguage::singleton->add_register_func("functions/by_type/"+type_name+"/"+E->get().name,create_basic_type_call_node);
		}
	}
}
