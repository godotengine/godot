#include "visual_script_nodes.h"
#include "global_constants.h"
#include "globals.h"
#include "scene/main/scene_main_loop.h"
#include "os/os.h"
#include "scene/main/node.h"

//////////////////////////////////////////
////////////////FUNCTION//////////////////
//////////////////////////////////////////


bool  VisualScriptFunction::_set(const StringName& p_name, const Variant& p_value) {


	if (p_name=="argument_count") {

		int new_argc=p_value;
		int argc = arguments.size();
		if (argc==new_argc)
			return true;

		arguments.resize(new_argc);

		for(int i=argc;i<new_argc;i++) {
			arguments[i].name="arg"+itos(i+1);
			arguments[i].type=Variant::NIL;
		}
		ports_changed_notify();
		_change_notify();
		return true;
	}
	if (String(p_name).begins_with("argument/")) {
		int idx = String(p_name).get_slice("/",1).to_int()-1;
		ERR_FAIL_INDEX_V(idx,arguments.size(),false);
		String what = String(p_name).get_slice("/",2);
		if (what=="type") {

			Variant::Type new_type = Variant::Type(int(p_value));
			arguments[idx].type=new_type;
			ports_changed_notify();

			return true;
		}

		if (what=="name") {

			arguments[idx].name=p_value;
			ports_changed_notify();
			return true;
		}


	}

	if (p_name=="stack/stackless") {
		set_stack_less(p_value);
		return true;
	}

	if (p_name=="stack/size") {
		stack_size=p_value;
		return true;
	}
	return false;
}

bool  VisualScriptFunction::_get(const StringName& p_name,Variant &r_ret) const {


	if (p_name=="argument_count") {
		r_ret = arguments.size();
		return true;
	}
	if (String(p_name).begins_with("argument/")) {
		int idx = String(p_name).get_slice("/",1).to_int()-1;
		ERR_FAIL_INDEX_V(idx,arguments.size(),false);
		String what = String(p_name).get_slice("/",2);
		if (what=="type") {
			r_ret = arguments[idx].type;
			return true;
		}
		if (what=="name") {
			r_ret = arguments[idx].name;
			return true;
		}



	}

	if (p_name=="stack/stackless") {
		r_ret=stack_less;
		return true;
	}

	if (p_name=="stack/size") {
		r_ret=stack_size;
		return true;
	}

	return false;
}
void  VisualScriptFunction::_get_property_list( List<PropertyInfo> *p_list) const {


	p_list->push_back(PropertyInfo(Variant::INT,"argument_count",PROPERTY_HINT_RANGE,"0,256"));
	String argt="Variant";
	for(int i=1;i<Variant::VARIANT_MAX;i++) {
		argt+=","+Variant::get_type_name(Variant::Type(i));
	}

	for(int i=0;i<arguments.size();i++) {
		p_list->push_back(PropertyInfo(Variant::INT,"argument/"+itos(i+1)+"/type",PROPERTY_HINT_ENUM,argt));
		p_list->push_back(PropertyInfo(Variant::STRING,"argument/"+itos(i+1)+"/name"));
	}
	if (!stack_less) {
		p_list->push_back(PropertyInfo(Variant::INT,"stack/size",PROPERTY_HINT_RANGE,"1,100000"));
	}
	p_list->push_back(PropertyInfo(Variant::BOOL,"stack/stackless"));

}


int VisualScriptFunction::get_output_sequence_port_count() const {

	return 1;
}

bool VisualScriptFunction::has_input_sequence_port() const{

	return false;
}

int VisualScriptFunction::get_input_value_port_count() const{

	return 0;
}
int VisualScriptFunction::get_output_value_port_count() const{

	return arguments.size();
}

String VisualScriptFunction::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptFunction::get_input_value_port_info(int p_idx) const{

	ERR_FAIL_V(PropertyInfo());
	return PropertyInfo();
}
PropertyInfo VisualScriptFunction::get_output_value_port_info(int p_idx) const{

	ERR_FAIL_INDEX_V(p_idx,arguments.size(),PropertyInfo());
	PropertyInfo out;
	out.type=arguments[p_idx].type;
	out.name=arguments[p_idx].name;
	return out;
}

String VisualScriptFunction::get_caption() const {

	return "Function";
}

String VisualScriptFunction::get_text() const {

	return get_name(); //use name as function name I guess
}

void VisualScriptFunction::add_argument(Variant::Type p_type,const String& p_name,int p_index){

	Argument arg;
	arg.name=p_name;
	arg.type=p_type;
	if (p_index>=0)
		arguments.insert(p_index,arg);
	else
		arguments.push_back(arg);

	ports_changed_notify();

}
void VisualScriptFunction::set_argument_type(int p_argidx,Variant::Type p_type){

	ERR_FAIL_INDEX(p_argidx,arguments.size());

	arguments[p_argidx].type=p_type;
	ports_changed_notify();
}
Variant::Type VisualScriptFunction::get_argument_type(int p_argidx) const {

	ERR_FAIL_INDEX_V(p_argidx,arguments.size(),Variant::NIL);
	return arguments[p_argidx].type;

}
void VisualScriptFunction::set_argument_name(int p_argidx,const String& p_name) {

	ERR_FAIL_INDEX(p_argidx,arguments.size());

	arguments[p_argidx].name=p_name;
	ports_changed_notify();

}
String VisualScriptFunction::get_argument_name(int p_argidx) const {

	ERR_FAIL_INDEX_V(p_argidx,arguments.size(),String());
	return arguments[p_argidx].name;

}
void VisualScriptFunction::remove_argument(int p_argidx) {

	ERR_FAIL_INDEX(p_argidx,arguments.size());

	arguments.remove(p_argidx);
	ports_changed_notify();

}

int VisualScriptFunction::get_argument_count() const {

	return arguments.size();
}

class VisualScriptNodeInstanceFunction : public VisualScriptNodeInstance {
public:

	VisualScriptFunction *node;
	VisualScriptInstance *instance;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		int ac = node->get_argument_count();

		for(int i=0;i<ac;i++) {
#ifdef DEBUG_ENABLED
			Variant::Type expected = node->get_argument_type(i);
			if (expected!=Variant::NIL) {
				if (!Variant::can_convert_strict(p_inputs[i]->get_type(),expected)) {
					r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
					r_error.expected=expected;
					r_error.argument=i;
					return 0;
				}
			}
#endif

			*p_outputs[i]=*p_inputs[i];
		}

		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptFunction::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceFunction * instance = memnew(VisualScriptNodeInstanceFunction );
	instance->node=this;
	instance->instance=p_instance;
	return instance;
}

VisualScriptFunction::VisualScriptFunction() {

	stack_size=256;
	stack_less=false;
}


void VisualScriptFunction::set_stack_less(bool p_enable) {
	stack_less=p_enable;
	_change_notify();
}

bool VisualScriptFunction::is_stack_less() const {
	return stack_less;
}

void VisualScriptFunction::set_stack_size(int p_size) {

	ERR_FAIL_COND(p_size <1 || p_size>100000);
	stack_size=p_size;
}

int VisualScriptFunction::get_stack_size() const {

	return stack_size;
}


//////////////////////////////////////////
////////////////OPERATOR//////////////////
//////////////////////////////////////////

int VisualScriptOperator::get_output_sequence_port_count() const {

	return 1;
}

bool VisualScriptOperator::has_input_sequence_port() const{

	return true;
}

int VisualScriptOperator::get_input_value_port_count() const{

	return (op==Variant::OP_BIT_NEGATE || op==Variant::OP_NOT || op==Variant::OP_NEGATE) ? 1 : 2;
}
int VisualScriptOperator::get_output_value_port_count() const{

	return 1;
}

String VisualScriptOperator::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptOperator::get_input_value_port_info(int p_idx) const{

	static const Variant::Type port_types[Variant::OP_MAX][2]={
		{Variant::NIL,Variant::NIL}, //OP_EQUAL,
		{Variant::NIL,Variant::NIL}, //OP_NOT_EQUAL,
		{Variant::NIL,Variant::NIL}, //OP_LESS,
		{Variant::NIL,Variant::NIL}, //OP_LESS_EQUAL,
		{Variant::NIL,Variant::NIL}, //OP_GREATER,
		{Variant::NIL,Variant::NIL}, //OP_GREATER_EQUAL,
		//mathematic
		{Variant::NIL,Variant::NIL}, //OP_ADD,
		{Variant::NIL,Variant::NIL}, //OP_SUBSTRACT,
		{Variant::NIL,Variant::NIL}, //OP_MULTIPLY,
		{Variant::NIL,Variant::NIL}, //OP_DIVIDE,
		{Variant::NIL,Variant::NIL}, //OP_NEGATE,
		{Variant::INT,Variant::INT}, //OP_MODULE,
		{Variant::STRING,Variant::STRING}, //OP_STRING_CONCAT,
		//bitwise
		{Variant::INT,Variant::INT}, //OP_SHIFT_LEFT,
		{Variant::INT,Variant::INT}, //OP_SHIFT_RIGHT,
		{Variant::INT,Variant::INT}, //OP_BIT_AND,
		{Variant::INT,Variant::INT}, //OP_BIT_OR,
		{Variant::INT,Variant::INT}, //OP_BIT_XOR,
		{Variant::INT,Variant::INT}, //OP_BIT_NEGATE,
		//logic
		{Variant::BOOL,Variant::BOOL}, //OP_AND,
		{Variant::BOOL,Variant::BOOL}, //OP_OR,
		{Variant::BOOL,Variant::BOOL}, //OP_XOR,
		{Variant::BOOL,Variant::BOOL}, //OP_NOT,
		//containment
		{Variant::NIL,Variant::NIL} //OP_IN,
	};

	ERR_FAIL_INDEX_V(p_idx,Variant::OP_MAX,PropertyInfo());

	PropertyInfo pinfo;
	pinfo.name=p_idx==0?"A":"B";
	pinfo.type=port_types[op][p_idx];
	return pinfo;
}
PropertyInfo VisualScriptOperator::get_output_value_port_info(int p_idx) const{
	static const Variant::Type port_types[Variant::OP_MAX]={
		//comparation
		Variant::BOOL, //OP_EQUAL,
		Variant::BOOL, //OP_NOT_EQUAL,
		Variant::BOOL, //OP_LESS,
		Variant::BOOL, //OP_LESS_EQUAL,
		Variant::BOOL, //OP_GREATER,
		Variant::BOOL, //OP_GREATER_EQUAL,
		//mathematic
		Variant::NIL, //OP_ADD,
		Variant::NIL, //OP_SUBSTRACT,
		Variant::NIL, //OP_MULTIPLY,
		Variant::NIL, //OP_DIVIDE,
		Variant::NIL, //OP_NEGATE,
		Variant::INT, //OP_MODULE,
		Variant::STRING, //OP_STRING_CONCAT,
		//bitwise
		Variant::INT, //OP_SHIFT_LEFT,
		Variant::INT, //OP_SHIFT_RIGHT,
		Variant::INT, //OP_BIT_AND,
		Variant::INT, //OP_BIT_OR,
		Variant::INT, //OP_BIT_XOR,
		Variant::INT, //OP_BIT_NEGATE,
		//logic
		Variant::BOOL, //OP_AND,
		Variant::BOOL, //OP_OR,
		Variant::BOOL, //OP_XOR,
		Variant::BOOL, //OP_NOT,
		//containment
		Variant::BOOL //OP_IN,
	};

	PropertyInfo pinfo;
	pinfo.name="";
	pinfo.type=port_types[op];
	return pinfo;

}

static const char* op_names[]={
	//comparation
	"Equal", //OP_EQUAL,
	"NotEqual", //OP_NOT_EQUAL,
	"Less", //OP_LESS,
	"LessEqual", //OP_LESS_EQUAL,
	"Greater", //OP_GREATER,
	"GreaterEq", //OP_GREATER_EQUAL,
	//mathematic
	"Add", //OP_ADD,
	"Subtract", //OP_SUBSTRACT,
	"Multiply", //OP_MULTIPLY,
	"Divide", //OP_DIVIDE,
	"Negate", //OP_NEGATE,
	"Remainder", //OP_MODULE,
	"Concat", //OP_STRING_CONCAT,
	//bitwise
	"ShiftLeft", //OP_SHIFT_LEFT,
	"ShiftRight", //OP_SHIFT_RIGHT,
	"BitAnd", //OP_BIT_AND,
	"BitOr", //OP_BIT_OR,
	"BitXor", //OP_BIT_XOR,
	"BitNeg", //OP_BIT_NEGATE,
	//logic
	"And", //OP_AND,
	"Or", //OP_OR,
	"Xor", //OP_XOR,
	"Not", //OP_NOT,
	//containment
	"In", //OP_IN,
};

String VisualScriptOperator::get_caption() const {



	return op_names[op];
}

String VisualScriptOperator::get_text() const {

	static const wchar_t* op_names[]={
		//comparation
		L"A = B", //OP_EQUAL,
		L"A \u2260 B", //OP_NOT_EQUAL,
		L"A < B", //OP_LESS,
		L"A \u2264 B", //OP_LESS_EQUAL,
		L"A > B", //OP_GREATER,
		L"A \u2265 B", //OP_GREATER_EQUAL,
		//mathematic
		L"A + B", //OP_ADD,
		L"A - B", //OP_SUBSTRACT,
		L"A x B", //OP_MULTIPLY,
		L"A \u00F7 B", //OP_DIVIDE,
		L"\u00AC A", //OP_NEGATE,
		L"A mod B", //OP_MODULE,
		L"A .. B", //OP_STRING_CONCAT,
		//bitwise
		L"A << B", //OP_SHIFT_LEFT,
		L"A >> B", //OP_SHIFT_RIGHT,
		L"A & B", //OP_BIT_AND,
		L"A | B", //OP_BIT_OR,
		L"A ^ B", //OP_BIT_XOR,
		L"~A", //OP_BIT_NEGATE,
		//logic
		L"A and B", //OP_AND,
		L"A or B", //OP_OR,
		L"A xor B", //OP_XOR,
		L"not A", //OP_NOT,

	};
	return op_names[op];
}

void VisualScriptOperator::set_operator(Variant::Operator p_op) {

	if (op==p_op)
		return;
	op=p_op;
	ports_changed_notify();

}

Variant::Operator VisualScriptOperator::get_operator() const{

	return op;
}


void VisualScriptOperator::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_operator","op"),&VisualScriptOperator::set_operator);
	ObjectTypeDB::bind_method(_MD("get_operator"),&VisualScriptOperator::get_operator);

	String types;
	for(int i=0;i<Variant::OP_MAX;i++) {
		if (i>0)
			types+=",";
		types+=op_names[i];
	}
	ADD_PROPERTY(PropertyInfo(Variant::INT,"operator_value/type",PROPERTY_HINT_ENUM,types),_SCS("set_operator"),_SCS("get_operator"));

}

class VisualScriptNodeInstanceOperator : public VisualScriptNodeInstance {
public:

	bool unary;
	Variant::Operator op;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		bool valid;
		if (unary) {

			Variant::evaluate(op,*p_inputs[0],Variant(),*p_outputs[0],valid);
		} else {
			Variant::evaluate(op,*p_inputs[0],*p_inputs[1],*p_outputs[0],valid);
		}

		if (!valid) {

			r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
			if (p_outputs[0]->get_type()==Variant::STRING) {
				r_error_str=*p_outputs[0];
			} else {
				if (unary)
					r_error_str=String(op_names[op])+RTR(": Invalid argument of type: ")+Variant::get_type_name(p_inputs[0]->get_type());
				else
					r_error_str=String(op_names[op])+RTR(": Invalid arguments: ")+"A: "+Variant::get_type_name(p_inputs[0]->get_type())+"  B: "+Variant::get_type_name(p_inputs[1]->get_type());
			}
		}

		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptOperator::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceOperator * instance = memnew(VisualScriptNodeInstanceOperator );
	instance->unary=get_input_value_port_count()==1;
	instance->op=op;
	return instance;
}

VisualScriptOperator::VisualScriptOperator() {

	op=Variant::OP_ADD;
}



template<Variant::Operator OP>
static Ref<VisualScriptNode> create_op_node(const String& p_name) {

	Ref<VisualScriptOperator> node;
	node.instance();
	node->set_operator(OP);
	return node;
}

//////////////////////////////////////////
////////////////VARIABLE GET//////////////////
//////////////////////////////////////////

int VisualScriptVariableGet::get_output_sequence_port_count() const {

	return 0;
}

bool VisualScriptVariableGet::has_input_sequence_port() const{

	return false;
}

int VisualScriptVariableGet::get_input_value_port_count() const{

	return 0;
}
int VisualScriptVariableGet::get_output_value_port_count() const{

	return 1;
}

String VisualScriptVariableGet::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptVariableGet::get_input_value_port_info(int p_idx) const{

	return PropertyInfo();
}

PropertyInfo VisualScriptVariableGet::get_output_value_port_info(int p_idx) const{

	PropertyInfo pinfo;
	pinfo.name="value";
	if (get_visual_script().is_valid() && get_visual_script()->has_variable(variable)) {
		PropertyInfo vinfo = get_visual_script()->get_variable_info(variable);
		pinfo.type=vinfo.type;
		pinfo.hint=vinfo.hint;
		pinfo.hint_string=vinfo.hint_string;
	}
	return pinfo;
}


String VisualScriptVariableGet::get_caption() const {

	return "Variable";
}

String VisualScriptVariableGet::get_text() const {

	return variable;
}

void VisualScriptVariableGet::set_variable(StringName p_variable) {

	if (variable==p_variable)
		return;
	variable=p_variable;
	ports_changed_notify();

}

StringName VisualScriptVariableGet::get_variable() const{

	return variable;
}

void VisualScriptVariableGet::_validate_property(PropertyInfo& property) const {

	if (property.name=="variable/name" && get_visual_script().is_valid()) {
		Ref<VisualScript> vs = get_visual_script();
		List<StringName> vars;
		vs->get_variable_list(&vars);

		String vhint;
		for (List<StringName>::Element *E=vars.front();E;E=E->next()) {
			if (vhint!=String())
				vhint+=",";

			vhint+=E->get().operator String();
		}

		property.hint=PROPERTY_HINT_ENUM;
		property.hint_string=vhint;
	}
}

void VisualScriptVariableGet::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_variable","name"),&VisualScriptVariableGet::set_variable);
	ObjectTypeDB::bind_method(_MD("get_variable"),&VisualScriptVariableGet::get_variable);


	ADD_PROPERTY(PropertyInfo(Variant::STRING,"variable/name"),_SCS("set_variable"),_SCS("get_variable"));

}

class VisualScriptNodeInstanceVariableGet : public VisualScriptNodeInstance {
public:

	VisualScriptVariableGet *node;
	VisualScriptInstance *instance;
	StringName variable;

	//virtual int get_working_memory_size() const { return 0; }
	virtual bool is_output_port_unsequenced(int p_idx) const { return true; }
	virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const {

		 if (instance->get_variable(variable,r_value)==false) {
			 r_error=RTR("VariableGet not found in script: ")+"'"+String(variable)+"'";
			 return false;
		 } else {
			 return true;
		 }
	}

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptVariableGet::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceVariableGet * instance = memnew(VisualScriptNodeInstanceVariableGet );
	instance->node=this;
	instance->instance=p_instance;
	instance->variable=variable;
	return instance;
}
VisualScriptVariableGet::VisualScriptVariableGet() {


}


//////////////////////////////////////////
////////////////VARIABLE GET//////////////////
//////////////////////////////////////////

int VisualScriptVariableSet::get_output_sequence_port_count() const {

	return 1;
}

bool VisualScriptVariableSet::has_input_sequence_port() const{

	return true;
}

int VisualScriptVariableSet::get_input_value_port_count() const{

	return 1;
}
int VisualScriptVariableSet::get_output_value_port_count() const{

	return 0;
}

String VisualScriptVariableSet::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptVariableSet::get_input_value_port_info(int p_idx) const{

	PropertyInfo pinfo;
	pinfo.name="set";
	if (get_visual_script().is_valid() && get_visual_script()->has_variable(variable)) {
		PropertyInfo vinfo = get_visual_script()->get_variable_info(variable);
		pinfo.type=vinfo.type;
		pinfo.hint=vinfo.hint;
		pinfo.hint_string=vinfo.hint_string;
	}
	return pinfo;
}

PropertyInfo VisualScriptVariableSet::get_output_value_port_info(int p_idx) const{

	return PropertyInfo();
}


String VisualScriptVariableSet::get_caption() const {

	return "VariableSet";
}

String VisualScriptVariableSet::get_text() const {

	return variable;
}

void VisualScriptVariableSet::set_variable(StringName p_variable) {

	if (variable==p_variable)
		return;
	variable=p_variable;
	ports_changed_notify();

}

StringName VisualScriptVariableSet::get_variable() const{

	return variable;
}

void VisualScriptVariableSet::_validate_property(PropertyInfo& property) const {

	if (property.name=="variable/name" && get_visual_script().is_valid()) {
		Ref<VisualScript> vs = get_visual_script();
		List<StringName> vars;
		vs->get_variable_list(&vars);

		String vhint;
		for (List<StringName>::Element *E=vars.front();E;E=E->next()) {
			if (vhint!=String())
				vhint+=",";

			vhint+=E->get().operator String();
		}

		property.hint=PROPERTY_HINT_ENUM;
		property.hint_string=vhint;
	}
}

void VisualScriptVariableSet::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_variable","name"),&VisualScriptVariableSet::set_variable);
	ObjectTypeDB::bind_method(_MD("get_variable"),&VisualScriptVariableSet::get_variable);


	ADD_PROPERTY(PropertyInfo(Variant::STRING,"variable/name"),_SCS("set_variable"),_SCS("get_variable"));

}

class VisualScriptNodeInstanceVariableSet : public VisualScriptNodeInstance {
public:

	VisualScriptVariableSet *node;
	VisualScriptInstance *instance;
	StringName variable;

	//virtual int get_working_memory_size() const { return 0; }
	virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return false; }

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		if (instance->set_variable(variable,*p_inputs[0])==false) {
			r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD			;
			r_error_str=RTR("VariableSet not found in script: ")+"'"+String(variable)+"'";
		}


		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptVariableSet::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceVariableSet * instance = memnew(VisualScriptNodeInstanceVariableSet );
	instance->node=this;
	instance->instance=p_instance;
	instance->variable=variable;
	return instance;
}
VisualScriptVariableSet::VisualScriptVariableSet() {


}



//////////////////////////////////////////
////////////////CONSTANT//////////////////
//////////////////////////////////////////

int VisualScriptConstant::get_output_sequence_port_count() const {

	return 0;
}

bool VisualScriptConstant::has_input_sequence_port() const{

	return false;
}

int VisualScriptConstant::get_input_value_port_count() const{

	return 0;
}
int VisualScriptConstant::get_output_value_port_count() const{

	return 1;
}

String VisualScriptConstant::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptConstant::get_input_value_port_info(int p_idx) const{

	return PropertyInfo();
}

PropertyInfo VisualScriptConstant::get_output_value_port_info(int p_idx) const{

	PropertyInfo pinfo;
	pinfo.name="get";
	pinfo.type=type;
	return pinfo;
}


String VisualScriptConstant::get_caption() const {

	return "Constant";
}

String VisualScriptConstant::get_text() const {

	return String(value);
}

void VisualScriptConstant::set_constant_type(Variant::Type p_type) {

	if (type==p_type)
		return;

	type=p_type;
	ports_changed_notify();
	Variant::CallError ce;
	value=Variant::construct(type,NULL,0,ce);
	_change_notify();

}

Variant::Type VisualScriptConstant::get_constant_type() const{

	return type;
}

void VisualScriptConstant::set_constant_value(Variant p_value){

	if (value==p_value)
		return;

	value=p_value;
	ports_changed_notify();
}
Variant VisualScriptConstant::get_constant_value() const{

	return value;
}

void VisualScriptConstant::_validate_property(PropertyInfo& property) const {


	if (property.name=="constant/value") {
		property.type=type;
		if (type==Variant::NIL)
			property.usage=0; //do not save if nil
	}
}

void VisualScriptConstant::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_constant_type","type"),&VisualScriptConstant::set_constant_type);
	ObjectTypeDB::bind_method(_MD("get_constant_type"),&VisualScriptConstant::get_constant_type);

	ObjectTypeDB::bind_method(_MD("set_constant_value","value"),&VisualScriptConstant::set_constant_value);
	ObjectTypeDB::bind_method(_MD("get_constant_value"),&VisualScriptConstant::get_constant_value);

	String argt="Null";
	for(int i=1;i<Variant::VARIANT_MAX;i++) {
		argt+=","+Variant::get_type_name(Variant::Type(i));
	}


	ADD_PROPERTY(PropertyInfo(Variant::INT,"constant/type",PROPERTY_HINT_ENUM,argt),_SCS("set_constant_type"),_SCS("get_constant_type"));
	ADD_PROPERTY(PropertyInfo(Variant::NIL,"constant/value"),_SCS("set_constant_value"),_SCS("get_constant_value"));

}

class VisualScriptNodeInstanceConstant : public VisualScriptNodeInstance {
public:

	Variant constant;
	//virtual int get_working_memory_size() const { return 0; }
	virtual bool is_output_port_unsequenced(int p_idx) const { return true; }
	virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const {

		*r_value=constant;

		return true;

	}

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptConstant::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceConstant * instance = memnew(VisualScriptNodeInstanceConstant );
	instance->constant=value;
	return instance;
}

VisualScriptConstant::VisualScriptConstant() {

	type=Variant::NIL;

}



//////////////////////////////////////////
////////////////INDEX////////////////////
//////////////////////////////////////////

int VisualScriptIndexGet::get_output_sequence_port_count() const {

	return 1;
}

bool VisualScriptIndexGet::has_input_sequence_port() const{

	return true;
}

int VisualScriptIndexGet::get_input_value_port_count() const{

	return 2;
}
int VisualScriptIndexGet::get_output_value_port_count() const{

	return 1;
}

String VisualScriptIndexGet::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptIndexGet::get_input_value_port_info(int p_idx) const{

	if (p_idx==0) {
		return PropertyInfo(Variant::NIL,"base");
	} else {
		return PropertyInfo(Variant::NIL,"index");

	}
}

PropertyInfo VisualScriptIndexGet::get_output_value_port_info(int p_idx) const{

	return PropertyInfo();
}


String VisualScriptIndexGet::get_caption() const {

	return "IndexGet";
}

String VisualScriptIndexGet::get_text() const {

	return String("get");
}


class VisualScriptNodeInstanceIndexGet : public VisualScriptNodeInstance {
public:


	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return true; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return false; }

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		bool valid;
		*p_outputs[0] = p_inputs[0]->get(*p_inputs[1],&valid);

		if (!valid) {
			r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
			r_error_str="Invalid get: "+p_inputs[0]->get_construct_string();
		}
		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptIndexGet::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceIndexGet * instance = memnew(VisualScriptNodeInstanceIndexGet );
	return instance;
}
VisualScriptIndexGet::VisualScriptIndexGet() {



}

//////////////////////////////////////////
////////////////INDEXSET//////////////////
//////////////////////////////////////////

int VisualScriptIndexSet::get_output_sequence_port_count() const {

	return 1;
}

bool VisualScriptIndexSet::has_input_sequence_port() const{

	return true;
}

int VisualScriptIndexSet::get_input_value_port_count() const{

	return 3;
}
int VisualScriptIndexSet::get_output_value_port_count() const{

	return 0;
}

String VisualScriptIndexSet::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptIndexSet::get_input_value_port_info(int p_idx) const{

	if (p_idx==0) {
		return PropertyInfo(Variant::NIL,"base");
	} else if (p_idx==1){
		return PropertyInfo(Variant::NIL,"index");

	} else {
		return PropertyInfo(Variant::NIL,"value");

	}
}

PropertyInfo VisualScriptIndexSet::get_output_value_port_info(int p_idx) const{

	return PropertyInfo();
}


String VisualScriptIndexSet::get_caption() const {

	return "IndexSet";
}

String VisualScriptIndexSet::get_text() const {

	return String("set");
}


class VisualScriptNodeInstanceIndexSet : public VisualScriptNodeInstance {
public:


	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return true; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return false; }

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		bool valid;
		*p_outputs[0]=*p_inputs[0];
		p_outputs[0]->set(*p_inputs[1],*p_inputs[2],&valid);

		if (!valid) {
			r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
			r_error_str="Invalid set: "+p_inputs[1]->get_construct_string();
		}
		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptIndexSet::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceIndexSet * instance = memnew(VisualScriptNodeInstanceIndexSet );
	return instance;
}
VisualScriptIndexSet::VisualScriptIndexSet() {



}


//////////////////////////////////////////
////////////////GLOBALCONSTANT///////////
//////////////////////////////////////////

int VisualScriptGlobalConstant::get_output_sequence_port_count() const {

	return 0;
}

bool VisualScriptGlobalConstant::has_input_sequence_port() const{

	return false;
}

int VisualScriptGlobalConstant::get_input_value_port_count() const{

	return 0;
}
int VisualScriptGlobalConstant::get_output_value_port_count() const{

	return 1;
}

String VisualScriptGlobalConstant::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptGlobalConstant::get_input_value_port_info(int p_idx) const{

	return PropertyInfo();
}

PropertyInfo VisualScriptGlobalConstant::get_output_value_port_info(int p_idx) const{

	return PropertyInfo(Variant::REAL,"value");
}


String VisualScriptGlobalConstant::get_caption() const {

	return "GlobalConst";
}

String VisualScriptGlobalConstant::get_text() const {

	return GlobalConstants::get_global_constant_name(index);
}

void VisualScriptGlobalConstant::set_global_constant(int p_which) {

	index=p_which;
	_change_notify();
	ports_changed_notify();
}

int VisualScriptGlobalConstant::get_global_constant() {
	return index;
}


class VisualScriptNodeInstanceGlobalConstant : public VisualScriptNodeInstance {
public:

	int index;
	//virtual int get_working_memory_size() const { return 0; }
	virtual bool is_output_port_unsequenced(int p_idx) const { return true; }
	virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const {

		*r_value = GlobalConstants::get_global_constant_value(index);
		return true;

	}

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {


		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptGlobalConstant::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceGlobalConstant * instance = memnew(VisualScriptNodeInstanceGlobalConstant );
	instance->index=index;
	return instance;
}

void VisualScriptGlobalConstant::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_global_constant","index"),&VisualScriptGlobalConstant::set_global_constant);
	ObjectTypeDB::bind_method(_MD("get_global_constant"),&VisualScriptGlobalConstant::get_global_constant);

	String cc;

	for(int i=0;i<GlobalConstants::get_global_constant_count();i++) {

		if (i>0)
			cc+=",";
		cc+=GlobalConstants::get_global_constant_name(i);
	}
	ADD_PROPERTY(PropertyInfo(Variant::INT,"constant",PROPERTY_HINT_ENUM,cc),_SCS("set_global_constant"),_SCS("get_global_constant"));
}

VisualScriptGlobalConstant::VisualScriptGlobalConstant() {

	index=0;
}



//////////////////////////////////////////
////////////////MATHCONSTANT///////////
//////////////////////////////////////////


const char* VisualScriptMathConstant::const_name[MATH_CONSTANT_MAX]={
	"One",
	"PI",
	"PIx2",
	"PI/2",
	"E",
	"Sqrt2",
};

double VisualScriptMathConstant::const_value[MATH_CONSTANT_MAX]={
	1.0,
	Math_PI,
	Math_PI*2,
	Math_PI*0.5,
	2.71828182845904523536,
	Math::sqrt(2.0)
};


int VisualScriptMathConstant::get_output_sequence_port_count() const {

	return 0;
}

bool VisualScriptMathConstant::has_input_sequence_port() const{

	return false;
}

int VisualScriptMathConstant::get_input_value_port_count() const{

	return 0;
}
int VisualScriptMathConstant::get_output_value_port_count() const{

	return 1;
}

String VisualScriptMathConstant::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptMathConstant::get_input_value_port_info(int p_idx) const{

	return PropertyInfo();
}

PropertyInfo VisualScriptMathConstant::get_output_value_port_info(int p_idx) const{

	return PropertyInfo(Variant::REAL,"value");
}


String VisualScriptMathConstant::get_caption() const {

	return "MathConst";
}

String VisualScriptMathConstant::get_text() const {

	return const_name[constant];
}

void VisualScriptMathConstant::set_math_constant(MathConstant p_which) {

	constant=p_which;
	_change_notify();
	ports_changed_notify();
}

VisualScriptMathConstant::MathConstant VisualScriptMathConstant::get_math_constant() {
	return constant;
}

class VisualScriptNodeInstanceMathConstant : public VisualScriptNodeInstance {
public:

	float value;
	//virtual int get_working_memory_size() const { return 0; }
	virtual bool is_output_port_unsequenced(int p_idx) const { return true; }
	virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const {

		*r_value = value;
		return true;

	}

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {


		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptMathConstant::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceMathConstant * instance = memnew(VisualScriptNodeInstanceMathConstant );
	instance->value=const_value[constant];
	return instance;
}


void VisualScriptMathConstant::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_math_constant","which"),&VisualScriptMathConstant::set_math_constant);
	ObjectTypeDB::bind_method(_MD("get_math_constant"),&VisualScriptMathConstant::get_math_constant);

	String cc;

	for(int i=0;i<MATH_CONSTANT_MAX;i++) {

		if (i>0)
			cc+=",";
		cc+=const_name[i];
	}
	ADD_PROPERTY(PropertyInfo(Variant::INT,"constant",PROPERTY_HINT_ENUM,cc),_SCS("set_math_constant"),_SCS("get_math_constant"));
}

VisualScriptMathConstant::VisualScriptMathConstant() {

	constant=MATH_CONSTANT_ONE;
}



//////////////////////////////////////////
////////////////GLOBALSINGLETON///////////
//////////////////////////////////////////

int VisualScriptEngineSingleton::get_output_sequence_port_count() const {

	return 0;
}

bool VisualScriptEngineSingleton::has_input_sequence_port() const{

	return false;
}

int VisualScriptEngineSingleton::get_input_value_port_count() const{

	return 0;
}
int VisualScriptEngineSingleton::get_output_value_port_count() const{

	return 1;
}

String VisualScriptEngineSingleton::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptEngineSingleton::get_input_value_port_info(int p_idx) const{

	return PropertyInfo();
}

PropertyInfo VisualScriptEngineSingleton::get_output_value_port_info(int p_idx) const{

	return PropertyInfo(Variant::OBJECT,"instance");
}


String VisualScriptEngineSingleton::get_caption() const {

	return "EngineSingleton";
}

String VisualScriptEngineSingleton::get_text() const {

	return singleton;
}

void VisualScriptEngineSingleton::set_singleton(const String& p_string) {

	singleton=p_string;

	_change_notify();
	ports_changed_notify();
}

String VisualScriptEngineSingleton::get_singleton() {
	return singleton;
}



class VisualScriptNodeInstanceEngineSingleton : public VisualScriptNodeInstance {
public:

	Object* singleton;

	//virtual int get_working_memory_size() const { return 0; }
	virtual bool is_output_port_unsequenced(int p_idx) const { return true; }
	virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const {

		*r_value=singleton;
		return true;
	}

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {


		return 0;
	}

};

VisualScriptNodeInstance* VisualScriptEngineSingleton::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceEngineSingleton * instance = memnew(VisualScriptNodeInstanceEngineSingleton );
	instance->singleton=Globals::get_singleton()->get_singleton_object(singleton);
	return instance;
}


void VisualScriptEngineSingleton::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_singleton","name"),&VisualScriptEngineSingleton::set_singleton);
	ObjectTypeDB::bind_method(_MD("get_singleton"),&VisualScriptEngineSingleton::get_singleton);

	String cc;

	List<Globals::Singleton> singletons;

	Globals::get_singleton()->get_singletons(&singletons);

	for (List<Globals::Singleton>::Element *E=singletons.front();E;E=E->next()) {
		if (E->get().name=="VS" || E->get().name=="PS" || E->get().name=="PS2D" || E->get().name=="AS" || E->get().name=="TS" || E->get().name=="SS" || E->get().name=="SS2D")
			continue; //skip these, too simple named

		if (cc!=String())
			cc+=",";
		cc+=E->get().name;
	}

	ADD_PROPERTY(PropertyInfo(Variant::STRING,"constant",PROPERTY_HINT_ENUM,cc),_SCS("set_singleton"),_SCS("get_singleton"));
}

VisualScriptEngineSingleton::VisualScriptEngineSingleton() {

	singleton=String();
}



//////////////////////////////////////////
////////////////GETNODE///////////
//////////////////////////////////////////

int VisualScriptSceneNode::get_output_sequence_port_count() const {

	return 0;
}

bool VisualScriptSceneNode::has_input_sequence_port() const{

	return false;
}

int VisualScriptSceneNode::get_input_value_port_count() const{

	return 0;
}
int VisualScriptSceneNode::get_output_value_port_count() const{

	return 1;
}

String VisualScriptSceneNode::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptSceneNode::get_input_value_port_info(int p_idx) const{

	return PropertyInfo();
}

PropertyInfo VisualScriptSceneNode::get_output_value_port_info(int p_idx) const{

	return PropertyInfo(Variant::OBJECT,"node");
}


String VisualScriptSceneNode::get_caption() const {

	return "SceneNode";
}

String VisualScriptSceneNode::get_text() const {

	return path.simplified();
}

void VisualScriptSceneNode::set_node_path(const NodePath& p_path) {

	path=p_path;
	_change_notify();
	ports_changed_notify();
}

NodePath VisualScriptSceneNode::get_node_path() {
	return path;
}


class VisualScriptNodeInstanceSceneNode : public VisualScriptNodeInstance {
public:

	VisualScriptSceneNode *node;
	VisualScriptInstance *instance;
	NodePath path;

	//virtual int get_working_memory_size() const { return 0; }
	virtual bool is_output_port_unsequenced(int p_idx) const { return true; }
	virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const {

		Node* node = instance->get_owner_ptr()->cast_to<Node>();
		if (!node) {
			r_error="Base object is not a Node!";
			return false;
		}



		Node* another = node->get_node(path);
		if (!node) {
			r_error="Path does not lead Node!";
			return false;
		}

		*r_value=another;
		return true;
	}

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptSceneNode::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceSceneNode * instance = memnew(VisualScriptNodeInstanceSceneNode );
	instance->node=this;
	instance->instance=p_instance;
	instance->path=path;
	return instance;
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

void VisualScriptSceneNode::_validate_property(PropertyInfo& property) const {

#ifdef TOOLS_ENABLED
	if (property.name=="node_path") {

		Ref<Script> script = get_visual_script();
		if (!script.is_valid())
			return;

		MainLoop * main_loop = OS::get_singleton()->get_main_loop();
		if (!main_loop)
			return;

		SceneTree *scene_tree = main_loop->cast_to<SceneTree>();

		if (!scene_tree)
			return;

		Node *edited_scene = scene_tree->get_edited_scene_root();

		if (!edited_scene)
			return;

		Node* script_node = _find_script_node(edited_scene,edited_scene,script);

		if (!script_node)
			return;

		property.hint_string=script_node->get_path();
	}
#endif
}

void VisualScriptSceneNode::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_node_path","path"),&VisualScriptSceneNode::set_node_path);
	ObjectTypeDB::bind_method(_MD("get_node_path"),&VisualScriptSceneNode::get_node_path);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,"node_path",PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE),_SCS("set_node_path"),_SCS("get_node_path"));
}

VisualScriptSceneNode::VisualScriptSceneNode() {

	path=String(".");
}


//////////////////////////////////////////
////////////////SceneTree///////////
//////////////////////////////////////////

int VisualScriptSceneTree::get_output_sequence_port_count() const {

	return 0;
}

bool VisualScriptSceneTree::has_input_sequence_port() const{

	return false;
}

int VisualScriptSceneTree::get_input_value_port_count() const{

	return 0;
}
int VisualScriptSceneTree::get_output_value_port_count() const{

	return 1;
}

String VisualScriptSceneTree::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptSceneTree::get_input_value_port_info(int p_idx) const{

	return PropertyInfo();
}

PropertyInfo VisualScriptSceneTree::get_output_value_port_info(int p_idx) const{

	return PropertyInfo(Variant::OBJECT,"instance");
}


String VisualScriptSceneTree::get_caption() const {

	return "SceneTree";
}

String VisualScriptSceneTree::get_text() const {

	return "";
}


class VisualScriptNodeInstanceSceneTree : public VisualScriptNodeInstance {
public:

	VisualScriptSceneTree *node;
	VisualScriptInstance *instance;

	//virtual int get_working_memory_size() const { return 0; }
	virtual bool is_output_port_unsequenced(int p_idx) const { return true; }
	virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const {

		Node* node = instance->get_owner_ptr()->cast_to<Node>();
		if (!node) {
			r_error="Base object is not a Node!";
			return false;
		}

		SceneTree* tree = node->get_tree();
		if (!tree) {
			r_error="Attempt to get SceneTree while node is not in the active tree.";
			return false;
		}

		*r_value=tree;
		return true;

	}

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {


		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptSceneTree::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceSceneTree * instance = memnew(VisualScriptNodeInstanceSceneTree );
	instance->node=this;
	instance->instance=p_instance;
	return instance;
}


void VisualScriptSceneTree::_validate_property(PropertyInfo& property) const {

}

void VisualScriptSceneTree::_bind_methods() {

}

VisualScriptSceneTree::VisualScriptSceneTree() {

}


//////////////////////////////////////////
////////////////RESPATH///////////
//////////////////////////////////////////

int VisualScriptResourcePath::get_output_sequence_port_count() const {

	return 0;
}

bool VisualScriptResourcePath::has_input_sequence_port() const{

	return false;
}

int VisualScriptResourcePath::get_input_value_port_count() const{

	return 0;
}
int VisualScriptResourcePath::get_output_value_port_count() const{

	return 1;
}

String VisualScriptResourcePath::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptResourcePath::get_input_value_port_info(int p_idx) const{

	return PropertyInfo();
}

PropertyInfo VisualScriptResourcePath::get_output_value_port_info(int p_idx) const{

	return PropertyInfo(Variant::STRING,"path");
}


String VisualScriptResourcePath::get_caption() const {

	return "ResourcePath";
}

String VisualScriptResourcePath::get_text() const {

	return path;
}

void VisualScriptResourcePath::set_resource_path(const String& p_path) {

	path=p_path;
	_change_notify();
	ports_changed_notify();
}

String VisualScriptResourcePath::get_resource_path() {
	return path;
}


class VisualScriptNodeInstanceResourcePath : public VisualScriptNodeInstance {
public:

	String path;

	//virtual int get_working_memory_size() const { return 0; }
	virtual bool is_output_port_unsequenced(int p_idx) const { return true; }
	virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const {
		*r_value = path;
		return true;
	}

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptResourcePath::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceResourcePath * instance = memnew(VisualScriptNodeInstanceResourcePath );
	instance->path=path;
	return instance;
}



void VisualScriptResourcePath::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_resource_path","path"),&VisualScriptResourcePath::set_resource_path);
	ObjectTypeDB::bind_method(_MD("get_resource_path"),&VisualScriptResourcePath::get_resource_path);

	ADD_PROPERTY(PropertyInfo(Variant::STRING,"path",PROPERTY_HINT_FILE),_SCS("set_resource_path"),_SCS("get_resource_path"));
}

VisualScriptResourcePath::VisualScriptResourcePath() {

	path="";
}



//////////////////////////////////////////
////////////////SELF///////////
//////////////////////////////////////////

int VisualScriptSelf::get_output_sequence_port_count() const {

	return 0;
}

bool VisualScriptSelf::has_input_sequence_port() const{

	return false;
}

int VisualScriptSelf::get_input_value_port_count() const{

	return 0;
}
int VisualScriptSelf::get_output_value_port_count() const{

	return 1;
}

String VisualScriptSelf::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptSelf::get_input_value_port_info(int p_idx) const{

	return PropertyInfo();
}

PropertyInfo VisualScriptSelf::get_output_value_port_info(int p_idx) const{

	return PropertyInfo(Variant::OBJECT,"instance");
}


String VisualScriptSelf::get_caption() const {

	return "Self";
}

String VisualScriptSelf::get_text() const {

	if (get_visual_script().is_valid())
		return get_visual_script()->get_instance_base_type();
	else
		return "";
}


class VisualScriptNodeInstanceSelf : public VisualScriptNodeInstance {
public:

	VisualScriptInstance* instance;

	//virtual int get_working_memory_size() const { return 0; }
	virtual bool is_output_port_unsequenced(int p_idx) const { return true; }
	virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const {

		*r_value = instance->get_owner_ptr();
		return true;
	}

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptSelf::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceSelf * instance = memnew(VisualScriptNodeInstanceSelf );
	instance->instance=p_instance;
	return instance;
}



void VisualScriptSelf::_bind_methods() {

}

VisualScriptSelf::VisualScriptSelf() {


}

//////////////////////////////////////////
////////////////CUSTOM (SCRIPTED)///////////
//////////////////////////////////////////

int VisualScriptCustomNode::get_output_sequence_port_count() const {

	if (get_script_instance() && get_script_instance()->has_method("_get_output_sequence_port_count")) {
		return get_script_instance()->call("_get_output_sequence_port_count");
	}
	return 0;
}

bool VisualScriptCustomNode::has_input_sequence_port() const{

	if (get_script_instance() && get_script_instance()->has_method("_has_input_sequence_port")) {
		return get_script_instance()->call("_has_input_sequence_port");
	}
	return false;
}

int VisualScriptCustomNode::get_input_value_port_count() const{

	if (get_script_instance() && get_script_instance()->has_method("_get_input_value_port_count")) {
		return get_script_instance()->call("_get_input_value_port_count");
	}
	return 0;
}
int VisualScriptCustomNode::get_output_value_port_count() const{

	if (get_script_instance() && get_script_instance()->has_method("_get_output_value_port_count")) {
		return get_script_instance()->call("_get_output_value_port_count");
	}
	return 0;
}

String VisualScriptCustomNode::get_output_sequence_port_text(int p_port) const {

	if (get_script_instance() && get_script_instance()->has_method("_get_output_sequence_port_text")) {
		return get_script_instance()->call("_get_output_sequence_port_text",p_port);
	}

	return String();
}

PropertyInfo VisualScriptCustomNode::get_input_value_port_info(int p_idx) const{

	PropertyInfo info;
	if (get_script_instance() && get_script_instance()->has_method("_get_input_value_port_type")) {
		info.type=Variant::Type(int(get_script_instance()->call("_get_input_value_port_type",p_idx)));
	}
	if (get_script_instance() && get_script_instance()->has_method("_get_input_value_port_name")) {
		info.name=get_script_instance()->call("_get_input_value_port_name",p_idx);
	}
	return info;
}

PropertyInfo VisualScriptCustomNode::get_output_value_port_info(int p_idx) const{

	PropertyInfo info;
	if (get_script_instance() && get_script_instance()->has_method("_get_output_value_port_type")) {
		info.type=Variant::Type(int(get_script_instance()->call("_get_output_value_port_type",p_idx)));
	}
	if (get_script_instance() && get_script_instance()->has_method("_get_output_value_port_name")) {
		info.name=get_script_instance()->call("_get_output_value_port_name",p_idx);
	}
	return info;
}


String VisualScriptCustomNode::get_caption() const {

	if (get_script_instance() && get_script_instance()->has_method("_get_caption")) {
		return get_script_instance()->call("_get_caption");
	}
	return "CustomNode";
}

String VisualScriptCustomNode::get_text() const {

	if (get_script_instance() && get_script_instance()->has_method("_get_text")) {
		return get_script_instance()->call("_get_text");
	}
	return "";
}

String VisualScriptCustomNode::get_category() const {

	if (get_script_instance() && get_script_instance()->has_method("_get_category")) {
		return get_script_instance()->call("_get_category");
	}
	return "custom";
}

class VisualScriptNodeInstanceCustomNode : public VisualScriptNodeInstance {
public:

	VisualScriptInstance* instance;
	VisualScriptCustomNode *node;
	int in_count;
	int out_count;
	int work_mem_size;
	Vector<bool> out_unsequenced;

	virtual int get_working_memory_size() const { return work_mem_size; }
	virtual bool is_output_port_unsequenced(int p_idx) const { return out_unsequenced[p_idx]; }
	virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const {

		if (!node->get_script_instance() || !node->get_script_instance()->has_method(VisualScriptLanguage::singleton->_get_output_port_unsequenced)) {
#ifdef DEBUG_ENABLED
			r_error=RTR("Custom node has no _get_output_port_unsequenced(idx,wmem), but unsequenced ports were specified.");
			return false;
		}
#endif

		Array work_mem(true);
		work_mem.resize(work_mem_size);

		*r_value = node->get_script_instance()->call(VisualScriptLanguage::singleton->_get_output_port_unsequenced,p_idx,work_mem);


		for(int i=0;i<work_mem_size;i++) {
			if (i<work_mem.size()) {
				p_working_mem[i]=work_mem[i];
			}
		}

		return true;

	}

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		if (node->get_script_instance()) {
#ifdef DEBUG_ENABLED
			if (!node->get_script_instance()->has_method(VisualScriptLanguage::singleton->_step)) {
				r_error_str=RTR("Custom node has no _step() method, can't process graph.");
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				return 0;
			}
#endif
			Array in_values(true);
			Array out_values(true);
			Array work_mem(true);

			in_values.resize(in_count);

			for(int i=0;i<in_count;i++) {
				in_values[i]=p_inputs[i];
			}

			out_values.resize(in_count);

			work_mem.resize(work_mem_size);

			for(int i=0;i<work_mem_size;i++) {
				work_mem[i]=p_working_mem[i];
			}

			int ret_out;

			Variant ret = node->get_script_instance()->call(VisualScriptLanguage::singleton->_step,in_values,out_values,p_start_mode,work_mem);
			if (ret.get_type()==Variant::STRING) {
				r_error_str=ret;
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				return 0;
			} else if (ret.is_num()) {
				ret_out=ret;
			} else {
				r_error_str=RTR("Invalid return value from _step(), must be integer (seq out), or string (error).");
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				return 0;
			}

			for(int i=0;i<out_count;i++) {
				if (i<out_values.size()) {
					*p_outputs[i]=out_values[i];
				}
			}

			for(int i=0;i<work_mem_size;i++) {
				if (i<work_mem.size()) {
					p_working_mem[i]=work_mem[i];
				}
			}

			return ret_out;

		}

		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptCustomNode::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceCustomNode * instance = memnew(VisualScriptNodeInstanceCustomNode );
	instance->instance=p_instance;
	instance->in_count=get_input_value_port_count();
	instance->out_count=get_output_value_port_count();

	for(int i=0;i<instance->out_count;i++) {
		bool unseq = get_script_instance() && get_script_instance()->has_method("_is_output_port_unsequenced") && bool(get_script_instance()->call("_is_output_port_unsequenced",i));
		instance->out_unsequenced.push_back(unseq);
	}

	if (get_script_instance() && get_script_instance()->has_method("_get_working_memory_size")) {
		instance->work_mem_size = get_script_instance()->call("_get_working_memory_size");
	} else {
		instance->work_mem_size=0;
	}

	return instance;
}



void VisualScriptCustomNode::_bind_methods() {

	BIND_VMETHOD( MethodInfo(Variant::INT,"_get_output_sequence_port_count") );
	BIND_VMETHOD( MethodInfo(Variant::BOOL,"_has_input_sequence_port") );

	BIND_VMETHOD( MethodInfo(Variant::STRING,"_get_output_sequence_port_text",PropertyInfo(Variant::INT,"idx")) );
	BIND_VMETHOD( MethodInfo(Variant::INT,"_get_input_value_port_count") );
	BIND_VMETHOD( MethodInfo(Variant::INT,"_get_output_value_port_count") );

	BIND_VMETHOD( MethodInfo(Variant::INT,"_get_input_value_port_type",PropertyInfo(Variant::INT,"idx")) );
	BIND_VMETHOD( MethodInfo(Variant::STRING,"_get_input_value_port_name",PropertyInfo(Variant::INT,"idx")) );

	BIND_VMETHOD( MethodInfo(Variant::INT,"_get_output_value_port_type",PropertyInfo(Variant::INT,"idx")) );
	BIND_VMETHOD( MethodInfo(Variant::STRING,"_get_output_value_port_name",PropertyInfo(Variant::INT,"idx")) );

	BIND_VMETHOD( MethodInfo(Variant::STRING,"_get_caption") );
	BIND_VMETHOD( MethodInfo(Variant::STRING,"_get_text") );
	BIND_VMETHOD( MethodInfo(Variant::STRING,"_get_category") );

	BIND_VMETHOD( MethodInfo(Variant::INT,"_get_working_memory_size") );
	BIND_VMETHOD( MethodInfo(Variant::INT,"_is_output_port_unsequenced",PropertyInfo(Variant::INT,"idx")) );
	BIND_VMETHOD( MethodInfo(Variant::INT,"_get_output_port_unsequenced",PropertyInfo(Variant::INT,"idx"),PropertyInfo(Variant::ARRAY,"work_mem")) );
	BIND_VMETHOD( MethodInfo(Variant::NIL,"_step:Variant",PropertyInfo(Variant::ARRAY,"inputs"),PropertyInfo(Variant::ARRAY,"outputs"),PropertyInfo(Variant::INT,"start_mode"),PropertyInfo(Variant::ARRAY,"working_mem")) );

	BIND_CONSTANT( START_MODE_BEGIN_SEQUENCE );
	BIND_CONSTANT( START_MODE_CONTINUE_SEQUENCE );
	BIND_CONSTANT( START_MODE_RESUME_YIELD );

	BIND_CONSTANT( STEP_PUSH_STACK_BIT );
	BIND_CONSTANT( STEP_GO_BACK_BIT );
	BIND_CONSTANT( STEP_NO_ADVANCE_BIT );
	BIND_CONSTANT( STEP_EXIT_FUNCTION_BIT );
	BIND_CONSTANT( STEP_YIELD_BIT );

}

VisualScriptCustomNode::VisualScriptCustomNode() {


}

//////////////////////////////////////////
////////////////SUBCALL///////////
//////////////////////////////////////////

int VisualScriptSubCall::get_output_sequence_port_count() const {

	return 1;
}

bool VisualScriptSubCall::has_input_sequence_port() const{

	return true;
}

int VisualScriptSubCall::get_input_value_port_count() const{

	Ref<Script> script = get_script();

	if (script.is_valid() && script->has_method(VisualScriptLanguage::singleton->_subcall)) {

		MethodInfo mi = script->get_method_info(VisualScriptLanguage::singleton->_subcall);
		return mi.arguments.size();
	}

	return 0;
}
int VisualScriptSubCall::get_output_value_port_count() const{

	return 1;
}

String VisualScriptSubCall::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptSubCall::get_input_value_port_info(int p_idx) const{

	Ref<Script> script = get_script();
	if (script.is_valid() && script->has_method(VisualScriptLanguage::singleton->_subcall)) {

		MethodInfo mi = script->get_method_info(VisualScriptLanguage::singleton->_subcall);
		return mi.arguments[p_idx];
	}

	return PropertyInfo();
}

PropertyInfo VisualScriptSubCall::get_output_value_port_info(int p_idx) const{

	Ref<Script> script = get_script();
	if (script.is_valid() && script->has_method(VisualScriptLanguage::singleton->_subcall)) {
		MethodInfo mi = script->get_method_info(VisualScriptLanguage::singleton->_subcall);
		return mi.return_val;
	}
	return PropertyInfo();
}


String VisualScriptSubCall::get_caption() const {

	return "SubCall";
}


String VisualScriptSubCall::get_text() const {

	Ref<Script> script = get_script();
	if (script.is_valid()) {
		if (script->get_name()!=String())
			return script->get_name();
		if (script->get_path().is_resource_file())
			return script->get_path().get_file();
		return script->get_type();
	}
	return "";
}

String VisualScriptSubCall::get_category() const {

	return "custom";
}

class VisualScriptNodeInstanceSubCall : public VisualScriptNodeInstance {
public:

	VisualScriptInstance* instance;
	VisualScriptSubCall *subcall;
	int input_args;
	bool valid;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return false; };

	virtual int step(const Variant** p_inputs,Variant** p_outputs,StartMode p_start_mode,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		if (!valid) {
			r_error_str="Node requires a script with a _subcall(<args>) method to work.";
			r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
			return 0;
		}
		*p_outputs[0]=subcall->call(VisualScriptLanguage::singleton->_subcall,p_inputs,input_args,r_error_str);
		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptSubCall::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceSubCall * instance = memnew(VisualScriptNodeInstanceSubCall );
	instance->instance=p_instance;
	Ref<Script> script = get_script();
	if (script.is_valid() && script->has_method(VisualScriptLanguage::singleton->_subcall)) {
		instance->valid=true;
		instance->input_args=get_input_value_port_count();
	} else {
		instance->valid=false;
	}
	return instance;
}



void VisualScriptSubCall::_bind_methods() {

	BIND_VMETHOD( MethodInfo(Variant::NIL,"_subcall",PropertyInfo(Variant::NIL,"arguments:Variant")) );

}

VisualScriptSubCall::VisualScriptSubCall() {


}


void register_visual_script_nodes() {

	VisualScriptLanguage::singleton->add_register_func("data/set_variable",create_node_generic<VisualScriptVariableGet>);
	VisualScriptLanguage::singleton->add_register_func("data/get_variable",create_node_generic<VisualScriptVariableSet>);
	VisualScriptLanguage::singleton->add_register_func("data/constant",create_node_generic<VisualScriptConstant>);
	VisualScriptLanguage::singleton->add_register_func("data/global_constant",create_node_generic<VisualScriptGlobalConstant>);
	VisualScriptLanguage::singleton->add_register_func("data/math_constant",create_node_generic<VisualScriptMathConstant>);
	VisualScriptLanguage::singleton->add_register_func("data/engine_singleton",create_node_generic<VisualScriptEngineSingleton>);
	VisualScriptLanguage::singleton->add_register_func("data/scene_node",create_node_generic<VisualScriptSceneNode>);
	VisualScriptLanguage::singleton->add_register_func("data/scene_tree",create_node_generic<VisualScriptSceneTree>);
	VisualScriptLanguage::singleton->add_register_func("data/resource_path",create_node_generic<VisualScriptResourcePath>);
	VisualScriptLanguage::singleton->add_register_func("data/self",create_node_generic<VisualScriptSelf>);
	VisualScriptLanguage::singleton->add_register_func("custom/custom_node",create_node_generic<VisualScriptCustomNode>);
	VisualScriptLanguage::singleton->add_register_func("custom/sub_call",create_node_generic<VisualScriptSubCall>);


	VisualScriptLanguage::singleton->add_register_func("index/get_index",create_node_generic<VisualScriptIndexGet>);
	VisualScriptLanguage::singleton->add_register_func("index/set_index",create_node_generic<VisualScriptIndexSet>);


	VisualScriptLanguage::singleton->add_register_func("operators/compare/equal",create_op_node<Variant::OP_EQUAL>);
	VisualScriptLanguage::singleton->add_register_func("operators/compare/not_equal",create_op_node<Variant::OP_NOT_EQUAL>);
	VisualScriptLanguage::singleton->add_register_func("operators/compare/less",create_op_node<Variant::OP_LESS>);
	VisualScriptLanguage::singleton->add_register_func("operators/compare/less_equal",create_op_node<Variant::OP_LESS_EQUAL>);
	VisualScriptLanguage::singleton->add_register_func("operators/compare/greater",create_op_node<Variant::OP_GREATER>);
	VisualScriptLanguage::singleton->add_register_func("operators/compare/greater_equal",create_op_node<Variant::OP_GREATER_EQUAL>);
	//mathematic
	VisualScriptLanguage::singleton->add_register_func("operators/math/add",create_op_node<Variant::OP_ADD>);
	VisualScriptLanguage::singleton->add_register_func("operators/math/subtract",create_op_node<Variant::OP_SUBSTRACT>);
	VisualScriptLanguage::singleton->add_register_func("operators/math/multiply",create_op_node<Variant::OP_MULTIPLY>);
	VisualScriptLanguage::singleton->add_register_func("operators/math/divide",create_op_node<Variant::OP_DIVIDE>);
	VisualScriptLanguage::singleton->add_register_func("operators/math/negate",create_op_node<Variant::OP_NEGATE>);
	VisualScriptLanguage::singleton->add_register_func("operators/math/remainder",create_op_node<Variant::OP_MODULE>);
	VisualScriptLanguage::singleton->add_register_func("operators/math/string_concat",create_op_node<Variant::OP_STRING_CONCAT>);
	//bitwise
	VisualScriptLanguage::singleton->add_register_func("operators/bitwise/shift_left",create_op_node<Variant::OP_SHIFT_LEFT>);
	VisualScriptLanguage::singleton->add_register_func("operators/bitwise/shift_right",create_op_node<Variant::OP_SHIFT_RIGHT>);
	VisualScriptLanguage::singleton->add_register_func("operators/bitwise/bit_and",create_op_node<Variant::OP_BIT_AND>);
	VisualScriptLanguage::singleton->add_register_func("operators/bitwise/bit_or",create_op_node<Variant::OP_BIT_OR>);
	VisualScriptLanguage::singleton->add_register_func("operators/bitwise/bit_xor",create_op_node<Variant::OP_BIT_XOR>);
	VisualScriptLanguage::singleton->add_register_func("operators/bitwise/bit_negate",create_op_node<Variant::OP_BIT_NEGATE>);
	//logic
	VisualScriptLanguage::singleton->add_register_func("operators/logic/and",create_op_node<Variant::OP_AND>);
	VisualScriptLanguage::singleton->add_register_func("operators/logic/or",create_op_node<Variant::OP_OR>);
	VisualScriptLanguage::singleton->add_register_func("operators/logic/xor",create_op_node<Variant::OP_XOR>);
	VisualScriptLanguage::singleton->add_register_func("operators/logic/not",create_op_node<Variant::OP_NOT>);
	VisualScriptLanguage::singleton->add_register_func("operators/logic/in",create_op_node<Variant::OP_IN>);


}
