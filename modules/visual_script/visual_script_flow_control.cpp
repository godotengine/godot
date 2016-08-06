#include "visual_script_flow_control.h"

//////////////////////////////////////////
////////////////RETURN////////////////////
//////////////////////////////////////////

int VisualScriptReturn::get_output_sequence_port_count() const {

	return 0;
}

bool VisualScriptReturn::has_input_sequence_port() const{

	return true;
}

int VisualScriptReturn::get_input_value_port_count() const{

	return with_value?1:0;
}
int VisualScriptReturn::get_output_value_port_count() const{

	return 0;
}

String VisualScriptReturn::get_output_sequence_port_text(int p_port) const {

	return String();
}

PropertyInfo VisualScriptReturn::get_input_value_port_info(int p_idx) const{

	PropertyInfo pinfo;
	pinfo.name="result";
	pinfo.type=type;
	return pinfo;
}
PropertyInfo VisualScriptReturn::get_output_value_port_info(int p_idx) const{
	return PropertyInfo();
}

String VisualScriptReturn::get_caption() const {

	return "Return";
}

String VisualScriptReturn::get_text() const {

	return get_name();
}

void VisualScriptReturn::set_return_type(Variant::Type p_type) {

	if (type==p_type)
		return;
	type=p_type;
	ports_changed_notify();

}

Variant::Type VisualScriptReturn::get_return_type() const{

	return type;
}

void VisualScriptReturn::set_enable_return_value(bool p_enable) {
	if (with_value==p_enable)
		return;

	with_value=p_enable;
	ports_changed_notify();
}

bool VisualScriptReturn::is_return_value_enabled() const {

	return with_value;
}

void VisualScriptReturn::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_return_type","type"),&VisualScriptReturn::set_return_type);
	ObjectTypeDB::bind_method(_MD("get_return_type"),&VisualScriptReturn::get_return_type);
	ObjectTypeDB::bind_method(_MD("set_enable_return_value","enable"),&VisualScriptReturn::set_enable_return_value);
	ObjectTypeDB::bind_method(_MD("is_return_value_enabled"),&VisualScriptReturn::is_return_value_enabled);

	String argt="Variant";
	for(int i=1;i<Variant::VARIANT_MAX;i++) {
		argt+=","+Variant::get_type_name(Variant::Type(i));
	}

	ADD_PROPERTY(PropertyInfo(Variant::BOOL,"return_value/enabled"),_SCS("set_enable_return_value"),_SCS("is_return_value_enabled"));
	ADD_PROPERTY(PropertyInfo(Variant::INT,"return_value/type",PROPERTY_HINT_ENUM,argt),_SCS("set_return_type"),_SCS("get_return_type"));

}

class VisualScriptNodeInstanceReturn : public VisualScriptNodeInstance {
public:

	VisualScriptReturn *node;
	VisualScriptInstance *instance;
	bool with_value;

	virtual int get_working_memory_size() const { return 1; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant** p_inputs,Variant** p_outputs,bool p_start_sequence,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		if (with_value) {
			*p_working_mem = *p_inputs[0];
		} else {
			*p_working_mem = Variant();
		}

		return 0;
	}


};

VisualScriptNodeInstance* VisualScriptReturn::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceReturn * instance = memnew(VisualScriptNodeInstanceReturn );
	instance->node=this;
	instance->instance=p_instance;
	instance->with_value=with_value;
	return instance;
}

VisualScriptReturn::VisualScriptReturn() {

	with_value=false;
	type=Variant::NIL;
}

template<bool with_value>
static Ref<VisualScriptNode> create_return_node(const String& p_name) {

	Ref<VisualScriptReturn> node;
	node.instance();
	node->set_enable_return_value(with_value);
	return node;
}



//////////////////////////////////////////
////////////////CONDITION/////////////////
//////////////////////////////////////////

int VisualScriptCondition::get_output_sequence_port_count() const {

	return 2;
}

bool VisualScriptCondition::has_input_sequence_port() const{

	return true;
}

int VisualScriptCondition::get_input_value_port_count() const{

	return 1;
}
int VisualScriptCondition::get_output_value_port_count() const{

	return 0;
}

String VisualScriptCondition::get_output_sequence_port_text(int p_port) const {

	if (p_port==0)
		return "true";
	else
		return "false";
}

PropertyInfo VisualScriptCondition::get_input_value_port_info(int p_idx) const{

	PropertyInfo pinfo;
	pinfo.name="cond";
	pinfo.type=Variant::BOOL;
	return pinfo;
}
PropertyInfo VisualScriptCondition::get_output_value_port_info(int p_idx) const{
	return PropertyInfo();
}

String VisualScriptCondition::get_caption() const {

	return "Condition";
}

String VisualScriptCondition::get_text() const {

	return "if (cond) is:  ";
}


void VisualScriptCondition::_bind_methods() {



}

class VisualScriptNodeInstanceCondition : public VisualScriptNodeInstance {
public:

	VisualScriptCondition *node;
	VisualScriptInstance *instance;

	//virtual int get_working_memory_size() const { return 1; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant** p_inputs,Variant** p_outputs,bool p_start_sequence,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		if (p_inputs[0]->operator bool())
			return 0;
		else
			return 1;
	}


};

VisualScriptNodeInstance* VisualScriptCondition::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceCondition * instance = memnew(VisualScriptNodeInstanceCondition );
	instance->node=this;
	instance->instance=p_instance;
	return instance;
}

VisualScriptCondition::VisualScriptCondition() {

}



//////////////////////////////////////////
////////////////WHILE/////////////////
//////////////////////////////////////////

int VisualScriptWhile::get_output_sequence_port_count() const {

	return 2;
}

bool VisualScriptWhile::has_input_sequence_port() const{

	return true;
}

int VisualScriptWhile::get_input_value_port_count() const{

	return 1;
}
int VisualScriptWhile::get_output_value_port_count() const{

	return 0;
}

String VisualScriptWhile::get_output_sequence_port_text(int p_port) const {

	if (p_port==0)
		return "repeat";
	else
		return "exit";
}

PropertyInfo VisualScriptWhile::get_input_value_port_info(int p_idx) const{

	PropertyInfo pinfo;
	pinfo.name="cond";
	pinfo.type=Variant::BOOL;
	return pinfo;
}
PropertyInfo VisualScriptWhile::get_output_value_port_info(int p_idx) const{
	return PropertyInfo();
}

String VisualScriptWhile::get_caption() const {

	return "While";
}

String VisualScriptWhile::get_text() const {

	return "while (cond): ";
}


void VisualScriptWhile::_bind_methods() {



}

class VisualScriptNodeInstanceWhile : public VisualScriptNodeInstance {
public:

	VisualScriptWhile *node;
	VisualScriptInstance *instance;

	//virtual int get_working_memory_size() const { return 1; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant** p_inputs,Variant** p_outputs,bool p_start_sequence,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		bool keep_going = p_inputs[0]->operator bool();

		if (keep_going)
			return 0|STEP_FLAG_PUSH_STACK_BIT;
		else
			return 1;
	}


};

VisualScriptNodeInstance* VisualScriptWhile::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceWhile * instance = memnew(VisualScriptNodeInstanceWhile );
	instance->node=this;
	instance->instance=p_instance;
	return instance;
}
VisualScriptWhile::VisualScriptWhile() {

}



//////////////////////////////////////////
////////////////ITERATOR/////////////////
//////////////////////////////////////////

int VisualScriptIterator::get_output_sequence_port_count() const {

	return 2;
}

bool VisualScriptIterator::has_input_sequence_port() const{

	return true;
}

int VisualScriptIterator::get_input_value_port_count() const{

	return 1;
}
int VisualScriptIterator::get_output_value_port_count() const{

	return 1;
}

String VisualScriptIterator::get_output_sequence_port_text(int p_port) const {

	if (p_port==0)
		return "each";
	else
		return "exit";
}

PropertyInfo VisualScriptIterator::get_input_value_port_info(int p_idx) const{

	PropertyInfo pinfo;
	pinfo.name="input";
	pinfo.type=Variant::NIL;
	return pinfo;
}
PropertyInfo VisualScriptIterator::get_output_value_port_info(int p_idx) const{
	PropertyInfo pinfo;
	pinfo.name="elem";
	pinfo.type=Variant::NIL;
	return pinfo;
}
String VisualScriptIterator::get_caption() const {

	return "Iterator";
}

String VisualScriptIterator::get_text() const {

	return "for (elem) in (input): ";
}


void VisualScriptIterator::_bind_methods() {



}

class VisualScriptNodeInstanceIterator : public VisualScriptNodeInstance {
public:

	VisualScriptIterator *node;
	VisualScriptInstance *instance;

	virtual int get_working_memory_size() const { return 2; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant** p_inputs,Variant** p_outputs,bool p_start_sequence,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		if (p_start_sequence) {
			p_working_mem[0]=*p_inputs[0];
			bool valid;
			bool can_iter = p_inputs[0]->iter_init(p_working_mem[1],valid);

			if (!valid) {
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				r_error_str=RTR("Input type not iterable: ")+Variant::get_type_name(p_inputs[0]->get_type());
				return 0;
			}

			if (!can_iter)
				return 1; //nothing to iterate


			*p_outputs[0]=p_working_mem[0].iter_get( p_working_mem[1],valid);

			if (!valid) {
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				r_error_str=RTR("Iterator became invalid");
				return 0;
			}


		} else {

			bool valid;
			bool can_iter = p_working_mem[0].iter_next(p_working_mem[1],valid);

			if (!valid) {
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				r_error_str=RTR("Iterator became invalid: ")+Variant::get_type_name(p_inputs[0]->get_type());
				return 0;
			}

			if (!can_iter)
				return 1; //nothing to iterate


			*p_outputs[0]=p_working_mem[0].iter_get( p_working_mem[1],valid);

			if (!valid) {
				r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
				r_error_str=RTR("Iterator became invalid");
				return 0;
			}

		}

		return 0|STEP_FLAG_PUSH_STACK_BIT; //go around
	}


};

VisualScriptNodeInstance* VisualScriptIterator::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceIterator * instance = memnew(VisualScriptNodeInstanceIterator );
	instance->node=this;
	instance->instance=p_instance;
	return instance;
}

VisualScriptIterator::VisualScriptIterator() {

}



//////////////////////////////////////////
////////////////SEQUENCE/////////////////
//////////////////////////////////////////

int VisualScriptSequence::get_output_sequence_port_count() const {

	return steps;
}

bool VisualScriptSequence::has_input_sequence_port() const{

	return true;
}

int VisualScriptSequence::get_input_value_port_count() const{

	return 0;
}
int VisualScriptSequence::get_output_value_port_count() const{

	return 1;
}

String VisualScriptSequence::get_output_sequence_port_text(int p_port) const {

	return itos(p_port+1);
}

PropertyInfo VisualScriptSequence::get_input_value_port_info(int p_idx) const{
	return PropertyInfo();
}
PropertyInfo VisualScriptSequence::get_output_value_port_info(int p_idx) const{
	return PropertyInfo(Variant::INT,"current");
}
String VisualScriptSequence::get_caption() const {

	return "Sequence";
}

String VisualScriptSequence::get_text() const {

	return "in order: ";
}

void VisualScriptSequence::set_steps(int p_steps) {

	ERR_FAIL_COND(p_steps<1);
	if (steps==p_steps)
		return;

	steps=p_steps;
	ports_changed_notify();

}

int VisualScriptSequence::get_steps() const {

	return steps;
}

void VisualScriptSequence::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_steps","steps"),&VisualScriptSequence::set_steps);
	ObjectTypeDB::bind_method(_MD("get_steps"),&VisualScriptSequence::get_steps);

	ADD_PROPERTY(PropertyInfo(Variant::INT,"steps",PROPERTY_HINT_RANGE,"1,64,1"),_SCS("set_steps"),_SCS("get_steps"));

}

class VisualScriptNodeInstanceSequence : public VisualScriptNodeInstance {
public:

	VisualScriptSequence *node;
	VisualScriptInstance *instance;
	int steps;

	virtual int get_working_memory_size() const { return 1; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant** p_inputs,Variant** p_outputs,bool p_start_sequence,Variant* p_working_mem,Variant::CallError& r_error,String& r_error_str) {

		if (p_start_sequence) {

			p_working_mem[0]=0;
		}

		int step = p_working_mem[0];

		*p_outputs[0]=step;

		if (step+1==steps)
			return step;
		else {
			p_working_mem[0]=step+1;
			return step|STEP_FLAG_PUSH_STACK_BIT;
		}

	}


};

VisualScriptNodeInstance* VisualScriptSequence::instance(VisualScriptInstance* p_instance) {

	VisualScriptNodeInstanceSequence * instance = memnew(VisualScriptNodeInstanceSequence );
	instance->node=this;
	instance->instance=p_instance;
	instance->steps=steps;
	return instance;
}
VisualScriptSequence::VisualScriptSequence() {

	steps=1;
}

void register_visual_script_flow_control_nodes() {

	VisualScriptLanguage::singleton->add_register_func("flow_control/return",create_return_node<false>);
	VisualScriptLanguage::singleton->add_register_func("flow_control/return_with_value",create_return_node<true>);
	VisualScriptLanguage::singleton->add_register_func("flow_control/condition",create_node_generic<VisualScriptCondition>);
	VisualScriptLanguage::singleton->add_register_func("flow_control/while",create_node_generic<VisualScriptWhile>);
	VisualScriptLanguage::singleton->add_register_func("flow_control/iterator",create_node_generic<VisualScriptIterator>);
	VisualScriptLanguage::singleton->add_register_func("flow_control/sequence",create_node_generic<VisualScriptSequence>);

}
