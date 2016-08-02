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
	emit_signal("ports_changed");

}

Variant::Type VisualScriptReturn::get_return_type() const{

	return type;
}

void VisualScriptReturn::set_enable_return_value(bool p_enable) {
	if (with_value==p_enable)
		return;

	with_value=p_enable;
	emit_signal("ports_changed");
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

VisualScriptNodeInstance* VisualScriptReturn::instance(VScriptInstance* p_instance) {

	return NULL;
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

VisualScriptNodeInstance* VisualScriptCondition::instance(VScriptInstance* p_instance) {

	return NULL;
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

VisualScriptNodeInstance* VisualScriptWhile::instance(VScriptInstance* p_instance) {

	return NULL;
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

VisualScriptNodeInstance* VisualScriptIterator::instance(VScriptInstance* p_instance) {

	return NULL;
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
	emit_signal("ports_changed");

}

int VisualScriptSequence::get_steps() const {

	return steps;
}

void VisualScriptSequence::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_steps","steps"),&VisualScriptSequence::set_steps);
	ObjectTypeDB::bind_method(_MD("get_steps"),&VisualScriptSequence::get_steps);

	ADD_PROPERTY(PropertyInfo(Variant::INT,"steps",PROPERTY_HINT_RANGE,"1,64,1"),_SCS("set_steps"),_SCS("get_steps"));

}

VisualScriptNodeInstance* VisualScriptSequence::instance(VScriptInstance* p_instance) {

	return NULL;
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
