#ifndef VISUAL_SCRIPT_FLOW_CONTROL_H
#define VISUAL_SCRIPT_FLOW_CONTROL_H

#include "visual_script.h"

class VisualScriptReturn : public VisualScriptNode {

	GDCLASS(VisualScriptReturn,VisualScriptNode)


	Variant::Type type;
	bool with_value;
protected:

	static void _bind_methods();
public:

	virtual int get_output_sequence_port_count() const;
	virtual bool has_input_sequence_port() const;


	virtual String get_output_sequence_port_text(int p_port) const;


	virtual int get_input_value_port_count() const;
	virtual int get_output_value_port_count() const;


	virtual PropertyInfo get_input_value_port_info(int p_idx) const;
	virtual PropertyInfo get_output_value_port_info(int p_idx) const;

	virtual String get_caption() const;
	virtual String get_text() const;
	virtual String get_category() const { return "flow_control"; }

	void set_return_type(Variant::Type);
	Variant::Type get_return_type() const;

	void set_enable_return_value(bool p_enable);
	bool is_return_value_enabled() const;


	virtual VisualScriptNodeInstance* instance(VisualScriptInstance* p_instance);

	VisualScriptReturn();
};


class VisualScriptCondition : public VisualScriptNode {

	GDCLASS(VisualScriptCondition,VisualScriptNode)



protected:

	static void _bind_methods();
public:

	virtual int get_output_sequence_port_count() const;
	virtual bool has_input_sequence_port() const;


	virtual String get_output_sequence_port_text(int p_port) const;


	virtual int get_input_value_port_count() const;
	virtual int get_output_value_port_count() const;


	virtual PropertyInfo get_input_value_port_info(int p_idx) const;
	virtual PropertyInfo get_output_value_port_info(int p_idx) const;

	virtual String get_caption() const;
	virtual String get_text() const;
	virtual String get_category() const { return "flow_control"; }


	virtual VisualScriptNodeInstance* instance(VisualScriptInstance* p_instance);

	VisualScriptCondition();
};


class VisualScriptWhile : public VisualScriptNode {

	GDCLASS(VisualScriptWhile,VisualScriptNode)



protected:

	static void _bind_methods();
public:

	virtual int get_output_sequence_port_count() const;
	virtual bool has_input_sequence_port() const;


	virtual String get_output_sequence_port_text(int p_port) const;


	virtual int get_input_value_port_count() const;
	virtual int get_output_value_port_count() const;


	virtual PropertyInfo get_input_value_port_info(int p_idx) const;
	virtual PropertyInfo get_output_value_port_info(int p_idx) const;

	virtual String get_caption() const;
	virtual String get_text() const;
	virtual String get_category() const { return "flow_control"; }


	virtual VisualScriptNodeInstance* instance(VisualScriptInstance* p_instance);

	VisualScriptWhile();
};



class VisualScriptIterator : public VisualScriptNode {

	GDCLASS(VisualScriptIterator,VisualScriptNode)



protected:

	static void _bind_methods();
public:

	virtual int get_output_sequence_port_count() const;
	virtual bool has_input_sequence_port() const;


	virtual String get_output_sequence_port_text(int p_port) const;


	virtual int get_input_value_port_count() const;
	virtual int get_output_value_port_count() const;


	virtual PropertyInfo get_input_value_port_info(int p_idx) const;
	virtual PropertyInfo get_output_value_port_info(int p_idx) const;

	virtual String get_caption() const;
	virtual String get_text() const;
	virtual String get_category() const { return "flow_control"; }


	virtual VisualScriptNodeInstance* instance(VisualScriptInstance* p_instance);

	VisualScriptIterator();
};



class VisualScriptSequence : public VisualScriptNode {

	GDCLASS(VisualScriptSequence,VisualScriptNode)


	int steps;

protected:

	static void _bind_methods();
public:

	virtual int get_output_sequence_port_count() const;
	virtual bool has_input_sequence_port() const;


	virtual String get_output_sequence_port_text(int p_port) const;


	virtual int get_input_value_port_count() const;
	virtual int get_output_value_port_count() const;


	virtual PropertyInfo get_input_value_port_info(int p_idx) const;
	virtual PropertyInfo get_output_value_port_info(int p_idx) const;

	virtual String get_caption() const;
	virtual String get_text() const;
	virtual String get_category() const { return "flow_control"; }

	void set_steps(int p_steps);
	int get_steps() const;

	virtual VisualScriptNodeInstance* instance(VisualScriptInstance* p_instance);

	VisualScriptSequence();
};




class VisualScriptSwitch : public VisualScriptNode {

	GDCLASS(VisualScriptSwitch,VisualScriptNode)

	struct Case {
		Variant::Type type;
		Case() { type=Variant::NIL; }
	};

	Vector<Case> case_values;

friend class VisualScriptNodeInstanceSwitch;
protected:

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

	static void _bind_methods();
public:

	virtual int get_output_sequence_port_count() const;
	virtual bool has_input_sequence_port() const;


	virtual String get_output_sequence_port_text(int p_port) const;
	virtual bool has_mixed_input_and_sequence_ports() const { return true; }


	virtual int get_input_value_port_count() const;
	virtual int get_output_value_port_count() const;


	virtual PropertyInfo get_input_value_port_info(int p_idx) const;
	virtual PropertyInfo get_output_value_port_info(int p_idx) const;

	virtual String get_caption() const;
	virtual String get_text() const;
	virtual String get_category() const { return "flow_control"; }

	virtual VisualScriptNodeInstance* instance(VisualScriptInstance* p_instance);


	VisualScriptSwitch();
};




class VisualScriptInputFilter : public VisualScriptNode {

	GDCLASS(VisualScriptInputFilter,VisualScriptNode)

	Vector<InputEvent> filters;


protected:
	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

public:

	virtual int get_output_sequence_port_count() const;
	virtual bool has_input_sequence_port() const;


	virtual String get_output_sequence_port_text(int p_port) const;


	virtual int get_input_value_port_count() const;
	virtual int get_output_value_port_count() const;


	virtual PropertyInfo get_input_value_port_info(int p_idx) const;
	virtual PropertyInfo get_output_value_port_info(int p_idx) const;

	virtual String get_caption() const;
	virtual String get_text() const;
	virtual String get_category() const { return "flow_control"; }

	virtual VisualScriptNodeInstance* instance(VisualScriptInstance* p_instance);


	VisualScriptInputFilter();
};





class VisualScriptTypeCast : public VisualScriptNode {

	GDCLASS(VisualScriptTypeCast,VisualScriptNode)


	StringName base_type;
	String script;

protected:

	static void _bind_methods();
public:

	virtual int get_output_sequence_port_count() const;
	virtual bool has_input_sequence_port() const;


	virtual String get_output_sequence_port_text(int p_port) const;


	virtual int get_input_value_port_count() const;
	virtual int get_output_value_port_count() const;


	virtual PropertyInfo get_input_value_port_info(int p_idx) const;
	virtual PropertyInfo get_output_value_port_info(int p_idx) const;

	virtual String get_caption() const;
	virtual String get_text() const;
	virtual String get_category() const { return "flow_control"; }

	void set_base_type(const StringName& p_type);
	StringName get_base_type() const;

	void set_base_script(const String& p_path);
	String get_base_script() const;

	virtual VisualScriptNodeInstance* instance(VisualScriptInstance* p_instance);


	VisualScriptTypeCast();
};




void register_visual_script_flow_control_nodes();



#endif // VISUAL_SCRIPT_FLOW_CONTROL_H
