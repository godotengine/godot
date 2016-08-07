#ifndef VISUAL_SCRIPT_FUNC_NODES_H
#define VISUAL_SCRIPT_FUNC_NODES_H

#include "visual_script.h"


class VisualScriptFunctionCall : public VisualScriptNode {

	OBJ_TYPE(VisualScriptFunctionCall,VisualScriptNode)
public:
	enum CallMode {
		CALL_MODE_SELF,
		CALL_MODE_NODE_PATH,
		CALL_MODE_INSTANCE,
		CALL_MODE_BASIC_TYPE,
	};
private:

	CallMode call_mode;
	StringName base_type;
	Variant::Type basic_type;
	NodePath base_path;
	StringName function;
	int use_default_args;

	Node *_get_base_node() const;
	StringName _get_base_type() const;

	void _update_defargs();
protected:
	virtual void _validate_property(PropertyInfo& property) const;

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
	virtual String get_category() const { return "functions"; }

	void set_basic_type(Variant::Type p_type);
	Variant::Type get_basic_type() const;

	void set_base_type(const StringName& p_type);
	StringName get_base_type() const;

	void set_function(const StringName& p_type);
	StringName get_function() const;

	void set_base_path(const NodePath& p_type);
	NodePath get_base_path() const;

	void set_call_mode(CallMode p_mode);
	CallMode get_call_mode() const;

	void set_use_default_args(int p_amount);
	int get_use_default_args() const;

	virtual VisualScriptNodeInstance* instance(VisualScriptInstance* p_instance);

	VisualScriptFunctionCall();
};

VARIANT_ENUM_CAST(VisualScriptFunctionCall::CallMode );


class VisualScriptPropertySet : public VisualScriptNode {

	OBJ_TYPE(VisualScriptPropertySet,VisualScriptNode)
public:
	enum CallMode {
		CALL_MODE_SELF,
		CALL_MODE_NODE_PATH,
		CALL_MODE_INSTANCE,
		CALL_MODE_BASIC_TYPE,


	};
private:

	CallMode call_mode;
	Variant::Type basic_type;	
	StringName base_type;
	NodePath base_path;
	StringName property;
	bool use_builtin_value;
	Variant builtin_value;
	InputEvent::Type event_type;

	Node *_get_base_node() const;
	StringName _get_base_type() const;

	void _update_base_type();

protected:
	virtual void _validate_property(PropertyInfo& property) const;

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
	virtual String get_category() const { return "functions"; }

	void set_base_type(const StringName& p_type);
	StringName get_base_type() const;

	void set_basic_type(Variant::Type p_type);
	Variant::Type get_basic_type() const;

	void set_event_type(InputEvent::Type p_type);
	InputEvent::Type get_event_type() const;

	void set_property(const StringName& p_type);
	StringName get_property() const;

	void set_base_path(const NodePath& p_type);
	NodePath get_base_path() const;

	void set_call_mode(CallMode p_mode);
	CallMode get_call_mode() const;

	void set_use_builtin_value(bool p_use);
	bool is_using_builtin_value() const;

	void set_builtin_value(const Variant &p_value);
	Variant get_builtin_value() const;

	virtual VisualScriptNodeInstance* instance(VisualScriptInstance* p_instance);

	VisualScriptPropertySet();
};

VARIANT_ENUM_CAST(VisualScriptPropertySet::CallMode );


class VisualScriptPropertyGet : public VisualScriptNode {

	OBJ_TYPE(VisualScriptPropertyGet,VisualScriptNode)
public:
	enum CallMode {
		CALL_MODE_SELF,
		CALL_MODE_NODE_PATH,
		CALL_MODE_INSTANCE,
		CALL_MODE_BASIC_TYPE

	};
private:

	CallMode call_mode;
	Variant::Type basic_type;
	StringName base_type;
	NodePath base_path;
	StringName property;
	InputEvent::Type event_type;

	void _update_base_type();
	Node *_get_base_node() const;
	StringName _get_base_type() const;


protected:
	virtual void _validate_property(PropertyInfo& property) const;

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
	virtual String get_category() const { return "functions"; }

	void set_base_type(const StringName& p_type);
	StringName get_base_type() const;

	void set_basic_type(Variant::Type p_type);
	Variant::Type get_basic_type() const;

	void set_event_type(InputEvent::Type p_type);
	InputEvent::Type get_event_type() const;

	void set_property(const StringName& p_type);
	StringName get_property() const;

	void set_base_path(const NodePath& p_type);
	NodePath get_base_path() const;

	void set_call_mode(CallMode p_mode);
	CallMode get_call_mode() const;

	virtual VisualScriptNodeInstance* instance(VisualScriptInstance* p_instance);

	VisualScriptPropertyGet();
};





VARIANT_ENUM_CAST(VisualScriptPropertyGet::CallMode );



class VisualScriptScriptCall : public VisualScriptNode {

	OBJ_TYPE(VisualScriptScriptCall,VisualScriptNode)
public:
	enum CallMode {
		CALL_MODE_SELF,
		CALL_MODE_NODE_PATH,
	};
private:

	CallMode call_mode;
	NodePath base_path;
	StringName function;
	int argument_count;


	Node *_get_base_node() const;


	void _update_argument_count();
protected:
	virtual void _validate_property(PropertyInfo& property) const;

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
	virtual String get_category() const { return "functions"; }

	void set_function(const StringName& p_type);
	StringName get_function() const;

	void set_base_path(const NodePath& p_type);
	NodePath get_base_path() const;

	void set_call_mode(CallMode p_mode);
	CallMode get_call_mode() const;

	void set_argument_count(int p_count);
	int get_argument_count() const;


	virtual VisualScriptNodeInstance* instance(VisualScriptInstance* p_instance);

	VisualScriptScriptCall();
};

VARIANT_ENUM_CAST(VisualScriptScriptCall::CallMode );




class VisualScriptEmitSignal : public VisualScriptNode {

	OBJ_TYPE(VisualScriptEmitSignal,VisualScriptNode)

private:

	StringName name;


protected:
	virtual void _validate_property(PropertyInfo& property) const;

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
	virtual String get_category() const { return "functions"; }

	void set_signal(const StringName& p_type);
	StringName get_signal() const;

	virtual VisualScriptNodeInstance* instance(VisualScriptInstance* p_instance);

	VisualScriptEmitSignal();
};



void register_visual_script_func_nodes();

#endif // VISUAL_SCRIPT_FUNC_NODES_H
