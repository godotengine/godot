/*************************************************************************/
/*  visual_shader.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef VISUAL_SHADER_H
#define VISUAL_SHADER_H

#include "core/string/string_builder.h"
#include "core/templates/safe_refcount.h"
#include "scene/gui/control.h"
#include "scene/resources/shader.h"

class VisualShaderNodeUniform;
class VisualShaderNode;

class VisualShader : public Shader {
	GDCLASS(VisualShader, Shader);

	friend class VisualShaderNodeVersionChecker;

	Dictionary engine_version;

public:
	enum Type {
		TYPE_VERTEX,
		TYPE_FRAGMENT,
		TYPE_LIGHT,
		TYPE_START,
		TYPE_PROCESS,
		TYPE_COLLIDE,
		TYPE_START_CUSTOM,
		TYPE_PROCESS_CUSTOM,
		TYPE_SKY,
		TYPE_FOG,
		TYPE_MAX
	};

	struct Connection {
		int from_node = 0;
		int from_port = 0;
		int to_node = 0;
		int to_port = 0;
	};

	struct DefaultTextureParam {
		StringName name;
		Ref<Texture2D> param;
	};

private:
	Type current_type;

	struct Node {
		Ref<VisualShaderNode> node;
		Vector2 position;
		List<int> prev_connected_nodes;
	};

	struct Graph {
		Map<int, Node> nodes;
		List<Connection> connections;
	} graph[TYPE_MAX];

	Shader::Mode shader_mode = Shader::MODE_SPATIAL;
	mutable String previous_code;

	Array _get_node_connections(Type p_type) const;

	Vector2 graph_offset;

	struct RenderModeEnums {
		Shader::Mode mode = Shader::Mode::MODE_MAX;
		const char *string;
	};

	HashMap<String, int> modes;
	Set<StringName> flags;

	static RenderModeEnums render_mode_enums[];

	mutable SafeFlag dirty;
	void _queue_update();

	union ConnectionKey {
		struct {
			uint64_t node : 32;
			uint64_t port : 32;
		};
		uint64_t key = 0;
		bool operator<(const ConnectionKey &p_key) const {
			return key < p_key.key;
		}
	};

	Error _write_node(Type p_type, StringBuilder &global_code, StringBuilder &global_code_per_node, Map<Type, StringBuilder> &global_code_per_func, StringBuilder &code, Vector<DefaultTextureParam> &def_tex_params, const VMap<ConnectionKey, const List<Connection>::Element *> &input_connections, const VMap<ConnectionKey, const List<Connection>::Element *> &output_connections, int node, Set<int> &processed, bool for_preview, Set<StringName> &r_classes) const;

	void _input_type_changed(Type p_type, int p_id);
	bool has_func_name(RenderingServer::ShaderMode p_mode, const String &p_func_name) const;

protected:
	virtual void _update_shader() const override;
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	virtual void reset_state() override;

public: // internal methods
	void set_shader_type(Type p_type);
	Type get_shader_type() const;

public:
	void set_engine_version(const Dictionary &p_version);
	Dictionary get_engine_version() const;

#ifndef DISABLE_DEPRECATED
	void update_engine_version(const Dictionary &p_new_version);
#endif /* DISABLE_DEPRECATED */

	enum {
		NODE_ID_INVALID = -1,
		NODE_ID_OUTPUT = 0,
	};

	void add_node(Type p_type, const Ref<VisualShaderNode> &p_node, const Vector2 &p_position, int p_id);
	void set_node_position(Type p_type, int p_id, const Vector2 &p_position);

	Vector2 get_node_position(Type p_type, int p_id) const;
	Ref<VisualShaderNode> get_node(Type p_type, int p_id) const;

	Vector<int> get_node_list(Type p_type) const;
	int get_valid_node_id(Type p_type) const;

	int find_node_id(Type p_type, const Ref<VisualShaderNode> &p_node) const;
	void remove_node(Type p_type, int p_id);
	void replace_node(Type p_type, int p_id, const StringName &p_new_class);

	bool is_node_connection(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) const;

	bool is_nodes_connected_relatively(const Graph *p_graph, int p_node, int p_target) const;
	bool can_connect_nodes(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) const;
	Error connect_nodes(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port);
	void disconnect_nodes(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port);
	void connect_nodes_forced(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port);
	bool is_port_types_compatible(int p_a, int p_b) const;

	void rebuild();
	void get_node_connections(Type p_type, List<Connection> *r_connections) const;

	void set_mode(Mode p_mode);
	virtual Mode get_mode() const override;

	virtual bool is_text_shader() const override;

	void set_graph_offset(const Vector2 &p_offset);
	Vector2 get_graph_offset() const;

	String generate_preview_shader(Type p_type, int p_node, int p_port, Vector<DefaultTextureParam> &r_default_tex_params) const;

	String validate_port_name(const String &p_port_name, VisualShaderNode *p_node, int p_port_id, bool p_output) const;
	String validate_uniform_name(const String &p_name, const Ref<VisualShaderNodeUniform> &p_uniform) const;

	VisualShader();
};

VARIANT_ENUM_CAST(VisualShader::Type)
///
///
///

class VisualShaderNode : public Resource {
	GDCLASS(VisualShaderNode, Resource);

	int port_preview = -1;

	Map<int, Variant> default_input_values;
	Map<int, bool> connected_input_ports;
	Map<int, int> connected_output_ports;
	Map<int, bool> expanded_output_ports;

protected:
	bool simple_decl = true;
	bool disabled = false;

	static void _bind_methods();

public:
	enum PortType {
		PORT_TYPE_SCALAR,
		PORT_TYPE_SCALAR_INT,
		PORT_TYPE_VECTOR,
		PORT_TYPE_BOOLEAN,
		PORT_TYPE_TRANSFORM,
		PORT_TYPE_SAMPLER,
		PORT_TYPE_MAX,
	};

	bool is_simple_decl() const;

	virtual String get_caption() const = 0;

	virtual int get_input_port_count() const = 0;
	virtual PortType get_input_port_type(int p_port) const = 0;
	virtual String get_input_port_name(int p_port) const = 0;

	virtual void set_input_port_default_value(int p_port, const Variant &p_value);
	Variant get_input_port_default_value(int p_port) const; // if NIL (default if node does not set anything) is returned, it means no default value is wanted if disconnected, thus no input var must be supplied (empty string will be supplied)
	Array get_default_input_values() const;
	virtual void set_default_input_values(const Array &p_values);
	virtual void remove_input_port_default_value(int p_port);
	virtual void clear_default_input_values();

	virtual int get_output_port_count() const = 0;
	virtual PortType get_output_port_type(int p_port) const = 0;
	virtual String get_output_port_name(int p_port) const = 0;

	virtual String get_input_port_default_hint(int p_port) const;

	void set_output_port_for_preview(int p_index);
	int get_output_port_for_preview() const;

	virtual bool is_port_separator(int p_index) const;

	bool is_output_port_connected(int p_port) const;
	void set_output_port_connected(int p_port, bool p_connected);
	bool is_input_port_connected(int p_port) const;
	void set_input_port_connected(int p_port, bool p_connected);
	virtual bool is_generate_input_var(int p_port) const;

	virtual bool has_output_port_preview(int p_port) const;

	virtual bool is_output_port_expandable(int p_port) const;
	void _set_output_ports_expanded(const Array &p_data);
	Array _get_output_ports_expanded() const;
	void _set_output_port_expanded(int p_port, bool p_expanded);
	bool _is_output_port_expanded(int p_port) const;
	int get_expanded_output_port_count() const;

	virtual bool is_code_generated() const;
	virtual bool is_show_prop_names() const;
	virtual bool is_use_prop_slots() const;

	bool is_disabled() const;
	void set_disabled(bool p_disabled = true);

	virtual Vector<StringName> get_editable_properties() const;
	virtual Map<StringName, String> get_editable_properties_names() const;

	virtual Vector<VisualShader::DefaultTextureParam> get_default_texture_parameters(VisualShader::Type p_type, int p_id) const;
	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const;
	virtual String generate_global_per_node(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const;
	virtual String generate_global_per_func(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const;
	// If no output is connected, the output var passed will be empty. If no input is connected and input is NIL, the input var passed will be empty.
	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const = 0;

	virtual String get_warning(Shader::Mode p_mode, VisualShader::Type p_type) const;

	VisualShaderNode();
};

VARIANT_ENUM_CAST(VisualShaderNode::PortType)

class VisualShaderNodeCustom : public VisualShaderNode {
	GDCLASS(VisualShaderNodeCustom, VisualShaderNode);

	struct Port {
		String name;
		int type = 0;
	};

	bool is_initialized = false;
	List<Port> input_ports;
	List<Port> output_ports;

	friend class VisualShaderEditor;

protected:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual void set_input_port_default_value(int p_port, const Variant &p_value) override;
	virtual void set_default_input_values(const Array &p_values) override;
	virtual void remove_input_port_default_value(int p_port) override;
	virtual void clear_default_input_values() override;

	GDVIRTUAL0RC(String, _get_name)
	GDVIRTUAL0RC(String, _get_description)
	GDVIRTUAL0RC(String, _get_category)
	GDVIRTUAL0RC(int, _get_return_icon_type)
	GDVIRTUAL0RC(int, _get_input_port_count)
	GDVIRTUAL1RC(int, _get_input_port_type, int)
	GDVIRTUAL1RC(String, _get_input_port_name, int)
	GDVIRTUAL0RC(int, _get_output_port_count)
	GDVIRTUAL1RC(int, _get_output_port_type, int)
	GDVIRTUAL1RC(String, _get_output_port_name, int)
	GDVIRTUAL4RC(String, _get_code, Vector<String>, TypedArray<String>, int, int)
	GDVIRTUAL1RC(String, _get_global_code, int)
	GDVIRTUAL0RC(bool, _is_highend)

protected:
	void _set_input_port_default_value(int p_port, const Variant &p_value);

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;
	virtual String generate_global_per_node(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;

	static void _bind_methods();

public:
	VisualShaderNodeCustom();
	void update_ports();

	bool _is_initialized();
	void _set_initialized(bool p_enabled);
};

/////

class VisualShaderNodeInput : public VisualShaderNode {
	GDCLASS(VisualShaderNodeInput, VisualShaderNode);

	friend class VisualShader;
	VisualShader::Type shader_type = VisualShader::TYPE_MAX;
	Shader::Mode shader_mode = Shader::MODE_MAX;

	struct Port {
		Shader::Mode mode = Shader::Mode::MODE_MAX;
		VisualShader::Type shader_type = VisualShader::Type::TYPE_MAX;
		PortType type = PortType::PORT_TYPE_MAX;
		const char *name;
		const char *string;
	};

	static const Port ports[];
	static const Port preview_ports[];

	String input_name = "[None]";

public:
	void set_shader_type(VisualShader::Type p_shader_type);
	void set_shader_mode(Shader::Mode p_shader_mode);

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const override;

public:
	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String get_caption() const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_input_name(String p_name);
	String get_input_name() const;
	String get_input_real_name() const;

	int get_input_index_count() const;
	PortType get_input_index_type(int p_index) const;
	String get_input_index_name(int p_index) const;

	PortType get_input_type_by_name(String p_name) const;

	virtual Vector<StringName> get_editable_properties() const override;

	VisualShaderNodeInput();
};

///

class VisualShaderNodeOutput : public VisualShaderNode {
	GDCLASS(VisualShaderNodeOutput, VisualShaderNode);

public:
	friend class VisualShader;
	VisualShader::Type shader_type = VisualShader::Type::TYPE_MAX;
	Shader::Mode shader_mode = Shader::Mode::MODE_MAX;

	struct Port {
		Shader::Mode mode = Shader::Mode::MODE_MAX;
		VisualShader::Type shader_type = VisualShader::Type::TYPE_MAX;
		PortType type = PortType::PORT_TYPE_MAX;
		const char *name;
		const char *string;
	};

	static const Port ports[];

public:
	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;
	Variant get_input_port_default_value(int p_port) const;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual bool is_port_separator(int p_index) const override;

	virtual String get_caption() const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeOutput();
};

class VisualShaderNodeUniform : public VisualShaderNode {
	GDCLASS(VisualShaderNodeUniform, VisualShaderNode);

public:
	enum Qualifier {
		QUAL_NONE,
		QUAL_GLOBAL,
		QUAL_INSTANCE,
		QUAL_MAX,
	};

private:
	String uniform_name = "";
	Qualifier qualifier = QUAL_NONE;
	bool global_code_generated = false;

protected:
	static void _bind_methods();
	String _get_qual_str() const;

public:
	void set_uniform_name(const String &p_name);
	String get_uniform_name() const;

	void set_qualifier(Qualifier p_qual);
	Qualifier get_qualifier() const;

	void set_global_code_generated(bool p_enabled);
	bool is_global_code_generated() const;

	virtual bool is_qualifier_supported(Qualifier p_qual) const = 0;
	virtual bool is_convertible_to_constant() const = 0;

	virtual Vector<StringName> get_editable_properties() const override;
	virtual String get_warning(Shader::Mode p_mode, VisualShader::Type p_type) const override;

	VisualShaderNodeUniform();
};

VARIANT_ENUM_CAST(VisualShaderNodeUniform::Qualifier)

class VisualShaderNodeUniformRef : public VisualShaderNode {
	GDCLASS(VisualShaderNodeUniformRef, VisualShaderNode);

public:
	enum UniformType {
		UNIFORM_TYPE_FLOAT,
		UNIFORM_TYPE_INT,
		UNIFORM_TYPE_BOOLEAN,
		UNIFORM_TYPE_VECTOR,
		UNIFORM_TYPE_TRANSFORM,
		UNIFORM_TYPE_COLOR,
		UNIFORM_TYPE_SAMPLER,
	};

	struct Uniform {
		String name;
		UniformType type;
	};

private:
	String uniform_name = "[None]";
	UniformType uniform_type = UniformType::UNIFORM_TYPE_FLOAT;

protected:
	static void _bind_methods();

public:
	static void add_uniform(const String &p_name, UniformType p_type);
	static void clear_uniforms();
	static bool has_uniform(const String &p_name);

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	void set_uniform_name(const String &p_name);
	String get_uniform_name() const;

	void _set_uniform_type(int p_uniform_type);
	int _get_uniform_type() const;

	int get_uniforms_count() const;
	String get_uniform_name_by_index(int p_idx) const;
	UniformType get_uniform_type_by_name(const String &p_name) const;
	UniformType get_uniform_type_by_index(int p_idx) const;
	PortType get_port_type_by_index(int p_idx) const;

	virtual Vector<StringName> get_editable_properties() const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeUniformRef();
};

class VisualShaderNodeResizableBase : public VisualShaderNode {
	GDCLASS(VisualShaderNodeResizableBase, VisualShaderNode);

protected:
	Vector2 size = Size2(0, 0);
	bool allow_v_resize = true;

protected:
	static void _bind_methods();

public:
	void set_size(const Vector2 &p_size);
	Vector2 get_size() const;

	bool is_allow_v_resize() const;
	void set_allow_v_resize(bool p_enabled);

	VisualShaderNodeResizableBase();
};

class VisualShaderNodeComment : public VisualShaderNodeResizableBase {
	GDCLASS(VisualShaderNodeComment, VisualShaderNodeResizableBase);

protected:
	String title = "Comment";
	String description = "";

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_title(const String &p_title);
	String get_title() const;

	void set_description(const String &p_description);
	String get_description() const;

	VisualShaderNodeComment();
};

class VisualShaderNodeGroupBase : public VisualShaderNodeResizableBase {
	GDCLASS(VisualShaderNodeGroupBase, VisualShaderNodeResizableBase);

private:
	void _apply_port_changes();

protected:
	String inputs = "";
	String outputs = "";
	bool editable = false;

	struct Port {
		PortType type = PortType::PORT_TYPE_MAX;
		String name;
	};

	Map<int, Port> input_ports;
	Map<int, Port> output_ports;
	Map<int, Control *> controls;

protected:
	static void _bind_methods();

public:
	void set_inputs(const String &p_inputs);
	String get_inputs() const;

	void set_outputs(const String &p_outputs);
	String get_outputs() const;

	bool is_valid_port_name(const String &p_name) const;

	void add_input_port(int p_id, int p_type, const String &p_name);
	void remove_input_port(int p_id);
	virtual int get_input_port_count() const override;
	bool has_input_port(int p_id) const;
	void clear_input_ports();

	void add_output_port(int p_id, int p_type, const String &p_name);
	void remove_output_port(int p_id);
	virtual int get_output_port_count() const override;
	bool has_output_port(int p_id) const;
	void clear_output_ports();

	void set_input_port_type(int p_id, int p_type);
	virtual PortType get_input_port_type(int p_id) const override;
	void set_input_port_name(int p_id, const String &p_name);
	virtual String get_input_port_name(int p_id) const override;

	void set_output_port_type(int p_id, int p_type);
	virtual PortType get_output_port_type(int p_id) const override;
	void set_output_port_name(int p_id, const String &p_name);
	virtual String get_output_port_name(int p_id) const override;

	int get_free_input_port_id() const;
	int get_free_output_port_id() const;

	void set_ctrl_pressed(Control *p_control, int p_index);
	Control *is_ctrl_pressed(int p_index);

	void set_editable(bool p_enabled);
	bool is_editable() const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeGroupBase();
};

class VisualShaderNodeExpression : public VisualShaderNodeGroupBase {
	GDCLASS(VisualShaderNodeExpression, VisualShaderNodeGroupBase);

protected:
	String expression = "";

	static void _bind_methods();

public:
	virtual String get_caption() const override;

	void set_expression(const String &p_expression);
	String get_expression() const;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	VisualShaderNodeExpression();
};

class VisualShaderNodeGlobalExpression : public VisualShaderNodeExpression {
	GDCLASS(VisualShaderNodeGlobalExpression, VisualShaderNodeExpression);

public:
	virtual String get_caption() const override;

	virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;

	VisualShaderNodeGlobalExpression();
};

extern String make_unique_id(VisualShader::Type p_type, int p_id, const String &p_name);

#endif // VISUAL_SHADER_H
