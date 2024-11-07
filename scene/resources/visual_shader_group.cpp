#include "visual_shader_group.h"

void VisualShaderGroup::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_group_name", "name"), &VisualShaderGroup::set_group_name);
	ClassDB::bind_method(D_METHOD("get_group_name"), &VisualShaderGroup::get_group_name);

	ClassDB::bind_method(D_METHOD("add_node", "node", "position", "id"), &VisualShaderGroup::add_node);
	ClassDB::bind_method(D_METHOD("get_node", "id"), &VisualShaderGroup::get_node);

	ClassDB::bind_method(D_METHOD("set_node_position", "id", "position"), &VisualShaderGroup::set_node_position);
	ClassDB::bind_method(D_METHOD("get_node_position", "id"), &VisualShaderGroup::get_node_position);

	ClassDB::bind_method(D_METHOD("get_node_list"), &VisualShaderGroup::get_node_ids);
	ClassDB::bind_method(D_METHOD("get_valid_node_id"), &VisualShaderGroup::get_valid_node_id);

	ClassDB::bind_method(D_METHOD("remove_node", "id"), &VisualShaderGroup::remove_node);
	ClassDB::bind_method(D_METHOD("replace_node", "id", "new_class"), &VisualShaderGroup::replace_node);

	ClassDB::bind_method(D_METHOD("is_node_connection", "from_node", "from_port", "to_node", "to_port"), &VisualShaderGroup::are_nodes_connected);
	ClassDB::bind_method(D_METHOD("can_connect_nodes", "from_node", "from_port", "to_node", "to_port"), &VisualShaderGroup::can_connect_nodes);

	ClassDB::bind_method(D_METHOD("connect_nodes", "from_node", "from_port", "to_node", "to_port"), &VisualShaderGroup::connect_nodes);
	ClassDB::bind_method(D_METHOD("disconnect_nodes", "from_node", "from_port", "to_node", "to_port"), &VisualShaderGroup::disconnect_nodes);
	ClassDB::bind_method(D_METHOD("connect_nodes_forced", "from_node", "from_port", "to_node", "to_port"), &VisualShaderGroup::connect_nodes_forced);

	// ClassDB::bind_method(D_METHOD("get_node_connections", "type"), &VisualShaderGroup::get_node_connections);

	ClassDB::bind_method(D_METHOD("attach_node_to_frame", "id", "frame"), &VisualShaderGroup::attach_node_to_frame);
	ClassDB::bind_method(D_METHOD("detach_node_from_frame", "id"), &VisualShaderGroup::detach_node_from_frame);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "group_name"), "set_group_name", "get_group_name");
	ADD_PROPERTY_DEFAULT("group_name", "Node group");
}

bool VisualShaderGroup::_set(const StringName &p_name, const Variant &p_value) {
	return graph->_set(p_name, p_value);
}

bool VisualShaderGroup::_get(const StringName &p_name, Variant &r_ret) const {
	return graph->_get(p_name, r_ret);
}

void VisualShaderGroup::_get_property_list(List<PropertyInfo> *p_list) const {
	graph->_get_property_list(p_list);
}

Ref<ShaderGraph> VisualShaderGroup::get_graph() const {
	return graph;
}

void VisualShaderGroup::set_group_name(const String &p_name) {
	if (group_name == p_name) {
		return;
	}

	group_name = p_name;
	emit_changed();
}

String VisualShaderGroup::get_group_name() const {
	return group_name;
}

void VisualShaderGroup::add_input_port(int p_id, VisualShaderNode::PortType p_type, const String &p_name) {
	input_ports[p_id] = Port{ p_type, p_name };
	emit_changed();
}

VisualShaderGroup::Port VisualShaderGroup::get_input_port(int p_id) const {
	return input_ports[p_id];
}

Vector<VisualShaderGroup::Port> VisualShaderGroup::get_input_ports() const {
	Vector<Port> ports;
	for (const KeyValue<int, Port> &E : input_ports) {
		ports.push_back(E.value);
	}
	return ports;
}

void VisualShaderGroup::remove_input_port(int p_id) {
	input_ports.erase(p_id);
	emit_changed();
}

void VisualShaderGroup::add_output_port(int p_id, VisualShaderNode::PortType p_type, const String &p_name) {
	output_ports[p_id] = Port{ p_type, p_name };
	emit_changed();
}

VisualShaderGroup::Port VisualShaderGroup::get_output_port(int p_id) const {
	return output_ports[p_id];
}

Vector<VisualShaderGroup::Port> VisualShaderGroup::get_output_ports() const {
	Vector<Port> ports;
	for (const KeyValue<int, Port> &E : output_ports) {
		ports.push_back(E.value);
	}
	return ports;
}

void VisualShaderGroup::remove_output_port(int p_id) {
	output_ports.erase(p_id);
	emit_changed();
}

void VisualShaderGroup::add_node(const Ref<VisualShaderNode> &p_node, const Vector2 &p_position, int p_id) {
	graph->add_node(p_node, p_position, p_id);
}

void VisualShaderGroup::set_node_position(int p_id, const Vector2 &p_position) {
	graph->set_node_position(p_id, p_position);
}

Vector2 VisualShaderGroup::get_node_position(int p_id) const {
	return graph->get_node_position(p_id);
}

Ref<VisualShaderNode> VisualShaderGroup::get_node(int p_id) const {
	return graph->get_node(p_id);
}

Vector<int> VisualShaderGroup::get_node_ids() const {
	return graph->get_node_ids();
}

int VisualShaderGroup::get_valid_node_id() const {
	return graph->get_valid_node_id();
}

int VisualShaderGroup::find_node_id(const Ref<VisualShaderNode> &p_node) const {
	return graph->find_node_id(p_node);
}

void VisualShaderGroup::remove_node(int p_id) {
	graph->remove_node(p_id);
}

void VisualShaderGroup::replace_node(int p_id, const StringName &p_new_class) {
	graph->replace_node(p_id, p_new_class);
}

bool VisualShaderGroup::are_nodes_connected(int p_from_node, int p_from_port, int p_to_node, int p_to_port) const {
	return graph->are_nodes_connected(p_from_node, p_from_port, p_to_node, p_to_port);
}

bool VisualShaderGroup::is_nodes_connected_relatively(int p_node, int p_target) const {
	return graph->is_nodes_connected_relatively(p_node, p_target);
}

bool VisualShaderGroup::can_connect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port) const {
	return graph->can_connect_nodes(p_from_node, p_from_port, p_to_node, p_to_port);
}

Error VisualShaderGroup::connect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	return graph->connect_nodes(p_from_node, p_from_port, p_to_node, p_to_port);
}

void VisualShaderGroup::disconnect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	graph->disconnect_nodes(p_from_node, p_from_port, p_to_node, p_to_port);
}

void VisualShaderGroup::connect_nodes_forced(int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	graph->connect_nodes_forced(p_from_node, p_from_port, p_to_node, p_to_port);
}

bool VisualShaderGroup::is_port_types_compatible(int p_a, int p_b) const {
	return graph->is_port_types_compatible(p_a, p_b);
}

void VisualShaderGroup::attach_node_to_frame(int p_node, int p_frame) {
	graph->attach_node_to_frame(p_node, p_frame);
}

void VisualShaderGroup::detach_node_from_frame(int p_node) {
	graph->detach_node_from_frame(p_node);
}

String VisualShaderGroup::get_reroute_parameter_name(int p_reroute_node) const {
	return graph->get_reroute_parameter_name(p_reroute_node);
}

void VisualShaderGroup::get_node_connections(List<ShaderGraph::Connection> *r_connections) const {
	graph->get_node_connections(r_connections);
}

VisualShaderGroup::VisualShaderGroup() {
	graph.instantiate();

	Ref<VisualShaderNodeGroupInput> input_node;
	input_node.instantiate();
	input_node->set_group(this);
	graph->nodes[NODE_ID_GROUP_INPUT].node = input_node;
	graph->nodes[NODE_ID_GROUP_INPUT].position = Vector2(0, 150);

	Ref<VisualShaderNodeGroupOutput> output_node;
	output_node.instantiate();
	output_node->set_group(this);
	graph->nodes[NODE_ID_GROUP_OUTPUT].node = output_node;
	graph->nodes[NODE_ID_GROUP_OUTPUT].position = Vector2(400, 150);

	group_name == TTR("Node group");
	add_input_port(0, VisualShaderNode::PORT_TYPE_SCALAR, "hardcoded_test in");
	add_output_port(0, VisualShaderNode::PORT_TYPE_SCALAR, "hardcoded_test out");
}

////////////// Group

void VisualShaderNodeGroup::_emit_changed() {
	emit_changed();
}

void VisualShaderNodeGroup::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_group", "group"), &VisualShaderNodeGroup::set_group);
	ClassDB::bind_method(D_METHOD("get_group"), &VisualShaderNodeGroup::get_group);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "group", PROPERTY_HINT_RESOURCE_TYPE, "VisualShaderGroup"), "set_group", "get_group");
}

String VisualShaderNodeGroup::get_caption() const {
	if (group.is_null()) {
		return TTR("Node group");
	}
	return group->get_group_name();
}

int VisualShaderNodeGroup::get_input_port_count() const {
	if (group.is_null()) {
		return 0;
	}
	return group->get_input_ports().size();
}

VisualShaderNode::PortType VisualShaderNodeGroup::get_input_port_type(int p_port) const {
	if (group.is_null()) {
		return PortType();
	}
	return group->get_input_port(p_port).type;
}

String VisualShaderNodeGroup::get_input_port_name(int p_port) const {
	if (group.is_null()) {
		return String();
	}
	return group->get_input_port(p_port).name;
}

int VisualShaderNodeGroup::get_output_port_count() const {
	if (group.is_null()) {
		return 0;
	}
	return group->get_output_ports().size();
}

VisualShaderNode::PortType VisualShaderNodeGroup::get_output_port_type(int p_port) const {
	if (group.is_null()) {
		return PortType();
	}
	return group->get_output_port(p_port).type;
}

String VisualShaderNodeGroup::get_output_port_name(int p_port) const {
	if (group.is_null()) {
		return String();
	}
	return group->get_output_port(p_port).name;
}

bool VisualShaderNodeGroup::is_show_prop_names() const {
	return true;
}

Vector<StringName> VisualShaderNodeGroup::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("group");
	return props;
}

bool VisualShaderNodeGroup::is_use_prop_slots() const {
	return true;
}

void VisualShaderNodeGroup::set_group(const Ref<VisualShaderGroup> &p_group) {
	if (group == p_group) {
		return;
	}
	group = p_group;
	emit_changed();
}

Ref<VisualShaderGroup> VisualShaderNodeGroup::get_group() const {
	return group;
}

String VisualShaderNodeGroup::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	// TODO: Implement.
	return String();
}

bool VisualShaderNodeGroup::is_output_port_expandable(int p_port) const {
	// TODO: Implement.
	return false;
}

VisualShaderNodeGroup::VisualShaderNodeGroup() {
	simple_decl = false;
}

// void VisualShaderNodeGroupInput::_bind_methods() {
// 	// TODO: Implement?
// }

// void VisualShaderNodeGroupInput::_validate_property(PropertyInfo &p_property) const {
// 	// TODO: Implement?
// }

void VisualShaderNodeGroupInput::set_group(VisualShaderGroup *p_group) {
	group = p_group;
}

VisualShaderGroup *VisualShaderNodeGroupInput::get_group() const {
	return group;
}

int VisualShaderNodeGroupInput::get_input_port_count() const {
	return 0;
}

VisualShaderNode::PortType VisualShaderNodeGroupInput::get_input_port_type(int p_port) const {
	return PortType();
}

String VisualShaderNodeGroupInput::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeGroupInput::get_output_port_count() const {
	if (!group) {
		return 0;
	}
	return group->get_input_ports().size();
}

VisualShaderNode::PortType VisualShaderNodeGroupInput::get_output_port_type(int p_port) const {
	if (!group) {
		return PortType();
	}
	return group->get_input_port(p_port).type;
}

String VisualShaderNodeGroupInput::get_output_port_name(int p_port) const {
	if (!group) {
		return String();
	}
	return group->get_input_port(p_port).name;
}

bool VisualShaderNodeGroupInput::is_output_port_expandable(int p_port) const {
	// TODO: Implement.
	return false;
}

String VisualShaderNodeGroupInput::get_caption() const {
	return "Group Input";
}

String VisualShaderNodeGroupInput::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	// TODO: Implement.
	return String();
}

Vector<StringName> VisualShaderNodeGroupInput::get_editable_properties() const {
	return Vector<StringName>();
}

VisualShaderNodeGroupInput::VisualShaderNodeGroupInput() {
}

void VisualShaderNodeGroupOutput::set_group(VisualShaderGroup *p_group) {
	group = p_group;
}

VisualShaderGroup *VisualShaderNodeGroupOutput::get_group() const {
	return group;
}

int VisualShaderNodeGroupOutput::get_input_port_count() const {
	if (!group) {
		return 0;
	}
	return group->get_output_ports().size();
}

VisualShaderNode::PortType VisualShaderNodeGroupOutput::get_input_port_type(int p_port) const {
	if (!group) {
		return PortType();
	}
	return group->get_output_port(p_port).type;
}

String VisualShaderNodeGroupOutput::get_input_port_name(int p_port) const {
	if (!group) {
		return String();
	}
	return group->get_output_port(p_port).name;
}

Variant VisualShaderNodeGroupOutput::get_input_port_default_value(int p_port) const {
	// TODO: Implement.
	return Variant();
}

int VisualShaderNodeGroupOutput::get_output_port_count() const {
	return 0;
}

VisualShaderNode::PortType VisualShaderNodeGroupOutput::get_output_port_type(int p_port) const {
	return PortType();
}

String VisualShaderNodeGroupOutput::get_output_port_name(int p_port) const {
	return String();
}

bool VisualShaderNodeGroupOutput::is_port_separator(int p_index) const {
	// TODO: Remove this?
	return false;
}

String VisualShaderNodeGroupOutput::get_caption() const {
	return "Group Output";
}

String VisualShaderNodeGroupOutput::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return String();
}

VisualShaderNodeGroupOutput::VisualShaderNodeGroupOutput() {
}
