/*************************************************************************/
/*  visual_shader_particle_nodes.cpp                                     */
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

#include "visual_shader_particle_nodes.h"

// VisualShaderNodeParticleEmitter

int VisualShaderNodeParticleEmitter::get_output_port_count() const {
	return 1;
}

VisualShaderNodeParticleEmitter::PortType VisualShaderNodeParticleEmitter::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeParticleEmitter::get_output_port_name(int p_port) const {
	if (p_port == 0) {
		return "position";
	}
	return String();
}

bool VisualShaderNodeParticleEmitter::has_output_port_preview(int p_port) const {
	return false;
}

VisualShaderNodeParticleEmitter::VisualShaderNodeParticleEmitter() {
}

// VisualShaderNodeParticleSphereEmitter

String VisualShaderNodeParticleSphereEmitter::get_caption() const {
	return "SphereEmitter";
}

int VisualShaderNodeParticleSphereEmitter::get_input_port_count() const {
	return 2;
}

VisualShaderNodeParticleSphereEmitter::PortType VisualShaderNodeParticleSphereEmitter::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeParticleSphereEmitter::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "radius";
	} else if (p_port == 1) {
		return "inner_radius";
	}
	return String();
}

String VisualShaderNodeParticleSphereEmitter::generate_global_per_node(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code;
	code += "vec3 __get_random_point_in_sphere(inout uint seed, float radius, float inner_radius) {\n";
	code += "	return __get_random_unit_vec3(seed) * __randf_range(seed, inner_radius, radius);\n";
	code += "}\n\n";
	return code;
}

String VisualShaderNodeParticleSphereEmitter::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	code += "	" + p_output_vars[0] + " = __get_random_point_in_sphere(__seed, " + (p_input_vars[0].is_empty() ? (String)get_input_port_default_value(0) : p_input_vars[0]) + ", " + (p_input_vars[1].is_empty() ? (String)get_input_port_default_value(1) : p_input_vars[1]) + ");\n";
	return code;
}

VisualShaderNodeParticleSphereEmitter::VisualShaderNodeParticleSphereEmitter() {
	set_input_port_default_value(0, 10.0);
	set_input_port_default_value(1, 0.0);
}

// VisualShaderNodeParticleBoxEmitter

String VisualShaderNodeParticleBoxEmitter::get_caption() const {
	return "BoxEmitter";
}

int VisualShaderNodeParticleBoxEmitter::get_input_port_count() const {
	return 1;
}

VisualShaderNodeParticleBoxEmitter::PortType VisualShaderNodeParticleBoxEmitter::get_input_port_type(int p_port) const {
	if (p_port == 0) {
		return PORT_TYPE_VECTOR;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeParticleBoxEmitter::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "extents";
	}
	return String();
}

String VisualShaderNodeParticleBoxEmitter::generate_global_per_node(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code;
	code += "vec3 __get_random_point_in_box(inout uint seed, vec3 extents) {\n";
	code += "	vec3 half_extents = extents / 2.0;\n";
	code += "	return vec3(__randf_range(seed, -half_extents.x, half_extents.x), __randf_range(seed, -half_extents.y, half_extents.y), __randf_range(seed, -half_extents.z, half_extents.z));\n";
	code += "}\n\n";
	return code;
}

String VisualShaderNodeParticleBoxEmitter::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	code += "	" + p_output_vars[0] + " = __get_random_point_in_box(__seed, " + (p_input_vars[0].is_empty() ? (String)get_input_port_default_value(0) : p_input_vars[0]) + ");\n";
	return code;
}

VisualShaderNodeParticleBoxEmitter::VisualShaderNodeParticleBoxEmitter() {
	set_input_port_default_value(0, Vector3(1.0, 1.0, 1.0));
}

// VisualShaderNodeParticleRingEmitter

String VisualShaderNodeParticleRingEmitter::get_caption() const {
	return "RingEmitter";
}

int VisualShaderNodeParticleRingEmitter::get_input_port_count() const {
	return 3;
}

VisualShaderNodeParticleRingEmitter::PortType VisualShaderNodeParticleRingEmitter::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeParticleRingEmitter::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "radius";
	} else if (p_port == 1) {
		return "inner_radius";
	} else if (p_port == 2) {
		return "height";
	}
	return String();
}

String VisualShaderNodeParticleRingEmitter::generate_global_per_node(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	String code;
	code += "vec3 __get_random_point_on_ring(inout uint seed, float radius, float inner_radius, float height) {\n";
	code += "	float angle = __rand_from_seed(seed) * PI * 2.0;\n";
	code += "	vec2 ring = vec2(sin(angle), cos(angle)) * __randf_range(seed, inner_radius, radius);\n";
	code += "	return vec3(ring.x, __randf_range(seed, min(0.0, height), max(0.0, height)), ring.y);\n";
	code += "}\n\n";
	return code;
}

String VisualShaderNodeParticleRingEmitter::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	code = "	" + p_output_vars[0] + " = __get_random_point_on_ring(__seed, " + (p_input_vars[0].is_empty() ? (String)get_input_port_default_value(0) : p_input_vars[0]) + ", " + (p_input_vars[1].is_empty() ? (String)get_input_port_default_value(1) : p_input_vars[1]) + ", " + (p_input_vars[2].is_empty() ? (String)get_input_port_default_value(2) : p_input_vars[2]) + ");\n";
	return code;
}

VisualShaderNodeParticleRingEmitter::VisualShaderNodeParticleRingEmitter() {
	set_input_port_default_value(0, 10.0);
	set_input_port_default_value(1, 0.0);
	set_input_port_default_value(2, 0.0);
}

// VisualShaderNodeParticleMultiplyByAxisAngle

void VisualShaderNodeParticleMultiplyByAxisAngle::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_degrees_mode", "enabled"), &VisualShaderNodeParticleMultiplyByAxisAngle::set_degrees_mode);
	ClassDB::bind_method(D_METHOD("is_degrees_mode"), &VisualShaderNodeParticleMultiplyByAxisAngle::is_degrees_mode);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "degrees_mode"), "set_degrees_mode", "is_degrees_mode");
}

String VisualShaderNodeParticleMultiplyByAxisAngle::get_caption() const {
	return "MultiplyByAxisAngle";
}

int VisualShaderNodeParticleMultiplyByAxisAngle::get_input_port_count() const {
	return 3;
}

VisualShaderNodeParticleMultiplyByAxisAngle::PortType VisualShaderNodeParticleMultiplyByAxisAngle::get_input_port_type(int p_port) const {
	if (p_port == 0 || p_port == 1) { // position, rotation_axis
		return PORT_TYPE_VECTOR;
	}
	return PORT_TYPE_SCALAR; // angle (degrees/radians)
}

String VisualShaderNodeParticleMultiplyByAxisAngle::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "position";
	}
	if (p_port == 1) {
		return "axis";
	}
	if (p_port == 2) {
		if (degrees_mode) {
			return "angle (degrees)";
		} else {
			return "angle (radians)";
		}
	}
	return String();
}

bool VisualShaderNodeParticleMultiplyByAxisAngle::is_show_prop_names() const {
	return true;
}

int VisualShaderNodeParticleMultiplyByAxisAngle::get_output_port_count() const {
	return 1;
}

VisualShaderNodeParticleMultiplyByAxisAngle::PortType VisualShaderNodeParticleMultiplyByAxisAngle::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeParticleMultiplyByAxisAngle::get_output_port_name(int p_port) const {
	return "position";
}

String VisualShaderNodeParticleMultiplyByAxisAngle::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	if (degrees_mode) {
		code += "	" + p_output_vars[0] + " = __build_rotation_mat3(" + (p_input_vars[1].is_empty() ? ("vec3" + (String)get_input_port_default_value(1)) : p_input_vars[1]) + ", radians(" + (p_input_vars[2].is_empty() ? (String)get_input_port_default_value(2) : p_input_vars[2]) + ")) * " + (p_input_vars[0].is_empty() ? "vec3(0.0)" : p_input_vars[0]) + ";\n";
	} else {
		code += "	" + p_output_vars[0] + " = __build_rotation_mat3(" + (p_input_vars[1].is_empty() ? ("vec3" + (String)get_input_port_default_value(1)) : p_input_vars[1]) + ", " + (p_input_vars[2].is_empty() ? (String)get_input_port_default_value(2) : p_input_vars[2]) + ") * " + (p_input_vars[0].is_empty() ? "vec3(0.0)" : p_input_vars[0]) + ";\n";
	}
	return code;
}

void VisualShaderNodeParticleMultiplyByAxisAngle::set_degrees_mode(bool p_enabled) {
	degrees_mode = p_enabled;
	emit_changed();
}

bool VisualShaderNodeParticleMultiplyByAxisAngle::is_degrees_mode() const {
	return degrees_mode;
}

Vector<StringName> VisualShaderNodeParticleMultiplyByAxisAngle::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("degrees_mode");
	props.push_back("axis_amount");
	return props;
}

bool VisualShaderNodeParticleMultiplyByAxisAngle::has_output_port_preview(int p_port) const {
	return false;
}

VisualShaderNodeParticleMultiplyByAxisAngle::VisualShaderNodeParticleMultiplyByAxisAngle() {
	set_input_port_default_value(1, Vector3(1, 0, 0));
	set_input_port_default_value(2, 0.0);
}

// VisualShaderNodeParticleConeVelocity

String VisualShaderNodeParticleConeVelocity::get_caption() const {
	return "ConeVelocity";
}

int VisualShaderNodeParticleConeVelocity::get_input_port_count() const {
	return 2;
}

VisualShaderNodeParticleConeVelocity::PortType VisualShaderNodeParticleConeVelocity::get_input_port_type(int p_port) const {
	if (p_port == 0) {
		return PORT_TYPE_VECTOR;
	} else if (p_port == 1) {
		return PORT_TYPE_SCALAR;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeParticleConeVelocity::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "direction";
	} else if (p_port == 1) {
		return "spread(degrees)";
	}
	return String();
}

int VisualShaderNodeParticleConeVelocity::get_output_port_count() const {
	return 1;
}

VisualShaderNodeParticleConeVelocity::PortType VisualShaderNodeParticleConeVelocity::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeParticleConeVelocity::get_output_port_name(int p_port) const {
	if (p_port == 0) {
		return "velocity";
	}
	return String();
}

bool VisualShaderNodeParticleConeVelocity::has_output_port_preview(int p_port) const {
	return false;
}

String VisualShaderNodeParticleConeVelocity::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	code += "	__radians = radians(" + (p_input_vars[1].is_empty() ? (String)get_input_port_default_value(1) : p_input_vars[1]) + ");\n";
	code += "	__scalar_buff1 = __rand_from_seed_m1_p1(__seed) * __radians;\n";
	code += "	__scalar_buff2 = __rand_from_seed_m1_p1(__seed) * __radians;\n";
	code += "	__vec3_buff1 = " + (p_input_vars[0].is_empty() ? "vec3" + (String)get_input_port_default_value(0) : p_input_vars[0]) + ";\n";
	code += "	__scalar_buff1 += __vec3_buff1.z != 0.0 ? atan(__vec3_buff1.x, __vec3_buff1.z) : sign(__vec3_buff1.x) * (PI / 2.0);\n";
	code += "	__scalar_buff2 += __vec3_buff1.z != 0.0 ? atan(__vec3_buff1.y, abs(__vec3_buff1.z)) : (__vec3_buff1.x != 0.0 ? atan(__vec3_buff1.y, abs(__vec3_buff1.x)) : sign(__vec3_buff1.y) * (PI / 2.0));\n";
	code += "	__vec3_buff1 = vec3(sin(__scalar_buff1), 0.0, cos(__scalar_buff1));\n";
	code += "	__vec3_buff2 = vec3(0.0, sin(__scalar_buff2), cos(__scalar_buff2));\n";
	code += "	__vec3_buff2.z = __vec3_buff2.z / max(0.0001, sqrt(abs(__vec3_buff2.z)));\n";
	code += "	" + p_output_vars[0] + " = normalize(vec3(__vec3_buff1.x * __vec3_buff2.z, __vec3_buff2.y, __vec3_buff1.z * __vec3_buff2.z));\n";
	return code;
}

VisualShaderNodeParticleConeVelocity::VisualShaderNodeParticleConeVelocity() {
	set_input_port_default_value(0, Vector3(1, 0, 0));
	set_input_port_default_value(1, 45.0);
}

// VisualShaderNodeParticleRandomness

void VisualShaderNodeParticleRandomness::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_op_type", "type"), &VisualShaderNodeParticleRandomness::set_op_type);
	ClassDB::bind_method(D_METHOD("get_op_type"), &VisualShaderNodeParticleRandomness::get_op_type);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "op_type", PROPERTY_HINT_ENUM, "Scalar,Vector"), "set_op_type", "get_op_type");

	BIND_ENUM_CONSTANT(OP_TYPE_SCALAR);
	BIND_ENUM_CONSTANT(OP_TYPE_VECTOR);
	BIND_ENUM_CONSTANT(OP_TYPE_MAX);
}

Vector<StringName> VisualShaderNodeParticleRandomness::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("op_type");
	return props;
}

String VisualShaderNodeParticleRandomness::get_caption() const {
	return "ParticleRandomness";
}

int VisualShaderNodeParticleRandomness::get_output_port_count() const {
	return 1;
}

VisualShaderNodeParticleRandomness::PortType VisualShaderNodeParticleRandomness::get_output_port_type(int p_port) const {
	if (op_type == OP_TYPE_VECTOR) {
		return PORT_TYPE_VECTOR;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeParticleRandomness::get_output_port_name(int p_port) const {
	return "random";
}

int VisualShaderNodeParticleRandomness::get_input_port_count() const {
	return 2;
}

VisualShaderNodeParticleRandomness::PortType VisualShaderNodeParticleRandomness::get_input_port_type(int p_port) const {
	if (op_type == OP_TYPE_VECTOR) {
		return PORT_TYPE_VECTOR;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeParticleRandomness::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "min";
	} else if (p_port == 1) {
		return "max";
	}
	return String();
}

String VisualShaderNodeParticleRandomness::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	if (op_type == OP_TYPE_SCALAR) {
		code += vformat("	%s = __randf_range(__seed, %s, %s);\n", p_output_vars[0], p_input_vars[0].is_empty() ? (String)get_input_port_default_value(0) : p_input_vars[0], p_input_vars[1].is_empty() ? (String)get_input_port_default_value(1) : p_input_vars[1]);
	} else if (op_type == OP_TYPE_VECTOR) {
		code += vformat("	%s = __randv_range(__seed, %s, %s);\n", p_output_vars[0], p_input_vars[0].is_empty() ? (String)get_input_port_default_value(0) : p_input_vars[0], p_input_vars[1].is_empty() ? (String)get_input_port_default_value(1) : p_input_vars[1]);
	}
	return code;
}

void VisualShaderNodeParticleRandomness::set_op_type(OpType p_op_type) {
	ERR_FAIL_INDEX(int(p_op_type), int(OP_TYPE_MAX));
	if (op_type == p_op_type) {
		return;
	}
	if (p_op_type == OP_TYPE_SCALAR) {
		set_input_port_default_value(0, 0.0);
		set_input_port_default_value(1, 1.0);
	} else {
		set_input_port_default_value(0, Vector3(-1.0, -1.0, -1.0));
		set_input_port_default_value(1, Vector3(1.0, 1.0, 1.0));
	}
	op_type = p_op_type;
	emit_changed();
}

VisualShaderNodeParticleRandomness::OpType VisualShaderNodeParticleRandomness::get_op_type() const {
	return op_type;
}

bool VisualShaderNodeParticleRandomness::has_output_port_preview(int p_port) const {
	return false;
}

VisualShaderNodeParticleRandomness::VisualShaderNodeParticleRandomness() {
	set_input_port_default_value(0, 0.0);
	set_input_port_default_value(1, 1.0);
}

// VisualShaderNodeParticleAccelerator

void VisualShaderNodeParticleAccelerator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &VisualShaderNodeParticleAccelerator::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &VisualShaderNodeParticleAccelerator::get_mode);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Linear,Radial,Tangential"), "set_mode", "get_mode");

	BIND_ENUM_CONSTANT(MODE_LINEAR);
	BIND_ENUM_CONSTANT(MODE_RADIAL)
	BIND_ENUM_CONSTANT(MODE_TANGENTIAL);
	BIND_ENUM_CONSTANT(MODE_MAX);
}

Vector<StringName> VisualShaderNodeParticleAccelerator::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("mode");
	return props;
}

String VisualShaderNodeParticleAccelerator::get_caption() const {
	return "ParticleAccelerator";
}

int VisualShaderNodeParticleAccelerator::get_output_port_count() const {
	return 1;
}

VisualShaderNodeParticleAccelerator::PortType VisualShaderNodeParticleAccelerator::get_output_port_type(int p_port) const {
	return PORT_TYPE_VECTOR;
}

String VisualShaderNodeParticleAccelerator::get_output_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeParticleAccelerator::get_input_port_count() const {
	return 3;
}

VisualShaderNodeParticleAccelerator::PortType VisualShaderNodeParticleAccelerator::get_input_port_type(int p_port) const {
	if (p_port == 0) {
		return PORT_TYPE_VECTOR;
	} else if (p_port == 1) {
		return PORT_TYPE_SCALAR;
	} else if (p_port == 2) {
		return PORT_TYPE_VECTOR;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeParticleAccelerator::get_input_port_name(int p_port) const {
	if (p_port == 0) {
		return "amount";
	} else if (p_port == 1) {
		return "randomness";
	} else if (p_port == 2) {
		return "axis";
	}
	return String();
}

String VisualShaderNodeParticleAccelerator::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	switch (mode) {
		case MODE_LINEAR:
			code += "	" + p_output_vars[0] + " = length(VELOCITY) > 0.0 ? " + "normalize(VELOCITY) * " + (p_input_vars[0].is_empty() ? "vec3" + (String)get_input_port_default_value(0) : p_input_vars[0]) + " * mix(1.0, __rand_from_seed(__seed), " + (p_input_vars[1].is_empty() ? (String)get_input_port_default_value(1) : p_input_vars[1]) + ") : vec3(0.0);\n";
			break;
		case MODE_RADIAL:
			code += "	" + p_output_vars[0] + " = length(__diff) > 0.0 ? __ndiff * " + (p_input_vars[0].is_empty() ? "vec3" + (String)get_input_port_default_value(0) : p_input_vars[0]) + " * mix(1.0, __rand_from_seed(__seed), " + (p_input_vars[1].is_empty() ? (String)get_input_port_default_value(1) : p_input_vars[1]) + ") : vec3(0.0);\n";
			break;
		case MODE_TANGENTIAL:
			code += "	__vec3_buff1 = cross(__ndiff, normalize(" + (p_input_vars[2].is_empty() ? "vec3" + (String)get_input_port_default_value(2) : p_input_vars[2]) + "));\n";
			code += "	" + p_output_vars[0] + " = length(__vec3_buff1) > 0.0 ? normalize(__vec3_buff1) * (" + (p_input_vars[0].is_empty() ? "vec3" + (String)get_input_port_default_value(0) : p_input_vars[0]) + " * mix(1.0, __rand_from_seed(__seed), " + (p_input_vars[1].is_empty() ? (String)get_input_port_default_value(1) : p_input_vars[1]) + ")) : vec3(0.0);\n";
			break;
		default:
			break;
	}

	return code;
}

void VisualShaderNodeParticleAccelerator::set_mode(Mode p_mode) {
	ERR_FAIL_INDEX(int(p_mode), int(MODE_MAX));
	if (mode == p_mode) {
		return;
	}
	mode = p_mode;
	emit_changed();
}

VisualShaderNodeParticleAccelerator::Mode VisualShaderNodeParticleAccelerator::get_mode() const {
	return mode;
}

bool VisualShaderNodeParticleAccelerator::has_output_port_preview(int p_port) const {
	return false;
}

VisualShaderNodeParticleAccelerator::VisualShaderNodeParticleAccelerator() {
	set_input_port_default_value(0, Vector3(1, 1, 1));
	set_input_port_default_value(1, 0.0);
	set_input_port_default_value(2, Vector3(0, -9.8, 0));
}

// VisualShaderNodeParticleOutput

String VisualShaderNodeParticleOutput::get_caption() const {
	if (shader_type == VisualShader::TYPE_START) {
		return "StartOutput";
	} else if (shader_type == VisualShader::TYPE_PROCESS) {
		return "ProcessOutput";
	} else if (shader_type == VisualShader::TYPE_COLLIDE) {
		return "CollideOutput";
	} else if (shader_type == VisualShader::TYPE_START_CUSTOM) {
		return "CustomStartOutput";
	} else if (shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
		return "CustomProcessOutput";
	}
	return String();
}

int VisualShaderNodeParticleOutput::get_input_port_count() const {
	if (shader_type == VisualShader::TYPE_START) {
		return 8;
	} else if (shader_type == VisualShader::TYPE_COLLIDE) {
		return 5;
	} else if (shader_type == VisualShader::TYPE_START_CUSTOM || shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
		return 6;
	} else { // TYPE_PROCESS
		return 7;
	}
	return 0;
}

VisualShaderNodeParticleOutput::PortType VisualShaderNodeParticleOutput::get_input_port_type(int p_port) const {
	switch (p_port) {
		case 0:
			if (shader_type == VisualShader::TYPE_START_CUSTOM || shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
				return PORT_TYPE_VECTOR; // custom.rgb
			}
			return PORT_TYPE_BOOLEAN; // active
		case 1:
			if (shader_type == VisualShader::TYPE_START_CUSTOM || shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
				break; // custom.a (scalar)
			}
			return PORT_TYPE_VECTOR; // velocity
		case 2:
			return PORT_TYPE_VECTOR; // color & velocity
		case 3:
			if (shader_type == VisualShader::TYPE_START_CUSTOM || shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
				return PORT_TYPE_VECTOR; // color
			}
			break; // alpha (scalar)
		case 4:
			if (shader_type == VisualShader::TYPE_START_CUSTOM || shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
				break; // alpha
			}
			if (shader_type == VisualShader::TYPE_PROCESS) {
				break; // scale
			}
			if (shader_type == VisualShader::TYPE_COLLIDE) {
				return PORT_TYPE_TRANSFORM; // transform
			}
			return PORT_TYPE_VECTOR; // position
		case 5:
			if (shader_type == VisualShader::TYPE_START_CUSTOM || shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
				return PORT_TYPE_TRANSFORM; // transform
			}
			if (shader_type == VisualShader::TYPE_PROCESS) {
				return PORT_TYPE_VECTOR; // rotation_axis
			}
			break; // scale (scalar)
		case 6:
			if (shader_type == VisualShader::TYPE_START) {
				return PORT_TYPE_VECTOR; // rotation_axis
			}
			break;
		case 7:
			break; // angle (scalar)
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeParticleOutput::get_input_port_name(int p_port) const {
	String port_name;
	switch (p_port) {
		case 0:
			if (shader_type == VisualShader::TYPE_START_CUSTOM || shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
				port_name = "custom";
				break;
			}
			port_name = "active";
			break;
		case 1:
			if (shader_type == VisualShader::TYPE_START_CUSTOM || shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
				port_name = "custom_alpha";
				break;
			}
			port_name = "velocity";
			break;
		case 2:
			if (shader_type == VisualShader::TYPE_START_CUSTOM || shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
				port_name = "velocity";
				break;
			}
			port_name = "color";
			break;
		case 3:
			if (shader_type == VisualShader::TYPE_START_CUSTOM || shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
				port_name = "color";
				break;
			}
			port_name = "alpha";
			break;
		case 4:
			if (shader_type == VisualShader::TYPE_START_CUSTOM || shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
				port_name = "alpha";
				break;
			}
			if (shader_type == VisualShader::TYPE_PROCESS) {
				port_name = "scale";
				break;
			}
			if (shader_type == VisualShader::TYPE_COLLIDE) {
				port_name = "transform";
				break;
			}
			port_name = "position";
			break;
		case 5:
			if (shader_type == VisualShader::TYPE_START_CUSTOM || shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
				port_name = "transform";
				break;
			}
			if (shader_type == VisualShader::TYPE_PROCESS) {
				port_name = "rotation_axis";
				break;
			}
			port_name = "scale";
			break;
		case 6:
			if (shader_type == VisualShader::TYPE_PROCESS) {
				port_name = "angle_in_radians";
				break;
			}
			port_name = "rotation_axis";
			break;
		case 7:
			port_name = "angle_in_radians";
			break;
		default:
			break;
	}
	if (!port_name.is_empty()) {
		return port_name.capitalize();
	}
	return String();
}

bool VisualShaderNodeParticleOutput::is_port_separator(int p_index) const {
	if (shader_type == VisualShader::TYPE_START || shader_type == VisualShader::TYPE_PROCESS) {
		String name = get_input_port_name(p_index);
		return bool(name == "Scale");
	}
	if (shader_type == VisualShader::TYPE_START_CUSTOM || shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
		String name = get_input_port_name(p_index);
		return bool(name == "Velocity");
	}
	return false;
}

String VisualShaderNodeParticleOutput::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	String tab = "	";

	if (shader_type == VisualShader::TYPE_START_CUSTOM || shader_type == VisualShader::TYPE_PROCESS_CUSTOM) {
		if (!p_input_vars[0].is_empty()) { // custom.rgb
			code += tab + "CUSTOM.rgb = " + p_input_vars[0] + ";\n";
		}
		if (!p_input_vars[1].is_empty()) { // custom.a
			code += tab + "CUSTOM.a = " + p_input_vars[1] + ";\n";
		}
		if (!p_input_vars[2].is_empty()) { // velocity
			code += tab + "VELOCITY = " + p_input_vars[2] + ";\n";
		}
		if (!p_input_vars[3].is_empty()) { // color.rgb
			code += tab + "COLOR.rgb = " + p_input_vars[3] + ";\n";
		}
		if (!p_input_vars[4].is_empty()) { // color.a
			code += tab + "COLOR.a = " + p_input_vars[4] + ";\n";
		}
		if (!p_input_vars[5].is_empty()) { // transform
			code += tab + "TRANSFORM = " + p_input_vars[5] + ";\n";
		}
	} else {
		if (!p_input_vars[0].is_empty()) { // active (begin)
			code += tab + "ACTIVE = " + p_input_vars[0] + ";\n";
			code += tab + "if(ACTIVE) {\n";
			tab += "	";
		}
		if (!p_input_vars[1].is_empty()) { // velocity
			code += tab + "VELOCITY = " + p_input_vars[1] + ";\n";
		}
		if (!p_input_vars[2].is_empty()) { // color
			code += tab + "COLOR.rgb = " + p_input_vars[2] + ";\n";
		}
		if (!p_input_vars[3].is_empty()) { // alpha
			code += tab + "COLOR.a = " + p_input_vars[3] + ";\n";
		}

		// position
		if (shader_type == VisualShader::TYPE_START) {
			code += tab + "if (RESTART_POSITION) {\n";
			if (!p_input_vars[4].is_empty()) {
				code += tab + "	TRANSFORM = mat4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(" + p_input_vars[4] + ", 1.0));\n";
			} else {
				code += tab + "	TRANSFORM = mat4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));\n";
			}
			code += tab + "	if (RESTART_VELOCITY) {\n";
			code += tab + "		VELOCITY = (EMISSION_TRANSFORM * vec4(VELOCITY, 0.0)).xyz;\n";
			code += tab + "	}\n";
			code += tab + "	TRANSFORM = EMISSION_TRANSFORM * TRANSFORM;\n";
			code += tab + "}\n";
		} else if (shader_type == VisualShader::TYPE_COLLIDE) { // position
			if (!p_input_vars[4].is_empty()) {
				code += tab + "TRANSFORM = " + p_input_vars[4] + ";\n";
			}
		}

		if (shader_type == VisualShader::TYPE_START || shader_type == VisualShader::TYPE_PROCESS) {
			int scale = 5;
			int rotation_axis = 6;
			int rotation = 7;
			if (shader_type == VisualShader::TYPE_PROCESS) {
				scale = 4;
				rotation_axis = 5;
				rotation = 6;
			}
			String op;
			if (shader_type == VisualShader::TYPE_START) {
				op = "*=";
			} else {
				op = "=";
			}

			if (!p_input_vars[rotation].is_empty()) { // rotation_axis & angle_in_radians
				String axis;
				if (p_input_vars[rotation_axis].is_empty()) {
					axis = "vec3(0, 1, 0)";
				} else {
					axis = p_input_vars[rotation_axis];
				}
				code += tab + "TRANSFORM " + op + " __build_rotation_mat4(" + axis + ", " + p_input_vars[rotation] + ");\n";
			}
			if (!p_input_vars[scale].is_empty()) { // scale
				code += tab + "TRANSFORM " + op + " mat4(vec4(" + p_input_vars[scale] + ", 0, 0, 0), vec4(0, " + p_input_vars[scale] + ", 0, 0), vec4(0, 0, " + p_input_vars[scale] + ", 0), vec4(0, 0, 0, 1));\n";
			}
		}
		if (!p_input_vars[0].is_empty()) { // active (end)
			code += "	}\n";
		}
	}
	return code;
}

VisualShaderNodeParticleOutput::VisualShaderNodeParticleOutput() {
}

// EmitParticle

Vector<StringName> VisualShaderNodeParticleEmit::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("flags");
	return props;
}

void VisualShaderNodeParticleEmit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_flags", "flags"), &VisualShaderNodeParticleEmit::set_flags);
	ClassDB::bind_method(D_METHOD("get_flags"), &VisualShaderNodeParticleEmit::get_flags);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "flags", PROPERTY_HINT_FLAGS, "Position,RotScale,Velocity,Color,Custom"), "set_flags", "get_flags");

	BIND_ENUM_CONSTANT(EMIT_FLAG_POSITION);
	BIND_ENUM_CONSTANT(EMIT_FLAG_ROT_SCALE);
	BIND_ENUM_CONSTANT(EMIT_FLAG_VELOCITY);
	BIND_ENUM_CONSTANT(EMIT_FLAG_COLOR);
	BIND_ENUM_CONSTANT(EMIT_FLAG_CUSTOM);
}

String VisualShaderNodeParticleEmit::get_caption() const {
	return "EmitParticle";
}

int VisualShaderNodeParticleEmit::get_input_port_count() const {
	return 7;
}

VisualShaderNodeParticleEmit::PortType VisualShaderNodeParticleEmit::get_input_port_type(int p_port) const {
	switch (p_port) {
		case 0:
			return PORT_TYPE_BOOLEAN;
		case 1:
			return PORT_TYPE_TRANSFORM;
		case 2:
			return PORT_TYPE_VECTOR;
		case 3:
			return PORT_TYPE_VECTOR;
		case 4:
			return PORT_TYPE_SCALAR;
		case 5:
			return PORT_TYPE_VECTOR;
		case 6:
			return PORT_TYPE_SCALAR;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeParticleEmit::get_input_port_name(int p_port) const {
	switch (p_port) {
		case 0:
			return "condition";
		case 1:
			return "transform";
		case 2:
			return "velocity";
		case 3:
			return "color";
		case 4:
			return "alpha";
		case 5:
			return "custom";
		case 6:
			return "custom_alpha";
	}
	return String();
}

int VisualShaderNodeParticleEmit::get_output_port_count() const {
	return 0;
}

VisualShaderNodeParticleEmit::PortType VisualShaderNodeParticleEmit::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeParticleEmit::get_output_port_name(int p_port) const {
	return String();
}

void VisualShaderNodeParticleEmit::add_flag(EmitFlags p_flag) {
	flags |= p_flag;
	emit_changed();
}

bool VisualShaderNodeParticleEmit::has_flag(EmitFlags p_flag) const {
	return flags & p_flag;
}

void VisualShaderNodeParticleEmit::set_flags(EmitFlags p_flags) {
	flags = (int)p_flags;
	emit_changed();
}

VisualShaderNodeParticleEmit::EmitFlags VisualShaderNodeParticleEmit::get_flags() const {
	return EmitFlags(flags);
}

bool VisualShaderNodeParticleEmit::is_show_prop_names() const {
	return true;
}

bool VisualShaderNodeParticleEmit::is_generate_input_var(int p_port) const {
	if (p_port == 0) {
		if (!is_input_port_connected(0)) {
			return false;
		}
	}
	return true;
}

String VisualShaderNodeParticleEmit::get_input_port_default_hint(int p_port) const {
	switch (p_port) {
		case 1:
			return "default";
		case 2:
			return "default";
		case 3:
			return "default";
		case 4:
			return "default";
		case 5:
			return "default";
		case 6:
			return "default";
	}
	return String();
}

String VisualShaderNodeParticleEmit::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	String code;
	String tab;
	bool default_condition = false;

	if (!is_input_port_connected(0)) {
		default_condition = true;
		if (get_input_port_default_value(0)) {
			tab = "	";
		} else {
			return code;
		}
	} else {
		tab = "		";
	}

	String transform;
	if (p_input_vars[1].is_empty()) {
		transform = "TRANSFORM";
	} else {
		transform = p_input_vars[1];
	}

	String velocity;
	if (p_input_vars[2].is_empty()) {
		velocity = "VELOCITY";
	} else {
		velocity = p_input_vars[2];
	}

	String color;
	if (p_input_vars[3].is_empty()) {
		color = "COLOR.rgb";
	} else {
		color = p_input_vars[3];
	}

	String alpha;
	if (p_input_vars[4].is_empty()) {
		alpha = "COLOR.a";
	} else {
		alpha = p_input_vars[4];
	}

	String custom;
	if (p_input_vars[5].is_empty()) {
		custom = "CUSTOM.rgb";
	} else {
		custom = p_input_vars[5];
	}

	String custom_alpha;
	if (p_input_vars[6].is_empty()) {
		custom_alpha = "CUSTOM.a";
	} else {
		custom_alpha = p_input_vars[6];
	}

	List<String> flags_arr;

	if (has_flag(EmitFlags::EMIT_FLAG_POSITION)) {
		flags_arr.push_back("FLAG_EMIT_POSITION");
	}
	if (has_flag(EmitFlags::EMIT_FLAG_ROT_SCALE)) {
		flags_arr.push_back("FLAG_EMIT_ROT_SCALE");
	}
	if (has_flag(EmitFlags::EMIT_FLAG_VELOCITY)) {
		flags_arr.push_back("FLAG_EMIT_VELOCITY");
	}
	if (has_flag(EmitFlags::EMIT_FLAG_COLOR)) {
		flags_arr.push_back("FLAG_EMIT_COLOR");
	}
	if (has_flag(EmitFlags::EMIT_FLAG_CUSTOM)) {
		flags_arr.push_back("FLAG_EMIT_CUSTOM");
	}

	String flags;

	for (int i = 0; i < flags_arr.size(); i++) {
		if (i > 0) {
			flags += "|";
		}
		flags += flags_arr[i];
	}

	if (flags.is_empty()) {
		flags = "uint(0)";
	}

	if (!default_condition) {
		code += "	if (" + p_input_vars[0] + ") {\n";
	}

	code += tab + "emit_subparticle(" + transform + ", " + velocity + ", vec4(" + color + ", " + alpha + "), vec4(" + custom + ", " + custom_alpha + "), " + flags + ");\n";

	if (!default_condition) {
		code += "	}\n";
	}

	return code;
}

VisualShaderNodeParticleEmit::VisualShaderNodeParticleEmit() {
	set_input_port_default_value(0, true);
}
