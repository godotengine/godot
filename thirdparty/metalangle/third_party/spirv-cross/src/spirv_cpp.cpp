/*
 * Copyright 2015-2020 Arm Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "spirv_cpp.hpp"

using namespace spv;
using namespace SPIRV_CROSS_NAMESPACE;
using namespace std;

void CompilerCPP::emit_buffer_block(const SPIRVariable &var)
{
	add_resource_name(var.self);

	auto &type = get<SPIRType>(var.basetype);
	auto instance_name = to_name(var.self);

	uint32_t descriptor_set = ir.meta[var.self].decoration.set;
	uint32_t binding = ir.meta[var.self].decoration.binding;

	emit_block_struct(type);
	auto buffer_name = to_name(type.self);

	statement("internal::Resource<", buffer_name, type_to_array_glsl(type), "> ", instance_name, "__;");
	statement_no_indent("#define ", instance_name, " __res->", instance_name, "__.get()");
	resource_registrations.push_back(
	    join("s.register_resource(", instance_name, "__", ", ", descriptor_set, ", ", binding, ");"));
	statement("");
}

void CompilerCPP::emit_interface_block(const SPIRVariable &var)
{
	add_resource_name(var.self);

	auto &type = get<SPIRType>(var.basetype);

	const char *qual = var.storage == StorageClassInput ? "StageInput" : "StageOutput";
	const char *lowerqual = var.storage == StorageClassInput ? "stage_input" : "stage_output";
	auto instance_name = to_name(var.self);
	uint32_t location = ir.meta[var.self].decoration.location;

	string buffer_name;
	auto flags = ir.meta[type.self].decoration.decoration_flags;
	if (flags.get(DecorationBlock))
	{
		emit_block_struct(type);
		buffer_name = to_name(type.self);
	}
	else
		buffer_name = type_to_glsl(type);

	statement("internal::", qual, "<", buffer_name, type_to_array_glsl(type), "> ", instance_name, "__;");
	statement_no_indent("#define ", instance_name, " __res->", instance_name, "__.get()");
	resource_registrations.push_back(join("s.register_", lowerqual, "(", instance_name, "__", ", ", location, ");"));
	statement("");
}

void CompilerCPP::emit_shared(const SPIRVariable &var)
{
	add_resource_name(var.self);

	auto instance_name = to_name(var.self);
	statement(CompilerGLSL::variable_decl(var), ";");
	statement_no_indent("#define ", instance_name, " __res->", instance_name);
}

void CompilerCPP::emit_uniform(const SPIRVariable &var)
{
	add_resource_name(var.self);

	auto &type = get<SPIRType>(var.basetype);
	auto instance_name = to_name(var.self);

	uint32_t descriptor_set = ir.meta[var.self].decoration.set;
	uint32_t binding = ir.meta[var.self].decoration.binding;
	uint32_t location = ir.meta[var.self].decoration.location;

	string type_name = type_to_glsl(type);
	remap_variable_type_name(type, instance_name, type_name);

	if (type.basetype == SPIRType::Image || type.basetype == SPIRType::SampledImage ||
	    type.basetype == SPIRType::AtomicCounter)
	{
		statement("internal::Resource<", type_name, type_to_array_glsl(type), "> ", instance_name, "__;");
		statement_no_indent("#define ", instance_name, " __res->", instance_name, "__.get()");
		resource_registrations.push_back(
		    join("s.register_resource(", instance_name, "__", ", ", descriptor_set, ", ", binding, ");"));
	}
	else
	{
		statement("internal::UniformConstant<", type_name, type_to_array_glsl(type), "> ", instance_name, "__;");
		statement_no_indent("#define ", instance_name, " __res->", instance_name, "__.get()");
		resource_registrations.push_back(
		    join("s.register_uniform_constant(", instance_name, "__", ", ", location, ");"));
	}

	statement("");
}

void CompilerCPP::emit_push_constant_block(const SPIRVariable &var)
{
	add_resource_name(var.self);

	auto &type = get<SPIRType>(var.basetype);
	auto &flags = ir.meta[var.self].decoration.decoration_flags;
	if (flags.get(DecorationBinding) || flags.get(DecorationDescriptorSet))
		SPIRV_CROSS_THROW("Push constant blocks cannot be compiled to GLSL with Binding or Set syntax. "
		                  "Remap to location with reflection API first or disable these decorations.");

	emit_block_struct(type);
	auto buffer_name = to_name(type.self);
	auto instance_name = to_name(var.self);

	statement("internal::PushConstant<", buffer_name, type_to_array_glsl(type), "> ", instance_name, ";");
	statement_no_indent("#define ", instance_name, " __res->", instance_name, ".get()");
	resource_registrations.push_back(join("s.register_push_constant(", instance_name, "__", ");"));
	statement("");
}

void CompilerCPP::emit_block_struct(SPIRType &type)
{
	// C++ can't do interface blocks, so we fake it by emitting a separate struct.
	// However, these structs are not allowed to alias anything, so remove it before
	// emitting the struct.
	//
	// The type we have here needs to be resolved to the non-pointer type so we can remove aliases.
	auto &self = get<SPIRType>(type.self);
	self.type_alias = 0;
	emit_struct(self);
}

void CompilerCPP::emit_resources()
{
	for (auto &id : ir.ids)
	{
		if (id.get_type() == TypeConstant)
		{
			auto &c = id.get<SPIRConstant>();

			bool needs_declaration = c.specialization || c.is_used_as_lut;

			if (needs_declaration)
			{
				if (!options.vulkan_semantics && c.specialization)
				{
					c.specialization_constant_macro_name =
					    constant_value_macro_name(get_decoration(c.self, DecorationSpecId));
				}
				emit_constant(c);
			}
		}
		else if (id.get_type() == TypeConstantOp)
		{
			emit_specialization_constant_op(id.get<SPIRConstantOp>());
		}
	}

	// Output all basic struct types which are not Block or BufferBlock as these are declared inplace
	// when such variables are instantiated.
	for (auto &id : ir.ids)
	{
		if (id.get_type() == TypeType)
		{
			auto &type = id.get<SPIRType>();
			if (type.basetype == SPIRType::Struct && type.array.empty() && !type.pointer &&
			    (!ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock) &&
			     !ir.meta[type.self].decoration.decoration_flags.get(DecorationBufferBlock)))
			{
				emit_struct(type);
			}
		}
	}

	statement("struct Resources : ", resource_type);
	begin_scope();

	// Output UBOs and SSBOs
	for (auto &id : ir.ids)
	{
		if (id.get_type() == TypeVariable)
		{
			auto &var = id.get<SPIRVariable>();
			auto &type = get<SPIRType>(var.basetype);

			if (var.storage != StorageClassFunction && type.pointer && type.storage == StorageClassUniform &&
			    !is_hidden_variable(var) &&
			    (ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock) ||
			     ir.meta[type.self].decoration.decoration_flags.get(DecorationBufferBlock)))
			{
				emit_buffer_block(var);
			}
		}
	}

	// Output push constant blocks
	for (auto &id : ir.ids)
	{
		if (id.get_type() == TypeVariable)
		{
			auto &var = id.get<SPIRVariable>();
			auto &type = get<SPIRType>(var.basetype);
			if (!is_hidden_variable(var) && var.storage != StorageClassFunction && type.pointer &&
			    type.storage == StorageClassPushConstant)
			{
				emit_push_constant_block(var);
			}
		}
	}

	// Output in/out interfaces.
	for (auto &id : ir.ids)
	{
		if (id.get_type() == TypeVariable)
		{
			auto &var = id.get<SPIRVariable>();
			auto &type = get<SPIRType>(var.basetype);

			if (var.storage != StorageClassFunction && !is_hidden_variable(var) && type.pointer &&
			    (var.storage == StorageClassInput || var.storage == StorageClassOutput) &&
			    interface_variable_exists_in_entry_point(var.self))
			{
				emit_interface_block(var);
			}
		}
	}

	// Output Uniform Constants (values, samplers, images, etc).
	for (auto &id : ir.ids)
	{
		if (id.get_type() == TypeVariable)
		{
			auto &var = id.get<SPIRVariable>();
			auto &type = get<SPIRType>(var.basetype);

			if (var.storage != StorageClassFunction && !is_hidden_variable(var) && type.pointer &&
			    (type.storage == StorageClassUniformConstant || type.storage == StorageClassAtomicCounter))
			{
				emit_uniform(var);
			}
		}
	}

	// Global variables.
	bool emitted = false;
	for (auto global : global_variables)
	{
		auto &var = get<SPIRVariable>(global);
		if (var.storage == StorageClassWorkgroup)
		{
			emit_shared(var);
			emitted = true;
		}
	}

	if (emitted)
		statement("");

	declare_undefined_values();

	statement("inline void init(spirv_cross_shader& s)");
	begin_scope();
	statement(resource_type, "::init(s);");
	for (auto &reg : resource_registrations)
		statement(reg);
	end_scope();
	resource_registrations.clear();

	end_scope_decl();

	statement("");
	statement("Resources* __res;");
	if (get_entry_point().model == ExecutionModelGLCompute)
		statement("ComputePrivateResources __priv_res;");
	statement("");

	// Emit regular globals which are allocated per invocation.
	emitted = false;
	for (auto global : global_variables)
	{
		auto &var = get<SPIRVariable>(global);
		if (var.storage == StorageClassPrivate)
		{
			if (var.storage == StorageClassWorkgroup)
				emit_shared(var);
			else
				statement(CompilerGLSL::variable_decl(var), ";");
			emitted = true;
		}
	}

	if (emitted)
		statement("");
}

string CompilerCPP::compile()
{
	// Do not deal with ES-isms like precision, older extensions and such.
	options.es = false;
	options.version = 450;
	backend.float_literal_suffix = true;
	backend.double_literal_suffix = false;
	backend.long_long_literal_suffix = true;
	backend.uint32_t_literal_suffix = true;
	backend.basic_int_type = "int32_t";
	backend.basic_uint_type = "uint32_t";
	backend.swizzle_is_function = true;
	backend.shared_is_implied = true;
	backend.unsized_array_supported = false;
	backend.explicit_struct_type = true;
	backend.use_initializer_list = true;

	fixup_type_alias();
	reorder_type_alias();
	build_function_control_flow_graphs_and_analyze();
	update_active_builtins();

	uint32_t pass_count = 0;
	do
	{
		if (pass_count >= 3)
			SPIRV_CROSS_THROW("Over 3 compilation loops detected. Must be a bug!");

		resource_registrations.clear();
		reset();

		// Move constructor for this type is broken on GCC 4.9 ...
		buffer.reset();

		emit_header();
		emit_resources();

		emit_function(get<SPIRFunction>(ir.default_entry_point), Bitset());

		pass_count++;
	} while (is_forcing_recompilation());

	// Match opening scope of emit_header().
	end_scope_decl();
	// namespace
	end_scope();

	// Emit C entry points
	emit_c_linkage();

	// Entry point in CPP is always main() for the time being.
	get_entry_point().name = "main";

	return buffer.str();
}

void CompilerCPP::emit_c_linkage()
{
	statement("");

	statement("spirv_cross_shader_t *spirv_cross_construct(void)");
	begin_scope();
	statement("return new ", impl_type, "();");
	end_scope();

	statement("");
	statement("void spirv_cross_destruct(spirv_cross_shader_t *shader)");
	begin_scope();
	statement("delete static_cast<", impl_type, "*>(shader);");
	end_scope();

	statement("");
	statement("void spirv_cross_invoke(spirv_cross_shader_t *shader)");
	begin_scope();
	statement("static_cast<", impl_type, "*>(shader)->invoke();");
	end_scope();

	statement("");
	statement("static const struct spirv_cross_interface vtable =");
	begin_scope();
	statement("spirv_cross_construct,");
	statement("spirv_cross_destruct,");
	statement("spirv_cross_invoke,");
	end_scope_decl();

	statement("");
	statement("const struct spirv_cross_interface *",
	          interface_name.empty() ? string("spirv_cross_get_interface") : interface_name, "(void)");
	begin_scope();
	statement("return &vtable;");
	end_scope();
}

void CompilerCPP::emit_function_prototype(SPIRFunction &func, const Bitset &)
{
	if (func.self != ir.default_entry_point)
		add_function_overload(func);

	local_variable_names = resource_names;
	string decl;

	auto &type = get<SPIRType>(func.return_type);
	decl += "inline ";
	decl += type_to_glsl(type);
	decl += " ";

	if (func.self == ir.default_entry_point)
	{
		decl += "main";
		processing_entry_point = true;
	}
	else
		decl += to_name(func.self);

	decl += "(";
	for (auto &arg : func.arguments)
	{
		add_local_variable_name(arg.id);

		decl += argument_decl(arg);
		if (&arg != &func.arguments.back())
			decl += ", ";

		// Hold a pointer to the parameter so we can invalidate the readonly field if needed.
		auto *var = maybe_get<SPIRVariable>(arg.id);
		if (var)
			var->parameter = &arg;
	}

	decl += ")";
	statement(decl);
}

string CompilerCPP::argument_decl(const SPIRFunction::Parameter &arg)
{
	auto &type = expression_type(arg.id);
	bool constref = !type.pointer || arg.write_count == 0;

	auto &var = get<SPIRVariable>(arg.id);

	string base = type_to_glsl(type);
	string variable_name = to_name(var.self);
	remap_variable_type_name(type, variable_name, base);

	for (uint32_t i = 0; i < type.array.size(); i++)
		base = join("std::array<", base, ", ", to_array_size(type, i), ">");

	return join(constref ? "const " : "", base, " &", variable_name);
}

string CompilerCPP::variable_decl(const SPIRType &type, const string &name, uint32_t /* id */)
{
	string base = type_to_glsl(type);
	remap_variable_type_name(type, name, base);
	bool runtime = false;

	for (uint32_t i = 0; i < type.array.size(); i++)
	{
		auto &array = type.array[i];
		if (!array && type.array_size_literal[i])
		{
			// Avoid using runtime arrays with std::array since this is undefined.
			// Runtime arrays cannot be passed around as values, so this is fine.
			runtime = true;
		}
		else
			base = join("std::array<", base, ", ", to_array_size(type, i), ">");
	}
	base += ' ';
	return base + name + (runtime ? "[1]" : "");
}

void CompilerCPP::emit_header()
{
	auto &execution = get_entry_point();

	statement("// This C++ shader is autogenerated by spirv-cross.");
	statement("#include \"spirv_cross/internal_interface.hpp\"");
	statement("#include \"spirv_cross/external_interface.h\"");
	// Needed to properly implement GLSL-style arrays.
	statement("#include <array>");
	statement("#include <stdint.h>");
	statement("");
	statement("using namespace spirv_cross;");
	statement("using namespace glm;");
	statement("");

	statement("namespace Impl");
	begin_scope();

	switch (execution.model)
	{
	case ExecutionModelGeometry:
	case ExecutionModelTessellationControl:
	case ExecutionModelTessellationEvaluation:
	case ExecutionModelGLCompute:
	case ExecutionModelFragment:
	case ExecutionModelVertex:
		statement("struct Shader");
		begin_scope();
		break;

	default:
		SPIRV_CROSS_THROW("Unsupported execution model.");
	}

	switch (execution.model)
	{
	case ExecutionModelGeometry:
		impl_type = "GeometryShader<Impl::Shader, Impl::Shader::Resources>";
		resource_type = "GeometryResources";
		break;

	case ExecutionModelVertex:
		impl_type = "VertexShader<Impl::Shader, Impl::Shader::Resources>";
		resource_type = "VertexResources";
		break;

	case ExecutionModelFragment:
		impl_type = "FragmentShader<Impl::Shader, Impl::Shader::Resources>";
		resource_type = "FragmentResources";
		break;

	case ExecutionModelGLCompute:
		impl_type = join("ComputeShader<Impl::Shader, Impl::Shader::Resources, ", execution.workgroup_size.x, ", ",
		                 execution.workgroup_size.y, ", ", execution.workgroup_size.z, ">");
		resource_type = "ComputeResources";
		break;

	case ExecutionModelTessellationControl:
		impl_type = "TessControlShader<Impl::Shader, Impl::Shader::Resources>";
		resource_type = "TessControlResources";
		break;

	case ExecutionModelTessellationEvaluation:
		impl_type = "TessEvaluationShader<Impl::Shader, Impl::Shader::Resources>";
		resource_type = "TessEvaluationResources";
		break;

	default:
		SPIRV_CROSS_THROW("Unsupported execution model.");
	}
}
