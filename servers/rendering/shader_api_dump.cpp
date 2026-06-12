/**************************************************************************/
/*  shader_api_dump.cpp                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "shader_api_dump.h"

#include "core/io/file_access.h"
#include "core/io/json.h"
#include "rendering_server.h"
#include "servers/rendering/shader_language.h"
#include "servers/rendering/shader_types.h"

#ifdef TOOLS_ENABLED

// Generates a JSON-compatible representation of Godot's shader API.
// A TypeScript definition of the expected data is provided below:
//
// {
//     datatypes: Array<{
//         name: string,
//         indexing_return_type: string,
//         size: number,
//         component_count: number,
//         indexing_size: number,
//         swizzle_field_count: number,
//         is_scalar: boolean,
//         is_float: boolean,
//         is_sampler: boolean,
//         operators: Array<{
//             name: string,
//             return_type: string,
//             right_type?: string, // only defined if a binary operator
//         }>,
//     }>,
//     modes: Array<{
//         name: string,
//         render_modes: Array<{
//             name: string,
//             options?: Array<string>
//         }>,
//         stencil_modes: Array<{
//             name: string,
//             options?: Array<string>
//         }>,
//         stages: Array<{
//             name: string,
//             can_discard: boolean,
//             main_function: boolean,
//             built_ins: Array<{
//                 name: string,
//                 type: string,
//                 constant: boolean,
//                 values?: Array<boolean | number>,
//             }>,
//             stage_functions: Array<{
//                 name: string,
//                 arguments: Array<{
//                     name: string,
//                     type: string,
//                 }>,
//                 return_type: string,
//                 skip_function: string,
//             }>,
//         }>,
//     }>,
//     functions: Array<{
//         name: string,
//         is_frag_only: boolean,
//         overloads: Array<{
//             return_type: string,
//             is_high_end: boolean,
//             arguments: Array<{
//                 name: string,
//                 type: string,
//                 is_out: boolean,
//                 is_const: boolean,
//                 min?: number, // only defined if is_const is true
//                 max?: number, // only defined if is_const is true
//             }>,
//         }>,
//     }>,
// }
Dictionary GDShaderAPIDump::generate_shader_api() {
	ShaderAPI api = {
		generate_datatypes(),
		generate_modes(),
		generate_functions()
	};
	return api_to_dictionary(api);
}

void GDShaderAPIDump::generate_shader_json_file(const String &p_path) {
	Dictionary api = generate_shader_api();
	Ref<JSON> json;
	json.instantiate();

	String text = json->stringify(api, "\t", false) + "\n";
	Ref<FileAccess> fa = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_MSG(fa.is_null(), vformat("Cannot open file '%s' for writing.", p_path));
	fa->store_string(text);
}

Vector<GDShaderAPIDump::ShaderAPIDatatype> GDShaderAPIDump::generate_datatypes() {
	using Operator = ShaderLanguage::Operator;
	using DataType = ShaderLanguage::DataType;

	Vector<ShaderAPIDatatype> datatypes;

	for (int type_index = DataType::TYPE_VOID; type_index < DataType::TYPE_MAX; type_index++) {
		DataType type = static_cast<DataType>(type_index);

		// No benefit to dumping void type.
		if (type == DataType::TYPE_VOID) {
			continue;
		}

		// Struct type is a generic representation of all structs that cannot be represented in JSON.
		if (type == DataType::TYPE_STRUCT) {
			continue;
		}

		// Iterate through all operators and all types to find all possible operator combinations.
		Vector<ShaderAPIDatatype::Operator> operators;
		int operator_index = Operator::OP_EQUAL;
		while (operator_index < Operator::OP_MAX) {
			Operator op = static_cast<Operator>(operator_index);

			int other_type_index = DataType::TYPE_VOID;
			while (other_type_index < DataType::TYPE_MAX) {
				DataType other_type = static_cast<DataType>(other_type_index);

				DataType return_type = DataType::TYPE_VOID;
				bool unary = false;
				bool valid = ShaderLanguage::get_datatype_operator_result(type, op, other_type, &return_type, &unary);
				if (valid) {
					ShaderAPIDatatype::Operator api_operator = {
						get_operator_name(op),
						return_type,
						unary ? DataType::TYPE_VOID : other_type,
					};
					operators.push_back(api_operator);
				}

				// If unary, don't iterate through every "other" type.
				if (unary) {
					break;
				}

				other_type_index++;
			}

			operator_index++;
		}

		ShaderAPIDatatype datatype = {
			ShaderLanguage::get_datatype_name(type),
			ShaderLanguage::get_datatype_indexed_type(type),
			operators,
			ShaderLanguage::get_datatype_size(type),
			ShaderLanguage::get_datatype_component_count(type),
			ShaderLanguage::get_datatype_indexed_size(type),
			ShaderLanguage::get_datatype_swizzle_field_count(type),
			ShaderLanguage::is_scalar_type(type),
			ShaderLanguage::is_float_type(type),
			ShaderLanguage::is_sampler_type(type),
		};
		datatypes.push_back(datatype);
	}

	return datatypes;
}

Vector<GDShaderAPIDump::ShaderAPIMode> GDShaderAPIDump::generate_modes() {
	using ShaderMode = RenderingServer::ShaderMode;

	Vector<ShaderAPIMode> modes;

	ShaderTypes *shader_types = ShaderTypes::get_singleton();
	const List<String> &types_list = shader_types->get_types_list();

	for (int shader_index = ShaderMode::SHADER_SPATIAL; shader_index < ShaderMode::SHADER_MAX; shader_index++) {
		ShaderMode shader_mode = static_cast<ShaderMode>(shader_index);

		Vector<ShaderAPIMode::Stage> stages;
		for (const KeyValue<StringName, ShaderLanguage::FunctionInfo> &name_to_function_info : shader_types->get_functions(shader_mode)) {
			const ShaderLanguage::FunctionInfo &function_info = name_to_function_info.value;

			Vector<ShaderAPIMode::Stage::BuiltIn> built_ins;
			for (const KeyValue<StringName, ShaderLanguage::BuiltInInfo> &name_to_built_in : function_info.built_ins) {
				ShaderAPIMode::Stage::BuiltIn built_in = {
					name_to_built_in.key,
					name_to_built_in.value.type,
					name_to_built_in.value.constant,
					name_to_built_in.value.values,
				};
				built_ins.push_back(built_in);
			}

			Vector<ShaderAPIMode::Stage::StageFunction> stage_functions;
			for (const KeyValue<StringName, ShaderLanguage::StageFunctionInfo> &name_to_stage_function : function_info.stage_functions) {
				Vector<ShaderAPIMode::Stage::StageFunction::Argument> arguments;
				for (const ShaderLanguage::StageFunctionInfo::Argument &argument : name_to_stage_function.value.arguments) {
					arguments.push_back({ argument.name,
							argument.type });
				}

				ShaderAPIMode::Stage::StageFunction stage_function = {
					name_to_stage_function.key,
					arguments,
					name_to_stage_function.value.return_type,
					name_to_stage_function.value.skip_function
				};
				stage_functions.push_back(stage_function);
			}

			ShaderAPIMode::Stage stage = {
				name_to_function_info.key,
				built_ins,
				stage_functions,
				function_info.can_discard,
				function_info.main_function
			};

			stages.push_back(stage);
		}

		ShaderAPIMode mode = {
			types_list.get(shader_index),
			stages,
			shader_types->get_modes(shader_mode),
			shader_types->get_stencil_modes(shader_mode),
		};

		modes.push_back(mode);
	}

	return modes;
}

Vector<GDShaderAPIDump::ShaderAPIFunction> GDShaderAPIDump::generate_functions() {
	HashMap<String, Vector<ShaderAPIFunction::Overload>> function_overloads;

	HashMap<String, Vector<int>> out_args = collect_out_arguments();
	HashMap<String, const ShaderLanguage::BuiltinFuncConstArgs *> const_int_args = collect_const_int_arguments();

	int idx = 0;
	while (ShaderLanguage::builtin_func_defs[idx].name) {
		const ShaderLanguage::BuiltinFuncDef &def = ShaderLanguage::builtin_func_defs[idx];

		String function_name = String(def.name);

		Vector<ShaderAPIFunction::Overload::Argument> arguments;
		int argument_idx = 0;
		const Vector<int> *out_arg_indexes = out_args.getptr(function_name);
		const ShaderLanguage::BuiltinFuncConstArgs **const_int_arg = const_int_args.getptr(function_name);
		while (def.args[argument_idx] != ShaderLanguage::DataType::TYPE_VOID) {
			bool is_out = out_arg_indexes != nullptr ? out_arg_indexes->has(argument_idx) : false;

			const ShaderLanguage::BuiltinFuncConstArgs *const_data = nullptr;
			if (const_int_arg != nullptr && (*const_int_arg)->arg == argument_idx) {
				const_data = (*const_int_arg);
			}

			ShaderAPIFunction::Overload::Argument argument = {
				String(def.args_names[argument_idx]),
				def.args[argument_idx],
				is_out,
				const_data
			};
			arguments.push_back(argument);

			argument_idx++;
		}

		// add overload to function
		ShaderAPIFunction::Overload overload = {
			arguments,
			def.rettype,
			def.high_end
		};

		if (function_overloads.has(function_name)) {
			function_overloads.get(function_name).push_back(overload);
		} else {
			function_overloads.insert(function_name, Vector({ overload }));
		}

		idx++;
	}

	Vector<ShaderAPIFunction> functions;

	HashSet<String> frag_only_functions = collect_frag_only_functions();
	for (KeyValue<String, Vector<GDShaderAPIDump::ShaderAPIFunction::Overload>> &f : function_overloads) {
		String name = f.key;
		ShaderAPIFunction function = {
			name,
			f.value,
			frag_only_functions.has(name)
		};
		functions.push_back(function);
	}

	return functions;
}

// The key is the function name.
// The value is a list of argument indexes that are "out".
HashMap<String, Vector<int>> GDShaderAPIDump::collect_out_arguments() {
	HashMap<String, Vector<int>> out_args;
	int idx = 0;
	while (ShaderLanguage::builtin_func_out_args[idx].name) {
		const ShaderLanguage::BuiltinFuncOutArgs &args = ShaderLanguage::builtin_func_out_args[idx];
		Vector<int> argument_indexes;
		for (int i = 0; i < ShaderLanguage::BuiltinFuncOutArgs::MAX_ARGS; i++) {
			if (args.arguments[i] >= 0) {
				argument_indexes.push_back(args.arguments[i]);
			}
		}
		out_args.insert(String(args.name), argument_indexes);
		idx++;
	}
	return out_args;
}

// The key is the function name.
// The value is a reference to the BuiltinFuncConstArgs where the argument index, min, and max can be obtained.
HashMap<String, const ShaderLanguage::BuiltinFuncConstArgs *> GDShaderAPIDump::collect_const_int_arguments() {
	HashMap<String, const ShaderLanguage::BuiltinFuncConstArgs *> const_int_args;
	int idx = 0;
	while (ShaderLanguage::builtin_func_const_args[idx].name) {
		const ShaderLanguage::BuiltinFuncConstArgs &arg = ShaderLanguage::builtin_func_const_args[idx];
		const_int_args.insert(String(arg.name), &arg);
		idx++;
	}
	return const_int_args;
}

HashSet<String> GDShaderAPIDump::collect_frag_only_functions() {
	HashSet<String> frag_only_functions;
	int idx = 0;
	while (ShaderLanguage::builtin_func_const_args[idx].name) {
		const ShaderLanguage::BuiltinEntry &entry = ShaderLanguage::frag_only_func_defs[idx];
		frag_only_functions.insert(String(entry.name));
		idx++;
	}
	return frag_only_functions;
}

String GDShaderAPIDump::get_operator_name(ShaderLanguage::Operator p_op) {
	switch (p_op) {
		case ShaderLanguage::Operator::OP_NEGATE:
			return "unary-";
		case ShaderLanguage::Operator::OP_POST_INCREMENT:
			return "post++";
		case ShaderLanguage::Operator::OP_POST_DECREMENT:
			return "post--";
		default:
			break;
	}
	return ShaderLanguage::get_operator_text(p_op);
}

Dictionary GDShaderAPIDump::api_to_dictionary(const ShaderAPI &p_api) {
	Dictionary api_dictionary;

	// datatypes
	{
		Array datatypes_array;
		for (const ShaderAPIDatatype &datatype : p_api.datatypes) {
			datatypes_array.push_back(api_datatype_to_dictionary(datatype));
		}
		api_dictionary["datatypes"] = datatypes_array;
	}

	// modes
	{
		Array modes_array;
		for (const ShaderAPIMode &mode : p_api.modes) {
			modes_array.push_back(api_mode_to_dictionary(mode));
		}
		api_dictionary["modes"] = modes_array;
	}

	// functions
	{
		Array functions_array;
		for (const ShaderAPIFunction &function : p_api.functions) {
			functions_array.push_back(api_function_to_dictionary(function));
		}
		api_dictionary["functions"] = functions_array;
	}

	return api_dictionary;
}

Dictionary GDShaderAPIDump::api_datatype_to_dictionary(const ShaderAPIDatatype &p_datatype) {
	Dictionary datatype;
	datatype["name"] = p_datatype.name;
	datatype["indexing_return_type"] = ShaderLanguage::get_datatype_name(p_datatype.indexing_return_type);
	datatype["size"] = p_datatype.size;
	datatype["component_count"] = p_datatype.component_count;
	datatype["indexing_size"] = p_datatype.indexing_size;
	datatype["swizzle_field_count"] = p_datatype.swizzle_field_count;
	datatype["is_scalar"] = p_datatype.is_scalar;
	datatype["is_float"] = p_datatype.is_float;
	datatype["is_sampler"] = p_datatype.is_sampler;

	Array operators;
	for (const ShaderAPIDatatype::Operator &op : p_datatype.operators) {
		Dictionary operator_data;
		operator_data["name"] = op.name;
		operator_data["return_type"] = ShaderLanguage::get_datatype_name(op.return_type);
		if (op.right_type != ShaderLanguage::DataType::TYPE_VOID) {
			operator_data["right_type"] = ShaderLanguage::get_datatype_name(op.right_type);
		}
		operators.push_back(operator_data);
	}
	datatype["operators"] = operators;

	return datatype;
}

Dictionary GDShaderAPIDump::api_mode_to_dictionary(const ShaderAPIMode &p_mode) {
	const auto mode_to_dictionary = [](const ShaderLanguage::ModeInfo &mode) -> Dictionary {
		Dictionary mode_data;
		mode_data["name"] = mode.name;
		if (!mode.options.is_empty()) {
			Array options;
			for (const StringName &option : mode.options) {
				options.push_back(option);
			}
			mode_data["options"] = options;
		}
		return mode_data;
	};

	Dictionary mode;
	mode["name"] = p_mode.name;

	Array stages;
	for (const ShaderAPIMode::Stage &function : p_mode.stages) {
		Dictionary function_data;
		function_data["name"] = function.name;
		function_data["can_discard"] = function.can_discard;
		function_data["main_function"] = function.main_function;

		// built_ins
		Array built_ins;
		for (const ShaderAPIMode::Stage::BuiltIn &built_in : function.built_ins) {
			Dictionary built_in_data;
			built_in_data["name"] = built_in.name;
			built_in_data["type"] = ShaderLanguage::get_datatype_name(built_in.type);
			built_in_data["constant"] = built_in.constant;
			if (!built_in.values.is_empty()) {
				Array values;
				for (const ShaderLanguage::Scalar &value : built_in.values) {
					switch (built_in.type) {
						case ShaderLanguage::TYPE_BOOL:
							values.push_back(value.boolean);
							break;
						case ShaderLanguage::TYPE_INT:
							values.push_back(value.sint);
							break;
						case ShaderLanguage::TYPE_UINT:
							values.push_back(value.uint);
							break;
						default:
							values.push_back(value.real);
							break;
					}
				}
				built_in_data["values"] = values;
			}
			built_ins.push_back(built_in_data);
		}
		function_data["built_ins"] = built_ins;

		// stage_functions
		Array stage_functions;
		for (const ShaderAPIMode::Stage::StageFunction &stage_function : function.stage_functions) {
			Dictionary stage_function_data;
			stage_function_data["name"] = stage_function.name;
			stage_function_data["return_type"] = ShaderLanguage::get_datatype_name(stage_function.return_type);
			stage_function_data["skip_function"] = stage_function.skip_function;

			Array arguments;
			for (const ShaderAPIMode::Stage::StageFunction::Argument &argument : stage_function.arguments) {
				Dictionary argument_data;
				argument_data["name"] = argument.name;
				argument_data["type"] = ShaderLanguage::get_datatype_name(argument.type);
				arguments.push_back(argument_data);
			}
			stage_function_data["arguments"] = arguments;

			stage_functions.push_back(stage_function_data);
		}
		function_data["stage_functions"] = stage_functions;

		stages.push_back(function_data);
	}
	mode["stages"] = stages;

	Array modes;
	for (const ShaderLanguage::ModeInfo &render_mode : p_mode.render_modes) {
		modes.push_back(mode_to_dictionary(render_mode));
	}
	mode["render_modes"] = modes;

	Array stencil_modes;
	for (const ShaderLanguage::ModeInfo &stencil_mode : p_mode.stencil_modes) {
		stencil_modes.push_back(mode_to_dictionary(stencil_mode));
	}
	mode["stencil_modes"] = stencil_modes;

	return mode;
}

Dictionary GDShaderAPIDump::api_function_to_dictionary(const ShaderAPIFunction &p_function) {
	Dictionary function;
	function["name"] = p_function.name;
	function["is_frag_only"] = p_function.is_frag_only;

	Array overloads;
	for (const ShaderAPIFunction::Overload &overload : p_function.overloads) {
		Dictionary overload_data;

		Array arguments;
		for (const ShaderAPIFunction::Overload::Argument &argument : overload.arguments) {
			Dictionary argument_data;
			argument_data["name"] = argument.name;
			argument_data["type"] = ShaderLanguage::get_datatype_name(argument.type);
			argument_data["is_out"] = argument.is_out;
			argument_data["is_const"] = argument.const_data != nullptr;
			if (argument.const_data != nullptr) {
				argument_data["min"] = argument.const_data->min;
				argument_data["max"] = argument.const_data->max;
			}
			arguments.push_back(argument_data);
		}
		overload_data["arguments"] = arguments;

		overload_data["return_type"] = ShaderLanguage::get_datatype_name(overload.return_type);
		overload_data["is_high_end"] = overload.is_high_end;

		overloads.push_back(overload_data);
	}
	function["overloads"] = overloads;

	return function;
}
#endif
