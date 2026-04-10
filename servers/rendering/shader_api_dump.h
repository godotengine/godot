/**************************************************************************/
/*  shader_api_dump.h                                                     */
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

#pragma once

#include "servers/rendering/shader_language.h"

#ifdef TOOLS_ENABLED

class GDShaderAPIDump {
public:
	static Dictionary generate_shader_api();
	static void generate_shader_json_file(const String &p_path);

private:
	// datatypes

	struct ShaderAPIDatatype {
		struct Operator {
			String name;
			ShaderLanguage::DataType return_type;
			ShaderLanguage::DataType right_type;
		};

		String name;
		ShaderLanguage::DataType indexing_return_type;
		Vector<Operator> operators;
		uint32_t size;
		uint32_t component_count;
		uint32_t indexing_size;
		uint32_t swizzle_field_count;
		bool is_scalar;
		bool is_float;
		bool is_sampler;
	};

	// modes

	struct ShaderAPIMode {
		struct Stage {
			struct BuiltIn {
				StringName name;
				ShaderLanguage::DataType type;
				bool constant;
				Vector<ShaderLanguage::Scalar> values;
			};

			struct StageFunction {
				struct Argument {
					StringName name;
					ShaderLanguage::DataType type;
				};

				StringName name;
				Vector<Argument> arguments;
				ShaderLanguage::DataType return_type;
				String skip_function;
			};

			StringName name;
			Vector<BuiltIn> built_ins;
			Vector<StageFunction> stage_functions;
			bool can_discard;
			bool main_function;
		};

		String name;
		Vector<Stage> stages;
		Vector<ShaderLanguage::ModeInfo> render_modes;
		Vector<ShaderLanguage::ModeInfo> stencil_modes;
	};

	// functions

	struct ShaderAPIFunction {
		struct Overload {
			struct Argument {
				String name;
				ShaderLanguage::DataType type;
				bool is_out;

				// if not nullptr, this argument is const.
				// min and max can be referenced from the pointer.
				const ShaderLanguage::BuiltinFuncConstArgs *const_data;
			};

			Vector<Argument> arguments;
			ShaderLanguage::DataType return_type;
			bool is_high_end;
		};

		String name;
		Vector<Overload> overloads;
		bool is_frag_only;
	};

	// api

	struct ShaderAPI {
		Vector<ShaderAPIDatatype> datatypes;
		Vector<ShaderAPIMode> modes;
		Vector<ShaderAPIFunction> functions;
	};

	static Vector<ShaderAPIDatatype> generate_datatypes();
	static Vector<ShaderAPIMode> generate_modes();
	static Vector<ShaderAPIFunction> generate_functions();

	static HashMap<String, Vector<int>> collect_out_arguments();
	static HashMap<String, const ShaderLanguage::BuiltinFuncConstArgs *> collect_const_int_arguments();
	static HashSet<String> collect_frag_only_functions();

	static String get_operator_name(ShaderLanguage::Operator p_op);

	static Dictionary api_to_dictionary(const ShaderAPI &p_api);
	static Dictionary api_datatype_to_dictionary(const ShaderAPIDatatype &p_datatype);
	static Dictionary api_mode_to_dictionary(const ShaderAPIMode &p_mode);
	static Dictionary api_function_to_dictionary(const ShaderAPIFunction &p_function);
};
#endif
