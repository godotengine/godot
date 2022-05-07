/*************************************************************************/
/*  shader_types.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SHADERTYPES_H
#define SHADERTYPES_H

#include "core/ordered_hash_map.h"
#include "servers/visual_server.h"
#include "shader_language.h"

class ShaderTypes {
	struct Type {
		Map<StringName, ShaderLanguage::FunctionInfo> functions;
		Vector<StringName> modes;
	};

	Map<VS::ShaderMode, Type> shader_modes;

	static ShaderTypes *singleton;

	Set<String> shader_types;

public:
	static ShaderTypes *get_singleton() { return singleton; }

	const Map<StringName, ShaderLanguage::FunctionInfo> &get_functions(VS::ShaderMode p_mode);
	const Vector<StringName> &get_modes(VS::ShaderMode p_mode);
	const Set<String> &get_types();

	ShaderTypes();
};

#endif // SHADERTYPES_H
