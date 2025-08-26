/**************************************************************************/
/*  editor_shader_language_plugin.h                                       */
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

#include "shader_editor.h"

class EditorShaderLanguagePlugin : public RefCounted {
	GDCLASS(EditorShaderLanguagePlugin, RefCounted);

	static Vector<Ref<EditorShaderLanguagePlugin>> shader_languages;
	static Vector<Vector2i> language_variation_map;

public:
	static void register_shader_language(const Ref<EditorShaderLanguagePlugin> &p_shader_language);
	static void clear_registered_shader_languages();
	static const Vector<Ref<EditorShaderLanguagePlugin>> get_shader_languages_read_only();
	static int get_shader_language_variation_count();
	static Ref<EditorShaderLanguagePlugin> get_shader_language_for_index(int p_index);
	static String get_file_extension_for_index(int p_index);

	virtual bool handles_shader(const Ref<Shader> &p_shader) const = 0;
	virtual bool handles_shader_include(const Ref<ShaderInclude> &p_shader_inc) const { return false; }
	virtual ShaderEditor *edit_shader(const Ref<Shader> &p_shader) = 0;
	virtual ShaderEditor *edit_shader_include(const Ref<ShaderInclude> &p_shader_inc) { return nullptr; }
	virtual Ref<Shader> create_new_shader(int p_variation_index, Shader::Mode p_shader_mode, int p_template_index) = 0;
	virtual Ref<ShaderInclude> create_new_shader_include() { return Ref<ShaderInclude>(); }
	virtual PackedStringArray get_language_variations() const = 0;
	virtual String get_file_extension(int p_variation_index) const { return "tres"; }
};
