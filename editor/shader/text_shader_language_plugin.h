/**************************************************************************/
/*  text_shader_language_plugin.h                                         */
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

#include "editor/shader/editor_shader_language_plugin.h"

class TextShaderLanguagePlugin : public EditorShaderLanguagePlugin {
	GDCLASS(TextShaderLanguagePlugin, EditorShaderLanguagePlugin);

public:
	virtual bool handles_shader(const Ref<Shader> &p_shader) const override;
	virtual bool handles_shader_include(const Ref<ShaderInclude> &p_shader_inc) const override;
	virtual ShaderEditor *edit_shader(const Ref<Shader> &p_shader) override;
	virtual ShaderEditor *edit_shader_include(const Ref<ShaderInclude> &p_shader_inc) override;
	virtual Ref<Shader> create_new_shader(int p_variation_index, Shader::Mode p_shader_mode, int p_template_index) override;
	virtual Ref<ShaderInclude> create_new_shader_include() override;
	virtual PackedStringArray get_language_variations() const override;
	virtual String get_file_extension(int p_variation_index) const override;
};
