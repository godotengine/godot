/**************************************************************************/
/*  shader.compat.inc                                                     */
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

#ifndef DISABLE_DEPRECATED

void Shader::_set_default_texture_parameter_bind_compat_95126(const StringName &p_name, const Ref<Texture2D> &p_texture, int p_index) {
	set_default_texture_parameter(p_name, p_texture, p_index);
}

Ref<Texture2D> Shader::_get_default_texture_parameter_bind_compat_95126(const StringName &p_name, int p_index) const {
	return get_default_texture_parameter(p_name, p_index);
}

void Shader::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("set_default_texture_parameter", "name", "texture", "index"), &Shader::_set_default_texture_parameter_bind_compat_95126, DEFVAL(0));
	ClassDB::bind_compatibility_method(D_METHOD("get_default_texture_parameter", "name", "index"), &Shader::_get_default_texture_parameter_bind_compat_95126, DEFVAL(0));
}

#endif // DISABLE_DEPRECATED
