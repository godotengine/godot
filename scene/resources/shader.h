/**************************************************************************/
/*  shader.h                                                              */
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

#ifndef SHADER_H
#define SHADER_H

#include "core/io/resource.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "scene/resources/texture.h"
#include "shader_include.h"

class Shader : public Resource {
	GDCLASS(Shader, Resource);
	OBJ_SAVE_TYPE(Shader);

public:
	enum Mode {
		MODE_SPATIAL,
		MODE_CANVAS_ITEM,
		MODE_PARTICLES,
		MODE_SKY,
		MODE_FOG,
		MODE_MAX
	};

private:
	RID shader;
	Mode mode = MODE_SPATIAL;
	HashSet<Ref<ShaderInclude>> include_dependencies;
	String code;

	// hack the name of performance
	// shaders keep a list of ShaderMaterial -> RenderingServer name translations, to make
	// conversion fast and save memory.
	mutable bool params_cache_dirty = true;
	mutable HashMap<StringName, StringName> params_cache; //map a shader param to a material param..
	HashMap<StringName, HashMap<int, Ref<Texture2D>>> default_textures;

	void _dependency_changed();
	virtual void _update_shader() const; //used for visual shader
protected:
	static void _bind_methods();

public:
	//void set_mode(Mode p_mode);
	virtual Mode get_mode() const;

	virtual void set_path(const String &p_path, bool p_take_over = false) override;

	void set_code(const String &p_code);
	String get_code() const;

	void get_shader_uniform_list(List<PropertyInfo> *p_params, bool p_get_groups = false) const;
	bool has_parameter(const StringName &p_name) const;

	void set_default_texture_parameter(const StringName &p_name, const Ref<Texture2D> &p_texture, int p_index = 0);
	Ref<Texture2D> get_default_texture_parameter(const StringName &p_name, int p_index = 0) const;
	void get_default_texture_parameter_list(List<StringName> *r_textures) const;

	virtual bool is_text_shader() const;

	// Finds the shader parameter name for the given property name, which should start with "shader_parameter/".
	_FORCE_INLINE_ StringName remap_parameter(const StringName &p_property) const {
		if (params_cache_dirty) {
			get_shader_uniform_list(nullptr);
		}

		String n = p_property;

		// Backwards compatibility with old shader parameter names.
		// Note: The if statements are important to make sure we are only replacing text exactly at index 0.
		if (n.find("param/") == 0) {
			n = n.replace_first("param/", "shader_parameter/");
		}
		if (n.find("shader_param/") == 0) {
			n = n.replace_first("shader_param/", "shader_parameter/");
		}
		if (n.find("shader_uniform/") == 0) {
			n = n.replace_first("shader_uniform/", "shader_parameter/");
		}

		{
			// Additional backwards compatibility for projects between #62972 and #64092 (about a month of v4.0 development).
			// These projects did not have any prefix for shader uniforms due to a bug.
			// This code should be removed during beta or rc of 4.0.
			const HashMap<StringName, StringName>::Iterator E = params_cache.find(n);
			if (E) {
				return E->value;
			}
		}

		if (n.begins_with("shader_parameter/")) {
			n = n.replace_first("shader_parameter/", "");
			const HashMap<StringName, StringName>::Iterator E = params_cache.find(n);
			if (E) {
				return E->value;
			}
		}

		return StringName();
	}

	virtual RID get_rid() const override;

	Shader();
	~Shader();
};

VARIANT_ENUM_CAST(Shader::Mode);

class ResourceFormatLoaderShader : public ResourceFormatLoader {
public:
	virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

class ResourceFormatSaverShader : public ResourceFormatSaver {
public:
	virtual Error save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags = 0);
	virtual void get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const;
	virtual bool recognize(const Ref<Resource> &p_resource) const;
};

#endif // SHADER_H
