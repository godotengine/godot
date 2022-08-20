/*************************************************************************/
/*  shader.h                                                             */
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
	bool has_uniform(const StringName &p_param) const;

	void set_default_texture_param(const StringName &p_uniform, const Ref<Texture2D> &p_texture, int p_index = 0);
	Ref<Texture2D> get_default_texture_param(const StringName &p_uniform, int p_index = 0) const;
	void get_default_texture_param_list(List<StringName> *r_textures) const;

	virtual bool is_text_shader() const;

	_FORCE_INLINE_ StringName remap_uniform(const StringName &p_uniform) const {
		if (params_cache_dirty) {
			get_shader_uniform_list(nullptr);
		}

		const HashMap<StringName, StringName>::Iterator E = params_cache.find(p_uniform);
		if (E) {
			return E->value;
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
