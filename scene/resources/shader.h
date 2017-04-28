/*************************************************************************/
/*  shader.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "io/resource_loader.h"
#include "resource.h"
#include "scene/resources/texture.h"

class Shader : public Resource {

	GDCLASS(Shader, Resource);
	OBJ_SAVE_TYPE(Shader);
	RES_BASE_EXTENSION("shd");

public:
	enum Mode {

		MODE_SPATIAL,
		MODE_CANVAS_ITEM,
		MODE_PARTICLES,
		MODE_MAX
	};

private:
	RID shader;
	Mode mode;

	// hack the name of performance
	// shaders keep a list of ShaderMaterial -> VisualServer name translations, to make
	// conversion fast and save memory.
	mutable bool params_cache_dirty;
	mutable Map<StringName, StringName> params_cache; //map a shader param to a material param..
	Map<StringName, Ref<Texture> > default_textures;

protected:
	static void _bind_methods();

public:
	//void set_mode(Mode p_mode);
	Mode get_mode() const;

	void set_code(const String &p_code);
	String get_code() const;

	void get_param_list(List<PropertyInfo> *p_params) const;
	bool has_param(const StringName &p_param) const;

	void set_default_texture_param(const StringName &p_param, const Ref<Texture> &p_texture);
	Ref<Texture> get_default_texture_param(const StringName &p_param) const;
	void get_default_texture_param_list(List<StringName> *r_textures) const;

	_FORCE_INLINE_ StringName remap_param(const StringName &p_param) const {
		if (params_cache_dirty)
			get_param_list(NULL);

		const Map<StringName, StringName>::Element *E = params_cache.find(p_param);
		if (E)
			return E->get();
		return StringName();
	}

	virtual RID get_rid() const;

	Shader();
	~Shader();
};

VARIANT_ENUM_CAST(Shader::Mode);

#endif // SHADER_H
