/*************************************************************************/
/*  material_editor_plugin.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "material_editor_plugin.h"

#include "scene/resources/particles_material.h"

String SpatialMaterialConversionPlugin::converts_to() const {

	return "ShaderMaterial";
}
bool SpatialMaterialConversionPlugin::handles(const Ref<Resource> &p_resource) const {

	Ref<SpatialMaterial> mat = p_resource;
	return mat.is_valid();
}
Ref<Resource> SpatialMaterialConversionPlugin::convert(const Ref<Resource> &p_resource) const {

	Ref<SpatialMaterial> mat = p_resource;
	ERR_FAIL_COND_V(!mat.is_valid(), Ref<Resource>());

	Ref<ShaderMaterial> smat;
	smat.instance();

	Ref<Shader> shader;
	shader.instance();

	String code = VS::get_singleton()->shader_get_code(mat->get_shader_rid());

	shader->set_code(code);

	smat->set_shader(shader);

	List<PropertyInfo> params;
	VS::get_singleton()->shader_get_param_list(mat->get_shader_rid(), &params);

	for (List<PropertyInfo>::Element *E = params.front(); E; E = E->next()) {

		// Texture parameter has to be treated specially since SpatialMaterial saved it
		// as RID but ShaderMaterial needs Texture itself
		Ref<Texture> texture = mat->get_texture_by_name(E->get().name);
		if (texture.is_valid()) {
			smat->set_shader_param(E->get().name, texture);
		} else {
			Variant value = VS::get_singleton()->material_get_param(mat->get_rid(), E->get().name);
			smat->set_shader_param(E->get().name, value);
		}
	}

	smat->set_render_priority(mat->get_render_priority());
	return smat;
}

String ParticlesMaterialConversionPlugin::converts_to() const {

	return "ShaderMaterial";
}
bool ParticlesMaterialConversionPlugin::handles(const Ref<Resource> &p_resource) const {

	Ref<ParticlesMaterial> mat = p_resource;
	return mat.is_valid();
}
Ref<Resource> ParticlesMaterialConversionPlugin::convert(const Ref<Resource> &p_resource) const {

	Ref<ParticlesMaterial> mat = p_resource;
	ERR_FAIL_COND_V(!mat.is_valid(), Ref<Resource>());

	Ref<ShaderMaterial> smat;
	smat.instance();

	Ref<Shader> shader;
	shader.instance();

	String code = VS::get_singleton()->shader_get_code(mat->get_shader_rid());

	shader->set_code(code);

	smat->set_shader(shader);

	List<PropertyInfo> params;
	VS::get_singleton()->shader_get_param_list(mat->get_shader_rid(), &params);

	for (List<PropertyInfo>::Element *E = params.front(); E; E = E->next()) {
		Variant value = VS::get_singleton()->material_get_param(mat->get_rid(), E->get().name);
		smat->set_shader_param(E->get().name, value);
	}

	smat->set_render_priority(mat->get_render_priority());
	return smat;
}

String CanvasItemMaterialConversionPlugin::converts_to() const {

	return "ShaderMaterial";
}
bool CanvasItemMaterialConversionPlugin::handles(const Ref<Resource> &p_resource) const {

	Ref<CanvasItemMaterial> mat = p_resource;
	return mat.is_valid();
}
Ref<Resource> CanvasItemMaterialConversionPlugin::convert(const Ref<Resource> &p_resource) const {

	Ref<CanvasItemMaterial> mat = p_resource;
	ERR_FAIL_COND_V(!mat.is_valid(), Ref<Resource>());

	Ref<ShaderMaterial> smat;
	smat.instance();

	Ref<Shader> shader;
	shader.instance();

	String code = VS::get_singleton()->shader_get_code(mat->get_shader_rid());

	shader->set_code(code);

	smat->set_shader(shader);

	List<PropertyInfo> params;
	VS::get_singleton()->shader_get_param_list(mat->get_shader_rid(), &params);

	for (List<PropertyInfo>::Element *E = params.front(); E; E = E->next()) {
		Variant value = VS::get_singleton()->material_get_param(mat->get_rid(), E->get().name);
		smat->set_shader_param(E->get().name, value);
	}

	smat->set_render_priority(mat->get_render_priority());
	return smat;
}
