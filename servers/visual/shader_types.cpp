#include "shader_types.h"


const Map< StringName, Map<StringName,ShaderLanguage::DataType> >& ShaderTypes::get_functions(VS::ShaderMode p_mode) {

	return shader_modes[p_mode].functions;
}

const Set<String>& ShaderTypes::get_modes(VS::ShaderMode p_mode) {

	return shader_modes[p_mode].modes;
}


ShaderTypes *ShaderTypes::singleton=NULL;

ShaderTypes::ShaderTypes()
{
	singleton=this;


	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["SRC_VERTEX"]=ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["SRC_NORMAL"]=ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["SRC_TANGENT"]=ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["SRC_BONES"]=ShaderLanguage::TYPE_IVEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["SRC_WEIGHTS"]=ShaderLanguage::TYPE_VEC4;

	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["POSITION"]=ShaderLanguage::TYPE_VEC4 ;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["VERTEX"]=ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["NORMAL"]=ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["TANGENT"]=ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["BINORMAL"]=ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["UV"]=ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["UV2"]=ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["COLOR"]=ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["POINT_SIZE"]=ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["INSTANCE_ID"]=ShaderLanguage::TYPE_INT;

	//builtins
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["WORLD_MATRIX"]=ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["INV_CAMERA_MATRIX"]=ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["PROJECTION_MATRIX"]=ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["TIME"]=ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["vertex"]["VIEWPORT_SIZE"]=ShaderLanguage::TYPE_VEC2;

	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["VERTEX"]=ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["FRAGCOORD"]=ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["FRONT_FACING"]=ShaderLanguage::TYPE_BOOL;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["NORMAL"]=ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["TANGENT"]=ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["BINORMAL"]=ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["NORMALMAP"]=ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["NORMALMAP_DEPTH"]=ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["UV"]=ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["UV2"]=ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["COLOR"]=ShaderLanguage::TYPE_VEC4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["NORMAL"]=ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["ALBEDO"]=ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["ALPHA"]=ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["METAL"]=ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["ROUGH"]=ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["EMISSION"]=ShaderLanguage::TYPE_VEC3;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["SPECIAL"]=ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["DISCARD"]=ShaderLanguage::TYPE_BOOL;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["SCREEN_UV"]=ShaderLanguage::TYPE_VEC2;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["POINT_COORD"]=ShaderLanguage::TYPE_VEC2;

	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["WORLD_MATRIX"]=ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["INV_CAMERA_MATRIX"]=ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["PROJECTION_MATRIX"]=ShaderLanguage::TYPE_MAT4;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["TIME"]=ShaderLanguage::TYPE_FLOAT;
	shader_modes[VS::SHADER_SPATIAL].functions["fragment"]["VIEWPORT_SIZE"]=ShaderLanguage::TYPE_VEC2;

	shader_modes[VS::SHADER_SPATIAL].modes.insert("blend_mix");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("blend_add");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("blend_sub");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("blend_mul");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("special_glow");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("special_subsurf");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("special_specular");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("depth_draw_opaque");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("depth_draw_always");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("depth_draw_never");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("depth_draw_alpha_prepass");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("cull_front");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("cull_back");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("cull_disable");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("lightmap_on_uv2");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("unshaded");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("ontop");

	shader_modes[VS::SHADER_SPATIAL].modes.insert("vertex_model_space");
	shader_modes[VS::SHADER_SPATIAL].modes.insert("vertex_camera_space");

}
