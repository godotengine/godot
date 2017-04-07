/*************************************************************************/
/*  rasterizer.cpp                                                       */
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
#include "rasterizer.h"
#include "os/os.h"
#include "print_string.h"

Rasterizer *(*Rasterizer::_create_func)() = NULL;

Rasterizer *Rasterizer::create() {

	return _create_func();
}

RasterizerStorage *RasterizerStorage::base_signleton = NULL;

RasterizerStorage::RasterizerStorage() {

	base_signleton = this;
}

#if 0

RID Rasterizer::create_default_material() {

	return material_create();
}


/* Fixed MAterial SHADER API */

RID Rasterizer::_create_shader(const SpatialMaterialShaderKey& p_key) {

	ERR_FAIL_COND_V(!p_key.valid,RID());
	Map<SpatialMaterialShaderKey,SpatialMaterialShader>::Element *E=fixed_material_shaders.find(p_key);

	if (E) {
		E->get().refcount++;
		return E->get().shader;
	}

	uint64_t t = OS::get_singleton()->get_ticks_usec();

	SpatialMaterialShader fms;
	fms.refcount=1;
	fms.shader=shader_create();

	//create shader code


	int texcoords_used=0;
	String code;

	static const char* _uv_str[4]={"UV","uv_xform","UV2","uv_sphere"};
#define _TEXUVSTR(m_idx) String(_uv_str[(p_key.texcoord_mask >> (m_idx * 2)) & 0x3])


	if (p_key.use_pointsize) {

		code+="UV=POINT_COORD;\n";
	}


	for(int i=0;i<VS::FIXED_MATERIAL_PARAM_MAX;i++) {

		if (p_key.texture_mask&(1<<i))
			texcoords_used|=(1<<((p_key.texcoord_mask>>(i*2))&0x3));
	}

	if (texcoords_used&(1<<VS::FIXED_MATERIAL_TEXCOORD_UV_TRANSFORM)) {

		code+="uniform mat4 fmp_uv_xform;\n";
		code+="vec2 uv_xform = (fmp_uv_xform * vec4(UV,0,1)).xy;\n";
	}

	/* HANDLE NORMAL MAPPING */


	if (p_key.texture_mask&(1<<VS::FIXED_MATERIAL_PARAM_NORMAL)) {

		String scode;
		scode+="uniform float fmp_normal;\n";
		scode+="uniform texture fmp_normal_tex;\n";
		String uv_str;
		if (((p_key.texcoord_mask>>(VS::FIXED_MATERIAL_PARAM_NORMAL*2))&0x3)==VS::FIXED_MATERIAL_TEXCOORD_SPHERE) {
			uv_str="uv"; //sorry not supported
		} else {
			uv_str=_TEXUVSTR(VS::FIXED_MATERIAL_PARAM_NORMAL);
		}
		if (p_key.use_xy_normalmap) {
			scode+="vec2 ywnormal=tex( fmp_normal_tex,"+uv_str+").wy * vec2(2.0,2.0) - vec2(1.0,1.0);\n";
			scode+="NORMALMAP=vec3(ywnormal,sqrt(1 - (ywnormal.x * ywnormal.x) - (ywnormal.y * ywnormal.y) ));\n";
		} else {
			scode+="NORMALMAP=tex( fmp_normal_tex,"+uv_str+").xyz * vec3(2.0,2.0,1.0) - vec3(1.0,1.0,0.0);\n";
		}
		scode+="NORMALMAP_DEPTH=fmp_normal;\n";

		code+=scode;
	}

	//handle sphere uv if used, do it here because it needs the normal, which may be transformed by a normal map

	if (texcoords_used&(1<<VS::FIXED_MATERIAL_TEXCOORD_SPHERE)) {

		String tcode;
		tcode="vec3 eye_normal = normalize(VERTEX);\n";
		tcode+="vec3 ref = (eye_normal - 2.0*dot(NORMAL, eye_normal)*NORMAL);\n";
		tcode+="ref.z+=1.0;\n";
		tcode+="vec2 uv_sphere = ref.xy*vec2(0.5,0.0-0.5)+vec2(0.5,0.0-0.5);\n";
		code+=tcode;
	}

	/* HANDLE DIFFUSE LIGHTING */

	code+="uniform color fmp_diffuse;\n";
	code+="color diffuse=fmp_diffuse;\n";

	if (p_key.use_color_array)
		code+="diffuse*=COLOR;\n";

	if (p_key.texture_mask&(1<<VS::FIXED_MATERIAL_PARAM_DIFFUSE)) {


		code+="uniform texture fmp_diffuse_tex;\n";
		code+="diffuse*=tex( fmp_diffuse_tex,"+_TEXUVSTR(VS::FIXED_MATERIAL_PARAM_DIFFUSE)+");\n";
	}

	if (p_key.texture_mask&(1<<VS::FIXED_MATERIAL_PARAM_DETAIL)) {

		String dcode;
		dcode+="uniform texture fmp_detail_tex;\n";
		dcode+="uniform float fmp_detail;\n";
		dcode+="color detail=tex( fmp_detail_tex,"+_TEXUVSTR(VS::FIXED_MATERIAL_PARAM_DETAIL)+");\n";
		//aways mix
		dcode+="diffuse=vec4(mix(diffuse.rgb,detail.rgb,detail.a*fmp_detail),diffuse.a);\n";

		code+=dcode;
	}

	if (p_key.use_alpha) {
		code+="DIFFUSE_ALPHA=diffuse;\n";
		if (p_key.discard_alpha) {
			code+="DISCARD=diffuse.a<0.5;\n";
		}
	} else {
		code+="DIFFUSE=diffuse.rgb;\n";
	}

	/* HANDLE SPECULAR LIGHTING */

	code+="uniform color fmp_specular;\n";
	code+="color specular=fmp_specular;\n";

	if (p_key.texture_mask&(1<<VS::FIXED_MATERIAL_PARAM_SPECULAR)) {

		String scode;
		scode+="uniform texture fmp_specular_tex;\n";
		scode+="specular*=tex( fmp_specular_tex,"+_TEXUVSTR(VS::FIXED_MATERIAL_PARAM_SPECULAR)+");\n";
		code+=scode;
	}

	code+="SPECULAR=specular.rgb;\n";

	code+="uniform float fmp_specular_exp;\n";
	code+="float specular_exp=fmp_specular_exp;\n";

	if (p_key.texture_mask&(1<<VS::FIXED_MATERIAL_PARAM_SPECULAR_EXP)) {

		String scode;
		scode+="uniform texture fmp_specular_exp_tex;\n";
		scode+="specular_exp*=tex( fmp_specular_exp_tex,"+_TEXUVSTR(VS::FIXED_MATERIAL_PARAM_SPECULAR_EXP)+").r;\n";
		code+=scode;
	}

	code+="SPEC_EXP=specular_exp;\n";

	/* HANDLE EMISSION LIGHTING */

	code+="uniform color fmp_emission;\n";
	code+="color emission=fmp_emission;\n";

	if (p_key.texture_mask&(1<<VS::FIXED_MATERIAL_PARAM_EMISSION)) {

		String scode;
		scode+="uniform texture fmp_emission_tex;\n";
		scode+="emission*=tex( fmp_emission_tex,"+_TEXUVSTR(VS::FIXED_MATERIAL_PARAM_EMISSION)+");\n";
		code+=scode;
	}

	code+="EMISSION=emission.rgb;\n";


	/* HANDLE GLOW */

	code+="uniform float fmp_glow;\n";
	code+="float glow=fmp_glow;\n";

	if (p_key.texture_mask&(1<<VS::FIXED_MATERIAL_PARAM_GLOW)) {

		String scode;
		scode+="uniform texture fmp_glow_tex;\n";
		scode+="glow*=tex( fmp_glow_tex,"+_TEXUVSTR(VS::FIXED_MATERIAL_PARAM_GLOW)+").r;\n";
		code+=scode;
	}

	code+="GLOW=glow;\n";

	if (p_key.texture_mask&(1<<VS::FIXED_MATERIAL_PARAM_SHADE_PARAM)) {

		String scode;
		scode+="uniform texture fmp_shade_param_tex;\n";
		scode+="SHADE_PARAM=tex( fmp_shade_param_tex,"+_TEXUVSTR(VS::FIXED_MATERIAL_PARAM_SHADE_PARAM)+").r;\n";
		code+=scode;
	} else {

		String scode;
		scode+="uniform float fmp_shade_param;\n";
		scode+="SHADE_PARAM=fmp_shade_param;\n";
		code+=scode;

	}


	//print_line("**FRAGMENT SHADER GENERATED code: \n"+code);

	String vcode;
	vcode="uniform float "+_fixed_material_param_names[VS::FIXED_MATERIAL_PARAM_SPECULAR_EXP]+";\n";
	vcode+="SPEC_EXP="+_fixed_material_param_names[VS::FIXED_MATERIAL_PARAM_SPECULAR_EXP]+";\n";
	if (p_key.use_pointsize) {

		vcode+="uniform float "+_fixed_material_point_size_name+";\n";
		vcode+="POINT_SIZE="+_fixed_material_point_size_name+";\n";
		//vcode+="POINT_SIZE=10.0;\n";
	}

	String lcode;

	switch(p_key.light_shader) {

		case VS::FIXED_MATERIAL_LIGHT_SHADER_LAMBERT: {
			//do nothing

		} break;
		case VS::FIXED_MATERIAL_LIGHT_SHADER_WRAP: {

			lcode+="float NdotL = max(0.0,((dot( NORMAL, LIGHT_DIR )+SHADE_PARAM)/(1.0+SHADE_PARAM)));";
			lcode+="vec3 half_vec = normalize(LIGHT_DIR + EYE_VEC);";
			lcode+="float eye_light = max(dot(NORMAL, half_vec),0.0);";
			lcode+="LIGHT = LIGHT_DIFFUSE * DIFFUSE * NdotL;";
			lcode+="if (NdotL > 0.0) {";
			lcode+="\tLIGHT+=LIGHT_SPECULAR * SPECULAR * pow( eye_light, SPECULAR_EXP );";
			lcode+="};";

		} break;
		case VS::FIXED_MATERIAL_LIGHT_SHADER_VELVET: {
			lcode+="float NdotL = max(0.0,dot( NORMAL, LIGHT_DIR ));";
			lcode+="vec3 half_vec = normalize(LIGHT_DIR + EYE_VEC);";
			lcode+="float eye_light = max(dot(NORMAL, half_vec),0.0);";
			lcode+="LIGHT = LIGHT_DIFFUSE * DIFFUSE * NdotL;";
			lcode+="float rim = (1.0-abs(dot(NORMAL,vec3(0,0,1))))*SHADE_PARAM;";
			lcode+="LIGHT += LIGHT_DIFFUSE * DIFFUSE * rim;";
			lcode+="if (NdotL > 0.0) {";
			lcode+="\tLIGHT+=LIGHT_SPECULAR * SPECULAR * pow( eye_light, SPECULAR_EXP );";
			lcode+="};";


		} break;
		case VS::FIXED_MATERIAL_LIGHT_SHADER_TOON: {

			lcode+="float NdotL = dot( NORMAL, LIGHT_DIR );";
			lcode+="vec3 light_ref = reflect( LIGHT_DIR, NORMAL );";
			lcode+="float eye_light = clamp( dot( light_ref, vec3(0,0,0)-EYE_VEC), 0.0, 1.0 );";
			lcode+="float NdotL_diffuse = smoothstep( max( SHADE_PARAM-0.05, 0.0-1.0), min( SHADE_PARAM+0.05, 1.0), NdotL );";
			lcode+="float spec_radius=clamp((1.0-(SPECULAR_EXP/64.0)),0.0,1.0);";
			lcode+="float NdotL_specular = smoothstep( max( spec_radius-0.05, 0.0), min( spec_radius+0.05, 1.0), eye_light )*max(NdotL,0);";
			lcode+="LIGHT = NdotL_diffuse * LIGHT_DIFFUSE*DIFFUSE + NdotL_specular * LIGHT_SPECULAR*SPECULAR;";

		} break;

	}

	//print_line("**VERTEX SHADER GENERATED code: \n"+vcode);

	shader_set_code(fms.shader,vcode,code,lcode,0,0);

	fixed_material_shaders[p_key]=fms;
	return fms.shader;
}

void Rasterizer::_free_shader(const SpatialMaterialShaderKey& p_key) {

	if (p_key.valid==0)
		return; //not a valid key

	Map<SpatialMaterialShaderKey,SpatialMaterialShader>::Element *E=fixed_material_shaders.find(p_key);

	ERR_FAIL_COND(!E);
	E->get().refcount--;
	if (E->get().refcount==0) {
		free(E->get().shader);
		fixed_material_shaders.erase(E);
	}

}


void Rasterizer::fixed_material_set_flag(RID p_material, VS::SpatialMaterialFlags p_flag, bool p_enabled) {


	Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);
	ERR_FAIL_COND(!E);
	SpatialMaterial &fm=*E->get();

	switch(p_flag) {

		case VS::FIXED_MATERIAL_FLAG_USE_ALPHA: fm.use_alpha=p_enabled; break;
		case VS::FIXED_MATERIAL_FLAG_USE_COLOR_ARRAY: fm.use_color_array=p_enabled; break;
		case VS::FIXED_MATERIAL_FLAG_USE_POINT_SIZE: fm.use_pointsize=p_enabled; break;
		case VS::FIXED_MATERIAL_FLAG_DISCARD_ALPHA: fm.discard_alpha=p_enabled; break;
		case VS::FIXED_MATERIAL_FLAG_USE_XY_NORMALMAP: fm.use_xy_normalmap=p_enabled; break;
	}

	if (!fm.dirty_list.in_list())
		fixed_material_dirty_list.add( &fm.dirty_list );

}

bool Rasterizer::fixed_material_get_flag(RID p_material, VS::SpatialMaterialFlags p_flag) const{

	const Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);
	ERR_FAIL_COND_V(!E,false);
	const SpatialMaterial &fm=*E->get();
	switch(p_flag) {

		case VS::FIXED_MATERIAL_FLAG_USE_ALPHA: return fm.use_alpha;; break;
		case VS::FIXED_MATERIAL_FLAG_USE_COLOR_ARRAY: return fm.use_color_array;; break;
		case VS::FIXED_MATERIAL_FLAG_USE_POINT_SIZE: return fm.use_pointsize;; break;
		case VS::FIXED_MATERIAL_FLAG_DISCARD_ALPHA: return fm.discard_alpha;; break;
		case VS::FIXED_MATERIAL_FLAG_USE_XY_NORMALMAP: return fm.use_xy_normalmap;; break;

	}


	return false;
}


RID Rasterizer::fixed_material_create() {

	RID mat = material_create();
	fixed_materials[mat]=memnew( SpatialMaterial() );
	SpatialMaterial &fm=*fixed_materials[mat];
	fm.self=mat;
	fm.get_key();
	material_set_flag(mat,VS::MATERIAL_FLAG_COLOR_ARRAY_SRGB,true);
	for(int i=0;i<VS::FIXED_MATERIAL_PARAM_MAX;i++) {

		material_set_param(mat,_fixed_material_param_names[i],fm.param[i]); //must be there
	}
	fixed_material_dirty_list.add(&fm.dirty_list);
	//print_line("FMC: "+itos(mat.get_id()));
	return mat;
}




void Rasterizer::fixed_material_set_parameter(RID p_material, VS::SpatialMaterialParam p_parameter, const Variant& p_value){

	Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);
	ERR_FAIL_COND(!E);
	SpatialMaterial &fm=*E->get();
	RID material=E->key();
	ERR_FAIL_INDEX(p_parameter,VS::FIXED_MATERIAL_PARAM_MAX);

	if ((p_parameter==VS::FIXED_MATERIAL_PARAM_DIFFUSE || p_parameter==VS::FIXED_MATERIAL_PARAM_SPECULAR || p_parameter==VS::FIXED_MATERIAL_PARAM_EMISSION)) {

		if (p_value.get_type()!=Variant::COLOR) {
			ERR_EXPLAIN(String(_fixed_material_param_names[p_parameter])+" expects Color");
			ERR_FAIL();
		}
	} else {

		if (!p_value.is_num()) {
			ERR_EXPLAIN(String(_fixed_material_param_names[p_parameter])+" expects scalar");
			ERR_FAIL();
		}
	}

	fm.param[p_parameter]=p_value;
	VS::get_singleton()->material_set_param(material,_fixed_material_param_names[p_parameter],p_value);


}
Variant Rasterizer::fixed_material_get_parameter(RID p_material,VS::SpatialMaterialParam p_parameter) const{

	const Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);
	ERR_FAIL_COND_V(!E,Variant());
	const SpatialMaterial &fm=*E->get();
	ERR_FAIL_INDEX_V(p_parameter,VS::FIXED_MATERIAL_PARAM_MAX,Variant());
	return fm.param[p_parameter];
}

void Rasterizer::fixed_material_set_texture(RID p_material,VS::SpatialMaterialParam p_parameter, RID p_texture){

	Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);
	if (!E) {

		print_line("Not found: "+itos(p_material.get_id()));
	}
	ERR_FAIL_COND(!E);
	SpatialMaterial &fm=*E->get();


	ERR_FAIL_INDEX(p_parameter,VS::FIXED_MATERIAL_PARAM_MAX);
	RID material=E->key();
	fm.texture[p_parameter]=p_texture;
	VS::get_singleton()->material_set_param(material,_fixed_material_tex_names[p_parameter],p_texture);

	if (!fm.dirty_list.in_list())
		fixed_material_dirty_list.add( &fm.dirty_list );




}
RID Rasterizer::fixed_material_get_texture(RID p_material,VS::SpatialMaterialParam p_parameter) const{

	const Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);
	ERR_FAIL_COND_V(!E,RID());
	const SpatialMaterial &fm=*E->get();
	ERR_FAIL_INDEX_V(p_parameter,VS::FIXED_MATERIAL_PARAM_MAX,RID());

	return fm.texture[p_parameter];
}


void Rasterizer::fixed_material_set_texcoord_mode(RID p_material,VS::SpatialMaterialParam p_parameter, VS::SpatialMaterialTexCoordMode p_mode) {

	Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);
	ERR_FAIL_COND(!E);
	SpatialMaterial &fm=*E->get();
	ERR_FAIL_INDEX(p_parameter,VS::FIXED_MATERIAL_PARAM_MAX);

	fm.get_key();

	fm.texture_tc[p_parameter]=p_mode;

	if (!fm.dirty_list.in_list())
		fixed_material_dirty_list.add( &fm.dirty_list );

}

VS::SpatialMaterialTexCoordMode Rasterizer::fixed_material_get_texcoord_mode(RID p_material,VS::SpatialMaterialParam p_parameter) const {

	const Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);
	ERR_FAIL_COND_V(!E,VS::FIXED_MATERIAL_TEXCOORD_UV);
	const SpatialMaterial &fm=*E->get();
	ERR_FAIL_INDEX_V(p_parameter,VS::FIXED_MATERIAL_PARAM_MAX,VS::FIXED_MATERIAL_TEXCOORD_UV);

	return fm.texture_tc[p_parameter];
}

void Rasterizer::fixed_material_set_uv_transform(RID p_material,const Transform& p_transform) {

	Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);
	ERR_FAIL_COND(!E);
	SpatialMaterial &fm=*E->get();
	RID material=E->key();

	VS::get_singleton()->material_set_param(material,_fixed_material_uv_xform_name,p_transform);

	fm.uv_xform=p_transform;

}



Transform Rasterizer::fixed_material_get_uv_transform(RID p_material) const {

	const Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);
	ERR_FAIL_COND_V(!E,Transform());
	const SpatialMaterial &fm=*E->get();

	return fm.uv_xform;
}

void Rasterizer::fixed_material_set_light_shader(RID p_material,VS::SpatialMaterialLightShader p_shader) {

	Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);
	ERR_FAIL_COND(!E);
	SpatialMaterial &fm=*E->get();

	fm.light_shader=p_shader;

	if (!fm.dirty_list.in_list())
		fixed_material_dirty_list.add( &fm.dirty_list );

}

VS::SpatialMaterialLightShader Rasterizer::fixed_material_get_light_shader(RID p_material) const {

	const Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);
	ERR_FAIL_COND_V(!E,VS::FIXED_MATERIAL_LIGHT_SHADER_LAMBERT);
	const SpatialMaterial &fm=*E->get();

	return fm.light_shader;
}

void Rasterizer::fixed_material_set_point_size(RID p_material,float p_size) {

	Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);
	ERR_FAIL_COND(!E);
	SpatialMaterial &fm=*E->get();
	RID material=E->key();

	VS::get_singleton()->material_set_param(material,_fixed_material_point_size_name,p_size);

	fm.point_size=p_size;


}

float Rasterizer::fixed_material_get_point_size(RID p_material) const{

	const Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);
	ERR_FAIL_COND_V(!E,1.0);
	const SpatialMaterial &fm=*E->get();

	return fm.point_size;

}

void Rasterizer::_update_fixed_materials() {


	while(fixed_material_dirty_list.first()) {

		SpatialMaterial &fm=*fixed_material_dirty_list.first()->self();

		SpatialMaterialShaderKey new_key = fm.get_key();
		if (new_key.key!=fm.current_key.key) {

			_free_shader(fm.current_key);
			RID new_rid = _create_shader(new_key);
			fm.current_key=new_key;
			material_set_shader(fm.self,new_rid);

			if (fm.texture[VS::FIXED_MATERIAL_PARAM_DETAIL].is_valid()) {
				//send these again just in case.
				material_set_param(fm.self,_fixed_material_param_names[VS::FIXED_MATERIAL_PARAM_DETAIL],fm.param[VS::FIXED_MATERIAL_PARAM_DETAIL]);
			}
			if (fm.texture[VS::FIXED_MATERIAL_PARAM_NORMAL].is_valid()) {
				//send these again just in case.
				material_set_param(fm.self,_fixed_material_param_names[VS::FIXED_MATERIAL_PARAM_NORMAL],fm.param[VS::FIXED_MATERIAL_PARAM_NORMAL]);
			}

			material_set_param(fm.self,_fixed_material_uv_xform_name,fm.uv_xform);
			if (fm.use_pointsize) {
				material_set_param(fm.self,_fixed_material_point_size_name,fm.point_size);
			}
		}

		fixed_material_dirty_list.remove(fixed_material_dirty_list.first());
	}
}


void Rasterizer::_free_fixed_material(const RID& p_material) {

	Map<RID,SpatialMaterial*>::Element *E = fixed_materials.find(p_material);

	if (E) {

		_free_shader(E->get()->current_key); //free shader
		if (E->get()->dirty_list.in_list())
			fixed_material_dirty_list.remove( &E->get()->dirty_list);
		memdelete(E->get());
		fixed_materials.erase(E); //free material
	}


}


void Rasterizer::flush_frame() {

	//not really necessary to implement
}

Rasterizer::Rasterizer() {

	static const char* fm_names[VS::FIXED_MATERIAL_PARAM_MAX]={
	"diffuse",
	"detail",
	"specular",
	"emission",
	"specular_exp",
	"glow",
	"normal",
	"shade_param"};

	for(int i=0;i<VS::FIXED_MATERIAL_PARAM_MAX;i++) {

		_fixed_material_param_names[i]=String("fmp_")+fm_names[i];
		_fixed_material_tex_names[i]=String("fmp_")+fm_names[i]+"_tex";
	}

	_fixed_material_uv_xform_name="fmp_uv_xform";
	_fixed_material_point_size_name="fmp_point_size";

	draw_viewport_func=NULL;

	ERR_FAIL_COND( sizeof(SpatialMaterialShaderKey)!=4);

}

RID Rasterizer::create_overdraw_debug_material() {
	RID mat = fixed_material_create();
	fixed_material_set_parameter( mat,VisualServer::FIXED_MATERIAL_PARAM_SPECULAR,Color(0,0,0) );
	fixed_material_set_parameter( mat,VisualServer::FIXED_MATERIAL_PARAM_DIFFUSE,Color(0.1,0.1,0.2) );
	fixed_material_set_parameter( mat,VisualServer::FIXED_MATERIAL_PARAM_EMISSION,Color(0,0,0) );
	fixed_material_set_flag( mat, VS::FIXED_MATERIAL_FLAG_USE_ALPHA, true);
	material_set_flag( mat, VisualServer::MATERIAL_FLAG_UNSHADED, true );
	material_set_blend_mode( mat,VisualServer::MATERIAL_BLEND_MODE_ADD );


	return mat;
}

#endif
