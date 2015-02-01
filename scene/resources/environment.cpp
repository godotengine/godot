/*************************************************************************/
/*  environment.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "environment.h"
#include "texture.h"
void Environment::set_background(BG p_bg) {

	ERR_FAIL_INDEX(p_bg,BG_MAX);
	bg_mode=p_bg;
	VS::get_singleton()->environment_set_background(environment,VS::EnvironmentBG(p_bg));
}

Environment::BG Environment::get_background() const{

	return bg_mode;
}

void Environment::set_background_param(BGParam p_param, const Variant& p_value){

	ERR_FAIL_INDEX(p_param,BG_PARAM_MAX);
	bg_param[p_param]=p_value;
	VS::get_singleton()->environment_set_background_param(environment,VS::EnvironmentBGParam(p_param),p_value);

}
Variant Environment::get_background_param(BGParam p_param) const{

	ERR_FAIL_INDEX_V(p_param,BG_PARAM_MAX,Variant());
	return bg_param[p_param];

}

void Environment::set_enable_fx(Fx p_effect,bool p_enabled){

	ERR_FAIL_INDEX(p_effect,FX_MAX);
	fx_enabled[p_effect]=p_enabled;
	VS::get_singleton()->environment_set_enable_fx(environment,VS::EnvironmentFx(p_effect),p_enabled);

}
bool Environment::is_fx_enabled(Fx p_effect) const{

	ERR_FAIL_INDEX_V(p_effect,FX_MAX,false);
	return fx_enabled[p_effect];

}

void Environment::fx_set_param(FxParam p_param,const Variant& p_value){

	ERR_FAIL_INDEX(p_param,FX_PARAM_MAX);
	fx_param[p_param]=p_value;
	VS::get_singleton()->environment_fx_set_param(environment,VS::EnvironmentFxParam(p_param),p_value);

}
Variant Environment::fx_get_param(FxParam p_param) const{

	ERR_FAIL_INDEX_V(p_param,FX_PARAM_MAX,Variant());
	return fx_param[p_param];

}

RID Environment::get_rid() const {

	return environment;
}

void Environment::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_background","bgmode"),&Environment::set_background);
	ObjectTypeDB::bind_method(_MD("get_background"),&Environment::get_background);

	ObjectTypeDB::bind_method(_MD("set_background_param","param","value"),&Environment::set_background_param);
	ObjectTypeDB::bind_method(_MD("get_background_param","param"),&Environment::get_background_param);

	ObjectTypeDB::bind_method(_MD("set_enable_fx","effect","enabled"),&Environment::set_enable_fx);
	ObjectTypeDB::bind_method(_MD("is_fx_enabled","effect"),&Environment::is_fx_enabled);

	ObjectTypeDB::bind_method(_MD("fx_set_param","param","value"),&Environment::fx_set_param);
	ObjectTypeDB::bind_method(_MD("fx_get_param","param"),&Environment::fx_get_param);

	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"ambient_light/enabled"),_SCS("set_enable_fx"),_SCS("is_fx_enabled"), FX_AMBIENT_LIGHT);
	ADD_PROPERTYI( PropertyInfo(Variant::COLOR,"ambient_light/color",PROPERTY_HINT_COLOR_NO_ALPHA),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_AMBIENT_LIGHT_COLOR);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"ambient_light/energy",PROPERTY_HINT_RANGE,"0,64,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_AMBIENT_LIGHT_ENERGY);


	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"fxaa/enabled"),_SCS("set_enable_fx"),_SCS("is_fx_enabled"), FX_FXAA);

	ADD_PROPERTY( PropertyInfo(Variant::INT,"background/mode",PROPERTY_HINT_ENUM,"Keep,Default Color,Color,Texture,Cubemap,Texture RGBE,Cubemap RGBE"),_SCS("set_background"),_SCS("get_background"));
	ADD_PROPERTYI( PropertyInfo(Variant::COLOR,"background/color"),_SCS("set_background_param"),_SCS("get_background_param"), BG_PARAM_COLOR);
	ADD_PROPERTYI( PropertyInfo(Variant::OBJECT,"background/texture",PROPERTY_HINT_RESOURCE_TYPE,"Texture"),_SCS("set_background_param"),_SCS("get_background_param"), BG_PARAM_TEXTURE);
	ADD_PROPERTYI( PropertyInfo(Variant::OBJECT,"background/cubemap",PROPERTY_HINT_RESOURCE_TYPE,"CubeMap"),_SCS("set_background_param"),_SCS("get_background_param"), BG_PARAM_CUBEMAP);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"background/energy",PROPERTY_HINT_RANGE,"0,128,0.01"),_SCS("set_background_param"),_SCS("get_background_param"), BG_PARAM_ENERGY);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"background/scale",PROPERTY_HINT_RANGE,"0.001,16,0.001"),_SCS("set_background_param"),_SCS("get_background_param"), BG_PARAM_SCALE);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"background/glow",PROPERTY_HINT_RANGE,"0.00,8,0.01"),_SCS("set_background_param"),_SCS("get_background_param"), BG_PARAM_GLOW);

	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"glow/enabled"),_SCS("set_enable_fx"),_SCS("is_fx_enabled"), FX_GLOW);
	ADD_PROPERTYI( PropertyInfo(Variant::INT,"glow/blur_passes",PROPERTY_HINT_RANGE,"1,4,1"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_GLOW_BLUR_PASSES);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"glow/blur_scale",PROPERTY_HINT_RANGE,"0.01,4,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_GLOW_BLUR_SCALE);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"glow/blur_strength",PROPERTY_HINT_RANGE,"0.01,4,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_GLOW_BLUR_STRENGTH);
	ADD_PROPERTYI( PropertyInfo(Variant::INT,"glow/blur_blend_mode",PROPERTY_HINT_ENUM,"Additive,Screen,SoftLight"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_GLOW_BLUR_BLEND_MODE);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"glow/bloom",PROPERTY_HINT_RANGE,"0,8,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_GLOW_BLOOM);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"glow/bloom_treshold",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_GLOW_BLOOM_TRESHOLD);
	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"dof_blur/enabled"),_SCS("set_enable_fx"),_SCS("is_fx_enabled"), FX_DOF_BLUR);
	ADD_PROPERTYI( PropertyInfo(Variant::INT,"dof_blur/blur_passes",PROPERTY_HINT_RANGE,"1,4,1"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_DOF_BLUR_PASSES);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"dof_blur/begin",PROPERTY_HINT_RANGE,"0,4096,0.1"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_DOF_BLUR_BEGIN);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"dof_blur/range",PROPERTY_HINT_RANGE,"0,4096,0.1"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_DOF_BLUR_RANGE);
	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"hdr/enabled"),_SCS("set_enable_fx"),_SCS("is_fx_enabled"), FX_HDR);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"hdr/tonemapper",PROPERTY_HINT_ENUM,"Linear,Log,Reinhardt,ReinhardtAutoWhite"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_HDR_TONEMAPPER);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"hdr/exposure",PROPERTY_HINT_RANGE,"0.01,16,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_HDR_EXPOSURE);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"hdr/white",PROPERTY_HINT_RANGE,"0.01,16,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_HDR_WHITE);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"hdr/glow_treshold",PROPERTY_HINT_RANGE,"0.00,8,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_HDR_GLOW_TRESHOLD);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"hdr/glow_scale",PROPERTY_HINT_RANGE,"0.00,16,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_HDR_GLOW_SCALE);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"hdr/min_luminance",PROPERTY_HINT_RANGE,"0.01,64,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_HDR_MIN_LUMINANCE);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"hdr/max_luminance",PROPERTY_HINT_RANGE,"0.01,64,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_HDR_MAX_LUMINANCE);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"hdr/exposure_adj_speed",PROPERTY_HINT_RANGE,"0.001,64,0.001"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_HDR_EXPOSURE_ADJUST_SPEED);
	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"fog/enabled"),_SCS("set_enable_fx"),_SCS("is_fx_enabled"), FX_FOG);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"fog/begin",PROPERTY_HINT_RANGE,"0.01,4096,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_FOG_BEGIN);
	ADD_PROPERTYI( PropertyInfo(Variant::COLOR,"fog/begin_color",PROPERTY_HINT_COLOR_NO_ALPHA),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_FOG_BEGIN_COLOR);
	ADD_PROPERTYI( PropertyInfo(Variant::COLOR,"fog/end_color",PROPERTY_HINT_COLOR_NO_ALPHA),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_FOG_END_COLOR);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"fog/attenuation",PROPERTY_HINT_EXP_EASING),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_FOG_ATTENUATION);
	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"fog/bg"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_FOG_BG);
	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"bcs/enabled"),_SCS("set_enable_fx"),_SCS("is_fx_enabled"), FX_BCS);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"bcs/brightness",PROPERTY_HINT_RANGE,"0.01,8,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_BCS_BRIGHTNESS);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"bcs/contrast",PROPERTY_HINT_RANGE,"0.01,8,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_BCS_CONTRAST);
	ADD_PROPERTYI( PropertyInfo(Variant::REAL,"bcs/saturation",PROPERTY_HINT_RANGE,"0.01,8,0.01"),_SCS("fx_set_param"),_SCS("fx_get_param"), FX_PARAM_BCS_SATURATION);
	ADD_PROPERTYI( PropertyInfo(Variant::BOOL,"srgb/enabled"),_SCS("set_enable_fx"),_SCS("is_fx_enabled"), FX_SRGB);





/*
		FX_PARAM_BLOOM_GLOW_BLUR_PASSES=VS::ENV_FX_PARAM_BLOOM_GLOW_BLUR_PASSES,
		FX_PARAM_BLOOM_AMOUNT=VS::ENV_FX_PARAM_BLOOM_AMOUNT,
		FX_PARAM_DOF_BLUR_PASSES=VS::ENV_FX_PARAM_DOF_BLUR_PASSES,
		FX_PARAM_DOF_BLUR_BEGIN=VS::ENV_FX_PARAM_DOF_BLUR_BEGIN,
		FX_PARAM_DOF_BLUR_END=VS::ENV_FX_PARAM_DOF_BLUR_END,
		FX_PARAM_HDR_EXPOSURE=VS::ENV_FX_PARAM_HDR_EXPOSURE,
		FX_PARAM_HDR_WHITE=VS::ENV_FX_PARAM_HDR_WHITE,
		FX_PARAM_HDR_GLOW_TRESHOLD=VS::ENV_FX_PARAM_HDR_GLOW_TRESHOLD,
		FX_PARAM_HDR_GLOW_SCALE=VS::ENV_FX_PARAM_HDR_GLOW_SCALE,
		FX_PARAM_HDR_MIN_LUMINANCE=VS::ENV_FX_PARAM_HDR_MIN_LUMINANCE,
		FX_PARAM_HDR_MAX_LUMINANCE=VS::ENV_FX_PARAM_HDR_MAX_LUMINANCE,
		FX_PARAM_HDR_EXPOSURE_ADJUST_SPEED=VS::ENV_FX_PARAM_HDR_EXPOSURE_ADJUST_SPEED,
		FX_PARAM_FOG_BEGIN=VS::ENV_FX_PARAM_FOG_BEGIN,
		FX_PARAM_FOG_ATTENUATION=VS::ENV_FX_PARAM_FOG_ATTENUATION,
		FX_PARAM_FOG_BEGIN_COLOR=VS::ENV_FX_PARAM_FOG_BEGIN_COLOR,
		FX_PARAM_FOG_END_COLOR=VS::ENV_FX_PARAM_FOG_END_COLOR,
		FX_PARAM_FOG_BG=VS::ENV_FX_PARAM_FOG_BG,
		FX_PARAM_GAMMA_VALUE=VS::ENV_FX_PARAM_GAMMA_VALUE,
		FX_PARAM_BRIGHTNESS_VALUE=VS::ENV_FX_PARAM_BRIGHTNESS_VALUE,
		FX_PARAM_CONTRAST_VALUE=VS::ENV_FX_PARAM_CONTRAST_VALUE,
		FX_PARAM_SATURATION_VALUE=VS::ENV_FX_PARAM_SATURATION_VALUE,
		FX_PARAM_MAX=VS::ENV_FX_PARAM_MAX
*/

	BIND_CONSTANT( BG_KEEP );
	BIND_CONSTANT( BG_DEFAULT_COLOR );
	BIND_CONSTANT( BG_COLOR );
	BIND_CONSTANT( BG_TEXTURE );
	BIND_CONSTANT( BG_CUBEMAP );
	BIND_CONSTANT( BG_TEXTURE_RGBE );
	BIND_CONSTANT( BG_CUBEMAP_RGBE );
	BIND_CONSTANT( BG_MAX );

	BIND_CONSTANT( BG_PARAM_COLOR );
	BIND_CONSTANT( BG_PARAM_TEXTURE );
	BIND_CONSTANT( BG_PARAM_CUBEMAP );
	BIND_CONSTANT( BG_PARAM_ENERGY );
	BIND_CONSTANT( BG_PARAM_GLOW );
	BIND_CONSTANT( BG_PARAM_MAX );


	BIND_CONSTANT( FX_AMBIENT_LIGHT );
	BIND_CONSTANT( FX_FXAA );
	BIND_CONSTANT( FX_GLOW );
	BIND_CONSTANT( FX_DOF_BLUR );
	BIND_CONSTANT( FX_HDR );
	BIND_CONSTANT( FX_FOG );
	BIND_CONSTANT( FX_BCS);
	BIND_CONSTANT( FX_SRGB );
	BIND_CONSTANT( FX_MAX );


	BIND_CONSTANT( FX_BLUR_BLEND_MODE_ADDITIVE );
	BIND_CONSTANT( FX_BLUR_BLEND_MODE_SCREEN );
	BIND_CONSTANT( FX_BLUR_BLEND_MODE_SOFTLIGHT );

	BIND_CONSTANT( FX_HDR_TONE_MAPPER_LINEAR );
	BIND_CONSTANT( FX_HDR_TONE_MAPPER_LOG );
	BIND_CONSTANT( FX_HDR_TONE_MAPPER_REINHARDT );
	BIND_CONSTANT( FX_HDR_TONE_MAPPER_REINHARDT_AUTOWHITE );

	BIND_CONSTANT( FX_PARAM_AMBIENT_LIGHT_COLOR );
	BIND_CONSTANT( FX_PARAM_AMBIENT_LIGHT_ENERGY );
	BIND_CONSTANT( FX_PARAM_GLOW_BLUR_PASSES );
	BIND_CONSTANT( FX_PARAM_GLOW_BLUR_SCALE );
	BIND_CONSTANT( FX_PARAM_GLOW_BLUR_STRENGTH );
	BIND_CONSTANT( FX_PARAM_GLOW_BLUR_BLEND_MODE );
	BIND_CONSTANT( FX_PARAM_GLOW_BLOOM);
	BIND_CONSTANT( FX_PARAM_GLOW_BLOOM_TRESHOLD);
	BIND_CONSTANT( FX_PARAM_DOF_BLUR_PASSES );
	BIND_CONSTANT( FX_PARAM_DOF_BLUR_BEGIN );
	BIND_CONSTANT( FX_PARAM_DOF_BLUR_RANGE );
	BIND_CONSTANT( FX_PARAM_HDR_TONEMAPPER);
	BIND_CONSTANT( FX_PARAM_HDR_EXPOSURE );
	BIND_CONSTANT( FX_PARAM_HDR_WHITE );
	BIND_CONSTANT( FX_PARAM_HDR_GLOW_TRESHOLD );
	BIND_CONSTANT( FX_PARAM_HDR_GLOW_SCALE );
	BIND_CONSTANT( FX_PARAM_HDR_MIN_LUMINANCE );
	BIND_CONSTANT( FX_PARAM_HDR_MAX_LUMINANCE );
	BIND_CONSTANT( FX_PARAM_HDR_EXPOSURE_ADJUST_SPEED );
	BIND_CONSTANT( FX_PARAM_FOG_BEGIN );
	BIND_CONSTANT( FX_PARAM_FOG_ATTENUATION );
	BIND_CONSTANT( FX_PARAM_FOG_BEGIN_COLOR );
	BIND_CONSTANT( FX_PARAM_FOG_END_COLOR );
	BIND_CONSTANT( FX_PARAM_FOG_BG );
	BIND_CONSTANT( FX_PARAM_BCS_BRIGHTNESS );
	BIND_CONSTANT( FX_PARAM_BCS_CONTRAST );
	BIND_CONSTANT( FX_PARAM_BCS_SATURATION );
	BIND_CONSTANT( FX_PARAM_MAX );

}

Environment::Environment() {

	environment = VS::get_singleton()->environment_create();

	set_background(BG_DEFAULT_COLOR);
	set_background_param(BG_PARAM_COLOR,Color(0,0,0));
	set_background_param(BG_PARAM_TEXTURE,Ref<ImageTexture>());
	set_background_param(BG_PARAM_CUBEMAP,Ref<CubeMap>());
	set_background_param(BG_PARAM_ENERGY,1.0);
	set_background_param(BG_PARAM_SCALE,1.0);
	set_background_param(BG_PARAM_GLOW,0.0);

	for(int i=0;i<FX_MAX;i++)
		set_enable_fx(Fx(i),false);

	fx_set_param(FX_PARAM_AMBIENT_LIGHT_COLOR,Color(0,0,0));
	fx_set_param(FX_PARAM_AMBIENT_LIGHT_ENERGY,1.0);
	fx_set_param(FX_PARAM_GLOW_BLUR_PASSES,1);
	fx_set_param(FX_PARAM_GLOW_BLUR_SCALE,1);
	fx_set_param(FX_PARAM_GLOW_BLUR_STRENGTH,1);
	fx_set_param(FX_PARAM_GLOW_BLOOM,0.0);
	fx_set_param(FX_PARAM_GLOW_BLOOM_TRESHOLD,0.5);
	fx_set_param(FX_PARAM_DOF_BLUR_PASSES,1);
	fx_set_param(FX_PARAM_DOF_BLUR_BEGIN,100.0);
	fx_set_param(FX_PARAM_DOF_BLUR_RANGE,10.0);
	fx_set_param(FX_PARAM_HDR_TONEMAPPER,FX_HDR_TONE_MAPPER_LINEAR);
	fx_set_param(FX_PARAM_HDR_EXPOSURE,0.4);
	fx_set_param(FX_PARAM_HDR_WHITE,1.0);
	fx_set_param(FX_PARAM_HDR_GLOW_TRESHOLD,0.95);
	fx_set_param(FX_PARAM_HDR_GLOW_SCALE,0.2);
	fx_set_param(FX_PARAM_HDR_MIN_LUMINANCE,0.4);
	fx_set_param(FX_PARAM_HDR_MAX_LUMINANCE,8.0);
	fx_set_param(FX_PARAM_HDR_EXPOSURE_ADJUST_SPEED,0.5);
	fx_set_param(FX_PARAM_FOG_BEGIN,100.0);
	fx_set_param(FX_PARAM_FOG_ATTENUATION,1.0);
	fx_set_param(FX_PARAM_FOG_BEGIN_COLOR,Color(0,0,0));
	fx_set_param(FX_PARAM_FOG_END_COLOR,Color(0,0,0));
	fx_set_param(FX_PARAM_FOG_BG,true);
	fx_set_param(FX_PARAM_BCS_BRIGHTNESS,1.0);
	fx_set_param(FX_PARAM_BCS_CONTRAST,1.0);
	fx_set_param(FX_PARAM_BCS_SATURATION,1.0);

}
Environment::~Environment() {

	VS::get_singleton()->free(environment);
}
