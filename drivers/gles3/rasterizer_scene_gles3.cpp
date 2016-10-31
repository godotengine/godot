#include "rasterizer_scene_gles3.h"
#include "globals.h"
#include "os/os.h"




static _FORCE_INLINE_ void store_matrix32(const Matrix32& p_mtx, float* p_array) {

	p_array[ 0]=p_mtx.elements[0][0];
	p_array[ 1]=p_mtx.elements[0][1];
	p_array[ 2]=0;
	p_array[ 3]=0;
	p_array[ 4]=p_mtx.elements[1][0];
	p_array[ 5]=p_mtx.elements[1][1];
	p_array[ 6]=0;
	p_array[ 7]=0;
	p_array[ 8]=0;
	p_array[ 9]=0;
	p_array[10]=1;
	p_array[11]=0;
	p_array[12]=p_mtx.elements[2][0];
	p_array[13]=p_mtx.elements[2][1];
	p_array[14]=0;
	p_array[15]=1;
}


static _FORCE_INLINE_ void store_transform(const Transform& p_mtx, float* p_array) {
	p_array[ 0]=p_mtx.basis.elements[0][0];
	p_array[ 1]=p_mtx.basis.elements[1][0];
	p_array[ 2]=p_mtx.basis.elements[2][0];
	p_array[ 3]=0;
	p_array[ 4]=p_mtx.basis.elements[0][1];
	p_array[ 5]=p_mtx.basis.elements[1][1];
	p_array[ 6]=p_mtx.basis.elements[2][1];
	p_array[ 7]=0;
	p_array[ 8]=p_mtx.basis.elements[0][2];
	p_array[ 9]=p_mtx.basis.elements[1][2];
	p_array[10]=p_mtx.basis.elements[2][2];
	p_array[11]=0;
	p_array[12]=p_mtx.origin.x;
	p_array[13]=p_mtx.origin.y;
	p_array[14]=p_mtx.origin.z;
	p_array[15]=1;
}

static _FORCE_INLINE_ void store_camera(const CameraMatrix& p_mtx, float* p_array) {

	for (int i=0;i<4;i++) {
		for (int j=0;j<4;j++) {

			p_array[i*4+j]=p_mtx.matrix[i][j];
		}
	}
}

/* SHADOW ATLAS API */

RID RasterizerSceneGLES3::shadow_atlas_create() {

	ShadowAtlas *shadow_atlas = memnew( ShadowAtlas );
	shadow_atlas->fbo=0;
	shadow_atlas->depth=0;
	shadow_atlas->size=0;
	shadow_atlas->smallest_subdiv=0;

	for(int i=0;i<4;i++) {
		shadow_atlas->size_order[i]=i;
	}


	return shadow_atlas_owner.make_rid(shadow_atlas);
}

void RasterizerSceneGLES3::shadow_atlas_set_size(RID p_atlas,int p_size){

	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_atlas);
	ERR_FAIL_COND(!shadow_atlas);
	ERR_FAIL_COND(p_size<0);
	if (p_size==shadow_atlas->size)
		return;

	if (shadow_atlas->fbo) {
		glDeleteTextures(1,&shadow_atlas->depth);
		glDeleteFramebuffers(1,&shadow_atlas->fbo);

		shadow_atlas->depth=0;
		shadow_atlas->fbo=0;
	}
	for(int i=0;i<4;i++) {
		//clear subdivisions
		shadow_atlas->quadrants[i].shadows.resize(0);
		shadow_atlas->quadrants[i].shadows.resize( 1<<shadow_atlas->quadrants[i].subdivision );
	}

	//erase shadow atlas reference from lights
	for (Map<RID,uint32_t>::Element *E=shadow_atlas->shadow_owners.front();E;E=E->next()) {
		LightInstance *li = light_instance_owner.getornull(E->key());
		ERR_CONTINUE(!li);
		li->shadow_atlases.erase(p_atlas);
	}

	//clear owners
	shadow_atlas->shadow_owners.clear();

	shadow_atlas->size=nearest_power_of_2(p_size);

	if (shadow_atlas->size)	{
		glGenFramebuffers(1, &shadow_atlas->fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, shadow_atlas->fbo);

		// Create a texture for storing the depth
		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &shadow_atlas->depth);
		glBindTexture(GL_TEXTURE_2D, shadow_atlas->depth);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_atlas->size, shadow_atlas->size, 0,
			     GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);

		//interpola nearest (though nvidia can improve this)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		// Remove artifact on the edges of the shadowmap
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		// We'll use a depth texture to store the depths in the shadow map
		// Attach the depth texture to FBO depth attachment point
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
				       GL_TEXTURE_2D, shadow_atlas->depth, 0);
	}
}


void RasterizerSceneGLES3::shadow_atlas_set_quadrant_subdivision(RID p_atlas,int p_quadrant,int p_subdivision){

	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_atlas);
	ERR_FAIL_COND(!shadow_atlas);
	ERR_FAIL_INDEX(p_quadrant,4);
	ERR_FAIL_INDEX(p_subdivision,16384);


	uint32_t subdiv = nearest_power_of_2(p_subdivision);
	if (subdiv&0xaaaaaaaa) { //sqrt(subdiv) must be integer
		subdiv<<=1;
	}

	subdiv=int(Math::sqrt(subdiv));

	//obtain the number that will be x*x

	if (shadow_atlas->quadrants[p_quadrant].subdivision==subdiv)
		return;

	//erase all data from quadrant
	for(int i=0;i<shadow_atlas->quadrants[p_quadrant].shadows.size();i++) {

		if (shadow_atlas->quadrants[p_quadrant].shadows[i].owner.is_valid()) {
			shadow_atlas->shadow_owners.erase(shadow_atlas->quadrants[p_quadrant].shadows[i].owner);
			LightInstance *li = light_instance_owner.getornull(shadow_atlas->quadrants[p_quadrant].shadows[i].owner);
			ERR_CONTINUE(!li);
			li->shadow_atlases.erase(p_atlas);
		}
	}

	shadow_atlas->quadrants[p_quadrant].shadows.resize(0);
	shadow_atlas->quadrants[p_quadrant].shadows.resize(subdiv*subdiv);
	shadow_atlas->quadrants[p_quadrant].subdivision=subdiv;

	//cache the smallest subdiv (for faster allocation in light update)

	shadow_atlas->smallest_subdiv=1<<30;

	for(int i=0;i<4;i++) {
		if (shadow_atlas->quadrants[i].subdivision) {
			shadow_atlas->smallest_subdiv=MIN(shadow_atlas->smallest_subdiv,shadow_atlas->quadrants[i].subdivision);
		}
	}

	if (shadow_atlas->smallest_subdiv==1<<30) {
		shadow_atlas->smallest_subdiv=0;
	}

	//resort the size orders, simple bublesort for 4 elements..

	int swaps=0;
	do {
		swaps=0;

		for(int i=0;i<3;i++) {
			if (shadow_atlas->quadrants[shadow_atlas->size_order[i]].subdivision > shadow_atlas->quadrants[shadow_atlas->size_order[i+1]].subdivision) {
				SWAP(shadow_atlas->size_order[i],shadow_atlas->size_order[i+1]);
				swaps++;
			}
		}
	} while(swaps>0);





}

bool RasterizerSceneGLES3::_shadow_atlas_find_shadow(ShadowAtlas *shadow_atlas,int *p_in_quadrants,int p_quadrant_count,int p_current_subdiv,uint64_t p_tick,int &r_quadrant,int &r_shadow) {


	for(int i=p_quadrant_count-1;i>=0;i--) {

		int qidx = p_in_quadrants[i];

		if (shadow_atlas->quadrants[qidx].subdivision==p_current_subdiv) {
			return false;
		}

		//look for an empty space
		int sc = shadow_atlas->quadrants[qidx].shadows.size();
		ShadowAtlas::Quadrant::Shadow *sarr = shadow_atlas->quadrants[qidx].shadows.ptr();

		int found_free_idx=-1; //found a free one
		int found_used_idx=-1; //found existing one, must steal it
		uint64_t min_pass; // pass of the existing one, try to use the least recently used one (LRU fashion)

		for(int j=0;j<sc;j++) {
			if (!sarr[j].owner.is_valid()) {
				found_free_idx=j;
				break;
			}

			LightInstance *sli = light_instance_owner.getornull(sarr[j].owner);
			ERR_CONTINUE(!sli);

			if (sli->last_scene_pass!=scene_pass) {

				//was just allocated, don't kill it so soon, wait a bit..
				if (p_tick-sarr[j].alloc_tick < shadow_atlas_realloc_tolerance_msec)
					continue;

				if (found_used_idx==-1 || sli->last_scene_pass<min_pass) {
					found_used_idx=j;
					min_pass=sli->last_scene_pass;
				}
			}
		}

		if (found_free_idx==-1 && found_used_idx==-1)
			continue; //nothing found

		if (found_free_idx==-1 && found_used_idx!=-1) {
			found_free_idx=found_used_idx;
		}

		r_quadrant=qidx;
		r_shadow=found_free_idx;

		return true;
	}

	return false;

}


uint32_t RasterizerSceneGLES3::shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version){

	ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_atlas);
	ERR_FAIL_COND_V(!shadow_atlas,ShadowAtlas::SHADOW_INVALID);

	LightInstance *li = light_instance_owner.getornull(p_light_intance);
	ERR_FAIL_COND_V(!li,ShadowAtlas::SHADOW_INVALID);

	if (shadow_atlas->size==0 || shadow_atlas->smallest_subdiv==0) {
		return ShadowAtlas::SHADOW_INVALID;
	}

	uint32_t quad_size = shadow_atlas->size>>1;
	int desired_fit = MAX(quad_size/shadow_atlas->smallest_subdiv,nearest_power_of_2(quad_size*p_coverage));

	int valid_quadrants[4];
	int valid_quadrant_count=0;
	int best_size=-1; //best size found
	int best_subdiv=-1; //subdiv for the best size

	//find the quadrants this fits into, and the best possible size it can fit into
	for(int i=0;i<4;i++) {
		int q = shadow_atlas->size_order[i];
		int sd = shadow_atlas->quadrants[q].subdivision;
		if (sd==0)
			continue; //unused

		int max_fit = quad_size / sd;

		if (best_size!=-1 && max_fit>best_size)
			break; //too large

		valid_quadrants[valid_quadrant_count++]=q;
		best_subdiv=sd;

		if (max_fit>=desired_fit) {
			best_size=max_fit;
		}
	}


	ERR_FAIL_COND_V(valid_quadrant_count==0,ShadowAtlas::SHADOW_INVALID);

	uint64_t tick = OS::get_singleton()->get_ticks_msec();


	//see if it already exists

	if (shadow_atlas->shadow_owners.has(p_light_intance)) {
		//it does!
		uint32_t key = shadow_atlas->shadow_owners[p_light_intance];
		uint32_t q = (key>>ShadowAtlas::QUADRANT_SHIFT)&0x3;
		uint32_t s = key&ShadowAtlas::SHADOW_INDEX_MASK;

		bool should_realloc=shadow_atlas->quadrants[q].subdivision!=best_subdiv && (shadow_atlas->quadrants[q].shadows[s].alloc_tick-tick > shadow_atlas_realloc_tolerance_msec);
		bool should_redraw=shadow_atlas->quadrants[q].shadows[s].version!=p_light_version;

		if (!should_realloc) {
			//already existing, see if it should redraw or it's just OK
			if (should_redraw) {
				key|=ShadowAtlas::SHADOW_INDEX_DIRTY_BIT;
			}

			return key;
		}

		int new_quadrant,new_shadow;

		//find a better place
		if (_shadow_atlas_find_shadow(shadow_atlas,valid_quadrants,valid_quadrant_count,shadow_atlas->quadrants[q].subdivision,tick,new_quadrant,new_shadow)) {
			//found a better place!
			ShadowAtlas::Quadrant::Shadow *sh = &shadow_atlas->quadrants[new_quadrant].shadows[new_shadow];
			if (sh->owner.is_valid()) {
				//is taken, but is invalid, erasing it
				shadow_atlas->shadow_owners.erase(sh->owner);
				LightInstance *sli = light_instance_owner.get(sh->owner);
				sli->shadow_atlases.erase(p_atlas);
			}

			sh->owner=p_light_intance;
			sh->alloc_tick=tick;
			sh->version=p_light_version;

			//make new key
			key=new_quadrant<<ShadowAtlas::QUADRANT_SHIFT;
			key|=new_shadow;
			//update it in map
			shadow_atlas->shadow_owners[p_light_intance]=key;
			//make it dirty, as it should redraw anyway
			key|=ShadowAtlas::SHADOW_INDEX_DIRTY_BIT;

			return key;
		}

		//no better place for this shadow found, keep current

		//already existing, see if it should redraw or it's just OK
		if (should_redraw) {
			key|=ShadowAtlas::SHADOW_INDEX_DIRTY_BIT;
		}

		return key;
	}

	int new_quadrant,new_shadow;

	//find a better place
	if (_shadow_atlas_find_shadow(shadow_atlas,valid_quadrants,valid_quadrant_count,-1,tick,new_quadrant,new_shadow)) {
		//found a better place!
		ShadowAtlas::Quadrant::Shadow *sh = &shadow_atlas->quadrants[new_quadrant].shadows[new_shadow];
		if (sh->owner.is_valid()) {
			//is taken, but is invalid, erasing it
			shadow_atlas->shadow_owners.erase(sh->owner);
			LightInstance *sli = light_instance_owner.get(sh->owner);
			sli->shadow_atlases.erase(p_atlas);
		}

		sh->owner=p_light_intance;
		sh->alloc_tick=tick;
		sh->version=p_light_version;

		//make new key
		uint32_t key=new_quadrant<<ShadowAtlas::QUADRANT_SHIFT;
		key|=new_shadow;
		//update it in map
		shadow_atlas->shadow_owners[p_light_intance]=key;
		//make it dirty, as it should redraw anyway
		key|=ShadowAtlas::SHADOW_INDEX_DIRTY_BIT;

		return key;
	}

	//no place to allocate this light, apologies

	return ShadowAtlas::SHADOW_INVALID;




}


/* ENVIRONMENT API */

RID RasterizerSceneGLES3::environment_create(){


	Environment *env = memnew( Environment );

	return environment_owner.make_rid(env);
}

void RasterizerSceneGLES3::environment_set_background(RID p_env,VS::EnvironmentBG p_bg){

	Environment *env=environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);
	env->bg_mode=p_bg;
}

void RasterizerSceneGLES3::environment_set_skybox(RID p_env, RID p_skybox, int p_radiance_size){

	Environment *env=environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	if (env->skybox_color.is_valid()) {
		env->skybox_color=RID();
	}
	if (env->skybox_radiance.is_valid()) {
		storage->free(env->skybox_radiance);
		env->skybox_radiance=RID();
	}


	if (p_skybox.is_valid()) {

		env->skybox_color=p_skybox;
		env->skybox_radiance=storage->texture_create_radiance_cubemap(p_skybox,p_radiance_size);
		//env->skybox_irradiance=storage->texture_create_pbr_cubemap(p_skybox,VS::PBR_CUBEMAP_IRRADIANCE,p_irradiance_size);
	}

}

void RasterizerSceneGLES3::environment_set_skybox_scale(RID p_env,float p_scale) {

	Environment *env=environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->skybox_scale=p_scale;

}

void RasterizerSceneGLES3::environment_set_bg_color(RID p_env,const Color& p_color){

	Environment *env=environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->bg_color=p_color;

}
void RasterizerSceneGLES3::environment_set_bg_energy(RID p_env,float p_energy) {

	Environment *env=environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->bg_energy=p_energy;

}

void RasterizerSceneGLES3::environment_set_canvas_max_layer(RID p_env,int p_max_layer){

	Environment *env=environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->canvas_max_layer=p_max_layer;

}
void RasterizerSceneGLES3::environment_set_ambient_light(RID p_env, const Color& p_color, float p_energy, float p_skybox_contribution){

	Environment *env=environment_owner.getornull(p_env);
	ERR_FAIL_COND(!env);

	env->ambient_color=p_color;
	env->ambient_energy=p_energy;
	env->ambient_skybox_contribution=p_skybox_contribution;

}

void RasterizerSceneGLES3::environment_set_glow(RID p_env,bool p_enable,int p_radius,float p_intensity,float p_strength,float p_bloom_treshold,VS::EnvironmentGlowBlendMode p_blend_mode){

}
void RasterizerSceneGLES3::environment_set_fog(RID p_env,bool p_enable,float p_begin,float p_end,RID p_gradient_texture){

}

void RasterizerSceneGLES3::environment_set_tonemap(RID p_env, bool p_enable, float p_exposure, float p_white, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale, VS::EnvironmentToneMapper p_tone_mapper){

}

void RasterizerSceneGLES3::environment_set_adjustment(RID p_env,bool p_enable,float p_brightness,float p_contrast,float p_saturation,RID p_ramp) {


}


RID RasterizerSceneGLES3::light_instance_create(RID p_light) {


	LightInstance *light_instance = memnew( LightInstance );

	light_instance->last_pass=0;
	light_instance->last_scene_pass=0;

	light_instance->light=p_light;
	light_instance->light_ptr=storage->light_owner.getornull(p_light);

	glGenBuffers(1, &light_instance->light_ubo);
	glBindBuffer(GL_UNIFORM_BUFFER, light_instance->light_ubo);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(LightInstance::LightDataUBO), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);


	ERR_FAIL_COND_V(!light_instance->light_ptr,RID());

	return light_instance_owner.make_rid(light_instance);
}

void RasterizerSceneGLES3::light_instance_set_transform(RID p_light_instance,const Transform& p_transform){

	LightInstance *light_instance = light_instance_owner.getornull(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	light_instance->transform=p_transform;
}

void RasterizerSceneGLES3::light_instance_mark_visible(RID p_light_instance) {

	LightInstance *light_instance = light_instance_owner.getornull(p_light_instance);
	ERR_FAIL_COND(!light_instance);

	light_instance->last_scene_pass=scene_pass;
}


////////////////////////////
////////////////////////////
////////////////////////////

bool RasterizerSceneGLES3::_setup_material(RasterizerStorageGLES3::Material* p_material,bool p_alpha_pass) {

	if (p_material->shader->spatial.cull_mode==RasterizerStorageGLES3::Shader::Spatial::CULL_MODE_DISABLED) {
		glDisable(GL_CULL_FACE);
	} else {
		glEnable(GL_CULL_FACE);
	}

	if (state.current_line_width!=p_material->line_width) {
		glLineWidth(p_material->line_width);
		state.current_line_width=p_material->line_width;
	}

	if (state.current_depth_draw!=p_material->shader->spatial.depth_draw_mode) {
		switch(p_material->shader->spatial.depth_draw_mode) {
			case RasterizerStorageGLES3::Shader::Spatial::DEPTH_DRAW_ALPHA_PREPASS:
			case RasterizerStorageGLES3::Shader::Spatial::DEPTH_DRAW_OPAQUE: {

				glDepthMask(!p_alpha_pass);
			} break;
			case RasterizerStorageGLES3::Shader::Spatial::DEPTH_DRAW_ALWAYS: {
				glDepthMask(GL_TRUE);
			} break;
			case RasterizerStorageGLES3::Shader::Spatial::DEPTH_DRAW_NEVER: {
				glDepthMask(GL_FALSE);
			} break;
		}

		state.current_depth_draw=p_material->shader->spatial.depth_draw_mode;
	}

	//glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);

	/*
	if (p_material->flags[VS::MATERIAL_FLAG_WIREFRAME])
		glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	*/

	//if (p_material->line_width)
	//	glLineWidth(p_material->line_width);

#if 0
	//blend mode
	if (state.current_blend_mode!=p_material->shader->spatial.blend_mode) {

		switch(p_material->shader->spatial.blend_mode) {

			 case RasterizerStorageGLES3::Shader::Spatial::BLEND_MODE_MIX: {
				glBlendEquation(GL_FUNC_ADD);
				if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
				} else {
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
				}

			 } break;
			 case RasterizerStorageGLES3::Shader::Spatial::BLEND_MODE_ADD: {

				glBlendEquation(GL_FUNC_ADD);
				glBlendFunc(p_alpha_pass?GL_SRC_ALPHA:GL_ONE,GL_ONE);

			 } break;
			 case RasterizerStorageGLES3::Shader::Spatial::BLEND_MODE_SUB: {

				glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
				glBlendFunc(GL_SRC_ALPHA,GL_ONE);
			 } break;
			case RasterizerStorageGLES3::Shader::Spatial::BLEND_MODE_MUL: {
				glBlendEquation(GL_FUNC_ADD);
				if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
					glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
				} else {
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
				}

			} break;
		}

		state.current_blend_mode=p_material->shader->spatial.blend_mode;

	}
#endif
	//material parameters


	state.scene_shader.set_custom_shader(p_material->shader->custom_code_id);
	bool rebind = state.scene_shader.bind();


	if (p_material->ubo_id) {

		glBindBufferBase(GL_UNIFORM_BUFFER,1,p_material->ubo_id);
	}



	int tc = p_material->textures.size();
	RID* textures = p_material->textures.ptr();
	ShaderLanguage::ShaderNode::Uniform::Hint* texture_hints = p_material->shader->texture_hints.ptr();

	for(int i=0;i<tc;i++) {

		glActiveTexture(GL_TEXTURE0+i);

		RasterizerStorageGLES3::Texture *t = storage->texture_owner.getornull( textures[i] );
		if (!t) {
			//check hints
			switch(texture_hints[i]) {
				case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK: {
					glBindTexture(GL_TEXTURE_2D,storage->resources.black_tex);
				} break;
				case ShaderLanguage::ShaderNode::Uniform::HINT_NORMAL: {
					glBindTexture(GL_TEXTURE_2D,storage->resources.normal_tex);
				} break;
				default: {
					glBindTexture(GL_TEXTURE_2D,storage->resources.white_tex);
				} break;
			}

			glBindTexture(GL_TEXTURE_2D,storage->resources.white_tex);
			continue;
		}

		if (storage->config.srgb_decode_supported) {
			//if SRGB decode extension is present, simply switch the texture to whathever is needed
			bool must_srgb=false;

			if (t->srgb && texture_hints[i]==ShaderLanguage::ShaderNode::Uniform::HINT_ALBEDO) {
				must_srgb=true;
			}

			if (t->using_srgb!=must_srgb) {
				if (must_srgb) {
					glTexParameteri(t->target,_TEXTURE_SRGB_DECODE_EXT,_DECODE_EXT);
#ifdef TOOLS_ENABLED
					if (!(t->flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)) {
						t->flags|=VS::TEXTURE_FLAG_CONVERT_TO_LINEAR;
						//notify that texture must be set to linear beforehand, so it works in other platforms when exported
					}
#endif

				} else {
					glTexParameteri(t->target,_TEXTURE_SRGB_DECODE_EXT,_SKIP_DECODE_EXT);
				}
				t->using_srgb=must_srgb;
			}
		}


		glBindTexture(t->target,t->tex_id);
	}


	return rebind;

}


void RasterizerSceneGLES3::_setup_geometry(RenderList::Element *e) {

	switch(e->instance->base_type) {

		case VS::INSTANCE_MESH: {

			RasterizerStorageGLES3::Surface *s = static_cast<RasterizerStorageGLES3::Surface*>(e->geometry);
			glBindVertexArray(s->array_id); // everything is so easy nowadays
		} break;
	}

}

static const GLenum gl_primitive[]={
	GL_POINTS,
	GL_LINES,
	GL_LINE_STRIP,
	GL_LINE_LOOP,
	GL_TRIANGLES,
	GL_TRIANGLE_STRIP,
	GL_TRIANGLE_FAN
};



void RasterizerSceneGLES3::_render_geometry(RenderList::Element *e) {

	switch(e->instance->base_type) {

		case VS::INSTANCE_MESH: {

			RasterizerStorageGLES3::Surface *s = static_cast<RasterizerStorageGLES3::Surface*>(e->geometry);

			if (s->index_array_len>0) {

				glDrawElements(gl_primitive[s->primitive],s->index_array_len, (s->array_len>=(1<<16))?GL_UNSIGNED_INT:GL_UNSIGNED_SHORT,0);

			} else {

				glDrawArrays(gl_primitive[s->primitive],0,s->array_len);

			}

		} break;
	}

}

void RasterizerSceneGLES3::_setup_light(LightInstance *p_light) {


	glBindBufferBase(GL_UNIFORM_BUFFER,3,p_light->light_ubo); //bind light uniform
}
void RasterizerSceneGLES3::_setup_transform(InstanceBase *p_instance,const Transform& p_view_transform,const CameraMatrix& p_projection) {

	if (p_instance->billboard || p_instance->billboard_y || p_instance->depth_scale) {

		Transform xf=p_instance->transform;
		if (p_instance->depth_scale) {

			if (p_projection.matrix[3][3]) {
				//orthogonal matrix, try to do about the same
				//with viewport size
				//real_t w = Math::abs( 1.0/(2.0*(p_projection.matrix[0][0])) );
				real_t h = Math::abs( 1.0/(2.0*p_projection.matrix[1][1]) );
				float sc = (h*2.0); //consistent with Y-fov
				xf.basis.scale( Vector3(sc,sc,sc));
			} else {
				//just scale by depth
				real_t sc = Plane(p_view_transform.origin,-p_view_transform.get_basis().get_axis(2)).distance_to(xf.origin);
				xf.basis.scale( Vector3(sc,sc,sc));
			}
		}

		if (p_instance->billboard) {

			Vector3 scale = xf.basis.get_scale();

			if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP]) {
				xf.set_look_at(xf.origin, xf.origin + p_view_transform.get_basis().get_axis(2), -p_view_transform.get_basis().get_axis(1));
			} else {
				xf.set_look_at(xf.origin, xf.origin + p_view_transform.get_basis().get_axis(2), p_view_transform.get_basis().get_axis(1));
			}

			xf.basis.scale(scale);
		}

		if (p_instance->billboard_y) {

			Vector3 scale = xf.basis.get_scale();
			Vector3 look_at =  p_view_transform.get_origin();
			look_at.y = 0.0;
			Vector3 look_at_norm = look_at.normalized();

			if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP]) {
				xf.set_look_at(xf.origin,xf.origin + look_at_norm, Vector3(0.0, -1.0, 0.0));
			} else {
				xf.set_look_at(xf.origin,xf.origin + look_at_norm, Vector3(0.0, 1.0, 0.0));
			}
			xf.basis.scale(scale);
		}
		state.scene_shader.set_uniform(SceneShaderGLES3::WORLD_TRANSFORM, xf);

	} else {
		state.scene_shader.set_uniform(SceneShaderGLES3::WORLD_TRANSFORM, p_instance->transform);
	}
}

void RasterizerSceneGLES3::_render_list(RenderList::Element **p_elements,int p_element_count,const Transform& p_view_transform,const CameraMatrix& p_projection,RasterizerStorageGLES3::Texture* p_base_env,bool p_reverse_cull,bool p_alpha_pass) {

	if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP]) {
		//p_reverse_cull=!p_reverse_cull;
		glFrontFace(GL_CCW);
	} else {
		glFrontFace(GL_CW);
	}

	bool shadow=false;

	glBindBufferBase(GL_UNIFORM_BUFFER,0,state.scene_ubo); //bind globals ubo


	glBindBufferBase(GL_UNIFORM_BUFFER,2,state.env_radiance_ubo); //bind environment radiance info
	glActiveTexture(GL_TEXTURE0+storage->config.max_texture_image_units-1);
	glBindTexture(GL_TEXTURE_2D,state.brdf_texture);

	if (p_base_env) {
		glActiveTexture(GL_TEXTURE0+storage->config.max_texture_image_units-2);
		glBindTexture(p_base_env->target,p_base_env->tex_id);
		state.scene_shader.set_conditional(SceneShaderGLES3::USE_RADIANCE_CUBEMAP,true);
	} else {
		state.scene_shader.set_conditional(SceneShaderGLES3::USE_RADIANCE_CUBEMAP,false);

	}



	state.scene_shader.set_conditional(SceneShaderGLES3::USE_SKELETON,false);

	state.current_blend_mode=-1;
	state.current_line_width=-1;
	state.current_depth_draw=-1;

	glDisable(GL_BLEND);

	RasterizerStorageGLES3::Material* prev_material=NULL;
	RasterizerStorageGLES3::Geometry* prev_geometry=NULL;
	VS::InstanceType prev_base_type = VS::INSTANCE_MAX;

	int prev_light_type=-1;
	int prev_light_index=-1;
	int prev_blend=-1;
	int current_blend_mode=-1;

	bool prev_additive=false;

	for (int i=0;i<p_element_count;i++) {

		RenderList::Element *e = p_elements[i];
		RasterizerStorageGLES3::Material* material= e->material;

		bool rebind=i==0;

		int light_type=(e->sort_key>>RenderList::SORT_KEY_LIGHT_TYPE_SHIFT)&0xF;
		int light_index=(e->sort_key>>RenderList::SORT_KEY_LIGHT_INDEX_SHIFT)&0xFFFF;

		bool additive=false;

		if (!shadow) {
#if 0
			if (texscreen_used && !texscreen_copied && material->shader_cache && material->shader_cache->valid && material->shader_cache->has_texscreen) {
				texscreen_copied=true;
				_copy_to_texscreen();

				//force reset state
				prev_material=NULL;
				prev_light=0x777E;
				prev_geometry_cmp=NULL;
				prev_light_type=0xEF;
				prev_skeleton =NULL;
				prev_sort_flags=0xFF;
				prev_morph_values=NULL;
				prev_receive_shadows_state=-1;
				glEnable(GL_BLEND);
				glDepthMask(GL_TRUE);
				glEnable(GL_DEPTH_TEST);
				glDisable(GL_SCISSOR_TEST);

			}
#endif
			if (light_type!=prev_light_type /* || receive_shadows_state!=prev_receive_shadows_state*/) {

				if (material->shader->spatial.unshaded/* || current_debug==VS::SCENARIO_DEBUG_SHADELESS*/) {
					state.scene_shader.set_conditional(SceneShaderGLES3::USE_FORWARD_LIGHTING,false);
					state.scene_shader.set_conditional(SceneShaderGLES3::USE_FORWARD_DIRECTIONAL,false);
					state.scene_shader.set_conditional(SceneShaderGLES3::USE_FORWARD_OMNI,false);
					state.scene_shader.set_conditional(SceneShaderGLES3::USE_FORWARD_SPOT,false);
					state.scene_shader.set_conditional(SceneShaderGLES3::SHADELESS,true);

					//state.scene_shader.set_conditional(SceneShaderGLES3::SHADELESS,true);
				} else {
					state.scene_shader.set_conditional(SceneShaderGLES3::SHADELESS,false);
					state.scene_shader.set_conditional(SceneShaderGLES3::USE_FORWARD_LIGHTING,light_type!=0xF);
					state.scene_shader.set_conditional(SceneShaderGLES3::USE_FORWARD_DIRECTIONAL,light_type==VS::LIGHT_DIRECTIONAL);
					state.scene_shader.set_conditional(SceneShaderGLES3::USE_FORWARD_OMNI,light_type==VS::LIGHT_OMNI);
					state.scene_shader.set_conditional(SceneShaderGLES3::USE_FORWARD_SPOT,light_type==VS::LIGHT_SPOT);
					/*
					if (receive_shadows_state==1) {
						state.scene_shader.set_conditional(SceneShaderGLES3::LIGHT_USE_SHADOW,(light_type&0x8));
						state.scene_shader.set_conditional(SceneShaderGLES3::LIGHT_USE_PSSM,(light_type&0x10));
						state.scene_shader.set_conditional(SceneShaderGLES3::LIGHT_USE_PSSM4,(light_type&0x20));
					}
					else {
						state.scene_shader.set_conditional(SceneShaderGLES3::LIGHT_USE_SHADOW,false);
						state.scene_shader.set_conditional(SceneShaderGLES3::LIGHT_USE_PSSM,false);
						state.scene_shader.set_conditional(SceneShaderGLES3::LIGHT_USE_PSSM4,false);
					}
					state.scene_shader.set_conditional(SceneShaderGLES3::SHADELESS,false);
					*/
				}

				rebind=true;
			}


			if (!*e->additive_ptr) {

				additive=false;
				*e->additive_ptr=true;
			} else {
				additive=true;
			}

			bool desired_blend=false;
			int desired_blend_mode=RasterizerStorageGLES3::Shader::Spatial::BLEND_MODE_MIX;

			if (additive) {
				desired_blend=true;
				desired_blend_mode=RasterizerStorageGLES3::Shader::Spatial::BLEND_MODE_ADD;
			} else {
				desired_blend=p_alpha_pass;
				desired_blend_mode=material->shader->spatial.blend_mode;
			}

			if (prev_blend!=desired_blend) {

				if (desired_blend) {
					glEnable(GL_BLEND);
					if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
						glColorMask(1,1,1,0);
					}
				} else {
					glDisable(GL_BLEND);
					glColorMask(1,1,1,1);
				}

				prev_blend=desired_blend;
			}

			if (desired_blend && desired_blend_mode!=current_blend_mode) {


				switch(desired_blend_mode) {

					 case RasterizerStorageGLES3::Shader::Spatial::BLEND_MODE_MIX: {
						glBlendEquation(GL_FUNC_ADD);
						if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
							glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
						}
						else {
							glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
						}

					 } break;
					 case RasterizerStorageGLES3::Shader::Spatial::BLEND_MODE_ADD: {

						glBlendEquation(GL_FUNC_ADD);
						glBlendFunc(p_alpha_pass?GL_SRC_ALPHA:GL_ONE,GL_ONE);

					 } break;
					 case RasterizerStorageGLES3::Shader::Spatial::BLEND_MODE_SUB: {

						glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
						glBlendFunc(GL_SRC_ALPHA,GL_ONE);
					 } break;
					case RasterizerStorageGLES3::Shader::Spatial::BLEND_MODE_MUL: {
						glBlendEquation(GL_FUNC_ADD);
						if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
							glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
						}
						else {
							glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
						}

					} break;

				}

				current_blend_mode=desired_blend_mode;
			}
		}

		if (light_index!=prev_light_index) {
			if (light_index!=0xFFFF) { //not unshaded
				_setup_light(light_instances[light_index]);
			}
		}

		if (material!=prev_material || rebind) {

			rebind = _setup_material(material,p_alpha_pass);
//			_rinfo.mat_change_count++;
		}


		if (prev_base_type != e->instance->base_type || prev_geometry!=e->geometry) {

			_setup_geometry(e);
		}

		if (rebind || prev_additive!=additive) {
			state.scene_shader.set_uniform(SceneShaderGLES3::NO_AMBIENT_LIGHT, additive);

		}

//		_set_cull(e->mirror,p_reverse_cull);

		state.scene_shader.set_uniform(SceneShaderGLES3::NORMAL_MULT, e->instance->mirror?-1.0:1.0);

		_setup_transform(e->instance,p_view_transform,p_projection);


//		_render(e->geometry, material, skeleton,e->owner,e->instance->transform);

		_render_geometry(e);

		prev_material=material;
		prev_base_type=e->instance->base_type;
		prev_geometry=e->geometry;
		prev_additive=additive;
		prev_light_type=light_type;
		prev_light_index=light_index;

	}

	//print_line("shaderchanges: "+itos(p_alpha_pass)+": "+itos(_rinfo.shader_change_count));


	glFrontFace(GL_CW);
	glBindVertexArray(0);

}

void RasterizerSceneGLES3::_add_geometry(  RasterizerStorageGLES3::Geometry* p_geometry,  InstanceBase *p_instance, RasterizerStorageGLES3::GeometryOwner *p_owner,int p_material) {

	RasterizerStorageGLES3::Material *m=NULL;
	RID m_src=p_instance->material_override.is_valid() ? p_instance->material_override :(p_material>=0?p_instance->materials[p_material]:p_geometry->material);


/*
#ifdef DEBUG_ENABLED
	if (current_debug==VS::SCENARIO_DEBUG_OVERDRAW) {
		m_src=overdraw_material;
	}

#endif
*/

	if (m_src.is_valid()) {
		m=storage->material_owner.getornull( m_src );

		if (!m->shader) {
			m=NULL;
		}
	}

	if (!m) {
		m=storage->material_owner.getptr( default_material );
	}

	ERR_FAIL_COND(!m);



	bool has_base_alpha=(m->shader->spatial.uses_alpha);
	bool has_blend_alpha=m->shader->spatial.blend_mode!=RasterizerStorageGLES3::Shader::Spatial::BLEND_MODE_MIX || m->shader->spatial.ontop;
	bool has_alpha = has_base_alpha || has_blend_alpha;
	bool shadow = false;

#if 0
	if (shadow) {

		if (has_blend_alpha || (has_base_alpha && m->depth_draw_mode!=VS::MATERIAL_DEPTH_DRAW_OPAQUE_PRE_PASS_ALPHA))
			return; //bye

		if (!m->shader_cache || (!m->shader_cache->writes_vertex && !m->shader_cache->uses_discard && m->depth_draw_mode!=VS::MATERIAL_DEPTH_DRAW_OPAQUE_PRE_PASS_ALPHA)) {
			//shader does not use discard and does not write a vertex position, use generic material
			if (p_instance->cast_shadows == VS::SHADOW_CASTING_SETTING_DOUBLE_SIDED)
				m = shadow_mat_double_sided_ptr;
			else
				m = shadow_mat_ptr;
			if (m->last_pass!=frame) {

				if (m->shader.is_valid()) {

					m->shader_cache=shader_owner.get(m->shader);
					if (m->shader_cache) {


						if (!m->shader_cache->valid)
							m->shader_cache=NULL;
					} else {
						m->shader=RID();
					}

				} else {
					m->shader_cache=NULL;
				}

				m->last_pass=frame;
			}
		}

		render_list = &opaque_render_list;
	/* notyet
		if (!m->shader_cache || m->shader_cache->can_zpass)
			render_list = &alpha_render_list;
		} else {
			render_list = &opaque_render_list;
		}*/

	} else {
		if (has_alpha) {
			render_list = &alpha_render_list;
		} else {
			render_list = &opaque_render_list;

		}
	}
#endif

	RenderList::Element *e = has_alpha ? render_list.add_alpha_element() : render_list.add_element();

	if (!e)
		return;

	e->geometry=p_geometry;
	e->material=m;
	e->instance=p_instance;
	e->owner=p_owner;
	e->additive=false;
	e->additive_ptr=&e->additive;
	e->sort_key=0;


	if (e->geometry->last_pass!=render_pass) {
		e->geometry->last_pass=render_pass;
		e->geometry->index=current_geometry_index++;
	}

	e->sort_key|=uint64_t(e->geometry->index)<<RenderList::SORT_KEY_GEOMETRY_INDEX_SHIFT;
	e->sort_key|=uint64_t(e->instance->base_type)<<RenderList::SORT_KEY_GEOMETRY_TYPE_SHIFT;

	if (e->material->last_pass!=render_pass) {
		e->material->last_pass=render_pass;
		e->material->index=current_material_index++;
	}

	e->sort_key|=uint64_t(e->material->index)<<RenderList::SORT_KEY_MATERIAL_INDEX_SHIFT;
	e->sort_key|=uint64_t(e->instance->depth_layer)<<RenderList::SORT_KEY_DEPTH_LAYER_SHIFT;

	//if (e->geometry->type==RasterizerStorageGLES3::Geometry::GEOMETRY_MULTISURFACE)
	//	e->sort_flags|=RenderList::SORT_FLAG_INSTANCING;

	bool mirror = e->instance->mirror;

	if (m->shader->spatial.cull_mode==RasterizerStorageGLES3::Shader::Spatial::CULL_MODE_FRONT) {
		mirror=!mirror;
	}

	if (mirror) {
		e->sort_key|=RenderList::SORT_KEY_MIRROR_FLAG;
	}

	//e->light_type=0xFF; // no lights!

	if (!shadow && !has_blend_alpha && has_alpha && m->shader->spatial.depth_draw_mode==RasterizerStorageGLES3::Shader::Spatial::DEPTH_DRAW_ALPHA_PREPASS) {

		//if nothing exists, add this element as opaque too
		RenderList::Element *oe = render_list.add_element();

		if (!oe)
			return;

		copymem(oe,e,sizeof(RenderList::Element));
		oe->additive_ptr=&oe->additive;
	}




	if (shadow || m->shader->spatial.unshaded /*|| current_debug==VS::SCENARIO_DEBUG_SHADELESS*/) {

		e->sort_key|=RenderList::SORT_KEY_LIGHT_INDEX_UNSHADED;
		e->sort_key|=uint64_t(0xF)<<RenderList::SORT_KEY_LIGHT_TYPE_SHIFT; //light type 0xF is no light?
		e->sort_key|=uint64_t(0xFFFF)<<RenderList::SORT_KEY_LIGHT_INDEX_SHIFT;
	} else {

		bool duplicate=false;
		bool lit=false;


		for(int i=0;i<directional_light_instance_count;i++) {

/*
			if (directional_lights[i]->base->shadow_enabled) {
				light_type|=0x8;
				if (directional_lights[i]->base->directional_shadow_mode==VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS)
					light_type|=0x10;
				else if (directional_lights[i]->base->directional_shadow_mode==VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS)
					light_type|=0x30;

			}
*/

			RenderList::Element *ec;
			if (duplicate) {
				ec = render_list.add_element();
				copymem(ec,e,sizeof(RenderList::Element));
			} else {

				ec=e;
				duplicate=true;
			}

			ec->additive_ptr=&e->additive;

			ec->sort_key&=~RenderList::SORT_KEY_LIGHT_MASK;
			ec->sort_key|=uint64_t(directional_light_instances[i]->light_index) << RenderList::SORT_KEY_LIGHT_INDEX_SHIFT;
			ec->sort_key|=uint64_t(VS::LIGHT_DIRECTIONAL) << RenderList::SORT_KEY_LIGHT_TYPE_SHIFT;

			lit=true;
		}


		const RID *liptr = p_instance->light_instances.ptr();
		int ilc=p_instance->light_instances.size();



		for(int i=0;i<ilc;i++) {

			LightInstance *li=light_instance_owner.getptr( liptr[i] );

			if (!li || li->last_pass!=render_pass) //lit by light not in visible scene
				continue;


//			if (li->base->shadow_enabled) {
//				light_type|=0x8;
//			}

			RenderList::Element *ec;
			if (duplicate) {

				ec = render_list.add_element();
				copymem(ec,e,sizeof(RenderList::Element));
			} else {

				duplicate=true;
				ec=e;
			}

			ec->additive_ptr=&e->additive;

			ec->sort_key&=~RenderList::SORT_KEY_LIGHT_MASK;
			ec->sort_key|=uint64_t(li->light_index) << RenderList::SORT_KEY_LIGHT_INDEX_SHIFT;
			ec->sort_key|=uint64_t(li->light_ptr->type) << RenderList::SORT_KEY_LIGHT_TYPE_SHIFT;

			lit=true;
		}


		if (!lit) {
			e->sort_key&=~RenderList::SORT_KEY_LIGHT_MASK;
			e->sort_key|=uint64_t(0xE)<<RenderList::SORT_KEY_LIGHT_TYPE_SHIFT; //light type 0xE is no light found
			e->sort_key|=uint64_t(0xFFFF)<<RenderList::SORT_KEY_LIGHT_INDEX_SHIFT;
		}


	}


}

void RasterizerSceneGLES3::_draw_skybox(RID p_skybox,CameraMatrix& p_projection,const Transform& p_transform,bool p_vflip,float p_scale) {

	RasterizerStorageGLES3::Texture *tex = storage->texture_owner.getornull(p_skybox);

	ERR_FAIL_COND(!tex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(tex->target,tex->tex_id);


	if (storage->config.srgb_decode_supported && tex->srgb && !tex->using_srgb) {

		glTexParameteri(tex->target,_TEXTURE_SRGB_DECODE_EXT,_DECODE_EXT);
		tex->using_srgb=true;
#ifdef TOOLS_ENABLED
		if (!(tex->flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)) {
			tex->flags|=VS::TEXTURE_FLAG_CONVERT_TO_LINEAR;
			//notify that texture must be set to linear beforehand, so it works in other platforms when exported
		}
#endif
	}

	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glDepthFunc(GL_LEQUAL);
	glColorMask(1,1,1,1);

	float flip_sign = p_vflip?-1:1;

	Vector3 vertices[8]={
		Vector3(-1,-1*flip_sign,1),
		Vector3( 0, 1, 0),
		Vector3( 1,-1*flip_sign,1),
		Vector3( 1, 1, 0),
		Vector3( 1, 1*flip_sign,1),
		Vector3( 1, 0, 0),
		Vector3(-1, 1*flip_sign,1),
		Vector3( 0, 0, 0)

	};



	//skybox uv vectors
	float vw,vh,zn;
	p_projection.get_viewport_size(vw,vh);
	zn=p_projection.get_z_near();

	float scale=p_scale;

	for(int i=0;i<4;i++) {

		Vector3 uv=vertices[i*2+1];
		uv.x=(uv.x*2.0-1.0)*vw*scale;
		uv.y=-(uv.y*2.0-1.0)*vh*scale;
		uv.z=-zn;
		vertices[i*2+1] = p_transform.basis.xform(uv).normalized();
		vertices[i*2+1].z = -vertices[i*2+1].z;
	}

	glBindBuffer(GL_ARRAY_BUFFER,state.skybox_verts);
	glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(Vector3)*8,vertices);
	glBindBuffer(GL_ARRAY_BUFFER,0); //unbind

	glBindVertexArray(state.skybox_array);

	storage->shaders.copy.set_conditional(CopyShaderGLES3::USE_CUBEMAP,true);
	storage->shaders.copy.bind();

	glDrawArrays(GL_TRIANGLE_FAN,0,4);

	glBindVertexArray(0);
	glColorMask(1,1,1,1);

	storage->shaders.copy.set_conditional(CopyShaderGLES3::USE_CUBEMAP,false);

}


void RasterizerSceneGLES3::_setup_environment(Environment *env,CameraMatrix& p_cam_projection,const Transform& p_cam_transform) {


	//store camera into ubo
	store_camera(p_cam_projection,state.ubo_data.projection_matrix);
	store_transform(p_cam_transform,state.ubo_data.camera_matrix);
	store_transform(p_cam_transform.affine_inverse(),state.ubo_data.camera_inverse_matrix);

	//time global variables
	for(int i=0;i<4;i++) {
		state.ubo_data.time[i]=storage->frame.time[i];
	}

	//bg and ambient
	if (env) {
		state.ubo_data.bg_energy=env->bg_energy;
		state.ubo_data.ambient_energy=env->ambient_energy;
		Color linear_ambient_color = env->ambient_color.to_linear();
		state.ubo_data.ambient_light_color[0]=linear_ambient_color.r;
		state.ubo_data.ambient_light_color[1]=linear_ambient_color.g;
		state.ubo_data.ambient_light_color[2]=linear_ambient_color.b;
		state.ubo_data.ambient_light_color[3]=linear_ambient_color.a;

		Color bg_color;

		switch(env->bg_mode) {
			case VS::ENV_BG_CLEAR_COLOR: {
				bg_color=storage->frame.clear_request_color.to_linear();
			} break;
			case VS::ENV_BG_COLOR: {
				bg_color=env->bg_color.to_linear();
			} break;
			default: {
				bg_color=Color(0,0,0,1);
			} break;
		}

		state.ubo_data.bg_color[0]=bg_color.r;
		state.ubo_data.bg_color[1]=bg_color.g;
		state.ubo_data.bg_color[2]=bg_color.b;
		state.ubo_data.bg_color[3]=bg_color.a;

		state.env_radiance_data.ambient_contribution=env->ambient_skybox_contribution;
	} else {
		state.ubo_data.bg_energy=1.0;
		state.ubo_data.ambient_energy=1.0;
		//use from clear color instead, since there is no ambient
		Color linear_ambient_color = storage->frame.clear_request_color.to_linear();
		state.ubo_data.ambient_light_color[0]=linear_ambient_color.r;
		state.ubo_data.ambient_light_color[1]=linear_ambient_color.g;
		state.ubo_data.ambient_light_color[2]=linear_ambient_color.b;
		state.ubo_data.ambient_light_color[3]=linear_ambient_color.a;

		state.ubo_data.bg_color[0]=linear_ambient_color.r;
		state.ubo_data.bg_color[1]=linear_ambient_color.g;
		state.ubo_data.bg_color[2]=linear_ambient_color.b;
		state.ubo_data.bg_color[3]=linear_ambient_color.a;

		state.env_radiance_data.ambient_contribution=0;

	}

	glBindBuffer(GL_UNIFORM_BUFFER, state.scene_ubo);
	glBufferSubData(GL_UNIFORM_BUFFER, 0,sizeof(State::SceneDataUBO), &state.ubo_data);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	//fill up environment

	store_transform(p_cam_transform,state.env_radiance_data.transform);


	glBindBuffer(GL_UNIFORM_BUFFER, state.env_radiance_ubo);
	glBufferSubData(GL_UNIFORM_BUFFER, 0,sizeof(State::EnvironmentRadianceUBO), &state.env_radiance_data);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

}

void RasterizerSceneGLES3::_setup_lights(RID *p_light_cull_result,int p_light_cull_count,const Transform& p_camera_inverse_transform,const CameraMatrix& p_camera_projection) {

	directional_light_instance_count=0;
	light_instance_count=0;

	Vector<float> lpercent;

	for(int i=0;i<p_light_cull_count;i++)	 {

		ERR_BREAK( i>=RenderList::MAX_LIGHTS );

		LightInstance *li = light_instance_owner.getptr(p_light_cull_result[i]);

		switch(li->light_ptr->type) {

			case VS::LIGHT_DIRECTIONAL: {

				ERR_FAIL_COND( directional_light_instance_count >= RenderList::MAX_LIGHTS);
				directional_light_instances[directional_light_instance_count++]=li;

				Color linear_col = li->light_ptr->color.to_linear();
				li->light_ubo_data.light_color_energy[0]=linear_col.r;
				li->light_ubo_data.light_color_energy[1]=linear_col.g;
				li->light_ubo_data.light_color_energy[2]=linear_col.b;
				li->light_ubo_data.light_color_energy[3]=li->light_ptr->param[VS::LIGHT_PARAM_ENERGY];

				//omni, keep at 0
				li->light_ubo_data.light_pos_inv_radius[0]=0.0;
				li->light_ubo_data.light_pos_inv_radius[1]=0.0;
				li->light_ubo_data.light_pos_inv_radius[2]=0.0;
				li->light_ubo_data.light_pos_inv_radius[3]=0.0;

				Vector3 direction = p_camera_inverse_transform.basis.xform(li->transform.basis.xform(Vector3(0,0,-1))).normalized();
				li->light_ubo_data.light_direction_attenuation[0]=direction.x;
				li->light_ubo_data.light_direction_attenuation[1]=direction.y;
				li->light_ubo_data.light_direction_attenuation[2]=direction.z;
				li->light_ubo_data.light_direction_attenuation[3]=1.0;

				li->light_ubo_data.light_params[0]=0;
				li->light_ubo_data.light_params[1]=li->light_ptr->param[VS::LIGHT_PARAM_SPECULAR];
				li->light_ubo_data.light_params[2]=0;
				li->light_ubo_data.light_params[3]=0;




#if 0
				if (li->light_ptr->shadow_enabled) {
					CameraMatrix bias;
					bias.set_light_bias();

					int passes=light_instance_get_shadow_passes(p_light_instance);

					for(int i=0;i<passes;i++) {
						Transform modelview=Transform(camera_transform_inverse * li->custom_transform[i]).inverse();
						li->shadow_projection[i] = bias * li->custom_projection[i] * modelview;
					}
					lights_use_shadow=true;
				}
#endif
			} break;
			case VS::LIGHT_OMNI: {

				Color linear_col = li->light_ptr->color.to_linear();
				li->light_ubo_data.light_color_energy[0]=linear_col.r;
				li->light_ubo_data.light_color_energy[1]=linear_col.g;
				li->light_ubo_data.light_color_energy[2]=linear_col.b;
				li->light_ubo_data.light_color_energy[3]=li->light_ptr->param[VS::LIGHT_PARAM_ENERGY];

				Vector3 pos = p_camera_inverse_transform.xform(li->transform.origin);

				//directional, keep at 0
				li->light_ubo_data.light_pos_inv_radius[0]=pos.x;
				li->light_ubo_data.light_pos_inv_radius[1]=pos.y;
				li->light_ubo_data.light_pos_inv_radius[2]=pos.z;
				li->light_ubo_data.light_pos_inv_radius[3]=1.0/MAX(0.001,li->light_ptr->param[VS::LIGHT_PARAM_RANGE]);

				li->light_ubo_data.light_direction_attenuation[0]=0;
				li->light_ubo_data.light_direction_attenuation[1]=0;
				li->light_ubo_data.light_direction_attenuation[2]=0;
				li->light_ubo_data.light_direction_attenuation[3]=li->light_ptr->param[VS::LIGHT_PARAM_ATTENUATION];

				li->light_ubo_data.light_params[0]=0;
				li->light_ubo_data.light_params[1]=li->light_ptr->param[VS::LIGHT_PARAM_SPECULAR];
				li->light_ubo_data.light_params[2]=0;
				li->light_ubo_data.light_params[3]=0;

#if 0

				Transform ai = p_camera_inverse_transform.affine_inverse();
				float zn = p_camera_projection.get_z_near();
				Plane p (ai.origin + ai.basis.get_axis(2) * -zn, -ai.basis.get_axis(2) );

				Vector3 point1 = li->transform.origin;
				Vector3 point2 = li->transform.origin+p_camera_inverse_transform.affine_inverse().basis.get_axis(1).normalized()*li->light_ptr->param[VS::LIGHT_PARAM_RANGE];

				p.intersects_segment(ai.origin,point1,&point1);
				p.intersects_segment(ai.origin,point2,&point2);
				float r = point1.distance_to(point2);

				float vp_w,vp_h;
				p_camera_projection.get_viewport_size(vp_w,vp_h);

				lpercent.push_back(r*2/((vp_h+vp_w)*0.5));

#endif

#if 0
				if (li->light_ptr->shadow_enabled) {
					li->shadow_projection[0] = Transform(camera_transform_inverse * li->transform).inverse();
					lights_use_shadow=true;
				}
#endif
			} break;
			case VS::LIGHT_SPOT: {

				Color linear_col = li->light_ptr->color.to_linear();
				li->light_ubo_data.light_color_energy[0]=linear_col.r;
				li->light_ubo_data.light_color_energy[1]=linear_col.g;
				li->light_ubo_data.light_color_energy[2]=linear_col.b;
				li->light_ubo_data.light_color_energy[3]=li->light_ptr->param[VS::LIGHT_PARAM_ENERGY];

				Vector3 pos = p_camera_inverse_transform.xform(li->transform.origin);

				//directional, keep at 0
				li->light_ubo_data.light_pos_inv_radius[0]=pos.x;
				li->light_ubo_data.light_pos_inv_radius[1]=pos.y;
				li->light_ubo_data.light_pos_inv_radius[2]=pos.z;
				li->light_ubo_data.light_pos_inv_radius[3]=1.0/MAX(0.001,li->light_ptr->param[VS::LIGHT_PARAM_RANGE]);

				Vector3 direction = p_camera_inverse_transform.basis.xform(li->transform.basis.xform(Vector3(0,0,-1))).normalized();
				li->light_ubo_data.light_direction_attenuation[0]=direction.x;
				li->light_ubo_data.light_direction_attenuation[1]=direction.y;
				li->light_ubo_data.light_direction_attenuation[2]=direction.z;
				li->light_ubo_data.light_direction_attenuation[3]=li->light_ptr->param[VS::LIGHT_PARAM_ATTENUATION];

				li->light_ubo_data.light_params[0]=li->light_ptr->param[VS::LIGHT_PARAM_SPOT_ATTENUATION];
				li->light_ubo_data.light_params[1]=li->light_ptr->param[VS::LIGHT_PARAM_SPECULAR];
				li->light_ubo_data.light_params[2]=0;
				li->light_ubo_data.light_params[3]=0;

#if 0
				if (li->light_ptr->shadow_enabled) {
					CameraMatrix bias;
					bias.set_light_bias();
					Transform modelview=Transform(camera_transform_inverse * li->transform).inverse();
					li->shadow_projection[0] = bias * li->projection * modelview;
					lights_use_shadow=true;
				}
#endif
			} break;

		}


		/* make light hash */

		// actually, not really a hash, but helps to sort the lights
		// and avoid recompiling redudant shader versions


		li->last_pass=render_pass;
		li->light_index=i;

		//update UBO for forward rendering, blit to texture for clustered

		glBindBuffer(GL_UNIFORM_BUFFER, li->light_ubo);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(LightInstance::LightDataUBO), &li->light_ubo_data);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);

		light_instances[i]=li;
		light_instance_count++;
	}
}

void RasterizerSceneGLES3::_copy_screen() {

	glBindVertexArray(storage->resources.quadie_array);
	glDrawArrays(GL_TRIANGLE_FAN,0,4);
	glBindVertexArray(0);

}

void RasterizerSceneGLES3::_copy_to_front_buffer(Environment *env) {

	//copy to front buffer
	glBindFramebuffer(GL_FRAMEBUFFER,storage->frame.current_rt->front.fbo);

	glDepthMask(GL_FALSE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glDepthFunc(GL_LEQUAL);
	glColorMask(1,1,1,1);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D,storage->frame.current_rt->buffers.diffuse);

	storage->shaders.copy.set_conditional(CopyShaderGLES3::DISABLE_ALPHA,true);

	if (!env) {
		//no environment, simply convert from linear to srgb
		storage->shaders.copy.set_conditional(CopyShaderGLES3::LINEAR_TO_SRGB,true);
	} else {
		storage->shaders.copy.set_conditional(CopyShaderGLES3::LINEAR_TO_SRGB,true);

	}

	storage->shaders.copy.bind();

	_copy_screen();


	//turn off everything used
	storage->shaders.copy.set_conditional(CopyShaderGLES3::LINEAR_TO_SRGB,false);
	storage->shaders.copy.set_conditional(CopyShaderGLES3::DISABLE_ALPHA,false);


}

void RasterizerSceneGLES3::render_scene(const Transform& p_cam_transform,CameraMatrix& p_cam_projection,bool p_cam_ortogonal,InstanceBase** p_cull_result,int p_cull_count,RID* p_light_cull_result,int p_light_cull_count,RID* p_directional_lights,int p_directional_light_count,RID p_environment){

	//first of all, make a new render pass
	render_pass++;

	//fill up ubo

	Environment *env = environment_owner.getornull(p_environment);

	_setup_environment(env,p_cam_projection,p_cam_transform);

	_setup_lights(p_light_cull_result,p_light_cull_count,p_cam_transform.affine_inverse(),p_cam_projection);

	render_list.clear();

	current_material_index=0;

	bool use_mrt=false;

	//fill list

	for(int i=0;i<p_cull_count;i++) {

		InstanceBase *inst = p_cull_result[i];
		switch(inst->base_type) {

			case VS::INSTANCE_MESH: {

				RasterizerStorageGLES3::Mesh *mesh = storage->mesh_owner.getptr(inst->base);
				ERR_CONTINUE(!mesh);

				int ssize = mesh->surfaces.size();

				for (int i=0;i<ssize;i++) {

					int mat_idx = inst->materials[i].is_valid() ? i : -1;
					RasterizerStorageGLES3::Surface *s = mesh->surfaces[i];
					_add_geometry(s,inst,NULL,mat_idx);
				}

				//mesh->last_pass=frame;

			} break;
			case VS::INSTANCE_MULTIMESH: {

			} break;
			case VS::INSTANCE_IMMEDIATE: {

			} break;

		}
	}

	//


	glEnable(GL_BLEND);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_SCISSOR_TEST);

	RasterizerStorageGLES3::Texture* env_radiance_tex=NULL;

	if (use_mrt) {

		glBindFramebuffer(GL_FRAMEBUFFER,storage->frame.current_rt->buffers.fbo);
		state.scene_shader.set_conditional(SceneShaderGLES3::USE_MULTIPLE_RENDER_TARGETS,true);

		Color black(0,0,0,0);
		glClearBufferfv(GL_COLOR,1,black.components); // specular
		glClearBufferfv(GL_COLOR,2,black.components); // normal metal rough

	} else {

		glBindFramebuffer(GL_FRAMEBUFFER,storage->frame.current_rt->buffers.alpha_fbo);
		state.scene_shader.set_conditional(SceneShaderGLES3::USE_MULTIPLE_RENDER_TARGETS,false);

	}


	glClearDepth(1.0);
	glClear(GL_DEPTH_BUFFER_BIT);

	Color clear_color(0,0,0,0);

	if (!env || env->bg_mode==VS::ENV_BG_CLEAR_COLOR) {

		if (storage->frame.clear_request) {

			clear_color = storage->frame.clear_request_color.to_linear();
			storage->frame.clear_request=false;

		}

	} else if (env->bg_mode==VS::ENV_BG_COLOR) {

		clear_color = env->bg_color.to_linear();
		storage->frame.clear_request=false;
	} else if (env->bg_mode==VS::ENV_BG_SKYBOX) {

		if (env->skybox_radiance.is_valid()) {
			env_radiance_tex = storage->texture_owner.getornull(env->skybox_radiance);
		}
		storage->frame.clear_request=false;

	} else {
		storage->frame.clear_request=false;
	}

	glClearBufferfv(GL_COLOR,0,clear_color.components); // specular


	state.texscreen_copied=false;

	glBlendEquation(GL_FUNC_ADD);

	if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	} else {
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	glDisable(GL_BLEND);

	render_list.sort_by_key(false);

	if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	} else {
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	_render_list(render_list.elements,render_list.element_count,p_cam_transform,p_cam_projection,env_radiance_tex,false,false);


	state.scene_shader.set_conditional(SceneShaderGLES3::USE_MULTIPLE_RENDER_TARGETS,false);

	if (env && env->bg_mode==VS::ENV_BG_SKYBOX) {

		if (use_mrt) {
			glBindFramebuffer(GL_FRAMEBUFFER,storage->frame.current_rt->buffers.alpha_fbo); //switch to alpha fbo for skybox, only diffuse/ambient matters
		}

		_draw_skybox(env->skybox_color,p_cam_projection,p_cam_transform,storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP],env->skybox_scale);
	}





	//_render_list_forward(&alpha_render_list,camera_transform,camera_transform_inverse,camera_projection,false,fragment_lighting,true);
	//glColorMask(1,1,1,1);

//	state.scene_shader.set_conditional( SceneShaderGLES3::USE_FOG,false);

	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	glEnable(GL_BLEND);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_SCISSOR_TEST);
	glBindFramebuffer(GL_FRAMEBUFFER,storage->frame.current_rt->buffers.alpha_fbo);

	render_list.sort_by_depth(true);

	_render_list(&render_list.elements[render_list.max_elements-render_list.alpha_element_count],render_list.alpha_element_count,p_cam_transform,p_cam_projection,env_radiance_tex,false,true);


	_copy_to_front_buffer(env);

#if 0
	if (use_fb) {



		for(int i=0;i<VS::ARRAY_MAX;i++) {
			glDisableVertexAttribArray(i);
		}
		glBindBuffer(GL_ARRAY_BUFFER,0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
		glDisable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
		glDisable(GL_SCISSOR_TEST);
		glDepthMask(false);

		if (current_env && current_env->fx_enabled[VS::ENV_FX_HDR]) {

			int hdr_tm = current_env->fx_param[VS::ENV_FX_PARAM_HDR_TONEMAPPER];
			switch(hdr_tm) {
				case VS::ENV_FX_HDR_TONE_MAPPER_LINEAR: {


				} break;
				case VS::ENV_FX_HDR_TONE_MAPPER_LOG: {
					copy_shader.set_conditional(CopyShaderGLES2::USE_LOG_TONEMAPPER,true);

				} break;
				case VS::ENV_FX_HDR_TONE_MAPPER_REINHARDT: {
					copy_shader.set_conditional(CopyShaderGLES2::USE_REINHARDT_TONEMAPPER,true);
				} break;
				case VS::ENV_FX_HDR_TONE_MAPPER_REINHARDT_AUTOWHITE: {

					copy_shader.set_conditional(CopyShaderGLES2::USE_REINHARDT_TONEMAPPER,true);
					copy_shader.set_conditional(CopyShaderGLES2::USE_AUTOWHITE,true);
				} break;
			}


			_process_hdr();
		}
		if (current_env && current_env->fx_enabled[VS::ENV_FX_GLOW]) {
			_process_glow_bloom();
			int glow_transfer_mode=current_env->fx_param[VS::ENV_FX_PARAM_GLOW_BLUR_BLEND_MODE];
			if (glow_transfer_mode==1)
				copy_shader.set_conditional(CopyShaderGLES2::USE_GLOW_SCREEN,true);
			if (glow_transfer_mode==2)
				copy_shader.set_conditional(CopyShaderGLES2::USE_GLOW_SOFTLIGHT,true);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, current_rt?current_rt->fbo:base_framebuffer);

		Size2 size;
		if (current_rt) {
			glBindFramebuffer(GL_FRAMEBUFFER, current_rt->fbo);
			glViewport( 0,0,viewport.width,viewport.height);
			size=Size2(viewport.width,viewport.height);
		} else {
			glBindFramebuffer(GL_FRAMEBUFFER, base_framebuffer);
			glViewport( viewport.x, window_size.height-(viewport.height+viewport.y), viewport.width,viewport.height );
			size=Size2(viewport.width,viewport.height);
		}

		//time to copy!!!
		copy_shader.set_conditional(CopyShaderGLES2::USE_BCS,current_env && current_env->fx_enabled[VS::ENV_FX_BCS]);
		copy_shader.set_conditional(CopyShaderGLES2::USE_SRGB,current_env && current_env->fx_enabled[VS::ENV_FX_SRGB]);
		copy_shader.set_conditional(CopyShaderGLES2::USE_GLOW,current_env && current_env->fx_enabled[VS::ENV_FX_GLOW]);
		copy_shader.set_conditional(CopyShaderGLES2::USE_HDR,current_env && current_env->fx_enabled[VS::ENV_FX_HDR]);
		copy_shader.set_conditional(CopyShaderGLES2::USE_NO_ALPHA,true);
		copy_shader.set_conditional(CopyShaderGLES2::USE_FXAA,current_env && current_env->fx_enabled[VS::ENV_FX_FXAA]);

		copy_shader.bind();
		//copy_shader.set_uniform(CopyShaderGLES2::SOURCE,0);

		if (current_env && current_env->fx_enabled[VS::ENV_FX_GLOW]) {

			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, framebuffer.blur[0].color );
			glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::GLOW_SOURCE),1);

		}

		if (current_env && current_env->fx_enabled[VS::ENV_FX_HDR]) {

			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, current_vd->lum_color );
			glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::HDR_SOURCE),2);
			copy_shader.set_uniform(CopyShaderGLES2::TONEMAP_EXPOSURE,float(current_env->fx_param[VS::ENV_FX_PARAM_HDR_EXPOSURE]));
			copy_shader.set_uniform(CopyShaderGLES2::TONEMAP_WHITE,float(current_env->fx_param[VS::ENV_FX_PARAM_HDR_WHITE]));

		}

		if (current_env && current_env->fx_enabled[VS::ENV_FX_FXAA])
			copy_shader.set_uniform(CopyShaderGLES2::PIXEL_SIZE,Size2(1.0/size.x,1.0/size.y));


		if (current_env && current_env->fx_enabled[VS::ENV_FX_BCS]) {

			Vector3 bcs;
			bcs.x=current_env->fx_param[VS::ENV_FX_PARAM_BCS_BRIGHTNESS];
			bcs.y=current_env->fx_param[VS::ENV_FX_PARAM_BCS_CONTRAST];
			bcs.z=current_env->fx_param[VS::ENV_FX_PARAM_BCS_SATURATION];
			copy_shader.set_uniform(CopyShaderGLES2::BCS,bcs);
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, framebuffer.color );
		glUniform1i(copy_shader.get_uniform_location(CopyShaderGLES2::SOURCE),0);

		_copy_screen_quad();

		copy_shader.set_conditional(CopyShaderGLES2::USE_BCS,false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_SRGB,false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_GLOW,false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_HDR,false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_NO_ALPHA,false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_FXAA,false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_GLOW_SCREEN,false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_GLOW_SOFTLIGHT,false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_REINHARDT_TONEMAPPER,false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_AUTOWHITE,false);
		copy_shader.set_conditional(CopyShaderGLES2::USE_LOG_TONEMAPPER,false);

		state.scene_shader.set_conditional(SceneShaderGLES3::USE_8BIT_HDR,false);


		if (current_env && current_env->fx_enabled[VS::ENV_FX_HDR] && GLOBAL_DEF("rasterizer/debug_hdr",false)) {
			_debug_luminances();
		}
	}

	current_env=NULL;
	current_debug=VS::SCENARIO_DEBUG_DISABLED;
	if (GLOBAL_DEF("rasterizer/debug_shadow_maps",false)) {
		_debug_shadows();
	}
//	_debug_luminances();
//	_debug_samplers();

	if (using_canvas_bg) {
		using_canvas_bg=false;
		glColorMask(1,1,1,1); //don't touch alpha
	}
#endif
}

void RasterizerSceneGLES3::set_scene_pass(uint64_t p_pass) {
	scene_pass=p_pass;
}

bool RasterizerSceneGLES3::free(RID p_rid) {

	if (light_instance_owner.owns(p_rid)) {


		LightInstance *light_instance = light_instance_owner.getptr(p_rid);

		//remove from shadow atlases..
		for(Set<RID>::Element *E=light_instance->shadow_atlases.front();E;E=E->next()) {
			ShadowAtlas *shadow_atlas = shadow_atlas_owner.get(E->get());
			ERR_CONTINUE(!shadow_atlas->shadow_owners.has(p_rid));
			uint32_t key = shadow_atlas->shadow_owners[p_rid];
			uint32_t q = (key>>ShadowAtlas::QUADRANT_SHIFT)&0x3;
			uint32_t s = key&ShadowAtlas::SHADOW_INDEX_MASK;

			shadow_atlas->quadrants[q].shadows[s].owner=RID();
			shadow_atlas->shadow_owners.erase(p_rid);
		}


		glDeleteBuffers(1,&light_instance->light_ubo);
		light_instance_owner.free(p_rid);
		memdelete(light_instance);

	} else if (shadow_atlas_owner.owns(p_rid)) {

		ShadowAtlas *shadow_atlas = shadow_atlas_owner.get(p_rid);
		shadow_atlas_set_size(p_rid,0);
		shadow_atlas_owner.free(p_rid);
		memdelete(shadow_atlas);

	} else {
		return false;
	}


	return true;

}

// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
static _FORCE_INLINE_ float radicalInverse_VdC(uint32_t bits) {
      bits = (bits << 16u) | (bits >> 16u);
      bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
      bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
      bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
      bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
      return float(bits) * 2.3283064365386963e-10f; // / 0x100000000
}

static _FORCE_INLINE_ Vector2 Hammersley(uint32_t i, uint32_t N) {
      return Vector2(float(i) / float(N), radicalInverse_VdC(i));
}

static _FORCE_INLINE_ Vector3 ImportanceSampleGGX(Vector2 Xi, float Roughness, Vector3 N) {
      float a = Roughness * Roughness; // DISNEY'S ROUGHNESS [see Burley'12 siggraph]

      // Compute distribution direction
      float Phi = 2.0f * M_PI * Xi.x;
      float CosTheta = Math::sqrt((1.0f - Xi.y) / (1.0f + (a*a - 1.0f) * Xi.y));
      float SinTheta = Math::sqrt((float)Math::abs(1.0f - CosTheta * CosTheta));

      // Convert to spherical direction
      Vector3 H;
      H.x = SinTheta * Math::cos(Phi);
      H.y = SinTheta * Math::sin(Phi);
      H.z = CosTheta;

      Vector3 UpVector = Math::abs(N.z) < 0.999 ? Vector3(0.0, 0.0, 1.0) : Vector3(1.0, 0.0, 0.0);
      Vector3 TangentX = UpVector.cross(N);
      TangentX.normalize();
      Vector3 TangentY = N.cross(TangentX);

      // Tangent to world space
      return TangentX * H.x + TangentY * H.y + N * H.z;
}

static _FORCE_INLINE_ float GGX(float NdotV, float a) {
	float k = a / 2.0;
	return NdotV / (NdotV * (1.0 - k) + k);
}

// http://graphicrants.blogspot.com.au/2013/08/specular-brdf-reference.html
float _FORCE_INLINE_ G_Smith(float a, float nDotV, float nDotL)
{
	return GGX(nDotL, a * a) * GGX(nDotV, a * a);
}

void RasterizerSceneGLES3::_generate_brdf() {

	int brdf_size=GLOBAL_DEF("rendering/gles3/brdf_texture_size",64);



	DVector<uint8_t> brdf;
	brdf.resize(brdf_size*brdf_size*2);

	DVector<uint8_t>::Write w = brdf.write();


	for(int i=0;i<brdf_size;i++) {
		for(int j=0;j<brdf_size;j++) {

			float Roughness = float(j)/(brdf_size-1);
			float NoV       = float(i+1)/(brdf_size); //avoid storing nov0

			Vector3 V;
			V.x = Math::sqrt( 1.0 - NoV * NoV );
			V.y = 0.0;
			V.z = NoV;

			Vector3 N = Vector3(0.0, 0.0, 1.0);

			float A = 0;
			float B = 0;

			for(int s=0;s<512;s++) {


				Vector2 xi = Hammersley(s,512);
				Vector3 H  = ImportanceSampleGGX( xi, Roughness, N );
				Vector3 L  = 2.0 * V.dot(H) * H - V;

				float NoL = CLAMP( L.z, 0.0, 1.0 );
				float NoH = CLAMP( H.z, 0.0, 1.0 );
				float VoH = CLAMP( V.dot(H), 0.0, 1.0 );

				if ( NoL > 0.0 ) {
					float G     = G_Smith( Roughness, NoV, NoL );
					float G_Vis = G * VoH / (NoH * NoV);
					float Fc    = pow(1.0 - VoH, 5.0);

					A += (1.0 - Fc) * G_Vis;
					B += Fc * G_Vis;
				}
			}

			A/=512.0;
			B/=512.0;

			int tofs = ((brdf_size-j-1)*brdf_size+i)*2;
			w[tofs+0]=CLAMP(A*255,0,255);
			w[tofs+1]=CLAMP(B*255,0,255);
		}
	}


	//set up brdf texture


	glGenTextures(1, &state.brdf_texture);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D,state.brdf_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, brdf_size, brdf_size, 0, GL_RG, GL_UNSIGNED_BYTE,w.ptr());
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D,0);

}

void RasterizerSceneGLES3::initialize() {

	state.scene_shader.init();

	default_shader = storage->shader_create(VS::SHADER_SPATIAL);
	default_material = storage->material_create();
	storage->material_set_shader(default_material,default_shader);

	glGenBuffers(1, &state.scene_ubo);
	glBindBuffer(GL_UNIFORM_BUFFER, state.scene_ubo);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(State::SceneDataUBO), &state.scene_ubo, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenBuffers(1, &state.env_radiance_ubo);
	glBindBuffer(GL_UNIFORM_BUFFER, state.env_radiance_ubo);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(State::EnvironmentRadianceUBO), &state.env_radiance_ubo, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);


	render_list.max_elements=GLOBAL_DEF("rendering/gles3/max_renderable_elements",(int)RenderList::DEFAULT_MAX_ELEMENTS);
	if (render_list.max_elements>1000000)
		render_list.max_elements=1000000;
	if (render_list.max_elements<1024)
		render_list.max_elements=1024;



	{
		//quad buffers

		glGenBuffers(1,&state.skybox_verts);
		glBindBuffer(GL_ARRAY_BUFFER,state.skybox_verts);
		glBufferData(GL_ARRAY_BUFFER,sizeof(Vector3)*8,NULL,GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER,0); //unbind


		glGenVertexArrays(1,&state.skybox_array);
		glBindVertexArray(state.skybox_array);
		glBindBuffer(GL_ARRAY_BUFFER,state.skybox_verts);
		glVertexAttribPointer(VS::ARRAY_VERTEX,3,GL_FLOAT,GL_FALSE,sizeof(Vector3)*2,0);
		glEnableVertexAttribArray(VS::ARRAY_VERTEX);
		glVertexAttribPointer(VS::ARRAY_TEX_UV,3,GL_FLOAT,GL_FALSE,sizeof(Vector3)*2,((uint8_t*)NULL)+sizeof(Vector3));
		glEnableVertexAttribArray(VS::ARRAY_TEX_UV);
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER,0); //unbind
	}
	render_list.init();
	_generate_brdf();

	shadow_atlas_realloc_tolerance_msec=500;
}

void RasterizerSceneGLES3::finalize(){


}


RasterizerSceneGLES3::RasterizerSceneGLES3()
{

}
