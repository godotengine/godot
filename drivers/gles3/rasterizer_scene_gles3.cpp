#include "rasterizer_scene_gles3.h"
#include "globals.h"
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



RID RasterizerSceneGLES3::light_instance_create(RID p_light) {


	return RID();
}

void RasterizerSceneGLES3::light_instance_set_transform(RID p_light_instance,const Transform& p_transform){


}


bool RasterizerSceneGLES3::_setup_material(RasterizerStorageGLES3::Material* p_material,bool p_alpha_pass) {

	if (p_material->shader->spatial.cull_mode==RasterizerStorageGLES3::Shader::Spatial::CULL_MODE_DISABLED) {
		glDisable(GL_CULL_FACE);
	} else {
		glEnable(GL_CULL_FACE);
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

	//material parameters

	state.scene_shader.set_custom_shader(p_material->shader->custom_code_id);
	bool rebind = state.scene_shader.bind();


	if (p_material->ubo_id) {
		glBindBufferBase(GL_UNIFORM_BUFFER,1,p_material->ubo_id);
	}



	int tc = p_material->textures.size();
	RID* textures = p_material->textures.ptr();

	for(int i=0;i<tc;i++) {

		glActiveTexture(GL_TEXTURE0+i);

		RasterizerStorageGLES3::Texture *t = storage->texture_owner.getornull( textures[i] );
		if (!t) {
			//check hints
			glBindTexture(GL_TEXTURE_2D,storage->resources.white_tex);
			continue;
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

void RasterizerSceneGLES3::_render_list(RenderList::Element **p_elements,int p_element_count,const Transform& p_view_transform,const CameraMatrix& p_projection,bool p_reverse_cull,bool p_alpha_pass) {

	if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_VFLIP]) {
		//p_reverse_cull=!p_reverse_cull;
		glFrontFace(GL_CCW);
	} else {
		glFrontFace(GL_CW);
	}

	glBindBufferBase(GL_UNIFORM_BUFFER,0,state.scene_ubo); //bind globals ubo

	state.scene_shader.set_conditional(SceneShaderGLES3::USE_SKELETON,false);

	state.current_blend_mode=-1;

	glDisable(GL_BLEND);

	RasterizerStorageGLES3::Material* prev_material=NULL;
	RasterizerStorageGLES3::Geometry* prev_geometry=NULL;
	VS::InstanceType prev_base_type = VS::INSTANCE_MAX;

	for (int i=0;i<p_element_count;i++) {

		RenderList::Element *e = p_elements[i];
		RasterizerStorageGLES3::Material* material= e->material;

		bool rebind=i==0;

		if (material!=prev_material || rebind) {

			rebind = _setup_material(material,p_alpha_pass);
//			_rinfo.mat_change_count++;
		}


		if (prev_base_type != e->instance->base_type || prev_geometry!=e->geometry) {

			_setup_geometry(e);
		}

//		_set_cull(e->mirror,p_reverse_cull);

		state.scene_shader.set_uniform(SceneShaderGLES3::NORMAL_MULT, e->instance->mirror?-1.0:1.0);
		state.scene_shader.set_uniform(SceneShaderGLES3::WORLD_TRANSFORM, e->instance->transform);


//		_render(e->geometry, material, skeleton,e->owner,e->instance->transform);

		_render_geometry(e);

		prev_material=material;
		prev_base_type=e->instance->base_type;
		prev_geometry=e->geometry;
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



	//bool has_base_alpha=(m->shader_cache && m->shader_cache->has_alpha);
	//bool has_blend_alpha=m->blend_mode!=VS::MATERIAL_BLEND_MODE_MIX || m->flags[VS::MATERIAL_FLAG_ONTOP];
	bool has_alpha = false; //has_base_alpha || has_blend_alpha;

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

	e->sort_key|=uint64_t(e->instance->base_type)<<RenderList::SORT_KEY_GEOMETRY_INDEX_SHIFT;
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

//	if (m->flags[VS::MATERIAL_FLAG_INVERT_FACES])
//		e->mirror=!e->mirror;

	if (mirror) {
		e->sort_key|=RenderList::SORT_KEY_MIRROR_FLAG;
	}

	//e->light_type=0xFF; // no lights!
	e->sort_key|=uint64_t(0xF)<<RenderList::SORT_KEY_LIGHT_TYPE_SHIFT; //light type 0xF is no light?
	e->sort_key|=uint64_t(0xFFFF)<<RenderList::SORT_KEY_LIGHT_INDEX_SHIFT;
/* prepass
	if (!shadow && !has_blend_alpha && has_alpha && m->depth_draw_mode==VS::MATERIAL_DEPTH_DRAW_OPAQUE_PRE_PASS_ALPHA) {

		//if nothing exists, add this element as opaque too
		RenderList::Element *oe = opaque_render_list.add_element();

		if (!oe)
			return;

		memcpy(oe,e,sizeof(RenderList::Element));
		oe->additive_ptr=&oe->additive;
	}
*/

#if 0
	if (shadow || m->flags[VS::MATERIAL_FLAG_UNSHADED] || current_debug==VS::SCENARIO_DEBUG_SHADELESS) {

		e->light_type=0x7F; //unshaded is zero
	} else {

		bool duplicate=false;


		for(int i=0;i<directional_light_count;i++) {
			uint16_t sort_key = directional_lights[i]->sort_key;
			uint8_t light_type = VS::LIGHT_DIRECTIONAL;
			if (directional_lights[i]->base->shadow_enabled) {
				light_type|=0x8;
				if (directional_lights[i]->base->directional_shadow_mode==VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS)
					light_type|=0x10;
				else if (directional_lights[i]->base->directional_shadow_mode==VS::LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS)
					light_type|=0x30;

			}

			RenderList::Element *ec;
			if (duplicate) {

				ec = render_list->add_element();
				memcpy(ec,e,sizeof(RenderList::Element));
			} else {

				ec=e;
				duplicate=true;
			}

			ec->light_type=light_type;
			ec->light=sort_key;
			ec->additive_ptr=&e->additive;

		}


		const RID *liptr = p_instance->light_instances.ptr();
		int ilc=p_instance->light_instances.size();



		for(int i=0;i<ilc;i++) {

			LightInstance *li=light_instance_owner.get( liptr[i] );
			if (!li || li->last_pass!=scene_pass) //lit by light not in visible scene
				continue;
			uint8_t light_type=li->base->type|0x40; //penalty to ensure directionals always go first
			if (li->base->shadow_enabled) {
				light_type|=0x8;
			}
			uint16_t sort_key =li->sort_key;

			RenderList::Element *ec;
			if (duplicate) {

				ec = render_list->add_element();
				memcpy(ec,e,sizeof(RenderList::Element));
			} else {

				duplicate=true;
				ec=e;
			}

			ec->light_type=light_type;
			ec->light=sort_key;
			ec->additive_ptr=&e->additive;

		}



	}

#endif
}

void RasterizerSceneGLES3::render_scene(const Transform& p_cam_transform,CameraMatrix& p_cam_projection,bool p_cam_ortogonal,InstanceBase** p_cull_result,int p_cull_count,RID* p_light_cull_result,int p_light_cull_count,RID* p_directional_lights,int p_directional_light_count,RID p_environment){


	//fill up ubo

	store_camera(p_cam_projection,state.ubo_data.projection_matrix);
	store_transform(p_cam_transform,state.ubo_data.camera_matrix);
	store_transform(p_cam_transform.affine_inverse(),state.ubo_data.camera_inverse_matrix);
	for(int i=0;i<4;i++) {
		state.ubo_data.time[i]=storage->frame.time[i];
	}


	glBindBuffer(GL_UNIFORM_BUFFER, state.scene_ubo);
	glBufferSubData(GL_UNIFORM_BUFFER, 0,sizeof(State::SceneDataUBO), &state.ubo_data);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);


	render_list.clear();

	render_pass++;
	current_material_index=0;

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
	glClearDepth(1.0);
	glBindFramebuffer(GL_FRAMEBUFFER,storage->frame.current_rt->front.fbo);


	if (true) {

		if (storage->frame.clear_request) {

			glClearColor( storage->frame.clear_request_color.r, storage->frame.clear_request_color.g, storage->frame.clear_request_color.b, storage->frame.clear_request_color.a );
			glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
			storage->frame.clear_request=false;

		}
	}

	state.current_depth_test=true;
	state.current_depth_mask=true;
	state.texscreen_copied=false;

	glBlendEquation(GL_FUNC_ADD);

	if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	} else {
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	glDisable(GL_BLEND);
	//current_blend_mode=VS::MATERIAL_BLEND_MODE_MIX;


	render_list.sort_by_key(false);

	//_render_list_forward(&opaque_render_list,camera_transform,camera_transform_inverse,camera_projection,false,fragment_lighting);
/*
	if (draw_tex_background) {

		//most 3D vendors recommend drawing a texture bg or skybox here,
		//after opaque geometry has been drawn
		//so the zbuffer can get rid of most pixels
		_draw_tex_bg();
	}
*/
	if (storage->frame.current_rt->flags[RasterizerStorage::RENDER_TARGET_TRANSPARENT]) {
		glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	} else {
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

//	glDisable(GL_BLEND);
//	current_blend_mode=VS::MATERIAL_BLEND_MODE_MIX;
//	state.scene_shader.set_conditional(SceneShaderGLES3::USE_GLOW,false);
//	if (current_env && current_env->fx_enabled[VS::ENV_FX_GLOW]) {
//		glColorMask(1,1,1,0); //don't touch alpha
//	}


	_render_list(render_list.elements,render_list.element_count,p_cam_transform,p_cam_projection,false,false);

	//_render_list_forward(&alpha_render_list,camera_transform,camera_transform_inverse,camera_projection,false,fragment_lighting,true);
	//glColorMask(1,1,1,1);

//	state.scene_shader.set_conditional( SceneShaderGLES3::USE_FOG,false);

	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
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

bool RasterizerSceneGLES3::free(RID p_rid) {

	return false;

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

	render_list.max_elements=GLOBAL_DEF("rendering/gles3/max_renderable_elements",(int)RenderList::DEFAULT_MAX_ELEMENTS);
	if (render_list.max_elements>1000000)
		render_list.max_elements=1000000;
	if (render_list.max_elements<1024)
		render_list.max_elements=1024;

	render_list.init();
}

void RasterizerSceneGLES3::finalize(){


}


RasterizerSceneGLES3::RasterizerSceneGLES3()
{

}
