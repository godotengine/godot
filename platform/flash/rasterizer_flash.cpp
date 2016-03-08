/*************************************************/
/*  rasterizer_gles2.cpp                          */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2016 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#include "rasterizer_flash.h"
#include "os/os.h"
#include "globals.h"
#include <stdio.h>
#include "servers/visual/particle_system_sw.h"
#include <string.h>

_FORCE_INLINE_ static void _set_color_attrib(const Color& p_color) {

}

RasterizerFlash::FX::FX() {

	bgcolor_active=false;
	bgcolor=Color(0,1,0,1);

	skybox_active=false;

	glow_active=false;
	glow_passes=4;
	glow_attenuation=0.7;
	glow_bloom=0.0;

	antialias_active=true;
	antialias_tolerance=15;

	ssao_active=true;
	ssao_attenuation=0.7;
	ssao_radius=0.18;
	ssao_max_distance=1.0;
	ssao_range_min=0.25;
	ssao_range_max=0.48;
	ssao_only=false;


	fog_active=false;
	fog_near=5;
	fog_far=100;
	fog_attenuation=1.0;
	fog_color_near=Color(1,1,1,1);
	fog_color_far=Color(1,1,1,1);
	fog_bg=false;

	toon_active=false;
	toon_treshold=0.4;
	toon_soft=0.001;

	edge_active=false;
	edge_color=Color(0,0,0,1);
	edge_size=1.0;

}


void RasterizerFlash::_draw_primitive(int p_points, const Vector3 *p_vertices, const Vector3 *p_normals, const Color* p_colors, const Vector3 *p_uvs,const Plane *p_tangents,int p_instanced) {

};

RID RasterizerFlash::texture_create() {

	Texture *texture = memnew(Texture);

	return texture_owner.make_rid( texture );

}

void RasterizerFlash::texture_allocate(RID p_texture,int p_width, int p_height,Image::Format p_format,uint32_t p_flags,int p_mipmap_count) {

}

void RasterizerFlash::texture_set_data(RID p_texture,const Image& p_image,VS::CubeMapSide p_cube_side) {

}

Image RasterizerFlash::texture_get_data(RID p_texture,VS::CubeMapSide p_cube_side) const {

	return Image();
}

void RasterizerFlash::texture_set_flags(RID p_texture,uint32_t p_flags) {

}
uint32_t RasterizerFlash::texture_get_flags(RID p_texture) const {

	Texture * texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture,0);

	return texture->flags;

}
Image::Format RasterizerFlash::texture_get_format(RID p_texture) const {

	Texture * texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture,Image::FORMAT_GRAYSCALE);

	return texture->format;
}
uint32_t RasterizerFlash::texture_get_width(RID p_texture) const {

	Texture * texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture,0);

	return texture->width;
}
uint32_t RasterizerFlash::texture_get_height(RID p_texture) const {

	Texture * texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture,0);

	return texture->height;
}

bool RasterizerFlash::texture_has_alpha(RID p_texture) const {

	Texture * texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture,0);

	return texture->has_alpha;

}

void RasterizerFlash::texture_set_size_override(RID p_texture,int p_width, int p_height) {

	Texture * texture = texture_owner.get(p_texture);

	ERR_FAIL_COND(!texture);

	ERR_FAIL_COND(p_width<=0 || p_width>4096);
	ERR_FAIL_COND(p_height<=0 || p_height>4096);
	//real texture size is in alloc width and height
	texture->width=p_width;
	texture->height=p_height;

}

/* SHADER API */

RID RasterizerFlash::shader_create(VS::ShaderMode p_mode) {

	Shader *shader = memnew( Shader );
	shader->mode=p_mode;
	RID rid = shader_owner.make_rid(shader);
	shader_set_mode(rid,p_mode);
	_shader_make_dirty(shader);

	return rid;

}



void RasterizerFlash::shader_set_mode(RID p_shader,VS::ShaderMode p_mode) {


}
VS::ShaderMode RasterizerFlash::shader_get_mode(RID p_shader) const {

	Shader *shader=shader_owner.get(p_shader);
	ERR_FAIL_COND_V(!shader,VS::SHADER_MATERIAL);
	return shader->mode;
}



void RasterizerFlash::shader_set_code(RID p_shader, const String& p_vertex, const String& p_fragment,int p_vertex_ofs,int p_fragment_ofs) {

	Shader *shader=shader_owner.get(p_shader);
	ERR_FAIL_COND(!shader);

#ifdef DEBUG_ENABLED
	if (shader->vertex_code==p_vertex && shader->fragment_code==p_fragment)
		return;
#endif
	shader->fragment_code=p_fragment;
	shader->vertex_code=p_vertex;
	shader->fragment_line=p_fragment_ofs;
	shader->vertex_line=p_vertex_ofs;
	_shader_make_dirty(shader);

}

String RasterizerFlash::shader_get_vertex_code(RID p_shader) const {

	Shader *shader=shader_owner.get(p_shader);
	ERR_FAIL_COND_V(!shader,String());
	return shader->vertex_code;

}

String RasterizerFlash::shader_get_fragment_code(RID p_shader) const {

	Shader *shader=shader_owner.get(p_shader);
	ERR_FAIL_COND_V(!shader,String());
	return shader->fragment_code;

}

void RasterizerFlash::_shader_make_dirty(Shader* p_shader) {

	if (p_shader->dirty_list.in_list())
		return;

	_shader_dirty_list.add(&p_shader->dirty_list);
}

void RasterizerFlash::shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const {

	Shader *shader=shader_owner.get(p_shader);
	ERR_FAIL_COND(!shader);


	if (shader->dirty_list.in_list())
		_update_shader(shader); // ok should be not anymore dirty


	Map<int,StringName> order;


	for(Map<StringName,ShaderLanguage::Uniform>::Element *E=shader->uniforms.front();E;E=E->next()) {


		order[E->get().order]=E->key();
	}


	for(Map<int,StringName>::Element *E=order.front();E;E=E->next()) {

		PropertyInfo pi;
		ShaderLanguage::Uniform &u=shader->uniforms[E->get()];
		pi.name=E->get();
		switch(u.type) {

			case ShaderLanguage::TYPE_VOID:
			case ShaderLanguage::TYPE_BOOL:
			case ShaderLanguage::TYPE_FLOAT:
			case ShaderLanguage::TYPE_VEC2:
			case ShaderLanguage::TYPE_VEC3:
			case ShaderLanguage::TYPE_MAT3:
			case ShaderLanguage::TYPE_MAT4:
			case ShaderLanguage::TYPE_VEC4:
				pi.type=u.default_value.get_type();
				break;
			case ShaderLanguage::TYPE_TEXTURE:
				pi.type=Variant::_RID;
				pi.hint=PROPERTY_HINT_RESOURCE_TYPE;
				pi.hint_string="Texture";
				break;
			case ShaderLanguage::TYPE_CUBEMAP:
				pi.type=Variant::_RID;
				pi.hint=PROPERTY_HINT_RESOURCE_TYPE;
				pi.hint_string="Texture";
				break;
		};

		p_param_list->push_back(pi);

	}

}


/* COMMON MATERIAL API */


RID RasterizerFlash::material_create() {

	return material_owner.make_rid( memnew( Material ) );
}

void RasterizerFlash::material_set_shader(RID p_material, RID p_shader) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);
	if (material->shader==p_shader)
		return;
	material->shader=p_shader;
	material->shader_version=0;

}

RID RasterizerFlash::material_get_shader(RID p_material) const {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material,RID());
	return material->shader;
}


void RasterizerFlash::material_set_param(RID p_material, const StringName& p_param, const Variant& p_value) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);

	Map<StringName,Material::UniformData>::Element *E=material->shader_params.find(p_param);
	if (E) {

		if (p_value.get_type()==Variant::NIL) {

			material->shader_params.erase(E);
			material->shader_version=0; //get default!
		} else {
			E->get().value=p_value;
		}
	} else {

		Material::UniformData ud;
		ud.index=-1;
		ud.value=p_value;
		ud.istexture=p_value.get_type()==Variant::_RID; /// cache it being texture
		material->shader_params[p_param]=ud; //may be got at some point, or erased

	}
}
Variant RasterizerFlash::material_get_param(RID p_material, const StringName& p_param) const {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material,Variant());


	if (material->shader.is_valid()) {
		//update shader params if necesary
		//make sure the shader is compiled and everything
		//so the actual parameters can be properly retrieved!
		material->shader_cache=shader_owner.get( material->shader );
		if (!material->shader_cache) {
			//invalidate
			material->shader=RID();
			material->shader_cache=NULL;
		} else {

			if (material->shader_cache->dirty_list.in_list())
				_update_shader(material->shader_cache);
			if (material->shader_cache->valid && material->shader_cache->version!=material->shader_version) {
				//validate
				_update_material_shader_params(material);
			}
		}
	}


	if (material->shader_params.has(p_param))
		return material->shader_params[p_param].value;
	else
		return Variant();
}


void RasterizerFlash::material_set_flag(RID p_material, VS::MaterialFlag p_flag,bool p_enabled) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);
	ERR_FAIL_INDEX(p_flag,VS::MATERIAL_FLAG_MAX);
	material->flags[p_flag]=p_enabled;

}
bool RasterizerFlash::material_get_flag(RID p_material,VS::MaterialFlag p_flag) const {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material,false);
	ERR_FAIL_INDEX_V(p_flag,VS::MATERIAL_FLAG_MAX,false);
	return material->flags[p_flag];


}

void RasterizerFlash::material_set_hint(RID p_material, VS::MaterialHint p_hint,bool p_enabled) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);
	ERR_FAIL_INDEX(p_hint,VS::MATERIAL_HINT_MAX);
	material->hints[p_hint]=p_enabled;

}

bool RasterizerFlash::material_get_hint(RID p_material,VS::MaterialHint p_hint) const {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material,false);
	ERR_FAIL_INDEX_V(p_hint,VS::MATERIAL_HINT_MAX,false);
	return material->hints[p_hint];

}

void RasterizerFlash::material_set_shade_model(RID p_material, VS::MaterialShadeModel p_model) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);
	material->shade_model=p_model;

};

VS::MaterialShadeModel RasterizerFlash::material_get_shade_model(RID p_material) const {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material,VS::MATERIAL_SHADE_MODEL_LAMBERT);
	return material->shade_model;
};


void RasterizerFlash::material_set_blend_mode(RID p_material,VS::MaterialBlendMode p_mode) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);
	material->blend_mode=p_mode;

}
VS::MaterialBlendMode RasterizerFlash::material_get_blend_mode(RID p_material) const {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material,VS::MATERIAL_BLEND_MODE_ADD);
	return material->blend_mode;
}

void RasterizerFlash::material_set_line_width(RID p_material,float p_line_width) {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND(!material);
	material->line_width=p_line_width;

}
float RasterizerFlash::material_get_line_width(RID p_material) const {

	Material *material = material_owner.get(p_material);
	ERR_FAIL_COND_V(!material,0);

	return material->line_width;
}



/* MESH API */

RID RasterizerFlash::mesh_create() {

	return mesh_owner.make_rid( memnew( Mesh ) );
}



void RasterizerFlash::mesh_add_surface(RID p_mesh,VS::PrimitiveType p_primitive,const Array& p_arrays,const Array& p_blend_shapes) {

}

Error RasterizerFlash::_surface_set_arrays(Surface *p_surface, uint8_t *p_mem,uint8_t *p_index_mem,const Array& p_arrays,bool p_main) {

	return FAILED;
}



void RasterizerFlash::mesh_add_custom_surface(RID p_mesh,const Variant& p_dat) {

	ERR_EXPLAIN("OpenGL Rasterizer does not support custom surfaces. Running on wrong platform?");
	ERR_FAIL_V();
}

Array RasterizerFlash::mesh_get_surface_arrays(RID p_mesh,int p_surface) const {

	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND_V(!mesh,Array());
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), Array() );
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V( !surface, Array() );

	return surface->data;


}
Array RasterizerFlash::mesh_get_surface_morph_arrays(RID p_mesh,int p_surface) const{

	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND_V(!mesh,Array());
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), Array() );
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V( !surface, Array() );

	return surface->morph_data;

}


void RasterizerFlash::mesh_set_morph_target_count(RID p_mesh,int p_amount) {

	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_COND( mesh->surfaces.size()!=0 );

	mesh->morph_target_count=p_amount;

}

int RasterizerFlash::mesh_get_morph_target_count(RID p_mesh) const{

	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND_V(!mesh,-1);

	return mesh->morph_target_count;

}

void RasterizerFlash::mesh_set_morph_target_mode(RID p_mesh,VS::MorphTargetMode p_mode) {

	ERR_FAIL_INDEX(p_mode,2);
	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND(!mesh);

	mesh->morph_target_mode=p_mode;

}

VS::MorphTargetMode RasterizerFlash::mesh_get_morph_target_mode(RID p_mesh) const {

	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND_V(!mesh,VS::MORPH_MODE_NORMALIZED);

	return mesh->morph_target_mode;

}



void RasterizerFlash::mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material,bool p_owned) {

	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_INDEX(p_surface, mesh->surfaces.size() );
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND( !surface);

	if (surface->material_owned && surface->material.is_valid())
		free(surface->material);

	surface->material_owned=p_owned;

	surface->material=p_material;
}

RID RasterizerFlash::mesh_surface_get_material(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND_V(!mesh,RID());
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), RID() );
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V( !surface, RID() );

	return surface->material;
}

int RasterizerFlash::mesh_surface_get_array_len(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND_V(!mesh,-1);
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), -1 );
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V( !surface, -1 );

	return surface->array_len;
}
int RasterizerFlash::mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND_V(!mesh,-1);
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), -1 );
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V( !surface, -1 );

	return surface->index_array_len;
}
uint32_t RasterizerFlash::mesh_surface_get_format(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND_V(!mesh,0);
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), 0 );
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V( !surface, 0 );

	return surface->format;
}
VS::PrimitiveType RasterizerFlash::mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const {

	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND_V(!mesh,VS::PRIMITIVE_POINTS);
	ERR_FAIL_INDEX_V(p_surface, mesh->surfaces.size(), VS::PRIMITIVE_POINTS );
	Surface *surface = mesh->surfaces[p_surface];
	ERR_FAIL_COND_V( !surface, VS::PRIMITIVE_POINTS );

	return surface->primitive;
}

void RasterizerFlash::mesh_remove_surface(RID p_mesh,int p_index) {

	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_INDEX(p_index, mesh->surfaces.size() );
	Surface *surface = mesh->surfaces[p_index];
	ERR_FAIL_COND( !surface);

	if (mesh->morph_target_count) {
		for(int i=0;i<mesh->morph_target_count;i++)
			memfree(surface->morph_targets_local[i].array);
		memfree( surface->morph_targets_local );
	}

	memdelete( mesh->surfaces[p_index] );
	mesh->surfaces.remove(p_index);

}
int RasterizerFlash::mesh_get_surface_count(RID p_mesh) const {

	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND_V(!mesh,-1);

	return mesh->surfaces.size();
}

AABB RasterizerFlash::mesh_get_aabb(RID p_mesh) const {

	Mesh *mesh = mesh_owner.get( p_mesh );
	ERR_FAIL_COND_V(!mesh,AABB());

	AABB aabb;

	for (int i=0;i<mesh->surfaces.size();i++) {

		if (i==0)
			aabb=mesh->surfaces[i]->aabb;
		else
			aabb.merge_with(mesh->surfaces[i]->aabb);
	}

	return aabb;
}
/* MULTIMESH API */

RID RasterizerFlash::multimesh_create() {

	return multimesh_owner.make_rid( memnew( MultiMesh ));
}

void RasterizerFlash::multimesh_set_instance_count(RID p_multimesh,int p_count) {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND(!multimesh);

	multimesh->elements.clear(); // make sure to delete everything, so it "fails" in all implementations
	multimesh->elements.resize(p_count);

}
int RasterizerFlash::multimesh_get_instance_count(RID p_multimesh) const {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND_V(!multimesh,-1);

	return multimesh->elements.size();
}

void RasterizerFlash::multimesh_set_mesh(RID p_multimesh,RID p_mesh) {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND(!multimesh);

	multimesh->mesh=p_mesh;

}
void RasterizerFlash::multimesh_set_aabb(RID p_multimesh,const AABB& p_aabb) {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	multimesh->aabb=p_aabb;
}
void RasterizerFlash::multimesh_instance_set_transform(RID p_multimesh,int p_index,const Transform& p_transform) {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	ERR_FAIL_INDEX(p_index,multimesh->elements.size());
	MultiMesh::Element &e=multimesh->elements[p_index];

	e.matrix[0]=p_transform.basis.elements[0][0];
	e.matrix[1]=p_transform.basis.elements[1][0];
	e.matrix[2]=p_transform.basis.elements[2][0];
	e.matrix[3]=0;
	e.matrix[4]=p_transform.basis.elements[0][1];
	e.matrix[5]=p_transform.basis.elements[1][1];
	e.matrix[6]=p_transform.basis.elements[2][1];
	e.matrix[7]=0;
	e.matrix[8]=p_transform.basis.elements[0][2];
	e.matrix[9]=p_transform.basis.elements[1][2];
	e.matrix[10]=p_transform.basis.elements[2][2];
	e.matrix[11]=0;
	e.matrix[12]=p_transform.origin.x;
	e.matrix[13]=p_transform.origin.y;
	e.matrix[14]=p_transform.origin.z;
	e.matrix[15]=1;

}
void RasterizerFlash::multimesh_instance_set_color(RID p_multimesh,int p_index,const Color& p_color) {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND(!multimesh)
	ERR_FAIL_INDEX(p_index,multimesh->elements.size());
	MultiMesh::Element &e=multimesh->elements[p_index];
	e.color[0]=CLAMP(p_color.r*255,0,255);
	e.color[1]=CLAMP(p_color.g*255,0,255);
	e.color[2]=CLAMP(p_color.b*255,0,255);
	e.color[3]=CLAMP(p_color.a*255,0,255);


}

RID RasterizerFlash::multimesh_get_mesh(RID p_multimesh) const {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND_V(!multimesh,RID());

	return multimesh->mesh;
}
AABB RasterizerFlash::multimesh_get_aabb(RID p_multimesh) const {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND_V(!multimesh,AABB());

	return multimesh->aabb;
}

Transform RasterizerFlash::multimesh_instance_get_transform(RID p_multimesh,int p_index) const {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND_V(!multimesh,Transform());

	ERR_FAIL_INDEX_V(p_index,multimesh->elements.size(),Transform());
	MultiMesh::Element &e=multimesh->elements[p_index];

	Transform tr;

	tr.basis.elements[0][0]=e.matrix[0];
	tr.basis.elements[1][0]=e.matrix[1];
	tr.basis.elements[2][0]=e.matrix[2];
	tr.basis.elements[0][1]=e.matrix[4];
	tr.basis.elements[1][1]=e.matrix[5];
	tr.basis.elements[2][1]=e.matrix[6];
	tr.basis.elements[0][2]=e.matrix[8];
	tr.basis.elements[1][2]=e.matrix[9];
	tr.basis.elements[2][2]=e.matrix[10];
	tr.origin.x=e.matrix[12];
	tr.origin.y=e.matrix[13];
	tr.origin.z=e.matrix[14];

	return tr;
}
Color RasterizerFlash::multimesh_instance_get_color(RID p_multimesh,int p_index) const {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND_V(!multimesh,Color());
	ERR_FAIL_INDEX_V(p_index,multimesh->elements.size(),Color());
	MultiMesh::Element &e=multimesh->elements[p_index];
	Color c;
	c.r=e.color[0]/255.0;
	c.g=e.color[1]/255.0;
	c.b=e.color[2]/255.0;
	c.a=e.color[3]/255.0;

	return c;

}

void RasterizerFlash::multimesh_set_visible_instances(RID p_multimesh,int p_visible) {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	multimesh->visible=p_visible;

}

int RasterizerFlash::multimesh_get_visible_instances(RID p_multimesh) const {

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND_V(!multimesh,-1);
	return multimesh->visible;

}


/* PARTICLES API */

RID RasterizerFlash::particles_create() {

	Particles *particles = memnew( Particles );
	ERR_FAIL_COND_V(!particles,RID());
	return particles_owner.make_rid(particles);
}

void RasterizerFlash::particles_set_amount(RID p_particles, int p_amount) {

	ERR_FAIL_COND(p_amount<1);
	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);
	particles->data.amount=p_amount;

}

int RasterizerFlash::particles_get_amount(RID p_particles) const {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,-1);
	return particles->data.amount;

}

void RasterizerFlash::particles_set_emitting(RID p_particles, bool p_emitting) {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);
	particles->data.emitting=p_emitting;;

}
bool RasterizerFlash::particles_is_emitting(RID p_particles) const {

	const Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,false);
	return particles->data.emitting;

}

void RasterizerFlash::particles_set_visibility_aabb(RID p_particles, const AABB& p_visibility) {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);
	particles->data.visibility_aabb=p_visibility;

}

void RasterizerFlash::particles_set_emission_half_extents(RID p_particles, const Vector3& p_half_extents) {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);

	particles->data.emission_half_extents=p_half_extents;
}
Vector3 RasterizerFlash::particles_get_emission_half_extents(RID p_particles) const {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,Vector3());

	return particles->data.emission_half_extents;
}

void RasterizerFlash::particles_set_emission_base_velocity(RID p_particles, const Vector3& p_base_velocity) {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);

	particles->data.emission_base_velocity=p_base_velocity;
}

Vector3 RasterizerFlash::particles_get_emission_base_velocity(RID p_particles) const {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,Vector3());

	return particles->data.emission_base_velocity;
}


void RasterizerFlash::particles_set_emission_points(RID p_particles, const DVector<Vector3>& p_points) {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);

	particles->data.emission_points=p_points;
}

DVector<Vector3> RasterizerFlash::particles_get_emission_points(RID p_particles) const {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,DVector<Vector3>());

	return particles->data.emission_points;

}

void RasterizerFlash::particles_set_gravity_normal(RID p_particles, const Vector3& p_normal) {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);

	particles->data.gravity_normal=p_normal;

}
Vector3 RasterizerFlash::particles_get_gravity_normal(RID p_particles) const {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,Vector3());

	return particles->data.gravity_normal;
}


AABB RasterizerFlash::particles_get_visibility_aabb(RID p_particles) const {

	const Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,AABB());
	return particles->data.visibility_aabb;

}

void RasterizerFlash::particles_set_variable(RID p_particles, VS::ParticleVariable p_variable,float p_value) {

	ERR_FAIL_INDEX(p_variable,VS::PARTICLE_VAR_MAX);

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);
	particles->data.particle_vars[p_variable]=p_value;

}
float RasterizerFlash::particles_get_variable(RID p_particles, VS::ParticleVariable p_variable) const {

	const Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,-1);
	return particles->data.particle_vars[p_variable];
}

void RasterizerFlash::particles_set_randomness(RID p_particles, VS::ParticleVariable p_variable,float p_randomness) {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);
	particles->data.particle_randomness[p_variable]=p_randomness;

}
float RasterizerFlash::particles_get_randomness(RID p_particles, VS::ParticleVariable p_variable) const {

	const Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,-1);
	return particles->data.particle_randomness[p_variable];

}

void RasterizerFlash::particles_set_color_phases(RID p_particles, int p_phases) {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);
	ERR_FAIL_COND( p_phases<0 || p_phases>VS::MAX_PARTICLE_COLOR_PHASES );
	particles->data.color_phase_count=p_phases;

}
int RasterizerFlash::particles_get_color_phases(RID p_particles) const {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,-1);
	return particles->data.color_phase_count;
}


void RasterizerFlash::particles_set_color_phase_pos(RID p_particles, int p_phase, float p_pos) {

	ERR_FAIL_INDEX(p_phase, VS::MAX_PARTICLE_COLOR_PHASES);
	if (p_pos<0.0)
		p_pos=0.0;
	if (p_pos>1.0)
		p_pos=1.0;

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);
	particles->data.color_phases[p_phase].pos=p_pos;

}
float RasterizerFlash::particles_get_color_phase_pos(RID p_particles, int p_phase) const {

	ERR_FAIL_INDEX_V(p_phase, VS::MAX_PARTICLE_COLOR_PHASES, -1.0);

	const Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,-1);
	return particles->data.color_phases[p_phase].pos;

}

void RasterizerFlash::particles_set_color_phase_color(RID p_particles, int p_phase, const Color& p_color) {

	ERR_FAIL_INDEX(p_phase, VS::MAX_PARTICLE_COLOR_PHASES);
	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);
	particles->data.color_phases[p_phase].color=p_color;

	//update alpha
	particles->has_alpha=false;
	for(int i=0;i<VS::MAX_PARTICLE_COLOR_PHASES;i++) {
		if (particles->data.color_phases[i].color.a<0.99)
			particles->has_alpha=true;
	}

}

Color RasterizerFlash::particles_get_color_phase_color(RID p_particles, int p_phase) const {

	ERR_FAIL_INDEX_V(p_phase, VS::MAX_PARTICLE_COLOR_PHASES, Color());

	const Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,Color());
	return particles->data.color_phases[p_phase].color;

}

void RasterizerFlash::particles_set_attractors(RID p_particles, int p_attractors) {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);
	ERR_FAIL_COND( p_attractors<0 || p_attractors>VisualServer::MAX_PARTICLE_ATTRACTORS );
	particles->data.attractor_count=p_attractors;

}
int RasterizerFlash::particles_get_attractors(RID p_particles) const {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,-1);
	return particles->data.attractor_count;
}

void RasterizerFlash::particles_set_attractor_pos(RID p_particles, int p_attractor, const Vector3& p_pos) {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);
	ERR_FAIL_INDEX(p_attractor,particles->data.attractor_count);
	particles->data.attractors[p_attractor].pos=p_pos;;
}
Vector3 RasterizerFlash::particles_get_attractor_pos(RID p_particles,int p_attractor) const {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,Vector3());
	ERR_FAIL_INDEX_V(p_attractor,particles->data.attractor_count,Vector3());
	return particles->data.attractors[p_attractor].pos;
}

void RasterizerFlash::particles_set_attractor_strength(RID p_particles, int p_attractor, float p_force) {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);
	ERR_FAIL_INDEX(p_attractor,particles->data.attractor_count);
	particles->data.attractors[p_attractor].force=p_force;
}

float RasterizerFlash::particles_get_attractor_strength(RID p_particles,int p_attractor) const {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,0);
	ERR_FAIL_INDEX_V(p_attractor,particles->data.attractor_count,0);
	return particles->data.attractors[p_attractor].force;
}

void RasterizerFlash::particles_set_material(RID p_particles, RID p_material,bool p_owned) {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);
	if (particles->material_owned && particles->material.is_valid())
		free(particles->material);

	particles->material_owned=p_owned;

	particles->material=p_material;

}
RID RasterizerFlash::particles_get_material(RID p_particles) const {

	const Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,RID());
	return particles->material;

}

void RasterizerFlash::particles_set_use_local_coordinates(RID p_particles, bool p_enable) {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);
	particles->data.local_coordinates=p_enable;

}

bool RasterizerFlash::particles_is_using_local_coordinates(RID p_particles) const {

	const Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,false);
	return particles->data.local_coordinates;
}
bool RasterizerFlash::particles_has_height_from_velocity(RID p_particles) const {

	const Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,false);
	return particles->data.height_from_velocity;
}

void RasterizerFlash::particles_set_height_from_velocity(RID p_particles, bool p_enable) {

	Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND(!particles);
	particles->data.height_from_velocity=p_enable;

}

AABB RasterizerFlash::particles_get_aabb(RID p_particles) const {

	const Particles* particles = particles_owner.get( p_particles );
	ERR_FAIL_COND_V(!particles,AABB());
	return particles->data.visibility_aabb;
}

/* SKELETON API */

RID RasterizerFlash::skeleton_create() {

	Skeleton *skeleton = memnew( Skeleton );
	ERR_FAIL_COND_V(!skeleton,RID());
	return skeleton_owner.make_rid( skeleton );
}
void RasterizerFlash::skeleton_resize(RID p_skeleton,int p_bones) {

	Skeleton *skeleton = skeleton_owner.get( p_skeleton );
	ERR_FAIL_COND(!skeleton);
	if (p_bones == skeleton->bones.size()) {
		return;
	};

	skeleton->bones.resize(p_bones);

}
int RasterizerFlash::skeleton_get_bone_count(RID p_skeleton) const {

	Skeleton *skeleton = skeleton_owner.get( p_skeleton );
	ERR_FAIL_COND_V(!skeleton, -1);
	return skeleton->bones.size();
}
void RasterizerFlash::skeleton_bone_set_transform(RID p_skeleton,int p_bone, const Transform& p_transform) {

	Skeleton *skeleton = skeleton_owner.get( p_skeleton );
	ERR_FAIL_COND(!skeleton);
	ERR_FAIL_INDEX( p_bone, skeleton->bones.size() );

	skeleton->bones[p_bone] = p_transform;
}

Transform RasterizerFlash::skeleton_bone_get_transform(RID p_skeleton,int p_bone) {

	Skeleton *skeleton = skeleton_owner.get( p_skeleton );
	ERR_FAIL_COND_V(!skeleton, Transform());
	ERR_FAIL_INDEX_V( p_bone, skeleton->bones.size(), Transform() );

	// something
	return skeleton->bones[p_bone];
}


/* LIGHT API */

RID RasterizerFlash::light_create(VS::LightType p_type) {

	Light *light = memnew( Light );
	light->type=p_type;
	return light_owner.make_rid(light);
}

VS::LightType RasterizerFlash::light_get_type(RID p_light) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light,VS::LIGHT_OMNI);
	return light->type;
}

void RasterizerFlash::light_set_color(RID p_light,VS::LightColor p_type, const Color& p_color) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);
	ERR_FAIL_INDEX( p_type, 3 );
	light->colors[p_type]=p_color;
}
Color RasterizerFlash::light_get_color(RID p_light,VS::LightColor p_type) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light, Color());
	ERR_FAIL_INDEX_V( p_type, 3, Color() );
	return light->colors[p_type];
}

void RasterizerFlash::light_set_shadow(RID p_light,bool p_enabled) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);
	light->shadow_enabled=p_enabled;
}

bool RasterizerFlash::light_has_shadow(RID p_light) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light,false);
	return light->shadow_enabled;
}

void RasterizerFlash::light_set_volumetric(RID p_light,bool p_enabled) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);
	light->volumetric_enabled=p_enabled;

}
bool RasterizerFlash::light_is_volumetric(RID p_light) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light,false);
	return light->volumetric_enabled;
}

void RasterizerFlash::light_set_projector(RID p_light,RID p_texture) {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND(!light);
	light->projector=p_texture;
}
RID RasterizerFlash::light_get_projector(RID p_light) const {

	Light *light = light_owner.get(p_light);
	ERR_FAIL_COND_V(!light,RID());
	return light->projector;
}

void RasterizerFlash::light_set_var(RID p_light, VS::LightParam p_var, float p_value) {

	Light * light = light_owner.get( p_light );
	ERR_FAIL_COND(!light);
	ERR_FAIL_INDEX( p_var, VS::LIGHT_PARAM_MAX );

	light->vars[p_var]=p_value;
}
float RasterizerFlash::light_get_var(RID p_light, VS::LightParam p_var) const {

	Light * light = light_owner.get( p_light );
	ERR_FAIL_COND_V(!light,0);

	ERR_FAIL_INDEX_V( p_var, VS::LIGHT_PARAM_MAX,0 );

	return light->vars[p_var];
}

void RasterizerFlash::light_set_operator(RID p_light,VS::LightOp p_op) {

};

VS::LightOp RasterizerFlash::light_get_operator(RID p_light) const {

	return VS::LightOp();
};

void RasterizerFlash::light_omni_set_shadow_mode(RID p_light,VS::LightOmniShadowMode p_mode) {

};

VS::LightOmniShadowMode RasterizerFlash::light_omni_get_shadow_mode(RID p_light) const {

	return VS::LIGHT_OMNI_SHADOW_DEFAULT;
};


void RasterizerFlash::light_directional_set_shadow_mode(RID p_light,VS::LightDirectionalShadowMode p_mode) {

};

VS::LightDirectionalShadowMode RasterizerFlash::light_directional_get_shadow_mode(RID p_light) const {

	return VS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL;
};

void RasterizerFlash::light_directional_set_shadow_max_distance(RID p_light,float p_distance) {

};

float RasterizerFlash::light_directional_get_shadow_max_distance(RID p_light) const {

	return 0;
};

void RasterizerFlash::light_directional_set_pssm_split_weight(RID p_light,float p_weight) {

};

float RasterizerFlash::light_directional_get_pssm_split_weight(RID p_light) const {

	return 0;
};

void RasterizerFlash::light_directional_set_shadow_param(RID p_light,VS::LightDirectionalShadowParam p_param, float p_value) {

};

float RasterizerFlash::light_directional_get_shadow_param(RID p_light,VS::LightDirectionalShadowParam p_param) const {

	return 0;
};

AABB RasterizerFlash::light_get_aabb(RID p_light) const {

	Light *light = light_owner.get( p_light );
	ERR_FAIL_COND_V(!light,AABB());

	switch( light->type ) {

		case VS::LIGHT_SPOT: {

			float len=light->vars[VS::LIGHT_PARAM_RADIUS];
			float size=Math::tan(Math::deg2rad(light->vars[VS::LIGHT_PARAM_SPOT_ANGLE]))*len;
			return AABB( Vector3( -size,-size,-len ), Vector3( size*2, size*2, len ) );
		} break;
		case VS::LIGHT_OMNI: {

			float r = light->vars[VS::LIGHT_PARAM_RADIUS];
			return AABB( -Vector3(r,r,r), Vector3(r,r,r)*2 );
		} break;
		case VS::LIGHT_DIRECTIONAL: {

			return AABB();
		} break;
		default: {}
	}

	ERR_FAIL_V( AABB() );
}


RID RasterizerFlash::light_instance_create(RID p_light) {

	Light *light = light_owner.get( p_light );
	ERR_FAIL_COND_V(!light, RID());

	LightInstance *light_instance = memnew( LightInstance );

	light_instance->light=p_light;
	light_instance->base=light;
	light_instance->last_pass=0;

	return light_instance_owner.make_rid( light_instance );
}
void RasterizerFlash::light_instance_set_transform(RID p_light_instance,const Transform& p_transform) {

	LightInstance *lighti = light_instance_owner.get( p_light_instance );
	ERR_FAIL_COND(!lighti);
	lighti->transform=p_transform;

}

void RasterizerFlash::light_instance_set_active_hint(RID p_light_instance) {

	LightInstance *lighti = light_instance_owner.get( p_light_instance );
	ERR_FAIL_COND(!lighti);
	if (lighti->base->type==VS::LIGHT_DIRECTIONAL)
		lighti->shadow_pass=scene_pass;
	else
		lighti->shadow_pass=frame;

}
bool RasterizerFlash::light_instance_has_shadow(RID p_light_instance) const {

	LightInstance *lighti = light_instance_owner.get( p_light_instance );
	ERR_FAIL_COND_V(!lighti, false);

	if (!lighti->base->shadow_enabled)
		return false;

	if (lighti->base->type==VS::LIGHT_DIRECTIONAL) {
		if (lighti->shadow_pass!=scene_pass)
			return false;

	} else {
		if (lighti->shadow_pass!=frame)
			return false;
	}



	return !lighti->shadow_buffers.empty();

}

bool RasterizerFlash::light_instance_assign_shadow(RID p_light_instance) {

	return false;
}


Rasterizer::ShadowType RasterizerFlash::light_instance_get_shadow_type(RID p_light_instance) const {

	LightInstance *lighti = light_instance_owner.get( p_light_instance );
	ERR_FAIL_COND_V(!lighti,Rasterizer::SHADOW_NONE);

	switch(lighti->base->type) {

		case VS::LIGHT_DIRECTIONAL: return SHADOW_PSM; break;
		case VS::LIGHT_OMNI: return SHADOW_DUAL_PARABOLOID; break;
		case VS::LIGHT_SPOT: return SHADOW_SIMPLE; break;
	}

	return Rasterizer::SHADOW_NONE;
}

int RasterizerFlash::light_instance_get_shadow_passes(RID p_light_instance) const {

	LightInstance *lighti = light_instance_owner.get( p_light_instance );
	ERR_FAIL_COND_V(!lighti,0);
	if (lighti->base->type==VS::LIGHT_OMNI)
		return 2; // dp
	else
		return 1;
}

void RasterizerFlash::light_instance_set_custom_transform(RID p_light_instance, int p_index, const CameraMatrix& p_camera, const Transform& p_transform, float p_split_near,float p_split_far) {

	LightInstance *lighti = light_instance_owner.get( p_light_instance );
	ERR_FAIL_COND(!lighti);

	ERR_FAIL_COND(lighti->base->type!=VS::LIGHT_DIRECTIONAL);
	ERR_FAIL_INDEX(p_index,1);

	lighti->custom_projection=p_camera;
	lighti->custom_transform=p_transform;

}

/* PARTICLES INSTANCE */

RID RasterizerFlash::particles_instance_create(RID p_particles) {

	ERR_FAIL_COND_V(!particles_owner.owns(p_particles),RID());
	ParticlesInstance *particles_instance = memnew( ParticlesInstance );
	ERR_FAIL_COND_V(!particles_instance, RID() );
	particles_instance->particles=p_particles;
	return particles_instance_owner.make_rid(particles_instance);
}

void RasterizerFlash::particles_instance_set_transform(RID p_particles_instance,const Transform& p_transform) {

	ParticlesInstance *particles_instance=particles_instance_owner.get(p_particles_instance);
	ERR_FAIL_COND(!particles_instance);
	particles_instance->transform=p_transform;
}


/* RENDER API */
/* all calls (inside begin/end shadow) are always warranted to be in the following order: */


void RasterizerFlash::begin_frame() {

}

void RasterizerFlash::clear_viewport(const Color& p_color) {

};

void RasterizerFlash::set_viewport(const VS::ViewportRect& p_viewport) {

	viewport=p_viewport;
}

void RasterizerFlash::begin_scene(bool p_copy_bg, RID p_fx,VS::ScenarioDebugMode p_debug) {

};

void RasterizerFlash::begin_shadow_map( RID p_light_instance, int p_shadow_pass ) {

}

void RasterizerFlash::set_camera(const Transform& p_world,const CameraMatrix& p_projection) {

	camera_transform=p_world;
	camera_transform_inverse=camera_transform.inverse();
	camera_projection=p_projection;
	camera_plane = Plane( camera_transform.origin, camera_transform.basis.get_axis(2) );
	camera_z_near=camera_projection.get_z_near();
	camera_z_far=camera_projection.get_z_far();
	camera_projection.get_viewport_size(camera_vp_size.x,camera_vp_size.y);
}

void RasterizerFlash::add_light( RID p_light_instance ) {

#define LIGHT_FADE_TRESHOLD 0.05

	ERR_FAIL_COND( light_instance_count >= MAX_SCENE_LIGHTS );

	LightInstance *li = light_instance_owner.get(p_light_instance);
	ERR_FAIL_COND(!li);


	if (li->base->type==VS::LIGHT_DIRECTIONAL) {

		ERR_FAIL_COND( directional_light_count >= RenderList::MAX_LIGHTS);
		directional_lights[directional_light_count++]=li;
	}

	/* make light hash */

	// actually, not really a hash, but helps to sort the lights
	// and avoid recompiling redudant shader versions


	li->last_pass=scene_pass;
	li->sort_key=light_instance_count;

	light_instances[light_instance_count++]=li;

}

void RasterizerFlash::_update_shader( Shader* p_shader) const {

}


void RasterizerFlash::_add_geometry( const Geometry* p_geometry, const InstanceData *p_instance, const Geometry *p_geometry_cmp, const GeometryOwner *p_owner) {

	Material *m=NULL;
	RID m_src=p_instance->material_override.is_valid() ? p_instance->material_override : p_geometry->material;

	if (m_src)
		m=material_owner.get( m_src );

	if (!m) {
		m=material_owner.get( default_material );
	}

	ERR_FAIL_COND(!m);

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


	LightInstance *lights[RenderList::MAX_LIGHTS];

	RenderList *render_list=NULL;

	bool has_alpha = m->blend_mode!=VS::MATERIAL_BLEND_MODE_MIX || (m->shader_cache && m->shader_cache->has_alpha) || m->flags[VS::MATERIAL_FLAG_ONTOP];

	if (has_alpha) {
		render_list = &alpha_render_list;
	} else {
		render_list = &opaque_render_list;

	}

	if (false && !has_alpha) {


		//hacemos aca el per picsel liting viejis
	}


	RenderList::Element *e = render_list->add_element();

	e->geometry=p_geometry;
	e->geometry_cmp=p_geometry_cmp;
	e->material=m;
	e->instance=p_instance;
	//e->depth=camera_plane.distance_to(p_world->origin);
	e->depth=camera_transform.origin.distance_to(p_instance->transform.origin);
	e->owner=p_owner;
	e->light_type=0;
	e->additive=false;
	e->additive_ptr=&e->additive;

	if (p_instance->skeleton.is_valid()) {
		e->skeleton=skeleton_owner.get(p_instance->skeleton);
		if (!e->skeleton)
			const_cast<InstanceData*>(p_instance)->skeleton=RID();
	} else {

			e->skeleton=NULL;

	}

	e->mirror=p_instance->mirror;
	if (m->flags[VS::MATERIAL_FLAG_INVERT_FACES])
		e->mirror=!e->mirror;

	e->light_key=0;
	e->light_type=0xFF; // no lights!
	e->light_count=0;

	if (m->flags[VS::MATERIAL_FLAG_UNSHADED]) {

		e->light_key-=1;
	} else {

		//setup lights
		uint16_t light_count=0;
		uint16_t sort_key[4];
		uint8_t light_types[4];

		int dlc = MIN(directional_light_count,RenderList::MAX_LIGHTS);;
		light_count=dlc;

		for(int i=0;i<dlc;i++) {
			sort_key[i]=directional_lights[i]->sort_key;
			light_types[i]=VS::LIGHT_DIRECTIONAL;
		}


		const RID *liptr = p_instance->light_instances.ptr();
		int ilc=p_instance->light_instances.size();



		for(int i=0;i<ilc;i++) {

			if (light_count>=RenderList::MAX_LIGHTS)
				break;

			LightInstance *li=light_instance_owner.get( liptr[i] );
			if (!li || li->last_pass!=scene_pass) //lit by light not in visible scene
				continue;
			light_types[light_count]=li->base->type;
			sort_key[light_count++]=li->sort_key;


		}

		if (light_count>dlc) {
			SortArray<uint16_t> light_sort;
			light_sort.sort(&sort_key[dlc],(light_count-dlc)); //generate an equal sort key
		}

		if (fragment_lighting && !has_alpha) {

			//per vertex lighting, all lights at the same time
			for(int i=0;i<light_count;i++) {

				RenderList::Element *ec;
				if (i>0) {

					ec = render_list->add_element();
					memcpy(ec,e,sizeof(RenderList::Element));
				} else {

					ec=e;
				}

				ec->light_type=light_types[i];
				ec->light_count=1;
				ec->lights[0]=sort_key[i];
				ec->additive_ptr=&e->additive;
			}

			/*for(int i=0;i<light_count;i++)
				e->lights[i]=sort_key[i];*/



		} else {
			//per vertex lighting, all lights at the same time
			e->light_count=light_count;
			for(int i=0;i<light_count;i++)
				e->lights[i]=sort_key[i];
		}

	}

}

void RasterizerFlash::add_mesh( const RID& p_mesh, const InstanceData *p_data) {

	Mesh *mesh = mesh_owner.get(p_mesh);
	ERR_FAIL_COND(!mesh);

	int ssize = mesh->surfaces.size();

	for (int i=0;i<ssize;i++) {

		Surface *s = mesh->surfaces[i];
		_add_geometry(s,p_data,s,NULL);
	}

	mesh->last_pass=frame;

}

void RasterizerFlash::add_multimesh( const RID& p_multimesh, const InstanceData *p_data){

	MultiMesh *multimesh = multimesh_owner.get(p_multimesh);
	ERR_FAIL_COND(!multimesh);

	if (!multimesh->mesh.is_valid())
		return;
	if (multimesh->elements.empty())
		return;

	Mesh *mesh = mesh_owner.get(multimesh->mesh);
	ERR_FAIL_COND(!mesh);

	int surf_count = mesh->surfaces.size();
	if (multimesh->last_pass!=scene_pass) {

		multimesh->cache_surfaces.resize(surf_count);
		for(int i=0;i<surf_count;i++) {

			multimesh->cache_surfaces[i].material=mesh->surfaces[i]->material;
			multimesh->cache_surfaces[i].has_alpha=mesh->surfaces[i]->has_alpha;
			multimesh->cache_surfaces[i].surface=mesh->surfaces[i];
		}

		multimesh->last_pass=scene_pass;
	}

	for(int i=0;i<surf_count;i++) {

		_add_geometry(&multimesh->cache_surfaces[i],p_data,multimesh->cache_surfaces[i].surface,multimesh);
	}


}

void RasterizerFlash::add_particles( const RID& p_particle_instance, const InstanceData *p_data){

	//print_line("adding particles");
	ParticlesInstance *particles_instance = particles_instance_owner.get(p_particle_instance);
	ERR_FAIL_COND(!particles_instance);
	Particles *p=particles_owner.get( particles_instance->particles );
	ERR_FAIL_COND(!p);

	_add_geometry(p,p_data,p,particles_instance);

}


void RasterizerFlash::_set_cull(bool p_front,bool p_reverse_cull) {

}


_FORCE_INLINE_ void RasterizerFlash::_update_material_shader_params(Material *p_material) const {


	Map<StringName,Material::UniformData> old_mparams=p_material->shader_params;
	Map<StringName,Material::UniformData> &mparams=p_material->shader_params;
	mparams.clear();
	int idx=0;
	for(Map<StringName,ShaderLanguage::Uniform>::Element *E=p_material->shader_cache->uniforms.front();E;E=E->next()) {

		Material::UniformData ud;

		bool keep=true;

		if (!old_mparams.has(E->key()))
			keep=false;
		else if (old_mparams[E->key()].value.get_type()!=E->value().default_value.get_type()) {

			if (old_mparams[E->key()].value.get_type()==Variant::OBJECT) {
				if (E->value().default_value.get_type()!=Variant::_RID) //hackfor textures
					keep=false;
			} else if (!old_mparams[E->key()].value.is_num() || !E->value().default_value.get_type())
				keep=false;
		}

		if (keep) {
			ud.value=old_mparams[E->key()].value;
			print_line("KEEP: "+String(E->key()));
		} else {
			ud.value=E->value().default_value;
			print_line("NEW: "+String(E->key())+" because: hasold-"+itos(old_mparams.has(E->key())));
			if (old_mparams.has(E->key()))
				print_line(" told "+Variant::get_type_name(old_mparams[E->key()].value.get_type())+" tnew "+Variant::get_type_name(E->value().default_value.get_type()));
		}

		ud.istexture=(E->get().type==ShaderLanguage::TYPE_TEXTURE || E->get().type==ShaderLanguage::TYPE_CUBEMAP);
		ud.index=idx++;
		mparams[E->key()]=ud;
	}

	p_material->shader_version=p_material->shader_cache->version;

}

bool RasterizerFlash::_setup_material(const Geometry *p_geometry,const Material *p_material,bool p_vertexlit,bool p_no_const_light) {

	return false;

}

void RasterizerFlash::_setup_lights(const uint16_t * p_lights,int p_light_count) {

}


Error RasterizerFlash::_setup_geometry(const Geometry *p_geometry, const Material* p_material, const Skeleton *p_skeleton,const float *p_morphs) {

	return OK;
};


void RasterizerFlash::_render(const Geometry *p_geometry,const Material *p_material, const Skeleton* p_skeleton, const GeometryOwner *p_owner) {

};


void RasterizerFlash::_render_list_forward(RenderList *p_render_list,bool p_reverse_cull,bool p_fragment_light) {

};



void RasterizerFlash::end_scene() {


}
void RasterizerFlash::end_shadow_map() {
}

void RasterizerFlash::end_frame() {

}

void RasterizerFlash::flush_frame() {
}

/* CANVAS API */

void RasterizerFlash::canvas_begin() {


}
void RasterizerFlash::canvas_set_opacity(float p_opacity) {

	canvas_opacity = p_opacity;
}

void RasterizerFlash::canvas_set_blend_mode(VS::MaterialBlendMode p_mode) {

}


void RasterizerFlash::canvas_begin_rect(const Matrix32& p_transform) {

}

void RasterizerFlash::canvas_set_clip(bool p_clip, const Rect2& p_rect) {


}

void RasterizerFlash::canvas_end_rect() {

	//glPopMatrix();
}


RasterizerFlash::Texture* RasterizerFlash::_bind_canvas_texture(const RID& p_texture) {

	return NULL;
}

void RasterizerFlash::canvas_draw_line(const Point2& p_from, const Point2& p_to,const Color& p_color,float p_width) {

}

void RasterizerFlash::_draw_gui_primitive(int p_points, const Vector2 *p_vertices, const Color* p_colors, const Vector2 *p_uvs) {

}

void RasterizerFlash::_draw_textured_quad(const Rect2& p_rect, const Rect2& p_src_region, const Size2& p_tex_size,bool p_h_flip, bool p_v_flip ) {
}

void RasterizerFlash::_draw_quad(const Rect2& p_rect) {

	Vector2 coords[4]= {
		Vector2( p_rect.pos.x,p_rect.pos.y ),
		Vector2( p_rect.pos.x+p_rect.size.width,p_rect.pos.y ),
		Vector2( p_rect.pos.x+p_rect.size.width,p_rect.pos.y+p_rect.size.height ),
		Vector2( p_rect.pos.x,p_rect.pos.y+p_rect.size.height )
	};

	_draw_gui_primitive(4,coords,0,0);

}

void RasterizerFlash::canvas_draw_rect(const Rect2& p_rect, int p_flags, const Rect2& p_source,RID p_texture,const Color& p_modulate) {

	Color m = p_modulate;
	m.a*=canvas_opacity;
	_set_color_attrib(m);
	Texture *texture = _bind_canvas_texture(p_texture);

	if ( texture ) {



		if (!(p_flags&CANVAS_RECT_REGION)) {

			Rect2 region = Rect2(0,0,texture->width,texture->height);
			_draw_textured_quad(p_rect,region,region.size,p_flags&CANVAS_RECT_FLIP_H,p_flags&CANVAS_RECT_FLIP_V);

		} else {

			_draw_textured_quad(p_rect, p_source, Size2(texture->width,texture->height),p_flags&CANVAS_RECT_FLIP_H,p_flags&CANVAS_RECT_FLIP_V );

		}
	} else {

		//glDisable(GL_TEXTURE_2D);
		_draw_quad( p_rect );
		//print_line("rect: "+p_rect);

	}


}

void RasterizerFlash::canvas_draw_style_box(const Rect2& p_rect, RID p_texture,const float *p_margin, bool p_draw_center,const Color& p_modulate) {

	Color m = p_modulate;
	m.a*=canvas_opacity;
	_set_color_attrib(m);

	Texture* texture=_bind_canvas_texture(p_texture);
	ERR_FAIL_COND(!texture);
	/* CORNERS */

	_draw_textured_quad( // top left
		Rect2( p_rect.pos, Size2(p_margin[MARGIN_LEFT],p_margin[MARGIN_TOP])),
		Rect2( Point2(), Size2(p_margin[MARGIN_LEFT],p_margin[MARGIN_TOP])),
		Size2( texture->width, texture->height ) );

	_draw_textured_quad( // top right
		Rect2( Point2( p_rect.pos.x + p_rect.size.width - p_margin[MARGIN_RIGHT], p_rect.pos.y), Size2(p_margin[MARGIN_RIGHT],p_margin[MARGIN_TOP])),
		Rect2( Point2(texture->width-p_margin[MARGIN_RIGHT],0), Size2(p_margin[MARGIN_RIGHT],p_margin[MARGIN_TOP])),
		Size2( texture->width, texture->height ) );


	_draw_textured_quad( // bottom left
		Rect2( Point2(p_rect.pos.x,p_rect.pos.y + p_rect.size.height - p_margin[MARGIN_BOTTOM]), Size2(p_margin[MARGIN_LEFT],p_margin[MARGIN_BOTTOM])),
		Rect2( Point2(0,texture->height-p_margin[MARGIN_BOTTOM]), Size2(p_margin[MARGIN_LEFT],p_margin[MARGIN_BOTTOM])),
		Size2( texture->width, texture->height ) );

	_draw_textured_quad( // bottom right
		Rect2( Point2( p_rect.pos.x + p_rect.size.width - p_margin[MARGIN_RIGHT], p_rect.pos.y + p_rect.size.height - p_margin[MARGIN_BOTTOM]), Size2(p_margin[MARGIN_RIGHT],p_margin[MARGIN_BOTTOM])),
		Rect2( Point2(texture->width-p_margin[MARGIN_RIGHT],texture->height-p_margin[MARGIN_BOTTOM]), Size2(p_margin[MARGIN_RIGHT],p_margin[MARGIN_BOTTOM])),
		Size2( texture->width, texture->height ) );

	Rect2 rect_center( p_rect.pos+Point2( p_margin[MARGIN_LEFT], p_margin[MARGIN_TOP]), Size2( p_rect.size.width - p_margin[MARGIN_LEFT] - p_margin[MARGIN_RIGHT], p_rect.size.height - p_margin[MARGIN_TOP] - p_margin[MARGIN_BOTTOM] ));

	Rect2 src_center( Point2( p_margin[MARGIN_LEFT], p_margin[MARGIN_TOP]), Size2( texture->width - p_margin[MARGIN_LEFT] - p_margin[MARGIN_RIGHT], texture->height - p_margin[MARGIN_TOP] - p_margin[MARGIN_BOTTOM] ));


	_draw_textured_quad( // top
		Rect2( Point2(rect_center.pos.x,p_rect.pos.y),Size2(rect_center.size.width,p_margin[MARGIN_TOP])),
		Rect2( Point2(p_margin[MARGIN_LEFT],0), Size2(src_center.size.width,p_margin[MARGIN_TOP])),
		Size2( texture->width, texture->height ) );

	_draw_textured_quad( // bottom
		Rect2( Point2(rect_center.pos.x,rect_center.pos.y+rect_center.size.height),Size2(rect_center.size.width,p_margin[MARGIN_BOTTOM])),
		Rect2( Point2(p_margin[MARGIN_LEFT],src_center.pos.y+src_center.size.height), Size2(src_center.size.width,p_margin[MARGIN_BOTTOM])),
		Size2( texture->width, texture->height ) );

	_draw_textured_quad( // left
		Rect2( Point2(p_rect.pos.x,rect_center.pos.y),Size2(p_margin[MARGIN_LEFT],rect_center.size.height)),
		Rect2( Point2(0,p_margin[MARGIN_TOP]), Size2(p_margin[MARGIN_LEFT],src_center.size.height)),
		Size2( texture->width, texture->height ) );

	_draw_textured_quad( // right
		Rect2( Point2(rect_center.pos.x+rect_center.size.width,rect_center.pos.y),Size2(p_margin[MARGIN_RIGHT],rect_center.size.height)),
		Rect2( Point2(src_center.pos.x+src_center.size.width,p_margin[MARGIN_TOP]), Size2(p_margin[MARGIN_RIGHT],src_center.size.height)),
		Size2( texture->width, texture->height ) );

	if (p_draw_center) {

		_draw_textured_quad(
			rect_center,
			src_center,
			Size2( texture->width, texture->height ));
	}

}

void RasterizerFlash::canvas_draw_primitive(const Vector<Point2>& p_points, const Vector<Color>& p_colors,const Vector<Point2>& p_uvs, RID p_texture,float p_width) {

	ERR_FAIL_COND(p_points.size()<1);
	_set_color_attrib(Color(1,1,1,canvas_opacity));
	_bind_canvas_texture(p_texture);
	_draw_gui_primitive(p_points.size(),p_points.ptr(),p_colors.ptr(),p_uvs.ptr());

}

void RasterizerFlash::canvas_draw_polygon(int p_vertex_count, const int* p_indices, const Vector2* p_vertices, const Vector2* p_uvs, const Color* p_colors,const RID& p_texture,bool p_singlecolor) {


};


void RasterizerFlash::canvas_set_transform(const Matrix32& p_transform) {

}

/* ENVIRONMENT */

RID RasterizerFlash::environment_create() {

	Environment * env = memnew( Environment );
	return environment_owner.make_rid(env);
}

void RasterizerFlash::environment_set_background(RID p_env,VS::EnvironmentBG p_bg) {

	ERR_FAIL_INDEX(p_bg,VS::ENV_BG_MAX);
	Environment * env = environment_owner.get(p_env);
	ERR_FAIL_COND(!env);
	env->bg_mode=p_bg;
}

VS::EnvironmentBG RasterizerFlash::environment_get_background(RID p_env) const{

	const Environment * env = environment_owner.get(p_env);
	ERR_FAIL_COND_V(!env,VS::ENV_BG_MAX);
	return env->bg_mode;
}

void RasterizerFlash::environment_set_background_param(RID p_env,VS::EnvironmentBGParam p_param, const Variant& p_value){

	ERR_FAIL_INDEX(p_param,VS::ENV_BG_PARAM_MAX);
	Environment * env = environment_owner.get(p_env);
	ERR_FAIL_COND(!env);
	env->bg_param[p_param]=p_value;

}
Variant RasterizerFlash::environment_get_background_param(RID p_env,VS::EnvironmentBGParam p_param) const{

	ERR_FAIL_INDEX_V(p_param,VS::ENV_BG_PARAM_MAX,Variant());
	const Environment * env = environment_owner.get(p_env);
	ERR_FAIL_COND_V(!env,Variant());
	return env->bg_param[p_param];

}

void RasterizerFlash::environment_set_enable_fx(RID p_env,VS::EnvironmentFx p_effect,bool p_enabled){

	ERR_FAIL_INDEX(p_effect,VS::ENV_FX_MAX);
	Environment * env = environment_owner.get(p_env);
	ERR_FAIL_COND(!env);
	env->fx_enabled[p_effect]=p_enabled;
}
bool RasterizerFlash::environment_is_fx_enabled(RID p_env,VS::EnvironmentFx p_effect) const{

	ERR_FAIL_INDEX_V(p_effect,VS::ENV_FX_MAX,false);
	const Environment * env = environment_owner.get(p_env);
	ERR_FAIL_COND_V(!env,false);
	return env->fx_enabled[p_effect];

}

void RasterizerFlash::environment_fx_set_param(RID p_env,VS::EnvironmentFxParam p_param,const Variant& p_value){

	ERR_FAIL_INDEX(p_param,VS::ENV_FX_PARAM_MAX);
	Environment * env = environment_owner.get(p_env);
	ERR_FAIL_COND(!env);
	env->fx_param[p_param]=p_value;
}
Variant RasterizerFlash::environment_fx_get_param(RID p_env,VS::EnvironmentFxParam p_param) const{

	ERR_FAIL_INDEX_V(p_param,VS::ENV_FX_PARAM_MAX,Variant());
	const Environment * env = environment_owner.get(p_env);
	ERR_FAIL_COND_V(!env,Variant());
	return env->fx_param[p_param];

}


/* FX */

RID RasterizerFlash::fx_create() {

	FX *fx = memnew( FX );
	ERR_FAIL_COND_V(!fx,RID());
	return fx_owner.make_rid(fx);

}
void RasterizerFlash::fx_get_effects(RID p_fx,List<String> *p_effects) const {

	FX *fx = fx_owner.get(p_fx);
	ERR_FAIL_COND(!fx);

	p_effects->clear();
	p_effects->push_back("bgcolor");
	p_effects->push_back("skybox");
	p_effects->push_back("antialias");
	//p_effects->push_back("hdr");
	p_effects->push_back("glow");	// glow has a bloom parameter, too
	p_effects->push_back("ssao");
	p_effects->push_back("fog");
	p_effects->push_back("dof_blur");
	p_effects->push_back("toon");
	p_effects->push_back("edge");

}
void RasterizerFlash::fx_set_active(RID p_fx,const String& p_effect, bool p_active) {

	FX *fx = fx_owner.get(p_fx);
	ERR_FAIL_COND(!fx);

	if (p_effect=="bgcolor")
		fx->bgcolor_active=p_active;
	else if (p_effect=="skybox")
		fx->skybox_active=p_active;
	else if (p_effect=="antialias")
		fx->antialias_active=p_active;
	else if (p_effect=="glow")
		fx->glow_active=p_active;
	else if (p_effect=="ssao")
		fx->ssao_active=p_active;
	else if (p_effect=="fog")
		fx->fog_active=p_active;
//	else if (p_effect=="dof_blur")
//		fx->dof_blur_active=p_active;
	else if (p_effect=="toon")
		fx->toon_active=p_active;
	else if (p_effect=="edge")
		fx->edge_active=p_active;
}
bool RasterizerFlash::fx_is_active(RID p_fx,const String& p_effect) const {

	FX *fx = fx_owner.get(p_fx);
	ERR_FAIL_COND_V(!fx,false);

	if (p_effect=="bgcolor")
		return fx->bgcolor_active;
	else if (p_effect=="skybox")
		return fx->skybox_active;
	else if (p_effect=="antialias")
		return fx->antialias_active;
	else if (p_effect=="glow")
		return fx->glow_active;
	else if (p_effect=="ssao")
		return fx->ssao_active;
	else if (p_effect=="fog")
		return fx->fog_active;
	//else if (p_effect=="dof_blur")
	//	return fx->dof_blur_active;
	else if (p_effect=="toon")
		return fx->toon_active;
	else if (p_effect=="edge")
		return fx->edge_active;

	return false;
}
void RasterizerFlash::fx_get_effect_params(RID p_fx,const String& p_effect,List<PropertyInfo> *p_params) const {

	FX *fx = fx_owner.get(p_fx);
	ERR_FAIL_COND(!fx);


	if (p_effect=="bgcolor") {

		p_params->push_back( PropertyInfo( Variant::COLOR, "color" ) );
	} else if (p_effect=="skybox") {
		p_params->push_back( PropertyInfo( Variant::_RID, "cubemap" ) );
	} else if (p_effect=="antialias") {

		p_params->push_back( PropertyInfo( Variant::REAL, "tolerance", PROPERTY_HINT_RANGE,"1,128,1" ) );

	} else if (p_effect=="glow") {

		p_params->push_back( PropertyInfo( Variant::INT, "passes", PROPERTY_HINT_RANGE,"1,4,1" ) );
		p_params->push_back( PropertyInfo( Variant::REAL, "attenuation", PROPERTY_HINT_RANGE,"0.01,8.0,0.01" ) );
		p_params->push_back( PropertyInfo( Variant::REAL, "bloom", PROPERTY_HINT_RANGE,"-1.0,1.0,0.01" ) );

	} else if (p_effect=="ssao") {

		p_params->push_back( PropertyInfo( Variant::REAL, "radius", PROPERTY_HINT_RANGE,"0.0,16.0,0.01" ) );
		p_params->push_back( PropertyInfo( Variant::REAL, "max_distance", PROPERTY_HINT_RANGE,"0.0,256.0,0.01" ) );
		p_params->push_back( PropertyInfo( Variant::REAL, "range_max", PROPERTY_HINT_RANGE,"0.0,1.0,0.01" ) );
		p_params->push_back( PropertyInfo( Variant::REAL, "range_min", PROPERTY_HINT_RANGE,"0.0,1.0,0.01" ) );
		p_params->push_back( PropertyInfo( Variant::REAL, "attenuation", PROPERTY_HINT_RANGE,"0.0,8.0,0.01" ) );

	} else if (p_effect=="fog") {

		p_params->push_back( PropertyInfo( Variant::REAL, "begin", PROPERTY_HINT_RANGE,"0.0,8192,0.01" ) );
		p_params->push_back( PropertyInfo( Variant::REAL, "end", PROPERTY_HINT_RANGE,"0.0,8192,0.01" ) );
		p_params->push_back( PropertyInfo( Variant::REAL, "attenuation", PROPERTY_HINT_RANGE,"0.0,8.0,0.01" ) );
		p_params->push_back( PropertyInfo( Variant::COLOR, "color_begin" ) );
		p_params->push_back( PropertyInfo( Variant::COLOR, "color_end" ) );
		p_params->push_back( PropertyInfo( Variant::BOOL, "fog_bg" ) );

//	} else if (p_effect=="dof_blur") {
//		return fx->dof_blur_active;
	} else if (p_effect=="toon") {
		p_params->push_back( PropertyInfo( Variant::REAL, "treshold", PROPERTY_HINT_RANGE,"0.0,1.0,0.01" ) );
		p_params->push_back( PropertyInfo( Variant::REAL, "soft", PROPERTY_HINT_RANGE,"0.001,1.0,0.001" ) );
	} else if (p_effect=="edge") {

	}
}
Variant RasterizerFlash::fx_get_effect_param(RID p_fx,const String& p_effect,const String& p_param) const {

	FX *fx = fx_owner.get(p_fx);
	ERR_FAIL_COND_V(!fx,Variant());

	if (p_effect=="bgcolor") {

		if (p_param=="color")
			return fx->bgcolor;
	} else if (p_effect=="skybox") {
		if (p_param=="cubemap")
			return fx->skybox_cubemap;
	} else if (p_effect=="antialias") {

		if (p_param=="tolerance")
			return fx->antialias_tolerance;

	} else if (p_effect=="glow") {

		if (p_param=="passes")
			return fx->glow_passes;
		if (p_param=="attenuation")
			return fx->glow_attenuation;
		if (p_param=="bloom")
			return fx->glow_bloom;

	} else if (p_effect=="ssao") {

		if (p_param=="attenuation")
			return fx->ssao_attenuation;
		if (p_param=="max_distance")
			return fx->ssao_max_distance;
		if (p_param=="range_max")
			return fx->ssao_range_max;
		if (p_param=="range_min")
			return fx->ssao_range_min;
		if (p_param=="radius")
			return fx->ssao_radius;

	} else if (p_effect=="fog") {

		if (p_param=="begin")
			return fx->fog_near;
		if (p_param=="end")
			return fx->fog_far;
		if (p_param=="attenuation")
			return fx->fog_attenuation;
		if (p_param=="color_begin")
			return fx->fog_color_near;
		if (p_param=="color_end")
			return fx->fog_color_far;
		if (p_param=="fog_bg")
			return fx->fog_bg;
//	} else if (p_effect=="dof_blur") {
//		return fx->dof_blur_active;
	} else if (p_effect=="toon") {
		if (p_param=="treshold")
			return fx->toon_treshold;
		if (p_param=="soft")
			return fx->toon_soft;

	} else if (p_effect=="edge") {

	}
	return Variant();
}
void RasterizerFlash::fx_set_effect_param(RID p_fx,const String& p_effect, const String& p_param, const Variant& p_value) {

	FX *fx = fx_owner.get(p_fx);
	ERR_FAIL_COND(!fx);

	if (p_effect=="bgcolor") {

		if (p_param=="color")
			fx->bgcolor=p_value;
	} else if (p_effect=="skybox") {
		if (p_param=="cubemap")
			fx->skybox_cubemap=p_value;

	} else if (p_effect=="antialias") {

		if (p_param=="tolerance")
			fx->antialias_tolerance=p_value;

	} else if (p_effect=="glow") {

		if (p_param=="passes")
			fx->glow_passes=p_value;
		if (p_param=="attenuation")
			fx->glow_attenuation=p_value;
		if (p_param=="bloom")
			fx->glow_bloom=p_value;

	} else if (p_effect=="ssao") {

		if (p_param=="attenuation")
			fx->ssao_attenuation=p_value;
		if (p_param=="radius")
			fx->ssao_radius=p_value;
		if (p_param=="max_distance")
			fx->ssao_max_distance=p_value;
		if (p_param=="range_max")
			fx->ssao_range_max=p_value;
		if (p_param=="range_min")
			fx->ssao_range_min=p_value;

	} else if (p_effect=="fog") {

		if (p_param=="begin")
			fx->fog_near=p_value;
		if (p_param=="end")
			fx->fog_far=p_value;
		if (p_param=="attenuation")
			fx->fog_attenuation=p_value;
		if (p_param=="color_begin")
			fx->fog_color_near=p_value;
		if (p_param=="color_end")
			fx->fog_color_far=p_value;
		if (p_param=="fog_bg")
			fx->fog_bg=p_value;
//	} else if (p_effect=="dof_blur") {
//		fx->dof_blur_active=p_value;
	} else if (p_effect=="toon") {

		if (p_param=="treshold")
			fx->toon_treshold=p_value;
		if (p_param=="soft")
			fx->toon_soft=p_value;

	} else if (p_effect=="edge") {

	}

}

/*MISC*/

bool RasterizerFlash::is_texture(const RID& p_rid) const {

	return texture_owner.owns(p_rid);
}
bool RasterizerFlash::is_material(const RID& p_rid) const {

	return material_owner.owns(p_rid);
}
bool RasterizerFlash::is_mesh(const RID& p_rid) const {

	return mesh_owner.owns(p_rid);
}
bool RasterizerFlash::is_multimesh(const RID& p_rid) const {

	return multimesh_owner.owns(p_rid);
}
bool RasterizerFlash::is_particles(const RID &p_beam) const {

	return particles_owner.owns(p_beam);
}

bool RasterizerFlash::is_light(const RID& p_rid) const {

	return light_owner.owns(p_rid);
}
bool RasterizerFlash::is_light_instance(const RID& p_rid) const {

	return light_instance_owner.owns(p_rid);
}
bool RasterizerFlash::is_particles_instance(const RID& p_rid) const {

	return particles_instance_owner.owns(p_rid);
}
bool RasterizerFlash::is_skeleton(const RID& p_rid) const {

	return skeleton_owner.owns(p_rid);
}
bool RasterizerFlash::is_environment(const RID& p_rid) const {

	return environment_owner.owns(p_rid);
}
bool RasterizerFlash::is_fx(const RID& p_rid) const {

	return fx_owner.owns(p_rid);
}
bool RasterizerFlash::is_shader(const RID& p_rid) const {

	return false;
}

void RasterizerFlash::free(const RID& p_rid) {

	if (texture_owner.owns(p_rid)) {

		// delete the texture
		Texture *texture = texture_owner.get(p_rid);

		_rinfo.texture_mem-=texture->total_data_size;
		texture_owner.free(p_rid);
		memdelete(texture);

	} else if (shader_owner.owns(p_rid)) {


	} else if (material_owner.owns(p_rid)) {

		Material *material = material_owner.get( p_rid );
		ERR_FAIL_COND(!material);

		_free_fixed_material(p_rid); //just in case
		material_owner.free(p_rid);
		memdelete(material);

	} else if (mesh_owner.owns(p_rid)) {


	} else if (multimesh_owner.owns(p_rid)) {

	       MultiMesh *multimesh = multimesh_owner.get(p_rid);
	       ERR_FAIL_COND(!multimesh);

	       multimesh_owner.free(p_rid);
	       memdelete(multimesh);

	} else if (particles_owner.owns(p_rid)) {

		Particles *particles = particles_owner.get(p_rid);
		ERR_FAIL_COND(!particles);

		particles_owner.free(p_rid);
		memdelete(particles);
	} else if (particles_instance_owner.owns(p_rid)) {

		ParticlesInstance *particles_isntance = particles_instance_owner.get(p_rid);
		ERR_FAIL_COND(!particles_isntance);

		particles_instance_owner.free(p_rid);
		memdelete(particles_isntance);

	} else if (skeleton_owner.owns(p_rid)) {

		Skeleton *skeleton = skeleton_owner.get( p_rid );
		ERR_FAIL_COND(!skeleton)

		skeleton_owner.free(p_rid);
		memdelete(skeleton);

	} else if (light_owner.owns(p_rid)) {

		Light *light = light_owner.get( p_rid );
		ERR_FAIL_COND(!light)

		light_owner.free(p_rid);
		memdelete(light);

	} else if (light_instance_owner.owns(p_rid)) {

		LightInstance *light_instance = light_instance_owner.get( p_rid );
		ERR_FAIL_COND(!light_instance);
		light_instance->clear_shadow_buffers();
		light_instance_owner.free(p_rid);
		memdelete( light_instance );

	} else if (fx_owner.owns(p_rid)) {

		FX *fx = fx_owner.get( p_rid );
		ERR_FAIL_COND(!fx);

		fx_owner.free(p_rid);
		memdelete( fx );

	} else if (environment_owner.owns(p_rid)) {

		Environment *env = environment_owner.get( p_rid );
		ERR_FAIL_COND(!env);

		environment_owner.free(p_rid);
		memdelete( env );

	};
}



void RasterizerFlash::ShadowBuffer::init(int p_size) {

}

void RasterizerFlash::_init_shadow_buffers() {

	int near_shadow_size=GLOBAL_DEF("rasterizer/near_shadow_size",512);
	int far_shadow_size=GLOBAL_DEF("rasterizer/far_shadow_size",64);

	near_shadow_buffers.resize( GLOBAL_DEF("rasterizer/near_shadow_count",4) );
	far_shadow_buffers.resize( GLOBAL_DEF("rasterizer/far_shadow_count",16) );

	shadow_near_far_split_size_ratio = GLOBAL_DEF("rasterizer/shadow_near_far_split_size_ratio",0.3);

	for (int i=0;i<near_shadow_buffers.size();i++) {

		near_shadow_buffers[i].init(near_shadow_size );
	}

	for (int i=0;i<far_shadow_buffers.size();i++) {

		far_shadow_buffers[i].init(far_shadow_size);
	}

}



void RasterizerFlash::_update_framebuffer() {

	return;

}

void RasterizerFlash::init() {

}

void RasterizerFlash::finish() {

	memdelete_arr(skinned_buffer);
}

int RasterizerFlash::get_render_info(VS::RenderInfo p_info) {

	switch(p_info) {

		case VS::INFO_OBJECTS_IN_FRAME: {

			return _rinfo.object_count;
		} break;
		case VS::INFO_VERTICES_IN_FRAME: {

			return _rinfo.vertex_count;
		} break;
		case VS::INFO_MATERIAL_CHANGES_IN_FRAME: {

			return _rinfo.mat_change_count;
		} break;
		case VS::INFO_SHADER_CHANGES_IN_FRAME: {

			return _rinfo.shader_change_count;
		} break;
		case VS::INFO_USAGE_VIDEO_MEM_TOTAL: {

			return 0;
		} break;
		case VS::INFO_VIDEO_MEM_USED: {

			return get_render_info(VS::INFO_TEXTURE_MEM_USED)+get_render_info(VS::INFO_VERTEX_MEM_USED);
		} break;
		case VS::INFO_TEXTURE_MEM_USED: {

			_rinfo.texture_mem;
		} break;
		case VS::INFO_VERTEX_MEM_USED: {

			return 0;
		} break;
	}

	return false;
}

RasterizerFlash::RasterizerFlash(bool p_compress_arrays,bool p_keep_ram_copy,bool p_default_fragment_lighting) {

	keep_copies=p_keep_ram_copy;
	pack_arrays=p_compress_arrays;
	fragment_lighting=GLOBAL_DEF("rasterizer/use_fragment_lighting",p_default_fragment_lighting);

	frame = 0;
};

RasterizerFlash::~RasterizerFlash() {

};


