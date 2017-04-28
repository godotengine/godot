/*************************************************************************/
/*  baked_light_baker.cpp                                                */
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
#include "baked_light_baker.h"

#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "io/marshalls.h"
#include <stdlib.h>
#include <cmath>

#if 0
void baked_light_baker_add_64f(double *dst,double value);
void baked_light_baker_add_64i(int64_t *dst,int64_t value);

//-separar en 2 testuras?
//*mejorar performance y threads
//*modos lineales
//*saturacion

_FORCE_INLINE_ static uint64_t get_uv_normal_bit(const Vector3& p_vector) {

	int lat = Math::fast_ftoi(Math::floor(Math::acos(p_vector.dot(Vector3(0,1,0)))*6.0/Math_PI+0.5));

	if (lat==0) {
		return 60;
	} else if (lat==6) {
		return 61;
	}

	int lon = Math::fast_ftoi(Math::floor( (Math_PI+Math::atan2(p_vector.x,p_vector.z))*12.0/(Math_PI*2.0) + 0.5))%12;

	return lon+(lat-1)*12;
}



_FORCE_INLINE_ static Vector3 get_bit_normal(int p_bit) {

	if (p_bit==61) {
		return Vector3(0,1,0);
	} else if (p_bit==62){
		return Vector3(0,-1,0);
	}

	float latang = ((p_bit / 12)+1)*Math_PI/6.0;

	Vector2 latv(Math::sin(latang),Math::cos(latang));

	float lonang = ((p_bit%12)*Math_PI*2.0/12.0)-Math_PI;

	Vector2 lonv(Math::sin(lonang),Math::cos(lonang));

	return Vector3(lonv.x*latv.x,latv.y,lonv.y*latv.x).normalized();

}


BakedLightBaker::MeshTexture* BakedLightBaker::_get_mat_tex(const Ref<Texture>& p_tex) {

	if (!tex_map.has(p_tex)) {

		Ref<ImageTexture> imgtex=p_tex;
		if (imgtex.is_null())
			return NULL;
		Image image=imgtex->get_data();
		if (image.empty())
			return NULL;

		if (image.get_format()!=Image::FORMAT_RGBA8) {
			if (image.get_format()>Image::FORMAT_INDEXED_ALPHA) {
				Error err = image.decompress();
				if (err)
					return NULL;
			}

			if (image.get_format()!=Image::FORMAT_RGBA8)
				image.convert(Image::FORMAT_RGBA8);
		}

		if (imgtex->get_flags()&Texture::FLAG_CONVERT_TO_LINEAR) {
			Image copy = image;
			copy.srgb_to_linear();
			image=copy;
		}

		PoolVector<uint8_t> dvt=image.get_data();
		PoolVector<uint8_t>::Read r=dvt.read();
		MeshTexture mt;
		mt.tex_w=image.get_width();
		mt.tex_h=image.get_height();
		int len = image.get_width()*image.get_height()*4;
		mt.tex.resize(len);
		copymem(mt.tex.ptr(),r.ptr(),len);

		textures.push_back(mt);
		tex_map[p_tex]=&textures.back()->get();
	}

	return tex_map[p_tex];
}


void BakedLightBaker::_add_mesh(const Ref<Mesh>& p_mesh,const Ref<Material>& p_mat_override,const Transform& p_xform,int p_baked_texture) {


	for(int i=0;i<p_mesh->get_surface_count();i++) {

		if (p_mesh->surface_get_primitive_type(i)!=Mesh::PRIMITIVE_TRIANGLES)
			continue;
		Ref<Material> mat = p_mat_override.is_valid()?p_mat_override:p_mesh->surface_get_material(i);

		MeshMaterial *matptr=NULL;
		int baked_tex=p_baked_texture;

		if (mat.is_valid()) {

			if (!mat_map.has(mat)) {

				MeshMaterial mm;

				Ref<SpatialMaterial> fm = mat;
				if (fm.is_valid()) {
					//fixed route
					mm.diffuse.color=fm->get_parameter(SpatialMaterial::PARAM_DIFFUSE);
					if (linear_color)
						mm.diffuse.color=mm.diffuse.color.to_linear();
					mm.diffuse.tex=_get_mat_tex(fm->get_texture(SpatialMaterial::PARAM_DIFFUSE));
					mm.specular.color=fm->get_parameter(SpatialMaterial::PARAM_SPECULAR);
					if (linear_color)
						mm.specular.color=mm.specular.color.to_linear();

					mm.specular.tex=_get_mat_tex(fm->get_texture(SpatialMaterial::PARAM_SPECULAR));
				} else {

					mm.diffuse.color=Color(1,1,1,1);
					mm.diffuse.tex=NULL;
					mm.specular.color=Color(0,0,0,1);
					mm.specular.tex=NULL;
				}

				materials.push_back(mm);
				mat_map[mat]=&materials.back()->get();

			}

			matptr=mat_map[mat];

		}


		int facecount=0;


		if (p_mesh->surface_get_format(i)&Mesh::ARRAY_FORMAT_INDEX) {

			facecount=p_mesh->surface_get_array_index_len(i);
		} else {

			facecount=p_mesh->surface_get_array_len(i);
		}

		ERR_CONTINUE((facecount==0 || (facecount%3)!=0));

		facecount/=3;

		int tbase=triangles.size();
		triangles.resize(facecount+tbase);


		Array a = p_mesh->surface_get_arrays(i);

		PoolVector<Vector3> vertices = a[Mesh::ARRAY_VERTEX];
		PoolVector<Vector3>::Read vr=vertices.read();
		PoolVector<Vector2> uv;
		PoolVector<Vector2>::Read uvr;
		PoolVector<Vector2> uv2;
		PoolVector<Vector2>::Read uv2r;
		PoolVector<Vector3> normal;
		PoolVector<Vector3>::Read normalr;
		bool read_uv=false;
		bool read_normal=false;

		if (p_mesh->surface_get_format(i)&Mesh::ARRAY_FORMAT_TEX_UV) {

			uv=a[Mesh::ARRAY_TEX_UV];
			uvr=uv.read();
			read_uv=true;

			if (mat.is_valid() && mat->get_flag(Material::FLAG_LIGHTMAP_ON_UV2) && p_mesh->surface_get_format(i)&Mesh::ARRAY_FORMAT_TEX_UV2) {

				uv2=a[Mesh::ARRAY_TEX_UV2];
				uv2r=uv2.read();

			} else {
				uv2r=uv.read();
				if (baked_light->get_transfer_lightmaps_only_to_uv2()) {
					baked_tex=-1;
				}
			}
		}

		if (p_mesh->surface_get_format(i)&Mesh::ARRAY_FORMAT_NORMAL) {

			normal=a[Mesh::ARRAY_NORMAL];
			normalr=normal.read();
			read_normal=true;
		}

		Matrix3 normal_xform = p_xform.basis.inverse().transposed();


		if (p_mesh->surface_get_format(i)&Mesh::ARRAY_FORMAT_INDEX) {

			PoolVector<int> indices = a[Mesh::ARRAY_INDEX];
			PoolVector<int>::Read ir = indices.read();

			for(int i=0;i<facecount;i++) {
				Triangle &t=triangles[tbase+i];
				t.vertices[0]=p_xform.xform(vr[ ir[i*3+0] ]);
				t.vertices[1]=p_xform.xform(vr[ ir[i*3+1] ]);
				t.vertices[2]=p_xform.xform(vr[ ir[i*3+2] ]);
				t.material=matptr;
				t.baked_texture=baked_tex;
				if (read_uv) {

					t.uvs[0]=uvr[ ir[i*3+0] ];
					t.uvs[1]=uvr[ ir[i*3+1] ];
					t.uvs[2]=uvr[ ir[i*3+2] ];

					t.bake_uvs[0]=uv2r[ ir[i*3+0] ];
					t.bake_uvs[1]=uv2r[ ir[i*3+1] ];
					t.bake_uvs[2]=uv2r[ ir[i*3+2] ];
				}
				if (read_normal) {

					t.normals[0]=normal_xform.xform(normalr[ ir[i*3+0] ]).normalized();
					t.normals[1]=normal_xform.xform(normalr[ ir[i*3+1] ]).normalized();
					t.normals[2]=normal_xform.xform(normalr[ ir[i*3+2] ]).normalized();
				}
			}

		} else {

			for(int i=0;i<facecount;i++) {
				Triangle &t=triangles[tbase+i];
				t.vertices[0]=p_xform.xform(vr[ i*3+0 ]);
				t.vertices[1]=p_xform.xform(vr[ i*3+1 ]);
				t.vertices[2]=p_xform.xform(vr[ i*3+2 ]);
				t.material=matptr;
				t.baked_texture=baked_tex;
				if (read_uv) {

					t.uvs[0]=uvr[ i*3+0 ];
					t.uvs[1]=uvr[ i*3+1 ];
					t.uvs[2]=uvr[ i*3+2 ];

					t.bake_uvs[0]=uv2r[ i*3+0 ];
					t.bake_uvs[1]=uv2r[ i*3+1 ];
					t.bake_uvs[2]=uv2r[ i*3+2 ];

				}
				if (read_normal) {

					t.normals[0]=normal_xform.xform(normalr[ i*3+0 ]).normalized();
					t.normals[1]=normal_xform.xform(normalr[ i*3+1 ]).normalized();
					t.normals[2]=normal_xform.xform(normalr[ i*3+2 ]).normalized();
				}
			}
		}
	}

}


void BakedLightBaker::_parse_geometry(Node* p_node) {

	if (p_node->cast_to<MeshInstance>()) {

		MeshInstance *meshi=p_node->cast_to<MeshInstance>();
		Ref<Mesh> mesh=meshi->get_mesh();
		if (mesh.is_valid()) {
			_add_mesh(mesh,meshi->get_material_override(),base_inv * meshi->get_global_transform(),meshi->get_baked_light_texture_id());
		}
	} else if (p_node->cast_to<Light>()) {

		Light *dl=p_node->cast_to<Light>();

		if (dl->get_bake_mode()!=Light::BAKE_MODE_DISABLED) {


			LightData dirl;
			dirl.type=VS::LightType(dl->get_light_type());
			dirl.diffuse=dl->get_color(DirectionalLight::COLOR_DIFFUSE);
			dirl.specular=dl->get_color(DirectionalLight::COLOR_SPECULAR);
			if (linear_color)
				dirl.diffuse=dirl.diffuse.to_linear();
			if (linear_color)
				dirl.specular=dirl.specular.to_linear();

			dirl.energy=dl->get_parameter(DirectionalLight::PARAM_ENERGY);
			dirl.pos=dl->get_global_transform().origin;
			dirl.up=dl->get_global_transform().basis.get_axis(1).normalized();
			dirl.left=dl->get_global_transform().basis.get_axis(0).normalized();
			dirl.dir=-dl->get_global_transform().basis.get_axis(2).normalized();
			dirl.spot_angle=dl->get_parameter(DirectionalLight::PARAM_SPOT_ANGLE);
			dirl.spot_attenuation=dl->get_parameter(DirectionalLight::PARAM_SPOT_ATTENUATION);
			dirl.attenuation=dl->get_parameter(DirectionalLight::PARAM_ATTENUATION);
			dirl.darkening=dl->get_parameter(DirectionalLight::PARAM_SHADOW_DARKENING);
			dirl.radius=dl->get_parameter(DirectionalLight::PARAM_RADIUS);
			dirl.bake_direct=dl->get_bake_mode()==Light::BAKE_MODE_FULL;
			dirl.rays_thrown=0;
			dirl.bake_shadow=dl->get_bake_mode()==Light::BAKE_MODE_INDIRECT_AND_SHADOWS;
			lights.push_back(dirl);
		}

	} else if (p_node->cast_to<Spatial>()){

		Spatial *sp = p_node->cast_to<Spatial>();

		Array arr = p_node->call("_get_baked_light_meshes");
		for(int i=0;i<arr.size();i+=2) {

			Transform xform=arr[i];
			Ref<Mesh> mesh=arr[i+1];
			_add_mesh(mesh,Ref<Material>(),base_inv * (sp->get_global_transform() * xform));
		}
	}

	for(int i=0;i<p_node->get_child_count();i++) {

		_parse_geometry(p_node->get_child(i));
	}
}


void BakedLightBaker::_fix_lights() {


	total_light_area=0;
	for(int i=0;i<lights.size();i++) {

		LightData &dl=lights[i];

		switch(dl.type) {
			case VS::LIGHT_DIRECTIONAL: {

				float up_max=-1e10;
				float dir_max=-1e10;
				float left_max=-1e10;
				float up_min=1e10;
				float dir_min=1e10;
				float left_min=1e10;

				for(int j=0;j<triangles.size();j++) {

					for(int k=0;k<3;k++) {

						Vector3 v = triangles[j].vertices[k];

						float up_d = dl.up.dot(v);
						float dir_d = dl.dir.dot(v);
						float left_d = dl.left.dot(v);

						if (up_d>up_max)
							up_max=up_d;
						if (up_d<up_min)
							up_min=up_d;

						if (left_d>left_max)
							left_max=left_d;
						if (left_d<left_min)
							left_min=left_d;

						if (dir_d>dir_max)
							dir_max=dir_d;
						if (dir_d<dir_min)
							dir_min=dir_d;

					}
				}

				//make a center point, then the upvector and leftvector
				dl.pos = dl.left*( left_max+left_min )*0.5 + dl.up*( up_max+up_min )*0.5 + dl.dir*(dir_min-(dir_max-dir_min));
				dl.left*=(left_max-left_min)*0.5;
				dl.up*=(up_max-up_min)*0.5;
				dl.length = (dir_max - dir_min)*10; //arbitrary number to keep it in scale
				dl.area=dl.left.length()*2*dl.up.length()*2;
				dl.constant=1.0/dl.area;
			} break;
			case VS::LIGHT_OMNI:
			case VS::LIGHT_SPOT: {

				dl.attenuation_table.resize(ATTENUATION_CURVE_LEN);
				for(int j=0;j<ATTENUATION_CURVE_LEN;j++) {
					dl.attenuation_table[j]=1.0-Math::pow(j/float(ATTENUATION_CURVE_LEN),dl.attenuation);
					float falloff=j*dl.radius/float(ATTENUATION_CURVE_LEN);
					if (falloff==0)
						falloff=0.000001;
					float intensity=4*Math_PI*(falloff*falloff);
					//dl.attenuation_table[j]*=falloff*falloff;
					dl.attenuation_table[j]*=1.0/(3.0/intensity);

				}
				if (dl.type==VS::LIGHT_OMNI) {

					dl.area=4.0*Math_PI*pow(dl.radius,2.0f);
					dl.constant=1.0/3.5;
				} else {


					float r = Math::tan(Math::deg2rad(dl.spot_angle))*dl.radius;
					float c = 1.0-(Math::deg2rad(dl.spot_angle)*0.5+0.5);
					dl.constant=1.0/3.5;
					dl.constant*=1.0/c;

					dl.area=Math_PI*r*r*c;
				}

			} break;


		}

		total_light_area+=dl.area;
	}
}

BakedLightBaker::BVH* BakedLightBaker::_parse_bvh(BVH** p_children, int p_size, int p_depth, int &max_depth) {

	if (p_depth>max_depth) {
		max_depth=p_depth;
	}

	if (p_size==1) {

		return p_children[0];
	} else if (p_size==0) {

		return NULL;
	}


	AABB aabb;
	aabb=p_children[0]->aabb;
	for(int i=1;i<p_size;i++) {

		aabb.merge_with(p_children[i]->aabb);
	}

	int li=aabb.get_longest_axis_index();

	switch(li) {

		case Vector3::AXIS_X: {
			SortArray<BVH*,BVHCmpX> sort_x;
			sort_x.nth_element(0,p_size,p_size/2,p_children);
			//sort_x.sort(&p_bb[p_from],p_size);
		} break;
		case Vector3::AXIS_Y: {
			SortArray<BVH*,BVHCmpY> sort_y;
			sort_y.nth_element(0,p_size,p_size/2,p_children);
			//sort_y.sort(&p_bb[p_from],p_size);
		} break;
		case Vector3::AXIS_Z: {
			SortArray<BVH*,BVHCmpZ> sort_z;
			sort_z.nth_element(0,p_size,p_size/2,p_children);
			//sort_z.sort(&p_bb[p_from],p_size);

		} break;
	}


	BVH* left = _parse_bvh(p_children,p_size/2,p_depth+1,max_depth);
	BVH* right = _parse_bvh(&p_children[p_size/2],p_size-p_size/2,p_depth+1,max_depth);

	BVH *_new = memnew(BVH);
	_new->aabb=aabb;
	_new->center=aabb.pos+aabb.size*0.5;
	_new->children[0]=left;
	_new->children[1]=right;
	_new->leaf=NULL;

	return _new;
}

void BakedLightBaker::_make_bvh() {

	Vector<BVH*> bases;
	bases.resize(triangles.size());
	int max_depth=0;
	for(int i=0;i<triangles.size();i++) {
		bases[i]=memnew( BVH );
		bases[i]->leaf=&triangles[i];
		bases[i]->aabb.pos=triangles[i].vertices[0];
		bases[i]->aabb.expand_to(triangles[i].vertices[1]);
		bases[i]->aabb.expand_to(triangles[i].vertices[2]);
		triangles[i].aabb=bases[i]->aabb;
		bases[i]->center=bases[i]->aabb.pos+bases[i]->aabb.size*0.5;
	}

	bvh=_parse_bvh(bases.ptr(),bases.size(),1,max_depth);

	ray_stack = memnew_arr(uint32_t,max_depth);
	bvh_stack = memnew_arr(BVH*,max_depth);

	bvh_depth = max_depth;
}

void BakedLightBaker::_octree_insert(int p_octant,Triangle* p_triangle, int p_depth) {




	uint32_t *stack=octant_stack;
	uint32_t *ptr_stack=octantptr_stack;
	Octant *octants=octant_pool.ptr();

	stack[0]=0;
	ptr_stack[0]=0;

	int stack_pos=0;


	while(true) {

		Octant *octant=&octants[ptr_stack[stack_pos]];
		if (stack[stack_pos]<8) {

			int i = stack[stack_pos];
			stack[stack_pos]++;



			//fit_aabb=fit_aabb.grow(bvh->aabb.size.x*0.0001);

			int child_idx =octant->children[i];
			bool encloses;
			if (!child_idx) {

				AABB aabb=octant->aabb;
				aabb.size*=0.5;
				if (i&1)
					aabb.pos.x+=aabb.size.x;
				if (i&2)
					aabb.pos.y+=aabb.size.y;
				if (i&4)
					aabb.pos.z+=aabb.size.z;

				aabb.grow_by(cell_size*octree_extra_margin);
				if (!aabb.intersects(p_triangle->aabb))
					continue;
				encloses=aabb.grow(cell_size*-octree_extra_margin*2.0).encloses(p_triangle->aabb);
				if (!encloses && !Face3(p_triangle->vertices[0],p_triangle->vertices[1],p_triangle->vertices[2]).intersects_aabb2(aabb))
					continue;
			} else {

				Octant *child=&octants[child_idx];
				AABB aabb=child->aabb;
				aabb.grow_by(cell_size*octree_extra_margin);
				if (!aabb.intersects(p_triangle->aabb))
					continue;
				encloses=aabb.grow(cell_size*-octree_extra_margin*2.0).encloses(p_triangle->aabb);
				if (!encloses && !Face3(p_triangle->vertices[0],p_triangle->vertices[1],p_triangle->vertices[2]).intersects_aabb2(aabb))
					continue;

			}

			if (encloses)
				stack[stack_pos]=8; // quick and dirty opt

			if (!child_idx) {


				if (octant_pool_size==octant_pool.size()) {
					octant_pool.resize(octant_pool_size+OCTANT_POOL_CHUNK);
					octants=octant_pool.ptr();
					octant=&octants[ptr_stack[stack_pos]];
				}
				child_idx=octant_pool_size++;
				octant->children[i]=child_idx;
				Octant *child=&octants[child_idx];

				child->aabb=octant->aabb;
				child->texture_x=0;
				child->texture_y=0;

				child->aabb.size*=0.5;
				if (i&1)
					child->aabb.pos.x+=child->aabb.size.x;
				if (i&2)
					child->aabb.pos.y+=child->aabb.size.y;
				if (i&4)
					child->aabb.pos.z+=child->aabb.size.z;


				child->full_accum[0]=0;
				child->full_accum[1]=0;
				child->full_accum[2]=0;
				child->sampler_ofs=0;



				if (stack_pos==octree_depth-1) {
					child->leaf=true;
					child->offset[0]=child->aabb.pos.x+child->aabb.size.x*0.5;
					child->offset[1]=child->aabb.pos.y+child->aabb.size.y*0.5;
					child->offset[2]=child->aabb.pos.z+child->aabb.size.z*0.5;
					child->next_leaf=leaf_list;


					for(int ci=0;ci<8;ci++) {
						child->normal_accum[ci][0]=0;
						child->normal_accum[ci][1]=0;
						child->normal_accum[ci][2]=0;

					}

					child->bake_neighbour=0;
					child->first_neighbour=true;
					leaf_list=child_idx;
					cell_count++;

					for(int ci=0;ci<8;ci++) {
						child->light_accum[ci][0]=0;
						child->light_accum[ci][1]=0;
						child->light_accum[ci][2]=0;
					}

					child->parent=ptr_stack[stack_pos];

				} else {

					child->leaf=false;
					for(int j=0;j<8;j++) {
						child->children[j]=0;
					}
				}
			}

			if (!octants[child_idx].leaf) {
				stack_pos++;
				stack[stack_pos]=0;
				ptr_stack[stack_pos]=child_idx;
			} else {

				Octant *child=&octants[child_idx];

				Vector3 n = Plane(p_triangle->vertices[0],p_triangle->vertices[1],p_triangle->vertices[2]).normal;


				for(int ci=0;ci<8;ci++) {

					Vector3 pos = child->aabb.pos;

					if (ci&1)
						pos.x+=child->aabb.size.x;
					if (ci&2)
						pos.y+=child->aabb.size.y;
					if (ci&4)
						pos.z+=child->aabb.size.z;


					pos.x=floor((pos.x+cell_size*0.5)/cell_size);
					pos.y=floor((pos.y+cell_size*0.5)/cell_size);
					pos.z=floor((pos.z+cell_size*0.5)/cell_size);

					{
						Map<Vector3,Vector3>::Element *E=endpoint_normal.find(pos);
						if (!E) {
							endpoint_normal[pos]=n;
						} else {
							E->get()+=n;
						}
					}

					{

						uint64_t bit = get_uv_normal_bit(n);

						Map<Vector3,uint64_t>::Element *E=endpoint_normal_bits.find(pos);
						if (!E) {
							endpoint_normal_bits[pos]=(1<<bit);
						} else {
							E->get()|=(1<<bit);
						}

					}

				}

			}


		} else {
			stack_pos--;
			if (stack_pos<0)
				break;
		}
	}


}


void BakedLightBaker::_make_octree() {


	AABB base = bvh->aabb;
	float lal=base.get_longest_axis_size();
	//must be square because we want square blocks
	base.size.x=lal;
	base.size.y=lal;
	base.size.z=lal;
	base.grow_by(lal*0.001); //for precision
	octree_aabb=base;

	cell_size=base.size.x;
	for(int i=0;i<octree_depth;i++)
		cell_size/=2.0;
	octant_stack = memnew_arr(uint32_t,octree_depth*2 );
	octantptr_stack = memnew_arr(uint32_t,octree_depth*2 );

	octant_pool.resize(OCTANT_POOL_CHUNK);
	octant_pool_size=1;
	Octant *root=octant_pool.ptr();
	root->leaf=false;
	root->aabb=octree_aabb;
	root->parent=-1;
	for(int i=0;i<8;i++)
		root->children[i]=0;

	EditorProgress ep("bake_octree",vformat(TTR("Parsing %d Triangles:"), triangles.size()),triangles.size());

	for(int i=0;i<triangles.size();i++) {

		_octree_insert(0,&triangles[i],octree_depth-1);
		if ((i%1000)==0) {

			ep.step(TTR("Triangle #")+itos(i),i);
		}
	}

	{
		uint32_t oct_idx=leaf_list;
		Octant *octants=octant_pool.ptr();
		while(oct_idx) {

			BakedLightBaker::Octant *oct = &octants[oct_idx];
			for(int ci=0;ci<8;ci++) {


				Vector3 pos = oct->aabb.pos;

				if (ci&1)
					pos.x+=oct->aabb.size.x;
				if (ci&2)
					pos.y+=oct->aabb.size.y;
				if (ci&4)
					pos.z+=oct->aabb.size.z;


				pos.x=floor((pos.x+cell_size*0.5)/cell_size);
				pos.y=floor((pos.y+cell_size*0.5)/cell_size);
				pos.z=floor((pos.z+cell_size*0.5)/cell_size);

				{
					Map<Vector3,Vector3>::Element *E=endpoint_normal.find(pos);
					if (!E) {
						//?
						print_line("lolwut?");
					} else {
						Vector3 n = E->get().normalized();
						oct->normal_accum[ci][0]=n.x;
						oct->normal_accum[ci][1]=n.y;
						oct->normal_accum[ci][2]=n.z;

					}

				}

				{

					Map<Vector3,uint64_t>::Element *E=endpoint_normal_bits.find(pos);
					if (!E) {
						//?
						print_line("lolwut?");
					} else {

						float max_aper=0;
						for(uint64_t i=0;i<62;i++) {

							if (!(E->get()&(1<<i)))
								continue;
							Vector3 ang_i = get_bit_normal(i);

							for(uint64_t j=0;j<62;j++) {

								if (i==j)
									continue;
								if (!(E->get()&(1<<j)))
									continue;
								Vector3 ang_j = get_bit_normal(j);
								float ang = Math::acos(ang_i.dot(ang_j));
								if (ang>max_aper)
									max_aper=ang;
							}
						}
						if (max_aper>0.75*Math_PI) {
							//angle too wide prevent problems and forget
							oct->normal_accum[ci][0]=0;
							oct->normal_accum[ci][1]=0;
							oct->normal_accum[ci][2]=0;
						}
					}
				}


			}

			oct_idx=oct->next_leaf;
		}
	}


}





void BakedLightBaker::_plot_light(ThreadStack& thread_stack,const Vector3& p_plot_pos, const AABB& p_plot_aabb, const Color& p_light,const Color& p_tint_light,bool p_only_full, const Plane& p_plane) {

	//stackless version

	uint32_t *stack=thread_stack.octant_stack;
	uint32_t *ptr_stack=thread_stack.octantptr_stack;
	Octant *octants=octant_pool.ptr();

	stack[0]=0;
	ptr_stack[0]=0;

	int stack_pos=0;


	while(true) {

		Octant &octant=octants[ptr_stack[stack_pos]];

		if (stack[stack_pos]==0) {


			Vector3 pos = octant.aabb.pos + octant.aabb.size*0.5;
			float md = 1<<(octree_depth - stack_pos );
			float r=cell_size*plot_size*md;
			float div = 1.0/(md*md*md);
			//div=1.0;


			float d = p_plot_pos.distance_to(pos);

			if ((p_plane.distance_to(pos)>-cell_size*1.75*md) && d<=r) {


				float intensity = 1.0 - (d/r)*(d/r); //not gauss but..

				baked_light_baker_add_64f(&octant.full_accum[0],p_tint_light.r*intensity*div);
				baked_light_baker_add_64f(&octant.full_accum[1],p_tint_light.g*intensity*div);
				baked_light_baker_add_64f(&octant.full_accum[2],p_tint_light.b*intensity*div);
			}
		}

		if (octant.leaf) {



			//if (p_plane.normal.dot(octant.aabb.get_support(p_plane.normal)) < p_plane.d-CMP_EPSILON) { //octants behind are no go


			if (!p_only_full) {
				float r=cell_size*plot_size;
				for(int i=0;i<8;i++) {
					Vector3 pos=octant.aabb.pos;
					if (i&1)
						pos.x+=octant.aabb.size.x;
					if (i&2)
						pos.y+=octant.aabb.size.y;
					if (i&4)
						pos.z+=octant.aabb.size.z;



					float d = p_plot_pos.distance_to(pos);

					if ((p_plane.distance_to(pos)>-cell_size*1.75) && d<=r) {


						float intensity = 1.0 - (d/r)*(d/r); //not gauss but..
						if (edge_damp>0) {
							Vector3 normal = Vector3(octant.normal_accum[i][0],octant.normal_accum[i][1],octant.normal_accum[i][2]);
							if (normal.x>0 || normal.y>0 || normal.z>0) {

								float damp = Math::abs(p_plane.normal.dot(normal));
								intensity*=pow(damp,edge_damp);

							}
						}

						//intensity*=1.0-Math::abs(p_plane.distance_to(pos))/(plot_size*cell_size);
						//intensity = Math::cos(d*Math_PI*0.5/r);

						baked_light_baker_add_64f(&octant.light_accum[i][0],p_light.r*intensity);
						baked_light_baker_add_64f(&octant.light_accum[i][1],p_light.g*intensity);
						baked_light_baker_add_64f(&octant.light_accum[i][2],p_light.b*intensity);


					}
				}
			}

			stack_pos--;
		} else if (stack[stack_pos]<8) {

			int i = stack[stack_pos];
			stack[stack_pos]++;

			if (!octant.children[i]) {
				continue;
			}

			Octant &child=octants[octant.children[i]];

			if (!child.aabb.intersects(p_plot_aabb))
				continue;

			if (child.aabb.encloses(p_plot_aabb)) {
				stack[stack_pos]=8; //don't test the rest
			}

			stack_pos++;
			stack[stack_pos]=0;
			ptr_stack[stack_pos]=octant.children[i];
		} else {
			stack_pos--;
			if (stack_pos<0)
				break;
		}
	}


}


float BakedLightBaker::_throw_ray(ThreadStack& thread_stack,bool p_bake_direct,const Vector3& p_begin, const Vector3& p_end,float p_rest,const Color& p_light,float *p_att_curve,float p_att_pos,int p_att_curve_len,int p_bounces,bool p_first_bounce,bool p_only_dist) {


	uint32_t* stack = thread_stack.ray_stack;
	BVH **bstack = thread_stack.bvh_stack;

	enum {
		TEST_AABB_BIT=0,
		VISIT_LEFT_BIT=1,
		VISIT_RIGHT_BIT=2,
		VISIT_DONE_BIT=3,


	};

	Vector3 n = (p_end-p_begin);
	float len=n.length();
	if (len==0)
		return 0;
	n/=len;



	real_t d=1e10;
	bool inters=false;
	Vector3 r_normal;
	Vector3 r_point;
	Vector3 end=p_end;

	Triangle *triangle=NULL;

	/*
	for(int i=0;i<max_depth;i++)
		stack[i]=0;
	*/

	int level=0;
	//AABB ray_aabb;
	//ray_aabb.pos=p_begin;
	//ray_aabb.expand_to(p_end);


	bstack[0]=bvh;
	stack[0]=TEST_AABB_BIT;


	while(true) {

		uint32_t mode = stack[level];
		const BVH &b = *bstack[level];
		bool done=false;

		switch(mode) {
			case TEST_AABB_BIT: {

				if (b.leaf) {


					Face3 f3(b.leaf->vertices[0],b.leaf->vertices[1],b.leaf->vertices[2]);


					Vector3 res;

					if (f3.intersects_segment(p_begin,end,&res)) {


						float nd = n.dot(res);
						if (nd<d) {

							d=nd;
							r_point=res;
							end=res;
							len=(p_begin-end).length();
							r_normal=f3.get_plane().get_normal();
							triangle=b.leaf;
							inters=true;
						}

					}

					stack[level]=VISIT_DONE_BIT;
				} else {


					bool valid = b.aabb.smits_intersect_ray(p_begin,n,0,len);
					//bool valid = b.aabb.intersects_segment(p_begin,p_end);
					//bool valid = b.aabb.intersects(ray_aabb);

					if (!valid) {

						stack[level]=VISIT_DONE_BIT;

					} else {

						stack[level]=VISIT_LEFT_BIT;
					}
				}

			} continue;
			case VISIT_LEFT_BIT: {

				stack[level]=VISIT_RIGHT_BIT;
				bstack[level+1]=b.children[0];
				stack[level+1]=TEST_AABB_BIT;
				level++;

			} continue;
			case VISIT_RIGHT_BIT: {

				stack[level]=VISIT_DONE_BIT;
				bstack[level+1]=b.children[1];
				stack[level+1]=TEST_AABB_BIT;
				level++;
			} continue;
			case VISIT_DONE_BIT: {

				if (level==0) {
					done=true;
					break;
				} else
					level--;

			} continue;
		}


		if (done)
			break;
	}



	if (inters) {

		if (p_only_dist) {

			return p_begin.distance_to(r_point);
		}


		//should check if there is normals first
		Vector2 uv;
		if (true) {

			triangle->get_uv_and_normal(r_point,uv,r_normal);

		} else {

		}

		if (n.dot(r_normal)>0)
			return -1;

		if (n.dot(r_normal)>0)
			r_normal=-r_normal;


		//ok...
		Color diffuse_at_point(0.8,0.8,0.8);
		Color specular_at_point(0.0,0.0,0.0);


		float dist = p_begin.distance_to(r_point);

		AABB aabb;
		aabb.pos=r_point;
		aabb.pos-=Vector3(1,1,1)*cell_size*plot_size;
		aabb.size=Vector3(2,2,2)*cell_size*plot_size;

		Color res_light=p_light;
		float att=1.0;
		float dp=(1.0-normal_damp)*n.dot(-r_normal)+normal_damp;

		if (p_att_curve) {

			p_att_pos+=dist;
			int cpos = Math::fast_ftoi((p_att_pos/p_att_curve_len)*ATTENUATION_CURVE_LEN);
			cpos=CLAMP(cpos,0,ATTENUATION_CURVE_LEN-1);
			att=p_att_curve[cpos];
		}


		res_light.r*=dp;
		res_light.g*=dp;
		res_light.b*=dp;

		//light is plotted before multiplication with diffuse, this way
		//the multiplication can happen with more detail in the shader



		if (triangle->material) {

			//triangle->get_uv(r_point);

			diffuse_at_point=triangle->material->diffuse.get_color(uv);
			specular_at_point=triangle->material->specular.get_color(uv);
		}


		diffuse_at_point.r=res_light.r*diffuse_at_point.r;
		diffuse_at_point.g=res_light.g*diffuse_at_point.g;
		diffuse_at_point.b=res_light.b*diffuse_at_point.b;

		if (p_bounces>0) {


			p_rest-=dist;
			if (p_rest<CMP_EPSILON)
				return 0;

			if (r_normal==-n)
				return 0; //todo change a little

			r_point+=r_normal*0.01;




			specular_at_point.r=res_light.r*specular_at_point.r;
			specular_at_point.g=res_light.g*specular_at_point.g;
			specular_at_point.b=res_light.b*specular_at_point.b;



			if (use_diffuse && (diffuse_at_point.r>CMP_EPSILON || diffuse_at_point.g>CMP_EPSILON || diffuse_at_point.b>CMP_EPSILON)) {
				//diffuse bounce

				Vector3 c1=r_normal.cross(n).normalized();
				Vector3 c2=r_normal.cross(c1).normalized();
				double r1 = double(rand())/RAND_MAX;
				double r2 = double(rand())/RAND_MAX;
				double r3 = double(rand())/RAND_MAX;
#if 0
				Vector3 next = - ((c1*(r1-0.5)) + (c2*(r2-0.5)) + (r_normal*(r3-0.5))).normalized()*0.5 + r_normal*0.5;

				if (next==Vector3())
					next=r_normal;
				Vector3 rn=next.normalized();

#else
				Vector3 rn = ((c1*(r1-0.5)) + (c2*(r2-0.5)) + (r_normal*r3*0.5)).normalized();
#endif


				_throw_ray(thread_stack,p_bake_direct,r_point,r_point+rn*p_rest,p_rest,diffuse_at_point,p_att_curve,p_att_pos,p_att_curve_len,p_bounces-1);
			}

			if (use_specular && (specular_at_point.r>CMP_EPSILON || specular_at_point.g>CMP_EPSILON || specular_at_point.b>CMP_EPSILON)) {
				//specular bounce

				//Vector3 c1=r_normal.cross(n).normalized();
				//Vector3 c2=r_normal.cross(c1).normalized();

				Vector3 rn = n - r_normal *r_normal.dot(n) * 2.0;

				_throw_ray(thread_stack,p_bake_direct,r_point,r_point+rn*p_rest,p_rest,specular_at_point,p_att_curve,p_att_pos,p_att_curve_len,p_bounces-1);
			}
		}

		//specular later
		//_plot_light_point(r_point,octree,octree_aabb,p_light);


		Color plot_light=res_light.linear_interpolate(diffuse_at_point,tint);
		plot_light.r*=att;
		plot_light.g*=att;
		plot_light.b*=att;
		Color tint_light=diffuse_at_point;
		tint_light.r*=att;
		tint_light.g*=att;
		tint_light.b*=att;

		bool skip=false;

		if (!p_first_bounce || p_bake_direct) {


			float r = plot_size * cell_size*2;
			if (dist<r) {
				//avoid accumulaiton of light on corners
				//plot_light=plot_light.linear_interpolate(Color(0,0,0,0),1.0-sd/plot_size*plot_size);
				skip=true;

			} else {


				Vector3 c1=r_normal.cross(n).normalized();
				Vector3 c2=r_normal.cross(c1).normalized();
				double r1 = double(rand())/RAND_MAX;
				double r2 = double(rand())/RAND_MAX;
				double r3 = double(rand())/RAND_MAX;
				Vector3 rn = ((c1*(r1-0.5)) + (c2*(r2-0.5)) + (r_normal*r3*0.25)).normalized();
				float d =_throw_ray(thread_stack,p_bake_direct,r_point,r_point+rn*p_rest,p_rest,diffuse_at_point,p_att_curve,p_att_pos,p_att_curve_len,p_bounces-1,false,true);
				r = plot_size*cell_size*ao_radius;
				if (d>0 && d<r) {
					//avoid accumulaiton of light on corners
					//plot_light=plot_light.linear_interpolate(Color(0,0,0,0),1.0-sd/plot_size*plot_size);
					skip=true;

				} else {
					//plot_light=Color(0,0,0,0);
				}
			}
		}


		Plane plane(r_point,r_normal);
		if (!skip)
			_plot_light(thread_stack,r_point,aabb,plot_light,tint_light,!(!p_first_bounce || p_bake_direct),plane);


		return dist;
	}

	return -1;

}




void BakedLightBaker::_make_octree_texture() {


	BakedLightBaker::Octant *octants=octant_pool.ptr();

	//find neighbours first, to have a better idea of what amount of space is needed
	{

		Vector<OctantHash> octant_hashing;
		octant_hashing.resize(octant_pool_size);
		Vector<uint32_t> hash_table;
		int hash_table_size=Math::larger_prime(16384);
		hash_table.resize(hash_table_size);
		uint32_t*hashptr = hash_table.ptr();
		OctantHash*octhashptr = octant_hashing.ptr();

		for(int i=0;i<hash_table_size;i++)
			hashptr[i]=0;


		//step 1 add to hash table

		uint32_t oct_idx=leaf_list;


		while(oct_idx) {

			BakedLightBaker::Octant *oct = &octants[oct_idx];
			uint64_t base=0;
			Vector3 pos = oct->aabb.pos - octree_aabb.pos; //make sure is always positive
			base=int((pos.x+cell_size*0.5)/cell_size);
			base<<=16;
			base|=int((pos.y+cell_size*0.5)/cell_size);
			base<<=16;
			base|=int((pos.z+cell_size*0.5)/cell_size);

			uint32_t hash = HashMapHasherDefault::hash(base);
			uint32_t idx = hash % hash_table_size;
			octhashptr[oct_idx].next=hashptr[idx];
			octhashptr[oct_idx].hash=hash;
			octhashptr[oct_idx].value=base;
			hashptr[idx]=oct_idx;

			oct_idx=oct->next_leaf;

		}

		//step 2 find neighbours
		oct_idx=leaf_list;
		int neighbours=0;


		while(oct_idx) {

			BakedLightBaker::Octant *oct = &octants[oct_idx];
			Vector3 pos = oct->aabb.pos - octree_aabb.pos; //make sure is always positive
			pos.x+=cell_size;
			uint64_t base=0;
			base=int((pos.x+cell_size*0.5)/cell_size);
			base<<=16;
			base|=int((pos.y+cell_size*0.5)/cell_size);
			base<<=16;
			base|=int((pos.z+cell_size*0.5)/cell_size);

			uint32_t hash = HashMapHasherDefault::hash(base);
			uint32_t idx = hash % hash_table_size;

			uint32_t bucket = hashptr[idx];

			while(bucket) {

				if (octhashptr[bucket].value==base) {

					oct->bake_neighbour=bucket;
					octants[bucket].first_neighbour=false;
					neighbours++;
					break;
				}

				bucket = octhashptr[bucket].next;
			}

			oct_idx=oct->next_leaf;

		}

		print_line("octant with neighbour: "+itos(neighbours));

	}


	//ok let's try to just create a texture

	int otex_w=256;

	while (true) {



		uint32_t oct_idx=leaf_list;

		int row=0;


		print_line("begin at row "+itos(row));
		int longest_line_reused=0;
		int col=0;
		int processed=0;

		//reset
		while(oct_idx) {

			BakedLightBaker::Octant *oct = &octants[oct_idx];
			oct->texture_x=0;
			oct->texture_y=0;
			oct_idx=oct->next_leaf;

		}

		oct_idx=leaf_list;
		//assign
		while(oct_idx) {

			BakedLightBaker::Octant *oct = &octants[oct_idx];
			if (oct->first_neighbour && oct->texture_x==0 && oct->texture_y==0) {
				//was not processed
				uint32_t current_idx=oct_idx;
				int reused=0;

				while(current_idx) {
					BakedLightBaker::Octant *o = &octants[current_idx];
					if (col+1 >= otex_w) {
						col=0;
						row+=4;
					}
					o->texture_x=col;
					o->texture_y=row;
					processed++;

					if (o->bake_neighbour) {
						reused++;
					}
					col+=o->bake_neighbour ? 1 : 2; //reuse neighbour
					current_idx=o->bake_neighbour;
				}

				if (reused>longest_line_reused) {
					longest_line_reused=reused;
				}
			}
			oct_idx=oct->next_leaf;
		}

		row+=4;

		if (otex_w < row) {

			otex_w*=2;
		} else {

			baked_light_texture_w=otex_w;
			baked_light_texture_h=nearest_power_of_2(row);
			print_line("w: "+itos(otex_w));
			print_line("h: "+itos(row));
			break;
		}


	}


	{

		otex_w=(1<<lattice_size)*(1<<lattice_size)*2; //make sure lattice fits horizontally
		Vector3 lattice_cell_size=octree_aabb.size;
		for(int i=0;i<lattice_size;i++) {

			lattice_cell_size*=0.5;
		}



		while(true) {

			//let's plot the leafs first, given the octree is not so obvious which size it will have
			int row=4+4*(1<<lattice_size);
			int col=0;

			col=0;
			row+=4;
			print_line("end at row "+itos(row));

			//put octree, no need for recursion, just loop backwards.
			int regular_octants=0;
			for(int i=octant_pool_size-1;i>=0;i--) {

				BakedLightBaker::Octant *oct = &octants[i];
				if (oct->leaf) //ignore leaf
					continue;
				if (oct->aabb.size.x>lattice_cell_size.x*1.1) { //bigger than latice, skip
					oct->texture_x=0;
					oct->texture_y=0;
				} else if (oct->aabb.size.x>lattice_cell_size.x*0.8) {
					//this is the initial lattice
					Vector3 pos = oct->aabb.pos - octree_aabb.pos; //make sure is always positive
					int x = int((pos.x+lattice_cell_size.x*0.5)/lattice_cell_size.x);
					int y = int((pos.y+lattice_cell_size.y*0.5)/lattice_cell_size.y);
					int z = int((pos.z+lattice_cell_size.z*0.5)/lattice_cell_size.z);
					//bug net
					ERR_FAIL_INDEX(x,(1<<lattice_size));
					ERR_FAIL_INDEX(y,(1<<lattice_size));
					ERR_FAIL_INDEX(z,(1<<lattice_size));

					/*int ofs = z*(1<<lattice_size)*(1<<lattice_size)+y*(1<<lattice_size)+x;
					ofs*=4;
					oct->texture_x=ofs%otex_w;
					oct->texture_y=(ofs/otex_w)*4+4;
					*/

					oct->texture_x=(x+(1<<lattice_size)*z)*2;
					oct->texture_y=4+y*4;
					//print_line("pos: "+itos(x)+","+itos(y)+","+itos(z)+" -  ofs"+itos(oct->texture_x)+","+itos(oct->texture_y));


				} else {
					//an everyday regular octant

					if (col+2 > otex_w) {
						col=0;
						row+=4;
					}

					oct->texture_x=col;
					oct->texture_y=row;
					col+=2;
					regular_octants++;


				}
			}
			print_line("octants end at row "+itos(row)+" totalling"+itos(regular_octants));

			//ok evaluation.

			if (otex_w<=2048 && row>2048) { //too big upwards, try bigger texture
				otex_w*=2;
				continue;
			} else {
				baked_octree_texture_w=otex_w;
				baked_octree_texture_h=row+4;
				break;
			}

		}


	}


	baked_octree_texture_h=nearest_power_of_2(baked_octree_texture_h);
	print_line("RESULT! "+itos(baked_octree_texture_w)+","+itos(baked_octree_texture_h));

}








double BakedLightBaker::get_normalization(int p_light_idx) const {

	double nrg=0;

	const LightData &dl=lights[p_light_idx];
	double cell_area = cell_size*cell_size;
	//nrg+= /*dl.energy */ (dl.rays_thrown * cell_area / dl.area);
	nrg=dl.rays_thrown * cell_area;
	nrg*=(Math_PI*plot_size*plot_size)*0.5; // damping of radial linear gradient kernel
	nrg*=dl.constant;
	//nrg*=5;


	return nrg;
}



double BakedLightBaker::get_modifier(int p_light_idx) const {

	double nrg=0;

	const LightData &dl=lights[p_light_idx];
	double cell_area = cell_size*cell_size;
	//nrg+= /*dl.energy */ (dl.rays_thrown * cell_area / dl.area);
	nrg=cell_area;
	nrg*=(Math_PI*plot_size*plot_size)*0.5; // damping of radial linear gradient kernel
	nrg*=dl.constant;
	//nrg*=5;


	return nrg;
}

void BakedLightBaker::throw_rays(ThreadStack& thread_stack,int p_amount) {



	for(int i=0;i<lights.size();i++) {

		LightData &dl=lights[i];


		int amount = p_amount * total_light_area / dl.area;
		double mod = 1.0/double(get_modifier(i));
		mod*=p_amount/float(amount);

		switch(dl.type) {

			case VS::LIGHT_DIRECTIONAL: {


				for(int j=0;j<amount;j++) {
					Vector3 from = dl.pos;
					double r1 = double(rand())/RAND_MAX;
					double r2 = double(rand())/RAND_MAX;
					from+=dl.up*(r1*2.0-1.0);
					from+=dl.left*(r2*2.0-1.0);
					Vector3 to = from+dl.dir*dl.length;
					Color col=dl.diffuse;
					float m = mod*dl.energy;
					col.r*=m;
					col.g*=m;
					col.b*=m;

					dl.rays_thrown++;
					baked_light_baker_add_64i(&total_rays,1);

					_throw_ray(thread_stack,dl.bake_direct,from,to,dl.length,col,NULL,0,0,max_bounces,true);
				}
			} break;
			case VS::LIGHT_OMNI: {


				for(int j=0;j<amount;j++) {
					Vector3 from = dl.pos;

					double r1 = double(rand())/RAND_MAX;
					double r2 = double(rand())/RAND_MAX;
					double r3 = double(rand())/RAND_MAX;

#if 0
					//crap is not uniform..
					Vector3 dir = Vector3(r1*2.0-1.0,r2*2.0-1.0,r3*2.0-1.0).normalized();

#else

					double phi = r1*Math_PI*2.0;
					double costheta = r2*2.0-1.0;
					double u = r3;

					double theta = acos( costheta );
					double r = 1.0 * pow( u,1/3.0 );

					Vector3 dir(
						r * sin( theta) * cos( phi ),
						r * sin( theta) * sin( phi ),
						r * cos( theta )
					);
					dir.normalize();

#endif
					Vector3 to = dl.pos+dir*dl.radius;
					Color col=dl.diffuse;
					float m = mod*dl.energy;
					col.r*=m;
					col.g*=m;
					col.b*=m;

					dl.rays_thrown++;
					baked_light_baker_add_64i(&total_rays,1);
					_throw_ray(thread_stack,dl.bake_direct,from,to,dl.radius,col,dl.attenuation_table.ptr(),0,dl.radius,max_bounces,true);
					//_throw_ray(i,from,to,dl.radius,col,NULL,0,dl.radius,max_bounces,true);
				}

			} break;
			case VS::LIGHT_SPOT: {

				for(int j=0;j<amount;j++) {
					Vector3 from = dl.pos;

					double r1 = double(rand())/RAND_MAX;
					//double r2 = double(rand())/RAND_MAX;
					double r3 = double(rand())/RAND_MAX;

					float d=Math::tan(Math::deg2rad(dl.spot_angle));

					float x = sin(r1*Math_PI*2.0)*d;
					float y = cos(r1*Math_PI*2.0)*d;

					Vector3 dir = r3*(dl.dir + dl.up*y + dl.left*x) + (1.0-r3)*dl.dir;
					dir.normalize();


					Vector3 to = dl.pos+dir*dl.radius;
					Color col=dl.diffuse;
					float m = mod*dl.energy;
					col.r*=m;
					col.g*=m;
					col.b*=m;

					dl.rays_thrown++;
					baked_light_baker_add_64i(&total_rays,1);
					_throw_ray(thread_stack,dl.bake_direct,from,to,dl.radius,col,dl.attenuation_table.ptr(),0,dl.radius,max_bounces,true);
					//_throw_ray(i,from,to,dl.radius,col,NULL,0,dl.radius,max_bounces,true);
				}

			} break;

		}
	}
}













void BakedLightBaker::bake(const Ref<BakedLight> &p_light, Node* p_node) {

	if (baking)
		return;
	cell_count=0;

	base_inv=p_node->cast_to<Spatial>()->get_global_transform().affine_inverse();
	EditorProgress ep("bake",TTR("Light Baker Setup:"),5);
	baked_light=p_light;
	lattice_size=baked_light->get_initial_lattice_subdiv();
	octree_depth=baked_light->get_cell_subdivision();
	plot_size=baked_light->get_plot_size();
	max_bounces=baked_light->get_bounces();
	use_diffuse=baked_light->get_bake_flag(BakedLight::BAKE_DIFFUSE);
	use_specular=baked_light->get_bake_flag(BakedLight::BAKE_SPECULAR);
	use_translucency=baked_light->get_bake_flag(BakedLight::BAKE_TRANSLUCENT);

	edge_damp=baked_light->get_edge_damp();
	normal_damp=baked_light->get_normal_damp();
	octree_extra_margin=baked_light->get_cell_extra_margin();
	tint=baked_light->get_tint();
	ao_radius=baked_light->get_ao_radius();
	ao_strength=baked_light->get_ao_strength();
	linear_color=baked_light->get_bake_flag(BakedLight::BAKE_LINEAR_COLOR);

	baked_textures.clear();
	for(int i=0;i<baked_light->get_lightmaps_count();i++) {
		BakeTexture bt;
		bt.width=baked_light->get_lightmap_gen_size(i).x;
		bt.height=baked_light->get_lightmap_gen_size(i).y;
		baked_textures.push_back(bt);
	}


	ep.step(TTR("Parsing Geometry"),0);
	_parse_geometry(p_node);
	mat_map.clear();
	tex_map.clear();
	print_line("\ttotal triangles: "+itos(triangles.size()));
	// no geometry
	if (triangles.size() == 0) {
		return;
	}
	ep.step(TTR("Fixing Lights"),1);
	_fix_lights();
	ep.step(TTR("Making BVH"),2);
	_make_bvh();
	ep.step(TTR("Creating Light Octree"),3);
	_make_octree();
	ep.step(TTR("Creating Octree Texture"),4);
	_make_octree_texture();
	baking=true;
	_start_thread();

}


void BakedLightBaker::update_octree_sampler(PoolVector<int> &p_sampler) {

	BakedLightBaker::Octant *octants=octant_pool.ptr();
	double norm = 1.0/double(total_rays);



	if (p_sampler.size()==0 || first_bake_to_map) {

		Vector<int> tmp_smp;
		tmp_smp.resize(32); //32 for header

		for(int i=0;i<32;i++) {
			tmp_smp[i]=0;
		}

		for(int i=octant_pool_size-1;i>=0;i--) {

			if (i==0)
				tmp_smp[1]=tmp_smp.size();

			Octant &octant=octants[i];
			octant.sampler_ofs = tmp_smp.size();
			int idxcol[2]={0,0};

			int r = CLAMP((octant.full_accum[0]*norm)*2048,0,32767);
			int g = CLAMP((octant.full_accum[1]*norm)*2048,0,32767);
			int b = CLAMP((octant.full_accum[2]*norm)*2048,0,32767);

			idxcol[0]|=r;
			idxcol[1]|=(g<<16)|b;

			if (octant.leaf) {
				tmp_smp.push_back(idxcol[0]);
				tmp_smp.push_back(idxcol[1]);
			} else {

				for(int j=0;j<8;j++) {
					if (octant.children[j]) {
						idxcol[0]|=(1<<(j+16));
					}
				}
				tmp_smp.push_back(idxcol[0]);
				tmp_smp.push_back(idxcol[1]);
				for(int j=0;j<8;j++) {
					if (octant.children[j]) {
						tmp_smp.push_back(octants[octant.children[j]].sampler_ofs);
						if (octants[octant.children[j]].sampler_ofs==0) {
							print_line("FUUUUUUUUCK");
						}
					}
				}
			}

		}

		p_sampler.resize(tmp_smp.size());
		PoolVector<int>::Write w = p_sampler.write();
		int ss = tmp_smp.size();
		for(int i=0;i<ss;i++) {
			w[i]=tmp_smp[i];
		}

		first_bake_to_map=false;

	}

	double gamma = baked_light->get_gamma_adjust();
	double mult = baked_light->get_energy_multiplier();
	float saturation = baked_light->get_saturation();

	PoolVector<int>::Write w = p_sampler.write();

	encode_uint32(octree_depth,(uint8_t*)&w[2]);
	encode_uint32(linear_color,(uint8_t*)&w[3]);

	encode_float(octree_aabb.pos.x,(uint8_t*)&w[4]);
	encode_float(octree_aabb.pos.y,(uint8_t*)&w[5]);
	encode_float(octree_aabb.pos.z,(uint8_t*)&w[6]);
	encode_float(octree_aabb.size.x,(uint8_t*)&w[7]);
	encode_float(octree_aabb.size.y,(uint8_t*)&w[8]);
	encode_float(octree_aabb.size.z,(uint8_t*)&w[9]);

	//norm*=multiplier;

	for(int i=octant_pool_size-1;i>=0;i--) {

		Octant &octant=octants[i];
		int idxcol[2]={w[octant.sampler_ofs],w[octant.sampler_ofs+1]};

		double rf=pow(octant.full_accum[0]*norm*mult,gamma);
		double gf=pow(octant.full_accum[1]*norm*mult,gamma);
		double bf=pow(octant.full_accum[2]*norm*mult,gamma);

		double gray = (rf+gf+bf)/3.0;
		rf = gray + (rf-gray)*saturation;
		gf = gray + (gf-gray)*saturation;
		bf = gray + (bf-gray)*saturation;


		int r = CLAMP((rf)*2048,0,32767);
		int g = CLAMP((gf)*2048,0,32767);
		int b = CLAMP((bf)*2048,0,32767);

		idxcol[0]=((idxcol[0]>>16)<<16)|r;
		idxcol[1]=(g<<16)|b;
		w[octant.sampler_ofs]=idxcol[0];
		w[octant.sampler_ofs+1]=idxcol[1];
	}

}

void BakedLightBaker::update_octree_images(PoolVector<uint8_t> &p_octree,PoolVector<uint8_t> &p_light) {


	int len = baked_octree_texture_w*baked_octree_texture_h*4;
	p_octree.resize(len);

	int ilen = baked_light_texture_w*baked_light_texture_h*4;
	p_light.resize(ilen);


	PoolVector<uint8_t>::Write w = p_octree.write();
	zeromem(w.ptr(),len);

	PoolVector<uint8_t>::Write iw = p_light.write();
	zeromem(iw.ptr(),ilen);

	float gamma = baked_light->get_gamma_adjust();
	float mult = baked_light->get_energy_multiplier();

	for(int i=0;i<len;i+=4) {
		w[i+0]=0xFF;
		w[i+1]=0;
		w[i+2]=0xFF;
		w[i+3]=0xFF;
	}

	for(int i=0;i<ilen;i+=4) {
		iw[i+0]=0xFF;
		iw[i+1]=0;
		iw[i+2]=0xFF;
		iw[i+3]=0xFF;
	}

	float multiplier=1.0;

	if (baked_light->get_format()==BakedLight::FORMAT_HDR8)
		multiplier=8;
	encode_uint32(baked_octree_texture_w,&w[0]);
	encode_uint32(baked_octree_texture_h,&w[4]);
	encode_uint32(0,&w[8]);
	encode_float(1<<lattice_size,&w[12]);
	encode_uint32(octree_depth-lattice_size,&w[16]);
	encode_uint32(multiplier,&w[20]);
	encode_uint16(baked_light_texture_w,&w[24]); //if present, use the baked light texture
	encode_uint16(baked_light_texture_h,&w[26]);
	encode_uint32(0,&w[28]); //baked light texture format

	encode_float(octree_aabb.pos.x,&w[32]);
	encode_float(octree_aabb.pos.y,&w[36]);
	encode_float(octree_aabb.pos.z,&w[40]);
	encode_float(octree_aabb.size.x,&w[44]);
	encode_float(octree_aabb.size.y,&w[48]);
	encode_float(octree_aabb.size.z,&w[52]);


	BakedLightBaker::Octant *octants=octant_pool.ptr();
	int octant_count=octant_pool_size;
	uint8_t *ptr = w.ptr();
	uint8_t *lptr = iw.ptr();


	int child_offsets[8]={
		0,
		4,
		baked_octree_texture_w*4,
		baked_octree_texture_w*4+4,
		baked_octree_texture_w*8+0,
		baked_octree_texture_w*8+4,
		baked_octree_texture_w*8+baked_octree_texture_w*4,
		baked_octree_texture_w*8+baked_octree_texture_w*4+4,
	};

	int lchild_offsets[8]={
		0,
		4,
		baked_light_texture_w*4,
		baked_light_texture_w*4+4,
		baked_light_texture_w*8+0,
		baked_light_texture_w*8+4,
		baked_light_texture_w*8+baked_light_texture_w*4,
		baked_light_texture_w*8+baked_light_texture_w*4+4,
	};

	/*Vector<double> norm_arr;
	norm_arr.resize(lights.size());

	for(int i=0;i<lights.size();i++) {
		norm_arr[i] =  1.0/get_normalization(i);
	}

	const double *normptr=norm_arr.ptr();
*/
	double norm = 1.0/double(total_rays);
	mult/=multiplier;
	double saturation = baked_light->get_saturation();

	for(int i=0;i<octant_count;i++) {

		Octant &oct=octants[i];
		if (oct.texture_x==0 && oct.texture_y==0)
			continue;


		if (oct.leaf) {

			int ofs = (oct.texture_y * baked_light_texture_w + oct.texture_x)<<2;
			ERR_CONTINUE(ofs<0 || ofs >ilen);
			//write colors
			for(int j=0;j<8;j++) {

				/*
				if (!oct.children[j])
					continue;
				*/
				uint8_t *iptr=&lptr[ofs+lchild_offsets[j]];

				float r=oct.light_accum[j][0]*norm;
				float g=oct.light_accum[j][1]*norm;
				float b=oct.light_accum[j][2]*norm;

				r=pow(r*mult,gamma);
				g=pow(g*mult,gamma);
				b=pow(b*mult,gamma);

				double gray = (r+g+b)/3.0;
				r = gray + (r-gray)*saturation;
				g = gray + (g-gray)*saturation;
				b = gray + (b-gray)*saturation;

				float ic[3]={
					r,
					g,
					b,
				};
				iptr[0]=CLAMP(ic[0]*255.0,0,255);
				iptr[1]=CLAMP(ic[1]*255.0,0,255);
				iptr[2]=CLAMP(ic[2]*255.0,0,255);
				iptr[3]=255;
			}

		} else {

			int ofs = (oct.texture_y * baked_octree_texture_w + oct.texture_x)<<2;
			ERR_CONTINUE(ofs<0 || ofs >len);

			//write indices
			for(int j=0;j<8;j++) {

				if (!oct.children[j])
					continue;
				Octant&choct=octants[oct.children[j]];
				uint8_t *iptr=&ptr[ofs+child_offsets[j]];

				iptr[0]=choct.texture_x>>8;
				iptr[1]=choct.texture_x&0xFF;
				iptr[2]=choct.texture_y>>8;
				iptr[3]=choct.texture_y&0xFF;

			}
		}

	}


}


void BakedLightBaker::_free_bvh(BVH* p_bvh) {

	if (!p_bvh->leaf) {
		if (p_bvh->children[0])
			_free_bvh(p_bvh->children[0]);
		if (p_bvh->children[1])
			_free_bvh(p_bvh->children[1]);
	}

	memdelete(p_bvh);

}


bool BakedLightBaker::is_baking() {

	return baking;
}

void BakedLightBaker::set_pause(bool p_pause){

	if (paused==p_pause)
		return;

	paused=p_pause;

	if (paused) {
		_stop_thread();
	} else {
		_start_thread();
	}
}
bool BakedLightBaker::is_paused() {

	return paused;

}

void BakedLightBaker::_bake_thread_func(void *arg) {

	BakedLightBaker *ble = (BakedLightBaker*)arg;



	ThreadStack thread_stack;

	thread_stack.ray_stack = memnew_arr(uint32_t,ble->bvh_depth);
	thread_stack.bvh_stack = memnew_arr(BVH*,ble->bvh_depth);
	thread_stack.octant_stack = memnew_arr(uint32_t,ble->octree_depth*2 );
	thread_stack.octantptr_stack = memnew_arr(uint32_t,ble->octree_depth*2 );

	while(!ble->bake_thread_exit) {

		ble->throw_rays(thread_stack,1000);
	}

	memdelete_arr(thread_stack.ray_stack );
	memdelete_arr(thread_stack.bvh_stack );
	memdelete_arr(thread_stack.octant_stack );
	memdelete_arr(thread_stack.octantptr_stack );

}

void BakedLightBaker::_start_thread() {

	if (threads.size()!=0)
		return;
	bake_thread_exit=false;

	int thread_count = EDITOR_DEF("light_baker/custom_bake_threads",0);
	if (thread_count<=0 || thread_count>64)
		thread_count=OS::get_singleton()->get_processor_count();

	//thread_count=1;
	threads.resize(thread_count);
	for(int i=0;i<threads.size();i++) {
		threads[i]=Thread::create(_bake_thread_func,this);
	}
}

void BakedLightBaker::_stop_thread() {

	if (threads.size()==0)
		return;
	bake_thread_exit=true;
	for(int i=0;i<threads.size();i++) {
		Thread::wait_to_finish(threads[i]);
		memdelete(threads[i]);
	}
	threads.clear();
}

void BakedLightBaker::_plot_pixel_to_lightmap(int x, int y, int width, int height, uint8_t *image, const Vector3& p_pos,const Vector3& p_normal,double *p_norm_ptr,float mult,float gamma) {


	uint8_t *ptr = &image[(y*width+x)*4];
	//int lc = lights.size();
	double norm = 1.0/double(total_rays);


	Color color;

	Octant *octants=octant_pool.ptr();


	int octant_idx=0;


	while(true) {

		Octant &octant=octants[octant_idx];

		if (octant.leaf) {

			Vector3 lpos = p_pos-octant.aabb.pos;
			lpos/=octant.aabb.size;

			Vector3 cols[8];

			for(int i=0;i<8;i++) {

				cols[i].x+=octant.light_accum[i][0]*norm;
				cols[i].y+=octant.light_accum[i][1]*norm;
				cols[i].z+=octant.light_accum[i][2]*norm;
			}


			/*Vector3 final = (cols[0] + (cols[1] - cols[0]) * lpos.y);
			final = final + ((cols[2] + (cols[3] - cols[2]) * lpos.y) - final)*lpos.x;

			Vector3 final2 = (cols[4+0] + (cols[4+1] - cols[4+0]) * lpos.y);
			final2 = final2 + ((cols[4+2] + (cols[4+3] - cols[4+2]) * lpos.y) - final2)*lpos.x;*/

			Vector3 finala = cols[0].linear_interpolate(cols[1],lpos.x);
			Vector3 finalb = cols[2].linear_interpolate(cols[3],lpos.x);
			Vector3 final = finala.linear_interpolate(finalb,lpos.y);

			Vector3 final2a = cols[4+0].linear_interpolate(cols[4+1],lpos.x);
			Vector3 final2b = cols[4+2].linear_interpolate(cols[4+3],lpos.x);
			Vector3 final2 = final2a.linear_interpolate(final2b,lpos.y);

			final = final.linear_interpolate(final2,lpos.z);
			if (baked_light->get_format()==BakedLight::FORMAT_HDR8)
				final*=8.0;


			color.r=pow(final.x*mult,gamma);
			color.g=pow(final.y*mult,gamma);
			color.b=pow(final.z*mult,gamma);
			color.a=1.0;

			int lc = lights.size();
			LightData *lv = lights.ptr();
			for(int i=0;i<lc;i++) {
				//shadow baking
				if (!lv[i].bake_shadow)
					continue;
				Vector3 from = p_pos+p_normal*0.01;
				Vector3 to;
				float att=0;
				switch(lv[i].type) {
					case VS::LIGHT_DIRECTIONAL: {
						to=from-lv[i].dir*lv[i].length;
					} break;
					case VS::LIGHT_OMNI: {
						to=lv[i].pos;
						float d = MIN(lv[i].radius,to.distance_to(from))/lv[i].radius;
						att=d;//1.0-d;
					} break;
					default: continue;
				}

				uint32_t* stack = ray_stack;
				BVH **bstack = bvh_stack;

				enum {
					TEST_RAY_BIT=0,
					VISIT_LEFT_BIT=1,
					VISIT_RIGHT_BIT=2,
					VISIT_DONE_BIT=3,


				};

				bool intersected=false;

				int level=0;

				Vector3 n = (to-from);
				float len=n.length();
				if (len==0)
					continue;
				n/=len;

				bstack[0]=bvh;
				stack[0]=TEST_RAY_BIT;


				while(!intersected) {

					uint32_t mode = stack[level];
					const BVH &b = *bstack[level];
					bool done=false;

					switch(mode) {
						case TEST_RAY_BIT: {

							if (b.leaf) {


								Face3 f3(b.leaf->vertices[0],b.leaf->vertices[1],b.leaf->vertices[2]);


								Vector3 res;

								if (f3.intersects_segment(from,to)) {
									intersected=true;
									done=true;
								}

								stack[level]=VISIT_DONE_BIT;
							} else {


								bool valid = b.aabb.smits_intersect_ray(from,n,0,len);
								//bool valid = b.aabb.intersects_segment(p_begin,p_end);
								//bool valid = b.aabb.intersects(ray_aabb);

								if (!valid) {

									stack[level]=VISIT_DONE_BIT;

								} else {

									stack[level]=VISIT_LEFT_BIT;
								}
							}

						} continue;
						case VISIT_LEFT_BIT: {

							stack[level]=VISIT_RIGHT_BIT;
							bstack[level+1]=b.children[0];
							stack[level+1]=TEST_RAY_BIT;
							level++;

						} continue;
						case VISIT_RIGHT_BIT: {

							stack[level]=VISIT_DONE_BIT;
							bstack[level+1]=b.children[1];
							stack[level+1]=TEST_RAY_BIT;
							level++;
						} continue;
						case VISIT_DONE_BIT: {

							if (level==0) {
								done=true;
								break;
							} else
								level--;

						} continue;
					}


					if (done)
						break;
				}



				if (intersected) {

					color.a=Math::lerp(MAX(0.01,lv[i].darkening),1.0,att);
				}

			}

			break;
		} else {

			Vector3 lpos = p_pos - octant.aabb.pos;
			Vector3 half = octant.aabb.size * 0.5;

			int ofs=0;

			if (lpos.x >= half.x)
				ofs|=1;
			if (lpos.y >= half.y)
				ofs|=2;
			if (lpos.z >= half.z)
				ofs|=4;

			octant_idx = octant.children[ofs];

			if (octant_idx==0)
				return;

		}
	}

	ptr[0]=CLAMP(color.r*255.0,0,255);
	ptr[1]=CLAMP(color.g*255.0,0,255);
	ptr[2]=CLAMP(color.b*255.0,0,255);
	ptr[3]=CLAMP(color.a*255.0,0,255);

}


Error BakedLightBaker::transfer_to_lightmaps() {

	if (!triangles.size() || baked_textures.size()==0)
		return ERR_UNCONFIGURED;

	EditorProgress ep("transfer_to_lightmaps",TTR("Transfer to Lightmaps:"),baked_textures.size()*2+triangles.size());

	for(int i=0;i<baked_textures.size();i++) {

		ERR_FAIL_COND_V( baked_textures[i].width<=0 || baked_textures[i].height<=0,ERR_UNCONFIGURED );

		baked_textures[i].data.resize( baked_textures[i].width*baked_textures[i].height*4 );
		zeromem(baked_textures[i].data.ptr(),baked_textures[i].data.size());
		ep.step(TTR("Allocating Texture #")+itos(i+1),i);
	}

	Vector<double> norm_arr;
	norm_arr.resize(lights.size());

	for(int i=0;i<lights.size();i++) {
		norm_arr[i] =  1.0/get_normalization(i);
	}
	float gamma = baked_light->get_gamma_adjust();
	float mult = baked_light->get_energy_multiplier();

	for(int i=0;i<triangles.size();i++) {

		if (i%200==0) {
			ep.step(TTR("Baking Triangle #")+itos(i),i+baked_textures.size());
		}
		Triangle &t=triangles[i];
		if (t.baked_texture<0 || t.baked_texture>=baked_textures.size())
			continue;

		BakeTexture &bt=baked_textures[t.baked_texture];
		Vector3 normal = Plane(t.vertices[0],t.vertices[1],t.vertices[2]).normal;


		int x[3];
		int y[3];

		Vector3 vertices[3]={
			t.vertices[0],
			t.vertices[1],
			t.vertices[2]
		};

		for(int j=0;j<3;j++) {

			x[j]=t.bake_uvs[j].x*bt.width;
			y[j]=t.bake_uvs[j].y*bt.height;
			x[j]=CLAMP(x[j],0,bt.width-1);
			y[j]=CLAMP(y[j],0,bt.height-1);
		}


		{

			// sort the points vertically
			if (y[1] > y[2])  {
				SWAP(x[1], x[2]);
				SWAP(y[1], y[2]);
				SWAP(vertices[1],vertices[2]);
			}
			if (y[0] > y[1]) {
				SWAP(x[0], x[1]);
				SWAP(y[0], y[1]);
				SWAP(vertices[0],vertices[1]);
			}
			if (y[1] > y[2]) {
				SWAP(x[1], x[2]);
				SWAP(y[1], y[2]);
				SWAP(vertices[1],vertices[2]);
			}

			double dx_far = double(x[2] - x[0]) / (y[2] - y[0] + 1);
			double dx_upper = double(x[1] - x[0]) / (y[1] - y[0] + 1);
			double dx_low = double(x[2] - x[1]) / (y[2] - y[1] + 1);
			double xf = x[0];
			double xt = x[0] + dx_upper; // if y[0] == y[1], special case
			for (int yi = y[0]; yi <= (y[2] > bt.height-1 ? bt.height-1 : y[2]); yi++)
			{
				if (yi >= 0) {
					for (int xi = (xf > 0 ? int(xf) : 0); xi <= (xt < bt.width ? xt : bt.width-1) ; xi++) {
						//pixels[int(x + y * width)] = color;

						Vector2 v0 = Vector2(x[1]-x[0],y[1]-y[0]);
						Vector2 v1 = Vector2(x[2]-x[0],y[2]-y[0]);
						//vertices[2] - vertices[0];
						Vector2 v2 = Vector2(xi-x[0],yi-y[0]);
						float d00 = v0.dot( v0);
						float d01 = v0.dot( v1);
						float d11 = v1.dot( v1);
						float d20 = v2.dot( v0);
						float d21 = v2.dot( v1);
						float denom = (d00 * d11 - d01 * d01);
						Vector3 pos;
						if (denom==0) {
							pos=t.vertices[0];
						} else {
							float v = (d11 * d20 - d01 * d21) / denom;
							float w = (d00 * d21 - d01 * d20) / denom;
							float u = 1.0f - v - w;
							pos = vertices[0]*u + vertices[1]*v  + vertices[2]*w;
						}
						_plot_pixel_to_lightmap(xi,yi,bt.width,bt.height,bt.data.ptr(),pos,normal,norm_arr.ptr(),mult,gamma);

					}

					for (int xi = (xf < bt.width ? int(xf) : bt.width-1); xi >= (xt > 0 ? xt : 0); xi--) {
						//pixels[int(x + y * width)] = color;
						Vector2 v0 = Vector2(x[1]-x[0],y[1]-y[0]);
						Vector2 v1 = Vector2(x[2]-x[0],y[2]-y[0]);
						//vertices[2] - vertices[0];
						Vector2 v2 = Vector2(xi-x[0],yi-y[0]);
						float d00 = v0.dot( v0);
						float d01 = v0.dot( v1);
						float d11 = v1.dot( v1);
						float d20 = v2.dot( v0);
						float d21 = v2.dot( v1);
						float denom = (d00 * d11 - d01 * d01);
						Vector3 pos;
						if (denom==0) {
							pos=t.vertices[0];
						} else {
							float v = (d11 * d20 - d01 * d21) / denom;
							float w = (d00 * d21 - d01 * d20) / denom;
							float u = 1.0f - v - w;
							pos = vertices[0]*u + vertices[1]*v  + vertices[2]*w;
						}

						_plot_pixel_to_lightmap(xi,yi,bt.width,bt.height,bt.data.ptr(),pos,normal,norm_arr.ptr(),mult,gamma);

					}
				}
				xf += dx_far;
				if (yi < y[1])
					xt += dx_upper;
				else
					xt += dx_low;
			}
		}

	}


	for(int i=0;i<baked_textures.size();i++) {


		{

			ep.step(TTR("Post-Processing Texture #")+itos(i),i+baked_textures.size()+triangles.size());

			BakeTexture &bt=baked_textures[i];

			Vector<uint8_t> copy_data=bt.data;
			uint8_t *data=bt.data.ptr();
			const int max_radius=8;
			const int shadow_radius=2;
			const int max_dist=0x7FFFFFFF;

			for(int x=0;x<bt.width;x++) {

				for(int y=0;y<bt.height;y++) {


					uint8_t a = copy_data[(y*bt.width+x)*4+3];

					if (a>0) {
						//blur shadow

						int from_x = MAX(0,x-shadow_radius);
						int to_x = MIN(bt.width-1,x+shadow_radius);
						int from_y = MAX(0,y-shadow_radius);
						int to_y = MIN(bt.height-1,y+shadow_radius);

						int sum=0;
						int sumc=0;

						for(int k=from_y;k<=to_y;k++) {
							for(int l=from_x;l<=to_x;l++) {

								const uint8_t * rp = &copy_data[(k*bt.width+l)<<2];

								sum+=rp[3];
								sumc++;
							}
						}

						sum/=sumc;
						data[(y*bt.width+x)*4+3]=sum;

					} else {

						int closest_dist=max_dist;
						uint8_t closest_color[4];

						int from_x = MAX(0,x-max_radius);
						int to_x = MIN(bt.width-1,x+max_radius);
						int from_y = MAX(0,y-max_radius);
						int to_y = MIN(bt.height-1,y+max_radius);

						for(int k=from_y;k<=to_y;k++) {
							for(int l=from_x;l<=to_x;l++) {

								int dy = y-k;
								int dx = x-l;
								int dist = dy*dy+dx*dx;
								if (dist>=closest_dist)
									continue;

								const uint8_t * rp = &copy_data[(k*bt.width+l)<<2];

								if (rp[3]==0)
									continue;

								closest_dist=dist;
								closest_color[0]=rp[0];
								closest_color[1]=rp[1];
								closest_color[2]=rp[2];
								closest_color[3]=rp[3];
							}
						}


						if (closest_dist!=max_dist) {

							data[(y*bt.width+x)*4+0]=closest_color[0];
							data[(y*bt.width+x)*4+1]=closest_color[1];
							data[(y*bt.width+x)*4+2]=closest_color[2];
							data[(y*bt.width+x)*4+3]=closest_color[3];
						}
					}
				}
			}
		}

		PoolVector<uint8_t> dv;
		dv.resize(baked_textures[i].data.size());
		{
			PoolVector<uint8_t>::Write w = dv.write();
			copymem(w.ptr(),baked_textures[i].data.ptr(),baked_textures[i].data.size());
		}

		Image img(baked_textures[i].width,baked_textures[i].height,0,Image::FORMAT_RGBA8,dv);
		Ref<ImageTexture> tex = memnew( ImageTexture );
		tex->create_from_image(img);
		baked_light->set_lightmap_texture(i,tex);
	}


	return OK;
}

void BakedLightBaker::clear() {



	_stop_thread();

	if (bvh)
		_free_bvh(bvh);

	if (ray_stack)
		memdelete_arr(ray_stack);
	if (octant_stack)
		memdelete_arr(octant_stack);
	if (octantptr_stack)
		memdelete_arr(octantptr_stack);
	if (bvh_stack)
		memdelete_arr(bvh_stack);
/*
 * ???
	for(int i=0;i<octant_pool.size();i++) {
		/*
		if (octant_pool[i].leaf) {
			memdelete_arr( octant_pool[i].light );
		}
		Vector<double> norm_arr;
		norm_arr.resize(lights.size());
		*/

		for(int i=0;i<lights.size();i++) {
			norm_arr[i] =  1.0/get_normalization(i);
		}

		const double *normptr=norm_arr.ptr();
	}
*/
	octant_pool.clear();
	octant_pool_size=0;
	bvh=NULL;
	leaf_list=0;
	cell_count=0;
	ray_stack=NULL;
	octant_stack=NULL;
	octantptr_stack=NULL;
	bvh_stack=NULL;
	materials.clear();
	materials.clear();
	textures.clear();
	lights.clear();
	triangles.clear();
	endpoint_normal.clear();
	endpoint_normal_bits.clear();
	baked_octree_texture_w=0;
	baked_octree_texture_h=0;
	paused=false;
	baking=false;

	bake_thread_exit=false;
	first_bake_to_map=true;
	baked_light=Ref<BakedLight>();
	total_rays=0;

}

BakedLightBaker::BakedLightBaker() {
	octree_depth=9;
	lattice_size=4;
	octant_pool.clear();
	octant_pool_size=0;
	bvh=NULL;
	leaf_list=0;
	cell_count=0;
	ray_stack=NULL;
	bvh_stack=NULL;
	octant_stack=NULL;
	octantptr_stack=NULL;
	plot_size=2.5;
	max_bounces=2;
	materials.clear();
	baked_octree_texture_w=0;
	baked_octree_texture_h=0;
	paused=false;
	baking=false;

	bake_thread_exit=false;
	total_rays=0;
	first_bake_to_map=true;
	linear_color=false;

}

BakedLightBaker::~BakedLightBaker() {

	clear();
}
#endif
