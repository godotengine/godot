
#include "baked_light_baker.h"
#include <stdlib.h>
#include <cmath>
#include "io/marshalls.h"
#include "tools/editor/editor_node.h"


BakedLightBaker::MeshTexture* BakedLightBaker::_get_mat_tex(const Ref<Texture>& p_tex) {

	if (!tex_map.has(p_tex)) {

		Ref<ImageTexture> imgtex=p_tex;
		if (imgtex.is_null())
			return NULL;
		Image image=imgtex->get_data();
		if (image.empty())
			return NULL;

		if (image.get_format()!=Image::FORMAT_RGBA) {
			if (image.get_format()>Image::FORMAT_INDEXED_ALPHA) {
				Error err = image.decompress();
				if (err)
					return NULL;
			}

			if (image.get_format()!=Image::FORMAT_RGBA)
				image.convert(Image::FORMAT_RGBA);
		}

		DVector<uint8_t> dvt=image.get_data();
		DVector<uint8_t>::Read r=dvt.read();
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


void BakedLightBaker::_add_mesh(const Ref<Mesh>& p_mesh,const Ref<Material>& p_mat_override,const Transform& p_xform) {


	for(int i=0;i<p_mesh->get_surface_count();i++) {

		if (p_mesh->surface_get_primitive_type(i)!=Mesh::PRIMITIVE_TRIANGLES)
			continue;
		Ref<Material> mat = p_mat_override.is_valid()?p_mat_override:p_mesh->surface_get_material(i);

		MeshMaterial *matptr=NULL;

		if (mat.is_valid()) {

			if (!mat_map.has(mat)) {

				MeshMaterial mm;

				Ref<FixedMaterial> fm = mat;
				if (fm.is_valid()) {
					//fixed route
					mm.diffuse.color=fm->get_parameter(FixedMaterial::PARAM_DIFFUSE);
					mm.diffuse.tex=_get_mat_tex(fm->get_texture(FixedMaterial::PARAM_DIFFUSE));
					mm.specular.color=fm->get_parameter(FixedMaterial::PARAM_SPECULAR);
					mm.specular.tex=_get_mat_tex(fm->get_texture(FixedMaterial::PARAM_SPECULAR));
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

		DVector<Vector3> vertices = a[Mesh::ARRAY_VERTEX];
		DVector<Vector3>::Read vr=vertices.read();
		DVector<Vector2> uv;
		DVector<Vector2>::Read uvr;
		DVector<Vector3> normal;
		DVector<Vector3>::Read normalr;
		bool read_uv=false;
		bool read_normal=false;

		if (p_mesh->surface_get_format(i)&Mesh::ARRAY_FORMAT_TEX_UV) {

			uv=a[Mesh::ARRAY_TEX_UV];
			uvr=uv.read();
			read_uv=true;
		}

		if (p_mesh->surface_get_format(i)&Mesh::ARRAY_FORMAT_NORMAL) {

			normal=a[Mesh::ARRAY_NORMAL];
			normalr=normal.read();
			read_normal=true;
		}

		Matrix3 normal_xform = p_xform.basis.inverse().transposed();


		if (p_mesh->surface_get_format(i)&Mesh::ARRAY_FORMAT_INDEX) {

			DVector<int> indices = a[Mesh::ARRAY_INDEX];
			DVector<int>::Read ir = indices.read();

			for(int i=0;i<facecount;i++) {
				Triangle &t=triangles[tbase+i];
				t.vertices[0]=p_xform.xform(vr[ ir[i*3+0] ]);
				t.vertices[1]=p_xform.xform(vr[ ir[i*3+1] ]);
				t.vertices[2]=p_xform.xform(vr[ ir[i*3+2] ]);
				t.material=matptr;
				if (read_uv) {

					t.uvs[0]=uvr[ ir[i*3+0] ];
					t.uvs[1]=uvr[ ir[i*3+1] ];
					t.uvs[2]=uvr[ ir[i*3+2] ];
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
				if (read_uv) {

					t.uvs[0]=uvr[ i*3+0 ];
					t.uvs[1]=uvr[ i*3+1 ];
					t.uvs[2]=uvr[ i*3+2 ];
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
			_add_mesh(mesh,meshi->get_material_override(),base_inv * meshi->get_global_transform());
		}
	} else if (p_node->cast_to<Light>()) {

		Light *dl=p_node->cast_to<Light>();

		if (dl->get_bake_mode()!=Light::BAKE_MODE_DISABLED) {


			LightData dirl;
			dirl.type=VS::LightType(dl->get_light_type());
			dirl.diffuse=dl->get_color(DirectionalLight::COLOR_DIFFUSE);
			dirl.specular=dl->get_color(DirectionalLight::COLOR_SPECULAR);
			dirl.energy=dl->get_parameter(DirectionalLight::PARAM_ENERGY);
			dirl.pos=dl->get_global_transform().origin;
			dirl.up=dl->get_global_transform().basis.get_axis(1).normalized();
			dirl.left=dl->get_global_transform().basis.get_axis(0).normalized();
			dirl.dir=-dl->get_global_transform().basis.get_axis(2).normalized();
			dirl.spot_angle=dl->get_parameter(DirectionalLight::PARAM_SPOT_ANGLE);
			dirl.spot_attenuation=dl->get_parameter(DirectionalLight::PARAM_SPOT_ATTENUATION);
			dirl.attenuation=dl->get_parameter(DirectionalLight::PARAM_ATTENUATION);
			dirl.radius=dl->get_parameter(DirectionalLight::PARAM_RADIUS);
			dirl.bake_direct=dl->get_bake_mode()==Light::BAKE_MODE_FULL;
			dirl.rays_thrown=0;
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

					dl.area=4.0*Math_PI*pow(dl.radius,2.0);
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

					int lz = lights.size();
					child->light = memnew_arr(OctantLight,lz);

					for(int li=0;li<lz;li++) {
						for(int ci=0;ci<8;ci++) {
							child->light[li].accum[ci][0]=0;
							child->light[li].accum[ci][1]=0;
							child->light[li].accum[ci][2]=0;
						}
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

					Map<Vector3,Vector3>::Element *E=endpoint_normal.find(pos);
					if (!E) {
						endpoint_normal[pos]=n;
					} else {
						E->get()+=n;
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

	EditorProgress ep("bake_octree","Parsing "+itos(triangles.size())+" Triangles:",triangles.size());

	for(int i=0;i<triangles.size();i++) {

		_octree_insert(0,&triangles[i],octree_depth-1);
		if ((i%1000)==0) {

			ep.step("Triangle# "+itos(i),i);
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

			oct_idx=oct->next_leaf;
		}
	}


}





void BakedLightBaker::_plot_light(int p_light_index, const Vector3& p_plot_pos, const AABB& p_plot_aabb, const Color& p_light, const Plane& p_plane) {

	//stackless version

	uint32_t *stack=octant_stack;
	uint32_t *ptr_stack=octantptr_stack;
	Octant *octants=octant_pool.ptr();

	stack[0]=0;
	ptr_stack[0]=0;

	int stack_pos=0;


	while(true) {

		Octant &octant=octants[ptr_stack[stack_pos]];

		if (octant.leaf) {



			//if (p_plane.normal.dot(octant.aabb.get_support(p_plane.normal)) < p_plane.d-CMP_EPSILON) { //octants behind are no go



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

				if (d<=r) {


					float intensity = 1.0 - (d/r)*(d/r); //not gauss but..
					float damp = Math::abs(p_plane.normal.dot(Vector3(octant.normal_accum[i][0],octant.normal_accum[i][1],octant.normal_accum[i][2])));
					intensity*=pow(damp,edge_damp);
					//intensity*=1.0-Math::abs(p_plane.distance_to(pos))/(plot_size*cell_size);
					octant.light[p_light_index].accum[i][0]+=p_light.r*intensity;
					octant.light[p_light_index].accum[i][1]+=p_light.g*intensity;
					octant.light[p_light_index].accum[i][2]+=p_light.b*intensity;
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


float BakedLightBaker::_throw_ray(int p_light_index,const Vector3& p_begin, const Vector3& p_end,float p_rest,const Color& p_light,float *p_att_curve,float p_att_pos,int p_att_curve_len,int p_bounces,bool p_first_bounce) {


	uint32_t* stack = ray_stack;
	BVH **bstack = bvh_stack;

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

	//for(int i=0;i<max_depth;i++)
	//	stack[i]=0;

	int level=0;
	//AABB ray_aabb;
	//ray_aabb.pos=p_begin;
	//ray_aabb.expand_to(p_end);


	const BVH *bvhptr = bvh;

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
	//				bool valid = b.aabb.intersects(ray_aabb);

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



		//should check if there is normals first
		Vector2 uv;
		if (true) {

			triangle->get_uv_and_normal(r_point,uv,r_normal);

		} else {

		}

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


		float ret=1e6;

		if (p_bounces>0) {


			p_rest-=dist;
			if (p_rest<CMP_EPSILON)
				return 0;

			if (r_normal==-n)
				return 0; //todo change a little

			r_point+=r_normal*0.01;



			if (triangle->material) {

				//triangle->get_uv(r_point);

				diffuse_at_point=triangle->material->diffuse.get_color(uv);
				specular_at_point=triangle->material->specular.get_color(uv);
			}


			diffuse_at_point.r=res_light.r*diffuse_at_point.r;
			diffuse_at_point.g=res_light.g*diffuse_at_point.g;
			diffuse_at_point.b=res_light.b*diffuse_at_point.b;

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


				ret=_throw_ray(p_light_index,r_point,r_point+rn*p_rest,p_rest,diffuse_at_point,p_att_curve,p_att_pos,p_att_curve_len,p_bounces-1);
			}

			if (use_specular && (specular_at_point.r>CMP_EPSILON || specular_at_point.g>CMP_EPSILON || specular_at_point.b>CMP_EPSILON)) {
				//specular bounce

				//Vector3 c1=r_normal.cross(n).normalized();
				//Vector3 c2=r_normal.cross(c1).normalized();

				Vector3 rn = n - r_normal *r_normal.dot(n) * 2.0;

				_throw_ray(p_light_index,r_point,r_point+rn*p_rest,p_rest,specular_at_point,p_att_curve,p_att_pos,p_att_curve_len,p_bounces-1);
			}
		}

		//specular later
//		_plot_light_point(r_point,octree,octree_aabb,p_light);


		Color plot_light=res_light;
		plot_light.r*=att;
		plot_light.g*=att;
		plot_light.b*=att;

		if (!p_first_bounce) {


			float r = plot_size * cell_size;
			if (ret<r) {
				//avoid accumulaiton of light on corners
				//plot_light=plot_light.linear_interpolate(Color(0,0,0,0),1.0-sd/plot_size*plot_size);
				plot_light=Color(0,0,0,0);
			}
		}


		if (!p_first_bounce || lights[p_light_index].bake_direct) {
			Plane plane(r_point,r_normal);
			//print_line(String(plot_light)+String(" ")+rtos(att));
			_plot_light(p_light_index,r_point,aabb,plot_light,plane);
		}


		return dist;
	}

	return 0;

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

			uint32_t hash = HashMapHahserDefault::hash(base);
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

			uint32_t hash = HashMapHahserDefault::hash(base);
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

	{

		int otex_w=(1<<lattice_size)*(1<<lattice_size)*2; //make sure lattice fits horizontally
		Vector3 lattice_cell_size=octree_aabb.size;
		for(int i=0;i<lattice_size;i++) {

			lattice_cell_size*=0.5;
		}



		while(true) {

			//let's plot the leafs first, given the octree is not so obvious which size it will have
			int row=4+4*(1<<lattice_size);


			uint32_t oct_idx=leaf_list;

			//untag
			while(oct_idx) {

				BakedLightBaker::Octant *oct = &octants[oct_idx];
				//0,0 also means unprocessed
				oct->texture_x=0;
				oct->texture_y=0;
				oct_idx=oct->next_leaf;

			}

			oct_idx=leaf_list;


			print_line("begin at row "+itos(row));
			int longest_line_reused=0;
			int col=0;
			int processed=0;

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

			print_line("processed "+itos(processed));

			print_line("longest reused: "+itos(longest_line_reused));

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
	double cell_area = cell_size*cell_size;;
	//nrg+= /*dl.energy */ (dl.rays_thrown * cell_area / dl.area);
	nrg=dl.rays_thrown * cell_area;
	nrg*=(Math_PI*plot_size*plot_size)*0.5; // damping of radial linear gradient kernel
	nrg*=dl.constant;
	//nrg*=5;
	print_line("CS: "+rtos(cell_size));

	return nrg;
}

void BakedLightBaker::throw_rays(int p_amount) {



	for(int i=0;i<lights.size();i++) {

		LightData &dl=lights[i];


		int amount = p_amount * total_light_area / dl.area;

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
					col.r*=dl.energy;
					col.g*=dl.energy;
					col.b*=dl.energy;
					dl.rays_thrown++;
					total_rays++;
					_throw_ray(i,from,to,dl.length,col,NULL,0,0,max_bounces,true);
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
					col.r*=dl.energy;
					col.g*=dl.energy;
					col.b*=dl.energy;

					dl.rays_thrown++;
					total_rays++;
					_throw_ray(i,from,to,dl.radius,col,dl.attenuation_table.ptr(),0,dl.radius,max_bounces,true);
//					_throw_ray(i,from,to,dl.radius,col,NULL,0,dl.radius,max_bounces,true);
				}

			} break;
			case VS::LIGHT_SPOT: {

				for(int j=0;j<amount;j++) {
					Vector3 from = dl.pos;

					double r1 = double(rand())/RAND_MAX;
					double r2 = double(rand())/RAND_MAX;
					double r3 = double(rand())/RAND_MAX;

					float d=Math::tan(Math::deg2rad(dl.spot_angle));

					float x = sin(r1*Math_PI*2.0)*d;
					float y = cos(r1*Math_PI*2.0)*d;

					Vector3 dir = r3*(dl.dir + dl.up*y + dl.left*x) + (1.0-r3)*dl.dir;
					dir.normalize();


					Vector3 to = dl.pos+dir*dl.radius;
					Color col=dl.diffuse;
					col.r*=dl.energy;
					col.g*=dl.energy;
					col.b*=dl.energy;

					dl.rays_thrown++;
					total_rays++;
					_throw_ray(i,from,to,dl.radius,col,dl.attenuation_table.ptr(),0,dl.radius,max_bounces,true);
	//					_throw_ray(i,from,to,dl.radius,col,NULL,0,dl.radius,max_bounces,true);
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
	EditorProgress ep("bake","Light Baker Setup:",5);
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



	ep.step("Parsing Geometry",0);
	_parse_geometry(p_node);
	mat_map.clear();
	tex_map.clear();
	print_line("\ttotal triangles: "+itos(triangles.size()));
	ep.step("Fixing Lights",1);
	_fix_lights();
	ep.step("Making BVH",2);
	_make_bvh();
	ep.step("Creating Light Octree",3);
	_make_octree();
	ep.step("Creating Octree Texture",4);
	_make_octree_texture();
	baking=true;
	_start_thread();

}


void BakedLightBaker::update_octree_image(DVector<uint8_t> &p_image) {


	int len = baked_octree_texture_w*baked_octree_texture_h*4;
	p_image.resize(len);
	DVector<uint8_t>::Write w = p_image.write();
	zeromem(w.ptr(),len);
	float gamma = baked_light->get_gamma_adjust();
	float mult = baked_light->get_energy_multiplier();

	for(int i=0;i<len;i+=4) {
		w[i+0]=0xFF;
		w[i+1]=0;
		w[i+2]=0xFF;
		w[i+3]=0xFF;
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

	encode_float(octree_aabb.pos.x,&w[32]);
	encode_float(octree_aabb.pos.y,&w[36]);
	encode_float(octree_aabb.pos.z,&w[40]);
	encode_float(octree_aabb.size.x,&w[44]);
	encode_float(octree_aabb.size.y,&w[48]);
	encode_float(octree_aabb.size.z,&w[52]);


	BakedLightBaker::Octant *octants=octant_pool.ptr();
	int octant_count=octant_pool_size;
	uint8_t *ptr = w.ptr();


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

	Vector<double> norm_arr;
	norm_arr.resize(lights.size());

	for(int i=0;i<lights.size();i++) {
		norm_arr[i] =  1.0/get_normalization(i);
	}

	const double *normptr=norm_arr.ptr();

	int lz=lights.size();
	mult/=multiplier;

	for(int i=0;i<octant_count;i++) {

		Octant &oct=octants[i];
		if (oct.texture_x==0 && oct.texture_y==0)
			continue;
		int ofs = (oct.texture_y * baked_octree_texture_w + oct.texture_x)<<2;

		if (oct.leaf) {

			//write colors
			for(int j=0;j<8;j++) {

				//if (!oct.children[j])
				//	continue;
				uint8_t *iptr=&ptr[ofs+child_offsets[j]];
				float r=0;
				float g=0;
				float b=0;

				for(int k=0;k<lz;k++) {
					r+=oct.light[k].accum[j][0]*normptr[k];
					g+=oct.light[k].accum[j][1]*normptr[k];
					b+=oct.light[k].accum[j][2]*normptr[k];
				}

				r=pow(r*mult,gamma);
				g=pow(g*mult,gamma);
				b=pow(b*mult,gamma);

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

	ble->rays_at_snap_time=ble->total_rays;
	ble->snap_time=OS::get_singleton()->get_ticks_usec();

	while(!ble->bake_thread_exit) {

		ble->throw_rays(1000);
		uint64_t t=OS::get_singleton()->get_ticks_usec();
		if (t-ble->snap_time>1000000) {

			double time = (t-ble->snap_time)/1000000.0;

			int rays=ble->total_rays-ble->rays_at_snap_time;
			ble->rays_sec=int(rays/time);
			ble->snap_time=OS::get_singleton()->get_ticks_usec();
			ble->rays_at_snap_time=ble->total_rays;
		}
	}

}

void BakedLightBaker::_start_thread() {

	if (thread!=NULL)
		return;
	bake_thread_exit=false;
	thread=Thread::create(_bake_thread_func,this);

}

void BakedLightBaker::_stop_thread() {

	if (thread==NULL)
		return;
	bake_thread_exit=true;
	Thread::wait_to_finish(thread);
	thread=NULL;
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

	for(int i=0;i<octant_pool.size();i++) {
		if (octant_pool[i].leaf) {
			memdelete_arr( octant_pool[i].light );
		}
	}
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
	triangles.clear();;
	endpoint_normal.clear();
	baked_octree_texture_w=0;
	baked_octree_texture_h=0;
	paused=false;
	baking=false;
	thread=NULL;
	bake_thread_exit=false;
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
	thread=NULL;
	bake_thread_exit=false;
	rays_at_snap_time=0;
	snap_time=0;
	rays_sec=0;
	total_rays=0;

}

BakedLightBaker::~BakedLightBaker() {

	clear();
}
