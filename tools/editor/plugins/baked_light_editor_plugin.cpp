#include "baked_light_editor_plugin.h"
#include "scene/gui/box_container.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/light.h"


class BakedLightBaker {
public:

	enum {

		ATTENUATION_CURVE_LEN=256
	};

	struct Octant {
		bool leaf;
		union {
			struct {
				float light_accum[3];
				float surface_area;
				Octant *next_leaf;
				float offset[3];
			};
			Octant* children[8];
		};
	};

	struct Triangle {

		Vector3 vertices[3];
		Vector2 uv[3];
	};


	struct BVH {

		AABB aabb;
		Vector3 center;
		Triangle *leaf;
		BVH*children[2];
	};


	struct BVHCmpX {

		bool operator()(const BVH* p_left, const BVH* p_right) const {

			return p_left->center.x < p_right->center.x;
		}
	};

	struct BVHCmpY {

		bool operator()(const BVH* p_left, const BVH* p_right) const {

			return p_left->center.y < p_right->center.y;
		}
	};
	struct BVHCmpZ {

		bool operator()(const BVH* p_left, const BVH* p_right) const {

			return p_left->center.z < p_right->center.z;
		}
	};


	struct DirLight {


		Vector3 pos;
		Vector3 up;
		Vector3 left;
		Vector3 dir;
		Color diffuse;
		Color specular;
		float energy;
		float length;
		int rays_thrown;

	};

	AABB octree_aabb;
	Octant *octree;
	BVH*bvh;
	Vector<Triangle> triangles;
	Transform base_inv;
	Octant *leaf_list;
	int octree_depth;
	int cell_count;
	uint32_t *ray_stack;
	BVH **bvh_stack;
	float cell_size;
	float plot_size; //multiplied by cell size
	Vector<DirLight> directional_lights;
	int max_bounces;



	void _add_mesh(const Ref<Mesh>& p_mesh,const Ref<Material>& p_mat_override,const Transform& p_xform);
	void _parse_geometry(Node* p_node);
	BVH* _parse_bvh(BVH** p_children,int p_size,int p_depth,int& max_depth);
	void _make_bvh();
	void _make_octree();
	void _octree_insert(const AABB& p_aabb,Octant *p_octant,Triangle* p_triangle, int p_depth);

	void _free_octree(Octant *p_octant) {

		if (!p_octant->leaf) {

			for(int i=0;i<8;i++) {
				if (p_octant->children[i])
					_free_octree(p_octant->children[i]);
			}
		}

		memdelete(p_octant);
	}

	void _free_bvh(BVH* p_bvh) {

		if (!p_bvh->leaf) {
			if (p_bvh->children[0])
				_free_bvh(p_bvh->children[0]);
			if (p_bvh->children[1])
				_free_bvh(p_bvh->children[1]);
		}

		memdelete(p_bvh);

	}

	void _fix_lights();


	void _plot_light(const Vector3& p_plot_pos,const AABB& p_plot_aabb, Octant *p_octant, const AABB& p_aabb,const Color& p_light);
	void _throw_ray(const Vector3& p_from, const Vector3& p_to,const Color& p_light,float *p_att_curve,float p_att_curve_len,int p_bounces);


	void throw_rays(int p_amount);
	float get_normalization() const;


	void bake(Node *p_base);


	void clear() {

		if (octree)
			_free_octree(octree);
		if (bvh)
			_free_bvh(bvh);

		if (ray_stack)
			memdelete_arr(ray_stack);
		if (bvh_stack)
			memdelete_arr(bvh_stack);

		octree=NULL;
		bvh=NULL;
		leaf_list=NULL;
		cell_count=0;
		ray_stack=NULL;
		bvh_stack=NULL;
	}

	BakedLightBaker() {
		octree_depth=8;
		octree=NULL;
		bvh=NULL;
		leaf_list=NULL;
		cell_count=0;
		ray_stack=NULL;
		bvh_stack=NULL;
		plot_size=2;
		max_bounces=3;
	}

	~BakedLightBaker() {

		clear();
	}

};


void BakedLightBaker::_add_mesh(const Ref<Mesh>& p_mesh,const Ref<Material>& p_mat_override,const Transform& p_xform) {


	for(int i=0;i<p_mesh->get_surface_count();i++) {

		if (p_mesh->surface_get_primitive_type(i)!=Mesh::PRIMITIVE_TRIANGLES)
			continue;
		Ref<Material> mat = p_mat_override.is_valid()?p_mat_override:p_mesh->surface_get_material(i);

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

		if (p_mesh->surface_get_format(i)&Mesh::ARRAY_FORMAT_INDEX) {

			DVector<int> indices = a[Mesh::ARRAY_INDEX];
			DVector<int>::Read ir = indices.read();

			for(int i=0;i<facecount;i++) {
				Triangle &t=triangles[tbase+i];
				t.vertices[0]=p_xform.xform(vr[ ir[i*3+0] ]);
				t.vertices[1]=p_xform.xform(vr[ ir[i*3+1] ]);
				t.vertices[2]=p_xform.xform(vr[ ir[i*3+2] ]);
			}

		} else {

			for(int i=0;i<facecount;i++) {
				Triangle &t=triangles[tbase+i];
				t.vertices[0]=p_xform.xform(vr[ i*3+0 ]);
				t.vertices[1]=p_xform.xform(vr[ i*3+1 ]);
				t.vertices[2]=p_xform.xform(vr[ i*3+2 ]);
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
	}

	if (p_node->cast_to<DirectionalLight>()) {

		DirectionalLight *dl=p_node->cast_to<DirectionalLight>();

		DirLight dirl;
		dirl.diffuse=dl->get_color(DirectionalLight::COLOR_DIFFUSE);
		dirl.specular=dl->get_color(DirectionalLight::COLOR_SPECULAR);
		dirl.energy=dl->get_parameter(DirectionalLight::PARAM_ENERGY);
		dirl.pos=dl->get_global_transform().origin;
		dirl.up=dl->get_global_transform().basis.get_axis(1).normalized();
		dirl.left=dl->get_global_transform().basis.get_axis(0).normalized();
		dirl.dir=-dl->get_global_transform().basis.get_axis(2).normalized();
		dirl.rays_thrown=0;
		directional_lights.push_back(dirl);

	}

	for(int i=0;i<p_node->get_child_count();i++) {

		_parse_geometry(p_node->get_child(i));
	}
}


void BakedLightBaker::_fix_lights() {


	for(int i=0;i<directional_lights.size();i++) {

		DirLight &dl=directional_lights[i];
		float up_max=-1e10;
		float dir_max=-1e10;
		float left_max=-1e10;
		float up_min=1e10;
		float dir_min=1e10;
		float left_min=1e10;

		for(int j=0;j<triangles.size();j++) {

			for(int k=0;k<3;k++) {

				Vector3 v = triangles[j].vertices[j];

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
		bases[i]->center=bases[i]->aabb.pos+bases[i]->aabb.size*0.5;
	}

	bvh=_parse_bvh(bases.ptr(),bases.size(),1,max_depth);
	ray_stack = memnew_arr(uint32_t,max_depth);
	bvh_stack = memnew_arr(BVH*,max_depth);
}

void BakedLightBaker::_octree_insert(const AABB& p_aabb,Octant *p_octant,Triangle* p_triangle, int p_depth) {

	if (p_octant->leaf) {

		if (p_aabb.has_point(p_triangle->vertices[0]) && p_aabb.has_point(p_triangle->vertices[1]) &&p_aabb.has_point(p_triangle->vertices[2])) {
			//face is completely enclosed, add area
			p_octant->surface_area+=Face3(p_triangle->vertices[0],p_triangle->vertices[1],p_triangle->vertices[2]).get_area();
		} else {
			//not completely enclosed, will need to be clipped..
			Vector<Vector3> poly;
			poly.push_back(p_triangle->vertices[0]);
			poly.push_back(p_triangle->vertices[1]);
			poly.push_back(p_triangle->vertices[2]);

			//clip
			for(int i=0;i<3;i++) {

				//top plane
				Plane p(0,0,0,0);
				p.normal[i]=1.0;
				p.d=p_aabb.pos[i]+p_aabb.size[i];
				poly=Geometry::clip_polygon(poly,p);

				//bottom plane
				p.normal[i]=-1.0;
				p.d=-p_aabb.pos[i];
				poly=Geometry::clip_polygon(poly,p);
			}
			//calculate area
			for(int i=2;i<poly.size();i++) {
				p_octant->surface_area+=Face3(poly[0],poly[i-1],poly[i]).get_area();
			}
		}

	} else {


		for(int i=0;i<8;i++) {

			AABB aabb=p_aabb;
			aabb.size*=0.5;
			if (i&1)
				aabb.pos.x+=aabb.size.x;
			if (i&2)
				aabb.pos.y+=aabb.size.y;
			if (i&4)
				aabb.pos.z+=aabb.size.z;

			AABB fit_aabb=aabb;
			//fit_aabb=fit_aabb.grow(bvh->aabb.size.x*0.0001);

			if (!Face3(p_triangle->vertices[0],p_triangle->vertices[1],p_triangle->vertices[2]).intersects_aabb(fit_aabb))
				continue;

			if (!p_octant->children[i]) {
				p_octant->children[i]=memnew(Octant);
				if (p_depth==0) {
					p_octant->children[i]->leaf=true;
					p_octant->children[i]->light_accum[0]=0;
					p_octant->children[i]->light_accum[1]=0;
					p_octant->children[i]->light_accum[2]=0;
					p_octant->children[i]->offset[0]=aabb.pos.x+aabb.size.x*0.5;
					p_octant->children[i]->offset[1]=aabb.pos.y+aabb.size.y*0.5;
					p_octant->children[i]->offset[2]=aabb.pos.z+aabb.size.z*0.5;
					p_octant->children[i]->surface_area=0;
					p_octant->children[i]->next_leaf=leaf_list;
					leaf_list=p_octant->children[i];
					cell_count++;
				} else {

					p_octant->children[i]->leaf=false;
					for(int j=0;j<8;j++) {
						p_octant->children[i]->children[j]=0;
					}
				}
			}

			_octree_insert(aabb,p_octant->children[i],p_triangle,p_depth-1);
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
	for(int i=0;i<=octree_depth;i++)
		cell_size/=2.0;

	octree = memnew( Octant );
	octree->leaf=false;
	for(int i=0;i<8;i++)
		octree->children[i]=NULL;

	for(int i=0;i<triangles.size();i++) {

		_octree_insert(octree_aabb,octree,&triangles[i],octree_depth-1);
	}

}


void BakedLightBaker::_plot_light(const Vector3& p_plot_pos,const AABB& p_plot_aabb, Octant *p_octant, const AABB& p_aabb,const Color& p_light) {


	if (p_octant->leaf) {

		float r=cell_size*plot_size;
		Vector3 center=p_aabb.pos+p_aabb.size*0.5;
		float d = p_plot_pos.distance_to(center);
		if (d>r)
			return; //oh crap! outside radius
		float intensity = 1.0 - (d/r)*(d/r); //not gauss but..
		p_octant->light_accum[0]+=p_light.r*intensity;
		p_octant->light_accum[1]+=p_light.g*intensity;
		p_octant->light_accum[2]+=p_light.b*intensity;

	} else {

		for(int i=0;i<8;i++) {

			if (!p_octant->children[i])
				continue;

			AABB aabb=p_aabb;
			aabb.size*=0.5;
			if (i&1)
				aabb.pos.x+=aabb.size.x;
			if (i&2)
				aabb.pos.y+=aabb.size.y;
			if (i&4)
				aabb.pos.z+=aabb.size.z;


			if (!aabb.intersects(p_plot_aabb))
				continue;

			_plot_light(p_plot_pos,p_plot_aabb,p_octant->children[i],aabb,p_light);

		}

	}
}


void BakedLightBaker::_throw_ray(const Vector3& p_begin, const Vector3& p_end,const Color& p_light,float *p_att_curve,float p_att_curve_len,int p_bounces) {


	uint32_t* stack = ray_stack;
	BVH **bstack = bvh_stack;

	enum {
		TEST_AABB_BIT=0,
		VISIT_LEFT_BIT=1,
		VISIT_RIGHT_BIT=2,
		VISIT_DONE_BIT=3,


	};

	Vector3 n = (p_end-p_begin).normalized();
	real_t d=1e10;
	bool inters=false;
	Vector3 r_normal;
	Vector3 r_point;

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

					if (f3.intersects_segment(p_begin,p_end,&res)) {


						float nd = n.dot(res);
						if (nd<d) {

							d=nd;
							r_point=res;
							r_normal=f3.get_plane().get_normal();
							inters=true;
						}

					}

					stack[level]=VISIT_DONE_BIT;
				} else {


					bool valid = b.aabb.intersects_segment(p_begin,p_end);
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

		//print_line("collision!");
		if (n.dot(r_normal)>0)
			r_normal=-r_normal;

		//ok...
		Color diffuse_at_point(0.8,0.8,0.8);
		Color specular_at_point(0.8,0.8,0.8);

		AABB aabb;
		aabb.pos=r_point;
		aabb.pos-=Vector3(1,1,1)*cell_size*plot_size;
		aabb.size=Vector3(2,2,2)*cell_size*plot_size;

		_plot_light(r_point,aabb,octree,octree_aabb,p_light);

	}

}






float BakedLightBaker::get_normalization() const {

	float nrg=0;
	for(int i=0;i<directional_lights.size();i++) {

		const DirLight &dl=directional_lights[i];
		float total_area = dl.left.length()*2*dl.up.length()*2;
		float cell_area = cell_size*cell_size;;
		nrg+= dl.energy * (dl.rays_thrown * cell_area / total_area);
		nrg*=5;
	}

	return nrg;
}

void BakedLightBaker::throw_rays(int p_amount) {



	for(int i=0;i<directional_lights.size();i++) {

		DirLight &dl=directional_lights[i];

		float sr = Math::sqrt(p_amount);
		float aspect = dl.up.length()/dl.left.length();


		for(int j=0;j<p_amount;j++) {
			Vector3 from = dl.pos;
			from+=dl.up*(Math::randf()*2.0-1.0);
			from+=dl.left*(Math::randf()*2.0-1.0);
			Vector3 to = from+dl.dir*dl.length;
			Color col=dl.diffuse;
			col.r*=dl.energy;
			col.g*=dl.energy;
			col.b*=dl.energy;
			dl.rays_thrown++;
			_throw_ray(from,to,col,NULL,0,max_bounces);
		}


	}
}













void BakedLightBaker::bake(Node* p_node) {

	cell_count=0;

	_parse_geometry(p_node);
	_fix_lights();
	_make_bvh();
	_make_octree();

}




void BakedLightEditor::_node_removed(Node *p_node) {

	if(p_node==node) {
		node=NULL;
		p_node->remove_child(preview);
		preview->set_mesh(Ref<Mesh>());
		hide();
	}

}



void BakedLightEditor::_menu_option(int p_option) {


	switch(p_option) {


		case MENU_OPTION_BAKE: {

			ERR_FAIL_COND(!node);
			preview->set_mesh(Ref<Mesh>());
			baker->base_inv=node->get_global_transform().affine_inverse();
			baker->bake(node);
			baker->throw_rays(100000);
			float norm =  baker->get_normalization();
			float max_lum=0;

			print_line("CELLS: "+itos(baker->cell_count));
			DVector<Color> colors;
			DVector<Vector3> vertices;
			colors.resize(baker->cell_count*36);
			vertices.resize(baker->cell_count*36);


			{
				DVector<Color>::Write cw=colors.write();
				DVector<Vector3>::Write vw=vertices.write();
				BakedLightBaker::Octant *oct = baker->leaf_list;
				int vert_idx=0;

				while(oct) {

					Color color;
					color.r=oct->light_accum[0]/norm;
					color.g=oct->light_accum[1]/norm;
					color.b=oct->light_accum[2]/norm;
					float lum = color.get_v();
					if (lum>max_lum)
						max_lum=lum;

					for (int i=0;i<6;i++) {


						Vector3 face_points[4];
						for (int j=0;j<4;j++) {

							float v[3];
							v[0]=1.0;
							v[1]=1-2*((j>>1)&1);
							v[2]=v[1]*(1-2*(j&1));

							for (int k=0;k<3;k++) {

								if (i<3)
									face_points[j][(i+k)%3]=v[k]*(i>=3?-1:1);
								else
									face_points[3-j][(i+k)%3]=v[k]*(i>=3?-1:1);
							}
						}

						for(int j=0;j<4;j++) {
							face_points[j]*=baker->cell_size;
							face_points[j]+=Vector3(oct->offset[0],oct->offset[1],oct->offset[2]);
						}

#define ADD_VTX(m_idx) \
	vw[vert_idx]=face_points[m_idx]; \
	cw[vert_idx]=color; \
	vert_idx++;

					//tri 1
						ADD_VTX(0);
						ADD_VTX(1);
						ADD_VTX(2);
					//tri 2
						ADD_VTX(2);
						ADD_VTX(3);
						ADD_VTX(0);

#undef ADD_VTX

					}

					oct=oct->next_leaf;
				}


			}

			print_line("max lum: "+rtos(max_lum));
			Array a;
			a.resize(Mesh::ARRAY_MAX);
			a[Mesh::ARRAY_VERTEX]=vertices;
			a[Mesh::ARRAY_COLOR]=colors;

			Ref<FixedMaterial> matcol = memnew( FixedMaterial );
			matcol->set_fixed_flag(FixedMaterial::FLAG_USE_COLOR_ARRAY,true);
			matcol->set_fixed_flag(FixedMaterial::FLAG_USE_ALPHA,true);
			matcol->set_flag(FixedMaterial::FLAG_UNSHADED,true);
			matcol->set_flag(FixedMaterial::FLAG_DOUBLE_SIDED,true);
			matcol->set_parameter(FixedMaterial::PARAM_DIFFUSE,Color(1,1,1));
			Ref<Mesh> m = memnew( Mesh );
			m->add_surface(Mesh::PRIMITIVE_TRIANGLES,a);
			m->surface_set_material(0,matcol);
			preview->set_mesh(m);





		} break;
		case MENU_OPTION_CLEAR: {



		} break;

	}
}


void BakedLightEditor::edit(BakedLight *p_baked_light) {

	if (node==p_baked_light)
		return;
	if (node) {
		node->remove_child(preview);
	}

	node=p_baked_light;

	if (node)
		node->add_child(preview);

}



void BakedLightEditor::_bind_methods() {

	ObjectTypeDB::bind_method("_menu_option",&BakedLightEditor::_menu_option);
}

BakedLightEditor::BakedLightEditor() {


	options = memnew( MenuButton );

	options->set_text("BakedLight");
	options->get_popup()->add_item("Bake..",MENU_OPTION_BAKE);
	options->get_popup()->add_item("Clear",MENU_OPTION_CLEAR);
	options->get_popup()->connect("item_pressed", this,"_menu_option");


	err_dialog = memnew( AcceptDialog );
	add_child(err_dialog);
	node=NULL;
	baker = memnew( BakedLightBaker );
	preview = memnew( MeshInstance );
}

BakedLightEditor::~BakedLightEditor() {

	memdelete(baker);
}

void BakedLightEditorPlugin::edit(Object *p_object) {

	baked_light_editor->edit(p_object->cast_to<BakedLight>());
}

bool BakedLightEditorPlugin::handles(Object *p_object) const {

	return p_object->is_type("BakedLight");
}

void BakedLightEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		baked_light_editor->show();
		baked_light_editor->options->show();
	} else {

		baked_light_editor->hide();
		baked_light_editor->options->show();
		baked_light_editor->edit(NULL);
		if (baked_light_editor->node) {
			baked_light_editor->node->remove_child(baked_light_editor->preview);
			baked_light_editor->node=NULL;
		}
	}

}

BakedLightEditorPlugin::BakedLightEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	baked_light_editor = memnew( BakedLightEditor );
	editor->get_viewport()->add_child(baked_light_editor);
	add_custom_control(CONTAINER_SPATIAL_EDITOR_MENU,baked_light_editor->options);
	baked_light_editor->hide();
	baked_light_editor->options->hide();
}


BakedLightEditorPlugin::~BakedLightEditorPlugin()
{
}


