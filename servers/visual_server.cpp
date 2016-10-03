/*************************************************************************/
/*  visual_server.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "visual_server.h"
#include "globals.h"
#include "method_bind_ext.inc"

VisualServer *VisualServer::singleton=NULL;
VisualServer* (*VisualServer::create_func)()=NULL;

VisualServer *VisualServer::get_singleton() {

	return singleton;
}



DVector<String> VisualServer::_shader_get_param_list(RID p_shader) const {

//remove at some point

	DVector<String> pl;


#if 0
	List<StringName> params;
	shader_get_param_list(p_shader,&params);


	for(List<StringName>::Element *E=params.front();E;E=E->next()) {

		pl.push_back(E->get());
	}
#endif
	return pl;
}

VisualServer *VisualServer::create() {

	ERR_FAIL_COND_V(singleton,NULL);

	if (create_func)
		return create_func();

	return NULL;
}

RID VisualServer::texture_create_from_image(const Image& p_image,uint32_t p_flags) {

	RID texture = texture_create();
	texture_allocate(texture,p_image.get_width(), p_image.get_height(), p_image.get_format(), p_flags); //if it has mipmaps, use, else generate
	ERR_FAIL_COND_V(!texture.is_valid(),texture);

	texture_set_data(texture, p_image );

	return texture;
}

RID VisualServer::get_test_texture() {

	if (test_texture.is_valid()) {
		return test_texture;
	};

#define TEST_TEXTURE_SIZE 256


	DVector<uint8_t> test_data;
	test_data.resize(TEST_TEXTURE_SIZE*TEST_TEXTURE_SIZE*3);

	{
		DVector<uint8_t>::Write w=test_data.write();

		for (int x=0;x<TEST_TEXTURE_SIZE;x++) {

			for (int y=0;y<TEST_TEXTURE_SIZE;y++) {

				Color c;
				int r=255-(x+y)/2;

				if ((x%(TEST_TEXTURE_SIZE/8))<2 ||(y%(TEST_TEXTURE_SIZE/8))<2) {

					c.r=y;
					c.g=r;
					c.b=x;

				} else {

					c.r=r;
					c.g=x;
					c.b=y;
				}

				w[(y*TEST_TEXTURE_SIZE+x)*3+0]=uint8_t(CLAMP(c.r*255,0,255));
				w[(y*TEST_TEXTURE_SIZE+x)*3+1]=uint8_t(CLAMP(c.g*255,0,255));
				w[(y*TEST_TEXTURE_SIZE+x)*3+2]=uint8_t(CLAMP(c.b*255,0,255));
			}
		}
	}

	Image data(TEST_TEXTURE_SIZE,TEST_TEXTURE_SIZE,false,Image::FORMAT_RGB8,test_data);

	test_texture = texture_create_from_image(data);

	return test_texture;
};

void VisualServer::_free_internal_rids() {

	if (test_texture.is_valid())
		free(test_texture);
	if (white_texture.is_valid())
		free(white_texture);
	if (test_material.is_valid())
		free(test_material);

	for(int i=0;i<16;i++) {
		if (material_2d[i].is_valid())
			free(material_2d[i]);
	}



}

RID VisualServer::_make_test_cube() {

	DVector<Vector3> vertices;
	DVector<Vector3> normals;
	DVector<float> tangents;
	DVector<Vector3> uvs;

	int vtx_idx=0;
#define ADD_VTX(m_idx);\
	vertices.push_back( face_points[m_idx] );\
	normals.push_back( normal_points[m_idx] );\
	tangents.push_back( normal_points[m_idx][1] );\
	tangents.push_back( normal_points[m_idx][2] );\
	tangents.push_back( normal_points[m_idx][0] );\
	tangents.push_back( 1.0 );\
	uvs.push_back( Vector3(uv_points[m_idx*2+0],uv_points[m_idx*2+1],0) );\
	vtx_idx++;\

	for (int i=0;i<6;i++) {


		Vector3 face_points[4];
		Vector3 normal_points[4];
		float uv_points[8]={0,0,0,1,1,1,1,0};

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
			normal_points[j]=Vector3();
			normal_points[j][i%3]=(i>=3?-1:1);
		}

	//tri 1
		ADD_VTX(0);
		ADD_VTX(1);
		ADD_VTX(2);
	//tri 2
		ADD_VTX(2);
		ADD_VTX(3);
		ADD_VTX(0);

	}

	RID test_cube = mesh_create();

	Array d;
	d.resize(VS::ARRAY_MAX);
	d[VisualServer::ARRAY_NORMAL]= normals ;
	d[VisualServer::ARRAY_TANGENT]= tangents ;
	d[VisualServer::ARRAY_TEX_UV]= uvs ;
	d[VisualServer::ARRAY_VERTEX]= vertices ;

	DVector<int> indices;
	indices.resize(vertices.size());
	for(int i=0;i<vertices.size();i++)
		indices.set(i,i);
	d[VisualServer::ARRAY_INDEX]=indices;

	mesh_add_surface_from_arrays( test_cube, PRIMITIVE_TRIANGLES,d );


/*
	test_material = fixed_material_create();
	//material_set_flag(material, MATERIAL_FLAG_BILLBOARD_TOGGLE,true);
	fixed_material_set_texture( test_material, FIXED_MATERIAL_PARAM_DIFFUSE, get_test_texture() );
	fixed_material_set_param( test_material, FIXED_MATERIAL_PARAM_SPECULAR_EXP, 70 );
	fixed_material_set_param( test_material, FIXED_MATERIAL_PARAM_EMISSION, Color(0.2,0.2,0.2) );

	fixed_material_set_param( test_material, FIXED_MATERIAL_PARAM_DIFFUSE, Color(1, 1, 1) );
	fixed_material_set_param( test_material, FIXED_MATERIAL_PARAM_SPECULAR, Color(1,1,1) );
*/
	mesh_surface_set_material(test_cube, 0, test_material );

	return test_cube;
}


RID VisualServer::make_sphere_mesh(int p_lats,int p_lons,float p_radius) {

	DVector<Vector3> vertices;
	DVector<Vector3> normals;

	for(int i = 1; i <= p_lats; i++) {
		double lat0 = Math_PI * (-0.5 + (double) (i - 1) / p_lats);
		double z0  = Math::sin(lat0);
		double zr0 =  Math::cos(lat0);

		double lat1 = Math_PI * (-0.5 + (double) i / p_lats);
		double z1 = Math::sin(lat1);
		double zr1 = Math::cos(lat1);

		for(int j = p_lons; j >= 1; j--) {

			double lng0 = 2 * Math_PI * (double) (j - 1) / p_lons;
			double x0 = Math::cos(lng0);
			double y0 = Math::sin(lng0);

			double lng1 = 2 * Math_PI * (double) (j) / p_lons;
			double x1 = Math::cos(lng1);
			double y1 = Math::sin(lng1);


			Vector3 v[4]={
				Vector3(x1 * zr0, z0, y1 *zr0),
				Vector3(x1 * zr1, z1, y1 *zr1),
				Vector3(x0 * zr1, z1, y0 *zr1),
				Vector3(x0 * zr0, z0, y0 *zr0)
			};

#define ADD_POINT(m_idx)\
	normals.push_back(v[m_idx]);	\
	vertices.push_back(v[m_idx]*p_radius);\

			ADD_POINT(0);
			ADD_POINT(1);
			ADD_POINT(2);

			ADD_POINT(2);
			ADD_POINT(3);
			ADD_POINT(0);
		}
	}

	RID mesh = mesh_create();
	Array d;
	d.resize(VS::ARRAY_MAX);

	d[ARRAY_VERTEX]=vertices;
	d[ARRAY_NORMAL]=normals;

	mesh_add_surface_from_arrays(mesh,PRIMITIVE_TRIANGLES,d);

	return mesh;
}


RID VisualServer::material_2d_get(bool p_shaded, bool p_transparent, bool p_cut_alpha, bool p_opaque_prepass) {

	int version=0;
	if (p_shaded)
		version=1;
	if (p_transparent)
		version|=2;
	if (p_cut_alpha)
		version|=4;
	if (p_opaque_prepass)
		version|=8;
	if (material_2d[version].is_valid())
		return material_2d[version];

	//not valid, make

/*	material_2d[version]=fixed_material_create();
	fixed_material_set_flag(material_2d[version],FIXED_MATERIAL_FLAG_USE_ALPHA,p_transparent);
	fixed_material_set_flag(material_2d[version],FIXED_MATERIAL_FLAG_USE_COLOR_ARRAY,true);
	fixed_material_set_flag(material_2d[version],FIXED_MATERIAL_FLAG_DISCARD_ALPHA,p_cut_alpha);
	material_set_flag(material_2d[version],MATERIAL_FLAG_UNSHADED,!p_shaded);
	material_set_flag(material_2d[version],MATERIAL_FLAG_DOUBLE_SIDED,true);
	material_set_depth_draw_mode(material_2d[version],p_opaque_prepass?MATERIAL_DEPTH_DRAW_OPAQUE_PRE_PASS_ALPHA:MATERIAL_DEPTH_DRAW_OPAQUE_ONLY);
	fixed_material_set_texture(material_2d[version],FIXED_MATERIAL_PARAM_DIFFUSE,get_white_texture());
	//material cut alpha?*/

	return material_2d[version];
}

RID VisualServer::get_white_texture() {

	if (white_texture.is_valid())
		return white_texture;

	DVector<uint8_t> wt;
	wt.resize(16*3);
	{
		DVector<uint8_t>::Write w =wt.write();
		for(int i=0;i<16*3;i++)
			w[i]=255;
	}
	Image white(4,4,0,Image::FORMAT_RGB8,wt);
	white_texture=texture_create();
	texture_allocate(white_texture,4,4,Image::FORMAT_RGB8);
	texture_set_data(white_texture,white);
	return white_texture;

}


void VisualServer::mesh_add_surface_from_arrays(RID p_mesh,PrimitiveType p_primitive,const Array& p_arrays,const Array& p_blend_shapes,uint32_t p_compress_format) {


}

void VisualServer::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("texture_create"),&VisualServer::texture_create);
	ObjectTypeDB::bind_method(_MD("texture_create_from_image"),&VisualServer::texture_create_from_image,DEFVAL( TEXTURE_FLAGS_DEFAULT ) );
	//ObjectTypeDB::bind_method(_MD("texture_allocate"),&VisualServer::texture_allocate,DEFVAL( TEXTURE_FLAGS_DEFAULT ) );
	//ObjectTypeDB::bind_method(_MD("texture_set_data"),&VisualServer::texture_blit_rect,DEFVAL( CUBEMAP_LEFT ) );
	//ObjectTypeDB::bind_method(_MD("texture_get_rect"),&VisualServer::texture_get_rect );
	ObjectTypeDB::bind_method(_MD("texture_set_flags"),&VisualServer::texture_set_flags );
	ObjectTypeDB::bind_method(_MD("texture_get_flags"),&VisualServer::texture_get_flags );
	ObjectTypeDB::bind_method(_MD("texture_get_width"),&VisualServer::texture_get_width );
	ObjectTypeDB::bind_method(_MD("texture_get_height"),&VisualServer::texture_get_height );

	ObjectTypeDB::bind_method(_MD("texture_set_shrink_all_x2_on_set_data","shrink"),&VisualServer::texture_set_shrink_all_x2_on_set_data );




}

void VisualServer::_canvas_item_add_style_box(RID p_item, const Rect2& p_rect, const Rect2& p_source, RID p_texture,const Vector<float>& p_margins, const Color& p_modulate) {

	ERR_FAIL_COND(p_margins.size()!=4);
	//canvas_item_add_style_box(p_item,p_rect,p_source,p_texture,Vector2(p_margins[0],p_margins[1]),Vector2(p_margins[2],p_margins[3]),true,p_modulate);
}

void VisualServer::_camera_set_orthogonal(RID p_camera,float p_size,float p_z_near,float p_z_far) {

	camera_set_orthogonal(p_camera,p_size,p_z_near,p_z_far);
}





void VisualServer::mesh_add_surface_from_mesh_data( RID p_mesh, const Geometry::MeshData& p_mesh_data) {

#if 1
	DVector<Vector3> vertices;
	DVector<Vector3> normals;

	for (int i=0;i<p_mesh_data.faces.size();i++) {

		const Geometry::MeshData::Face& f = p_mesh_data.faces[i];

		for (int j=2;j<f.indices.size();j++) {

#define _ADD_VERTEX(m_idx)\
	vertices.push_back( p_mesh_data.vertices[ f.indices[m_idx] ] );\
	normals.push_back( f.plane.normal );

			_ADD_VERTEX( 0 );
			_ADD_VERTEX( j-1 );
			_ADD_VERTEX( j );
		}
	}

	Array d;
	d.resize(VS::ARRAY_MAX);
	d[ARRAY_VERTEX]=vertices;
	d[ARRAY_NORMAL]=normals;
	mesh_add_surface_from_arrays(p_mesh,PRIMITIVE_TRIANGLES, d);

#else


	DVector<Vector3> vertices;



	for (int i=0;i<p_mesh_data.edges.size();i++) {

		const Geometry::MeshData::Edge& f = p_mesh_data.edges[i];
		vertices.push_back(p_mesh_data.vertices[ f.a]);
		vertices.push_back(p_mesh_data.vertices[ f.b]);
	}

	Array d;
	d.resize(VS::ARRAY_MAX);
	d[ARRAY_VERTEX]=vertices;
	mesh_add_surface(p_mesh,PRIMITIVE_LINES, d);




#endif

}

void VisualServer::mesh_add_surface_from_planes( RID p_mesh, const DVector<Plane>& p_planes) {


	Geometry::MeshData mdata = Geometry::build_convex_mesh(p_planes);
	mesh_add_surface_from_mesh_data(p_mesh,mdata);

}

RID VisualServer::instance_create2(RID p_base, RID p_scenario) {

	RID instance = instance_create();
	instance_set_base(instance,p_base);
	instance_set_scenario(instance,p_scenario);
	return instance;
}


VisualServer::VisualServer() {

//	ERR_FAIL_COND(singleton);
	singleton=this;

}


VisualServer::~VisualServer() {

	singleton=NULL;
}
