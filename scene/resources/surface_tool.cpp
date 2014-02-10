/*************************************************************************/
/*  surface_tool.cpp                                                     */
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
#include "surface_tool.h"

#define _VERTEX_SNAP 0.0001
#define EQ_VERTEX_DIST 0.00001



bool SurfaceTool::compare(const Vertex& p_a,const Vertex& p_b) const {

	if (p_a.vertex.distance_to(p_b.vertex)>EQ_VERTEX_DIST)
		return false;

	if (format&Mesh::ARRAY_FORMAT_TEX_UV) {

		if (p_a.uv.distance_to(p_b.uv)>EQ_VERTEX_DIST)
			return false;
	}

	if (format&Mesh::ARRAY_FORMAT_TEX_UV2) {

		if (p_a.uv2.distance_to(p_b.uv2)>EQ_VERTEX_DIST)
			return false;
	}

	if (format&Mesh::ARRAY_FORMAT_NORMAL) {
		if (p_a.normal.distance_to(p_b.normal)>EQ_VERTEX_DIST)
			return false;
	}

	if (format&Mesh::ARRAY_FORMAT_TANGENT) {
		if (p_a.binormal.distance_to(p_b.binormal)>EQ_VERTEX_DIST)
			return false;
		if (p_a.tangent.distance_to(p_b.tangent)>EQ_VERTEX_DIST)
			return false;
	}

	if (format&Mesh::ARRAY_FORMAT_COLOR) {
		if (p_a.color!=p_b.color)
			return false;
	}

	if (format&Mesh::ARRAY_FORMAT_BONES) {
		for(int i=0;i<4;i++) {
			if (Math::abs(p_a.bones[i]-p_b.bones[i])>CMP_EPSILON)
				return false;
		}
	}

	if (format&Mesh::ARRAY_FORMAT_WEIGHTS) {
		for(int i=0;i<4;i++) {
			if (Math::abs(p_a.weights[i]-p_b.weights[i])>CMP_EPSILON)
				return false;
		}
	}


	return true;
}

void SurfaceTool::begin(Mesh::PrimitiveType p_primitive) {

	clear();

	primitive=p_primitive;
	begun=true;
	first=true;
}

void SurfaceTool::add_vertex( const Vector3& p_vertex) {

	ERR_FAIL_COND(!begun);

	Vertex vtx;
	vtx.vertex=p_vertex;
	vtx.color=last_color;
	vtx.normal=last_normal;
	vtx.uv=last_uv;
	vtx.weights=last_weights;
	vtx.bones=last_bones;
	vtx.tangent=last_tangent.normal;
	vtx.binormal=last_tangent.normal.cross(last_normal).normalized() * last_tangent.d;
	vertex_array.push_back(vtx);
	first=false;
	format|=Mesh::ARRAY_FORMAT_VERTEX;

}
void SurfaceTool::add_color( Color p_color ) {

	ERR_FAIL_COND(!begun);

	ERR_FAIL_COND( !first && !(format&Mesh::ARRAY_FORMAT_COLOR));

	format|=Mesh::ARRAY_FORMAT_COLOR;
	last_color=p_color;
}
void SurfaceTool::add_normal( const Vector3& p_normal) {

	ERR_FAIL_COND(!begun);

	ERR_FAIL_COND( !first && !(format&Mesh::ARRAY_FORMAT_NORMAL));

	format|=Mesh::ARRAY_FORMAT_NORMAL;
	last_normal=p_normal;
}


void SurfaceTool::add_tangent( const Plane& p_tangent ) {

	ERR_FAIL_COND(!begun);
	ERR_FAIL_COND( !first && !(format&Mesh::ARRAY_FORMAT_TANGENT));

	format|=Mesh::ARRAY_FORMAT_TANGENT;
	last_tangent=p_tangent;


}


void SurfaceTool::add_uv( const Vector2& p_uv) {

	ERR_FAIL_COND(!begun);
	ERR_FAIL_COND( !first && !(format&Mesh::ARRAY_FORMAT_TEX_UV));

	format|=Mesh::ARRAY_FORMAT_TEX_UV;
	last_uv=p_uv;

}

void SurfaceTool::add_uv2( const Vector2& p_uv2) {

	ERR_FAIL_COND(!begun);
	ERR_FAIL_COND( !first && !(format&Mesh::ARRAY_FORMAT_TEX_UV2));

	format|=Mesh::ARRAY_FORMAT_TEX_UV2;
	last_uv2=p_uv2;

}

void SurfaceTool::add_bones( const Vector<int>& p_bones) {

	ERR_FAIL_COND(!begun);
	ERR_FAIL_COND(p_bones.size()!=4);
	ERR_FAIL_COND( !first && !(format&Mesh::ARRAY_FORMAT_BONES));

	format|=Mesh::ARRAY_FORMAT_BONES;
	last_bones=p_bones;
}

void SurfaceTool::add_weights( const Vector<float>& p_weights) {

	ERR_FAIL_COND(!begun);

	ERR_FAIL_COND(p_weights.size()!=4);
	ERR_FAIL_COND( !first && !(format&Mesh::ARRAY_FORMAT_WEIGHTS));

	format|=Mesh::ARRAY_FORMAT_WEIGHTS;
	last_weights=p_weights;

}


void SurfaceTool::add_index( int p_index) {

	ERR_FAIL_COND(!begun);

	format|=Mesh::ARRAY_FORMAT_INDEX;
	index_array.push_back(p_index);
}

Ref<Mesh> SurfaceTool::commit(const Ref<Mesh>& p_existing) {


	Ref<Mesh> mesh;
	if (p_existing.is_valid())
		mesh=p_existing;
	else
		mesh= Ref<Mesh>( memnew( Mesh ) );

	int varr_len=vertex_array.size();


	if (varr_len==0)
		return mesh;

	int surface = mesh->get_surface_count();

	Array a;
	a.resize(Mesh::ARRAY_MAX);

	for (int i=0;i<Mesh::ARRAY_MAX;i++) {

		switch(format&(1<<i)) {

			case Mesh::ARRAY_FORMAT_VERTEX:
			case Mesh::ARRAY_FORMAT_NORMAL: {

				DVector<Vector3> array;
				array.resize(varr_len);
				DVector<Vector3>::Write w = array.write();

				int idx=0;
				for(List< Vertex >::Element *E=vertex_array.front();E;E=E->next(),idx++) {

					const Vertex &v=E->get();

					switch(i) {
						case Mesh::ARRAY_VERTEX: {
							w[idx]=v.vertex;
						} break;
						case Mesh::ARRAY_NORMAL: {
							w[idx]=v.normal;
						} break;
					}

				}

				w=DVector<Vector3>::Write();
				a[i]=array;

			} break;

			case Mesh::ARRAY_FORMAT_TEX_UV:
			case Mesh::ARRAY_FORMAT_TEX_UV2: {

				DVector<Vector2> array;
				array.resize(varr_len);
				DVector<Vector2>::Write w = array.write();

				int idx=0;
				for(List< Vertex >::Element *E=vertex_array.front();E;E=E->next(),idx++) {

					const Vertex &v=E->get();

					switch(i) {

						case Mesh::ARRAY_TEX_UV: {
							w[idx]=v.uv;
						} break;
						case Mesh::ARRAY_TEX_UV2: {
							w[idx]=v.uv2;
						} break;
					}

				}

				w=DVector<Vector2>::Write();
				a[i]=array;
			} break;
			case Mesh::ARRAY_FORMAT_TANGENT: {


				DVector<float> array;
				array.resize(varr_len*4);
				DVector<float>::Write w = array.write();

				int idx=0;
				for(List< Vertex >::Element *E=vertex_array.front();E;E=E->next(),idx+=4) {

					const Vertex &v=E->get();

					w[idx+0]=v.tangent.x;
					w[idx+1]=v.tangent.y;
					w[idx+2]=v.tangent.z;
					float d = v.binormal.dot(v.normal.cross(v.tangent));
					w[idx+3]=d<0 ? -1 : 1;
				}

				w=DVector<float>::Write();
				a[i]=array;

			} break;
			case Mesh::ARRAY_FORMAT_COLOR:  {

				DVector<Color> array;
				array.resize(varr_len);
				DVector<Color>::Write w = array.write();

				int idx=0;
				for(List< Vertex >::Element *E=vertex_array.front();E;E=E->next(),idx++) {

					const Vertex &v=E->get();
					w[idx]=v.color;
				}

				w=DVector<Color>::Write();
				a[i]=array;
			} break;	
			case Mesh::ARRAY_FORMAT_BONES:
			case Mesh::ARRAY_FORMAT_WEIGHTS: {


				DVector<float> array;
				array.resize(varr_len*4);
				DVector<float>::Write w = array.write();

				int idx=0;
				for(List< Vertex >::Element *E=vertex_array.front();E;E=E->next(),idx+=4) {

					const Vertex &v=E->get();

					for(int j=0;j<4;j++) {
						switch(i) {
							case Mesh::ARRAY_WEIGHTS: {
								ERR_CONTINUE( v.weights.size()!=4 );
								w[idx+j]=v.weights[j];
							} break;
							case Mesh::ARRAY_BONES: {
								ERR_CONTINUE( v.bones.size()!=4 );
								w[idx+j]=v.bones[j];
							} break;
						}
					}

				}

				w=DVector<float>::Write();
				a[i]=array;

			} break;
			case Mesh::ARRAY_FORMAT_INDEX: {

				ERR_CONTINUE( index_array.size() ==0 );

				DVector<int> array;
				array.resize(index_array.size());
				DVector<int>::Write w = array.write();

				int idx=0;
				for(List< int>::Element *E=index_array.front();E;E=E->next(),idx++) {

					w[idx]=E->get();
				}

				w=DVector<int>::Write();
				a[i]=array;
			} break;

			default: {}
		}

	}

	mesh->add_surface(primitive,a);
	if (material.is_valid())
		mesh->surface_set_material(surface,material);

	return mesh;
}

void SurfaceTool::index() {

#if 0
	printf("indexing..\n");
	ERR_FAIL_COND( format & Surface::ARRAY_FORMAT_INDEX ); // already indexed

	index_array.clear();
	DVector< Vertex > indexed_vertex_array;

	int vertex_array_len = vertex_array.size();
	vertex_array.read_lock();
	const Vertex*vertex_array_ptr = vertex_array.read();

	for (int i=0;i<vertex_array_len;i++) {

		int index_pos=-1;

		int indexed_vertex_array_len=indexed_vertex_array.size();

		if (indexed_vertex_array_len) {

			indexed_vertex_array.read_lock();
			const Vertex* indexed_vertex_array_ptr=indexed_vertex_array.read();

			for (int j=0;j<indexed_vertex_array_len;j++) {

				if (vertex_array_ptr[i].same_as(indexed_vertex_array_ptr[j])) {

					index_pos=j;
					break;
				}
			}

			indexed_vertex_array.read_unlock();
		}

		if (index_pos==-1) {

			index_pos=indexed_vertex_array.size();
			indexed_vertex_array.push_back(vertex_array_ptr[i]);
		} else {

			indexed_vertex_array.write_lock();
			indexed_vertex_array.write()[index_pos].normal+=vertex_array_ptr[i].normal;
			indexed_vertex_array.write()[index_pos].binormal+=vertex_array_ptr[i].binormal;
			indexed_vertex_array.write()[index_pos].tangent+=vertex_array_ptr[i].tangent;
			indexed_vertex_array.write_unlock();
		}

		index_array.push_back(index_pos);
	}

	int idxvertsize=indexed_vertex_array.size();
	indexed_vertex_array.write_lock();
	Vertex* idxvert=indexed_vertex_array.write();
	for (int i=0;i<idxvertsize;i++) {

		idxvert[i].normal.normalize();
		idxvert[i].tangent.normalize();
		idxvert[i].binormal.normalize();
	}
	indexed_vertex_array.write_unlock();

	vertex_array.read_unlock();

	format|=Surface::ARRAY_FORMAT_INDEX;
	vertex_array=indexed_vertex_array;

	printf("indexing.. end\n");
#endif
}

void SurfaceTool::deindex() {


}


void SurfaceTool::_create_list(const Ref<Mesh>& p_existing, int p_surface, List<Vertex> *r_vertex, List<int> *r_index, int& lformat) {

	Array arr = p_existing->surface_get_arrays(p_surface);
	ERR_FAIL_COND( arr.size() !=VS::ARRAY_MAX );

	DVector<Vector3> varr = arr[VS::ARRAY_VERTEX];
	DVector<Vector3> narr = arr[VS::ARRAY_NORMAL];
	DVector<float> tarr = arr[VS::ARRAY_TANGENT];
	DVector<Color> carr = arr[VS::ARRAY_COLOR];
	DVector<Vector2> uvarr = arr[VS::ARRAY_TEX_UV];
	DVector<Vector2> uv2arr = arr[VS::ARRAY_TEX_UV2];
	DVector<int> barr = arr[VS::ARRAY_BONES];
	DVector<float> warr = arr[VS::ARRAY_WEIGHTS];

	int vc = varr.size();

	if (vc==0)
		return;
	lformat=0;

	DVector<Vector3>::Read rv;
	if (varr.size()) {
		lformat|=VS::ARRAY_FORMAT_VERTEX;
		rv=varr.read();
	}
	DVector<Vector3>::Read rn;
	if (narr.size()) {
		lformat|=VS::ARRAY_FORMAT_NORMAL;
		rn=narr.read();
	}
	DVector<float>::Read rt;
	if (tarr.size()) {
		lformat|=VS::ARRAY_FORMAT_TANGENT;
		rt=tarr.read();
	}
	DVector<Color>::Read rc;
	if (carr.size()) {
		lformat|=VS::ARRAY_FORMAT_COLOR;
		rc=carr.read();
	}

	DVector<Vector2>::Read ruv;
	if (uvarr.size()) {
		lformat|=VS::ARRAY_FORMAT_TEX_UV;
		ruv=uvarr.read();
	}

	DVector<Vector2>::Read ruv2;
	if (uv2arr.size()) {
		lformat|=VS::ARRAY_FORMAT_TEX_UV2;
		ruv2=uv2arr.read();
	}

	DVector<int>::Read rb;
	if (barr.size()) {
		lformat|=VS::ARRAY_FORMAT_BONES;
		rb=barr.read();
	}

	DVector<float>::Read rw;
	if (warr.size()) {
		lformat|=VS::ARRAY_FORMAT_WEIGHTS;
		rw=warr.read();
	}

	for(int i=0;i<vc;i++) {

		Vertex v;
		if (lformat&VS::ARRAY_FORMAT_VERTEX)
			v.vertex=varr[i];
		if (lformat&VS::ARRAY_FORMAT_NORMAL)
			v.normal=narr[i];
		if (lformat&VS::ARRAY_FORMAT_TANGENT) {
			Plane p( tarr[i*4+0],  tarr[i*4+1],  tarr[i*4+2],  tarr[i*4+3] );
			v.tangent=p.normal;
			v.binormal=p.normal.cross(last_normal).normalized() * p.d;
		}
		if (lformat&VS::ARRAY_FORMAT_COLOR)
			v.color=carr[i];
		if (lformat&VS::ARRAY_FORMAT_TEX_UV)
			v.uv=uvarr[i];
		if (lformat&VS::ARRAY_FORMAT_TEX_UV2)
			v.uv2=uv2arr[i];
		if (lformat&VS::ARRAY_FORMAT_BONES) {
			Vector<int> b;
			b.resize(4);
			b[0]=barr[i*4+0];
			b[1]=barr[i*4+1];
			b[2]=barr[i*4+2];
			b[3]=barr[i*4+3];
			v.bones=b;
		}
		if (lformat&VS::ARRAY_FORMAT_WEIGHTS) {
			Vector<float> w;
			w.resize(4);
			w[0]=warr[i*4+0];
			w[1]=warr[i*4+1];
			w[2]=warr[i*4+2];
			w[3]=warr[i*4+3];
			v.weights=w;
		}

		r_vertex->push_back(v);
	}

	//indices

	DVector<int> idx= arr[VS::ARRAY_INDEX];
	int is = idx.size();
	if (is) {

		lformat|=VS::ARRAY_FORMAT_INDEX;
		DVector<int>::Read iarr=idx.read();
		for(int i=0;i<is;i++) {
			r_index->push_back(iarr[i]);
		}

	}


}


void SurfaceTool::create_from(const Ref<Mesh>& p_existing, int p_surface) {

	clear();
	primitive=p_existing->surface_get_primitive_type(p_surface);
	_create_list(p_existing,p_surface,&vertex_array,&index_array,format);

}

void SurfaceTool::append_from(const Ref<Mesh>& p_existing, int p_surface,const Transform& p_xform) {

	if (vertex_array.size()==0) {
		primitive=p_existing->surface_get_primitive_type(p_surface);
		format=0;
	}

	int nformat;
	List<Vertex> nvertices;
	List<int> nindices;
	_create_list(p_existing,p_surface,&nvertices,&nindices,nformat);
	format|=nformat;
	int vfrom = vertex_array.size();


	for(List<Vertex>::Element *E=nvertices.front();E;E=E->next()) {

		Vertex v=E->get();
		v.vertex=p_xform.xform(v.vertex);
		if (nformat&VS::ARRAY_FORMAT_NORMAL) {
			v.normal=p_xform.basis.xform(v.normal);
		}
		if (nformat&VS::ARRAY_FORMAT_TANGENT) {
			v.tangent=p_xform.basis.xform(v.tangent);
			v.binormal=p_xform.basis.xform(v.binormal);
		}

		vertex_array.push_back(v);
	}

	for(List<int>::Element *E=nindices.front();E;E=E->next()) {

		int dst_index = E->get()+vfrom;
		//if (dst_index <0 || dst_index>=vertex_array.size()) {
		//	print_line("invalid index!");
		//}
		index_array.push_back(dst_index);
	}
	if (index_array.size()%3)
		print_line("IA not div of 3?");

}


void SurfaceTool::generate_tangents() {

	ERR_FAIL_COND(!(format&Mesh::ARRAY_FORMAT_TEX_UV));

#if 0

	int len=vertex_array.size();
	vertex_array.write_lock();
	Vertex *vertexptr=vertex_array.write();

	for (int i=0;i<len/3;i++) {


		Vector3 v1 = vertexptr[i*3+0].vertex;
		Vector3 v2 = vertexptr[i*3+1].vertex;
		Vector3 v3 = vertexptr[i*3+2].vertex;

		Vector3 w1 = vertexptr[i*3+0].uv[0];
		Vector3 w2 = vertexptr[i*3+1].uv[0];
		Vector3 w3 = vertexptr[i*3+2].uv[0];


		float x1 = v2.x - v1.x;
		float x2 = v3.x - v1.x;
		float y1 = v2.y - v1.y;
		float y2 = v3.y - v1.y;
		float z1 = v2.z - v1.z;
		float z2 = v3.z - v1.z;

		float s1 = w2.x - w1.x;
		float s2 = w3.x - w1.x;
		float t1 = w2.y - w1.y;
		float t2 = w3.y - w1.y;

		float r  = (s1 * t2 - s2 * t1);

		Vector3 binormal,tangent;

		if (r==0) {
			binormal=Vector3(0,0,0);
			tangent=Vector3(0,0,0);
		} else {
			tangent = Vector3((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r,
			(t2 * z1 - t1 * z2) * r);
			binormal = Vector3((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r,
			(s1 * z2 - s2 * z1) * r);
		}

		tangent.normalize();
		binormal.normalize();
		Vector3 normal=Plane( v1, v2, v3 ).normal;

		Vector3 tangentp = tangent - normal * normal.dot( tangent );
		Vector3 binormalp = binormal - normal * (normal.dot(binormal)) - tangent * (tangent.dot(binormal));

		tangentp.normalize();
		binormalp.normalize();


		for (int j=0;j<3;j++) {
			vertexptr[i*3+j].normal=normal;
			vertexptr[i*3+j].binormal=binormalp;
			vertexptr[i*3+j].tangent=tangentp;
		}
	}

	format|=Surface::ARRAY_FORMAT_TANGENT;
	printf("adding tangents to the format\n");

	vertex_array.write_unlock();
#endif
}

void SurfaceTool::generate_flat_normals() {

}
void SurfaceTool::generate_smooth_normals() {

}

void SurfaceTool::set_material(const Ref<Material>& p_material) {

	material=p_material;
}

void SurfaceTool::clear() {

	begun=false;
	primitive=Mesh::PRIMITIVE_LINES;
	format=0;
	last_bones.clear();;
	last_weights.clear();
	index_array.clear();
	vertex_array.clear();

}

void SurfaceTool::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("begin","primitive"),&SurfaceTool::begin);
	ObjectTypeDB::bind_method(_MD("add_vertex","vertex"),&SurfaceTool::add_vertex);
	ObjectTypeDB::bind_method(_MD("add_color","color"),&SurfaceTool::add_color);
	ObjectTypeDB::bind_method(_MD("add_normal","normal"),&SurfaceTool::add_normal);
	ObjectTypeDB::bind_method(_MD("add_tangent","tangent"),&SurfaceTool::add_tangent);
	ObjectTypeDB::bind_method(_MD("add_uv","uv"),&SurfaceTool::add_uv);
	ObjectTypeDB::bind_method(_MD("add_uv2","uv2"),&SurfaceTool::add_uv2);
	ObjectTypeDB::bind_method(_MD("add_bones","bones"),&SurfaceTool::add_bones);
	ObjectTypeDB::bind_method(_MD("add_weights","weights"),&SurfaceTool::add_weights);
	ObjectTypeDB::bind_method(_MD("set_material","material:Material"),&SurfaceTool::set_material);
	ObjectTypeDB::bind_method(_MD("index"),&SurfaceTool::index);
	ObjectTypeDB::bind_method(_MD("deindex"),&SurfaceTool::deindex);
	ObjectTypeDB::bind_method(_MD("generate_flat_normals"),&SurfaceTool::generate_flat_normals);
	ObjectTypeDB::bind_method(_MD("generate_smooth_normals"),&SurfaceTool::generate_smooth_normals);
	ObjectTypeDB::bind_method(_MD("generate_tangents"),&SurfaceTool::generate_tangents);
	ObjectTypeDB::bind_method(_MD("commit:Mesh","existing:Mesh"),&SurfaceTool::commit,DEFVAL( RefPtr() ));
	ObjectTypeDB::bind_method(_MD("clear"),&SurfaceTool::clear);

}


SurfaceTool::SurfaceTool() {

	first=false;
	begun=false;
	primitive=Mesh::PRIMITIVE_LINES;
	format=0;

}

