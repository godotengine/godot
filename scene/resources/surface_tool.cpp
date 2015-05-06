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


bool SurfaceTool::Vertex::operator==(const Vertex& p_b) const {


	if (vertex!=p_b.vertex)
		return false;

	if (uv!=p_b.uv)
		return false;

	if (uv2!=p_b.uv2)
		return false;

	if (normal!=p_b.normal)
		return false;

	if (binormal!=p_b.binormal)
		return false;

	if (color!=p_b.color)
		return false;

	if (bones.size()!=p_b.bones.size())
		return false;

	for(int i=0;i<bones.size();i++) {
		if (bones[i]!=p_b.bones[i])
			return false;
	}

	for(int i=0;i<weights.size();i++) {
		if (weights[i]!=p_b.weights[i])
			return false;
	}

	return true;
}


uint32_t SurfaceTool::VertexHasher::hash(const Vertex &p_vtx) {

	uint32_t h = hash_djb2_buffer((const uint8_t*)&p_vtx.vertex,sizeof(real_t)*3);
	h = hash_djb2_buffer((const uint8_t*)&p_vtx.normal,sizeof(real_t)*3,h);
	h = hash_djb2_buffer((const uint8_t*)&p_vtx.binormal,sizeof(real_t)*3,h);
	h = hash_djb2_buffer((const uint8_t*)&p_vtx.tangent,sizeof(real_t)*3,h);
	h = hash_djb2_buffer((const uint8_t*)&p_vtx.uv,sizeof(real_t)*2,h);
	h = hash_djb2_buffer((const uint8_t*)&p_vtx.uv2,sizeof(real_t)*2,h);
	h = hash_djb2_buffer((const uint8_t*)&p_vtx.color,sizeof(real_t)*4,h);
	h = hash_djb2_buffer((const uint8_t*)p_vtx.bones.ptr(),p_vtx.bones.size()*sizeof(int),h);
	h = hash_djb2_buffer((const uint8_t*)p_vtx.weights.ptr(),p_vtx.weights.size()*sizeof(float),h);
	return h;
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

void SurfaceTool::add_smooth_group(bool p_smooth) {

	ERR_FAIL_COND(!begun);
	if (index_array.size()) {
		smooth_groups[index_array.size()]=p_smooth;
	} else {

		smooth_groups[vertex_array.size()]=p_smooth;
	}
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

	if (index_array.size())
		return; //already indexed


	HashMap<Vertex,int,VertexHasher> indices;
	List<Vertex> new_vertices;

	for(List< Vertex >::Element *E=vertex_array.front();E;E=E->next()) {

		int *idxptr=indices.getptr(E->get());
		int idx;
		if (!idxptr) {
			idx=indices.size();
			new_vertices.push_back(E->get());
			indices[E->get()]=idx;
		} else {
			idx=*idxptr;
		}

		index_array.push_back(idx);

	}

	vertex_array.clear();
	vertex_array=new_vertices;

	format|=Mesh::ARRAY_FORMAT_INDEX;
}

void SurfaceTool::deindex() {

	if (index_array.size()==0)
		return; //nothing to deindex
	Vector< Vertex > varr;
	varr.resize(vertex_array.size());
	int idx=0;
	for (List< Vertex >::Element *E=vertex_array.front();E;E=E->next()) {

		varr[idx++]=E->get();
	}
	vertex_array.clear();
	for (List<int>::Element *E=index_array.front();E;E=E->next()) {

		ERR_FAIL_INDEX(E->get(),varr.size());
		vertex_array.push_back(varr[E->get()]);
	}
	format&=~Mesh::ARRAY_FORMAT_INDEX;
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
	ERR_FAIL_COND(!(format&Mesh::ARRAY_FORMAT_NORMAL));


	if (index_array.size()) {

		Vector<List<Vertex>::Element*> vtx;
		vtx.resize(vertex_array.size());
		int idx=0;
		for (List<Vertex>::Element *E=vertex_array.front();E;E=E->next()) {
			vtx[idx++]=E;
			E->get().binormal=Vector3();
			E->get().tangent=Vector3();
		}

		for (List<int>::Element *E=index_array.front();E;) {

			int i[3];
			i[0]=E->get();
			E=E->next();
			ERR_FAIL_COND(!E);
			i[1]=E->get();
			E=E->next();
			ERR_FAIL_COND(!E);
			i[2]=E->get();
			E=E->next();
			ERR_FAIL_COND(!E);


			Vector3 v1 = vtx[ i[0] ]->get().vertex;
			Vector3 v2 = vtx[ i[1] ]->get().vertex;
			Vector3 v3 = vtx[ i[2] ]->get().vertex;

			Vector2 w1 = vtx[ i[0] ]->get().uv;
			Vector2 w2 = vtx[ i[1] ]->get().uv;
			Vector2 w3 = vtx[ i[2] ]->get().uv;


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
				vtx[ i[j] ]->get().binormal+=binormalp;
				vtx[ i[j] ]->get().tangent+=tangentp;

			}
		}

		for (List<Vertex>::Element *E=vertex_array.front();E;E=E->next()) {
			E->get().binormal.normalize();
			E->get().tangent.normalize();
		}


	} else {


		for (List<Vertex>::Element *E=vertex_array.front();E;) {

			List< Vertex >::Element *v[3];
			v[0]=E;
			v[1]=v[0]->next();
			ERR_FAIL_COND(!v[1]);
			v[2]=v[1]->next();
			ERR_FAIL_COND(!v[2]);
			E=v[2]->next();

			Vector3 v1 = v[0]->get().vertex;
			Vector3 v2 = v[1]->get().vertex;
			Vector3 v3 = v[2]->get().vertex;

			Vector2 w1 = v[0]->get().uv;
			Vector2 w2 = v[1]->get().uv;
			Vector2 w3 = v[2]->get().uv;


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
				v[j]->get().binormal=binormalp;
				v[j]->get().tangent=tangentp;

			}
		}
	}

	format|=Mesh::ARRAY_FORMAT_TANGENT;

}

void SurfaceTool::generate_normals() {

	ERR_FAIL_COND(primitive!=Mesh::PRIMITIVE_TRIANGLES);

	bool was_indexed=index_array.size();

	deindex();

	HashMap<Vertex,Vector3,VertexHasher> vertex_hash;

	int count=0;
	bool smooth=false;
	if (smooth_groups.has(0))
		smooth=smooth_groups[0];

	print_line("SMOOTH BEGIN? "+itos(smooth));

	List< Vertex >::Element *B=vertex_array.front();
	for(List< Vertex >::Element *E=B;E;) {

		List< Vertex >::Element *v[3];
		v[0]=E;
		v[1]=v[0]->next();
		ERR_FAIL_COND(!v[1]);
		v[2]=v[1]->next();
		ERR_FAIL_COND(!v[2]);
		E=v[2]->next();

		Vector3 normal = Plane(v[0]->get().vertex,v[1]->get().vertex,v[2]->get().vertex).normal;

		if (smooth) {

			for(int i=0;i<3;i++) {

				Vector3 *lv=vertex_hash.getptr(v[i]->get());
				if (!lv) {
					vertex_hash.set(v[i]->get(),normal);
				} else {
					(*lv)+=normal;
				}
			}
		} else {

			for(int i=0;i<3;i++) {

				v[i]->get().normal=normal;

			}
		}
		count+=3;

		if (smooth_groups.has(count) || !E) {

			if (vertex_hash.size()) {

				while (B!=E) {


					Vector3* lv=vertex_hash.getptr(B->get());
					if (lv) {
						B->get().normal=lv->normalized();
					}

					B=B->next();
				}

			} else {
				B=E;
			}

			vertex_hash.clear();
			if (E) {
				smooth=smooth_groups[count];
				print_line("SMOOTH AT "+itos(count)+": "+itos(smooth));

			}
		}

	}

	format|=Mesh::ARRAY_FORMAT_NORMAL;

	if (was_indexed) {
		index();
		smooth_groups.clear();
	}

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
	smooth_groups.clear();

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
	ObjectTypeDB::bind_method(_MD("add_smooth_group","smooth"),&SurfaceTool::add_smooth_group);
	ObjectTypeDB::bind_method(_MD("set_material","material:Material"),&SurfaceTool::set_material);
	ObjectTypeDB::bind_method(_MD("index"),&SurfaceTool::index);
	ObjectTypeDB::bind_method(_MD("deindex"),&SurfaceTool::deindex);
	///ObjectTypeDB::bind_method(_MD("generate_flat_normals"),&SurfaceTool::generate_flat_normals);
	ObjectTypeDB::bind_method(_MD("generate_normals"),&SurfaceTool::generate_normals);
	ObjectTypeDB::bind_method(_MD("commit:Mesh","existing:Mesh"),&SurfaceTool::commit,DEFVAL( RefPtr() ));
	ObjectTypeDB::bind_method(_MD("clear"),&SurfaceTool::clear);

}


SurfaceTool::SurfaceTool() {

	first=false;
	begun=false;
	primitive=Mesh::PRIMITIVE_LINES;
	format=0;

}

