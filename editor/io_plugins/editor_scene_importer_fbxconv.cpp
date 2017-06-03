/*************************************************************************/
/*  editor_scene_importer_fbxconv.cpp                                    */
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
#include "editor_scene_importer_fbxconv.h"

#include "editor/editor_settings.h"
#include "os/file_access.h"
#include "os/os.h"
#include "scene/3d/mesh_instance.h"
#include "scene/animation/animation_player.h"

#if 0
String EditorSceneImporterFBXConv::_id(const String& p_id) const {

	return p_id.replace(":","_").replace("/","_");
}

uint32_t EditorSceneImporterFBXConv::get_import_flags() const {

	return IMPORT_SCENE|IMPORT_ANIMATION;
}
void EditorSceneImporterFBXConv::get_extensions(List<String> *r_extensions) const{

	r_extensions->push_back("fbx");
	r_extensions->push_back("g3dj");
}


Color EditorSceneImporterFBXConv::_get_color(const Array& a) {

	if (a.size()<3)
		return Color();
	Color c;
	c.r=a[0];
	c.g=a[1];
	c.b=a[2];
	if (a.size()>=4)
		c.a=a[3];
	return c;

}

Transform EditorSceneImporterFBXConv::_get_transform_mixed(const Dictionary& d,const Dictionary& dbase) {




	Array translation;

	if (d.has("translation"))
		translation=d["translation"];
	else if (dbase.has("translation"))
		translation=dbase["translation"];

	Array rotation;

	if (d.has("rotation"))
		rotation=d["rotation"];
	else if (dbase.has("rotation"))
		rotation=dbase["rotation"];

	Array scale;

	if (d.has("scale"))
		scale=d["scale"];
	else if (dbase.has("scale"))
		scale=dbase["scale"];

	Transform t;


	if (translation.size()) {
		Array tr = translation;
		if (tr.size()>=3) {
			t.origin.x=tr[0];
			t.origin.y=tr[1];
			t.origin.z=tr[2];
		}
	}

	if (rotation.size()) {

		Array r = rotation;
		if (r.size()>=4) {

			Quat q;
			q.x = r[0];
			q.y = r[1];
			q.z = r[2];
			q.w = r[3];
			t.basis=Matrix3(q);
		}
	}


	if (scale.size()) {

		Array sc = scale;
		if (sc.size()>=3) {
			Vector3 s;
			s.x=sc[0];
			s.y=sc[1];
			s.z=sc[2];
			t.basis.scale(s);
		}
	}

	return t;


}

Transform EditorSceneImporterFBXConv::_get_transform(const Dictionary& d) {


	Transform t;

	if (d.has("translation")) {
		Array tr = d["translation"];
		if (tr.size()>=3) {
			t.origin.x=tr[0];
			t.origin.y=tr[1];
			t.origin.z=tr[2];
		}
	}

	if (d.has("rotation")) {

		Array r = d["rotation"];
		if (r.size()>=4) {

			Quat q;
			q.x = r[0];
			q.y = r[1];
			q.z = r[2];
			q.w = r[3];
			t.basis=Matrix3(q);
		}
	}


	if (d.has("scale")) {

		Array sc = d["scale"];
		if (sc.size()>=3) {
			Vector3 s;
			s.x=sc[0];
			s.y=sc[1];
			s.z=sc[2];
			t.basis.scale(s);
		}
	}

	return t;
}


void EditorSceneImporterFBXConv::_detect_bones_in_nodes(State& state,const Array& p_nodes) {


	for(int i=0;i<p_nodes.size();i++) {

		Dictionary d = p_nodes[i];
		if (d.has("isBone") && bool(d["isBone"])) {

			String bone_name=_id(d["id"]);
			print_line("IS BONE: "+bone_name);
			if (!state.bones.has(bone_name)) {
				state.bones.insert(bone_name,BoneInfo());
			}

			if (!state.bones[bone_name].has_rest) {
				state.bones[bone_name].rest=_get_transform(d).affine_inverse();
			}

			state.bones[bone_name].node=d;

			//state.bones[name].rest=_get_transform(b);
		}

		if (d.has("parts")) {

			Array parts=d["parts"];
			for(int j=0;j<parts.size();j++) {

				Dictionary p=parts[j];
				if (p.has("bones")) {
					Array bones=p["bones"];
					//omfg
					for(int k=0;k<bones.size();k++) {

						Dictionary b = bones[k];
						if (b.has("node")) {

							String name = _id(b["node"]);
							if (!state.bones.has(name)) {
								state.bones.insert(name,BoneInfo());
							}

							state.bones[name].rest=_get_transform(b);
							state.bones[name].has_rest=true;
						}
					}
				}

			}
		}

		if (d.has("children")) {

			_detect_bones_in_nodes(state,d["children"]);
		}
	}

}

void EditorSceneImporterFBXConv::_parse_skeletons(const String& p_name,State& state, const Array &p_nodes, Skeleton *p_skeleton,int p_parent) {



	for(int i=0;i<p_nodes.size();i++) {


		Dictionary d = p_nodes[i];
		int bone_idx=-1;
		String id;
		Skeleton* skeleton=p_skeleton;
		if (d.has("id")) {

			id=_id(d["id"]);
			if (state.bones.has(id)) {
				//BONER
				if (!skeleton) {
					skeleton=memnew( Skeleton );
					state.skeletons[id]=skeleton;
				}
				bone_idx = skeleton->get_bone_count();
				skeleton->add_bone(id);
				skeleton->set_bone_parent(bone_idx,p_parent);
				skeleton->set_bone_rest(bone_idx,state.bones[id].rest);
				state.bones[id].skeleton=skeleton;
			}
		}

		if (d.has("children")) {

			_parse_skeletons(id,state,d["children"],skeleton,bone_idx);
		}
	}

}

void EditorSceneImporterFBXConv::_detect_bones(State& state) {
	//This format should mark when a node is a bone,
	//which is the only thing that Collada does right.
	//think about others implementing a parser.
	//Just _one_ string and you avoid loads of lines of code to other people.

	for(int i=0;i<state.animations.size();i++) {

		Dictionary an = state.animations[i];
		if (an.has("bones")) {

			Array bo=an["bones"];
			for(int j=0;j<bo.size();j++) {

				Dictionary b=bo[j];
				if (b.has("boneId")) {

					String id = b["boneId"];
					if (!state.bones.has(id)) {
						state.bones.insert(id,BoneInfo());
					}
					state.bones[id].has_anim_chan=true; //used in anim


				}
			}
		}
	}

	_detect_bones_in_nodes(state,state.nodes);
	_parse_skeletons("",state,state.nodes,NULL,-1);

	print_line("found bones: "+itos(state.bones.size()));
	print_line("found skeletons: "+itos(state.skeletons.size()));
}

Error EditorSceneImporterFBXConv::_parse_bones(State& state,const Array &p_bones,Skeleton* p_skeleton) {



	return OK;
}


void EditorSceneImporterFBXConv::_add_surface(State& state,Ref<Mesh>& m,const Dictionary &part) {

	if (part.has("meshpartid")) {

		String id = part["meshpartid"];
		ERR_FAIL_COND(!state.surface_cache.has(id));


		Ref<Material> mat;
		if (part.has("materialid")) {
			String matid=part["materialid"];
			if (state.material_cache.has(matid)) {
				mat=state.material_cache[matid];
			}
		}
		int idx = m->get_surface_count();

		Array array = state.surface_cache[id].array;
		PoolVector<float> indices = array[Mesh::ARRAY_BONES];
		if (indices.size() && part.has("bones")) {


			Map<int,int> index_map;

			Array bones=part["bones"];

			for(int i=0;i<bones.size();i++) {

				Dictionary bone=bones[i];
				String name=_id(bone["node"]);

				if (state.bones.has(name)) {
					int idx=state.bones[name].skeleton->find_bone(name);
					if (idx==-1)
						idx=0;
					index_map[i]=idx;
				}
			}



			int ilen=indices.size();
			{
				PoolVector<float>::Write iw=indices.write();
				for(int j=0;j<ilen;j++) {
					int b = iw[j];
					ERR_CONTINUE(!index_map.has(b));
					b=index_map[b];
					iw[j]=b;
				}
			}

			array[Mesh::ARRAY_BONES]=indices;


		}

		m->add_surface(state.surface_cache[id].primitive,array);
		m->surface_set_material(idx,mat);
		m->surface_set_name(idx,id);
	}

}

Error EditorSceneImporterFBXConv::_parse_nodes(State& state,const Array &p_nodes,Node* p_base) {

	for(int i=0;i<p_nodes.size();i++) {

		Dictionary n = p_nodes[i];
		Spatial *node=NULL;
		bool skip=false;

		String id;
		if (n.has("id")) {
			id=_id(n["id"]);
		}

		print_line("ID: "+id);

		if (state.skeletons.has(id)) {

			Skeleton *skeleton = state.skeletons[id];
			node=skeleton;
			skeleton->localize_rests();
			print_line("IS SKELETON! ");
		} else if (state.bones.has(id)) {
			if (p_base)
				node=p_base->cast_to<Spatial>();
			if (!state.bones[id].has_anim_chan) {
				print_line("no has anim "+id);
			}
			skip=true;
		} else if (n.has("parts")) {
			//is a mesh
			MeshInstance *mesh = memnew( MeshInstance );
			node=mesh;

			Array parts=n["parts"];
			String long_identifier;
			for(int j=0;j<parts.size();j++) {

				Dictionary part=parts[j];
				if (part.has("meshpartid")) {
					String partid=part["meshpartid"];
					long_identifier+=partid;
				}
			}

			Ref<Mesh> m;

			if (state.mesh_cache.has(long_identifier)) {
				m=state.mesh_cache[long_identifier];
			} else {
				m = Ref<Mesh>( memnew( Mesh ) );

				//and parts are surfaces
				for(int j=0;j<parts.size();j++) {

					Dictionary part=parts[j];
					if (part.has("meshpartid")) {
						_add_surface(state,m,part);
					}
				}


				state.mesh_cache[long_identifier]=m;
			}

			mesh->set_mesh(m);
		}

		if (!skip) {

			if (!node) {
				node = memnew( Spatial );
			}

			node->set_name(id);
			node->set_transform(_get_transform(n));
			p_base->add_child(node);
			node->set_owner(state.scene);
		}


		if (n.has("children")) {
			Error err = _parse_nodes(state,n["children"],node);
			if (err)
				return err;
		}
	}

	return OK;
}


void EditorSceneImporterFBXConv::_parse_materials(State& state) {

	for(int i=0;i<state.materials.size();i++)	 {

		Dictionary material = state.materials[i];

		ERR_CONTINUE(!material.has("id"));
		String id = _id(material["id"]);

		Ref<SpatialMaterial> mat = memnew( SpatialMaterial );

		if (material.has("diffuse")) {
			mat->set_parameter(SpatialMaterial::PARAM_DIFFUSE,_get_color(material["diffuse"]));
		}

		if (material.has("specular")) {
			mat->set_parameter(SpatialMaterial::PARAM_SPECULAR,_get_color(material["specular"]));
		}

		if (material.has("emissive")) {
			mat->set_parameter(SpatialMaterial::PARAM_EMISSION,_get_color(material["emissive"]));
		}

		if (material.has("shininess")) {
			float exp = material["shininess"];
			mat->set_parameter(SpatialMaterial::PARAM_SPECULAR_EXP,exp);
		}

		if (material.has("opacity")) {
			Color c = mat->get_parameter(SpatialMaterial::PARAM_DIFFUSE);
			c.a=material["opacity"];
			mat->set_parameter(SpatialMaterial::PARAM_DIFFUSE,c);
		}


		if (material.has("textures")) {

			Array textures = material["textures"];
			for(int j=0;j<textures.size();j++) {

				Dictionary texture=textures[j];
				Ref<Texture> tex;
				if (texture.has("filename")) {


					String filename=texture["filename"];
					String path=state.base_path+"/"+filename.replace("\\","/");
					if (state.texture_cache.has(path)) {
						tex=state.texture_cache[path];
					} else {
						tex = ResourceLoader::load(path,"ImageTexture");
						if (tex.is_null()) {
							if (state.missing_deps)
								state.missing_deps->push_back(path);
						}
						state.texture_cache[path]=tex; //add anyway
					}
				}

				if (tex.is_valid() && texture.has("type")) {

					String type=texture["type"];
					if (type=="DIFFUSE")
						mat->set_texture(SpatialMaterial::PARAM_DIFFUSE,tex);
					else if (type=="SPECULAR")
						mat->set_texture(SpatialMaterial::PARAM_SPECULAR,tex);
					else if (type=="SHININESS")
						mat->set_texture(SpatialMaterial::PARAM_SPECULAR_EXP,tex);
					else if (type=="NORMAL")
						mat->set_texture(SpatialMaterial::PARAM_NORMAL,tex);
					else if (type=="EMISSIVE")
						mat->set_texture(SpatialMaterial::PARAM_EMISSION,tex);
				}

			}
		}

		state.material_cache[id]=mat;

	}

}

void EditorSceneImporterFBXConv::_parse_surfaces(State& state) {

	for(int i=0;i<state.meshes.size();i++)	 {

		Dictionary mesh = state.meshes[i];

		ERR_CONTINUE(!mesh.has("attributes"));
		ERR_CONTINUE(!mesh.has("vertices"));
		ERR_CONTINUE(!mesh.has("parts"));

		print_line("MESH #"+itos(i));

		Array attrlist=mesh["attributes"];
		Array vertices=mesh["vertices"];
		bool exists[Mesh::ARRAY_MAX];
		int ofs[Mesh::ARRAY_MAX];
		int weight_max=0;
		int binormal_ofs=-1;
		int weight_ofs[4];

		for(int j=0;j<Mesh::ARRAY_MAX;j++) {
			exists[j]=false;
			ofs[j]=0;
		}
		exists[Mesh::ARRAY_INDEX]=true;
		float stride=0;

		for(int j=0;j<attrlist.size();j++) {

			String attr=attrlist[j];
			if (attr=="POSITION") {
				exists[Mesh::ARRAY_VERTEX]=true;
				ofs[Mesh::ARRAY_VERTEX]=stride;
				stride+=3;
			} else if (attr=="NORMAL") {
				exists[Mesh::ARRAY_NORMAL]=true;
				ofs[Mesh::ARRAY_NORMAL]=stride;
				stride+=3;
			} else if (attr=="COLOR") {
				exists[Mesh::ARRAY_COLOR]=true;
				ofs[Mesh::ARRAY_COLOR]=stride;
				stride+=4;
			} else if (attr=="COLORPACKED") {
				stride+=1; //ignore
			} else if (attr=="TANGENT") {
				exists[Mesh::ARRAY_TANGENT]=true;
				ofs[Mesh::ARRAY_TANGENT]=stride;
				stride+=3;
			} else if (attr=="BINORMAL") {
				binormal_ofs=stride;
				stride+=3;
			} else if (attr=="TEXCOORD0") {
				exists[Mesh::ARRAY_TEX_UV]=true;
				ofs[Mesh::ARRAY_TEX_UV]=stride;
				stride+=2;
			} else if (attr=="TEXCOORD1") {
				exists[Mesh::ARRAY_TEX_UV2]=true;
				ofs[Mesh::ARRAY_TEX_UV2]=stride;
				stride+=2;
			} else if (attr.begins_with("TEXCOORD")) {
				stride+=2;
			} else if (attr.begins_with("BLENDWEIGHT")) {
				int idx=attr.replace("BLENDWEIGHT","").to_int();
				if (idx==0) {
					exists[Mesh::ARRAY_BONES]=true;
					ofs[Mesh::ARRAY_BONES]=stride;
					exists[Mesh::ARRAY_WEIGHTS]=true;
					ofs[Mesh::ARRAY_WEIGHTS]=stride+1;
				} if (idx<4) {
					weight_ofs[idx]=stride;
					weight_max=MAX(weight_max,idx+1);
				}

				stride+=2;
			}

			print_line("ATTR "+attr+" OFS: "+itos(stride));

		}

		Array parts=mesh["parts"];

		for(int j=0;j<parts.size();j++) {



			Dictionary part=parts[j];
			ERR_CONTINUE(!part.has("indices"));
			ERR_CONTINUE(!part.has("id"));

			print_line("PART: "+String(part["id"]));
			Array indices=part["indices"];
			Map<int,int> iarray;
			Map<int,int> array;

			for(int k=0;k<indices.size();k++) {

				int idx = indices[k];
				if (!iarray.has(idx)) {
					int map_to=array.size();
					iarray[idx]=map_to;
					array[map_to]=idx;
				}
			}

			print_line("indices total "+itos(indices.size())+" vertices used: "+itos(array.size()));

			Array arrays;
			arrays.resize(Mesh::ARRAY_MAX);



			for(int k=0;k<Mesh::ARRAY_MAX;k++) {


				if (!exists[k])
					continue;
				print_line("exists: "+itos(k));
				int lofs = ofs[k];
				switch(k) {

					case Mesh::ARRAY_VERTEX:
					case Mesh::ARRAY_NORMAL: {

						PoolVector<Vector3> vtx;
						vtx.resize(array.size());
						{
							int len=array.size();
							PoolVector<Vector3>::Write w = vtx.write();
							for(int l=0;l<len;l++) {

								int pos = array[l];
								w[l].x=vertices[pos*stride+lofs+0];
								w[l].y=vertices[pos*stride+lofs+1];
								w[l].z=vertices[pos*stride+lofs+2];
							}
						}
						arrays[k]=vtx;

					} break;
					case Mesh::ARRAY_TANGENT: {

						if (binormal_ofs<0)
							break;

						PoolVector<float> tangents;
						tangents.resize(array.size()*4);
						{
							int len=array.size();

							PoolVector<float>::Write w = tangents.write();
							for(int l=0;l<len;l++) {

								int pos = array[l];
								Vector3 n;
								n.x=vertices[pos*stride+ofs[Mesh::ARRAY_NORMAL]+0];
								n.y=vertices[pos*stride+ofs[Mesh::ARRAY_NORMAL]+1];
								n.z=vertices[pos*stride+ofs[Mesh::ARRAY_NORMAL]+2];
								Vector3 t;
								t.x=vertices[pos*stride+lofs+0];
								t.y=vertices[pos*stride+lofs+1];
								t.z=vertices[pos*stride+lofs+2];
								Vector3 bi;
								bi.x=vertices[pos*stride+binormal_ofs+0];
								bi.y=vertices[pos*stride+binormal_ofs+1];
								bi.z=vertices[pos*stride+binormal_ofs+2];
								float d = bi.dot(n.cross(t));

								w[l*4+0]=t.x;
								w[l*4+1]=t.y;
								w[l*4+2]=t.z;
								w[l*4+3]=d;

							}
						}
						arrays[k]=tangents;

					} break;
					case Mesh::ARRAY_COLOR: {

						PoolVector<Color> cols;
						cols.resize(array.size());
						{
							int len=array.size();
							PoolVector<Color>::Write w = cols.write();
							for(int l=0;l<len;l++) {

								int pos = array[l];
								w[l].r=vertices[pos*stride+lofs+0];
								w[l].g=vertices[pos*stride+lofs+1];
								w[l].b=vertices[pos*stride+lofs+2];
								w[l].a=vertices[pos*stride+lofs+3];
							}
						}
						arrays[k]=cols;

					} break;
					case Mesh::ARRAY_TEX_UV:
					case Mesh::ARRAY_TEX_UV2: {

						PoolVector<Vector2> uvs;
						uvs.resize(array.size());
						{
							int len=array.size();
							PoolVector<Vector2>::Write w = uvs.write();
							for(int l=0;l<len;l++) {

								int pos = array[l];
								w[l].x=vertices[pos*stride+lofs+0];
								w[l].y=vertices[pos*stride+lofs+1];
								w[l].y=1.0-w[l].y;
							}
						}
						arrays[k]=uvs;

					} break;
					case Mesh::ARRAY_BONES:
					case Mesh::ARRAY_WEIGHTS: {

						PoolVector<float> arr;
						arr.resize(array.size()*4);
						int po=k==Mesh::ARRAY_WEIGHTS?1:0;
						lofs=ofs[Mesh::ARRAY_BONES];
						{
							int len=array.size();

							PoolVector<float>::Write w = arr.write();
							for(int l=0;l<len;l++) {

								int pos = array[l];

								for(int m=0;m<4;m++) {

									float val=0;
									if (m<=weight_max)
										val=vertices[pos*stride+lofs+m*2+po];
									w[l*4+m]=val;
								}
							}
						}

						arrays[k]=arr;
					} break;
					case Mesh::ARRAY_INDEX: {

						PoolVector<int> arr;
						arr.resize(indices.size());
						{
							int len=indices.size();

							PoolVector<int>::Write w = arr.write();
							for(int l=0;l<len;l++) {

								w[l]=iarray[ indices[l] ];
							}
						}

						arrays[k]=arr;

					} break;


				}


			}

			Mesh::PrimitiveType pt=Mesh::PRIMITIVE_TRIANGLES;

			if (part.has("type")) {
				String type=part["type"];
				if (type=="LINES")
					pt=Mesh::PRIMITIVE_LINES;
				else if (type=="POINTS")
					pt=Mesh::PRIMITIVE_POINTS;
				else if (type=="TRIANGLE_STRIP")
					pt=Mesh::PRIMITIVE_TRIANGLE_STRIP;
				else if (type=="LINE_STRIP")
					pt=Mesh::PRIMITIVE_LINE_STRIP;
			}

			if (pt==Mesh::PRIMITIVE_TRIANGLES) {
				PoolVector<int> ia=arrays[Mesh::ARRAY_INDEX];
				int len=ia.size();
				{
					PoolVector<int>::Write w=ia.write();
					for(int l=0;l<len;l+=3) {
						SWAP(w[l+1],w[l+2]);
					}
				}
				arrays[Mesh::ARRAY_INDEX]=ia;


			}
			SurfaceInfo si;
			si.array=arrays;
			si.primitive=pt;
			state.surface_cache[_id(part["id"])]=si;

		}
	}
}


Error EditorSceneImporterFBXConv::_parse_animations(State& state) {

	AnimationPlayer *ap = memnew( AnimationPlayer );

	state.scene->add_child(ap);
	ap->set_owner(state.scene);

	for(int i=0;i<state.animations.size();i++) {

		Dictionary anim = state.animations[i];
		ERR_CONTINUE(!anim.has("id"));
		Ref<Animation> an = memnew( Animation );
		an->set_name(_id(anim["id"]));


		if (anim.has("bones")) {

			Array bone_tracks = anim["bones"];
			for(int j=0;j<bone_tracks.size();j++) {
				Dictionary bone_track=bone_tracks[j];
				String bone = bone_track["boneId"];
				if (!bone_track.has("keyframes"))
					continue;
				if (!state.bones.has(bone))
					continue;

				Skeleton *sk = state.bones[bone].skeleton;

				if (!sk)
					continue;
				int bone_idx=sk->find_bone(bone);
				if (bone_idx==-1)
					continue;



				String path = state.scene->get_path_to(sk);
				path+=":"+bone;
				an->add_track(Animation::TYPE_TRANSFORM);
				int tidx = an->get_track_count()-1;
				an->track_set_path(tidx,path);


				Dictionary parent_xform_dict;
				Dictionary xform_dict;

				if (state.bones.has(bone)) {
					xform_dict=state.bones[bone].node;
				}


				Array parent_keyframes;
				if (sk->get_bone_parent(bone_idx)!=-1) {
					String parent_name = sk->get_bone_name(sk->get_bone_parent(bone_idx));
					if (state.bones.has(parent_name)) {
						parent_xform_dict=state.bones[parent_name].node;
					}

					print_line("parent for "+bone+"? "+parent_name+" XFD: "+String(Variant(parent_xform_dict)));
					for(int k=0;k<bone_tracks.size();k++) {
						Dictionary d = bone_tracks[k];
						if (d["boneId"]==parent_name) {
							parent_keyframes=d["keyframes"];
							print_line("found keyframes");
							break;
						}
					}


				}

				print_line("BONE XFD "+String(Variant(xform_dict)));

				Array keyframes=bone_track["keyframes"];

				for(int k=0;k<keyframes.size();k++) {

					Dictionary key=keyframes[k];
					Transform xform=_get_transform_mixed(key,xform_dict);
					float time = key["keytime"];
					time=time/1000.0;
#if 0
					if (parent_keyframes.size()) {
						//localize
						print_line(itos(k)+" localizate for: "+bone);

						float prev_kt=-1;
						float kt;
						int idx=0;

						for(int l=0;l<parent_keyframes.size();l++) {

							Dictionary d=parent_keyframes[l];
							kt=d["keytime"];
							kt=kt/1000.0;
							if (kt>time)
								break;
							prev_kt=kt;
							idx++;

						}

						Transform t;
						if (idx==0) {
							t=_get_transform_mixed(parent_keyframes[0],parent_xform_dict);
						} else if (idx==parent_keyframes.size()){
							t=_get_transform_mixed(parent_keyframes[idx-1],parent_xform_dict);
						} else {
							t=_get_transform_mixed(parent_keyframes[idx-1],parent_xform_dict);
							float d = (time-prev_kt)/(kt-prev_kt);
							if (d>0) {
								Transform t2=_get_transform_mixed(parent_keyframes[idx],parent_xform_dict);
								t=t.interpolate_with(t2,d);
							} else {
								print_line("exact: "+rtos(kt));
							}
						}

						xform = t.affine_inverse() * xform; //localize
					} else if (!parent_xform_dict.empty()) {
						Transform t = _get_transform(parent_xform_dict);
						xform = t.affine_inverse() * xform; //localize
					}
#endif

					xform = sk->get_bone_rest(bone_idx).affine_inverse() * xform;


					Quat q = xform.basis;
					q.normalize();
					Vector3 s = xform.basis.get_scale();
					Vector3 l = xform.origin;



					an->transform_track_insert_key(tidx,time,l,q,s);

				}

			}


		}


		ap->add_animation(_id(anim["id"]),an);

	}

	return OK;
}

Error EditorSceneImporterFBXConv::_parse_json(State& state, const String &p_path) {

	//not the happiest....
	Vector<uint8_t> data = FileAccess::get_file_as_array(p_path);
	ERR_FAIL_COND_V(!data.size(),ERR_FILE_CANT_OPEN);
	String str;
	bool utferr = str.parse_utf8((const char*)data.ptr(),data.size());
	ERR_FAIL_COND_V(utferr,ERR_PARSE_ERROR);

	Dictionary dict;
	Error err = dict.parse_json(str);
	str=String(); //free mem immediately
	ERR_FAIL_COND_V(err,err);

	if (dict.has("meshes"))
		state.meshes=dict["meshes"];
	if (dict.has("materials"))
		state.materials=dict["materials"];
	if (dict.has("nodes"))
		state.nodes=dict["nodes"];
	if (dict.has("animations"))
		state.animations=dict["animations"];


	state.scene = memnew( Spatial );
	_detect_bones(state);
	_parse_surfaces(state);
	_parse_materials(state);
	err = _parse_nodes(state,state.nodes,state.scene);
	if (err)
		return err;

	if (state.import_animations) {
		err = _parse_animations(state);
		if (err)
			return err;
	}

	print_line("JSON PARSED O-K!");

	return OK;
}

Error EditorSceneImporterFBXConv::_parse_fbx(State& state,const String& p_path) {

	state.base_path=p_path.get_base_dir();

	if (p_path.to_lower().ends_with("g3dj")) {
		return _parse_json(state,p_path.basename()+".g3dj");
	}

	String tool = EDITOR_DEF("fbxconv/path","");
	ERR_FAIL_COND_V( !FileAccess::exists(tool),ERR_UNCONFIGURED);
	String wine = EDITOR_DEF("fbxconv/use_wine","");

	List<String> args;
	String path=p_path;
	if (wine!="") {
		List<String> wpargs;
		wpargs.push_back("-w");
		wpargs.push_back(p_path);
		String pipe; //winepath to convert to windows path
		int wpres;
		Error wperr = OS::get_singleton()->execute(wine+"path",wpargs,true,NULL,&pipe,&wpres);
		ERR_FAIL_COND_V(wperr!=OK,ERR_CANT_CREATE);
		ERR_FAIL_COND_V(wpres!=0,ERR_CANT_CREATE);
		path=pipe.strip_edges();
		args.push_back(tool);
		tool=wine;
	}

	args.push_back("-o");
	args.push_back("G3DJ");
	args.push_back(path);

	int res;
	Error err = OS::get_singleton()->execute(tool,args,true,NULL,NULL,&res);
	ERR_FAIL_COND_V(err!=OK,ERR_CANT_CREATE);
	ERR_FAIL_COND_V(res!=0,ERR_CANT_CREATE);

	return _parse_json(state,p_path.basename()+".g3dj");


}

Node* EditorSceneImporterFBXConv::import_scene(const String& p_path,uint32_t p_flags,List<String> *r_missing_deps,Error* r_err){

	State state;
	state.scene=NULL;
	state.missing_deps=r_missing_deps;
	state.import_animations=p_flags&IMPORT_ANIMATION;
	Error err = _parse_fbx(state,p_path);
	if (err!=OK) {
		if (r_err)
			*r_err=err;
		return NULL;
	}


	return state.scene;
}
Ref<Animation> EditorSceneImporterFBXConv::import_animation(const String& p_path,uint32_t p_flags){


	return Ref<Animation>();
}


EditorSceneImporterFBXConv::EditorSceneImporterFBXConv() {

	EDITOR_DEF("fbxconv/path","");
#ifndef WINDOWS_ENABLED
	EDITOR_DEF("fbxconv/use_wine","");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING,"fbxconv/use_wine",PROPERTY_HINT_GLOBAL_FILE));
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING,"fbxconv/path",PROPERTY_HINT_GLOBAL_FILE));
#else
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING,"fbxconv/path",PROPERTY_HINT_GLOBAL_FILE,"exe"));
#endif

}
#endif
