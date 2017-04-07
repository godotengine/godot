/*************************************************************************/
/*  editor_import_collada.cpp                                            */
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
#include "editor_import_collada.h"

#include "editor/collada/collada.h"
#include "editor/editor_node.h"
#include "os/os.h"
#include "scene/3d/camera.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/path.h"
#include "scene/3d/skeleton.h"
#include "scene/3d/spatial.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/packed_scene.h"
#include <iostream>

struct ColladaImport {

	Collada collada;
	Spatial *scene;

	Vector<Ref<Animation> > animations;

	struct NodeMap {
		//String path;
		Spatial *node;
		int bone;
		List<int> anim_tracks;

		NodeMap() {
			node = NULL;
			bone = -1;
		}
	};

	bool found_ambient;
	Color ambient;
	bool found_directional;
	bool force_make_tangents;
	bool apply_mesh_xform_to_vertices;
	bool use_mesh_builtin_materials;
	float bake_fps;

	Map<String, NodeMap> node_map; //map from collada node to engine node
	Map<String, String> node_name_map; //map from collada node to engine node
	Map<String, Ref<Mesh> > mesh_cache;
	Map<String, Ref<Curve3D> > curve_cache;
	Map<String, Ref<Material> > material_cache;
	Map<Collada::Node *, Skeleton *> skeleton_map;

	Map<Skeleton *, Map<String, int> > skeleton_bone_map;

	Set<String> valid_animated_nodes;
	Vector<int> valid_animated_properties;
	Map<String, bool> bones_with_animation;

	Error _populate_skeleton(Skeleton *p_skeleton, Collada::Node *p_node, int &r_bone, int p_parent);
	Error _create_scene_skeletons(Collada::Node *p_node);
	Error _create_scene(Collada::Node *p_node, Spatial *p_parent);
	Error _create_resources(Collada::Node *p_node);
	Error _create_material(const String &p_material);
	Error _create_mesh_surfaces(bool p_optimize, Ref<Mesh> &p_mesh, const Map<String, Collada::NodeGeometry::Material> &p_material_map, const Collada::MeshData &meshdata, const Transform &p_local_xform, const Vector<int> &bone_remap, const Collada::SkinControllerData *p_skin_data, const Collada::MorphControllerData *p_morph_data, Vector<Ref<Mesh> > p_morph_meshes = Vector<Ref<Mesh> >(), bool p_for_morph = false, bool p_use_mesh_material = false);
	Error load(const String &p_path, int p_flags, bool p_force_make_tangents = false);
	void _fix_param_animation_tracks();
	void create_animation(int p_clip, bool p_make_tracks_in_all_bones, bool p_import_value_tracks);
	void create_animations(bool p_make_tracks_in_all_bones, bool p_import_value_tracks);

	Set<String> tracks_in_clips;
	Vector<String> missing_textures;

	void _pre_process_lights(Collada::Node *p_node);

	ColladaImport() {

		found_ambient = false;
		found_directional = false;
		force_make_tangents = false;
		apply_mesh_xform_to_vertices = true;
		bake_fps = 15;
	}
};

Error ColladaImport::_populate_skeleton(Skeleton *p_skeleton, Collada::Node *p_node, int &r_bone, int p_parent) {

	if (p_node->type != Collada::Node::TYPE_JOINT)
		return OK;

	Collada::NodeJoint *joint = static_cast<Collada::NodeJoint *>(p_node);

	print_line("populating joint " + joint->name);
	p_skeleton->add_bone(p_node->name);
	if (p_parent >= 0)
		p_skeleton->set_bone_parent(r_bone, p_parent);

	NodeMap nm;
	nm.node = p_skeleton;
	nm.bone = r_bone;
	node_map[p_node->id] = nm;
	node_name_map[p_node->name] = p_node->id;

	skeleton_bone_map[p_skeleton][joint->sid] = r_bone;

	if (collada.state.bone_rest_map.has(joint->sid)) {

		p_skeleton->set_bone_rest(r_bone, collada.fix_transform(collada.state.bone_rest_map[joint->sid]));
		//should map this bone to something for animation?
	} else {
		print_line("no rest: " + joint->sid);
		WARN_PRINT("Joint has no rest..");
	}

	int id = r_bone++;
	for (int i = 0; i < p_node->children.size(); i++) {

		Error err = _populate_skeleton(p_skeleton, p_node->children[i], r_bone, id);
		if (err)
			return err;
	}

	return OK;
}

void ColladaImport::_pre_process_lights(Collada::Node *p_node) {

	if (p_node->type == Collada::Node::TYPE_LIGHT) {

		Collada::NodeLight *light = static_cast<Collada::NodeLight *>(p_node);
		if (collada.state.light_data_map.has(light->light)) {

			Collada::LightData &ld = collada.state.light_data_map[light->light];
			if (ld.mode == Collada::LightData::MODE_AMBIENT) {
				found_ambient = true;
				ambient = ld.color;
			}
			if (ld.mode == Collada::LightData::MODE_DIRECTIONAL) {
				found_directional = true;
			}
		}
	}

	for (int i = 0; i < p_node->children.size(); i++)
		_pre_process_lights(p_node->children[i]);
}

Error ColladaImport::_create_scene_skeletons(Collada::Node *p_node) {

	if (p_node->type == Collada::Node::TYPE_SKELETON) {

		Skeleton *sk = memnew(Skeleton);
		int bone = 0;

		for (int i = 0; i < p_node->children.size(); i++) {

			_populate_skeleton(sk, p_node->children[i], bone, -1);
		}
		sk->localize_rests(); //after creating skeleton, rests must be localized...!
		skeleton_map[p_node] = sk;
	}

	for (int i = 0; i < p_node->children.size(); i++) {

		Error err = _create_scene_skeletons(p_node->children[i]);
		if (err)
			return err;
	}
	return OK;
}

Error ColladaImport::_create_scene(Collada::Node *p_node, Spatial *p_parent) {

	Spatial *node = NULL;

	switch (p_node->type) {

		case Collada::Node::TYPE_NODE: {

			node = memnew(Spatial);
		} break;
		case Collada::Node::TYPE_JOINT: {

			return OK; // do nothing
		} break;
		case Collada::Node::TYPE_LIGHT: {

			//node = memnew( Light)
			Collada::NodeLight *light = static_cast<Collada::NodeLight *>(p_node);
			if (collada.state.light_data_map.has(light->light)) {

				Collada::LightData &ld = collada.state.light_data_map[light->light];

				if (ld.mode == Collada::LightData::MODE_AMBIENT) {

					if (found_directional)
						return OK; //do nothing not needed

					if (!bool(GLOBAL_DEF("collada/use_ambient", false)))
						return OK;
					//well, it's an ambient light..
					Light *l = memnew(DirectionalLight);
					//l->set_color(Light::COLOR_AMBIENT,ld.color);
					//l->set_color(Light::COLOR_DIFFUSE,Color(0,0,0));
					//l->set_color(Light::COLOR_SPECULAR,Color(0,0,0));
					node = l;

				} else if (ld.mode == Collada::LightData::MODE_DIRECTIONAL) {

					//well, it's an ambient light..
					Light *l = memnew(DirectionalLight);
					/*
					if (found_ambient) //use it here
						l->set_color(Light::COLOR_AMBIENT,ambient);

					l->set_color(Light::COLOR_DIFFUSE,ld.color);
					l->set_color(Light::COLOR_SPECULAR,Color(1,1,1));
					*/
					node = l;
				} else {

					Light *l;

					if (ld.mode == Collada::LightData::MODE_OMNI)
						l = memnew(OmniLight);
					else {
						l = memnew(SpotLight);
						//l->set_parameter(Light::PARAM_SPOT_ANGLE,ld.spot_angle);
						//l->set_parameter(Light::PARAM_SPOT_ATTENUATION,ld.spot_exp);
					}

					//
					//l->set_color(Light::COLOR_DIFFUSE,ld.color);
					//l->set_color(Light::COLOR_SPECULAR,Color(1,1,1));
					//l->approximate_opengl_attenuation(ld.constant_att,ld.linear_att,ld.quad_att);
					node = l;
				}

			} else {

				node = memnew(Spatial);
			}
		} break;
		case Collada::Node::TYPE_CAMERA: {

			Collada::NodeCamera *cam = static_cast<Collada::NodeCamera *>(p_node);
			Camera *camera = memnew(Camera);

			if (collada.state.camera_data_map.has(cam->camera)) {

				const Collada::CameraData &cd = collada.state.camera_data_map[cam->camera];

				switch (cd.mode) {

					case Collada::CameraData::MODE_ORTHOGONAL: {

						if (cd.orthogonal.y_mag) {

							camera->set_keep_aspect_mode(Camera::KEEP_HEIGHT);
							camera->set_orthogonal(cd.orthogonal.y_mag * 2.0, cd.z_near, cd.z_far);

						} else if (!cd.orthogonal.y_mag && cd.orthogonal.x_mag) {

							camera->set_keep_aspect_mode(Camera::KEEP_WIDTH);
							camera->set_orthogonal(cd.orthogonal.x_mag * 2.0, cd.z_near, cd.z_far);
						}

					} break;
					case Collada::CameraData::MODE_PERSPECTIVE: {

						if (cd.perspective.y_fov) {

							camera->set_perspective(cd.perspective.y_fov, cd.z_near, cd.z_far);

						} else if (!cd.perspective.y_fov && cd.perspective.x_fov) {

							camera->set_perspective(cd.perspective.x_fov / cd.aspect, cd.z_near, cd.z_far);
						}

					} break;
				}
			}

			node = camera;

		} break;
		case Collada::Node::TYPE_GEOMETRY: {

			Collada::NodeGeometry *ng = static_cast<Collada::NodeGeometry *>(p_node);

			if (collada.state.curve_data_map.has(ng->source)) {

				node = memnew(Path);
			} else {
				//mesh since nothing else
				node = memnew(MeshInstance);
				node->cast_to<MeshInstance>()->set_flag(GeometryInstance::FLAG_USE_BAKED_LIGHT, true);
			}
		} break;
		case Collada::Node::TYPE_SKELETON: {

			ERR_FAIL_COND_V(!skeleton_map.has(p_node), ERR_CANT_CREATE);
			Skeleton *sk = skeleton_map[p_node];
			node = sk;
		} break;
	}

	if (p_node->name != "")
		node->set_name(p_node->name);
	NodeMap nm;
	nm.node = node;
	node_map[p_node->id] = nm;
	node_name_map[p_node->name] = p_node->id;
	Transform xf = p_node->default_transform;

	xf = collada.fix_transform(xf) * p_node->post_transform;
	node->set_transform(xf);
	p_parent->add_child(node);
	node->set_owner(scene);

	if (p_node->empty_draw_type != "") {
		node->set_meta("empty_draw_type", Variant(p_node->empty_draw_type));
	}

	for (int i = 0; i < p_node->children.size(); i++) {

		Error err = _create_scene(p_node->children[i], node);
		if (err)
			return err;
	}
	return OK;
}

Error ColladaImport::_create_material(const String &p_target) {

	ERR_FAIL_COND_V(material_cache.has(p_target), ERR_ALREADY_EXISTS);
	ERR_FAIL_COND_V(!collada.state.material_map.has(p_target), ERR_INVALID_PARAMETER);
	Collada::Material &src_mat = collada.state.material_map[p_target];
	ERR_FAIL_COND_V(!collada.state.effect_map.has(src_mat.instance_effect), ERR_INVALID_PARAMETER);
	Collada::Effect &effect = collada.state.effect_map[src_mat.instance_effect];

	Ref<SpatialMaterial> material = memnew(SpatialMaterial);

	if (src_mat.name != "")
		material->set_name(src_mat.name);
	else if (effect.name != "")
		material->set_name(effect.name);

	// DIFFUSE

	if (effect.diffuse.texture != "") {

		String texfile = effect.get_texture_path(effect.diffuse.texture, collada);
		if (texfile != "") {

			Ref<Texture> texture = ResourceLoader::load(texfile, "Texture");
			if (texture.is_valid()) {

				material->set_texture(SpatialMaterial::TEXTURE_ALBEDO, texture);
				material->set_albedo(Color(1, 1, 1, 1));
				//material->set_parameter(SpatialMaterial::PARAM_DIFFUSE,Color(1,1,1,1));
			} else {
				missing_textures.push_back(texfile.get_file());
			}
		}
	} else {
		//material->set_parameter(SpatialMaterial::PARAM_DIFFUSE,effect.diffuse.color);
	}

	// SPECULAR

	if (effect.specular.texture != "") {

		String texfile = effect.get_texture_path(effect.specular.texture, collada);
		if (texfile != "") {

			Ref<Texture> texture = ResourceLoader::load(texfile, "Texture");
			if (texture.is_valid()) {
				material->set_texture(SpatialMaterial::TEXTURE_SPECULAR, texture);
				material->set_specular(Color(1, 1, 1, 1));

				//material->set_texture(SpatialMaterial::PARAM_SPECULAR,texture);
				//material->set_parameter(SpatialMaterial::PARAM_SPECULAR,Color(1,1,1,1));
			} else {
				missing_textures.push_back(texfile.get_file());
			}
		}
	} else {
		material->set_metalness(effect.specular.color.get_v());
	}

	// EMISSION

	if (effect.emission.texture != "") {

		String texfile = effect.get_texture_path(effect.emission.texture, collada);
		if (texfile != "") {

			Ref<Texture> texture = ResourceLoader::load(texfile, "Texture");
			if (texture.is_valid()) {

				material->set_feature(SpatialMaterial::FEATURE_EMISSION, true);
				material->set_texture(SpatialMaterial::TEXTURE_EMISSION, texture);
				material->set_emission(Color(1, 1, 1, 1));

				//material->set_parameter(SpatialMaterial::PARAM_EMISSION,Color(1,1,1,1));
			} else {
				missing_textures.push_back(texfile.get_file());
			}
		}
	} else {
		if (effect.emission.color != Color()) {
			material->set_feature(SpatialMaterial::FEATURE_EMISSION, true);
			material->set_emission(effect.emission.color);
		}
	}

	// NORMAL

	if (effect.bump.texture != "") {

		String texfile = effect.get_texture_path(effect.bump.texture, collada);
		if (texfile != "") {

			Ref<Texture> texture = ResourceLoader::load(texfile, "Texture");
			if (texture.is_valid()) {
				material->set_feature(SpatialMaterial::FEATURE_NORMAL_MAPPING, true);
				material->set_texture(SpatialMaterial::TEXTURE_NORMAL, texture);
				//material->set_emission(Color(1,1,1,1));

				//material->set_texture(SpatialMaterial::PARAM_NORMAL,texture);
			} else {
				//missing_textures.push_back(texfile.get_file());
			}
		}
	}

	float roughness = Math::sqrt(1.0 - ((Math::log(effect.shininess) / Math::log(2.0)) / 8.0)); //not very right..
	material->set_roughness(roughness);

	if (effect.double_sided) {
		material->set_cull_mode(SpatialMaterial::CULL_DISABLED);
	}
	material->set_flag(SpatialMaterial::FLAG_UNSHADED, effect.unshaded);

	material_cache[p_target] = material;
	return OK;
}

static void _generate_normals(const PoolVector<int> &p_indices, const PoolVector<Vector3> &p_vertices, PoolVector<Vector3> &r_normals) {

	r_normals.resize(p_vertices.size());
	PoolVector<Vector3>::Write narrayw = r_normals.write();

	int iacount = p_indices.size() / 3;
	PoolVector<int>::Read index_arrayr = p_indices.read();
	PoolVector<Vector3>::Read vertex_arrayr = p_vertices.read();

	for (int idx = 0; idx < iacount; idx++) {

		Vector3 v[3] = {
			vertex_arrayr[index_arrayr[idx * 3 + 0]],
			vertex_arrayr[index_arrayr[idx * 3 + 1]],
			vertex_arrayr[index_arrayr[idx * 3 + 2]]
		};

		Vector3 normal = Plane(v[0], v[1], v[2]).normal;

		narrayw[index_arrayr[idx * 3 + 0]] += normal;
		narrayw[index_arrayr[idx * 3 + 1]] += normal;
		narrayw[index_arrayr[idx * 3 + 2]] += normal;
	}

	int vlen = p_vertices.size();

	for (int idx = 0; idx < vlen; idx++) {
		narrayw[idx].normalize();
	}
}

static void _generate_tangents_and_binormals(const PoolVector<int> &p_indices, const PoolVector<Vector3> &p_vertices, const PoolVector<Vector3> &p_uvs, const PoolVector<Vector3> &p_normals, PoolVector<real_t> &r_tangents) {

	int vlen = p_vertices.size();

	Vector<Vector3> tangents;
	tangents.resize(vlen);
	Vector<Vector3> binormals;
	binormals.resize(vlen);

	int iacount = p_indices.size() / 3;

	PoolVector<int>::Read index_arrayr = p_indices.read();
	PoolVector<Vector3>::Read vertex_arrayr = p_vertices.read();
	PoolVector<Vector3>::Read narrayr = p_normals.read();
	PoolVector<Vector3>::Read uvarrayr = p_uvs.read();

	for (int idx = 0; idx < iacount; idx++) {

		Vector3 v1 = vertex_arrayr[index_arrayr[idx * 3 + 0]];
		Vector3 v2 = vertex_arrayr[index_arrayr[idx * 3 + 1]];
		Vector3 v3 = vertex_arrayr[index_arrayr[idx * 3 + 2]];

		Vector3 w1 = uvarrayr[index_arrayr[idx * 3 + 0]];
		Vector3 w2 = uvarrayr[index_arrayr[idx * 3 + 1]];
		Vector3 w3 = uvarrayr[index_arrayr[idx * 3 + 2]];

		real_t x1 = v2.x - v1.x;
		real_t x2 = v3.x - v1.x;
		real_t y1 = v2.y - v1.y;
		real_t y2 = v3.y - v1.y;
		real_t z1 = v2.z - v1.z;
		real_t z2 = v3.z - v1.z;

		real_t s1 = w2.x - w1.x;
		real_t s2 = w3.x - w1.x;
		real_t t1 = w2.y - w1.y;
		real_t t2 = w3.y - w1.y;

		real_t r = (s1 * t2 - s2 * t1);

		Vector3 tangent;
		Vector3 binormal;

		if (r == 0) {

			binormal = Vector3();
			tangent = Vector3();
		} else {
			tangent = Vector3((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r,
					(t2 * z1 - t1 * z2) * r)
							  .normalized();
			binormal = Vector3((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r,
					(s1 * z2 - s2 * z1) * r)
							   .normalized();
		}

		tangents[index_arrayr[idx * 3 + 0]] += tangent;
		binormals[index_arrayr[idx * 3 + 0]] += binormal;
		tangents[index_arrayr[idx * 3 + 1]] += tangent;
		binormals[index_arrayr[idx * 3 + 1]] += binormal;
		tangents[index_arrayr[idx * 3 + 2]] += tangent;
		binormals[index_arrayr[idx * 3 + 2]] += binormal;

		//print_line(itos(idx)+" tangent: "+tangent);
		//print_line(itos(idx)+" binormal: "+binormal);
	}

	r_tangents.resize(vlen * 4);
	PoolVector<real_t>::Write tarrayw = r_tangents.write();

	for (int idx = 0; idx < vlen; idx++) {
		Vector3 tangent = tangents[idx];
		Vector3 bingen = narrayr[idx].cross(tangent);
		float dir;
		if (bingen.dot(binormals[idx]) < 0)
			dir = -1.0;
		else
			dir = +1.0;

		tarrayw[idx * 4 + 0] = tangent.x;
		tarrayw[idx * 4 + 1] = tangent.y;
		tarrayw[idx * 4 + 2] = tangent.z;
		tarrayw[idx * 4 + 3] = dir;
	}
}

Error ColladaImport::_create_mesh_surfaces(bool p_optimize, Ref<Mesh> &p_mesh, const Map<String, Collada::NodeGeometry::Material> &p_material_map, const Collada::MeshData &meshdata, const Transform &p_local_xform, const Vector<int> &bone_remap, const Collada::SkinControllerData *skin_controller, const Collada::MorphControllerData *p_morph_data, Vector<Ref<Mesh> > p_morph_meshes, bool p_for_morph, bool p_use_mesh_material) {

	bool local_xform_mirror = p_local_xform.basis.determinant() < 0;

	if (p_morph_data) {

		//add morphie target
		ERR_FAIL_COND_V(!p_morph_data->targets.has("MORPH_TARGET"), ERR_INVALID_DATA);
		String mt = p_morph_data->targets["MORPH_TARGET"];
		ERR_FAIL_COND_V(!p_morph_data->sources.has(mt), ERR_INVALID_DATA);
		int morph_targets = p_morph_data->sources[mt].sarray.size();
		for (int i = 0; i < morph_targets; i++) {

			String target = p_morph_data->sources[mt].sarray[i];
			ERR_FAIL_COND_V(!collada.state.mesh_data_map.has(target), ERR_INVALID_DATA);
			String name = collada.state.mesh_data_map[target].name;

			p_mesh->add_blend_shape(name);
		}
		if (p_morph_data->mode == "RELATIVE")
			p_mesh->set_blend_shape_mode(Mesh::BLEND_SHAPE_MODE_RELATIVE);
		else if (p_morph_data->mode == "NORMALIZED")
			p_mesh->set_blend_shape_mode(Mesh::BLEND_SHAPE_MODE_NORMALIZED);
	}

	int surface = 0;
	for (int p_i = 0; p_i < meshdata.primitives.size(); p_i++) {

		const Collada::MeshData::Primitives &p = meshdata.primitives[p_i];

		/* VERTEX SOURCE */
		ERR_FAIL_COND_V(!p.sources.has("VERTEX"), ERR_INVALID_DATA);

		String vertex_src_id = p.sources["VERTEX"].source;
		int vertex_ofs = p.sources["VERTEX"].offset;

		ERR_FAIL_COND_V(!meshdata.vertices.has(vertex_src_id), ERR_INVALID_DATA);

		ERR_FAIL_COND_V(!meshdata.vertices[vertex_src_id].sources.has("POSITION"), ERR_INVALID_DATA);
		String position_src_id = meshdata.vertices[vertex_src_id].sources["POSITION"];

		ERR_FAIL_COND_V(!meshdata.sources.has(position_src_id), ERR_INVALID_DATA);

		const Collada::MeshData::Source *vertex_src = &meshdata.sources[position_src_id];

		/* NORMAL SOURCE */

		const Collada::MeshData::Source *normal_src = NULL;
		int normal_ofs = 0;

		if (p.sources.has("NORMAL")) {

			String normal_source_id = p.sources["NORMAL"].source;
			normal_ofs = p.sources["NORMAL"].offset;
			ERR_FAIL_COND_V(!meshdata.sources.has(normal_source_id), ERR_INVALID_DATA);
			normal_src = &meshdata.sources[normal_source_id];
		}

		const Collada::MeshData::Source *binormal_src = NULL;
		int binormal_ofs = 0;

		if (p.sources.has("TEXBINORMAL")) {

			String binormal_source_id = p.sources["TEXBINORMAL"].source;
			binormal_ofs = p.sources["TEXBINORMAL"].offset;
			ERR_FAIL_COND_V(!meshdata.sources.has(binormal_source_id), ERR_INVALID_DATA);
			binormal_src = &meshdata.sources[binormal_source_id];
		}

		const Collada::MeshData::Source *tangent_src = NULL;
		int tangent_ofs = 0;

		if (p.sources.has("TEXTANGENT")) {

			String tangent_source_id = p.sources["TEXTANGENT"].source;
			tangent_ofs = p.sources["TEXTANGENT"].offset;
			ERR_FAIL_COND_V(!meshdata.sources.has(tangent_source_id), ERR_INVALID_DATA);
			tangent_src = &meshdata.sources[tangent_source_id];
		}

		const Collada::MeshData::Source *uv_src = NULL;
		int uv_ofs = 0;

		if (p.sources.has("TEXCOORD0")) {

			String uv_source_id = p.sources["TEXCOORD0"].source;
			uv_ofs = p.sources["TEXCOORD0"].offset;
			ERR_FAIL_COND_V(!meshdata.sources.has(uv_source_id), ERR_INVALID_DATA);
			uv_src = &meshdata.sources[uv_source_id];
		}

		const Collada::MeshData::Source *uv2_src = NULL;
		int uv2_ofs = 0;

		if (p.sources.has("TEXCOORD1")) {

			String uv2_source_id = p.sources["TEXCOORD1"].source;
			uv2_ofs = p.sources["TEXCOORD1"].offset;
			ERR_FAIL_COND_V(!meshdata.sources.has(uv2_source_id), ERR_INVALID_DATA);
			uv2_src = &meshdata.sources[uv2_source_id];
		}

		const Collada::MeshData::Source *color_src = NULL;
		int color_ofs = 0;

		if (p.sources.has("COLOR")) {

			String color_source_id = p.sources["COLOR"].source;
			color_ofs = p.sources["COLOR"].offset;
			ERR_FAIL_COND_V(!meshdata.sources.has(color_source_id), ERR_INVALID_DATA);
			color_src = &meshdata.sources[color_source_id];
		}

		//find largest source..

		/************************/
		/* ADD WEIGHTS IF EXIST */
		/************************/

		Map<int, Vector<Collada::Vertex::Weight> > pre_weights;

		bool has_weights = false;

		if (skin_controller) {

			const Collada::SkinControllerData::Source *weight_src = NULL;
			int weight_ofs = 0;

			if (skin_controller->weights.sources.has("WEIGHT")) {

				String weight_id = skin_controller->weights.sources["WEIGHT"].source;
				weight_ofs = skin_controller->weights.sources["WEIGHT"].offset;
				if (skin_controller->sources.has(weight_id)) {

					weight_src = &skin_controller->sources[weight_id];
				}
			}

			int joint_ofs = 0;

			if (skin_controller->weights.sources.has("JOINT")) {

				joint_ofs = skin_controller->weights.sources["JOINT"].offset;
			}

			//should be OK, given this was pre-checked.

			int index_ofs = 0;
			int wstride = skin_controller->weights.sources.size();
			for (int w_i = 0; w_i < skin_controller->weights.sets.size(); w_i++) {

				int amount = skin_controller->weights.sets[w_i];

				Vector<Collada::Vertex::Weight> weights;

				for (int a_i = 0; a_i < amount; a_i++) {

					Collada::Vertex::Weight w;

					int read_from = index_ofs + a_i * wstride;
					ERR_FAIL_INDEX_V(read_from + wstride - 1, skin_controller->weights.indices.size(), ERR_INVALID_DATA);
					int weight_index = skin_controller->weights.indices[read_from + weight_ofs];
					ERR_FAIL_INDEX_V(weight_index, weight_src->array.size(), ERR_INVALID_DATA);

					w.weight = weight_src->array[weight_index];

					int bone_index = skin_controller->weights.indices[read_from + joint_ofs];
					if (bone_index == -1)
						continue; //ignore this weight (refers to bind shape)
					ERR_FAIL_INDEX_V(bone_index, bone_remap.size(), ERR_INVALID_DATA);

					w.bone_idx = bone_remap[bone_index];

					weights.push_back(w);
				}

				/* FIX WEIGHTS */

				weights.sort();

				if (weights.size() > 4) {
					//cap to 4 and make weights add up 1
					weights.resize(4);
				}

				//make sure weights always add up to 1
				float total = 0;
				for (int i = 0; i < weights.size(); i++)
					total += weights[i].weight;
				if (total)
					for (int i = 0; i < weights.size(); i++)
						weights[i].weight /= total;

				if (weights.size() == 0 || total == 0) { //if nothing, add a weight to bone 0
					//no weights assigned
					Collada::Vertex::Weight w;
					w.bone_idx = 0;
					w.weight = 1.0;
					weights.clear();
					weights.push_back(w);
				}

				pre_weights[w_i] = weights;

				/*
				for(Set<int>::Element *E=vertex_map[w_i].front();E;E=E->next()) {

					int dst = E->get();
					ERR_EXPLAIN("invalid vertex index in array");
					ERR_FAIL_INDEX_V(dst,vertex_array.size(),ERR_INVALID_DATA);
					vertex_array[dst].weights=weights;

				}*/

				index_ofs += wstride * amount;
			}

			//vertices need to be localized
			has_weights = true;
		}

		Set<Collada::Vertex> vertex_set; //vertex set will be the vertices
		List<int> indices_list; //indices will be the indices
		//Map<int,Set<int> > vertex_map; //map vertices (for setting skinning/morph)

		/**************************/
		/* CREATE PRIMITIVE ARRAY */
		/**************************/

		// The way collada uses indices is more optimal, and friendlier with 3D modelling software,
		// because it can index everything, not only vertices (similar to how the WII works).
		// This is, however, more incompatible with standard video cards, so arrays must be converted.
		// Must convert to GL/DX format.

		int _prim_ofs = 0;
		int vertidx = 0;
		for (int p_i = 0; p_i < p.count; p_i++) {

			int amount;
			if (p.polygons.size()) {

				ERR_FAIL_INDEX_V(p_i, p.polygons.size(), ERR_INVALID_DATA);
				amount = p.polygons[p_i];
			} else {
				amount = 3; //triangles;
			}

			//COLLADA_PRINT("amount: "+itos(amount));

			int prev2[2] = { 0, 0 };

			for (int j = 0; j < amount; j++) {

				int src = _prim_ofs;
				//_prim_ofs+=p.sources.size()

				ERR_FAIL_INDEX_V(src, p.indices.size(), ERR_INVALID_DATA);

				Collada::Vertex vertex;
				if (!p_optimize)
					vertex.uid = vertidx++;

				int vertex_index = p.indices[src + vertex_ofs]; //used for index field (later used by controllers)
				int vertex_pos = (vertex_src->stride ? vertex_src->stride : 3) * vertex_index;
				ERR_FAIL_INDEX_V(vertex_pos, vertex_src->array.size(), ERR_INVALID_DATA);
				vertex.vertex = Vector3(vertex_src->array[vertex_pos + 0], vertex_src->array[vertex_pos + 1], vertex_src->array[vertex_pos + 2]);

				if (pre_weights.has(vertex_index)) {
					vertex.weights = pre_weights[vertex_index];
				}

				if (normal_src) {

					int normal_pos = (normal_src->stride ? normal_src->stride : 3) * p.indices[src + normal_ofs];
					ERR_FAIL_INDEX_V(normal_pos, normal_src->array.size(), ERR_INVALID_DATA);
					vertex.normal = Vector3(normal_src->array[normal_pos + 0], normal_src->array[normal_pos + 1], normal_src->array[normal_pos + 2]);
					vertex.normal = vertex.normal.snapped(0.001);

					if (tangent_src && binormal_src) {

						int binormal_pos = (binormal_src->stride ? binormal_src->stride : 3) * p.indices[src + binormal_ofs];
						ERR_FAIL_INDEX_V(binormal_pos, binormal_src->array.size(), ERR_INVALID_DATA);
						Vector3 binormal = Vector3(binormal_src->array[binormal_pos + 0], binormal_src->array[binormal_pos + 1], binormal_src->array[binormal_pos + 2]);

						int tangent_pos = (tangent_src->stride ? tangent_src->stride : 3) * p.indices[src + tangent_ofs];
						ERR_FAIL_INDEX_V(tangent_pos, tangent_src->array.size(), ERR_INVALID_DATA);
						Vector3 tangent = Vector3(tangent_src->array[tangent_pos + 0], tangent_src->array[tangent_pos + 1], tangent_src->array[tangent_pos + 2]);

						vertex.tangent.normal = tangent;
						vertex.tangent.d = vertex.normal.cross(tangent).dot(binormal) > 0 ? 1 : -1;
					}
				}

				if (uv_src) {

					int uv_pos = (uv_src->stride ? uv_src->stride : 2) * p.indices[src + uv_ofs];
					ERR_FAIL_INDEX_V(uv_pos, uv_src->array.size(), ERR_INVALID_DATA);
					vertex.uv = Vector3(uv_src->array[uv_pos + 0], 1.0 - uv_src->array[uv_pos + 1], 0);
				}

				if (uv2_src) {

					int uv2_pos = (uv2_src->stride ? uv2_src->stride : 2) * p.indices[src + uv2_ofs];
					ERR_FAIL_INDEX_V(uv2_pos, uv2_src->array.size(), ERR_INVALID_DATA);
					vertex.uv2 = Vector3(uv2_src->array[uv2_pos + 0], 1.0 - uv2_src->array[uv2_pos + 1], 0);
				}

				if (color_src) {

					int color_pos = (color_src->stride ? color_src->stride : 3) * p.indices[src + color_ofs]; // colors are RGB in collada..
					ERR_FAIL_INDEX_V(color_pos, color_src->array.size(), ERR_INVALID_DATA);
					vertex.color = Color(color_src->array[color_pos + 0], color_src->array[color_pos + 1], color_src->array[color_pos + 2], (color_src->stride > 3) ? color_src->array[color_pos + 3] : 1.0);
				}

#ifndef NO_UP_AXIS_SWAP
				if (collada.state.up_axis == Vector3::AXIS_Z) {

					SWAP(vertex.vertex.z, vertex.vertex.y);
					vertex.vertex.z = -vertex.vertex.z;
					SWAP(vertex.normal.z, vertex.normal.y);
					vertex.normal.z = -vertex.normal.z;
					SWAP(vertex.tangent.normal.z, vertex.tangent.normal.y);
					vertex.tangent.normal.z = -vertex.tangent.normal.z;
				}

#endif

				vertex.fix_unit_scale(collada);
				int index = 0;
				//COLLADA_PRINT("vertex: "+vertex.vertex);

				if (vertex_set.has(vertex)) {

					index = vertex_set.find(vertex)->get().idx;
				} else {

					index = vertex_set.size();
					vertex.idx = index;
					vertex_set.insert(vertex);
				}

				/*	if (!vertex_map.has(vertex_index))
					vertex_map[vertex_index]=Set<int>();
				vertex_map[vertex_index].insert(index); //should be outside..*/
				//build triangles if needed
				if (j == 0)
					prev2[0] = index;

				if (j >= 2) {
					//insert indices in reverse order (collada uses CCW as frontface)
					if (local_xform_mirror) {

						indices_list.push_back(prev2[0]);
						indices_list.push_back(prev2[1]);
						indices_list.push_back(index);

					} else {
						indices_list.push_back(prev2[0]);
						indices_list.push_back(index);
						indices_list.push_back(prev2[1]);
					}
				}

				prev2[1] = index;
				_prim_ofs += p.vertex_size;
			}
		}

		Vector<Collada::Vertex> vertex_array; //there we go, vertex array

		vertex_array.resize(vertex_set.size());
		for (Set<Collada::Vertex>::Element *F = vertex_set.front(); F; F = F->next()) {

			vertex_array[F->get().idx] = F->get();
		}

		if (has_weights) {

			//if skeleton, localize
			Transform local_xform = p_local_xform;
			for (int i = 0; i < vertex_array.size(); i++) {

				vertex_array[i].vertex = local_xform.xform(vertex_array[i].vertex);
				vertex_array[i].normal = local_xform.basis.xform(vertex_array[i].normal).normalized();
				vertex_array[i].tangent.normal = local_xform.basis.xform(vertex_array[i].tangent.normal).normalized();
				if (local_xform_mirror) {
					//i shouldn't do this? wtf?
					//vertex_array[i].normal*=-1.0;
					//vertex_array[i].tangent.normal*=-1.0;
				}
			}
		}

		PoolVector<int> index_array;
		index_array.resize(indices_list.size());
		PoolVector<int>::Write index_arrayw = index_array.write();

		int iidx = 0;
		for (List<int>::Element *F = indices_list.front(); F; F = F->next()) {

			index_arrayw[iidx++] = F->get();
		}

		index_arrayw = PoolVector<int>::Write();

		/*****************/
		/* MAKE SURFACES  */
		/*****************/

		{

			Ref<SpatialMaterial> material;

			//find material
			Mesh::PrimitiveType primitive = Mesh::PRIMITIVE_TRIANGLES;

			{

				if (p_material_map.has(p.material)) {
					String target = p_material_map[p.material].target;

					if (!material_cache.has(target)) {
						Error err = _create_material(target);
						if (!err)
							material = material_cache[target];
					} else
						material = material_cache[target];

				} else if (p.material != "") {
					print_line("Warning, unreferenced material in geometry instance: " + p.material);
				}
			}

			PoolVector<Vector3> final_vertex_array;
			PoolVector<Vector3> final_normal_array;
			PoolVector<float> final_tangent_array;
			PoolVector<Color> final_color_array;
			PoolVector<Vector3> final_uv_array;
			PoolVector<Vector3> final_uv2_array;
			PoolVector<int> final_bone_array;
			PoolVector<float> final_weight_array;

			uint32_t final_format = 0;

			//create format
			final_format = Mesh::ARRAY_FORMAT_VERTEX | Mesh::ARRAY_FORMAT_INDEX;

			if (normal_src) {
				final_format |= Mesh::ARRAY_FORMAT_NORMAL;
				if (uv_src && binormal_src && tangent_src) {
					final_format |= Mesh::ARRAY_FORMAT_TANGENT;
				}
			}

			if (color_src)
				final_format |= Mesh::ARRAY_FORMAT_COLOR;
			if (uv_src)
				final_format |= Mesh::ARRAY_FORMAT_TEX_UV;
			if (uv2_src)
				final_format |= Mesh::ARRAY_FORMAT_TEX_UV2;

			if (has_weights) {
				final_format |= Mesh::ARRAY_FORMAT_WEIGHTS;
				final_format |= Mesh::ARRAY_FORMAT_BONES;
			}

			//set arrays

			int vlen = vertex_array.size();
			{ //vertices

				PoolVector<Vector3> varray;
				varray.resize(vertex_array.size());

				PoolVector<Vector3>::Write varrayw = varray.write();

				for (int k = 0; k < vlen; k++)
					varrayw[k] = vertex_array[k].vertex;

				varrayw = PoolVector<Vector3>::Write();
				final_vertex_array = varray;
			}

			if (uv_src) { //compute uv first, may be needed for computing tangent/bionrmal
				PoolVector<Vector3> uvarray;
				uvarray.resize(vertex_array.size());
				PoolVector<Vector3>::Write uvarrayw = uvarray.write();

				for (int k = 0; k < vlen; k++) {
					uvarrayw[k] = vertex_array[k].uv;
				}

				uvarrayw = PoolVector<Vector3>::Write();
				final_uv_array = uvarray;
			}

			if (uv2_src) { //compute uv first, may be needed for computing tangent/bionrmal
				PoolVector<Vector3> uv2array;
				uv2array.resize(vertex_array.size());
				PoolVector<Vector3>::Write uv2arrayw = uv2array.write();

				for (int k = 0; k < vlen; k++) {
					uv2arrayw[k] = vertex_array[k].uv2;
				}

				uv2arrayw = PoolVector<Vector3>::Write();
				final_uv2_array = uv2array;
			}

			if (normal_src) {
				PoolVector<Vector3> narray;
				narray.resize(vertex_array.size());
				PoolVector<Vector3>::Write narrayw = narray.write();

				for (int k = 0; k < vlen; k++) {
					narrayw[k] = vertex_array[k].normal;
				}

				narrayw = PoolVector<Vector3>::Write();
				final_normal_array = narray;

				/*
				PoolVector<Vector3> altnaray;
				_generate_normals(index_array,final_vertex_array,altnaray);

				for(int i=0;i<altnaray.size();i++)
					print_line(rtos(altnaray[i].dot(final_normal_array[i])));
				*/

			} else if (primitive == Mesh::PRIMITIVE_TRIANGLES) {
				//generate normals (even if unused later)

				_generate_normals(index_array, final_vertex_array, final_normal_array);
				if (OS::get_singleton()->is_stdout_verbose())
					print_line("Collada: Triangle mesh lacks normals, so normals were generated.");
				final_format |= Mesh::ARRAY_FORMAT_NORMAL;
			}

			if (final_normal_array.size() && uv_src && binormal_src && tangent_src && !force_make_tangents) {

				PoolVector<real_t> tarray;
				tarray.resize(vertex_array.size() * 4);
				PoolVector<real_t>::Write tarrayw = tarray.write();

				for (int k = 0; k < vlen; k++) {
					tarrayw[k * 4 + 0] = vertex_array[k].tangent.normal.x;
					tarrayw[k * 4 + 1] = vertex_array[k].tangent.normal.y;
					tarrayw[k * 4 + 2] = vertex_array[k].tangent.normal.z;
					tarrayw[k * 4 + 3] = vertex_array[k].tangent.d;
				}

				tarrayw = PoolVector<real_t>::Write();

				final_tangent_array = tarray;
			} else if (final_normal_array.size() && primitive == Mesh::PRIMITIVE_TRIANGLES && final_uv_array.size() && (force_make_tangents || (material.is_valid()))) {
				//if this uses triangles, there are uvs and the material is using a normalmap, generate tangents and binormals, because they WILL be needed
				//generate binormals/tangents
				_generate_tangents_and_binormals(index_array, final_vertex_array, final_uv_array, final_normal_array, final_tangent_array);
				final_format |= Mesh::ARRAY_FORMAT_TANGENT;
				if (OS::get_singleton()->is_stdout_verbose())
					print_line("Collada: Triangle mesh lacks tangents (And normalmap was used), so tangents were generated.");
			}

			if (color_src) {
				PoolVector<Color> colorarray;
				colorarray.resize(vertex_array.size());
				PoolVector<Color>::Write colorarrayw = colorarray.write();

				for (int k = 0; k < vlen; k++) {
					colorarrayw[k] = vertex_array[k].color;
				}

				colorarrayw = PoolVector<Color>::Write();

				final_color_array = colorarray;
			}

			if (has_weights) {
				PoolVector<float> weightarray;
				PoolVector<int> bonearray;

				weightarray.resize(vertex_array.size() * 4);
				PoolVector<float>::Write weightarrayw = weightarray.write();
				bonearray.resize(vertex_array.size() * 4);
				PoolVector<int>::Write bonearrayw = bonearray.write();

				for (int k = 0; k < vlen; k++) {
					float sum = 0;

					for (int l = 0; l < VS::ARRAY_WEIGHTS_SIZE; l++) {
						if (l < vertex_array[k].weights.size()) {
							weightarrayw[k * VS::ARRAY_WEIGHTS_SIZE + l] = vertex_array[k].weights[l].weight;
							sum += weightarrayw[k * VS::ARRAY_WEIGHTS_SIZE + l];
							bonearrayw[k * VS::ARRAY_WEIGHTS_SIZE + l] = int(vertex_array[k].weights[l].bone_idx);
							//COLLADA_PRINT(itos(k)+": "+rtos(bonearrayw[k*VS::ARRAY_WEIGHTS_SIZE+l])+":"+rtos(weightarray[k*VS::ARRAY_WEIGHTS_SIZE+l]));
						} else {

							weightarrayw[k * VS::ARRAY_WEIGHTS_SIZE + l] = 0;
							bonearrayw[k * VS::ARRAY_WEIGHTS_SIZE + l] = 0;
						}
					}
					/*
					if (sum<0.8)
						COLLADA_PRINT("ERROR SUMMING INDEX "+itos(k)+" had weights: "+itos(vertex_array[k].weights.size()));
					*/
				}

				weightarrayw = PoolVector<float>::Write();
				bonearrayw = PoolVector<int>::Write();

				final_weight_array = weightarray;
				final_bone_array = bonearray;
			}

			////////////////////////////
			// FINALLY CREATE SUFRACE //
			////////////////////////////

			Array d;
			d.resize(VS::ARRAY_MAX);

			d[Mesh::ARRAY_INDEX] = index_array;
			d[Mesh::ARRAY_VERTEX] = final_vertex_array;

			if (final_normal_array.size())
				d[Mesh::ARRAY_NORMAL] = final_normal_array;
			if (final_tangent_array.size())
				d[Mesh::ARRAY_TANGENT] = final_tangent_array;
			if (final_uv_array.size())
				d[Mesh::ARRAY_TEX_UV] = final_uv_array;
			if (final_uv2_array.size())
				d[Mesh::ARRAY_TEX_UV2] = final_uv2_array;
			if (final_color_array.size())
				d[Mesh::ARRAY_COLOR] = final_color_array;
			if (final_weight_array.size())
				d[Mesh::ARRAY_WEIGHTS] = final_weight_array;
			if (final_bone_array.size())
				d[Mesh::ARRAY_BONES] = final_bone_array;

			Array mr;

////////////////////////////
// THEN THE MORPH TARGETS //
////////////////////////////
#if 0
			if (p_morph_data) {

				//add morphie target
				ERR_FAIL_COND_V( !p_morph_data->targets.has("MORPH_TARGET"), ERR_INVALID_DATA );
				String mt = p_morph_data->targets["MORPH_TARGET"];
				ERR_FAIL_COND_V( !p_morph_data->sources.has(mt), ERR_INVALID_DATA);
				int morph_targets = p_morph_data->sources[mt].sarray.size();
				mr.resize(morph_targets);

				for(int j=0;j<morph_targets;j++) {

					Array mrt;
					mrt.resize(VS::ARRAY_MAX);

					String target = p_morph_data->sources[mt].sarray[j];
					ERR_FAIL_COND_V( !collada.state.mesh_data_map.has(target), ERR_INVALID_DATA );
					String name = collada.state.mesh_data_map[target].name;
					Collada::MeshData &md = collada.state.mesh_data_map[target];

					// collada in itself supports morphing everything. However, the spec is unclear and no examples or exporters that
					// morph anything but "POSITIONS" seem to exit. Because of this, normals and binormals/tangents have to be regenerated here,
					// which may result in inaccurate (but most of the time good enough) results.

					PoolVector<Vector3> vertices;
					vertices.resize(vlen);

					ERR_FAIL_COND_V( md.vertices.size() != 1, ERR_INVALID_DATA);
					String vertex_src_id=md.vertices.front()->key();
					ERR_FAIL_COND_V(!md.vertices[vertex_src_id].sources.has("POSITION"),ERR_INVALID_DATA);
					String position_src_id = md.vertices[vertex_src_id].sources["POSITION"];

					ERR_FAIL_COND_V(!md.sources.has(position_src_id),ERR_INVALID_DATA);

					const Collada::MeshData::Source *m=&md.sources[position_src_id];

					ERR_FAIL_COND_V( m->array.size() != vertex_src->array.size(), ERR_INVALID_DATA);
					int stride=m->stride;
					if (stride==0)
						stride=3;


					//read vertices from morph target
					PoolVector<Vector3>::Write vertw = vertices.write();

					for(int m_i=0;m_i<m->array.size()/stride;m_i++) {

						int pos = m_i*stride;
						Vector3 vtx( m->array[pos+0], m->array[pos+1], m->array[pos+2] );

#ifndef NO_UP_AXIS_SWAP
						if (collada.state.up_axis==Vector3::AXIS_Z) {

							SWAP( vtx.z, vtx.y );
							vtx.z = -vtx.z;

						}
#endif

						Collada::Vertex vertex;
						vertex.vertex=vtx;
						vertex.fix_unit_scale(collada);
						vtx=vertex.vertex;

						vtx = p_local_xform.xform(vtx);


						if (vertex_map.has(m_i)) { //vertex may no longer be here, don't bother converting


							for (Set<int> ::Element *E=vertex_map[m_i].front() ; E; E=E->next() ) {

								vertw[E->get()]=vtx;
							}
						}
					}


					//vertices are in place, now generate everything else
					vertw = PoolVector<Vector3>::Write();
					PoolVector<Vector3> normals;
					PoolVector<float> tangents;
					print_line("vertex source id: "+vertex_src_id);
					if(md.vertices[vertex_src_id].sources.has("NORMAL")){
						//has normals
						normals.resize(vlen);
						//std::cout << "has normals" << std::endl;
						String normal_src_id = md.vertices[vertex_src_id].sources["NORMAL"];
						//std::cout << "normals source: "<< normal_src_id.utf8().get_data() <<std::endl;
						ERR_FAIL_COND_V(!md.sources.has(normal_src_id),ERR_INVALID_DATA);

						const Collada::MeshData::Source *m=&md.sources[normal_src_id];

						ERR_FAIL_COND_V( m->array.size() != vertex_src->array.size(), ERR_INVALID_DATA);
						int stride=m->stride;
						if (stride==0)
							stride=3;


						//read normals from morph target
						PoolVector<Vector3>::Write vertw = normals.write();

						for(int m_i=0;m_i<m->array.size()/stride;m_i++) {

							int pos = m_i*stride;
							Vector3 vtx( m->array[pos+0], m->array[pos+1], m->array[pos+2] );

#ifndef NO_UP_AXIS_SWAP
							if (collada.state.up_axis==Vector3::AXIS_Z) {

								SWAP( vtx.z, vtx.y );
								vtx.z = -vtx.z;

							}
#endif

							Collada::Vertex vertex;
							vertex.vertex=vtx;
							vertex.fix_unit_scale(collada);
							vtx=vertex.vertex;

							vtx = p_local_xform.xform(vtx);


							if (vertex_map.has(m_i)) { //vertex may no longer be here, don't bother converting


								for (Set<int> ::Element *E=vertex_map[m_i].front() ; E; E=E->next() ) {

									vertw[E->get()]=vtx;
								}
							}
						}

						print_line("using built-in normals");
					}else{
						print_line("generating normals");
						_generate_normals(index_array,vertices,normals);//no normals
					}
					if (final_tangent_array.size() && final_uv_array.size()) {

						_generate_tangents_and_binormals(index_array,vertices,final_uv_array,normals,tangents);

					}

					mrt[Mesh::ARRAY_VERTEX]=vertices;

					mrt[Mesh::ARRAY_NORMAL]=normals;
					if (tangents.size())
						mrt[Mesh::ARRAY_TANGENT]=tangents;
					if (final_uv_array.size())
						mrt[Mesh::ARRAY_TEX_UV]=final_uv_array;
					if (final_uv2_array.size())
						mrt[Mesh::ARRAY_TEX_UV2]=final_uv2_array;
					if (final_color_array.size())
						mrt[Mesh::ARRAY_COLOR]=final_color_array;

					mr[j]=mrt;

				}

			}

#endif
			for (int mi = 0; mi < p_morph_meshes.size(); mi++) {

				//print_line("want surface "+itos(mi)+" has "+itos(p_morph_meshes[mi]->get_surface_count()));
				Array a = p_morph_meshes[mi]->surface_get_arrays(surface);
				//add valid weight and bone arrays if they exist, TODO check if they are unique to shape (generally not)

				if (final_weight_array.size())
					a[Mesh::ARRAY_WEIGHTS] = final_weight_array;
				if (final_bone_array.size())
					a[Mesh::ARRAY_BONES] = final_bone_array;

				a[Mesh::ARRAY_INDEX] = Variant();
				//a.resize(Mesh::ARRAY_MAX); //no need for index
				mr.push_back(a);
			}

			p_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, d, mr, p_for_morph ? 0 : Mesh::ARRAY_COMPRESS_DEFAULT);

			if (material.is_valid()) {
				if (p_use_mesh_material) {
					p_mesh->surface_set_material(surface, material);
				}
				p_mesh->surface_set_name(surface, material->get_name());
			}
		}

		/*****************/
		/* FIND MATERIAL */
		/*****************/

		surface++;
	}

	return OK;
}

Error ColladaImport::_create_resources(Collada::Node *p_node) {

	if (p_node->type == Collada::Node::TYPE_GEOMETRY && node_map.has(p_node->id)) {

		Spatial *node = node_map[p_node->id].node;
		Collada::NodeGeometry *ng = static_cast<Collada::NodeGeometry *>(p_node);

		if (node->cast_to<Path>()) {

			Path *path = node->cast_to<Path>();

			String curve = ng->source;

			if (curve_cache.has(ng->source)) {

				path->set_curve(curve_cache[ng->source]);
			} else {

				Ref<Curve3D> c = memnew(Curve3D);

				const Collada::CurveData &cd = collada.state.curve_data_map[ng->source];

				ERR_FAIL_COND_V(!cd.control_vertices.has("POSITION"), ERR_INVALID_DATA);
				ERR_FAIL_COND_V(!cd.control_vertices.has("IN_TANGENT"), ERR_INVALID_DATA);
				ERR_FAIL_COND_V(!cd.control_vertices.has("OUT_TANGENT"), ERR_INVALID_DATA);
				ERR_FAIL_COND_V(!cd.control_vertices.has("INTERPOLATION"), ERR_INVALID_DATA);

				ERR_FAIL_COND_V(!cd.sources.has(cd.control_vertices["POSITION"]), ERR_INVALID_DATA);
				const Collada::CurveData::Source &vertices = cd.sources[cd.control_vertices["POSITION"]];
				ERR_FAIL_COND_V(vertices.stride != 3, ERR_INVALID_DATA);

				ERR_FAIL_COND_V(!cd.sources.has(cd.control_vertices["IN_TANGENT"]), ERR_INVALID_DATA);
				const Collada::CurveData::Source &in_tangents = cd.sources[cd.control_vertices["IN_TANGENT"]];
				ERR_FAIL_COND_V(in_tangents.stride != 3, ERR_INVALID_DATA);

				ERR_FAIL_COND_V(!cd.sources.has(cd.control_vertices["OUT_TANGENT"]), ERR_INVALID_DATA);
				const Collada::CurveData::Source &out_tangents = cd.sources[cd.control_vertices["OUT_TANGENT"]];
				ERR_FAIL_COND_V(out_tangents.stride != 3, ERR_INVALID_DATA);

				ERR_FAIL_COND_V(!cd.sources.has(cd.control_vertices["INTERPOLATION"]), ERR_INVALID_DATA);
				const Collada::CurveData::Source &interps = cd.sources[cd.control_vertices["INTERPOLATION"]];
				ERR_FAIL_COND_V(interps.stride != 1, ERR_INVALID_DATA);

				const Collada::CurveData::Source *tilts = NULL;
				if (cd.control_vertices.has("TILT") && cd.sources.has(cd.control_vertices["TILT"]))
					tilts = &cd.sources[cd.control_vertices["TILT"]];

				if (tilts) {
					print_line("FOUND TILTS!!!");
				}
				int pc = vertices.array.size() / 3;
				for (int i = 0; i < pc; i++) {

					Vector3 pos(vertices.array[i * 3 + 0], vertices.array[i * 3 + 1], vertices.array[i * 3 + 2]);
					Vector3 in(in_tangents.array[i * 3 + 0], in_tangents.array[i * 3 + 1], in_tangents.array[i * 3 + 2]);
					Vector3 out(out_tangents.array[i * 3 + 0], out_tangents.array[i * 3 + 1], out_tangents.array[i * 3 + 2]);

#ifndef NO_UP_AXIS_SWAP
					if (collada.state.up_axis == Vector3::AXIS_Z) {

						SWAP(pos.y, pos.z);
						pos.z = -pos.z;
						SWAP(in.y, in.z);
						in.z = -in.z;
						SWAP(out.y, out.z);
						out.z = -out.z;
					}
#endif
					pos *= collada.state.unit_scale;
					in *= collada.state.unit_scale;
					out *= collada.state.unit_scale;

					c->add_point(pos, in - pos, out - pos);
					if (tilts)
						c->set_point_tilt(i, tilts->array[i]);
				}

				curve_cache[ng->source] = c;
				path->set_curve(c);
			}
		}

		if (node->cast_to<MeshInstance>()) {

			Collada::NodeGeometry *ng = static_cast<Collada::NodeGeometry *>(p_node);

			MeshInstance *mi = node->cast_to<MeshInstance>();

			ERR_FAIL_COND_V(!mi, ERR_BUG);

			Collada::SkinControllerData *skin = NULL;
			Collada::MorphControllerData *morph = NULL;
			String meshid;
			Transform apply_xform;
			Vector<int> bone_remap;
			Vector<Ref<Mesh> > morphs;

			print_line("mesh: " + String(mi->get_name()));

			if (ng->controller) {

				print_line("has controller");

				String ngsource = ng->source;

				if (collada.state.skin_controller_data_map.has(ngsource)) {

					ERR_FAIL_COND_V(!collada.state.skin_controller_data_map.has(ngsource), ERR_INVALID_DATA);
					skin = &collada.state.skin_controller_data_map[ngsource];

					Vector<String> skeletons = ng->skeletons;

					ERR_FAIL_COND_V(skeletons.empty(), ERR_INVALID_DATA);

					String skname = skeletons[0];
					if (!node_map.has(skname)) {
						print_line("no node for skeleton " + skname);
					}
					ERR_FAIL_COND_V(!node_map.has(skname), ERR_INVALID_DATA);
					NodeMap nmsk = node_map[skname];
					Skeleton *sk = nmsk.node->cast_to<Skeleton>();
					ERR_FAIL_COND_V(!sk, ERR_INVALID_DATA);
					ERR_FAIL_COND_V(!skeleton_bone_map.has(sk), ERR_INVALID_DATA);
					Map<String, int> &bone_remap_map = skeleton_bone_map[sk];

					meshid = skin->base;

					if (collada.state.morph_controller_data_map.has(meshid)) {
						//it's a morph!!
						morph = &collada.state.morph_controller_data_map[meshid];
						ngsource = meshid;
						meshid = morph->mesh;
					} else {
						ngsource = "";
					}

					if (apply_mesh_xform_to_vertices) {
						apply_xform = collada.fix_transform(p_node->default_transform);
						node->set_transform(Transform());
					} else {
						apply_xform = Transform();
					}

					Collada::SkinControllerData::Source *joint_src = NULL;

					ERR_FAIL_COND_V(!skin->weights.sources.has("JOINT"), ERR_INVALID_DATA);

					String joint_id = skin->weights.sources["JOINT"].source;
					ERR_FAIL_COND_V(!skin->sources.has(joint_id), ERR_INVALID_DATA);

					joint_src = &skin->sources[joint_id];

					bone_remap.resize(joint_src->sarray.size());

					for (int i = 0; i < bone_remap.size(); i++) {

						String str = joint_src->sarray[i];
						if (!bone_remap_map.has(str)) {
							print_line("bone not found for remap: " + str);
							print_line("in skeleton: " + skname);
						}
						ERR_FAIL_COND_V(!bone_remap_map.has(str), ERR_INVALID_DATA);
						bone_remap[i] = bone_remap_map[str];
					}
				}

				if (collada.state.morph_controller_data_map.has(ngsource)) {
					print_line("is morph " + ngsource);
					//it's a morph!!
					morph = &collada.state.morph_controller_data_map[ngsource];
					meshid = morph->mesh;
					printf("KKmorph: %p\n", morph);
					print_line("morph mshid: " + meshid);

					Vector<String> targets;

					morph->targets.has("MORPH_TARGET");
					String target = morph->targets["MORPH_TARGET"];
					bool valid = false;
					if (morph->sources.has(target)) {
						valid = true;
						Vector<String> names = morph->sources[target].sarray;
						for (int i = 0; i < names.size(); i++) {

							String meshid = names[i];
							if (collada.state.mesh_data_map.has(meshid)) {
								Ref<Mesh> mesh = Ref<Mesh>(memnew(Mesh));
								const Collada::MeshData &meshdata = collada.state.mesh_data_map[meshid];
								Error err = _create_mesh_surfaces(false, mesh, ng->material_map, meshdata, apply_xform, bone_remap, skin, NULL, Vector<Ref<Mesh> >(), true);
								ERR_FAIL_COND_V(err, err);

								morphs.push_back(mesh);
							} else {
								valid = false;
							}
						}
					}

					if (!valid)
						morphs.clear();

					ngsource = "";
				}

				if (ngsource != "") {
					ERR_EXPLAIN("Controller Instance Source '" + ngsource + "' is neither skin or morph!");
					ERR_FAIL_V(ERR_INVALID_DATA);
				}

			} else {
				meshid = ng->source;
			}

			Ref<Mesh> mesh;
			if (mesh_cache.has(meshid)) {
				mesh = mesh_cache[meshid];
			} else {
				if (collada.state.mesh_data_map.has(meshid)) {
					//bleh, must ignore invalid

					ERR_FAIL_COND_V(!collada.state.mesh_data_map.has(meshid), ERR_INVALID_DATA);
					mesh = Ref<Mesh>(memnew(Mesh));
					const Collada::MeshData &meshdata = collada.state.mesh_data_map[meshid];
					mesh->set_name(meshdata.name);
					Error err = _create_mesh_surfaces(morphs.size() == 0, mesh, ng->material_map, meshdata, apply_xform, bone_remap, skin, morph, morphs, false, use_mesh_builtin_materials);
					ERR_FAIL_COND_V(err, err);

					mesh_cache[meshid] = mesh;
				} else {

					print_line("Warning, will not import geometry: " + meshid);
				}
			}

			if (!mesh.is_null()) {

				mi->set_mesh(mesh);
				if (!use_mesh_builtin_materials) {
					const Collada::MeshData &meshdata = collada.state.mesh_data_map[meshid];

					for (int i = 0; i < meshdata.primitives.size(); i++) {

						String matname = meshdata.primitives[i].material;

						if (ng->material_map.has(matname)) {
							String target = ng->material_map[matname].target;

							Ref<Material> material;
							if (!material_cache.has(target)) {
								Error err = _create_material(target);
								if (!err)
									material = material_cache[target];
							} else
								material = material_cache[target];

							mi->set_surface_material(i, material);
						} else if (matname != "") {
							print_line("Warning, unreferenced material in geometry instance: " + matname);
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < p_node->children.size(); i++) {

		Error err = _create_resources(p_node->children[i]);
		if (err)
			return err;
	}
	return OK;
}

Error ColladaImport::load(const String &p_path, int p_flags, bool p_force_make_tangents) {

	Error err = collada.load(p_path, p_flags);
	ERR_FAIL_COND_V(err, err);

	force_make_tangents = p_force_make_tangents;
	ERR_FAIL_COND_V(!collada.state.visual_scene_map.has(collada.state.root_visual_scene), ERR_INVALID_DATA);
	Collada::VisualScene &vs = collada.state.visual_scene_map[collada.state.root_visual_scene];

	scene = memnew(Spatial); // root

	//determine what's going on with the lights
	for (int i = 0; i < vs.root_nodes.size(); i++) {

		_pre_process_lights(vs.root_nodes[i]);
	}
	//import scene

	for (int i = 0; i < vs.root_nodes.size(); i++) {

		Error err = _create_scene_skeletons(vs.root_nodes[i]);
		if (err != OK) {
			memdelete(scene);
			ERR_FAIL_COND_V(err, err);
		}
	}

	for (int i = 0; i < vs.root_nodes.size(); i++) {

		Error err = _create_scene(vs.root_nodes[i], scene);
		if (err != OK) {
			memdelete(scene);
			ERR_FAIL_COND_V(err, err);
		}

		Error err2 = _create_resources(vs.root_nodes[i]);
		if (err2 != OK) {
			memdelete(scene);
			ERR_FAIL_COND_V(err2, err2);
		}
	}

	//optatively, set unit scale in the root
	scene->set_transform(collada.get_root_transform());

	return OK;
}

void ColladaImport::_fix_param_animation_tracks() {

	for (Map<String, Collada::Node *>::Element *E = collada.state.scene_map.front(); E; E = E->next()) {

		Collada::Node *n = E->get();
		switch (n->type) {

			case Collada::Node::TYPE_NODE: {
				// ? do nothing
			} break;
			case Collada::Node::TYPE_JOINT: {

			} break;
			case Collada::Node::TYPE_SKELETON: {

			} break;
			case Collada::Node::TYPE_LIGHT: {

			} break;
			case Collada::Node::TYPE_CAMERA: {

			} break;
			case Collada::Node::TYPE_GEOMETRY: {

				Collada::NodeGeometry *ng = static_cast<Collada::NodeGeometry *>(n);
				// test source(s)
				String source = ng->source;

				while (source != "") {

					if (collada.state.skin_controller_data_map.has(source)) {

						const Collada::SkinControllerData &skin = collada.state.skin_controller_data_map[source];

						//nothing to animate here i think

						source = skin.base;
					} else if (collada.state.morph_controller_data_map.has(source)) {

						const Collada::MorphControllerData &morph = collada.state.morph_controller_data_map[source];

						if (morph.targets.has("MORPH_WEIGHT") && morph.targets.has("MORPH_TARGET")) {

							String weights = morph.targets["MORPH_WEIGHT"];
							String targets = morph.targets["MORPH_TARGET"];
							//fails here

							if (morph.sources.has(targets) && morph.sources.has(weights)) {
								const Collada::MorphControllerData::Source &weight_src = morph.sources[weights];
								const Collada::MorphControllerData::Source &target_src = morph.sources[targets];

								ERR_FAIL_COND(weight_src.array.size() != target_src.sarray.size());

								for (int i = 0; i < weight_src.array.size(); i++) {

									String track_name = weights + "(" + itos(i) + ")";
									String mesh_name = target_src.sarray[i];
									if (collada.state.mesh_name_map.has(mesh_name) && collada.state.referenced_tracks.has(track_name)) {

										const Vector<int> &rt = collada.state.referenced_tracks[track_name];

										for (int rti = 0; rti < rt.size(); rti++) {
											Collada::AnimationTrack *at = &collada.state.animation_tracks[rt[rti]];

											at->target = E->key();
											at->param = "morph/" + collada.state.mesh_name_map[mesh_name];
											at->property = true;
											//at->param
										}
									}
								}
							}
						}
						source = morph.mesh;
					} else {

						source = ""; // for now nothing else supported
					}
				}

			} break;
		}
	}
}

void ColladaImport::create_animations(bool p_make_tracks_in_all_bones, bool p_import_value_tracks) {

	_fix_param_animation_tracks();
	for (int i = 0; i < collada.state.animation_clips.size(); i++) {

		for (int j = 0; j < collada.state.animation_clips[i].tracks.size(); j++)
			tracks_in_clips.insert(collada.state.animation_clips[i].tracks[j]);
	}

	for (int i = 0; i < collada.state.animation_tracks.size(); i++) {

		Collada::AnimationTrack &at = collada.state.animation_tracks[i];
		//print_line("CHANNEL: "+at.target+" PARAM: "+at.param);

		String node;

		if (!node_map.has(at.target)) {

			if (node_name_map.has(at.target)) {

				node = node_name_map[at.target];
			} else {
				print_line("Coudlnt find node: " + at.target);
				continue;
			}
		} else {
			node = at.target;
		}

		if (at.property) {

			valid_animated_properties.push_back(i);

		} else {

			node_map[node].anim_tracks.push_back(i);
			valid_animated_nodes.insert(node);
		}
	}

	create_animation(-1, p_make_tracks_in_all_bones, p_import_value_tracks);
	//print_line("clipcount: "+itos(collada.state.animation_clips.size()));
	for (int i = 0; i < collada.state.animation_clips.size(); i++)
		create_animation(i, p_make_tracks_in_all_bones, p_import_value_tracks);
}

void ColladaImport::create_animation(int p_clip, bool p_make_tracks_in_all_bones, bool p_import_value_tracks) {

	Ref<Animation> animation = Ref<Animation>(memnew(Animation));

	if (p_clip == -1) {

		//print_line("default");
		animation->set_name("default");
	} else {
		//print_line("clip name: "+collada.state.animation_clips[p_clip].name);
		animation->set_name(collada.state.animation_clips[p_clip].name);
	}

	for (Map<String, NodeMap>::Element *E = node_map.front(); E; E = E->next()) {

		if (E->get().bone < 0)
			continue;
		bones_with_animation[E->key()] = false;
	}
	//store and validate tracks

	if (p_clip == -1) {
		//main anim
	}

	Set<int> track_filter;

	if (p_clip == -1) {

		for (int i = 0; i < collada.state.animation_clips.size(); i++) {

			int tc = collada.state.animation_clips[i].tracks.size();
			for (int j = 0; j < tc; j++) {

				String n = collada.state.animation_clips[i].tracks[j];
				if (collada.state.by_id_tracks.has(n)) {

					const Vector<int> &ti = collada.state.by_id_tracks[n];
					for (int k = 0; k < ti.size(); k++) {
						track_filter.insert(ti[k]);
					}
				}
			}
		}
	} else {

		int tc = collada.state.animation_clips[p_clip].tracks.size();
		for (int j = 0; j < tc; j++) {

			String n = collada.state.animation_clips[p_clip].tracks[j];
			if (collada.state.by_id_tracks.has(n)) {

				const Vector<int> &ti = collada.state.by_id_tracks[n];
				for (int k = 0; k < ti.size(); k++) {
					track_filter.insert(ti[k]);
				}
			}
		}
	}

	//animation->set_loop(true);
	//create animation tracks

	Vector<float> base_snapshots;

	float f = 0;
	float snapshot_interval = 1.0 / bake_fps; //should be customizable somewhere...

	float anim_length = collada.state.animation_length;
	if (p_clip >= 0 && collada.state.animation_clips[p_clip].end)
		anim_length = collada.state.animation_clips[p_clip].end;

	while (f < anim_length) {

		base_snapshots.push_back(f);

		f += snapshot_interval;

		if (f >= anim_length) {
			base_snapshots.push_back(anim_length);
		}
	}

	//print_line("anim len: "+rtos(anim_length));
	animation->set_length(anim_length);

	bool tracks_found = false;

	for (Set<String>::Element *E = valid_animated_nodes.front(); E; E = E->next()) {

		// take snapshots

		if (!collada.state.scene_map.has(E->get())) {

			continue;
		}

		NodeMap &nm = node_map[E->get()];
		String path = scene->get_path_to(nm.node);

		if (nm.bone >= 0) {
			Skeleton *sk = static_cast<Skeleton *>(nm.node);
			String name = sk->get_bone_name(nm.bone);
			path = path + ":" + name;
		}

		bool found_anim = false;

		Collada::Node *cn = collada.state.scene_map[E->get()];
		if (cn->ignore_anim) {

			continue;
		}

		animation->add_track(Animation::TYPE_TRANSFORM);
		int track = animation->get_track_count() - 1;
		animation->track_set_path(track, path);
		animation->track_set_imported(track, true); //helps merging later

		Vector<float> snapshots = base_snapshots;

		if (nm.anim_tracks.size() == 1) {
			//use snapshot keys from anim track instead, because this was most likely exported baked
			Collada::AnimationTrack &at = collada.state.animation_tracks[nm.anim_tracks.front()->get()];
			snapshots.clear();
			for (int i = 0; i < at.keys.size(); i++)
				snapshots.push_back(at.keys[i].time);
		}

		for (int i = 0; i < snapshots.size(); i++) {

			for (List<int>::Element *ET = nm.anim_tracks.front(); ET; ET = ET->next()) {
				//apply tracks

				if (p_clip == -1) {

					if (track_filter.has(ET->get())) {

						continue;
					}
				} else {

					if (!track_filter.has(ET->get()))
						continue;
				}

				found_anim = true;

				Collada::AnimationTrack &at = collada.state.animation_tracks[ET->get()];

				int xform_idx = -1;
				for (int j = 0; j < cn->xform_list.size(); j++) {

					if (cn->xform_list[j].id == at.param) {

						xform_idx = j;
						break;
					}
				}

				if (xform_idx == -1) {
					print_line("couldnt find matching node " + at.target + " xform for track " + at.param);
					continue;
				}

				ERR_CONTINUE(xform_idx == -1);

				Vector<float> data = at.get_value_at_time(snapshots[i]);
				ERR_CONTINUE(data.empty());

				Collada::Node::XForm &xf = cn->xform_list[xform_idx];

				if (at.component == "ANGLE") {
					ERR_CONTINUE(data.size() != 1);
					ERR_CONTINUE(xf.op != Collada::Node::XForm::OP_ROTATE);
					ERR_CONTINUE(xf.data.size() < 4);
					xf.data[3] = data[0];
				} else if (at.component == "X" || at.component == "Y" || at.component == "Z") {
					int cn = at.component[0] - 'X';
					ERR_CONTINUE(cn >= xf.data.size());
					ERR_CONTINUE(data.size() > 1);
					xf.data[cn] = data[0];
				} else if (data.size() == xf.data.size()) {

					xf.data = data;
				} else {

					if (data.size() != xf.data.size()) {
						print_line("component " + at.component + " datasize " + itos(data.size()) + " xfdatasize " + itos(xf.data.size()));
					}

					ERR_CONTINUE(data.size() != xf.data.size());
				}
			}

			Transform xform = cn->compute_transform(collada);
			xform = collada.fix_transform(xform) * cn->post_transform;

			if (nm.bone >= 0) {
				//make bone transform relative to rest (in case of skeleton)
				Skeleton *sk = nm.node->cast_to<Skeleton>();
				if (sk) {

					xform = sk->get_bone_rest(nm.bone).affine_inverse() * xform;
				} else {

					ERR_PRINT("INVALID SKELETON!!!!");
				}
			}

			Quat q = xform.basis;
			q.normalize();
			Vector3 s = xform.basis.get_scale();
			Vector3 l = xform.origin;

			animation->transform_track_insert_key(track, snapshots[i], l, q, s);
		}

		if (nm.bone >= 0) {
			if (found_anim)
				bones_with_animation[E->get()] = true;
		}

		if (found_anim)
			tracks_found = true;
		else {
			animation->remove_track(track);
		}
	}

	if (p_make_tracks_in_all_bones) {

		//some bones may lack animation, but since we don't store pose as a property, we must add keyframes!
		for (Map<String, bool>::Element *E = bones_with_animation.front(); E; E = E->next()) {

			if (E->get())
				continue;

			//print_line("BONE LACKS ANIM: "+E->key());

			NodeMap &nm = node_map[E->key()];
			String path = scene->get_path_to(nm.node);
			ERR_CONTINUE(nm.bone < 0);
			Skeleton *sk = static_cast<Skeleton *>(nm.node);
			String name = sk->get_bone_name(nm.bone);
			path = path + ":" + name;

			Collada::Node *cn = collada.state.scene_map[E->key()];
			if (cn->ignore_anim) {
				print_line("warning, ignoring animation on node: " + path);
				continue;
			}

			animation->add_track(Animation::TYPE_TRANSFORM);
			int track = animation->get_track_count() - 1;
			animation->track_set_path(track, path);
			animation->track_set_imported(track, true); //helps merging later

			Transform xform = cn->compute_transform(collada);
			xform = collada.fix_transform(xform) * cn->post_transform;

			xform = sk->get_bone_rest(nm.bone).affine_inverse() * xform;

			Quat q = xform.basis;
			q.normalize();
			Vector3 s = xform.basis.get_scale();
			Vector3 l = xform.origin;

			animation->transform_track_insert_key(track, 0, l, q, s);

			tracks_found = true;
		}
	}

	if (p_import_value_tracks) {
		for (int i = 0; i < valid_animated_properties.size(); i++) {

			int ti = valid_animated_properties[i];

			if (p_clip == -1) {

				if (track_filter.has(ti))
					continue;
			} else {

				if (!track_filter.has(ti))
					continue;
			}

			Collada::AnimationTrack &at = collada.state.animation_tracks[ti];

			// take snapshots
			if (!collada.state.scene_map.has(at.target))
				continue;

			NodeMap &nm = node_map[at.target];
			String path = scene->get_path_to(nm.node);

			animation->add_track(Animation::TYPE_VALUE);
			int track = animation->get_track_count() - 1;

			path = path + ":" + at.param;
			animation->track_set_path(track, path);
			animation->track_set_imported(track, true); //helps merging later

			for (int i = 0; i < at.keys.size(); i++) {

				float time = at.keys[i].time;
				Variant value;
				Vector<float> data = at.keys[i].data;
				if (data.size() == 1) {
					//push a float
					value = data[0];

				} else if (data.size() == 16) {
					//matrix
					print_line("value keys for matrices not supported");
				} else {

					print_line("don't know what to do with this amount of value keys: " + itos(data.size()));
				}

				animation->track_insert_key(track, time, value);
			}

			tracks_found = true;
		}
	}

	if (tracks_found) {

		animations.push_back(animation);
	}
}

/*********************************************************************************/
/*************************************** SCENE ***********************************/
/*********************************************************************************/

#define DEBUG_ANIMATION

uint32_t EditorSceneImporterCollada::get_import_flags() const {

	return IMPORT_SCENE | IMPORT_ANIMATION;
}
void EditorSceneImporterCollada::get_extensions(List<String> *r_extensions) const {

	r_extensions->push_back("dae");
}
Node *EditorSceneImporterCollada::import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err) {

	ColladaImport state;
	uint32_t flags = Collada::IMPORT_FLAG_SCENE;
	if (p_flags & IMPORT_ANIMATION)
		flags |= Collada::IMPORT_FLAG_ANIMATION;

	state.use_mesh_builtin_materials = !(p_flags & IMPORT_MATERIALS_IN_INSTANCES);
	state.bake_fps = p_bake_fps;

	Error err = state.load(p_path, flags, p_flags & EditorSceneImporter::IMPORT_GENERATE_TANGENT_ARRAYS);

	ERR_FAIL_COND_V(err != OK, NULL);

	if (state.missing_textures.size()) {

		/*
		for(int i=0;i<state.missing_textures.size();i++) {
			EditorNode::add_io_error("Texture Not Found: "+state.missing_textures[i]);
		}
		*/

		if (r_missing_deps) {

			for (int i = 0; i < state.missing_textures.size(); i++) {
				//EditorNode::add_io_error("Texture Not Found: "+state.missing_textures[i]);
				r_missing_deps->push_back(state.missing_textures[i]);
			}
		}
	}

	if (p_flags & IMPORT_ANIMATION) {

		state.create_animations(p_flags & IMPORT_ANIMATION_FORCE_ALL_TRACKS_IN_ALL_CLIPS, p_flags & EditorSceneImporter::IMPORT_ANIMATION_KEEP_VALUE_TRACKS);
		AnimationPlayer *ap = memnew(AnimationPlayer);
		for (int i = 0; i < state.animations.size(); i++) {
			String name;
			if (state.animations[i]->get_name() == "")
				name = "default";
			else
				name = state.animations[i]->get_name();

			if (p_flags & IMPORT_ANIMATION_DETECT_LOOP) {

				if (name.begins_with("loop") || name.ends_with("loop") || name.begins_with("cycle") || name.ends_with("cycle")) {
					state.animations[i]->set_loop(true);
				}
			}

			ap->add_animation(name, state.animations[i]);
		}
		state.scene->add_child(ap);
		ap->set_owner(state.scene);
	}

	return state.scene;
}

Ref<Animation> EditorSceneImporterCollada::import_animation(const String &p_path, uint32_t p_flags) {

	ColladaImport state;

	state.use_mesh_builtin_materials = false;

	Error err = state.load(p_path, Collada::IMPORT_FLAG_ANIMATION, p_flags & EditorSceneImporter::IMPORT_GENERATE_TANGENT_ARRAYS);
	ERR_FAIL_COND_V(err != OK, RES());

	state.create_animations(p_flags & EditorSceneImporter::IMPORT_ANIMATION_FORCE_ALL_TRACKS_IN_ALL_CLIPS, p_flags & EditorSceneImporter::IMPORT_ANIMATION_KEEP_VALUE_TRACKS);
	if (state.scene)
		memdelete(state.scene);

	if (state.animations.size() == 0)
		return Ref<Animation>();
	Ref<Animation> anim = state.animations[0];
	anim = state.animations[0];
	print_line("Anim Load OK");
	String base = p_path.get_basename().to_lower();
	if (p_flags & IMPORT_ANIMATION_DETECT_LOOP) {

		if (base.begins_with("loop") || base.ends_with("loop") || base.begins_with("cycle") || base.ends_with("cycle")) {
			anim->set_loop(true);
		}
	}

	return anim;
}

EditorSceneImporterCollada::EditorSceneImporterCollada() {
}
