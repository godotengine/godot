/**************************************************************************/
/*  resource_importer_obj.cpp                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "resource_importer_obj.h"

#include "core/io/file_access.h"
#include "core/io/resource_saver.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/resources/importer_mesh.h"
#include "scene/resources/mesh.h"
#include "scene/resources/surface_tool.h"

uint32_t EditorOBJImporter::get_import_flags() const {
	return IMPORT_SCENE;
}

static Error _parse_material_library(const String &p_path, HashMap<String, Ref<StandardMaterial3D>> &material_map, List<String> *r_missing_deps) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_OPEN, vformat("Couldn't open MTL file '%s', it may not exist or not be readable.", p_path));

	Ref<StandardMaterial3D> current;
	String current_name;
	String base_path = p_path.get_base_dir();
	while (true) {
		String l = f->get_line().strip_edges();

		if (l.begins_with("newmtl ")) {
			//vertex

			current_name = l.replace("newmtl", "").strip_edges();
			current.instantiate();
			current->set_name(current_name);
			material_map[current_name] = current;
		} else if (l.begins_with("Ka ")) {
			//uv
			WARN_PRINT("OBJ: Ambient light for material '" + current_name + "' is ignored in PBR");

		} else if (l.begins_with("Kd ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_INVALID_DATA);
			Color c = current->get_albedo();
			c.r = v[1].to_float();
			c.g = v[2].to_float();
			c.b = v[3].to_float();
			current->set_albedo(c);
		} else if (l.begins_with("Ks ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_INVALID_DATA);
			float r = v[1].to_float();
			float g = v[2].to_float();
			float b = v[3].to_float();
			float metalness = MAX(r, MAX(g, b));
			current->set_metallic(metalness);
		} else if (l.begins_with("Ns ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() != 2, ERR_INVALID_DATA);
			float s = v[1].to_float();
			current->set_metallic((1000.0 - s) / 1000.0);
		} else if (l.begins_with("d ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() != 2, ERR_INVALID_DATA);
			float d = v[1].to_float();
			Color c = current->get_albedo();
			c.a = d;
			current->set_albedo(c);
			if (c.a < 0.99) {
				current->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
			}
		} else if (l.begins_with("Tr ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() != 2, ERR_INVALID_DATA);
			float d = v[1].to_float();
			Color c = current->get_albedo();
			c.a = 1.0 - d;
			current->set_albedo(c);
			if (c.a < 0.99) {
				current->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
			}

		} else if (l.begins_with("map_Ka ")) {
			//uv
			WARN_PRINT("OBJ: Ambient light texture for material '" + current_name + "' is ignored in PBR");

		} else if (l.begins_with("map_Kd ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);

			String p = l.replace("map_Kd", "").replace("\\", "/").strip_edges();
			String path;
			if (p.is_absolute_path()) {
				path = p;
			} else {
				path = base_path.path_join(p);
			}

			Ref<Texture2D> texture = ResourceLoader::load(path);

			if (texture.is_valid()) {
				current->set_texture(StandardMaterial3D::TEXTURE_ALBEDO, texture);
			} else if (r_missing_deps) {
				r_missing_deps->push_back(path);
			}

		} else if (l.begins_with("map_Ks ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);

			String p = l.replace("map_Ks", "").replace("\\", "/").strip_edges();
			String path;
			if (p.is_absolute_path()) {
				path = p;
			} else {
				path = base_path.path_join(p);
			}

			Ref<Texture2D> texture = ResourceLoader::load(path);

			if (texture.is_valid()) {
				current->set_texture(StandardMaterial3D::TEXTURE_METALLIC, texture);
			} else if (r_missing_deps) {
				r_missing_deps->push_back(path);
			}

		} else if (l.begins_with("map_Ns ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);

			String p = l.replace("map_Ns", "").replace("\\", "/").strip_edges();
			String path;
			if (p.is_absolute_path()) {
				path = p;
			} else {
				path = base_path.path_join(p);
			}

			Ref<Texture2D> texture = ResourceLoader::load(path);

			if (texture.is_valid()) {
				current->set_texture(StandardMaterial3D::TEXTURE_ROUGHNESS, texture);
			} else if (r_missing_deps) {
				r_missing_deps->push_back(path);
			}
		} else if (l.begins_with("map_bump ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);

			String p = l.replace("map_bump", "").replace("\\", "/").strip_edges();
			String path = base_path.path_join(p);

			Ref<Texture2D> texture = ResourceLoader::load(path);

			if (texture.is_valid()) {
				current->set_feature(StandardMaterial3D::FEATURE_NORMAL_MAPPING, true);
				current->set_texture(StandardMaterial3D::TEXTURE_NORMAL, texture);
			} else if (r_missing_deps) {
				r_missing_deps->push_back(path);
			}
		} else if (f->eof_reached()) {
			break;
		}
	}

	return OK;
}

static Error _parse_obj(const String &p_path, List<Ref<Mesh>> &r_meshes, bool p_single_mesh, bool p_generate_tangents, bool p_optimize, Vector3 p_scale_mesh, Vector3 p_offset_mesh, bool p_disable_compression, List<String> *r_missing_deps) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_OPEN, vformat("Couldn't open OBJ file '%s', it may not exist or not be readable.", p_path));

	// Avoid trying to load/interpret potential build artifacts from Visual Studio (e.g. when compiling native plugins inside the project tree)
	// This should only match, if it's indeed a COFF file header
	// https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#machine-types
	const int first_bytes = f->get_16();
	static const Vector<int> coff_header_machines{
		0x0, // IMAGE_FILE_MACHINE_UNKNOWN
		0x8664, // IMAGE_FILE_MACHINE_AMD64
		0x1c0, // IMAGE_FILE_MACHINE_ARM
		0x14c, // IMAGE_FILE_MACHINE_I386
		0x200, // IMAGE_FILE_MACHINE_IA64
	};
	ERR_FAIL_COND_V_MSG(coff_header_machines.find(first_bytes) != -1, ERR_FILE_CORRUPT, vformat("Couldn't read OBJ file '%s', it seems to be binary, corrupted, or empty.", p_path));
	f->seek(0);

	Ref<ArrayMesh> mesh;
	mesh.instantiate();

	bool generate_tangents = p_generate_tangents;
	Vector3 scale_mesh = p_scale_mesh;
	Vector3 offset_mesh = p_offset_mesh;

	Vector<Vector3> vertices;
	Vector<Vector3> normals;
	Vector<Vector2> uvs;
	Vector<Color> colors;
	const String default_name = "Mesh";
	String name = default_name;

	HashMap<String, HashMap<String, Ref<StandardMaterial3D>>> material_map;

	Ref<SurfaceTool> surf_tool = memnew(SurfaceTool);
	surf_tool->begin(Mesh::PRIMITIVE_TRIANGLES);

	String current_material_library;
	String current_material;
	String current_group;
	uint32_t smooth_group = 0;
	bool smoothing = true;
	const uint32_t no_smoothing_smooth_group = (uint32_t)-1;

	while (true) {
		String l = f->get_line().strip_edges();
		while (l.length() && l[l.length() - 1] == '\\') {
			String add = f->get_line().strip_edges();
			l += add;
			if (add.is_empty()) {
				break;
			}
		}

		if (l.begins_with("v ")) {
			//vertex
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_FILE_CORRUPT);
			Vector3 vtx;
			vtx.x = v[1].to_float() * scale_mesh.x + offset_mesh.x;
			vtx.y = v[2].to_float() * scale_mesh.y + offset_mesh.y;
			vtx.z = v[3].to_float() * scale_mesh.z + offset_mesh.z;
			vertices.push_back(vtx);
			//vertex color
			if (v.size() >= 7) {
				while (colors.size() < vertices.size() - 1) {
					colors.push_back(Color(1.0, 1.0, 1.0));
				}
				Color c;
				c.r = v[4].to_float();
				c.g = v[5].to_float();
				c.b = v[6].to_float();
				colors.push_back(c);
			} else if (!colors.is_empty()) {
				colors.push_back(Color(1.0, 1.0, 1.0));
			}
		} else if (l.begins_with("vt ")) {
			//uv
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 3, ERR_FILE_CORRUPT);
			Vector2 uv;
			uv.x = v[1].to_float();
			uv.y = 1.0 - v[2].to_float();
			uvs.push_back(uv);
		} else if (l.begins_with("vn ")) {
			//normal
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_FILE_CORRUPT);
			Vector3 nrm;
			nrm.x = v[1].to_float();
			nrm.y = v[2].to_float();
			nrm.z = v[3].to_float();
			normals.push_back(nrm);
		} else if (l.begins_with("f ")) {
			//vertex

			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_FILE_CORRUPT);

			//not very fast, could be sped up

			Vector<String> face[3];
			face[0] = v[1].split("/");
			face[1] = v[2].split("/");
			ERR_FAIL_COND_V(face[0].size() == 0, ERR_FILE_CORRUPT);

			ERR_FAIL_COND_V(face[0].size() != face[1].size(), ERR_FILE_CORRUPT);
			for (int i = 2; i < v.size() - 1; i++) {
				face[2] = v[i + 1].split("/");

				ERR_FAIL_COND_V(face[0].size() != face[2].size(), ERR_FILE_CORRUPT);
				for (int j = 0; j < 3; j++) {
					int idx = j;

					if (idx < 2) {
						idx = 1 ^ idx;
					}

					if (face[idx].size() == 3) {
						int norm = face[idx][2].to_int() - 1;
						if (norm < 0) {
							norm += normals.size() + 1;
						}
						ERR_FAIL_INDEX_V(norm, normals.size(), ERR_FILE_CORRUPT);
						surf_tool->set_normal(normals[norm]);
						if (generate_tangents && uvs.is_empty()) {
							// We can't generate tangents without UVs, so create dummy tangents.
							Vector3 tan = Vector3(0.0, 1.0, 0.0).cross(normals[norm]);
							surf_tool->set_tangent(Plane(tan.x, tan.y, tan.z, 1.0));
						}
					} else {
						// No normals, use a dummy normal since normals will be generated.
						if (generate_tangents && uvs.is_empty()) {
							// We can't generate tangents without UVs, so create dummy tangents.
							surf_tool->set_tangent(Plane(1.0, 0.0, 0.0, 1.0));
						}
					}

					if (face[idx].size() >= 2 && !face[idx][1].is_empty()) {
						int uv = face[idx][1].to_int() - 1;
						if (uv < 0) {
							uv += uvs.size() + 1;
						}
						ERR_FAIL_INDEX_V(uv, uvs.size(), ERR_FILE_CORRUPT);
						surf_tool->set_uv(uvs[uv]);
					}

					int vtx = face[idx][0].to_int() - 1;
					if (vtx < 0) {
						vtx += vertices.size() + 1;
					}
					ERR_FAIL_INDEX_V(vtx, vertices.size(), ERR_FILE_CORRUPT);

					Vector3 vertex = vertices[vtx];
					if (!colors.is_empty()) {
						surf_tool->set_color(colors[vtx]);
					}
					surf_tool->set_smooth_group(smoothing ? smooth_group : no_smoothing_smooth_group);
					surf_tool->add_vertex(vertex);
				}

				face[1] = face[2];
			}
		} else if (l.begins_with("s ")) { //smoothing
			String what = l.substr(2, l.length()).strip_edges();
			bool do_smooth;
			if (what == "off") {
				do_smooth = false;
			} else {
				do_smooth = true;
			}
			if (do_smooth != smoothing) {
				smoothing = do_smooth;
				if (smoothing) {
					smooth_group++;
				}
			}
		} else if (/*l.begins_with("g ") ||*/ l.begins_with("usemtl ") || (l.begins_with("o ") || f->eof_reached())) { //commit group to mesh
			uint64_t mesh_flags = RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES;

			if (p_disable_compression) {
				mesh_flags = 0;
			}
			//groups are too annoying
			if (surf_tool->get_vertex_array().size()) {
				//another group going on, commit it
				if (normals.size() == 0) {
					surf_tool->generate_normals();
				}

				if (generate_tangents && uvs.size()) {
					surf_tool->generate_tangents();
				}

				surf_tool->index();

				print_verbose("OBJ: Current material library " + current_material_library + " has " + itos(material_map.has(current_material_library)));
				print_verbose("OBJ: Current material " + current_material + " has " + itos(material_map.has(current_material_library) && material_map[current_material_library].has(current_material)));

				if (material_map.has(current_material_library) && material_map[current_material_library].has(current_material)) {
					Ref<StandardMaterial3D> &material = material_map[current_material_library][current_material];
					if (!colors.is_empty()) {
						material->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
					}
					surf_tool->set_material(material);
				}

				mesh = surf_tool->commit(mesh, mesh_flags);

				if (!current_material.is_empty()) {
					mesh->surface_set_name(mesh->get_surface_count() - 1, current_material.get_basename());
				} else if (!current_group.is_empty()) {
					mesh->surface_set_name(mesh->get_surface_count() - 1, current_group);
				}

				print_verbose("OBJ: Added surface :" + mesh->surface_get_name(mesh->get_surface_count() - 1));
				surf_tool->clear();
				surf_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
			}

			if (l.begins_with("o ") || f->eof_reached()) {
				if (!p_single_mesh) {
					if (mesh->get_surface_count() > 0) {
						mesh->set_name(name);
						r_meshes.push_back(mesh);
						mesh.instantiate();
					}
					name = default_name;
					current_group = "";
					current_material = "";
				}
			}

			if (f->eof_reached()) {
				break;
			}

			if (l.begins_with("o ")) {
				name = l.substr(2, l.length()).strip_edges();
			}

			if (l.begins_with("usemtl ")) {
				current_material = l.replace("usemtl", "").strip_edges();
			}

			if (l.begins_with("g ")) {
				current_group = l.substr(2, l.length()).strip_edges();
			}

		} else if (l.begins_with("mtllib ")) { //parse material

			current_material_library = l.replace("mtllib", "").strip_edges();
			if (!material_map.has(current_material_library)) {
				HashMap<String, Ref<StandardMaterial3D>> lib;
				String lib_path = current_material_library;
				if (lib_path.is_relative_path()) {
					lib_path = p_path.get_base_dir().path_join(current_material_library);
				}
				Error err = _parse_material_library(lib_path, lib, r_missing_deps);
				if (err == OK) {
					material_map[current_material_library] = lib;
				}
			}
		}
	}

	if (p_single_mesh) {
		r_meshes.push_back(mesh);
	}

	return OK;
}

Node *EditorOBJImporter::import_scene(const String &p_path, uint32_t p_flags, const HashMap<StringName, Variant> &p_options, List<String> *r_missing_deps, Error *r_err) {
	List<Ref<Mesh>> meshes;

	Error err = _parse_obj(p_path, meshes, false, p_flags & IMPORT_GENERATE_TANGENT_ARRAYS, false, Vector3(1, 1, 1), Vector3(0, 0, 0), p_flags & IMPORT_FORCE_DISABLE_MESH_COMPRESSION, r_missing_deps);

	if (err != OK) {
		if (r_err) {
			*r_err = err;
		}
		return nullptr;
	}

	Node3D *scene = memnew(Node3D);

	for (const Ref<Mesh> &m : meshes) {
		Ref<ImporterMesh> mesh;
		mesh.instantiate();
		mesh->set_name(m->get_name());
		for (int i = 0; i < m->get_surface_count(); i++) {
			mesh->add_surface(m->surface_get_primitive_type(i), m->surface_get_arrays(i), Array(), Dictionary(), m->surface_get_material(i), String(), m->surface_get_format(i));
		}

		ImporterMeshInstance3D *mi = memnew(ImporterMeshInstance3D);
		mi->set_mesh(mesh);
		mi->set_name(m->get_name());
		scene->add_child(mi, true);
		mi->set_owner(scene);
	}

	if (r_err) {
		*r_err = OK;
	}

	return scene;
}

void EditorOBJImporter::get_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("obj");
}

EditorOBJImporter::EditorOBJImporter() {
}

////////////////////////////////////////////////////

String ResourceImporterOBJ::get_importer_name() const {
	return "wavefront_obj";
}

String ResourceImporterOBJ::get_visible_name() const {
	return "OBJ as Mesh";
}

void ResourceImporterOBJ::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("obj");
}

String ResourceImporterOBJ::get_save_extension() const {
	return "mesh";
}

String ResourceImporterOBJ::get_resource_type() const {
	return "Mesh";
}

int ResourceImporterOBJ::get_format_version() const {
	return 1;
}

int ResourceImporterOBJ::get_preset_count() const {
	return 0;
}

String ResourceImporterOBJ::get_preset_name(int p_idx) const {
	return "";
}

void ResourceImporterOBJ::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "generate_tangents"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::VECTOR3, "scale_mesh"), Vector3(1, 1, 1)));
	r_options->push_back(ImportOption(PropertyInfo(Variant::VECTOR3, "offset_mesh"), Vector3(0, 0, 0)));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "optimize_mesh"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force_disable_mesh_compression"), false));
}

bool ResourceImporterOBJ::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return true;
}

Error ResourceImporterOBJ::import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	List<Ref<Mesh>> meshes;

	Error err = _parse_obj(p_source_file, meshes, true, p_options["generate_tangents"], p_options["optimize_mesh"], p_options["scale_mesh"], p_options["offset_mesh"], p_options["force_disable_mesh_compression"], nullptr);

	ERR_FAIL_COND_V(err != OK, err);
	ERR_FAIL_COND_V(meshes.size() != 1, ERR_BUG);

	String save_path = p_save_path + ".mesh";

	err = ResourceSaver::save(meshes.front()->get(), save_path);

	ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot save Mesh to file '" + save_path + "'.");

	r_gen_files->push_back(save_path);

	return OK;
}

ResourceImporterOBJ::ResourceImporterOBJ() {
}
