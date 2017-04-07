/*************************************************************************/
/*  resource_importer_obj.cpp                                            */
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
#include "resource_importer_obj.h"

#include "io/resource_saver.h"
#include "os/file_access.h"
#include "scene/resources/mesh.h"
#include "scene/resources/surface_tool.h"
#include "scene/resources/surface_tool.h"

String ResourceImporterOBJ::get_importer_name() const {

	return "obj_mesh";
}

String ResourceImporterOBJ::get_visible_name() const {

	return "OBJ As Mesh";
}
void ResourceImporterOBJ::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("obj");
}
String ResourceImporterOBJ::get_save_extension() const {
	return "msh";
}

String ResourceImporterOBJ::get_resource_type() const {

	return "Mesh";
}

bool ResourceImporterOBJ::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {

	return true;
}

int ResourceImporterOBJ::get_preset_count() const {
	return 0;
}
String ResourceImporterOBJ::get_preset_name(int p_idx) const {

	return String();
}

void ResourceImporterOBJ::get_import_options(List<ImportOption> *r_options, int p_preset) const {

	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "generate/tangents"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "generate/normals"), true));
	//not for nowp
	//r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL,"import/materials")));
	//r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL,"import/textures")));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force/flip_faces"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force/smooth_shading"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force/weld_vertices"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "force/weld_tolerance", PROPERTY_HINT_RANGE, "0.00001,16,0.00001"), 0.0001));
	//r_options->push_back(PropertyInfo(Variant::INT,"compress/bitrate",PROPERTY_HINT_ENUM,"64,96,128,192"));
}

Error ResourceImporterOBJ::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files) {

	FileAccessRef f = FileAccess::open(p_source_file, FileAccess::READ);
	ERR_FAIL_COND_V(!f, ERR_CANT_OPEN);

	Ref<Mesh> mesh = Ref<Mesh>(memnew(Mesh));
	Map<String, Ref<Material> > name_map;

	bool generate_normals = p_options["generate/normals"];
	bool generate_tangents = p_options["generate/tangents"];
	bool flip_faces = p_options["force/flip_faces"];
	bool force_smooth = p_options["force/smooth_shading"];
	bool weld_vertices = p_options["force/weld_vertices"];
	float weld_tolerance = p_options["force/weld_tolerance"];
	Vector<Vector3> vertices;
	Vector<Vector3> normals;
	Vector<Vector2> uvs;
	String name;

	Ref<SurfaceTool> surf_tool = memnew(SurfaceTool);
	surf_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
	if (force_smooth)
		surf_tool->add_smooth_group(true);
	int has_index_data = false;

	while (true) {

		String l = f->get_line().strip_edges();

		if (l.begins_with("v ")) {
			//vertex
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_INVALID_DATA);
			Vector3 vtx;
			vtx.x = v[1].to_float();
			vtx.y = v[2].to_float();
			vtx.z = v[3].to_float();
			vertices.push_back(vtx);
		} else if (l.begins_with("vt ")) {
			//uv
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 3, ERR_INVALID_DATA);
			Vector2 uv;
			uv.x = v[1].to_float();
			uv.y = 1.0 - v[2].to_float();
			uvs.push_back(uv);

		} else if (l.begins_with("vn ")) {
			//normal
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_INVALID_DATA);
			Vector3 nrm;
			nrm.x = v[1].to_float();
			nrm.y = v[2].to_float();
			nrm.z = v[3].to_float();
			normals.push_back(nrm);
		} else if (l.begins_with("f ")) {
			//vertex

			has_index_data = true;
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_INVALID_DATA);

			//not very fast, could be sped up

			Vector<String> face[3];
			face[0] = v[1].split("/");
			face[1] = v[2].split("/");
			ERR_FAIL_COND_V(face[0].size() == 0, ERR_PARSE_ERROR);
			ERR_FAIL_COND_V(face[0].size() != face[1].size(), ERR_PARSE_ERROR);
			for (int i = 2; i < v.size() - 1; i++) {

				face[2] = v[i + 1].split("/");
				ERR_FAIL_COND_V(face[0].size() != face[2].size(), ERR_PARSE_ERROR);
				for (int j = 0; j < 3; j++) {

					int idx = j;

					if (!flip_faces && idx < 2) {
						idx = 1 ^ idx;
					}

					if (face[idx].size() == 3) {
						int norm = face[idx][2].to_int() - 1;
						ERR_FAIL_INDEX_V(norm, normals.size(), ERR_PARSE_ERROR);
						surf_tool->add_normal(normals[norm]);
					}

					if (face[idx].size() >= 2 && face[idx][1] != String()) {

						int uv = face[idx][1].to_int() - 1;
						ERR_FAIL_INDEX_V(uv, uvs.size(), ERR_PARSE_ERROR);
						surf_tool->add_uv(uvs[uv]);
					}

					int vtx = face[idx][0].to_int() - 1;
					ERR_FAIL_INDEX_V(vtx, vertices.size(), ERR_PARSE_ERROR);

					Vector3 vertex = vertices[vtx];
					if (weld_vertices)
						vertex = vertex.snapped(weld_tolerance);
					surf_tool->add_vertex(vertex);
				}

				face[1] = face[2];
			}
		} else if (l.begins_with("s ") && !force_smooth) { //smoothing
			String what = l.substr(2, l.length()).strip_edges();
			if (what == "off")
				surf_tool->add_smooth_group(false);
			else
				surf_tool->add_smooth_group(true);

		} else if (l.begins_with("o ") || f->eof_reached()) { //new surface or done

			if (has_index_data) {
				//new object/surface
				if (generate_normals || force_smooth)
					surf_tool->generate_normals();
				if (uvs.size() && (normals.size() || generate_normals) && generate_tangents)
					surf_tool->generate_tangents();

				surf_tool->index();
				mesh = surf_tool->commit(mesh);
				if (name == "")
					name = vformat(TTR("Surface %d"), mesh->get_surface_count() - 1);
				mesh->surface_set_name(mesh->get_surface_count() - 1, name);
				name = "";
				surf_tool->clear();
				surf_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
				if (force_smooth)
					surf_tool->add_smooth_group(true);

				has_index_data = false;

				if (f->eof_reached())
					break;
			}

			if (l.begins_with("o ")) //name
				name = l.substr(2, l.length()).strip_edges();
		}
	}

	/*
	TODO, check existing materials and merge?
	//re-apply materials if exist
	for(int i=0;i<mesh->get_surface_count();i++) {

		String n = mesh->surface_get_name(i);
		if (name_map.has(n))
			mesh->surface_set_material(i,name_map[n]);
	}
*/

	Error err = ResourceSaver::save(p_save_path + ".msh", mesh);

	return err;
}

ResourceImporterOBJ::ResourceImporterOBJ() {
}
