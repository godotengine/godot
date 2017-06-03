/*************************************************************************/
/*  editor_mesh_import_plugin.cpp                                        */
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
#include "editor_mesh_import_plugin.h"

#if 0

#include "editor/editor_dir_dialog.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_node.h"
#include "editor/property_editor.h"
//#include "scene/resources/sample.h"
#include "io/marshalls.h"
#include "io/resource_saver.h"
#include "os/file_access.h"
#include "scene/resources/surface_tool.h"

class _EditorMeshImportOptions : public Object {

	GDCLASS(_EditorMeshImportOptions,Object);
public:


	bool generate_tangents;
	bool generate_normals;
	bool flip_faces;
	bool smooth_shading;
	bool weld_vertices;
	bool import_material;
	bool import_textures;
	float weld_tolerance;


	bool _set(const StringName& p_name, const Variant& p_value) {

		String n = p_name;
		if (n=="generate/tangents")
			generate_tangents=p_value;
		else if (n=="generate/normals")
			generate_normals=p_value;
		else if (n=="import/materials")
			import_material=p_value;
		else if (n=="import/textures")
			import_textures=p_value;
		else if (n=="force/flip_faces")
			flip_faces=p_value;
		else if (n=="force/smooth_shading")
			smooth_shading=p_value;
		else if (n=="force/weld_vertices")
			weld_vertices=p_value;
		else if (n=="force/weld_tolerance")
			weld_tolerance=p_value;
		else
			return false;

		return true;

	}

	bool _get(const StringName& p_name,Variant &r_ret) const{

		String n = p_name;
		if (n=="generate/tangents")
			r_ret=generate_tangents;
		else if (n=="generate/normals")
			r_ret=generate_normals;
		else if (n=="import/materials")
			r_ret=import_material;
		else if (n=="import/textures")
			r_ret=import_textures;
		else if (n=="force/flip_faces")
			r_ret=flip_faces;
		else if (n=="force/smooth_shading")
			r_ret=smooth_shading;
		else if (n=="force/weld_vertices")
			r_ret=weld_vertices;
		else if (n=="force/weld_tolerance")
			r_ret=weld_tolerance;
		else
			return false;

		return true;

	}
	void _get_property_list( List<PropertyInfo> *p_list) const{

		p_list->push_back(PropertyInfo(Variant::BOOL,"generate/tangents"));
		p_list->push_back(PropertyInfo(Variant::BOOL,"generate/normals"));
		//not for nowp
		//p_list->push_back(PropertyInfo(Variant::BOOL,"import/materials"));
		//p_list->push_back(PropertyInfo(Variant::BOOL,"import/textures"));
		p_list->push_back(PropertyInfo(Variant::BOOL,"force/flip_faces"));
		p_list->push_back(PropertyInfo(Variant::BOOL,"force/smooth_shading"));
		p_list->push_back(PropertyInfo(Variant::BOOL,"force/weld_vertices"));
		p_list->push_back(PropertyInfo(Variant::REAL,"force/weld_tolerance",PROPERTY_HINT_RANGE,"0.00001,16,0.00001"));
		//p_list->push_back(PropertyInfo(Variant::BOOL,"compress/enable"));
		//p_list->push_back(PropertyInfo(Variant::INT,"compress/bitrate",PROPERTY_HINT_ENUM,"64,96,128,192"));


	}


	static void _bind_methods() {


		ADD_SIGNAL( MethodInfo("changed"));
	}


	_EditorMeshImportOptions() {

		generate_tangents=true;
		generate_normals=false;
		flip_faces=false;
		smooth_shading=false;
		weld_vertices=true;
		weld_tolerance=0.0001;
		import_material=false;
		import_textures=false;

	}


};

class EditorMeshImportDialog : public ConfirmationDialog {

	GDCLASS(EditorMeshImportDialog,ConfirmationDialog);

	EditorMeshImportPlugin *plugin;

	LineEdit *import_path;
	LineEdit *save_path;
	EditorFileDialog *file_select;
	EditorDirDialog *save_select;
	AcceptDialog *error_dialog;
	PropertyEditor *option_editor;

	_EditorMeshImportOptions *options;


public:

	void _choose_files(const Vector<String>& p_path) {

		String files;
		for(int i=0;i<p_path.size();i++) {

			if (i>0)
				files+=",";
			files+=p_path[i];
		}
		/*
		if (p_path.size()) {
			String srctex=p_path[0];
			String ipath = EditorImportDB::get_singleton()->find_source_path(srctex);

			if (ipath!="")
				save_path->set_text(ipath.get_base_dir());
		}*/
		import_path->set_text(files);

	}
	void _choose_save_dir(const String& p_path) {

		save_path->set_text(p_path);
	}

	void _browse() {

		file_select->popup_centered_ratio();
	}

	void _browse_target() {

		save_select->popup_centered_ratio();
	}

	void popup_import(const String& p_path) {

		popup_centered(Size2(400,400)*EDSCALE);

		if (p_path!="") {

			Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(p_path);
			ERR_FAIL_COND(!rimd.is_valid());

			save_path->set_text(p_path.get_base_dir());
			List<String> opts;
			rimd->get_options(&opts);
			for(List<String>::Element *E=opts.front();E;E=E->next()) {

				options->_set(E->get(),rimd->get_option(E->get()));
			}

			String src = "";
			for(int i=0;i<rimd->get_source_count();i++) {
				if (i>0)
					src+=",";
				src+=EditorImportPlugin::expand_source_path(rimd->get_source_path(i));
			}
			import_path->set_text(src);
		}
	}

	void _import() {

		Vector<String> meshes = import_path->get_text().split(",");
		if (meshes.size()==0) {
			error_dialog->set_text(TTR("No meshes to import!"));
			error_dialog->popup_centered_minsize();
			return;
		}

		String dst = save_path->get_text();
		if (dst=="") {
			error_dialog->set_text(TTR("Save path is empty!"));
			error_dialog->popup_centered_minsize();
			return;
		}

		for(int i=0;i<meshes.size();i++) {

			Ref<ResourceImportMetadata> imd = memnew( ResourceImportMetadata );

			List<PropertyInfo> pl;
			options->_get_property_list(&pl);
			for(List<PropertyInfo>::Element *E=pl.front();E;E=E->next()) {

				Variant v;
				String opt=E->get().name;
				options->_get(opt,v);
				imd->set_option(opt,v);

			}

			imd->add_source(EditorImportPlugin::validate_source_path(meshes[i]));

			String file_path = dst.plus_file(meshes[i].get_file().get_basename()+".msh");

			plugin->import(file_path,imd);
		}

		hide();
	}

	void _notification(int p_what) {


		if (p_what==NOTIFICATION_ENTER_TREE) {

			option_editor->edit(options);
		}
	}

	static void _bind_methods() {

		ClassDB::bind_method("_choose_files",&EditorMeshImportDialog::_choose_files);
		ClassDB::bind_method("_choose_save_dir",&EditorMeshImportDialog::_choose_save_dir);
		ClassDB::bind_method("_import",&EditorMeshImportDialog::_import);
		ClassDB::bind_method("_browse",&EditorMeshImportDialog::_browse);
		ClassDB::bind_method("_browse_target",&EditorMeshImportDialog::_browse_target);
	}

	EditorMeshImportDialog(EditorMeshImportPlugin *p_plugin) {

		plugin=p_plugin;

		set_title(TTR("Single Mesh Import"));
		set_hide_on_ok(false);

		VBoxContainer *vbc = memnew( VBoxContainer );
		add_child(vbc);
		//set_child_rect(vbc);

		HBoxContainer *hbc = memnew( HBoxContainer );
		vbc->add_margin_child(TTR("Source Mesh(es):"),hbc);

		import_path = memnew( LineEdit );
		import_path->set_h_size_flags(SIZE_EXPAND_FILL);
		hbc->add_child(import_path);

		Button * import_choose = memnew( Button );
		import_choose->set_text(" .. ");
		hbc->add_child(import_choose);

		import_choose->connect("pressed", this,"_browse");

		hbc = memnew( HBoxContainer );
		vbc->add_margin_child(TTR("Target Path:"),hbc);

		save_path = memnew( LineEdit );
		save_path->set_h_size_flags(SIZE_EXPAND_FILL);
		hbc->add_child(save_path);

		Button * save_choose = memnew( Button );
		save_choose->set_text(" .. ");
		hbc->add_child(save_choose);

		save_choose->connect("pressed", this,"_browse_target");

		file_select = memnew( EditorFileDialog );
		file_select->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
		file_select->set_mode(EditorFileDialog::MODE_OPEN_FILES);
		file_select->add_filter("*.obj ; Wavefront OBJ");
		add_child(file_select);
		file_select->connect("files_selected", this,"_choose_files");

		save_select = memnew( EditorDirDialog );
		add_child(save_select);
		save_select->connect("dir_selected", this,"_choose_save_dir");

		get_ok()->connect("pressed", this,"_import");
		get_ok()->set_text(TTR("Import"));

		error_dialog = memnew( AcceptDialog );
		add_child(error_dialog);

		options = memnew( _EditorMeshImportOptions );

		option_editor = memnew( PropertyEditor );
		option_editor->hide_top_label();
		vbc->add_margin_child(TTR("Options:"),option_editor,true);
	}

	~EditorMeshImportDialog() {
		memdelete(options);
	}

};


String EditorMeshImportPlugin::get_name() const {

	return "mesh";
}
String EditorMeshImportPlugin::get_visible_name() const{

	return TTR("Mesh");
}
void EditorMeshImportPlugin::import_dialog(const String& p_from){

	dialog->popup_import(p_from);
}
Error EditorMeshImportPlugin::import(const String& p_path, const Ref<ResourceImportMetadata>& p_from){


	ERR_FAIL_COND_V(p_from->get_source_count()!=1,ERR_INVALID_PARAMETER);

	Ref<ResourceImportMetadata> from=p_from;

	String src_path=EditorImportPlugin::expand_source_path(from->get_source_path(0));
	FileAccessRef f = FileAccess::open(src_path,FileAccess::READ);
	ERR_FAIL_COND_V(!f,ERR_CANT_OPEN);

	Ref<Mesh> mesh;
	Map<String,Ref<Material> > name_map;

	if (FileAccess::exists(p_path)) {
		mesh=ResourceLoader::load(p_path,"Mesh");
		if (mesh.is_valid()) {
			for(int i=0;i<mesh->get_surface_count();i++) {

				if (!mesh->surface_get_material(i).is_valid())
					continue;
				String name;
				if (mesh->surface_get_name(i)!="")
					name=mesh->surface_get_name(i);
				else
					name=vformat(TTR("Surface %d"),i+1);

				name_map[name]=mesh->surface_get_material(i);
			}

			while(mesh->get_surface_count()) {
				mesh->surface_remove(0);
			}
		}
	}

	if (!mesh.is_valid())
		mesh = Ref<Mesh>( memnew( Mesh ) );


	bool generate_normals=from->get_option("generate/normals");
	bool generate_tangents=from->get_option("generate/tangents");
	bool flip_faces=from->get_option("force/flip_faces");
	bool force_smooth=from->get_option("force/smooth_shading");
	bool weld_vertices=from->get_option("force/weld_vertices");
	float weld_tolerance=from->get_option("force/weld_tolerance");
	Vector<Vector3> vertices;
	Vector<Vector3> normals;
	Vector<Vector2> uvs;
	String name;

	Ref<SurfaceTool> surf_tool = memnew( SurfaceTool) ;
	surf_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
	if (force_smooth)
		surf_tool->add_smooth_group(true);
	int has_index_data=false;

	while(true) {


		String l = f->get_line().strip_edges();

		if (l.begins_with("v ")) {
			//vertex
			Vector<String> v = l.split(" ",false);
			ERR_FAIL_COND_V(v.size()<4,ERR_INVALID_DATA);
			Vector3 vtx;
			vtx.x=v[1].to_float();
			vtx.y=v[2].to_float();
			vtx.z=v[3].to_float();
			vertices.push_back(vtx);
		} else  if (l.begins_with("vt ")) {
			//uv
			Vector<String> v = l.split(" ",false);
			ERR_FAIL_COND_V(v.size()<3,ERR_INVALID_DATA);
			Vector2 uv;
			uv.x=v[1].to_float();
			uv.y=1.0-v[2].to_float();
			uvs.push_back(uv);

		} else if (l.begins_with("vn ")) {
			//normal
			Vector<String> v = l.split(" ",false);
			ERR_FAIL_COND_V(v.size()<4,ERR_INVALID_DATA);
			Vector3 nrm;
			nrm.x=v[1].to_float();
			nrm.y=v[2].to_float();
			nrm.z=v[3].to_float();
			normals.push_back(nrm);
		} if (l.begins_with("f ")) {
				//vertex

			has_index_data=true;
			Vector<String> v = l.split(" ",false);
			ERR_FAIL_COND_V(v.size()<4,ERR_INVALID_DATA);

			//not very fast, could be sped up


			Vector<String> face[3];
			face[0] = v[1].split("/");
			face[1] = v[2].split("/");
			ERR_FAIL_COND_V(face[0].size()==0,ERR_PARSE_ERROR);
			ERR_FAIL_COND_V(face[0].size()!=face[1].size(),ERR_PARSE_ERROR);
			for(int i=2;i<v.size()-1;i++) {

				face[2] = v[i+1].split("/");
				ERR_FAIL_COND_V(face[0].size()!=face[2].size(),ERR_PARSE_ERROR);
				for(int j=0;j<3;j++) {

					int idx=j;

					if (!flip_faces && idx<2) {
						idx=1^idx;
					}


					if (face[idx].size()==3) {
						int norm = face[idx][2].to_int()-1;
						ERR_FAIL_INDEX_V(norm,normals.size(),ERR_PARSE_ERROR);
						surf_tool->add_normal(normals[norm]);
					}

					if (face[idx].size()>=2 && face[idx][1]!=String()) {

						int uv = face[idx][1].to_int()-1;
						ERR_FAIL_INDEX_V(uv,uvs.size(),ERR_PARSE_ERROR);
						surf_tool->add_uv(uvs[uv]);
					}

					int vtx = face[idx][0].to_int()-1;
					ERR_FAIL_INDEX_V(vtx,vertices.size(),ERR_PARSE_ERROR);

					Vector3 vertex = vertices[vtx];
					if (weld_vertices)
						vertex=vertex.snapped(weld_tolerance);
					surf_tool->add_vertex(vertex);
				}

				face[1]=face[2];
			}
		} else if (l.begins_with("s ") && !force_smooth) { //smoothing
			String what = l.substr(2,l.length()).strip_edges();
			if (what=="off")
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
				if (name=="")
					name=vformat(TTR("Surface %d"),mesh->get_surface_count()-1);
				mesh->surface_set_name(mesh->get_surface_count()-1,name);
				name="";
				surf_tool->clear();
				surf_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
				if (force_smooth)
					surf_tool->add_smooth_group(true);

				has_index_data=false;

				if (f->eof_reached())
					break;
			}

			if (l.begins_with("o ")) //name
				name=l.substr(2,l.length()).strip_edges();
		}
	}


	from->set_source_md5(0,FileAccess::get_md5(src_path));
	from->set_editor(get_name());
	mesh->set_import_metadata(from);

	//re-apply materials if exist
	for(int i=0;i<mesh->get_surface_count();i++) {

		String n = mesh->surface_get_name(i);
		if (name_map.has(n))
			mesh->surface_set_material(i,name_map[n]);
	}

	Error err = ResourceSaver::save(p_path,mesh);

	return err;
}


void EditorMeshImportPlugin::import_from_drop(const Vector<String>& p_drop, const String &p_dest_path) {


	Vector<String> files;
	for(int i=0;i<p_drop.size();i++) {
		String ext = p_drop[i].get_extension().to_lower();
		String file = p_drop[i].get_file();
		if (ext=="obj") {

			files.push_back(p_drop[i]);
		}
	}

	if (files.size()) {
		import_dialog();
		dialog->_choose_files(files);
		dialog->_choose_save_dir(p_dest_path);
	}
}

EditorMeshImportPlugin::EditorMeshImportPlugin(EditorNode* p_editor) {

	dialog = memnew( EditorMeshImportDialog(this));
	p_editor->get_gui_base()->add_child(dialog);
}
#endif
