#include "editor_export_scene.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/dir_access.h"
#include "os/file_access.h"
#include "tools/editor/editor_settings.h"
#include "scene/resources/packed_scene.h"
#include "globals.h"

Vector<uint8_t> EditorSceneExportPlugin::custom_export(String& p_path,const Ref<EditorExportPlatform> &p_platform) {

	if (!EditorImportExport::get_singleton()->get_convert_text_scenes()) {
		return Vector<uint8_t>();
	}


	String extension = p_path.extension();

	//step 1 check if scene

	if (extension=="xml" || extension=="xres") {

		String type = ResourceLoader::get_resource_type(p_path);

		if (type!="PackedScene")
			return Vector<uint8_t>();

	} else if (extension!="tscn" && extension!="xscn") {
		return Vector<uint8_t>();
	}

	//step 2 check if cached

	uint64_t sd=0;
	String smd5;
	String gp = Globals::get_singleton()->globalize_path(p_path);
	String md5=gp.md5_text();
	String tmp_path = EditorSettings::get_singleton()->get_settings_path().plus_file("tmp/");

	bool valid=false;
	{
		//if existing, make sure it's valid
		FileAccessRef f = FileAccess::open(tmp_path+"scnexp-"+md5+".txt",FileAccess::READ);
		if (f) {

			uint64_t d = f->get_line().strip_edges().to_int64();
			sd = FileAccess::get_modified_time(p_path);

			if (d==sd) {
				valid=true;
			} else {
				String cmd5 = f->get_line().strip_edges();
				smd5 = FileAccess::get_md5(p_path);
				if (cmd5==smd5) {
					valid=true;
				}
			}


		}
	}

	if (!valid) {
		//cache failed, convert
		DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);

		String copy = p_path+".convert."+extension;

		// a copy will allow loading the internal resources without conflicting with opened scenes
		da->copy(p_path,copy);

		//@todo for tscn use something more efficient

		Ref<PackedScene> copyres =  ResourceLoader::load(copy,"PackedScene");

		da->remove(copy);

		memdelete(da);

		ERR_FAIL_COND_V(!copyres.is_valid(),Vector<uint8_t>());

		Error err = ResourceSaver::save(tmp_path+"scnexp-"+md5+".scn",copyres);

		copyres=Ref<PackedScene>();

		ERR_FAIL_COND_V(err!=OK,Vector<uint8_t>());

		FileAccessRef f = FileAccess::open(tmp_path+"scnexp-"+md5+".txt",FileAccess::WRITE);

		if (sd==0)
			sd = FileAccess::get_modified_time(p_path);
		if (smd5==String())
			smd5 = FileAccess::get_md5(p_path);

		f->store_line(String::num(sd));
		f->store_line(smd5);
		f->store_line(gp); //source path for reference
	}


	Vector<uint8_t> ret = FileAccess::get_file_as_array(tmp_path+"scnexp-"+md5+".scn");

	p_path+=".optimized.scn";

	return ret;

}


EditorSceneExportPlugin::EditorSceneExportPlugin()
{
}
