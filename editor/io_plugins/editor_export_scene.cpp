/*************************************************************************/
/*  editor_export_scene.cpp                                              */
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
#include "editor_export_scene.h"
#if 0
#include "editor/editor_settings.h"
#include "global_config.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/dir_access.h"
#include "os/file_access.h"
#include "scene/resources/packed_scene.h"

Vector<uint8_t> EditorSceneExportPlugin::custom_export(String& p_path,const Ref<EditorExportPlatform> &p_platform) {

	if (!EditorImportExport::get_singleton()->get_convert_text_scenes()) {
		return Vector<uint8_t>();
	}


	String extension = p_path.get_extension();

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
	String gp = GlobalConfig::get_singleton()->globalize_path(p_path);
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

	p_path+=".converted.scn";

	return ret;

}


EditorSceneExportPlugin::EditorSceneExportPlugin()
{
}
#endif
