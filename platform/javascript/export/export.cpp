/*************************************************************************/
/*  export.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "version.h"
#include "export.h"
#include "tools/editor/editor_settings.h"
#include "tools/editor/editor_export.h"
#include "tools/editor/editor_node.h"
#include "io/zip_io.h"
#include "io/marshalls.h"
#include "globals.h"
#include "os/file_access.h"
#include "os/os.h"
#include "platform/javascript/logo.h"
#include "string.h"


#if 0
class EditorExportPlatformJavaScript : public EditorExportPlatform {

	GDCLASS( EditorExportPlatformJavaScript,EditorExportPlatform );

	String custom_release_package;
	String custom_debug_package;

	enum PackMode {
		PACK_SINGLE_FILE,
		PACK_MULTIPLE_FILES
	};

	void _fix_html(Vector<uint8_t>& p_html, const String& p_name, bool p_debug);

	PackMode pack_mode;

	bool show_run;

	int max_memory;
	int version_code;

	String html_title;
	String html_head_include;
	String html_font_family;
	String html_style_include;
	bool html_controls_enabled;

	Ref<ImageTexture> logo;

protected:

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

public:

	virtual String get_name() const { return "HTML5"; }
	virtual ImageCompression get_image_compression() const { return IMAGE_COMPRESSION_BC; }
	virtual Ref<Texture> get_logo() const { return logo; }


	virtual bool poll_devices() { return show_run?true:false;}
	virtual int get_device_count() const { return show_run?1:0; };
	virtual String get_device_name(int p_device) const  { return "Run in Browser"; }
	virtual String get_device_info(int p_device) const { return "Run exported HTML in the system's default browser."; }
	virtual Error run(int p_device,int p_flags=0);

	virtual bool requires_password(bool p_debug) const { return false; }
	virtual String get_binary_extension() const { return "html"; }
	virtual Error export_project(const String& p_path,bool p_debug,int p_flags=0);

	virtual bool can_export(String *r_error=NULL) const;

	EditorExportPlatformJavaScript();
	~EditorExportPlatformJavaScript();
};

bool EditorExportPlatformJavaScript::_set(const StringName& p_name, const Variant& p_value) {

	String n=p_name;

	if (n=="custom_package/debug")
		custom_debug_package=p_value;
	else if (n=="custom_package/release")
		custom_release_package=p_value;
	else if (n=="browser/enable_run")
		show_run=p_value;
	else if (n=="options/memory_size")
		max_memory=p_value;
	else if (n=="html/title")
		html_title=p_value;
	else if (n=="html/head_include")
		html_head_include=p_value;
	else if (n=="html/font_family")
		html_font_family=p_value;
	else if (n=="html/style_include")
		html_style_include=p_value;
	else if (n=="html/controls_enabled")
		html_controls_enabled=p_value;
	else
		return false;

	return true;
}

bool EditorExportPlatformJavaScript::_get(const StringName& p_name,Variant &r_ret) const{

	String n=p_name;

	if (n=="custom_package/debug")
		r_ret=custom_debug_package;
	else if (n=="custom_package/release")
		r_ret=custom_release_package;
	else if (n=="browser/enable_run")
		r_ret=show_run;
	else if (n=="options/memory_size")
		r_ret=max_memory;
	else if (n=="html/title")
		r_ret=html_title;
	else if (n=="html/head_include")
		r_ret=html_head_include;
	else if (n=="html/font_family")
		r_ret=html_font_family;
	else if (n=="html/style_include")
		r_ret=html_style_include;
	else if (n=="html/controls_enabled")
		r_ret=html_controls_enabled;
	else
		return false;

	return true;
}
void EditorExportPlatformJavaScript::_get_property_list( List<PropertyInfo> *p_list) const{

	p_list->push_back( PropertyInfo( Variant::STRING, "custom_package/debug", PROPERTY_HINT_GLOBAL_FILE,"zip"));
	p_list->push_back( PropertyInfo( Variant::STRING, "custom_package/release", PROPERTY_HINT_GLOBAL_FILE,"zip"));
	p_list->push_back( PropertyInfo( Variant::INT, "options/memory_size",PROPERTY_HINT_ENUM,"32mb,64mb,128mb,256mb,512mb,1024mb"));
	p_list->push_back( PropertyInfo( Variant::BOOL, "browser/enable_run"));
	p_list->push_back( PropertyInfo( Variant::STRING, "html/title"));
	p_list->push_back( PropertyInfo( Variant::STRING, "html/head_include",PROPERTY_HINT_MULTILINE_TEXT));
	p_list->push_back( PropertyInfo( Variant::STRING, "html/font_family"));
	p_list->push_back( PropertyInfo( Variant::STRING, "html/style_include",PROPERTY_HINT_MULTILINE_TEXT));
	p_list->push_back( PropertyInfo( Variant::BOOL, "html/controls_enabled"));


	//p_list->push_back( PropertyInfo( Variant::INT, "resources/pack_mode", PROPERTY_HINT_ENUM,"Copy,Single Exec.,Pack (.pck),Bundles (Optical)"));

}


void EditorExportPlatformJavaScript::_fix_html(Vector<uint8_t>& p_html, const String& p_name, bool p_debug) {


	String str;
	String strnew;
	str.parse_utf8((const char*)p_html.ptr(),p_html.size());
	Vector<String> lines=str.split("\n");
	for(int i=0;i<lines.size();i++) {

		String current_line = lines[i];
		current_line = current_line.replace("$GODOT_TMEM",itos((1<<(max_memory+5))*1024*1024));
		current_line = current_line.replace("$GODOT_BASE",p_name);
		current_line = current_line.replace("$GODOT_CANVAS_WIDTH",GlobalConfig::get_singleton()->get("display/window/width"));
		current_line = current_line.replace("$GODOT_CANVAS_HEIGHT",GlobalConfig::get_singleton()->get("display/window/height"));
		current_line = current_line.replace("$GODOT_HEAD_TITLE",!html_title.empty()?html_title:(String) GlobalConfig::get_singleton()->get("application/name"));
		current_line = current_line.replace("$GODOT_HEAD_INCLUDE",html_head_include);
		current_line = current_line.replace("$GODOT_STYLE_FONT_FAMILY",html_font_family);
		current_line = current_line.replace("$GODOT_STYLE_INCLUDE",html_style_include);
		current_line = current_line.replace("$GODOT_CONTROLS_ENABLED",html_controls_enabled?"true":"false");
		current_line = current_line.replace("$GODOT_DEBUG_ENABLED",p_debug?"true":"false");
		strnew += current_line+"\n";
	}

	CharString cs = strnew.utf8();
	p_html.resize(cs.length());
	for(int i=9;i<cs.length();i++) {
		p_html[i]=cs[i];
	}
}

static void _fix_files(Vector<uint8_t>& html,uint64_t p_data_size) {


	String str;
	String strnew;
	str.parse_utf8((const char*)html.ptr(),html.size());
	Vector<String> lines=str.split("\n");
	for(int i=0;i<lines.size();i++) {
		if (lines[i].find("$DPLEN")!=-1) {
			strnew+=lines[i].replace("$DPLEN",itos(p_data_size));
		} else {
			strnew+=lines[i]+"\n";
		}
	}

	CharString cs = strnew.utf8();
	html.resize(cs.length());
	for(int i=9;i<cs.length();i++) {
		html[i]=cs[i];
	}

}

struct JSExportData {

	EditorProgress *ep;
	FileAccess *f;

};



Error EditorExportPlatformJavaScript::export_project(const String& p_path, bool p_debug, int p_flags) {


	String src_template;

	EditorProgress ep("export","Exporting for javascript",104);

	if (p_debug)
		src_template=custom_debug_package;
	else
		src_template=custom_release_package;

	if (src_template=="") {
		String err;
		if (p_debug) {
			src_template=find_export_template("javascript_debug.zip", &err);
		} else {
			src_template=find_export_template("javascript_release.zip", &err);
		}
		if (src_template=="") {
			EditorNode::add_io_error(err);
			return ERR_FILE_NOT_FOUND;
		}
	}

	FileAccess *src_f=NULL;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

	ep.step("Exporting to HTML5",0);

	ep.step("Finding Files..",1);

	FileAccess *f=FileAccess::open(p_path.get_base_dir()+"/data.pck",FileAccess::WRITE);
	if (!f) {
		EditorNode::add_io_error("Could not create file for writing:\n"+p_path.get_basename()+"_files.js");
		return ERR_FILE_CANT_WRITE;
	}
	Error err = save_pack(f);
	size_t len = f->get_len();
	memdelete(f);
	if (err)
		return err;


	unzFile pkg = unzOpen2(src_template.utf8().get_data(), &io);
	if (!pkg) {

		EditorNode::add_io_error("Could not find template HTML5 to export:\n"+src_template);
		return ERR_FILE_NOT_FOUND;
	}

	ERR_FAIL_COND_V(!pkg, ERR_CANT_OPEN);
	int ret = unzGoToFirstFile(pkg);


	while(ret==UNZ_OK) {

		//get filename
		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(pkg,&info,fname,16384,NULL,0,NULL,0);

		String file=fname;

		Vector<uint8_t> data;
		data.resize(info.uncompressed_size);

		//read
		unzOpenCurrentFile(pkg);
		unzReadCurrentFile(pkg,data.ptr(),data.size());
		unzCloseCurrentFile(pkg);

		//write

		if (file=="godot.html") {

			_fix_html(data,p_path.get_file().get_basename(), p_debug);
			file=p_path.get_file();
		}
		if (file=="godotfs.js") {

			_fix_files(data,len);
			file=p_path.get_file().get_basename()+"fs.js";
		}
		if (file=="godot.js") {

			file=p_path.get_file().get_basename()+".js";
		}

		if (file=="godot.asm.js") {

			file=p_path.get_file().get_basename()+".asm.js";
		}

		if (file=="godot.mem") {

			file=p_path.get_file().get_basename()+".mem";
		}

		if (file=="godot.wasm") {

			file=p_path.get_file().get_basename()+".wasm";
		}

		String dst = p_path.get_base_dir().plus_file(file);
		FileAccess *f=FileAccess::open(dst,FileAccess::WRITE);
		if (!f) {
			EditorNode::add_io_error("Could not create file for writing:\n"+dst);
			unzClose(pkg);
			return ERR_FILE_CANT_WRITE;
		}
		f->store_buffer(data.ptr(),data.size());
		memdelete(f);


		ret = unzGoToNextFile(pkg);
	}



	return OK;

}


Error EditorExportPlatformJavaScript::run(int p_device, int p_flags) {

	String path = EditorSettings::get_singleton()->get_settings_path()+"/tmp/tmp_export.html";
	Error err = export_project(path,true,p_flags);
	if (err)
		return err;

	OS::get_singleton()->shell_open(path);

	return OK;
}


EditorExportPlatformJavaScript::EditorExportPlatformJavaScript() {

	show_run=false;
	Image img( _javascript_logo );
	logo = Ref<ImageTexture>( memnew( ImageTexture ));
	logo->create_from_image(img);
	max_memory=3;
	html_title="";
	html_font_family="'Droid Sans',arial,sans-serif";
	html_controls_enabled=true;
	pack_mode=PACK_SINGLE_FILE;
}

bool EditorExportPlatformJavaScript::can_export(String *r_error) const {


	bool valid=true;
	String err;

	if (!exists_export_template("javascript_debug.zip") || !exists_export_template("javascript_release.zip")) {
		valid=false;
		err+="No export templates found.\nDownload and install export templates.\n";
	}

	if (custom_debug_package!="" && !FileAccess::exists(custom_debug_package)) {
		valid=false;
		err+="Custom debug package not found.\n";
	}

	if (custom_release_package!="" && !FileAccess::exists(custom_release_package)) {
		valid=false;
		err+="Custom release package not found.\n";
	}

	if (r_error)
		*r_error=err;

	return valid;
}


EditorExportPlatformJavaScript::~EditorExportPlatformJavaScript() {

}

#endif
void register_javascript_exporter() {


	//Ref<EditorExportPlatformJavaScript> exporter = Ref<EditorExportPlatformJavaScript>( memnew(EditorExportPlatformJavaScript) );
	//EditorImportExport::get_singleton()->add_export_platform(exporter);


}


