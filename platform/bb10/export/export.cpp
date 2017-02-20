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
#include "platform/bb10/logo.h"
#include "io/xml_parser.h"

#define MAX_DEVICES 5
#if 0
class EditorExportPlatformBB10 : public EditorExportPlatform {

	GDCLASS( EditorExportPlatformBB10,EditorExportPlatform );

	String custom_package;

	int version_code;
	String version_name;
	String package;
	String name;
	String category;
	String description;
	String author_name;
	String author_id;
	String icon;



	struct Device {

		int index;
		String name;
		String description;
	};

	Vector<Device> devices;
	bool devices_changed;
	Mutex *device_lock;
	Thread *device_thread;
	Ref<ImageTexture> logo;

	volatile bool quit_request;


	static void _device_poll_thread(void *ud);

	void _fix_descriptor(Vector<uint8_t>& p_manifest);
protected:

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

public:

	virtual String get_name() const { return "BlackBerry 10"; }
	virtual ImageCompression get_image_compression() const { return IMAGE_COMPRESSION_ETC1; }
	virtual Ref<Texture> get_logo() const { return logo; }


	virtual bool poll_devices();
	virtual int get_device_count() const;
	virtual String get_device_name(int p_device) const;
	virtual String get_device_info(int p_device) const;
	virtual Error run(int p_device,int p_flags=0);

	virtual bool requires_password(bool p_debug) const { return !p_debug; }
	virtual String get_binary_extension() const { return "bar"; }
	virtual Error export_project(const String& p_path,bool p_debug,int p_flags=0);

	virtual bool can_export(String *r_error=NULL) const;

	EditorExportPlatformBB10();
	~EditorExportPlatformBB10();
};

bool EditorExportPlatformBB10::_set(const StringName& p_name, const Variant& p_value) {

	String n=p_name;

	if (n=="version/code")
		version_code=p_value;
	else if (n=="version/name")
		version_name=p_value;
	else if (n=="package/unique_name")
		package=p_value;
	else if (n=="package/category")
		category=p_value;
	else if (n=="package/name")
		name=p_value;
	else if (n=="package/description")
		description=p_value;
	else if (n=="package/icon")
		icon=p_value;
	else if (n=="package/custom_template")
		custom_package=p_value;
	else if (n=="release/author")
		author_name=p_value;
	else if (n=="release/author_id")
		author_id=p_value;
	else
		return false;

	return true;
}

bool EditorExportPlatformBB10::_get(const StringName& p_name,Variant &r_ret) const{

	String n=p_name;

	if (n=="version/code")
		r_ret=version_code;
	else if (n=="version/name")
		r_ret=version_name;
	else if (n=="package/unique_name")
		r_ret=package;
	else if (n=="package/category")
		r_ret=category;
	else if (n=="package/name")
		r_ret=name;
	else if (n=="package/description")
		r_ret=description;
	else if (n=="package/icon")
		r_ret=icon;
	else if (n=="package/custom_template")
		r_ret=custom_package;
	else if (n=="release/author")
		r_ret=author_name;
	else if (n=="release/author_id")
		r_ret=author_id;
	else
		return false;

	return true;
}
void EditorExportPlatformBB10::_get_property_list( List<PropertyInfo> *p_list) const{

	p_list->push_back( PropertyInfo( Variant::INT, "version/code", PROPERTY_HINT_RANGE,"1,65535,1"));
	p_list->push_back( PropertyInfo( Variant::STRING, "version/name") );
	p_list->push_back( PropertyInfo( Variant::STRING, "package/unique_name") );
	p_list->push_back( PropertyInfo( Variant::STRING, "package/category") );
	p_list->push_back( PropertyInfo( Variant::STRING, "package/name") );
	p_list->push_back( PropertyInfo( Variant::STRING, "package/description",PROPERTY_HINT_MULTILINE_TEXT) );
	p_list->push_back( PropertyInfo( Variant::STRING, "package/icon",PROPERTY_HINT_FILE,"png") );
	p_list->push_back( PropertyInfo( Variant::STRING, "package/custom_template", PROPERTY_HINT_GLOBAL_FILE,"zip"));
	p_list->push_back( PropertyInfo( Variant::STRING, "release/author") );
	p_list->push_back( PropertyInfo( Variant::STRING, "release/author_id") );

	//p_list->push_back( PropertyInfo( Variant::INT, "resources/pack_mode", PROPERTY_HINT_ENUM,"Copy,Single Exec.,Pack (.pck),Bundles (Optical)"));

}

void EditorExportPlatformBB10::_fix_descriptor(Vector<uint8_t>& p_descriptor) {

	String fpath =  EditorSettings::get_singleton()->get_settings_path().plus_file("tmp_bar-settings.xml");
	{
		FileAccessRef f = FileAccess::open(fpath,FileAccess::WRITE);
		f->store_buffer(p_descriptor.ptr(),p_descriptor.size());
	}

	Ref<XMLParser> parser = memnew( XMLParser );
	Error err = parser->open(fpath);
	ERR_FAIL_COND(err!=OK);

	String txt;
	err = parser->read();
	Vector<String> depth;

	while(err!=ERR_FILE_EOF) {

		ERR_FAIL_COND(err!=OK);

		switch(parser->get_node_type()) {

			case XMLParser::NODE_NONE: {
				print_line("???");
			} break;
			case XMLParser::NODE_ELEMENT: {
				String e="<";
				e+=parser->get_node_name();
				for(int i=0;i<parser->get_attribute_count();i++) {
					e+=" ";
					e+=parser->get_attribute_name(i)+"=\"";
					e+=parser->get_attribute_value(i)+"\" ";
				}



				if (parser->is_empty()) {
					e+="/";
				} else {
					depth.push_back(parser->get_node_name());
				}

				e+=">";
				txt+=e;

			} break;
			case XMLParser::NODE_ELEMENT_END: {

				txt+="</"+parser->get_node_name()+">";
				if (depth.size() && depth[depth.size()-1]==parser->get_node_name()) {
					depth.resize(depth.size()-1);
				}


			} break;
			case XMLParser::NODE_TEXT: {
				if (depth.size()==2 && depth[0]=="qnx" && depth[1]=="id") {

					txt+=package;
				} else if (depth.size()==2 && depth[0]=="qnx" && depth[1]=="name") {

					String aname;
					if (this->name!="") {
						aname=this->name;
					} else {
						aname = GlobalConfig::get_singleton()->get("application/name");

					}

					if (aname=="") {
						aname=_MKSTR(VERSION_NAME);
					}

					txt+=aname;

				} else if (depth.size()==2 && depth[0]=="qnx" && depth[1]=="versionNumber") {
					txt+=itos(version_code);
				} else if (depth.size()==2 && depth[0]=="qnx" && depth[1]=="description") {
					txt+=description;
				} else if (depth.size()==2 && depth[0]=="qnx" && depth[1]=="author") {
					txt+=author_name;
				} else if (depth.size()==2 && depth[0]=="qnx" && depth[1]=="authorId") {
					txt+=author_id;
				} else if (depth.size()==2 && depth[0]=="qnx" && depth[1]=="category") {
					txt+=category;
				} else {
					txt+=parser->get_node_data();
				}
			} break;
			case XMLParser::NODE_COMMENT: {
				txt+="<!--"+parser->get_node_name()+"-->";
			} break;
			case XMLParser::NODE_CDATA: {
				//ignore
				//print_line("cdata");
			} break;
			case XMLParser::NODE_UNKNOWN: {
				//ignore
				txt+="<"+parser->get_node_name()+">";
			} break;
		}

		err = parser->read();
	}


	CharString cs = txt.utf8();
	p_descriptor.resize(cs.length());
	for(int i=0;i<cs.length();i++)
		p_descriptor[i]=cs[i];

}



Error EditorExportPlatformBB10::export_project(const String& p_path, bool p_debug, int p_flags) {


	EditorProgress ep("export","Exporting for BlackBerry 10",104);

	String src_template=custom_package;

	if (src_template=="") {
		String err;
		src_template = find_export_template("bb10.zip", &err);
		if (src_template=="") {
			EditorNode::add_io_error(err);
			return ERR_FILE_NOT_FOUND;
		}
	}

	FileAccess *src_f=NULL;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

	ep.step("Creating FileSystem for BAR",0);

	unzFile pkg = unzOpen2(src_template.utf8().get_data(), &io);
	if (!pkg) {

		EditorNode::add_io_error("Could not find template zip to export:\n"+src_template);
		return ERR_FILE_NOT_FOUND;
	}

	DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	da->change_dir(EditorSettings::get_singleton()->get_settings_path());


	if (da->change_dir("tmp")!=OK) {
		da->make_dir("tmp");
		if (da->change_dir("tmp")!=OK)
			return ERR_CANT_CREATE;
	}

	if (da->change_dir("bb10_export")!=OK) {
		da->make_dir("bb10_export");
		if (da->change_dir("bb10_export")!=OK) {
			return ERR_CANT_CREATE;
		}
	}


	String bar_dir = da->get_current_dir();
	if (bar_dir.ends_with("/")) {
		bar_dir=bar_dir.substr(0,bar_dir.length()-1);
	}

	//THIS IS SUPER, SUPER DANGEROUS!!!!
	//CAREFUL WITH THIS CODE, MIGHT DELETE USERS HARD DRIVE OR HOME DIR
	//EXTRA CHECKS ARE IN PLACE EVERYWERE TO MAKE SURE NOTHING BAD HAPPENS BUT STILL....
	//BE SUPER CAREFUL WITH THIS PLEASE!!!
	//BLACKBERRY THIS IS YOUR FAULT FOR NOT MAKING A BETTER WAY!!

	bool berr = bar_dir.ends_with("bb10_export");
	if (berr) {
		if (da->list_dir_begin()) {
			EditorNode::add_io_error("Can't ensure that dir is empty:\n"+bar_dir);
			ERR_FAIL_COND_V(berr,FAILED);
		};

		String f = da->get_next();
		while (f != "") {

			if (f == "." || f == "..") {
				f = da->get_next();
				continue;
			};
			Error err = da->remove(bar_dir + "/" + f);
			if (err != OK) {
				EditorNode::add_io_error("Can't ensure that dir is empty:\n"+bar_dir);
				ERR_FAIL_COND_V(err!=OK,err);
			};
			f = da->get_next();
		};

		da->list_dir_end();

	} else {
		print_line("ARE YOU CRAZY??? THIS IS A SERIOUS BUG HERE!!!");
		ERR_FAIL_V(ERR_OMFG_THIS_IS_VERY_VERY_BAD);
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

		if (file=="bar-descriptor.xml") {

			_fix_descriptor(data);
		}

		if (file=="icon.png") {
			bool found=false;

			if (this->icon!="" && this->icon.ends_with(".png")) {

				FileAccess *f = FileAccess::open(this->icon,FileAccess::READ);
				if (f) {

					data.resize(f->get_len());
					f->get_buffer(data.ptr(),data.size());
					memdelete(f);
					found=true;
				}

			}

			if (!found) {

				String appicon = GlobalConfig::get_singleton()->get("application/icon");
				if (appicon!="" && appicon.ends_with(".png")) {
					FileAccess*f = FileAccess::open(appicon,FileAccess::READ);
					if (f) {
						data.resize(f->get_len());
						f->get_buffer(data.ptr(),data.size());
						memdelete(f);
					}
				}
			}
		}


		if (file.find("/")) {

			da->make_dir_recursive(file.get_base_dir());
		}

		FileAccessRef wf = FileAccess::open(bar_dir.plus_file(file),FileAccess::WRITE);
		wf->store_buffer(data.ptr(),data.size());

		ret = unzGoToNextFile(pkg);
	}

	ep.step("Adding Files..",2);

	FileAccess* dst = FileAccess::open(bar_dir+"/data.pck", FileAccess::WRITE);
	if (!dst) {
		EditorNode::add_io_error("Can't copy executable file to:\n "+p_path);
		return ERR_FILE_CANT_WRITE;
	}
	save_pack(dst, false, 1024);
	dst->close();
	memdelete(dst);

	ep.step("Creating BAR Package..",104);

	String bb_packager=EditorSettings::get_singleton()->get("export/blackberry/host_tools");
	bb_packager=bb_packager.plus_file("blackberry-nativepackager");
	if (OS::get_singleton()->get_name()=="Windows")
		bb_packager+=".bat";


	if (!FileAccess::exists(bb_packager)) {
		EditorNode::add_io_error("Can't find packager:\n"+bb_packager);
		return ERR_CANT_OPEN;
	}

	List<String> args;
	args.push_back("-package");
	args.push_back(p_path);
	if (p_debug) {

		String debug_token=EditorSettings::get_singleton()->get("export/blackberry/debug_token");
		if (!FileAccess::exists(debug_token)) {
			EditorNode::add_io_error("Debug token not found!");
		} else {
			args.push_back("-debugToken");
			args.push_back(debug_token);
		}
		args.push_back("-devMode");
		args.push_back("-configuration");
		args.push_back("Device-Debug");
	} else {

		args.push_back("-configuration");
		args.push_back("Device-Release");
	}
	args.push_back(bar_dir.plus_file("bar-descriptor.xml"));

	int ec;

	Error err = OS::get_singleton()->execute(bb_packager,args,true,NULL,NULL,&ec);

	if (err!=OK)
		return err;
	if (ec!=0)
		return ERR_CANT_CREATE;

	return OK;

}

bool EditorExportPlatformBB10::poll_devices() {

	bool dc=devices_changed;
	devices_changed=false;
	return dc;
}

int EditorExportPlatformBB10::get_device_count() const {

	device_lock->lock();
	int dc=devices.size();
	device_lock->unlock();

	return dc;

}
String EditorExportPlatformBB10::get_device_name(int p_device) const {

	ERR_FAIL_INDEX_V(p_device,devices.size(),"");
	device_lock->lock();
	String s=devices[p_device].name;
	device_lock->unlock();
	return s;
}
String EditorExportPlatformBB10::get_device_info(int p_device) const {

	ERR_FAIL_INDEX_V(p_device,devices.size(),"");
	device_lock->lock();
	String s=devices[p_device].description;
	device_lock->unlock();
	return s;
}

void EditorExportPlatformBB10::_device_poll_thread(void *ud) {

	EditorExportPlatformBB10 *ea=(EditorExportPlatformBB10 *)ud;

	while(!ea->quit_request) {

		String bb_deploy=EditorSettings::get_singleton()->get("export/blackberry/host_tools");
		bb_deploy=bb_deploy.plus_file("blackberry-deploy");
		bool windows = OS::get_singleton()->get_name()=="Windows";
		if (windows)
			bb_deploy+=".bat";

		if (FileAccess::exists(bb_deploy)) {

			Vector<Device> devices;


			for (int i=0;i<MAX_DEVICES;i++) {

				String host = EditorSettings::get_singleton()->get("export/blackberry/device_"+itos(i+1)+"/host");
				if (host==String())
					continue;
				String pass = EditorSettings::get_singleton()->get("export/blackberry/device_"+itos(i+1)+"/password");
				if (pass==String())
					continue;

				List<String> args;
				args.push_back("-listDeviceInfo");
				args.push_back(host);
				args.push_back("-password");
				args.push_back(pass);


				int ec;
				String dp;

				Error err = OS::get_singleton()->execute(bb_deploy,args,true,NULL,&dp,&ec);

				if (err==OK && ec==0) {

					Device dev;
					dev.index=i;
					String descr;
					Vector<String> ls=dp.split("\n");

					for(int i=0;i<ls.size();i++) {

						String l = ls[i].strip_edges();
						if (l.begins_with("modelfullname::")) {
							dev.name=l.get_slice("::",1);
							descr+="Model: "+dev.name+"\n";
						}
						if (l.begins_with("modelnumber::")) {
							String s = l.get_slice("::",1);
							dev.name+=" ("+s+")";
							descr+="Model Number: "+s+"\n";
						}
						if (l.begins_with("scmbundle::"))
							descr+="OS Version: "+l.get_slice("::",1)+"\n";
						if (l.begins_with("[n]debug_token_expiration::"))
							descr+="Debug Token Expires:: "+l.get_slice("::",1)+"\n";

					}

					dev.description=descr;
					devices.push_back(dev);
				}

			}

			bool changed=false;


			ea->device_lock->lock();

			if (ea->devices.size()!=devices.size()) {
				changed=true;
			} else {

				for(int i=0;i<ea->devices.size();i++) {

					if (ea->devices[i].index!=devices[i].index) {
						changed=true;
						break;
					}
				}
			}

			if (changed) {

				ea->devices=devices;
				ea->devices_changed=true;
			}

			ea->device_lock->unlock();
		}


		uint64_t wait = 3000000;
		uint64_t time = OS::get_singleton()->get_ticks_usec();
		while(OS::get_singleton()->get_ticks_usec() - time < wait ) {
			OS::get_singleton()->delay_usec(1000);
			if (ea->quit_request)
				break;
		}
	}

}

Error EditorExportPlatformBB10::run(int p_device, int p_flags) {

	ERR_FAIL_INDEX_V(p_device,devices.size(),ERR_INVALID_PARAMETER);

	String bb_deploy=EditorSettings::get_singleton()->get("export/blackberry/host_tools");
	bb_deploy=bb_deploy.plus_file("blackberry-deploy");
	if (OS::get_singleton()->get_name()=="Windows")
		bb_deploy+=".bat";

	if (!FileAccess::exists(bb_deploy)) {
		EditorNode::add_io_error("Blackberry Deploy not found:\n"+bb_deploy);
		return ERR_FILE_NOT_FOUND;
	}


	device_lock->lock();


	EditorProgress ep("run","Running on "+devices[p_device].name,3);

	//export_temp
	ep.step("Exporting APK",0);

	String export_to=EditorSettings::get_singleton()->get_settings_path().plus_file("/tmp/tmpexport.bar");
	Error err = export_project(export_to,true,p_flags);
	if (err) {
		device_lock->unlock();
		return err;
	}
#if 0
	ep.step("Uninstalling..",1);

	print_line("Uninstalling previous version: "+devices[p_device].name);
	List<String> args;
	args.push_back("-s");
	args.push_back(devices[p_device].id);
	args.push_back("uninstall");
	args.push_back(package);
	int rv;
	err = OS::get_singleton()->execute(adb,args,true,NULL,NULL,&rv);

	if (err || rv!=0) {
		EditorNode::add_io_error("Could not install to device.");
		device_lock->unlock();
		return ERR_CANT_CREATE;
	}

	print_line("Installing into device (please wait..): "+devices[p_device].name);

#endif
	ep.step("Installing to Device (please wait..)..",2);

	List<String> args;
	args.clear();
	args.push_back("-installApp");
	args.push_back("-launchApp");
	args.push_back("-device");
	String host = EditorSettings::get_singleton()->get("export/blackberry/device_"+itos(p_device+1)+"/host");
	String pass = EditorSettings::get_singleton()->get("export/blackberry/device_"+itos(p_device+1)+"/password");
	args.push_back(host);
	args.push_back("-password");
	args.push_back(pass);
	args.push_back(export_to);

	int rv;
	err = OS::get_singleton()->execute(bb_deploy,args,true,NULL,NULL,&rv);
	if (err || rv!=0) {
		EditorNode::add_io_error("Could not install to device.");
		device_lock->unlock();
		return ERR_CANT_CREATE;
	}

	device_lock->unlock();
	return OK;


}


EditorExportPlatformBB10::EditorExportPlatformBB10() {

	version_code=1;
	version_name="1.0";
	package="com.godot.noname";
	category="core.games";
	name="";
	author_name="Cert. Name";
	author_id="Cert. ID";
	description="Game made with Godot Engine";

	device_lock = Mutex::create();
	quit_request=false;

	device_thread=Thread::create(_device_poll_thread,this);
	devices_changed=true;

	Image img( _bb10_logo );
	logo = Ref<ImageTexture>( memnew( ImageTexture ));
	logo->create_from_image(img);
}

bool EditorExportPlatformBB10::can_export(String *r_error) const {

	bool valid=true;
	String bb_deploy=EditorSettings::get_singleton()->get("export/blackberry/host_tools");
	String err;

	if (!FileAccess::exists(bb_deploy.plus_file("blackberry-deploy"))) {

		valid=false;
		err+="Blackberry host tools not configured in editor settings.\n";
	}

	if (!exists_export_template("bb10.zip")) {
		valid=false;
		err+="No export template found.\nDownload and install export templates.\n";
	}

	String debug_token=EditorSettings::get_singleton()->get("export/blackberry/debug_token");

	if (!FileAccess::exists(debug_token)) {
		valid=false;
		err+="No debug token set, will not be able to test on device.\n";
	}


	if (custom_package!="" && !FileAccess::exists(custom_package)) {
		valid=false;
		err+="Custom release package not found.\n";
	}

	if (r_error)
		*r_error=err;

	return valid;
}


EditorExportPlatformBB10::~EditorExportPlatformBB10() {

	quit_request=true;
	Thread::wait_to_finish(device_thread);
	memdelete(device_lock);
	memdelete(device_thread);
}

#endif
void register_bb10_exporter() {
#if 0
	EDITOR_DEF("export/blackberry/host_tools","");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING,"export/blackberry/host_tools",PROPERTY_HINT_GLOBAL_DIR));
	EDITOR_DEF("export/blackberry/debug_token","");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING,"export/blackberry/debug_token",PROPERTY_HINT_GLOBAL_FILE,"bar"));
	EDITOR_DEF("export/blackberry/device_1/host","");
	EDITOR_DEF("export/blackberry/device_1/password","");
	EDITOR_DEF("export/blackberry/device_2/host","");
	EDITOR_DEF("export/blackberry/device_2/password","");
	EDITOR_DEF("export/blackberry/device_3/host","");
	EDITOR_DEF("export/blackberry/device_3/password","");
	EDITOR_DEF("export/blackberry/device_4/host","");
	EDITOR_DEF("export/blackberry/device_4/password","");
	EDITOR_DEF("export/blackberry/device_5/host","");
	EDITOR_DEF("export/blackberry/device_5/password","");

	Ref<EditorExportPlatformBB10> exporter = Ref<EditorExportPlatformBB10>( memnew(EditorExportPlatformBB10) );
	EditorImportExport::get_singleton()->add_export_platform(exporter);

#endif
}


