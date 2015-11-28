/*************************************************************************/
/*  export.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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

#include "export.h"
#include "platform/windows/logo.h"
#include "os/os.h"
#include "globals.h"
#include "tools/editor/editor_node.h"
#include "tools/pe_bliss/pe_bliss_godot.h"

/**
	@author Masoud BaniHashemian <masoudbh3@gmail.com>
*/


void EditorExportPlatformWindows::store_16(DVector<uint8_t>& vector, uint16_t value) {
	const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
	int size = vector.size();
	vector.resize( size + 2 );
	DVector<uint8_t>::Write w = vector.write();
	w[size]=bytes[0];
	w[size+1]=bytes[1];
}
void EditorExportPlatformWindows::store_32(DVector<uint8_t>& vector, uint32_t value) {
	const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
	int size = vector.size();
	vector.resize( size + 4 );
	DVector<uint8_t>::Write w = vector.write();
	w[size]=bytes[0];
	w[size+1]=bytes[1];
	w[size+2]=bytes[2];
	w[size+3]=bytes[3];
}

bool EditorExportPlatformWindows::_set(const StringName& p_name, const Variant& p_value) {

	String n = p_name;

	if (n=="icon/icon_ico") {

		icon_ico=p_value;
	} else if (n=="icon/icon_png") {

		icon_png=p_value;
	} else if (n=="icon/icon_png16x16") {

		icon16=p_value;
	} else if (n=="icon/icon_png32x32") {

		icon32=p_value;
	} else if (n=="icon/icon_png48x48") {

		icon48=p_value;
	} else if (n=="icon/icon_png64x64") {

		icon64=p_value;
	} else if (n=="icon/icon_png128x128") {

		icon128=p_value;
	} else if (n=="icon/icon_png256x256") {

		icon256=p_value;
	} else if (n=="version_info/version_major") {

		version_major=p_value;
	} else if (n=="version_info/version_minor") {

		version_minor=p_value;
	} else if (n=="version_info/version_text") {

		version_text=p_value;
	} else if (n=="version_info/company_name") {

		company_name=p_value;
	} else if (n=="version_info/file_description") {

		file_description=p_value;
	} else if (n=="version_info/product_name") {

		product_name=p_value;
	} else if (n=="version_info/legal_copyright") {

		legal_copyright=p_value;
	} else if (n=="version_info/add_godot_version") {

		set_godot_version=p_value;
	} else
		return false;

	return true;

}

bool EditorExportPlatformWindows::_get(const StringName& p_name,Variant &r_ret) const {

	String n = p_name;

	if (n=="icon/icon_ico") {

		r_ret=icon_ico;
	} else if (n=="icon/icon_png") {

		r_ret=icon_png;
	} else if (n=="icon/icon_png16x16") {

		r_ret=icon16;
	} else if (n=="icon/icon_png32x32") {

		r_ret=icon32;
	} else if (n=="icon/icon_png48x48") {

		r_ret=icon48;
	} else if (n=="icon/icon_png64x64") {

		r_ret=icon64;
	} else if (n=="icon/icon_png128x128") {

		r_ret=icon128;
	} else if (n=="icon/icon_png256x256") {

		r_ret=icon256;
	} else if (n=="version_info/version_major") {

		r_ret=version_major;
	} else if (n=="version_info/version_minor") {

		r_ret=version_minor;
	} else if (n=="version_info/version_text") {

		r_ret=version_text;
	} else if (n=="version_info/company_name") {

		r_ret=company_name;
	} else if (n=="version_info/file_description") {

		r_ret=file_description;
	} else if (n=="version_info/product_name") {

		r_ret=product_name;
	} else if (n=="version_info/legal_copyright") {

		r_ret=legal_copyright;
	} else if (n=="version_info/add_godot_version") {

		r_ret=set_godot_version;
	} else
		return false;

	return true;

}

void EditorExportPlatformWindows::_get_property_list( List<PropertyInfo> *p_list) const {

	p_list->push_back( PropertyInfo( Variant::STRING, "icon/icon_ico",PROPERTY_HINT_FILE,"ico") );
	p_list->push_back( PropertyInfo( Variant::STRING, "icon/icon_png",PROPERTY_HINT_FILE,"png") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "icon/icon_png16x16") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "icon/icon_png32x32") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "icon/icon_png48x48") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "icon/icon_png64x64") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "icon/icon_png128x128") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "icon/icon_png256x256") );
	p_list->push_back( PropertyInfo( Variant::INT, "version_info/version_major", PROPERTY_HINT_RANGE,"0,65535,1"));
	p_list->push_back( PropertyInfo( Variant::INT, "version_info/version_minor", PROPERTY_HINT_RANGE,"0,65535,0"));
	p_list->push_back( PropertyInfo( Variant::STRING, "version_info/version_text") );
	p_list->push_back( PropertyInfo( Variant::STRING, "version_info/company_name") );
	p_list->push_back( PropertyInfo( Variant::STRING, "version_info/file_description") );
	p_list->push_back( PropertyInfo( Variant::STRING, "version_info/product_name") );
	p_list->push_back( PropertyInfo( Variant::STRING, "version_info/legal_copyright") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "version_info/add_godot_version") );
	
}

Error EditorExportPlatformWindows::export_project(const String& p_path, bool p_debug, int p_flags) {

	Error err = EditorExportPlatformPC::export_project(p_path, p_debug, p_flags);
	if(err != OK)
	{
		return err;
	}
	EditorProgress ep("editexe","Edit EXE File",102);
	ep.step("Create ico file..",0);
	
		DVector<uint8_t> icon_content;
		if (this->icon_ico!="" && this->icon_ico.ends_with(".ico")) {
			FileAccess *f = FileAccess::open(this->icon_ico,FileAccess::READ);
			if (f) {
				icon_content.resize(f->get_len());
				DVector<uint8_t>::Write write = icon_content.write();
				f->get_buffer(write.ptr(),icon_content.size());
				f->close();
				memdelete(f);
			}
		} else if (this->icon_png!="" && this->icon_png.ends_with(".png") && (icon16 || icon32 || icon48 || icon64 || icon128 || icon256)) {
			#ifdef PNG_ENABLED
			Vector<Image> pngs;
			Image png;
			Error err_png = png.load(this->icon_png);
			if (err_png==OK && !png.empty()) {
				if(icon256) {
					Image icon_256(png);
					if(!(png.get_height()==256 && png.get_width()==256)) icon_256.resize(256,256);
					pngs.push_back(icon_256);
				}
				if(icon128) {
					Image icon_128(png);
					if(!(png.get_height()==128 && png.get_width()==128)) icon_128.resize(128,128);
					pngs.push_back(icon_128);
				}
				if(icon64) {
					Image icon_64(png);
					if(!(png.get_height()==64 && png.get_width()==64)) icon_64.resize(64,64);
					pngs.push_back(icon_64);
				}
				if(icon48) {
					Image icon_48(png);
					if(!(png.get_height()==48 && png.get_width()==48)) icon_48.resize(48,48);
					pngs.push_back(icon_48);
				}
				if(icon32) {
					Image icon_32(png);
					if(!(png.get_height()==32 && png.get_width()==32)) icon_32.resize(32,32);
					pngs.push_back(icon_32);
				}
				if(icon16) {
					Image icon_16(png);
					if(!(png.get_height()==16 && png.get_width()==16)) icon_16.resize(16,16);
					pngs.push_back(icon_16);
				}
				// create icon according to https://www.daubnet.com/en/file-format-ico
				store_16(icon_content,0); //Reserved
				store_16(icon_content,1); //Type
				store_16(icon_content,pngs.size()); //Count
				int offset = 6+pngs.size()*16;
				//List of bitmaps 
				for(int i=0;i<pngs.size();i++) {
					int w = pngs[i].get_width();
					int h = pngs[i].get_height();
					icon_content.push_back(w<256?w:0); //width
					icon_content.push_back(h<256?h:0); //height
					icon_content.push_back(0); //ColorCount = 0
					icon_content.push_back(0); //Reserved
					store_16(icon_content,1); //Planes
					store_16(icon_content,32); //BitCount (bit per pixel)
					int size = 40 + (w * h * 4) + (w * h / 8);
					store_32(icon_content,size); //Size of (InfoHeader + ANDbitmap + XORbitmap) 
					store_32(icon_content,offset); //FileOffset
					offset += size;
				}
				//Write bmp files.
				for(int i=0;i<pngs.size();i++) {
					int w = pngs[i].get_width();
					int h = pngs[i].get_height();
					store_32(icon_content,40); //Size of InfoHeader structure = 40
					store_32(icon_content,w); //Width
					store_32(icon_content,h*2); //Height
					store_16(icon_content,1); //Planes
					store_16(icon_content,32); //BitCount
					store_32(icon_content,0); //Compression
					store_32(icon_content,w*h*4); //ImageSize = Size of Image in Bytes
					store_32(icon_content,0); //unused = 0 
					store_32(icon_content,0); //unused = 0 
					store_32(icon_content,0); //unused = 0 
					store_32(icon_content,0); //unused = 0 
					//XORBitmap
					for(int y=h-1;y>=0;y--) {
						for(int x=0;x<w;x++) {
							store_32(icon_content,pngs[i].get_pixel(x,y).to_32());
						}
					}
					//ANDBitmap
					for(int m=0;m<(w * h / 8);m+=4) store_32(icon_content,0x00000000); // Add empty ANDBitmap , TODO create full ANDBitmap Structure if need.
				}
			}
			#endif
		}
	
	ep.step("Add rsrc..",50);
	
		String basename = Globals::get_singleton()->get("application/name");
		product_name=product_name.replace("$genname",basename);
		String godot_version;
		if(set_godot_version) godot_version = String( VERSION_MKSTRING );
		String ret = pe_bliss_add_resrc(p_path.utf8(), version_major, version_minor,
																						company_name, file_description, legal_copyright, version_text,
																						product_name, godot_version, icon_content);
		if (ret.empty()) {
			return OK;
		} else {
			EditorNode::add_io_error(ret);
			return ERR_FILE_CANT_WRITE;
		}
}

EditorExportPlatformWindows::EditorExportPlatformWindows() {

	icon16=true;
	icon32=true;
	icon48=true;
	icon64=true;
	icon128=true;
	icon256=true;
	product_name="$genname";
	company_name="Godot Engine";
	file_description="Created With Godot Engine";
	version_text="1.0";
	OS::Date date = OS::get_singleton()->get_date();
	legal_copyright="Copyright (c) 2007-";
	legal_copyright+=String::num(date.year);
	legal_copyright+=" Juan Linietsky, Ariel Manzur";
	version_major=1;
	version_minor=0;
	set_godot_version=true;
}



void register_windows_exporter() {

	Image img(_windows_logo);
	Ref<ImageTexture> logo = memnew( ImageTexture );
	logo->create_from_image(img);

	{
		Ref<EditorExportPlatformWindows> exporter = Ref<EditorExportPlatformWindows>( memnew(EditorExportPlatformWindows) );
		exporter->set_binary_extension("exe");
		exporter->set_release_binary32("windows_32_release.exe");
		exporter->set_debug_binary32("windows_32_debug.exe");
		exporter->set_release_binary64("windows_64_release.exe");
		exporter->set_debug_binary64("windows_64_debug.exe");
		exporter->set_name("Windows Desktop");
		exporter->set_logo(logo);
		EditorImportExport::get_singleton()->add_export_platform(exporter);
	}


}
