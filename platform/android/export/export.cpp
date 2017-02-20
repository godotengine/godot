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
#include "platform/android/logo.h"
#include <string.h>
#if 0

static const char* android_perms[]={
"ACCESS_CHECKIN_PROPERTIES",
"ACCESS_COARSE_LOCATION",
"ACCESS_FINE_LOCATION",
"ACCESS_LOCATION_EXTRA_COMMANDS",
"ACCESS_MOCK_LOCATION",
"ACCESS_NETWORK_STATE",
"ACCESS_SURFACE_FLINGER",
"ACCESS_WIFI_STATE",
"ACCOUNT_MANAGER",
"ADD_VOICEMAIL",
"AUTHENTICATE_ACCOUNTS",
"BATTERY_STATS",
"BIND_ACCESSIBILITY_SERVICE",
"BIND_APPWIDGET",
"BIND_DEVICE_ADMIN",
"BIND_INPUT_METHOD",
"BIND_NFC_SERVICE",
"BIND_NOTIFICATION_LISTENER_SERVICE",
"BIND_PRINT_SERVICE",
"BIND_REMOTEVIEWS",
"BIND_TEXT_SERVICE",
"BIND_VPN_SERVICE",
"BIND_WALLPAPER",
"BLUETOOTH",
"BLUETOOTH_ADMIN",
"BLUETOOTH_PRIVILEGED",
"BRICK",
"BROADCAST_PACKAGE_REMOVED",
"BROADCAST_SMS",
"BROADCAST_STICKY",
"BROADCAST_WAP_PUSH",
"CALL_PHONE",
"CALL_PRIVILEGED",
"CAMERA",
"CAPTURE_AUDIO_OUTPUT",
"CAPTURE_SECURE_VIDEO_OUTPUT",
"CAPTURE_VIDEO_OUTPUT",
"CHANGE_COMPONENT_ENABLED_STATE",
"CHANGE_CONFIGURATION",
"CHANGE_NETWORK_STATE",
"CHANGE_WIFI_MULTICAST_STATE",
"CHANGE_WIFI_STATE",
"CLEAR_APP_CACHE",
"CLEAR_APP_USER_DATA",
"CONTROL_LOCATION_UPDATES",
"DELETE_CACHE_FILES",
"DELETE_PACKAGES",
"DEVICE_POWER",
"DIAGNOSTIC",
"DISABLE_KEYGUARD",
"DUMP",
"EXPAND_STATUS_BAR",
"FACTORY_TEST",
"FLASHLIGHT",
"FORCE_BACK",
"GET_ACCOUNTS",
"GET_PACKAGE_SIZE",
"GET_TASKS",
"GET_TOP_ACTIVITY_INFO",
"GLOBAL_SEARCH",
"HARDWARE_TEST",
"INJECT_EVENTS",
"INSTALL_LOCATION_PROVIDER",
"INSTALL_PACKAGES",
"INSTALL_SHORTCUT",
"INTERNAL_SYSTEM_WINDOW",
"INTERNET",
"KILL_BACKGROUND_PROCESSES",
"LOCATION_HARDWARE",
"MANAGE_ACCOUNTS",
"MANAGE_APP_TOKENS",
"MANAGE_DOCUMENTS",
"MASTER_CLEAR",
"MEDIA_CONTENT_CONTROL",
"MODIFY_AUDIO_SETTINGS",
"MODIFY_PHONE_STATE",
"MOUNT_FORMAT_FILESYSTEMS",
"MOUNT_UNMOUNT_FILESYSTEMS",
"NFC",
"PERSISTENT_ACTIVITY",
"PROCESS_OUTGOING_CALLS",
"READ_CALENDAR",
"READ_CALL_LOG",
"READ_CONTACTS",
"READ_EXTERNAL_STORAGE",
"READ_FRAME_BUFFER",
"READ_HISTORY_BOOKMARKS",
"READ_INPUT_STATE",
"READ_LOGS",
"READ_PHONE_STATE",
"READ_PROFILE",
"READ_SMS",
"READ_SOCIAL_STREAM",
"READ_SYNC_SETTINGS",
"READ_SYNC_STATS",
"READ_USER_DICTIONARY",
"REBOOT",
"RECEIVE_BOOT_COMPLETED",
"RECEIVE_MMS",
"RECEIVE_SMS",
"RECEIVE_WAP_PUSH",
"RECORD_AUDIO",
"REORDER_TASKS",
"RESTART_PACKAGES",
"SEND_RESPOND_VIA_MESSAGE",
"SEND_SMS",
"SET_ACTIVITY_WATCHER",
"SET_ALARM",
"SET_ALWAYS_FINISH",
"SET_ANIMATION_SCALE",
"SET_DEBUG_APP",
"SET_ORIENTATION",
"SET_POINTER_SPEED",
"SET_PREFERRED_APPLICATIONS",
"SET_PROCESS_LIMIT",
"SET_TIME",
"SET_TIME_ZONE",
"SET_WALLPAPER",
"SET_WALLPAPER_HINTS",
"SIGNAL_PERSISTENT_PROCESSES",
"STATUS_BAR",
"SUBSCRIBED_FEEDS_READ",
"SUBSCRIBED_FEEDS_WRITE",
"SYSTEM_ALERT_WINDOW",
"TRANSMIT_IR",
"UNINSTALL_SHORTCUT",
"UPDATE_DEVICE_STATS",
"USE_CREDENTIALS",
"USE_SIP",
"VIBRATE",
"WAKE_LOCK",
"WRITE_APN_SETTINGS",
"WRITE_CALENDAR",
"WRITE_CALL_LOG",
"WRITE_CONTACTS",
"WRITE_EXTERNAL_STORAGE",
"WRITE_GSERVICES",
"WRITE_HISTORY_BOOKMARKS",
"WRITE_PROFILE",
"WRITE_SECURE_SETTINGS",
"WRITE_SETTINGS",
"WRITE_SMS",
"WRITE_SOCIAL_STREAM",
"WRITE_SYNC_SETTINGS",
"WRITE_USER_DICTIONARY",
NULL};

class EditorExportPlatformAndroid : public EditorExportPlatform {

	GDCLASS( EditorExportPlatformAndroid,EditorExportPlatform );


	enum {
		MAX_USER_PERMISSIONS=20,
		SCREEN_SMALL=0,
		SCREEN_NORMAL=1,
		SCREEN_LARGE=2,
		SCREEN_XLARGE=3,
		SCREEN_MAX=4
	};

	String custom_release_package;
	String custom_debug_package;

	int version_code;
	String version_name;
	String package;
	String name;
	String icon;
	String cmdline;
	bool _signed;
	bool apk_expansion;
	bool remove_prev;
	bool use_32_fb;
	bool immersive;
	bool export_arm;
	bool export_x86;
	String apk_expansion_salt;
	String apk_expansion_pkey;
	int orientation;

	String release_keystore;
	String release_password;
	String release_username;

	struct APKExportData {

		zipFile apk;
		EditorProgress *ep;
	};

	struct Device {

		String id;
		String name;
		String description;
	};

	Vector<Device> devices;
	bool devices_changed;
	Mutex *device_lock;
	Thread *device_thread;
	Ref<ImageTexture> logo;

	Set<String> perms;
	String user_perms[MAX_USER_PERMISSIONS];
	bool screen_support[SCREEN_MAX];

	volatile bool quit_request;


	static void _device_poll_thread(void *ud);

	String get_package_name();

	String get_project_name() const;
	void _fix_manifest(Vector<uint8_t>& p_manifest, bool p_give_internet);
	void _fix_resources(Vector<uint8_t>& p_manifest);
	static Error save_apk_file(void *p_userdata,const String& p_path, const Vector<uint8_t>& p_data,int p_file,int p_total);
	static bool _should_compress_asset(const String& p_path, const Vector<uint8_t>& p_data);

protected:

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

public:

	virtual String get_name() const { return "Android"; }
	virtual ImageCompression get_image_compression() const { return IMAGE_COMPRESSION_ETC1; }
	virtual Ref<Texture> get_logo() const { return logo; }


	virtual bool poll_devices();
	virtual int get_device_count() const;
	virtual String get_device_name(int p_device) const;
	virtual String get_device_info(int p_device) const;
	virtual Error run(int p_device,int p_flags=0);

	virtual bool requires_password(bool p_debug) const { return !p_debug; }
	virtual String get_binary_extension() const { return "apk"; }
	virtual Error export_project(const String& p_path, bool p_debug, int p_flags=0);

	virtual bool can_export(String *r_error=NULL) const;

	EditorExportPlatformAndroid();
	~EditorExportPlatformAndroid();
};

bool EditorExportPlatformAndroid::_set(const StringName& p_name, const Variant& p_value) {

	String n=p_name;

	if (n=="one_click_deploy/clear_previous_install")
		remove_prev=p_value;
	else if (n=="custom_package/debug")
		custom_debug_package=p_value;
	else if (n=="custom_package/release")
		custom_release_package=p_value;
	else if (n=="version/code")
		version_code=p_value;
	else if (n=="version/name")
		version_name=p_value;
	else if (n=="command_line/extra_args")
		cmdline=p_value;
	else if (n=="package/unique_name")
		package=p_value;
	else if (n=="package/name")
		name=p_value;
	else if (n=="package/icon")
		icon=p_value;
	else if (n=="package/signed")
		_signed=p_value;
	else if (n=="architecture/arm")
		export_arm=p_value;
	else if (n=="architecture/x86")
		export_x86=p_value;
	else if (n=="screen/use_32_bits_view")
		use_32_fb=p_value;
	else if (n=="screen/immersive_mode")
		immersive=p_value;
	else if (n=="screen/orientation")
		orientation=p_value;
	else if (n=="screen/support_small")
		screen_support[SCREEN_SMALL]=p_value;
	else if (n=="screen/support_normal")
		screen_support[SCREEN_NORMAL]=p_value;
	else if (n=="screen/support_large")
		screen_support[SCREEN_LARGE]=p_value;
	else if (n=="screen/support_xlarge")
		screen_support[SCREEN_XLARGE]=p_value;
	else if (n=="keystore/release")
		release_keystore=p_value;
	else if (n=="keystore/release_user")
		release_username=p_value;
	else if (n=="keystore/release_password")
		release_password=p_value;
	else if (n=="apk_expansion/enable")
		apk_expansion=p_value;
	else if (n=="apk_expansion/SALT")
		apk_expansion_salt=p_value;
	else if (n=="apk_expansion/public_key")
		apk_expansion_pkey=p_value;
	else if (n.begins_with("permissions/")) {

		String what = n.get_slicec('/',1).to_upper();
		bool state = p_value;
		if (state)
			perms.insert(what);
		else
			perms.erase(what);
	} else if (n.begins_with("user_permissions/")) {

		int which = n.get_slicec('/',1).to_int();
		ERR_FAIL_INDEX_V(which,MAX_USER_PERMISSIONS,false);
		user_perms[which]=p_value;

	} else
		return false;

	return true;
}

bool EditorExportPlatformAndroid::_get(const StringName& p_name,Variant &r_ret) const{

	String n=p_name;
	if (n=="one_click_deploy/clear_previous_install")
		r_ret=remove_prev;
	else if (n=="custom_package/debug")
		r_ret=custom_debug_package;
	else if (n=="custom_package/release")
		r_ret=custom_release_package;
	else if (n=="version/code")
		r_ret=version_code;
	else if (n=="version/name")
		r_ret=version_name;
	else if (n=="command_line/extra_args")
		r_ret=cmdline;
	else if (n=="package/unique_name")
		r_ret=package;
	else if (n=="package/name")
		r_ret=name;
	else if (n=="package/icon")
		r_ret=icon;
	else if (n=="package/signed")
		r_ret=_signed;
	else if (n=="architecture/arm")
		r_ret=export_arm;
	else if (n=="architecture/x86")
		r_ret=export_x86;
	else if (n=="screen/use_32_bits_view")
		r_ret=use_32_fb;
	else if (n=="screen/immersive_mode")
		r_ret=immersive;
	else if (n=="screen/orientation")
		r_ret=orientation;
	else if (n=="screen/support_small")
		r_ret=screen_support[SCREEN_SMALL];
	else if (n=="screen/support_normal")
		r_ret=screen_support[SCREEN_NORMAL];
	else if (n=="screen/support_large")
		r_ret=screen_support[SCREEN_LARGE];
	else if (n=="screen/support_xlarge")
		r_ret=screen_support[SCREEN_XLARGE];
	else if (n=="keystore/release")
		r_ret=release_keystore;
	else if (n=="keystore/release_user")
		r_ret=release_username;
	else if (n=="keystore/release_password")
		r_ret=release_password;
	else if (n=="apk_expansion/enable")
		r_ret=apk_expansion;
	else if (n=="apk_expansion/SALT")
		r_ret=apk_expansion_salt;
	else if (n=="apk_expansion/public_key")
		r_ret=apk_expansion_pkey;
	else if (n.begins_with("permissions/")) {

		String what = n.get_slicec('/',1).to_upper();
		r_ret = perms.has(what);
	} else if (n.begins_with("user_permissions/")) {

		int which = n.get_slicec('/',1).to_int();
		ERR_FAIL_INDEX_V(which,MAX_USER_PERMISSIONS,false);
		r_ret=user_perms[which];
	} else
		return false;

	return true;
}

void EditorExportPlatformAndroid::_get_property_list( List<PropertyInfo> *p_list) const{

	p_list->push_back( PropertyInfo( Variant::BOOL, "one_click_deploy/clear_previous_install"));
	p_list->push_back( PropertyInfo( Variant::STRING, "custom_package/debug", PROPERTY_HINT_GLOBAL_FILE,"apk"));
	p_list->push_back( PropertyInfo( Variant::STRING, "custom_package/release", PROPERTY_HINT_GLOBAL_FILE,"apk"));
	p_list->push_back( PropertyInfo( Variant::STRING, "command_line/extra_args"));
	p_list->push_back( PropertyInfo( Variant::INT, "version/code", PROPERTY_HINT_RANGE,"1,65535,1"));
	p_list->push_back( PropertyInfo( Variant::STRING, "version/name") );
	p_list->push_back( PropertyInfo( Variant::STRING, "package/unique_name") );
	p_list->push_back( PropertyInfo( Variant::STRING, "package/name") );
	p_list->push_back( PropertyInfo( Variant::STRING, "package/icon",PROPERTY_HINT_FILE,"png") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "package/signed") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "architecture/arm") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "architecture/x86") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "screen/use_32_bits_view") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "screen/immersive_mode") );
	p_list->push_back( PropertyInfo( Variant::INT, "screen/orientation",PROPERTY_HINT_ENUM,"Landscape,Portrait") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "screen/support_small") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "screen/support_normal") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "screen/support_large") );
	p_list->push_back( PropertyInfo( Variant::BOOL, "screen/support_xlarge") );
	p_list->push_back( PropertyInfo( Variant::STRING, "keystore/release",PROPERTY_HINT_GLOBAL_FILE,"keystore") );
	p_list->push_back( PropertyInfo( Variant::STRING, "keystore/release_user" ) );
	p_list->push_back( PropertyInfo( Variant::STRING, "keystore/release_password" ) );
	p_list->push_back( PropertyInfo( Variant::BOOL, "apk_expansion/enable" ) );
	p_list->push_back( PropertyInfo( Variant::STRING, "apk_expansion/SALT" ) );
	p_list->push_back( PropertyInfo( Variant::STRING, "apk_expansion/public_key",PROPERTY_HINT_MULTILINE_TEXT ) );

	const char **perms = android_perms;
	while(*perms) {

		p_list->push_back( PropertyInfo( Variant::BOOL, "permissions/"+String(*perms).to_lower()));
		perms++;
	}

	for(int i=0;i<MAX_USER_PERMISSIONS;i++) {

		p_list->push_back( PropertyInfo( Variant::STRING, "user_permissions/"+itos(i)));
	}

	//p_list->push_back( PropertyInfo( Variant::INT, "resources/pack_mode", PROPERTY_HINT_ENUM,"Copy,Single Exec.,Pack (.pck),Bundles (Optical)"));

}


static String _parse_string(const uint8_t *p_bytes,bool p_utf8) {

	uint32_t offset=0;
	uint32_t len = decode_uint16(&p_bytes[offset]);

	if (p_utf8) {
		//don't know how to read extended utf8, this will have to be for now
		len>>=8;

	}
	offset+=2;
	//printf("len %i, unicode: %i\n",len,int(p_utf8));

	if (p_utf8) {

		Vector<uint8_t> str8;
		str8.resize(len+1);
		for(uint32_t i=0;i<len;i++) {
			str8[i]=p_bytes[offset+i];
		}
		str8[len]=0;
		String str;
		str.parse_utf8((const char*)str8.ptr());
		return str;
	} else {

		String str;
		for(uint32_t i=0;i<len;i++) {
			CharType c = decode_uint16(&p_bytes[offset+i*2]);
			if (c==0)
				break;
			str += String::chr(c);
		}
		return str;
	}

}

void EditorExportPlatformAndroid::_fix_resources(Vector<uint8_t>& p_manifest) {


	const int UTF8_FLAG = 0x00000100;
	print_line("*******************GORRRGLE***********************");

	uint32_t header = decode_uint32(&p_manifest[0]);
	uint32_t filesize = decode_uint32(&p_manifest[4]);
	uint32_t string_block_len = decode_uint32(&p_manifest[16]);
	uint32_t string_count = decode_uint32(&p_manifest[20]);
	uint32_t string_flags = decode_uint32(&p_manifest[28]);
	const uint32_t string_table_begins = 40;

	Vector<String> string_table;

	//printf("stirng block len: %i\n",string_block_len);
	//printf("stirng count: %i\n",string_count);
	//printf("flags: %x\n",string_flags);

	for(uint32_t i=0;i<string_count;i++) {

		uint32_t offset = decode_uint32(&p_manifest[string_table_begins+i*4]);
		offset+=string_table_begins+string_count*4;

		String str = _parse_string(&p_manifest[offset],string_flags&UTF8_FLAG);

		if (str.begins_with("godot-project-name")) {


			if (str=="godot-project-name") {
				//project name
				str = get_project_name();

			} else {

				String lang = str.substr(str.find_last("-")+1,str.length()).replace("-","_");
				String prop = "application/name_"+lang;
				if (GlobalConfig::get_singleton()->has(prop)) {
					str = GlobalConfig::get_singleton()->get(prop);
				} else {
					str = get_project_name();
				}
			}
		}

		string_table.push_back(str);

	}

	//write a new string table, but use 16 bits
	Vector<uint8_t> ret;
	ret.resize(string_table_begins+string_table.size()*4);

	for(uint32_t i=0;i<string_table_begins;i++) {

		ret[i]=p_manifest[i];
	}

	int ofs=0;
	for(int i=0;i<string_table.size();i++) {

		encode_uint32(ofs,&ret[string_table_begins+i*4]);
		ofs+=string_table[i].length()*2+2+2;
	}

	ret.resize(ret.size()+ofs);
	uint8_t *chars=&ret[ret.size()-ofs];
	for(int i=0;i<string_table.size();i++) {

		String s = string_table[i];
		encode_uint16(s.length(),chars);
		chars+=2;
		for(int j=0;j<s.length();j++) {
			encode_uint16(s[j],chars);
			chars+=2;
		}
		encode_uint16(0,chars);
		chars+=2;
	}

	//pad
	while(ret.size()%4)
		ret.push_back(0);

	//change flags to not use utf8
	encode_uint32(string_flags&~0x100,&ret[28]);
	//change length
	encode_uint32(ret.size()-12,&ret[16]);
	//append the rest...
	int rest_from = 12+string_block_len;
	int rest_to = ret.size();
	int rest_len = (p_manifest.size() - rest_from);
	ret.resize(ret.size() + (p_manifest.size() - rest_from) );
	for(int i=0;i<rest_len;i++) {
		ret[rest_to+i]=p_manifest[rest_from+i];
	}
	//finally update the size
	encode_uint32(ret.size(),&ret[4]);


	p_manifest=ret;
	//printf("end\n");
}

String EditorExportPlatformAndroid::get_project_name() const {

	String aname;
	if (this->name!="") {
		aname=this->name;
	} else {
		aname = GlobalConfig::get_singleton()->get("application/name");

	}

	if (aname=="") {
		aname=_MKSTR(VERSION_NAME);
	}

	return aname;
}


void EditorExportPlatformAndroid::_fix_manifest(Vector<uint8_t>& p_manifest,bool p_give_internet) {


	const int CHUNK_AXML_FILE = 0x00080003;
	const int CHUNK_RESOURCEIDS = 0x00080180;
	const int CHUNK_STRINGS = 0x001C0001;
	const int CHUNK_XML_END_NAMESPACE = 0x00100101;
	const int CHUNK_XML_END_TAG = 0x00100103;
	const int CHUNK_XML_START_NAMESPACE = 0x00100100;
	const int CHUNK_XML_START_TAG = 0x00100102;
	const int CHUNK_XML_TEXT = 0x00100104;
	const int UTF8_FLAG = 0x00000100;

	Vector<String> string_table;

	uint32_t ofs=0;


	uint32_t header = decode_uint32(&p_manifest[ofs]);
	uint32_t filesize = decode_uint32(&p_manifest[ofs+4]);
	ofs+=8;

	//print_line("FILESIZE: "+itos(filesize)+" ACTUAL: "+itos(p_manifest.size()));

	uint32_t string_count;
	uint32_t styles_count;
	uint32_t string_flags;
	uint32_t string_data_offset;

	uint32_t styles_offset;
	uint32_t string_table_begins;
	uint32_t string_table_ends;
	Vector<uint8_t> stable_extra;

	while(ofs < (uint32_t)p_manifest.size()) {

		uint32_t chunk = decode_uint32(&p_manifest[ofs]);
		uint32_t size = decode_uint32(&p_manifest[ofs+4]);


		switch(chunk) {

			case CHUNK_STRINGS: {


				int iofs=ofs+8;

				string_count=decode_uint32(&p_manifest[iofs]);
				styles_count=decode_uint32(&p_manifest[iofs+4]);
				string_flags=decode_uint32(&p_manifest[iofs+8]);
				string_data_offset=decode_uint32(&p_manifest[iofs+12]);
				styles_offset=decode_uint32(&p_manifest[iofs+16]);
/*
				printf("string count: %i\n",string_count);
				printf("flags: %i\n",string_flags);
				printf("sdata ofs: %i\n",string_data_offset);
				printf("styles ofs: %i\n",styles_offset);
*/
				uint32_t st_offset=iofs+20;
				string_table.resize(string_count);
				uint32_t string_end=0;

				string_table_begins=st_offset;


				for(uint32_t i=0;i<string_count;i++) {

					uint32_t string_at = decode_uint32(&p_manifest[st_offset+i*4]);
					string_at+=st_offset+string_count*4;

					ERR_EXPLAIN("Unimplemented, can't read utf8 string table.");
					ERR_FAIL_COND(string_flags&UTF8_FLAG);

					if (string_flags&UTF8_FLAG) {



					} else {
						uint32_t len = decode_uint16(&p_manifest[string_at]);
						Vector<CharType> ucstring;
						ucstring.resize(len+1);
						for(uint32_t j=0;j<len;j++) {
							uint16_t c=decode_uint16(&p_manifest[string_at+2+2*j]);
							ucstring[j]=c;
						}
						string_end=MAX(string_at+2+2*len,string_end);
						ucstring[len]=0;
						string_table[i]=ucstring.ptr();
					}


					//print_line("String "+itos(i)+": "+string_table[i]);
				}

				for(uint32_t i=string_end;i<(ofs+size);i++) {
					stable_extra.push_back(p_manifest[i]);
				}

				//printf("stable extra: %i\n",int(stable_extra.size()));
				string_table_ends=ofs+size;

				//print_line("STABLE SIZE: "+itos(size)+" ACTUAL: "+itos(string_table_ends));

			} break;
			case CHUNK_XML_START_TAG: {

				int iofs=ofs+8;
				uint32_t line=decode_uint32(&p_manifest[iofs]);
				uint32_t nspace=decode_uint32(&p_manifest[iofs+8]);
				uint32_t name=decode_uint32(&p_manifest[iofs+12]);
				uint32_t check=decode_uint32(&p_manifest[iofs+16]);

				String tname=string_table[name];

				//printf("NSPACE: %i\n",nspace);
				//printf("NAME: %i (%s)\n",name,tname.utf8().get_data());
				//printf("CHECK: %x\n",check);
				uint32_t attrcount=decode_uint32(&p_manifest[iofs+20]);
				iofs+=28;
				//printf("ATTRCOUNT: %x\n",attrcount);
				for(uint32_t i=0;i<attrcount;i++) {
					uint32_t attr_nspace=decode_uint32(&p_manifest[iofs]);
					uint32_t attr_name=decode_uint32(&p_manifest[iofs+4]);
					uint32_t attr_value=decode_uint32(&p_manifest[iofs+8]);
					uint32_t attr_flags=decode_uint32(&p_manifest[iofs+12]);
					uint32_t attr_resid=decode_uint32(&p_manifest[iofs+16]);


					String value;
					if (attr_value!=0xFFFFFFFF)
						value=string_table[attr_value];
					else
						value="Res #"+itos(attr_resid);
					String attrname = string_table[attr_name];
					String nspace;
					if (attr_nspace!=0xFFFFFFFF)
						nspace=string_table[attr_nspace];
					else
						nspace="";

					//printf("ATTR %i NSPACE: %i\n",i,attr_nspace);
					//printf("ATTR %i NAME: %i (%s)\n",i,attr_name,attrname.utf8().get_data());
					//printf("ATTR %i VALUE: %i (%s)\n",i,attr_value,value.utf8().get_data());
					//printf("ATTR %i FLAGS: %x\n",i,attr_flags);
					//printf("ATTR %i RESID: %x\n",i,attr_resid);

					//replace project information
					if (tname=="manifest" && attrname=="package") {

						print_line("FOUND package");
						string_table[attr_value]=get_package_name();
					}

					//print_line("tname: "+tname);
					//print_line("nspace: "+nspace);
					//print_line("attrname: "+attrname);
					if (tname=="manifest" && /*nspace=="android" &&*/ attrname=="versionCode") {

						print_line("FOUND versionCode");
						encode_uint32(version_code,&p_manifest[iofs+16]);
					}


					if (tname=="manifest" && /*nspace=="android" &&*/ attrname=="versionName") {

						print_line("FOUND versionName");
						if (attr_value==0xFFFFFFFF) {
							WARN_PRINT("Version name in a resource, should be plaintext")
						} else
							string_table[attr_value]=version_name;
					}

					if (tname=="activity" && /*nspace=="android" &&*/ attrname=="screenOrientation") {

						encode_uint32(orientation==0?0:1,&p_manifest[iofs+16]);
						/*
						print_line("FOUND screen orientation");
						if (attr_value==0xFFFFFFFF) {
							WARN_PRINT("Version name in a resource, should be plaintext")
						} else {
							string_table[attr_value]=(orientation==0?"landscape":"portrait");
						}*/
					}

					if (tname=="uses-permission" && /*nspace=="android" &&*/ attrname=="name") {

						if (value.begins_with("godot.custom")) {

							int which = value.get_slice(".",2).to_int();
							if (which>=0 && which<MAX_USER_PERMISSIONS && user_perms[which].strip_edges()!="") {

								string_table[attr_value]=user_perms[which].strip_edges();
							}

						} else if (value.begins_with("godot.")) {
							String perm = value.get_slice(".",1);

							if (perms.has(perm) || (p_give_internet && perm=="INTERNET")) {

								print_line("PERM: "+perm);
								string_table[attr_value]="android.permission."+perm;
							}

						}
					}

					if (tname=="supports-screens" ) {

						if (attrname=="smallScreens") {

							encode_uint32(screen_support[SCREEN_SMALL]?0xFFFFFFFF:0,&p_manifest[iofs+16]);

						} else if (attrname=="normalScreens") {

							encode_uint32(screen_support[SCREEN_NORMAL]?0xFFFFFFFF:0,&p_manifest[iofs+16]);

						} else if (attrname=="largeScreens") {

							encode_uint32(screen_support[SCREEN_LARGE]?0xFFFFFFFF:0,&p_manifest[iofs+16]);

						} else if (attrname=="xlargeScreens") {

							encode_uint32(screen_support[SCREEN_XLARGE]?0xFFFFFFFF:0,&p_manifest[iofs+16]);

						}
					}


					iofs+=20;
				}

			} break;
		}
		//printf("chunk %x: size: %d\n",chunk,size);

		ofs+=size;
	}

	//printf("end\n");

	//create new andriodmanifest binary

	Vector<uint8_t> ret;
	ret.resize(string_table_begins+string_table.size()*4);

	for(uint32_t i=0;i<string_table_begins;i++) {

		ret[i]=p_manifest[i];
	}

	ofs=0;
	for(int i=0;i<string_table.size();i++) {

		encode_uint32(ofs,&ret[string_table_begins+i*4]);
		ofs+=string_table[i].length()*2+2+2;
		//print_line("ofs: "+itos(i)+": "+itos(ofs));
	}
	ret.resize(ret.size()+ofs);
	uint8_t *chars=&ret[ret.size()-ofs];
	for(int i=0;i<string_table.size();i++) {

		String s = string_table[i];
		//print_line("savint string :"+s);
		encode_uint16(s.length(),chars);
		chars+=2;
		for(int j=0;j<s.length();j++) { //include zero?
			encode_uint16(s[j],chars);
			chars+=2;
		}
		encode_uint16(0,chars);
		chars+=2;

	}


	for(int i=0;i<stable_extra.size();i++) {
		ret.push_back(stable_extra[i]);
	}

	while(ret.size()%4)
		ret.push_back(0);


	uint32_t new_stable_end=ret.size();

	uint32_t extra = (p_manifest.size()-string_table_ends);
	ret.resize(new_stable_end + extra);
	for(uint32_t i=0;i<extra;i++)
		ret[new_stable_end+i]=p_manifest[string_table_ends+i];

	while(ret.size()%4)
		ret.push_back(0);
	encode_uint32(ret.size(),&ret[4]); //update new file size

	encode_uint32(new_stable_end-8,&ret[12]); //update new string table size

	//print_line("file size: "+itos(ret.size()));

	p_manifest=ret;






#if 0
	uint32_t header[9];
	for(int i=0;i<9;i++) {
		header[i]=decode_uint32(&p_manifest[i*4]);
	}

	//print_line("STO: "+itos(header[3]));
	uint32_t st_offset=9*4;
	//ERR_FAIL_COND(header[3]!=0x24)
	uint32_t string_count=header[4];


	string_table.resize(string_count);

	for(int i=0;i<string_count;i++) {

		uint32_t string_at = decode_uint32(&p_manifest[st_offset+i*4]);
		string_at+=st_offset+string_count*4;
		uint32_t len = decode_uint16(&p_manifest[string_at]);
		Vector<CharType> ucstring;
		ucstring.resize(len+1);
		for(int j=0;j<len;j++) {
			uint16_t c=decode_uint16(&p_manifest[string_at+2+2*j]);
			ucstring[j]=c;
		}
		ucstring[len]=0;
		string_table[i]=ucstring.ptr();
	}


#endif

}



Error EditorExportPlatformAndroid::save_apk_file(void *p_userdata,const String& p_path, const Vector<uint8_t>& p_data,int p_file,int p_total) {

	APKExportData *ed=(APKExportData*)p_userdata;
	String dst_path=p_path;
	dst_path=dst_path.replace_first("res://","assets/");

	zipOpenNewFileInZip(ed->apk,
		dst_path.utf8().get_data(),
		NULL,
		NULL,
		0,
		NULL,
		0,
		NULL,
		_should_compress_asset(p_path,p_data) ? Z_DEFLATED : 0,
		Z_DEFAULT_COMPRESSION);


	zipWriteInFileInZip(ed->apk,p_data.ptr(),p_data.size());
	zipCloseFileInZip(ed->apk);
	ed->ep->step("File: "+p_path,3+p_file*100/p_total);
	return OK;

}

bool EditorExportPlatformAndroid::_should_compress_asset(const String& p_path, const Vector<uint8_t>& p_data) {

	/*
	 *  By not compressing files with little or not benefit in doing so,
	 *  a performance gain is expected at runtime. Moreover, if the APK is
	 *  zip-aligned, assets stored as they are can be efficiently read by
	 *  Android by memory-mapping them.
	 */

	// -- Unconditional uncompress to mimic AAPT plus some other

	static const char* unconditional_compress_ext[] = {
		// From https://github.com/android/platform_frameworks_base/blob/master/tools/aapt/Package.cpp
		// These formats are already compressed, or don't compress well:
		".jpg", ".jpeg", ".png", ".gif",
		".wav", ".mp2", ".mp3", ".ogg", ".aac",
		".mpg", ".mpeg", ".mid", ".midi", ".smf", ".jet",
		".rtttl", ".imy", ".xmf", ".mp4", ".m4a",
		".m4v", ".3gp", ".3gpp", ".3g2", ".3gpp2",
		".amr", ".awb", ".wma", ".wmv",
		// Godot-specific:
		".webp", // Same reasoning as .png
		".cfb", // Don't let small config files slow-down startup
		// Trailer for easier processing
		NULL
	};

	for (const char** ext=unconditional_compress_ext; *ext; ++ext) {
		if (p_path.to_lower().ends_with(String(*ext))) {
			return false;
		}
	}

	// -- Compressed resource?

	if (p_data.size() >= 4 && p_data[0]=='R' && p_data[1]=='S' && p_data[2]=='C' && p_data[3]=='C') {
		// Already compressed
		return false;
	}

	// --- TODO: Decide on texture resources according to their image compression setting

	return true;
}



Error EditorExportPlatformAndroid::export_project(const String& p_path, bool p_debug, int p_flags) {

	String src_apk;

	EditorProgress ep("export","Exporting for Android",105);

	if (p_debug)
		src_apk=custom_debug_package;
	else
		src_apk=custom_release_package;

	if (src_apk=="") {
		String err;
		if (p_debug) {
			src_apk=find_export_template("android_debug.apk", &err);
		} else {
			src_apk=find_export_template("android_release.apk", &err);
		}
		if (src_apk=="") {
			EditorNode::add_io_error(err);
			return ERR_FILE_NOT_FOUND;
		}
	}

	FileAccess *src_f=NULL;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);



	ep.step("Creating APK",0);

	unzFile pkg = unzOpen2(src_apk.utf8().get_data(), &io);
	if (!pkg) {

		EditorNode::add_io_error("Could not find template APK to export:\n"+src_apk);
		return ERR_FILE_NOT_FOUND;
	}

	ERR_FAIL_COND_V(!pkg, ERR_CANT_OPEN);
	int ret = unzGoToFirstFile(pkg);

	zlib_filefunc_def io2=io;
	FileAccess *dst_f=NULL;
	io2.opaque=&dst_f;
	String unaligned_path=EditorSettings::get_singleton()->get_settings_path()+"/tmp/tmpexport-unaligned.apk";
	zipFile	unaligned_apk=zipOpen2(unaligned_path.utf8().get_data(),APPEND_STATUS_CREATE,NULL,&io2);


	while(ret==UNZ_OK) {

		//get filename
		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(pkg,&info,fname,16384,NULL,0,NULL,0);

		bool skip=false;

		String file=fname;

		Vector<uint8_t> data;
		data.resize(info.uncompressed_size);

		//read
		unzOpenCurrentFile(pkg);
		unzReadCurrentFile(pkg,data.ptr(),data.size());
		unzCloseCurrentFile(pkg);

		//write

		if (file=="AndroidManifest.xml") {

			_fix_manifest(data,p_flags&(EXPORT_DUMB_CLIENT|EXPORT_REMOTE_DEBUG));
		}

		if (file=="resources.arsc") {

			_fix_resources(data);
		}

		if (file=="res/drawable/icon.png") {
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

		if (file=="lib/x86/libgodot_android.so" && !export_x86) {
			skip=true;
		}

		if (file.match("lib/armeabi*/libgodot_android.so") && !export_arm) {
			skip=true;
		}

		if (file.begins_with("META-INF") && _signed) {
			skip=true;
		}

		print_line("ADDING: "+file);

		if (!skip) {

			// Respect decision on compression made by AAPT for the export template
			const bool uncompressed = info.compression_method == 0;

			zipOpenNewFileInZip(unaligned_apk,
				file.utf8().get_data(),
				NULL,
				NULL,
				0,
				NULL,
				0,
				NULL,
				uncompressed ? 0 : Z_DEFLATED,
				Z_DEFAULT_COMPRESSION);

			zipWriteInFileInZip(unaligned_apk,data.ptr(),data.size());
			zipCloseFileInZip(unaligned_apk);
		}

		ret = unzGoToNextFile(pkg);
	}


	ep.step("Adding Files..",1);
	Error err=OK;
	Vector<String> cl = cmdline.strip_edges().split(" ");
	for(int i=0;i<cl.size();i++) {
		if (cl[i].strip_edges().length()==0) {
			cl.remove(i);
			i--;
		}
	}

	gen_export_flags(cl,p_flags);

	if (p_flags&EXPORT_DUMB_CLIENT) {

		/*String host = EditorSettings::get_singleton()->get("filesystem/file_server/host");
		int port = EditorSettings::get_singleton()->get("filesystem/file_server/post");
		String passwd = EditorSettings::get_singleton()->get("filesystem/file_server/password");
		cl.push_back("-rfs");
		cl.push_back(host+":"+itos(port));
		if (passwd!="") {
			cl.push_back("-rfs_pass");
			cl.push_back(passwd);
		}*/


	} else {
		//all files

		if (apk_expansion) {

			String apkfname="main."+itos(version_code)+"."+get_package_name()+".obb";
			String fullpath=p_path.get_base_dir().plus_file(apkfname);
			FileAccess *pf = FileAccess::open(fullpath,FileAccess::WRITE);
			if (!pf) {
				EditorNode::add_io_error("Could not write expansion package file: "+apkfname);
				return OK;
			}
			err = save_pack(pf);
			memdelete(pf);

			cl.push_back("-use_apk_expansion");
			cl.push_back("-apk_expansion_md5");
			cl.push_back(FileAccess::get_md5(fullpath));
			cl.push_back("-apk_expansion_key");
			cl.push_back(apk_expansion_pkey.strip_edges());

		} else {

			APKExportData ed;
			ed.ep=&ep;
			ed.apk=unaligned_apk;

			err = export_project_files(save_apk_file,&ed,false);
		}


	}

	if (use_32_fb)
		cl.push_back("-use_depth_32");

	if (immersive)
		cl.push_back("-use_immersive");

	if (cl.size()) {
		//add comandline
		Vector<uint8_t> clf;
		clf.resize(4);
		encode_uint32(cl.size(),&clf[0]);
		for(int i=0;i<cl.size();i++) {

			CharString txt = cl[i].utf8();
			int base = clf.size();
			clf.resize(base+4+txt.length());
			encode_uint32(txt.length(),&clf[base]);
			copymem(&clf[base+4],txt.ptr(),txt.length());
			print_line(itos(i)+" param: "+cl[i]);
		}

		zipOpenNewFileInZip(unaligned_apk,
			"assets/_cl_",
			NULL,
			NULL,
			0,
			NULL,
			0,
			NULL,
			0, // No compress (little size gain and potentially slower startup)
			Z_DEFAULT_COMPRESSION);

		zipWriteInFileInZip(unaligned_apk,clf.ptr(),clf.size());
		zipCloseFileInZip(unaligned_apk);

	}

	zipClose(unaligned_apk,NULL);
	unzClose(pkg);

	if (err) {
		return err;
	}



	if (_signed) {


		String jarsigner=EditorSettings::get_singleton()->get("export/android/jarsigner");
		if (!FileAccess::exists(jarsigner)) {
			EditorNode::add_io_error("'jarsigner' could not be found.\nPlease supply a path in the editor settings.\nResulting apk is unsigned.");
			return OK;
		}

		String keystore;
		String password;
		String user;
		if (p_debug) {
			keystore=EditorSettings::get_singleton()->get("export/android/debug_keystore");
			password=EditorSettings::get_singleton()->get("export/android/debug_keystore_pass");
			user=EditorSettings::get_singleton()->get("export/android/debug_keystore_user");

			ep.step("Signing Debug APK..",103);

		} else {
			keystore=release_keystore;
			password=release_password;
			user=release_username;

			ep.step("Signing Release APK..",103);

		}

		if (!FileAccess::exists(keystore)) {
			EditorNode::add_io_error("Could not find keystore, unable to export.");
			return ERR_FILE_CANT_OPEN;
		}

		List<String> args;
		args.push_back("-digestalg");
		args.push_back("SHA1");
		args.push_back("-sigalg");
		args.push_back("MD5withRSA");
		String tsa_url=EditorSettings::get_singleton()->get("export/android/timestamping_authority_url");
		if (tsa_url != "") {
			args.push_back("-tsa");
			args.push_back(tsa_url);
		}
		args.push_back("-verbose");
		args.push_back("-keystore");
		args.push_back(keystore);
		args.push_back("-storepass");
		args.push_back(password);
		args.push_back(unaligned_path);
		args.push_back(user);
		int retval;
		OS::get_singleton()->execute(jarsigner,args,true,NULL,NULL,&retval);
		if (retval) {
			EditorNode::add_io_error("'jarsigner' returned with error #"+itos(retval));
			return ERR_CANT_CREATE;
		}

		ep.step("Verifying APK..",104);

		args.clear();
		args.push_back("-verify");
		args.push_back(unaligned_path);
		args.push_back("-verbose");

		OS::get_singleton()->execute(jarsigner,args,true,NULL,NULL,&retval);
		if (retval) {
			EditorNode::add_io_error("'jarsigner' verification of APK failed. Make sure to use jarsigner from Java 6.");
			return ERR_CANT_CREATE;
		}

	}



	// Let's zip-align (must be done after signing)

	static const int ZIP_ALIGNMENT = 4;

	ep.step("Aligning APK..",105);

	unzFile tmp_unaligned = unzOpen2(unaligned_path.utf8().get_data(), &io);
	if (!tmp_unaligned) {

		EditorNode::add_io_error("Could not find temp unaligned APK.");
		return ERR_FILE_NOT_FOUND;
	}

	ERR_FAIL_COND_V(!tmp_unaligned, ERR_CANT_OPEN);
	ret = unzGoToFirstFile(tmp_unaligned);

	io2=io;
	dst_f=NULL;
	io2.opaque=&dst_f;
	zipFile	final_apk=zipOpen2(p_path.utf8().get_data(),APPEND_STATUS_CREATE,NULL,&io2);

	// Take files from the unaligned APK and write them out to the aligned one
	// in raw mode, i.e. not uncompressing and recompressing, aligning them as needed,
	// following what is done in https://github.com/android/platform_build/blob/master/tools/zipalign/ZipAlign.cpp
	int bias = 0;
	while(ret==UNZ_OK) {

		unz_file_info info;
		memset(&info, 0, sizeof(info));

		char fname[16384];
		char extra[16384];
		ret = unzGetCurrentFileInfo(tmp_unaligned,&info,fname,16384,extra,16384-ZIP_ALIGNMENT,NULL,0);

		String file=fname;

		Vector<uint8_t> data;
		data.resize(info.compressed_size);

		// read
		int method, level;
		unzOpenCurrentFile2(tmp_unaligned, &method, &level, 1); // raw read
		long file_offset = unzGetCurrentFileZStreamPos64(tmp_unaligned);
		unzReadCurrentFile(tmp_unaligned,data.ptr(),data.size());
		unzCloseCurrentFile(tmp_unaligned);

		// align
		int padding = 0;
		if (!info.compression_method) {
			// Uncompressed file => Align
			long new_offset = file_offset + bias;
            padding = (ZIP_ALIGNMENT - (new_offset % ZIP_ALIGNMENT)) % ZIP_ALIGNMENT;
		}

		memset(extra + info.size_file_extra, 0, padding);

		// write
		zipOpenNewFileInZip2(final_apk,
			file.utf8().get_data(),
			NULL,
			extra,
			info.size_file_extra + padding,
			NULL,
			0,
			NULL,
			method,
			level,
			1); // raw write
		zipWriteInFileInZip(final_apk,data.ptr(),data.size());
		zipCloseFileInZipRaw(final_apk,info.uncompressed_size,info.crc);

		bias += padding;

		ret = unzGoToNextFile(tmp_unaligned);
	}

	zipClose(final_apk,NULL);
	unzClose(tmp_unaligned);

	if (err) {
		return err;
	}

	return OK;

}


bool EditorExportPlatformAndroid::poll_devices() {

	bool dc=devices_changed;
	devices_changed=false;
	return dc;
}

int EditorExportPlatformAndroid::get_device_count() const {

	device_lock->lock();
	int dc=devices.size();
	device_lock->unlock();

	return dc;

}

String EditorExportPlatformAndroid::get_device_name(int p_device) const {

	ERR_FAIL_INDEX_V(p_device,devices.size(),"");
	device_lock->lock();
	String s=devices[p_device].name;
	device_lock->unlock();
	return s;
}

String EditorExportPlatformAndroid::get_device_info(int p_device) const {

	ERR_FAIL_INDEX_V(p_device,devices.size(),"");
	device_lock->lock();
	String s=devices[p_device].description;
	device_lock->unlock();
	return s;
}

void EditorExportPlatformAndroid::_device_poll_thread(void *ud) {

	EditorExportPlatformAndroid *ea=(EditorExportPlatformAndroid *)ud;


	while(!ea->quit_request) {

		String adb=EditorSettings::get_singleton()->get("export/android/adb");
		if (FileAccess::exists(adb)) {

			String devices;
			List<String> args;
			args.push_back("devices");
			int ec;
			OS::get_singleton()->execute(adb,args,true,NULL,&devices,&ec);
			Vector<String> ds = devices.split("\n");
			Vector<String> ldevices;
			for(int i=1;i<ds.size();i++) {

				String d = ds[i];
				int dpos = d.find("device");
				if (dpos==-1)
					continue;
				d=d.substr(0,dpos).strip_edges();
				//print_line("found devuce: "+d);
				ldevices.push_back(d);
			}

			ea->device_lock->lock();

			bool different=false;

			if (devices.size()!=ldevices.size()) {

				different=true;
			} else {

				for(int i=0;i<ea->devices.size();i++) {

					if (ea->devices[i].id!=ldevices[i]) {
						different=true;
						break;
					}
				}
			}

			if (different) {


				Vector<Device> ndevices;

				for(int i=0;i<ldevices.size();i++) {

					Device d;
					d.id=ldevices[i];
					for(int j=0;j<ea->devices.size();j++) {
						if (ea->devices[j].id==ldevices[i]) {
							d.description=ea->devices[j].description;
							d.name=ea->devices[j].name;
						}
					}

					if (d.description=="") {
						//in the oven, request!
						args.clear();
						args.push_back("-s");
						args.push_back(d.id);
						args.push_back("shell");
						args.push_back("cat");
						args.push_back("/system/build.prop");
						int ec;
						String dp;

						OS::get_singleton()->execute(adb,args,true,NULL,&dp,&ec);

						Vector<String> props = dp.split("\n");
						String vendor;
						String device;
						d.description+"Device ID: "+d.id+"\n";
						for(int j=0;j<props.size();j++) {

							String p = props[j];
							if (p.begins_with("ro.product.model=")) {
								device=p.get_slice("=",1).strip_edges();
							} else if (p.begins_with("ro.product.brand=")) {
								vendor=p.get_slice("=",1).strip_edges().capitalize();
							} else if (p.begins_with("ro.build.display.id=")) {
								d.description+="Build: "+p.get_slice("=",1).strip_edges()+"\n";
							} else if (p.begins_with("ro.build.version.release=")) {
								d.description+="Release: "+p.get_slice("=",1).strip_edges()+"\n";
							} else if (p.begins_with("ro.product.cpu.abi=")) {
								d.description+="CPU: "+p.get_slice("=",1).strip_edges()+"\n";
							} else if (p.begins_with("ro.product.manufacturer=")) {
								d.description+="Manufacturer: "+p.get_slice("=",1).strip_edges()+"\n";
							} else if (p.begins_with("ro.board.platform=")) {
								d.description+="Chipset: "+p.get_slice("=",1).strip_edges()+"\n";
							} else if (p.begins_with("ro.opengles.version=")) {
								uint32_t opengl = p.get_slice("=",1).to_int();
								d.description+="OpenGL: "+itos(opengl>>16)+"."+itos((opengl>>8)&0xFF)+"."+itos((opengl)&0xFF)+"\n";
							}
						}

						d.name=vendor+" "+device;
						//print_line("name: "+d.name);
						//print_line("description: "+d.description);

					}

					ndevices.push_back(d);

				}

				ea->devices=ndevices;
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

	if (EditorSettings::get_singleton()->get("export/android/shutdown_adb_on_exit")) {
		String adb=EditorSettings::get_singleton()->get("export/android/adb");
		if (!FileAccess::exists(adb)) {
			return; //adb not configured
		}

		List<String> args;
		args.push_back("kill-server");
		OS::get_singleton()->execute(adb,args,true);
	};
}

Error EditorExportPlatformAndroid::run(int p_device, int p_flags) {

	ERR_FAIL_INDEX_V(p_device,devices.size(),ERR_INVALID_PARAMETER);
	device_lock->lock();

	EditorProgress ep("run","Running on "+devices[p_device].name,3);

	String adb=EditorSettings::get_singleton()->get("export/android/adb");
	if (adb=="") {

		EditorNode::add_io_error("ADB executable not configured in settings, can't run.");
		device_lock->unlock();
		return ERR_UNCONFIGURED;
	}

	//export_temp
	ep.step("Exporting APK",0);


	bool use_adb_over_usb = bool(EDITOR_DEF("export/android/use_remote_debug_over_adb",true));

	if (use_adb_over_usb) {
		p_flags|=EXPORT_REMOTE_DEBUG_LOCALHOST;
	}

	String export_to=EditorSettings::get_singleton()->get_settings_path()+"/tmp/tmpexport.apk";
	Error err = export_project(export_to,true,p_flags);
	if (err) {
		device_lock->unlock();
		return err;
	}

	List<String> args;
	int rv;

	if (remove_prev) {
		ep.step("Uninstalling..",1);

		print_line("Uninstalling previous version: "+devices[p_device].name);

		args.push_back("-s");
		args.push_back(devices[p_device].id);
		args.push_back("uninstall");
		args.push_back(get_package_name());

		err = OS::get_singleton()->execute(adb,args,true,NULL,NULL,&rv);
#if 0
	if (err || rv!=0) {
		EditorNode::add_io_error("Could not install to device.");
		device_lock->unlock();
		return ERR_CANT_CREATE;
	}
#endif
	}

	print_line("Installing into device (please wait..): "+devices[p_device].name);
	ep.step("Installing to Device (please wait..)..",2);

	args.clear();
	args.push_back("-s");
	args.push_back(devices[p_device].id);
	args.push_back("install");
	args.push_back("-r");
	args.push_back(export_to);

	err = OS::get_singleton()->execute(adb,args,true,NULL,NULL,&rv);
	if (err || rv!=0) {
		EditorNode::add_io_error("Could not install to device.");
		device_lock->unlock();
		return ERR_CANT_CREATE;
	}

	if (use_adb_over_usb) {

		args.clear();
		args.push_back("reverse");
		args.push_back("--remove-all");
		err = OS::get_singleton()->execute(adb,args,true,NULL,NULL,&rv);

		int port = GlobalConfig::get_singleton()->get("network/debug/remote_port");
		args.clear();
		args.push_back("reverse");
		args.push_back("tcp:"+itos(port));
		args.push_back("tcp:"+itos(port));

		err = OS::get_singleton()->execute(adb,args,true,NULL,NULL,&rv);
		print_line("Reverse result: "+itos(rv));

		int fs_port = EditorSettings::get_singleton()->get("filesystem/file_server/port");

		args.clear();
		args.push_back("reverse");
		args.push_back("tcp:"+itos(fs_port));
		args.push_back("tcp:"+itos(fs_port));

		err = OS::get_singleton()->execute(adb,args,true,NULL,NULL,&rv);
		print_line("Reverse result2: "+itos(rv));

	}


	ep.step("Running on Device..",3);
	args.clear();
	args.push_back("-s");
	args.push_back(devices[p_device].id);
	args.push_back("shell");
	args.push_back("am");
	args.push_back("start");
	args.push_back("--user 0");
	args.push_back("-a");
	args.push_back("android.intent.action.MAIN");
	args.push_back("-n");
	args.push_back(get_package_name()+"/org.godotengine.godot.Godot");

	err = OS::get_singleton()->execute(adb,args,true,NULL,NULL,&rv);
	if (err || rv!=0) {
		EditorNode::add_io_error("Could not execute ondevice.");
		device_lock->unlock();
		return ERR_CANT_CREATE;
	}
	device_lock->unlock();
	return OK;
}

String EditorExportPlatformAndroid::get_package_name() {

	String pname = package;
	String basename = GlobalConfig::get_singleton()->get("application/name");
	basename=basename.to_lower();

	String name;
	bool first=true;
	for(int i=0;i<basename.length();i++) {
		CharType c = basename[i];
		if (c>='0' && c<='9' && first) {
			continue;
		}
		if ((c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9')) {
			name+=String::chr(c);
			first=false;
		}
	}
	if (name=="")
		name="noname";

	pname=pname.replace("$genname",name);
	return pname;

}

EditorExportPlatformAndroid::EditorExportPlatformAndroid() {

	version_code=1;
	version_name="1.0";
	package="org.godotengine.$genname";
	name="";
	_signed=true;
	apk_expansion=false;
	device_lock = Mutex::create();
	quit_request=false;
	orientation=0;
	remove_prev=true;
	use_32_fb=true;
	immersive=true;

	export_arm=true;
	export_x86=false;


	device_thread=Thread::create(_device_poll_thread,this);
	devices_changed=true;

	Image img( _android_logo );
	logo = Ref<ImageTexture>( memnew( ImageTexture ));
	logo->create_from_image(img);

	for(int i=0;i<4;i++)
		screen_support[i]=true;
}

bool EditorExportPlatformAndroid::can_export(String *r_error) const {

	bool valid=true;
	String adb=EditorSettings::get_singleton()->get("export/android/adb");
	String err;

	if (!FileAccess::exists(adb)) {

		valid=false;
		err+="ADB executable not configured in editor settings.\n";
	}

	String js = EditorSettings::get_singleton()->get("export/android/jarsigner");

	if (!FileAccess::exists(js)) {

		valid=false;
		err+="OpenJDK 6 jarsigner not configured in editor settings.\n";
	}

	String dk = EditorSettings::get_singleton()->get("export/android/debug_keystore");

	if (!FileAccess::exists(dk)) {

		valid=false;
		err+="Debug Keystore not configured in editor settings.\n";
	}

	if (!exists_export_template("android_debug.apk") || !exists_export_template("android_release.apk")) {
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

	if (apk_expansion) {

		/*
		if (apk_expansion_salt=="") {
			valid=false;
			err+="Invalid SALT for apk expansion.\n";
		}
		*/
		if (apk_expansion_pkey=="") {
			valid=false;
			err+="Invalid public key for apk expansion.\n";
		}
	}

	if (r_error)
		*r_error=err;

	return valid;
}


EditorExportPlatformAndroid::~EditorExportPlatformAndroid() {

	quit_request=true;
	Thread::wait_to_finish(device_thread);
	memdelete(device_lock);
	memdelete(device_thread);
}

#endif

void register_android_exporter() {

#if 0
	String exe_ext=OS::get_singleton()->get_name()=="Windows"?"exe":"";
	EDITOR_DEF("export/android/adb","");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING,"android/adb",PROPERTY_HINT_GLOBAL_FILE,exe_ext));
	EDITOR_DEF("export/android/jarsigner","");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING,"android/jarsigner",PROPERTY_HINT_GLOBAL_FILE,exe_ext));
	EDITOR_DEF("export/android/debug_keystore","");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING,"android/debug_keystore",PROPERTY_HINT_GLOBAL_FILE,"keystore"));
	EDITOR_DEF("export/android/debug_keystore_user","androiddebugkey");
	EDITOR_DEF("export/android/debug_keystore_pass","android");
	//EDITOR_DEF("android/release_keystore","");
	//EDITOR_DEF("android/release_username","");
	//EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING,"android/release_keystore",PROPERTY_HINT_GLOBAL_FILE,"*.keystore"));
	EDITOR_DEF("export/android/timestamping_authority_url","");
	EDITOR_DEF("export/android/use_remote_debug_over_adb",false);
	EDITOR_DEF("export/android/shutdown_adb_on_exit",true);

	Ref<EditorExportPlatformAndroid> exporter = Ref<EditorExportPlatformAndroid>( memnew(EditorExportPlatformAndroid) );
	EditorImportExport::get_singleton()->add_export_platform(exporter);
#endif
}

