/*************************************************************************/
/*  export.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor/editor_export.h"
#include "platform/sailfish/logo.gen.h"
#include "scene/resources/texture.h"
#include "core/io/marshalls.h"
#include "core/io/zip_io.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "core/dictionary.h"
#include "core/project_settings.h"
#include "core/version.h"
#include "editor/editor_export.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "modules/regex/regex.h"

#define prop_sailfish_sdk_path "sailfish_sdk/sdk_path"
#define prop_custom_binary_arm       "custom_binary/arm"
#define prop_custom_binary_arm_debug "custom_binary/arm_debug"
#define prop_custom_binary_x86       "custom_binary/x86"
#define prop_custom_binary_x86_debug "custom_binary/x86_debug"
#define prop_version_release "version/release"
#define prop_version_string  "version/string"

class EditorExportPlatformSailfish : public EditorExportPlatform {
	GDCLASS(EditorExportPlatformSailfish, EditorExportPlatform)

	enum TargetArch {
		arch_armv7hl,
		arch_i486,
		arch_x86 = arch_i486,
		arch_unkown
	};

	struct Device {
		String     address;
		String     name;
		TargetArch arch;
	};

	struct MerTarget {
		String     name;
		Array      version; // array of 4 integers
		TargetArch arch;
	};
public:
	EditorExportPlatformSailfish() {
		// Ref<Image> img = memnew(Image(_sailfish_logo));
		// logo.instance();
		// logo->create_from_image(img);
	}

	void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) override {
		// EditorNode::print("Could not unzip temporary unaligned APK.");
		print_verbose("get_preset_features, path " + p_preset->get_export_path() );
	}

	virtual void get_platform_features(List<String> *r_features) override {
		r_features->push_back("mobile");
		r_features->push_back(get_os_name());
	}

	String get_os_name() const override {
		return "SailfishOS";
	}

	String get_name() const override {	
		return "SailfishOS";
	}

	void set_logo(Ref<Texture> logo) {
		this->logo = logo;
	}

	Ref<Texture> get_logo() const override {
		return logo;
	}

	void get_export_options(List<ExportOption> *r_options) override {
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, prop_sailfish_sdk_path, PROPERTY_HINT_GLOBAL_DIR), ""));
		// r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "sailfish_sdk/arm_target", PROPERTY_HINT_ENUM), ""));
		// r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "sailfish_sdk/x86_target", PROPERTY_HINT_ENUM), ""));

		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, prop_custom_binary_arm, PROPERTY_HINT_GLOBAL_DIR), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, prop_custom_binary_arm_debug, PROPERTY_HINT_GLOBAL_DIR), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, prop_custom_binary_x86, PROPERTY_HINT_GLOBAL_DIR), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, prop_custom_binary_x86_debug, PROPERTY_HINT_GLOBAL_DIR), ""));

		r_options->push_back(ExportOption(PropertyInfo(Variant::INT,    prop_version_release, PROPERTY_HINT_RANGE, "1,40096,1,or_greater"), 1));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, prop_version_string, PROPERTY_HINT_PLACEHOLDER_TEXT, "1.0.0"), "1.0.0"));

		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/name", PROPERTY_HINT_PLACEHOLDER_TEXT, "harbour-$genname"), "harbour-$genname"));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/game_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Game Name [default if blank]"), ""));
	}

	bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const override {
		return true;
	}
	List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const override {
		List<String> ext;
		return ext;
	}
	Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0) override {
		ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);
		String src_binary;
		String sdk_path;
		String sdk_configs_path = OS::get_singleton()->get_config_path();
		String mer_sdk_tools;
		List<MerTarget> mer_target; // Mer targets list
		String arm_template;
		String x86_template;

		if(p_debug) 
			arm_template = String( p_preset->get(prop_custom_binary_arm_debug) );
		else
			arm_template = String( p_preset->get(prop_custom_binary_arm) );
		
		if(p_debug) 
			x86_template = String( p_preset->get(prop_custom_binary_x86_debug) );
		else
			x86_template = String( p_preset->get(prop_custom_binary_x86) );

#ifdef WINDOWS_ENABLED
		sdk_configs_path +=  String("\\SailfishOS-SDK\\");
		mer_sdk_tools = sdk_configs_path + String("\\mer-sdk-tools\\Sailfish OS Build Engine\\");
#else		
		sdk_configs_path +=  String("/SailfishOS-SDK/");
		mer_sdk_tools = sdk_configs_path + String("/mer-sdk-tools/Sailfish OS Build Engine/");
#endif
		Error err;
		DirAccessRef dir = DirAccess::open(mer_sdk_tools,&err);
		if( err != Error::OK )
		{
			print_error("Cant open MerSDK targets tools dir: " + mer_sdk_tools);
			return err;
		}
		// List existing in mersdk_tools tagets folders
		print_verbose("Mer SDK targets:");
		dir->list_dir_begin();
		String entry = dir->get_next();
		while(  entry != String() ) {
			if(!dir->current_is_dir() || entry == String(".") || entry == String(".."))
			{
				entry = dir->get_next();
				continue;
			}
			print_verbose( entry );
			MerTarget target;
			target.name = entry;
			//possible SailfishOS-3.2.0.12-armv7hl/
			//possible SailfishOS-3.2.0.12-i486/
			RegEx regex("SailfishOS-([0-9]+)\\.([0-9]+)\\.([0-9]+)\\.([0-9]+)-(armv7hl|i486)");
			Array matches = regex.search_all(entry);
			// print_verbose( String("Matches size: ") + Variant(matches.size()) );
			if( matches.size() == 1 ) {
				Ref<RegExMatch> rem = ((Ref<RegExMatch>)matches[0]);
				Array names = rem->get_strings();
				// print_verbose( String("match[0] strings size: ") + Variant(names.size()) );
				if( names.size() < 6 ) {
					print_verbose("Wrong match");
					for( int d = 0; d < names.size(); d++ )
					{
						print_verbose( String("match[0].strings[") + Variant(d) + String("]: ") + String(names[d]) );
					}
					target.arch = arch_unkown;
				}
				else {
					Array version_array;
					version_array.push_back( int(names[1]) );
					version_array.push_back( int(names[2]) );
					version_array.push_back( int(names[3]) );
					version_array.push_back( int(names[4]) );
					target.version = version_array;
					print_verbose( String(" Version is {0}.{1}.{2}.{3}").format(version_array) );
					print_verbose( String(" Arch is ") + names[5] );

					if( names[5] == String("armv7hl") ) {
						target.arch = arch_armv7hl;
					}
					else if( names[5] == String("i486") ) {
						target.arch = arch_i486;
					}
					else
						target.arch = arch_unkown;
				}
			}
			else {
				for(int i = 0; i < matches.size(); i++ ) {
					Ref<RegExMatch> rem = ((Ref<RegExMatch>)matches[i]);
					Array names = rem->get_strings();
					for( int d = 0; d < names.size(); d++ )
					{
						print_verbose( String("match[") + Variant(i) + String("].strings[") + Variant(d) + String("]: ") + String(names[d]) );
					}
				}
			}
			// verify targets, and its versions (SDL-2.0.9 from Sailfish 3.1.0)
			if( int(target.version[0]) >= 3 && int(target.version[1]) >= 1 ) {
				if( target.arch == arch_armv7hl && arm_template == String() )
					print_line( String("No arm tempalte for ")  + target.name );
				else if( target.arch == arch_x86 && x86_template == String() )
					print_line( String("No x86 tempalte for ")  + target.name );
				else 
					mer_target.push_back(target);
			}
			else {
				print_error( String("Too old Mer target ")  + target.name );
			}
			entry = dir->get_next();
		}
		dir->list_dir_end();

		EditorProgress ep("export", "Exporting for SailfishOS", 105, true);
		List<PropertyInfo> props = p_preset->get_properties();
		// for(int i = 0; i < props.size(); i++ ) {
		// 	PropertyInfo current = props[i];
		// 	print_verbose(String("Property is ") + current.name );

		// 	if( current.name == prop_sailfish_sdk_path ) {
		// 		// sdk_path = current.hint;
		// 		print_verbose("SDK path property: ");
		// 		print_verbose("hint_string: " + current.hint_string);
		// 		print_verbose("hint       : " + current.hint);
		// 		print_verbose("type       : " + current.type);
		// 		// OS::get_singleton()->print("SDK path is %s\n", sdk_path..c_str() );
		// 		// print_verbose(String("SDK path is ") + sdk_path);
		// 	}
		// }
		sdk_path = String(p_preset->get( prop_sailfish_sdk_path));
		print_verbose(String("SDK path is ") + sdk_path);
		print_verbose(String("Platfrom config path: ") + sdk_configs_path );
		
		
		// for( int target_num = 0; mer_target.size(); terget_num++ )
		// {
			
		// }

		return Error::OK;
	}

	void resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, Set<String> &p_features) override {

	}
protected:
	Ref<ImageTexture> logo;
};

void register_sailfish_exporter() {

	Ref<EditorExportPlatformSailfish> platform;
	Ref<EditorExportPlatformPC> p;
	platform.instance();

	Ref<Image> img = memnew(Image(_sailfish_logo));
	Ref<ImageTexture> logo;
	logo.instance();
	logo->create_from_image(img);
	platform->set_logo(logo);
	// platform->set_name("SailfishOS/SDL");
	// p->set_extension("arm", "binary_format/arm");
	// platform->set_extension("x86", "binary_format/i486");
	// platform->set_release_32("godot.sailfish.opt.arm");
	// platform->set_debug_32("godot.sailfish.opt.debug.arm");
	// platform->set_release_64("godot.sailfish.opt.x86");
	// platform->set_debug_64("godot.sailfish.opt.debug.x86");
	// platform->set_os_name("SailfishOS");
	// platform->set_chmod_flags(0755);

	EDITOR_DEF("export/sailfish/sdk_path", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/sailfish/sdk_path", PROPERTY_HINT_GLOBAL_DIR));

	EditorExport::get_singleton()->add_export_platform(platform);
}
