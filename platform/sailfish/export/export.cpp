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
#include "core/io/xml_parser.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/thread.h"
#include "core/os/os.h"
#include "core/dictionary.h"
#include "core/project_settings.h"
#include "core/version.h"
#include "main/main.h"
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
#define prop_name          "package/name"
#define prop_launcher_name "package/game_name"

#ifdef WINDOWS_ENABLED
const String separator("\\");
#else
const String separator("/");
#endif

const String spec_file_tempalte =
"Name:       %{_gd_application_name}\n"
"Summary:    %{_gd_launcher_name}\n"
"Version:    %{_gd_version}\n"
"Release:    %{_gd_release}\n"
"Group:      Game\n"
"License:    LICENSE\n"
"BuildArch:  %{_gd_architecture}\n"
"URL:        http://example.org/\n"
"Requires:   SDL2 >= 2.0.9\n"
"Requires:   freetype\n"
"Requires:   libpng\n"
"Requires:   openssl\n"
"Requires:   zlib\n"
"Requires:   glib2\n"
"Requires:   libaudioresource\n"
"#Requires:   libkeepalive-glib\n"
"%description\n"
"%{_gd_description}\n"
"%prep\n"
"echo \"Nothing to do here. Skip this step\"\n"
"%build\n"
"echo \"Nothing to do here. Skip this step\"\n"
"%install\n"
"rm -rf %{buildroot}\n"
"mkdir -p %{buildroot}\n"
"mkdir -p %{buildroot}%{_bindir}\n"
"mkdir -p %{buildroot}%{_datadir}/%{name}\n"
"mkdir -p %{buildroot}/usr/share/applications\n"
"cp -r %{_gd_shared_path}%{_gd_export_path}/usr/share/applications/* %{buildroot}/usr/share/applications/\n"
"cp -r %{_gd_shared_path}%{_gd_export_path}%{_bindir}/* %{buildroot}%{_bindir}/\n"
"cp -r %{_gd_shared_path}%{_gd_export_path}%{_datadir}/%{name}/* %{buildroot}%{_datadir}/%{name}/\n"
"%files\n"
"%defattr(644,root,root,-)\n"
"%attr(755,root,root) %{_bindir}/%{name}\n"
"%attr(644,root,root) %{_datadir}/%{name}/%{name}.png\n"
"%attr(644,root,root) %{_datadir}/%{name}/%{name}.pck\n"
"%attr(644,root,root) /usr/share/applications/%{name}.desktop\n"
"%changelog\n"
"* %{_gd_date} Godot Game Engine\n"
"- application %{name} packed to RPM\n"
"#$changelog$"
;
// --define "_datadir  /home/nemo/.local/share" if need install to user folder
const String desktop_file_template =
"[Desktop Entry]\n"
"Type=Application\n"
"X-Nemo-Application-Type=Game\n"
"Icon=%{_datadir}/%{name}/%{name}.png\n"
"Exec=%{_bindir}/%{name} --main-pack %{_datadir}/%{name}/%{name}.pck\n"
"Name=%{_gd_launcher_name}\n"
"Name[en]=%{_gd_launcher_name}\n"
;

static void _execute_thread(void *p_ud) {
    
    EditorNode::ExecuteThreadArgs *eta = (EditorNode::ExecuteThreadArgs *)p_ud;
    Error err = OS::get_singleton()->execute(eta->path, eta->args, true, NULL, &eta->output, &eta->exitcode, true, eta->execute_output_mutex);
    print_verbose("Thread exit status: " + itos(eta->exitcode));
    if (err != OK) {
        eta->exitcode = err;
    }
    
    eta->done = true;
}

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
        MerTarget() {
            arch = arch_unkown;
            name = "SailfishOS";
            version[0] = 3;
            version[1] = 2;
            version[2] = 0;
            version[3] = 12;
        }
        
		String     name;
		int        version[4]; // array of 4 integers
		TargetArch arch;
	};
    
    struct NativePackage {
        MerTarget target;
        String    name;
        String    launcher_name;
        String    version;
        String    release;
        String    description;
    };
protected:
    String get_current_date() const {
        // TODO implement get date function
        return String("Thu Dec 19 2019");
    }
    
    String mertarget_to_text(const MerTarget &target) const {
        Array args;
        for(int i = 0; i < 4; i++)
            args.push_back(target.version[i]);
        bool error;
        return target.name + String("-%d.%d.%d.%d-").sprintf(args, &error) + arch_to_text(target.arch);
    }
    
    String arch_to_text(TargetArch arch) const {
        switch(arch) {
            case arch_armv7hl:
                return "armv7hl";
                break;
            case arch_i486:
                return "i486";
                break;
            default:
                return "noarch";
                break;
        }
        return "noarch";
    }
    
    String get_sdk_config_path() const {
        String sdk_configs_path = OS::get_singleton()->get_config_path();
        //sdk_configs_path +=  String("/SailfishOS-SDK/"); // old SailfishSDK , before 3.0.7
        //mer_sdk_tools = sdk_configs_path + String("/mer-sdk-tools/Sailfish OS Build Engine/"); // old SailfishSDK , before 3.0.7
# ifdef OSX_ENABLED
        sdk_configs_path = OS::get_singleton()->get_environment("HOME") + String("/.config");
# endif
        sdk_configs_path += separator + sdk_config_dir;
        return sdk_configs_path;
    }
    
    String get_absolute_export_path(const String &realitive_export_path) const {
        String export_path = realitive_export_path;
        String project_path = ProjectSettings::get_singleton()->get_resource_path();
        
        if( project_path.find_last(separator) == project_path.length() - 1 )
            project_path = project_path.left( project_path.find_last(separator) );
        // make from realitive path an absolute path
        if( export_path.find( String(".") + separator ) == 0 ) {
            
            export_path = project_path + separator + export_path.substr(2, export_path.length() - 2);
        }
        else {
            
            int count_out_dir = 0;
            while( export_path.find(String("..") + separator) == 0 ) {
                count_out_dir++;
                export_path = export_path.substr(3, export_path.length() - 3);
            }
            for( int i = 0; i < count_out_dir; i++ ) {
                
                int pos = project_path.find_last(separator);
                if( pos >= 0 ) {
                    
                    project_path = project_path.left(pos);
                }
            }
            export_path = project_path + separator + export_path;
        }
        return export_path;
    }
    
    String get_sfdk_path(const Ref<EditorExportPreset> &p_preset) const {
        
        String sfdk_path = String(p_preset->get(prop_sailfish_sdk_path));
#ifdef WINDOWS_ENABLED
        sfdk_path += String("\\bin\\sfdk.exe");
#else
        sfdk_path += String("/bin/sfdk");
#endif
        return sfdk_path;
    }
    
    int execute_task(const String &p_path, const List<String> &p_arguments, List<String> &r_output) {
        EditorNode::ExecuteThreadArgs eta;
        eta.path = p_path;
        eta.args = p_arguments;
        eta.execute_output_mutex = Mutex::create();
        eta.exitcode = 255;
        eta.done = false;
        
        int prev_len = 0;
        
        eta.execute_output_thread = Thread::create(_execute_thread, &eta);
        
        ERR_FAIL_COND_V(!eta.execute_output_thread, 0);
        
        while (!eta.done) {
            eta.execute_output_mutex->lock();
            if (prev_len != eta.output.length()) {
                String to_add = eta.output.substr(prev_len, eta.output.length());
                prev_len = eta.output.length();
                r_output.push_back(to_add);
                Main::iteration();
            }
            eta.execute_output_mutex->unlock();
            OS::get_singleton()->delay_usec(1000);
        }
        
        Thread::wait_to_finish(eta.execute_output_thread);
        memdelete(eta.execute_output_thread);
        memdelete(eta.execute_output_mutex);
        
        return eta.exitcode;
    }
    
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

		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, prop_custom_binary_arm, PROPERTY_HINT_GLOBAL_FILE), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, prop_custom_binary_arm_debug, PROPERTY_HINT_GLOBAL_FILE), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, prop_custom_binary_x86, PROPERTY_HINT_GLOBAL_FILE), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, prop_custom_binary_x86_debug, PROPERTY_HINT_GLOBAL_FILE), ""));

		r_options->push_back(ExportOption(PropertyInfo(Variant::INT,    prop_version_release, PROPERTY_HINT_RANGE, "1,40096,1,or_greater"), 1));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, prop_version_string, PROPERTY_HINT_PLACEHOLDER_TEXT, "1.0.0"), "1.0.0"));

		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, prop_name, PROPERTY_HINT_PLACEHOLDER_TEXT, "harbour-$genname"), "harbour-$genname"));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, prop_launcher_name, PROPERTY_HINT_PLACEHOLDER_TEXT, "Game Name [default if blank]"), ""));
	}

	bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const override {
		String arm_template;
		String x86_template;
        Error err;
		bool p_debug = false;
        

		if(p_debug) 
			arm_template = String( p_preset->get(prop_custom_binary_arm_debug) );
		else
			arm_template = String( p_preset->get(prop_custom_binary_arm) );
		
		if(p_debug) 
			x86_template = String( p_preset->get(prop_custom_binary_x86_debug) );
		else
			x86_template = String( p_preset->get(prop_custom_binary_x86) );

		// print_verbose( String("arm_binary: ") + arm_template );
		// print_verbose( String("x86_binary: ") + x86_template );

		if( arm_template.empty() && x86_template.empty() ) {
			r_error = TTR("Cant export without SailfishOS export templates");
			r_missing_templates = true;
			return false;
		}
        else {
            bool one_tamplate = false;
            if(!arm_template.empty())
            {
                FileAccessRef template_file = FileAccess::open(arm_template,FileAccess::READ,&err);
                if( err != Error::OK ) {
                    arm_template.clear();
                }
                else {
//                    template_file->get_
                    one_tamplate = true;
                }
            }
            
            if(!x86_template.empty())
            {
                FileAccessRef template_file = FileAccess::open(x86_template,FileAccess::READ,&err);
                if( err != Error::OK ) {
                    x86_template.clear();
                }
                else
                    one_tamplate = true;
            }
            if( !one_tamplate ) {
                r_error = TTR("Template files not exists");
                return false;
            }
        }
        
        // here need check if SDK is exists
        String sfdk_path = String(p_preset->get( prop_sailfish_sdk_path));
        if( !DirAccess::exists(sfdk_path) ) {
            r_error = TTR("Wrong SailfishSDK path");
            return false;
        }
        
        // check SDK version, minimum is 3.0.7
        FileAccessRef sdk_release_file = FileAccess::open(sfdk_path + separator + String("sdk-release"), FileAccess::READ, &err);
        
        if( err != Error::OK ) {
            r_error = TTR("Wrong SailfishSDK path: cant find \"sdk-release\" file");
            return false;
        }
        bool wrong_sdk_version = false;
        String current_line;
        Vector<int> sdk_version;
        sdk_version.resize(3);
        while( !sdk_release_file->eof_reached() ) {
            current_line = sdk_release_file->get_line();
            Vector<String> splitted = current_line.split("=");
            if( splitted.size() < 2 )
                continue;
            
            if( splitted[0] == String("SDK_RELEASE") ) {
                
                RegEx regex("([0-9]+)\\.([0-9]+)\\.([0-9]+)");
                Array matches = regex.search_all(splitted[1]);
                // print_verbose( String("Matches size: ") + Variant(matches.size()) );
                if( matches.size() == 1 ) {
                    Ref<RegExMatch> rem = ((Ref<RegExMatch>)matches[0]);
                    Array names = rem->get_strings();
                    if( names.size() >= 4 ) {
                        if( int(names[1]) < 3 ) {
                            r_error = TTR("Minimum SailfishSDK version is 3.0.7, current is ") + current_line.split("=")[1];
                            wrong_sdk_version = true;
                        }
                        else if( int(names[2]) > 0 ) {
                            wrong_sdk_version = false;
                        }
                        else if( int(names[3]) < 7 ) {
                            r_error = TTR("Minimum SailfishSDK version is 3.0.7, current is ") + current_line.split("=")[1];
                            wrong_sdk_version = true;
                        }
                        sdk_version.set(0, int(names[1]));
                        sdk_version.set(1, int(names[2]));
                        sdk_version.set(2, int(names[3]));
                    }
                }
                else {
                    r_error = TTR("Cant parse \"sdk-release\" file in SailfishSDK direktory");
                    wrong_sdk_version = true;
                }
            }
            else if( splitted[0] == String("SDK_CONFIG_DIR") ) {
                sdk_config_dir = splitted[1];
            }
        }
        sdk_release_file->close();
        if( wrong_sdk_version ) {
            //r_error = TTR("Wrong SailfishSDK path: cant find \"sdk-release\" file");
            return false;
        }
        
        DirAccessRef da = DirAccess::open(sfdk_path, &err);
#ifdef WINDOWS_ENABLED
        sfdk_path += String("\\bin\\sfdk.exe");
#else
        sfdk_path += String("/bin/sfdk");
#endif
        if( err != Error::OK || !da || !da->file_exists(sfdk_path) ) {
            r_error = TTR("Wrong SailfishSDK path or sfdk tool not exists");
            return false;
        }
        
		String xml_path;
        String sdk_configs_path = get_sdk_config_path();
#ifdef WINDOWS_ENABLED
        xml_path = sdk_configs_path + String("\\libsfdk\\");
#else
        xml_path = sdk_configs_path + String("/libsfdk/");
#endif
		xml_path += String("buildengines.xml");

		XMLParser *xml_parser = memnew(XMLParser);
		if( xml_parser->open(xml_path) != Error::OK ) {
			memdelete(xml_parser);
			r_error = TTR("Cant open XML file: ") + xml_path;
			return false; 
		}

        while( xml_parser->read() == Error::OK ) {
			if( xml_parser->get_node_type() == XMLParser::NodeType::NODE_ELEMENT )
			{
				if( xml_parser->get_node_name() != String("value") ) {
                    String debug_string = xml_parser->get_node_name();
                    print_verbose(String("Node skipping is: ") + debug_string);
					//xml_parser->skip_section();
					continue;
				}
				if( xml_parser->has_attribute("key")) {
					if( xml_parser->get_attribute_value("key") == String("SharedHome") ) {
                        xml_parser->read();
						shared_home = xml_parser->get_node_data();
					}
					else if( xml_parser->get_attribute_value("key") == String("SharedSrc") ) {
                        xml_parser->read();
						shared_src = xml_parser->get_node_data();
					}
				}
			}
		}
		xml_parser->close();
		memdelete(xml_parser);
        
        String icon = ProjectSettings::get_singleton()->get("application/config/icon");
        bool result = true;
        String suffix = icon.right(icon.length() - 3).to_lower();
        if( suffix != String("png") ) {
            r_error += TTR("Icon file should be PNG. Set up custom icon for Sailfish, or change icon of project");
            result = false;
        }
        
        String export_path = get_absolute_export_path( p_preset->get_export_path() );
        
        if( !shared_home.empty() && export_path.find(shared_home) >= 0 ) {
//            result = result;
            return true && result;
        }
        else if( !shared_src.empty() && export_path.find(shared_src) >= 0 ) {
//            result = result;
            return true && result;
        }
//        else
//            result = false;
        r_error += TTR("Export path is outside of Shared Home in SailfishSDK (choose export path inside shared home):\nSharedHome: ") + shared_home + String("\nShareedSource: ") + shared_src;
        
		return false;
	}
    
	List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const override {
		List<String> ext;
		ext.push_back("rpm");
		return ext;
	}
    
	Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0) override {
		ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);
        
        EditorProgress ep("export", "Exporting for SailfishOS", 100, true);
        
//        String src_binary;
		String sdk_path = p_preset->get(prop_sailfish_sdk_path);
		String sdk_configs_path = OS::get_singleton()->get_config_path();
        String sfdk_tool = get_sfdk_path(p_preset);
//        String mer_sdk_tools;
		List<MerTarget> mer_target; // Mer targets list
		String arm_template;
		String x86_template;
        
        ep.step("checking export template binaries.",5);

        if(p_debug) {
			arm_template = String( p_preset->get(prop_custom_binary_arm_debug) );
            if(arm_template.empty()) {
                print_error("Debug armv7hl template path is emty. Try use release template.");
            }
        }
        if( arm_template.empty() ) {
			arm_template = String( p_preset->get(prop_custom_binary_arm) );
            if(arm_template.empty()) {
                print_error("No arm template setuped");
            }
        }
		
        if(p_debug) {
            x86_template = String( p_preset->get(prop_custom_binary_x86_debug) );
            if(x86_template.empty()) {
                print_error("Debug i486 template path is emty. Try use release template.");
            }
        }
        if( x86_template.empty() ) {
            x86_template = String( p_preset->get(prop_custom_binary_x86) );
            if(x86_template.empty()) {
                print_error("No i486 template setuped");
            }
        }
        if( arm_template.empty() && x86_template.empty() ) {
            return ERR_CANT_CREATE;
        }

        ep.step("found export template binaries.",10);
        List<String> args;
        List<String> output_list;
        args.push_back("tools");
        args.push_back("list");
        ep.step("check sfdk targets.",25);
//        int result = EditorNode::get_singleton()->execute_and_show_output(TTR("Run sfdk tool"), sfdk_tool, args, true, false);
        int result = execute_task(sfdk_tool, args, output_list);
        if (result != 0) {
            EditorNode::get_singleton()->show_warning(TTR("Building of Sailfish RPM failed, check output for the error.\nAlternatively visit docs.godotengine.org for Sailfish build documentation."));
            return ERR_CANT_CREATE;
        }
        else {
            List<String>::Element *e = output_list.front();
            while( e ) {
                print_verbose( e->get() );
                e = e->next();
            }
            // TODO parse export targets, and choose rightd one
            // first create rpm root dir, if its exists create another
            NativePackage pack;
            pack.release = p_preset->get(prop_version_release);
            pack.description = ProjectSettings::get_singleton()->get("application/config/description");
            pack.launcher_name = p_preset->get(prop_launcher_name);
            pack.name = p_preset->get(prop_name);
            pack.version = p_preset->get(prop_version_string);
            // TODO arch should be generated from current MerTarget
            pack.target.arch = arch_armv7hl;
            pack.target.version[0] = 3;
            pack.target.version[1] = 2;
            pack.target.version[2] = 1;
            pack.target.version[3] = 20;
            String target_string  = mertarget_to_text(pack.target);
            String export_path    = get_absolute_export_path(p_preset->get_export_path());
            String broot_path     = export_path + String("_buildroot");
            String rpm_prefix_path = broot_path.left( broot_path.find_last(separator) );
            String export_path_part;
            String sdk_shared_path;
            String rpm_dir_path   = broot_path + separator + String("rpm");
            String pck_path       = broot_path + separator + pack.name + String(".pck");
            String template_path  = broot_path + separator;
            String spec_file_path = rpm_dir_path + separator + pack.name + String(".spec");
            String desktop_file_path = broot_path + String("/usr/share/applications/").replace("/", separator) + pack.name + String(".desktop");
            String icon_file_path;
            String data_dir = "/usr/share";
            String bin_dir = "/usr/bin";
#ifdef WINDOWS_ENABLED
            String data_dir_native = data_dir.replace("/","\\");
            String bin_dir_native = bin_dir.replace("/","\\");
#else
            String &data_dir_native = data_dir;
            String &bin_dir_native = bin_dir;
#endif
            Error err;
            if( !shared_home.empty() ) {
                if( export_path.find(shared_home) == 0 ) {
                    export_path_part = export_path.substr(shared_home.length(), export_path.length() - shared_home.length()).replace(separator,"/") + String("_buildroot");
                    sdk_shared_path = String("/home/mersdk/share");
                }
            }
//            DirAccessRef broot = DirAccess::open(broot_path, err);
            DirAccessRef broot = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
            err = broot->make_dir_recursive(rpm_dir_path);
            if( err != Error::OK ) {
                print_error( String("Cant create directory: ") + broot_path );
                return ERR_CANT_CREATE;
            }
            
            err = broot->make_dir_recursive(broot_path + data_dir_native + separator + pack.name);
            err = broot->make_dir_recursive(broot_path + bin_dir_native);
            err = broot->make_dir_recursive(broot_path + String("/usr/share/applications/").replace("/", separator));

            ep.step("copy export tempalte to buildroot",30);
            if ( pack.target.arch == arch_armv7hl ) {
                
                if( !arm_template.empty() ) {
                    
                    if( broot->copy(arm_template,broot_path + bin_dir_native + separator + pack.name) != Error::OK ) {
                        
                        print_error( String("Cant copy armv7hl template binary to: ") + broot_path + bin_dir_native + separator + pack.name );
                        return ERR_CANT_CREATE;
                    }
                }
                else {
                    
                    String debugs = (p_debug == true)?String(" debug"):String("");
                    print_error( String("armv7hl") + debugs + String(" template is emty") );
                    return ERR_CANT_CREATE;
                }
            }
            else if ( pack.target.arch == arch_x86 ) {
                if( !x86_template.empty() ) {
                    
                    if( broot->copy(x86_template,broot_path + bin_dir_native + separator + pack.name) != Error::OK ) {
                        
                        print_error( String("Cant copy x86 template binary to: ") + broot_path + bin_dir_native + separator + pack.name );
                        return ERR_CANT_CREATE;
                    }
                }
                else {
                    
                    String debugs = (p_debug == true)?String(" debug"):String("");
                    print_error( String("i486") + debugs + String(" template is emty") );
                    return ERR_CANT_CREATE;
                }
            }
            else {
                
                print_error("Wrong architecture of package");
                return ERR_CANT_CREATE;
            }
                
            ep.step("create *.pck file.",35);
            pck_path = broot_path + separator + data_dir_native + separator + pack.name + separator + pack.name + String(".pck");
            err = export_pack(p_preset, p_debug, pck_path);
            if( err != Error::OK ) {
                print_error( String("Cant create *.pck: ") + pck_path );
                return err;
            }
            ep.step(String("generate ") + pack.name + String(".spec file"), 45);
            {
                FileAccessRef spec_file = FileAccess::open(spec_file_path, FileAccess::WRITE, &err);
                if( err != Error::OK ) {
                    print_error( String("Cant create *.spec: ") + spec_file_path );
                    return ERR_CANT_CREATE;
                }
                String spec_text = spec_file_tempalte.replace("%{_gd_application_name}", pack.name);
                spec_text = spec_text.replace("%{_gd_launcher_name}", pack.launcher_name);
                spec_text = spec_text.replace("%{_gd_version}", pack.version);
                spec_text = spec_text.replace("%{_gd_release}", pack.release);
                spec_text = spec_text.replace("%{_gd_architecture}", arch_to_text(pack.target.arch) );
                spec_text = spec_text.replace("%{_gd_description}", pack.description);
                spec_text = spec_text.replace("%{_gd_shared_path}", sdk_shared_path);
                spec_text = spec_text.replace("%{_gd_export_path}", export_path_part);
                spec_text = spec_text.replace("%{_gd_date}", get_current_date() );
                spec_text = spec_text.replace("%{_datadir}", data_dir);
                spec_text = spec_text.replace("%{_bindir}", bin_dir);
                
                spec_file->store_string(spec_text);
                spec_file->flush();
                spec_file->close();
            }
            ep.step(String("generate ") + pack.name + String(".desktop file"), 50);
            {
                //desktop_file_path = broot_path + desktop_file_path;
                FileAccessRef desktop_file = FileAccess::open(desktop_file_path, FileAccess::WRITE, &err);
                if( err != Error::OK ) {
                    print_error( String("Cant create *.desktop: ") + desktop_file_path );
                    return ERR_CANT_CREATE;
                }
                String file_text = desktop_file_template.replace("%{_gd_launcher_name}", pack.launcher_name);
                file_text = file_text.replace("%{name}", pack.name);
                file_text = file_text.replace("%{_datadir}", data_dir);
                file_text = file_text.replace("%{_bindir}", bin_dir);
                desktop_file->store_string(file_text);
                desktop_file->flush();
                desktop_file->close();
            }
            ep.step(String("copy project icon"), 55);
            String icon = ProjectSettings::get_singleton()->get("application/config/icon");
            icon_file_path = broot_path + separator + data_dir_native + separator + pack.name + separator + pack.name + String(".png");
            if( broot->copy(icon, icon_file_path) != Error::OK ) {
                print_error( String("Cant copy icon file \"") + icon + String("\" to \"") + icon_file_path + String("\""));
                return ERR_CANT_CREATE;
            }
            ep.step( String("setup sfdk tool for ") + arch_to_text(pack.target.arch) + String(" package build"), 60 );
            
            {
                args.clear();
                args.push_back("-c");
                args.push_back(String("target=") + target_string);
                args.push_back("-c");
                args.push_back(String("output-prefix=\"") + rpm_prefix_path + String("\""));
                args.push_back("--specfile");
                args.push_back(spec_file_path);
                args.push_back("package");
                int result = EditorNode::get_singleton()->execute_and_show_output(TTR("Run sfdk tool"), sfdk_tool, args, true, false);
//                if( execute_task(sfdk_tool, args, result) != Error::OK )
                if( result != 0 )
                {
                    return ERR_CANT_CREATE;
                }
            }
            ep.step( String("remove temp directory"), 90 );
            {
                DirAccessRef rmdir = DirAccess::open(broot_path, &err);
                if( err != Error::OK ) {
                    print_error("cant open dir");
                }
                rmdir->erase_contents_recursive();//rmdir->remove(<#String p_name#>)
                rmdir->remove(broot_path);
            }
            ep.step( String("build success"), 100 );
        }
        
        return Error::OK;
/*
#ifdef WINDOWS_ENABLED
		sdk_configs_path +=  String("\\SailfishSDK\\");
		mer_sdk_tools = sdk_configs_path + String("\\libsfdk\\build-target-tools\\Sailfish OS Build Engine\\");
#else		
		//sdk_configs_path +=  String("/SailfishOS-SDK/"); // old SailfishSDK , before 3.0.7
		sdk_configs_path +=  String("/SailfishSDK/");
		//mer_sdk_tools = sdk_configs_path + String("/mer-sdk-tools/Sailfish OS Build Engine/"); // old SailfishSDK , before 3.0.7
		mer_sdk_tools = sdk_configs_path + String("/libsfdk/build-target-tools/Sailfish OS Build Engine/");
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
				print_error( String("Too old Sailfish target ")  + target.name );
			}
			entry = dir->get_next();
		}
		dir->list_dir_end();

//        EditorProgress ep("export", "Exporting for SailfishOS", 105, true);
		List<PropertyInfo> props = p_preset->get_properties();

		sdk_path = String(p_preset->get( prop_sailfish_sdk_path));
//        print_verbose(String("SDK path is ") + sdk_path);
//        print_verbose(String("Platfrom config path: ") + sdk_configs_path );
		
		
		for( int target_num = 0; mer_target.size(); target_num++ )
		{
			String merssh = sdk_path + "/bin/sfdk";
			String build_command = mer_sdk_tools + "/" + mer_target[target_num].name + "/rpmbuild";
			List<String> cmdline;
			cmdline.push_back("--help");
			cmdline.push_back("--version");
			int result = EditorNode::get_singleton()->execute_and_show_output(TTR("Building Sailfish RPM (rpmbuild)"), build_command, cmdline);
			if (result != 0) {
				EditorNode::get_singleton()->show_warning(TTR("Building of Sailfish RPM failed, check output for the error.\nAlternatively visit docs.godotengine.org for Sailfish build documentation."));
				return ERR_CANT_CREATE;
			}
		}*/

		return Error::OK;
	}

	void resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, Set<String> &p_features) override {

	}
protected:
	Ref<ImageTexture> logo;
    mutable String shared_home;
    mutable String shared_src;
    mutable String sdk_config_dir;
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
