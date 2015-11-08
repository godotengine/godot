#include "pe_bliss/pe_bliss.h"
#include "pe_bliss/pe_bliss_resources.h"
#include "core/ustring.h"
#include "core/dvector.h"
#include "os/file_access.h"

using namespace pe_bliss;

String pe_bliss_add_resrc(const char* p_path, int version_major, int version_minor,
												String& company_name, String& file_description,
												String& legal_copyright, String& version_text, 
												String& product_name, String& godot_version,
												DVector<uint8_t>& icon_content) {
	try
	{
		pe_base image(pe_factory::create_pe(p_path));
		
		const section_list& pe_sections = image.get_image_sections();
		uint32_t end_of_pe = 0;
		FileAccess *dst;
		DVector<uint8_t> overlay_data;
		if(image.has_overlay())
		{
			end_of_pe = pe_sections.back().get_pointer_to_raw_data() + pe_sections.back().get_size_of_raw_data();
			dst=FileAccess::open(p_path,FileAccess::READ);
			if (dst) {
				overlay_data.resize(dst->get_len()-end_of_pe);
				dst->seek(end_of_pe);
				DVector<uint8_t>::Write overlay_data_write = overlay_data.write();
				dst->get_buffer(overlay_data_write.ptr(),overlay_data.size());
				dst->close();
				memdelete(dst);
			}
		}
		resource_directory root;
		if(image.has_resources())
		{
			root = resource_directory(get_resources(image));
		}
		pe_resource_manager res(root);
		if(image.has_resources())
		{
			if(icon_content.size()) {
				if(res.resource_exists(pe_resource_viewer::resource_icon))
				{
					res.remove_resource_type(pe_resource_viewer::resource_icon);
				}
				if(res.resource_exists(pe_resource_viewer::resource_icon_group))
				{
					res.remove_resource_type(pe_resource_viewer::resource_icon_group);
				}
			}
			if(res.resource_exists(pe_resource_viewer::resource_version))
			{
				res.remove_resource_type(pe_resource_viewer::resource_version);
			}
		}
		file_version_info file_info;
		file_info.set_file_os(file_version_info::file_os_nt_win32);
		file_info.set_file_type(file_version_info::file_type_application);
		unsigned int ver = version_major << 16;
		ver = ver + version_minor;
		file_info.set_file_version_ms(ver);
		file_info.set_file_version_ls(0x00000000);
		file_info.set_product_version_ms(ver);
		file_info.set_product_version_ls(0x00000000);
		lang_string_values_map strings;
		translation_values_map translations;
		version_info_editor version(strings, translations);
		version.add_translation(version_info_editor::default_language_translation);
		version.set_company_name(company_name.c_str());
		version.set_file_description(file_description.c_str());
		if (!product_name.empty()) {
			version.set_internal_name((product_name+String(".exe")).c_str());
			version.set_original_filename((product_name+String(".exe")).c_str());
			version.set_product_name(product_name.c_str());
		}
		version.set_legal_copyright(legal_copyright.c_str());
		version.set_product_version(version_text.c_str());
		if(!godot_version.empty()) version.set_property(L"Godot Engine Version", godot_version.c_str() );
		resource_version_info_writer(res).set_version_info(file_info, strings, translations, 1033, 1200);
		if(icon_content.size()) {
			std::string icon;
			icon.resize(icon_content.size());
			for(int i=0; i<icon_content.size(); i++)
			{
				icon[i] = icon_content[i];
			}
			resource_cursor_icon_writer(res).add_icon(icon, L"MAIN_ICON", 1033);
		}
		if(image.has_resources())
		{
			rebuild_resources(image, root, image.section_from_directory(pe_win::image_directory_entry_resource));
		} else {
			section new_resources;
			new_resources.get_raw_data().resize(1);
			new_resources.set_name(".rsrc");
			new_resources.readable(true);
			section& attached_section = image.add_section(new_resources);
			rebuild_resources(image, root, attached_section);
		}
		rebuild_pe(image, p_path);
		if(image.has_overlay() && end_of_pe) {
			dst=FileAccess::open(p_path,FileAccess::READ_WRITE);
			if (dst) {
				dst->seek_end();
				DVector<uint8_t>::Read overlay_data_read = overlay_data.read();
				dst->store_buffer(overlay_data_read.ptr(),overlay_data.size());
				dst->close();
				memdelete(dst);
			}
		}
		return String();
	} catch(const pe_exception& e) {
		String ret("Error In Add rsrc Section : ");
		return ret + String(e.what());
	}
}
