#include "tools/editor/editor_import_export.h"

class EditorExportPlatformWindows : public EditorExportPlatformPC {
	OBJ_TYPE( EditorExportPlatformWindows,EditorExportPlatformPC );
	
private:
	String icon_ico;
	String icon_png;
	bool icon16;
	bool icon32;
	bool icon48;
	bool icon64;
	bool icon128;
	bool icon256;
	String company_name;
	String file_description;
	String product_name;
	String legal_copyright;
	String version_text;
	int version_major;
	int version_minor;
	bool set_godot_version;
	void store_16(DVector<uint8_t>& vector, uint16_t value); ///< store 16 bits uint 
	void store_32(DVector<uint8_t>& vector, uint32_t value); ///< store 32 bits uint 
	
protected:
	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;
	
public:
	Error export_project(const String& p_path, bool p_debug, bool p_dumb=false, bool p_remote_debug=false);
	EditorExportPlatformWindows();
};

void register_windows_exporter();

