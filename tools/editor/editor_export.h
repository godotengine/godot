/*************************************************************************/
/*  editor_import_export.h                                               */
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
#ifndef EDITOR_EXPORT_H
#define EDITOR_EXPORT_H



#include "resource.h"
#include "scene/main/node.h"
#include "scene/resources/texture.h"
#include "scene/main/timer.h"

class EditorProgress;
class FileAccess;
class EditorExportPlatform;

class EditorExportPreset : public Reference {

	GDCLASS( EditorExportPreset,Reference )
public:
	enum ExportFilter {
		EXPORT_ALL_RESOURCES,
		EXPORT_SELECTED_SCENES,
		EXPORT_SELECTED_RESOURCES,
		EXPORT_ALL_FILES,
	};

private:

	Ref<EditorExportPlatform> platform;
	ExportFilter export_filter;
	String include_filter;
	String exclude_filter;

	String exporter;
	Set<String> selected_files;
	bool runnable;

	Vector<String> patches;

friend class EditorExport;
friend class EditorExportPlatform;

	List<PropertyInfo> properties;
	Map<StringName,Variant> values;

	String name;
protected:
	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

public:

	Ref<EditorExportPlatform> get_platform();
	bool has(const StringName& p_property) const { return values.has(p_property); }

	Vector<String> get_files_to_export() const;

	void add_export_file(const String& p_path);
	void remove_export_file(const String& p_path);
	bool has_export_file(const String& p_path);

	void set_name(const String& p_name);
	String get_name() const;

	void set_runnable(bool p_enable);
	bool is_runnable() const;

	void set_export_filter(ExportFilter p_filter);
	ExportFilter get_export_filter() const;

	void set_include_filter(const String& p_include);
	String get_include_filter() const;

	void set_exclude_filter(const String& p_exclude);
	String get_exclude_filter() const;

	void add_patch(const String& p_path,int p_at_pos=-1);
	void set_patch(int p_index,const String& p_path);
	String get_patch(int p_index);
	void remove_patch(int p_idx);
	Vector<String> get_patches() const;

	const List<PropertyInfo>& get_properties() const { return properties; }

	EditorExportPreset();
};


class EditorExportPlatform : public Reference {

	GDCLASS( EditorExportPlatform,Reference )

public:

	typedef Error (*EditorExportSaveFunction)(void *p_userdata,const String& p_path, const Vector<uint8_t>& p_data,int p_file,int p_total);

private:

	struct SavedData {

		String path;
		uint64_t ofs;
		uint64_t size;
	};

	struct PackData {

		FileAccess *f;
		Vector<SavedData> file_ofs;
		EditorProgress *ep;
	};

	struct ZipData {

		void* zip;
		EditorProgress *ep;
		int count;

	};

	void gen_debug_flags(Vector<String> &r_flags, int p_flags);
	static Error _save_pack_file(void *p_userdata,const String& p_path, const Vector<uint8_t>& p_data,int p_file,int p_total);
	static Error _save_zip_file(void *p_userdata,const String& p_path, const Vector<uint8_t>& p_data,int p_file,int p_total);


protected:

	virtual void get_preset_features(const Ref<EditorExportPreset>& p_preset,List<String> *r_features)=0;
	String find_export_template(String template_file_name, String *err=NULL) const;

public:


	struct ExportOption {
		PropertyInfo option;
		Variant default_value;

		ExportOption(const PropertyInfo& p_info,const Variant& p_default) { option=p_info; default_value=p_default; }
		ExportOption() {}
	};

	virtual Ref<EditorExportPreset> create_preset();

	virtual void get_export_options(List<ExportOption> *r_options)=0;
	virtual String get_name() const =0;
	virtual Ref<Texture> get_logo() const =0;


	Error export_project_files(const Ref<EditorExportPreset>& p_preset,EditorExportSaveFunction p_func, void* p_udata);

	Error save_pack(const Ref<EditorExportPreset>& p_preset,FileAccess *p_where);
	Error save_zip(const Ref<EditorExportPreset>& p_preset,const String& p_path);


	virtual bool poll_devices() { return false; }
	virtual int get_device_count() const { return 0; }
	virtual String get_device_name(int p_device) const { return ""; }
	virtual String get_device_info(int p_device) const { return ""; }

	enum DebugFlags {
		DEBUG_FLAG_DUMB_CLIENT=1,
		DEBUG_FLAG_REMOTE_DEBUG=2,
		DEBUG_FLAG_REMOTE_DEBUG_LOCALHOST=4,
		DEBUG_FLAG_VIEW_COLLISONS=8,
		DEBUG_FLAG_VIEW_NAVIGATION=16,
	};

	virtual Error run(int p_device,int p_debug_flags) { return OK; }

	virtual bool can_export(String *r_error=NULL) const=0;

	virtual String get_binary_extension() const=0;
	virtual Error export_project(const Ref<EditorExportPreset>& p_preset,const String& p_path,int p_flags=0)=0;

	EditorExportPlatform();
};


class EditorExport : public Node {
	GDCLASS(EditorExport,Node);

	Vector<Ref<EditorExportPlatform> > export_platforms;
	Vector<Ref<EditorExportPreset> > export_presets;

	Timer *save_timer;
	bool block_save;

	static EditorExport *singleton;

	void _save();
protected:

friend class EditorExportPreset;
	void save_presets();

	void _notification(int p_what);
	static void _bind_methods();
public:

	static EditorExport * get_singleton() { return singleton; }

	void add_export_platform(const Ref<EditorExportPlatform>& p_platform);
	int get_export_platform_count();
	Ref<EditorExportPlatform> get_export_platform(int p_idx);


	void add_export_preset(const Ref<EditorExportPreset>& p_preset,int p_at_pos=-1);
	int get_export_preset_count() const;
	Ref<EditorExportPreset> get_export_preset(int p_idx);
	void remove_export_preset(int p_idx);

	void load_config();

	EditorExport();
	~EditorExport();
};




class EditorExportPlatformPC : public EditorExportPlatform {

	GDCLASS( EditorExportPlatformPC,EditorExportPlatform )

	Ref<ImageTexture> logo;
	String name;
	String extension;



public:

	virtual void get_preset_features(const Ref<EditorExportPreset>& p_preset,List<String>* r_features);

	virtual void get_export_options(List<ExportOption> *r_options);

	virtual String get_name() const;
	virtual Ref<Texture> get_logo() const;

	virtual bool can_export(String *r_error=NULL) const;
	virtual String get_binary_extension() const;
	virtual Error export_project(const Ref<EditorExportPreset>& p_preset,const String& p_path,int p_flags=0);

	void set_extension(const String& p_extension);
	void set_name(const String& p_name);

	void set_logo(const Ref<Texture>& p_loco);

	EditorExportPlatformPC();
};


#endif // EDITOR_IMPORT_EXPORT_H
