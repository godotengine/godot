/*************************************************************************/
/*  editor_texture_import_plugin.h                                       */
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
#ifndef EDITOR_TEXTURE_IMPORT_PLUGIN_H
#define EDITOR_TEXTURE_IMPORT_PLUGIN_H

#if 0
#include "editor/editor_dir_dialog.h"
#include "editor/editor_file_system.h"
#include "editor/editor_import_export.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/slider.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/tree.h"



class EditorNode;
class EditorTextureImportDialog;

class EditorTextureImportPlugin : public EditorImportPlugin {

	GDCLASS(EditorTextureImportPlugin,EditorImportPlugin);
public:


	enum Mode {
		MODE_TEXTURE_2D,
		MODE_TEXTURE_3D,
		MODE_ATLAS,
		MODE_LARGE,
		MODE_MAX
	};



private:

	EditorNode *editor;
	EditorTextureImportDialog *dialog;
	static EditorTextureImportPlugin *singleton;
	//used by other importers such as mesh

	Error _process_texture_data(Ref<ImageTexture> &texture, int format, float quality, int flags,EditorExportPlatform::ImageCompression p_compr,int tex_flags,float shrink);
	void compress_image(EditorExportPlatform::ImageCompression p_mode,Image& image,bool p_smaller);

	uint32_t texture_flags_to_export_flags(uint32_t p_tex_flags) const;
public:


	static EditorTextureImportPlugin *get_singleton() { return singleton; }

	enum ImageFormat {

		IMAGE_FORMAT_UNCOMPRESSED,
		IMAGE_FORMAT_COMPRESS_DISK_LOSSLESS,
		IMAGE_FORMAT_COMPRESS_DISK_LOSSY,
		IMAGE_FORMAT_COMPRESS_RAM,
	};

	enum ImageFlags {

		IMAGE_FLAG_STREAM_FORMAT=1,
		IMAGE_FLAG_FIX_BORDER_ALPHA=2,
		IMAGE_FLAG_ALPHA_BIT=4, //hint for compressions that use a bit for alpha
		IMAGE_FLAG_COMPRESS_EXTRA=8, // used for pvrtc2
		IMAGE_FLAG_NO_MIPMAPS=16, //normal for 2D games
		IMAGE_FLAG_REPEAT=32, //usually disabled in 2D
		IMAGE_FLAG_FILTER=64, //almost always enabled
		IMAGE_FLAG_PREMULT_ALPHA=128,//almost always enabled
		IMAGE_FLAG_CONVERT_TO_LINEAR=256, //convert image to linear
		IMAGE_FLAG_CONVERT_NORMAL_TO_XY=512, //convert image to linear
		IMAGE_FLAG_USE_ANISOTROPY=1024, //convert image to linear
	};

	virtual String get_name() const;
	virtual String get_visible_name() const;
	virtual void import_dialog(const String& p_from="");
	virtual Error import(const String& p_path, const Ref<ResourceImportMetadata>& p_from);
	virtual Error import2(const String& p_path, const Ref<ResourceImportMetadata>& p_from,EditorExportPlatform::ImageCompression p_compr, bool p_external=false);
	virtual Vector<uint8_t> custom_export(const String& p_path,const Ref<EditorExportPlatform> &p_platform);

	virtual void import_from_drop(const Vector<String>& p_drop,const String& p_dest_path);
	virtual void reimport_multiple_files(const Vector<String>& p_list);
	virtual bool can_reimport_multiple_files() const;

	EditorTextureImportPlugin(EditorNode* p_editor=NULL);
};


class EditorTextureExportPlugin : public EditorExportPlugin {

	GDCLASS( EditorTextureExportPlugin, EditorExportPlugin);


public:

	virtual Vector<uint8_t> custom_export(String& p_path,const Ref<EditorExportPlatform> &p_platform);
	EditorTextureExportPlugin();
};

class EditorImportTextureOptions : public VBoxContainer {

	GDCLASS( EditorImportTextureOptions, VBoxContainer );


	OptionButton *format;
	VBoxContainer *quality_vb;
	HSlider *quality;
	Tree *flags;
	Vector<TreeItem*> items;


	bool updating;

	void _changedp(int p_value);
	void _changed();


protected:
	static void _bind_methods();
	void _notification(int p_what);

public:



	void set_format(EditorTextureImportPlugin::ImageFormat p_format);
	EditorTextureImportPlugin::ImageFormat get_format() const;

	void set_flags(uint32_t p_flags);
	uint32_t get_flags() const;

	void set_quality(float p_quality);
	float get_quality() const;

	void show_2d_notice();

	EditorImportTextureOptions();


};
#endif // EDITOR_TEXTURE_IMPORT_PLUGIN_H
#endif
