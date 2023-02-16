/**************************************************************************/
/*  resource_importer_psd.cpp                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "resource_importer_psd.h"

#include "core/io/file_access.h"
#include "core/io/resource_saver.h"
#include "scene/resources/texture.h"

#ifdef TOOLS_ENABLED
#include "editor/import/audio_stream_import_settings.h"
#endif

String ResourceImporterPSD::get_importer_name() const {
    return "psd";
}

String ResourceImporterPSD::get_visible_name() const {
    return "PSD";
}

void ResourceImporterPSD::get_recognized_extensions(List<String> *p_extensions) const {
    p_extensions->push_back("psd");
}

String ResourceImporterPSD::get_save_extension() const {
    return "psdstr";
}

String ResourceImporterPSD::get_resource_type() const {
    return "PSDTexture";
}

bool ResourceImporterPSD::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
    return true;
}

int ResourceImporterPSD::get_preset_count() const {
    return 0;
}

String ResourceImporterPSD::get_preset_name(int p_idx) const {
    return String();
}

void ResourceImporterPSD::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
    
}


#ifdef TOOLS_ENABLED
bool ResourceImporterPSD::has_advanced_options() const {
    return true;
}
void ResourceImporterPSD::show_advanced_options(const String &p_path) {
    
}
#endif


Ref<PSDTexture> ResourceImporterPSD::import_psd(const String &p_path) {
    Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
    ERR_FAIL_COND_V(f.is_null(), Ref<PSDTexture>());

    uint64_t len = f->get_length();

    Vector<uint8_t> data;
    data.resize(len);
    uint8_t *w = data.ptrw();

    f->get_buffer(w, len);

    Ref<PSDTexture> psd_texture;
    psd_texture.instantiate();

    psd_texture->set_data(data);
    ERR_FAIL_COND_V(!psd_texture->get_data().size(), Ref<PSDTexture>());

    return psd_texture;
}

Error ResourceImporterPSD::import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
    Ref<PSDTexture> psd_texture = import_psd(p_source_file);


    return ResourceSaver::save(psd_texture, p_save_path + ".psdstr");
}

ResourceImporterPSD::ResourceImporterPSD() {
}
