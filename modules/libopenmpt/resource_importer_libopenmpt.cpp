/*************************************************************************/
/*  resource_importer_libopenmpt.cpp                                     */
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

#include "resource_importer_libopenmpt.h"

#include "io/resource_saver.h"
#include "os/file_access.h"
#include "scene/resources/texture.h"

String ResourceImporterLibopenmpt::get_importer_name() const {

	return "libopenmpt";
}

String ResourceImporterLibopenmpt::get_visible_name() const {

	return "Libopenmpt";
}
void ResourceImporterLibopenmpt::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("669");
	p_extensions->push_back("amf");
	p_extensions->push_back("ams");
	p_extensions->push_back("dbm");
	p_extensions->push_back("digi");
	p_extensions->push_back("dmf");
	p_extensions->push_back("dsm");
	p_extensions->push_back("dtm");
	p_extensions->push_back("far");
	p_extensions->push_back("gdm");
	p_extensions->push_back("ice");
	p_extensions->push_back("imf");
	p_extensions->push_back("it");

	// ITP file format is libopenmpt's own file format. It requires support
	// for external file access, though. Disabling for now.
	//p_extensions->push_back("itp");

	// j2b requires zlib or miniz (see https://lib.openmpt.org/doc/dependencies.html)
#if defined(MPT_WITH_ZLIB) || defined(MPT_WITH_MINIZ)
	p_extensions->push_back("j2b");
#endif

	p_extensions->push_back("m15");
	p_extensions->push_back("mdl");
	p_extensions->push_back("med");

	// There seems to be some support for midi files in place, but
	// according to https://lib.openmpt.org/libopenmpt/faq/ its usage
	// is discouraged. As there are additional dependencies required to
	// get his to work at all, it is questionable if this is worth the effort.
	// Disabling midi support for now.
	//p_extensions->push_back("mid");
	//p_extensions->push_back("midi");

	p_extensions->push_back("mmcmp");
	p_extensions->push_back("mms");

	// mo3 requires vorbis (see https://lib.openmpt.org/doc/dependencies.html)
#if defined(MPT_WITH_VORBIS)
	p_extensions->push_back("mo3");
#endif

	p_extensions->push_back("mod");
	p_extensions->push_back("mptm");
	p_extensions->push_back("mt2");
	p_extensions->push_back("mtm");
	p_extensions->push_back("nst");
	p_extensions->push_back("okt");
	p_extensions->push_back("plm");
	p_extensions->push_back("ppm");
	p_extensions->push_back("psm");
	p_extensions->push_back("pt36");
	p_extensions->push_back("ptm");
	p_extensions->push_back("s3m");
	p_extensions->push_back("sfx");
	p_extensions->push_back("sfx2");
	p_extensions->push_back("st26");
	p_extensions->push_back("stk");
	p_extensions->push_back("stm");
	p_extensions->push_back("stp");
	p_extensions->push_back("ult");
	p_extensions->push_back("umx");
	p_extensions->push_back("wow");
	p_extensions->push_back("xm");
	p_extensions->push_back("xpk");
}

String ResourceImporterLibopenmpt::get_save_extension() const {
	return "libopenmptstr";
}

String ResourceImporterLibopenmpt::get_resource_type() const {

	return "AudioStreamLibopenmpt";
}

bool ResourceImporterLibopenmpt::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {

	return true;
}

int ResourceImporterLibopenmpt::get_preset_count() const {
	return 0;
}
String ResourceImporterLibopenmpt::get_preset_name(int p_idx) const {

	return String();
}

void ResourceImporterLibopenmpt::get_import_options(List<ImportOption> *r_options, int p_preset) const {

	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "loop"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "loop_offset"), 0));
}

Error ResourceImporterLibopenmpt::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files) {

	bool loop = p_options["loop"];
	float loop_offset = p_options["loop_offset"];

	FileAccess *f = FileAccess::open(p_source_file, FileAccess::READ);
	if (!f) {
		ERR_FAIL_COND_V(!f, ERR_CANT_OPEN);
	}

	size_t len = f->get_len();

	PoolVector<uint8_t> data;
	data.resize(len);
	PoolVector<uint8_t>::Write w = data.write();

	f->get_buffer(w.ptr(), len);

	memdelete(f);

	try {

		Ref<AudioStreamLibopenmpt> libopenmpt_stream;
		libopenmpt_stream.instance();
		libopenmpt_stream->set_data(data);
		libopenmpt_stream->set_loop(loop);
		libopenmpt_stream->set_loop_offset(loop_offset);
		return ResourceSaver::save(p_save_path + ".libopenmptstr", libopenmpt_stream);

	}
	catch (const std::exception & e) {

		// libopenmpt throws an exception, if the file/stream format cannot be
		// loaded.
		ERR_PRINT(e.what() ? e.what() : "unknown error");
		return ERR_FILE_CORRUPT;
	}
}

ResourceImporterLibopenmpt::ResourceImporterLibopenmpt() {
}
