/**************************************************************************/
/*  doc_translation.cpp                                                   */
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

#include "doc_translation.h"

#include "core/io/compression.h"
#include "core/io/file_access_memory.h"
#include "core/io/translation_loader_po.h"
#include "core/string/translation_server.h"
#include "editor/translations/doc_translations.gen.h"

void load_doc_translations(const String &p_locale) {
	const Ref<TranslationDomain> domain = TranslationServer::get_singleton()->get_or_add_domain("godot.documentation");

	const DocTranslationList *dtl = _doc_translations;
	while (dtl->data) {
		if (dtl->lang == p_locale) {
			Vector<uint8_t> data;
			data.resize(dtl->uncomp_size);
			const int64_t ret = Compression::decompress(data.ptrw(), dtl->uncomp_size, dtl->data, dtl->comp_size, Compression::MODE_DEFLATE);
			ERR_FAIL_COND_MSG(ret == -1, "Compressed file is corrupt.");

			Ref<FileAccessMemory> fa;
			fa.instantiate();
			fa->open_custom(data.ptr(), data.size());

			Ref<Translation> tr = TranslationLoaderPO::load_translation(fa);

			if (tr.is_valid()) {
				tr->set_locale(dtl->lang);
				domain->add_translation(tr);
				break;
			}
		}

		dtl++;
	}
}
