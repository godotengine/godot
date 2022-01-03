/*************************************************************************/
/*  resource_importer_bmfont.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "resource_importer_bmfont.h"

#include "core/io/image_loader.h"
#include "core/io/resource_saver.h"

String ResourceImporterBMFont::get_importer_name() const {
	return "font_data_bmfont";
}

String ResourceImporterBMFont::get_visible_name() const {
	return "Font Data (AngelCode BMFont)";
}

void ResourceImporterBMFont::get_recognized_extensions(List<String> *p_extensions) const {
	if (p_extensions) {
		p_extensions->push_back("font");
		p_extensions->push_back("fnt");
	}
}

String ResourceImporterBMFont::get_save_extension() const {
	return "fontdata";
}

String ResourceImporterBMFont::get_resource_type() const {
	return "FontData";
}

bool ResourceImporterBMFont::get_option_visibility(const String &p_path, const String &p_option, const Map<StringName, Variant> &p_options) const {
	return true;
}

void ResourceImporterBMFont::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "compress"), true));
}

void _convert_packed_8bit(Ref<FontData> &r_font, Ref<Image> &p_source, int p_page, int p_sz) {
	int w = p_source->get_width();
	int h = p_source->get_height();

	PackedByteArray imgdata = p_source->get_data();
	const uint8_t *r = imgdata.ptr();

	PackedByteArray imgdata_r;
	imgdata_r.resize(w * h * 2);
	uint8_t *wr = imgdata_r.ptrw();

	PackedByteArray imgdata_g;
	imgdata_g.resize(w * h * 2);
	uint8_t *wg = imgdata_g.ptrw();

	PackedByteArray imgdata_b;
	imgdata_b.resize(w * h * 2);
	uint8_t *wb = imgdata_b.ptrw();

	PackedByteArray imgdata_a;
	imgdata_a.resize(w * h * 2);
	uint8_t *wa = imgdata_a.ptrw();

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int ofs_src = (i * w + j) * 4;
			int ofs_dst = (i * w + j) * 2;
			wr[ofs_dst + 0] = 255;
			wr[ofs_dst + 1] = r[ofs_src + 0];
			wg[ofs_dst + 0] = 255;
			wg[ofs_dst + 1] = r[ofs_src + 1];
			wb[ofs_dst + 0] = 255;
			wb[ofs_dst + 1] = r[ofs_src + 2];
			wa[ofs_dst + 0] = 255;
			wa[ofs_dst + 1] = r[ofs_src + 3];
		}
	}
	Ref<Image> img_r = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_r));
	r_font->set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 0, img_r);
	Ref<Image> img_g = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_g));
	r_font->set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 1, img_g);
	Ref<Image> img_b = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_b));
	r_font->set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 2, img_b);
	Ref<Image> img_a = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_a));
	r_font->set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 3, img_a);
}

void _convert_packed_4bit(Ref<FontData> &r_font, Ref<Image> &p_source, int p_page, int p_sz) {
	int w = p_source->get_width();
	int h = p_source->get_height();

	PackedByteArray imgdata = p_source->get_data();
	const uint8_t *r = imgdata.ptr();

	PackedByteArray imgdata_r;
	imgdata_r.resize(w * h * 2);
	uint8_t *wr = imgdata_r.ptrw();

	PackedByteArray imgdata_g;
	imgdata_g.resize(w * h * 2);
	uint8_t *wg = imgdata_g.ptrw();

	PackedByteArray imgdata_b;
	imgdata_b.resize(w * h * 2);
	uint8_t *wb = imgdata_b.ptrw();

	PackedByteArray imgdata_a;
	imgdata_a.resize(w * h * 2);
	uint8_t *wa = imgdata_a.ptrw();

	PackedByteArray imgdata_ro;
	imgdata_ro.resize(w * h * 2);
	uint8_t *wro = imgdata_ro.ptrw();

	PackedByteArray imgdata_go;
	imgdata_go.resize(w * h * 2);
	uint8_t *wgo = imgdata_go.ptrw();

	PackedByteArray imgdata_bo;
	imgdata_bo.resize(w * h * 2);
	uint8_t *wbo = imgdata_bo.ptrw();

	PackedByteArray imgdata_ao;
	imgdata_ao.resize(w * h * 2);
	uint8_t *wao = imgdata_ao.ptrw();

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int ofs_src = (i * w + j) * 4;
			int ofs_dst = (i * w + j) * 2;
			wr[ofs_dst + 0] = 255;
			wro[ofs_dst + 0] = 255;
			if (r[ofs_src + 0] > 0x0F) {
				wr[ofs_dst + 1] = (r[ofs_src + 0] - 0x0F) * 2;
				wro[ofs_dst + 1] = 0;
			} else {
				wr[ofs_dst + 1] = 0;
				wro[ofs_dst + 1] = r[ofs_src + 0] * 2;
			}
			wg[ofs_dst + 0] = 255;
			wgo[ofs_dst + 0] = 255;
			if (r[ofs_src + 1] > 0x0F) {
				wg[ofs_dst + 1] = (r[ofs_src + 1] - 0x0F) * 2;
				wgo[ofs_dst + 1] = 0;
			} else {
				wg[ofs_dst + 1] = 0;
				wgo[ofs_dst + 1] = r[ofs_src + 1] * 2;
			}
			wb[ofs_dst + 0] = 255;
			wbo[ofs_dst + 0] = 255;
			if (r[ofs_src + 2] > 0x0F) {
				wb[ofs_dst + 1] = (r[ofs_src + 2] - 0x0F) * 2;
				wbo[ofs_dst + 1] = 0;
			} else {
				wb[ofs_dst + 1] = 0;
				wbo[ofs_dst + 1] = r[ofs_src + 2] * 2;
			}
			wa[ofs_dst + 0] = 255;
			wao[ofs_dst + 0] = 255;
			if (r[ofs_src + 3] > 0x0F) {
				wa[ofs_dst + 1] = (r[ofs_src + 3] - 0x0F) * 2;
				wao[ofs_dst + 1] = 0;
			} else {
				wa[ofs_dst + 1] = 0;
				wao[ofs_dst + 1] = r[ofs_src + 3] * 2;
			}
		}
	}
	Ref<Image> img_r = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_r));
	r_font->set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 0, img_r);
	Ref<Image> img_g = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_g));
	r_font->set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 1, img_g);
	Ref<Image> img_b = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_b));
	r_font->set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 2, img_b);
	Ref<Image> img_a = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_a));
	r_font->set_texture_image(0, Vector2i(p_sz, 0), p_page * 4 + 3, img_a);

	Ref<Image> img_ro = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_ro));
	r_font->set_texture_image(0, Vector2i(p_sz, 1), p_page * 4 + 0, img_ro);
	Ref<Image> img_go = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_go));
	r_font->set_texture_image(0, Vector2i(p_sz, 1), p_page * 4 + 1, img_go);
	Ref<Image> img_bo = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_bo));
	r_font->set_texture_image(0, Vector2i(p_sz, 1), p_page * 4 + 2, img_bo);
	Ref<Image> img_ao = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_ao));
	r_font->set_texture_image(0, Vector2i(p_sz, 1), p_page * 4 + 3, img_ao);
}

void _convert_rgba_4bit(Ref<FontData> &r_font, Ref<Image> &p_source, int p_page, int p_sz) {
	int w = p_source->get_width();
	int h = p_source->get_height();

	PackedByteArray imgdata = p_source->get_data();
	const uint8_t *r = imgdata.ptr();

	PackedByteArray imgdata_g;
	imgdata_g.resize(w * h * 4);
	uint8_t *wg = imgdata_g.ptrw();

	PackedByteArray imgdata_o;
	imgdata_o.resize(w * h * 4);
	uint8_t *wo = imgdata_o.ptrw();

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int ofs = (i * w + j) * 4;

			if (r[ofs + 0] > 0x7F) {
				wg[ofs + 0] = r[ofs + 0];
				wo[ofs + 0] = 0;
			} else {
				wg[ofs + 0] = 0;
				wo[ofs + 0] = r[ofs + 0] * 2;
			}
			if (r[ofs + 1] > 0x7F) {
				wg[ofs + 1] = r[ofs + 1];
				wo[ofs + 1] = 0;
			} else {
				wg[ofs + 1] = 0;
				wo[ofs + 1] = r[ofs + 1] * 2;
			}
			if (r[ofs + 2] > 0x7F) {
				wg[ofs + 2] = r[ofs + 2];
				wo[ofs + 2] = 0;
			} else {
				wg[ofs + 2] = 0;
				wo[ofs + 2] = r[ofs + 2] * 2;
			}
			if (r[ofs + 3] > 0x7F) {
				wg[ofs + 3] = r[ofs + 3];
				wo[ofs + 3] = 0;
			} else {
				wg[ofs + 3] = 0;
				wo[ofs + 3] = r[ofs + 3] * 2;
			}
		}
	}
	Ref<Image> img_g = memnew(Image(w, h, 0, Image::FORMAT_RGBA8, imgdata_g));
	r_font->set_texture_image(0, Vector2i(p_sz, 0), p_page, img_g);

	Ref<Image> img_o = memnew(Image(w, h, 0, Image::FORMAT_RGBA8, imgdata_o));
	r_font->set_texture_image(0, Vector2i(p_sz, 1), p_page, img_o);
}

void _convert_mono_8bit(Ref<FontData> &r_font, Ref<Image> &p_source, int p_page, int p_ch, int p_sz, int p_ol) {
	int w = p_source->get_width();
	int h = p_source->get_height();

	PackedByteArray imgdata = p_source->get_data();
	const uint8_t *r = imgdata.ptr();

	int size = 4;
	if (p_source->get_format() == Image::FORMAT_L8) {
		size = 1;
		p_ch = 0;
	}

	PackedByteArray imgdata_g;
	imgdata_g.resize(w * h * 2);
	uint8_t *wg = imgdata_g.ptrw();

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int ofs_src = (i * w + j) * size;
			int ofs_dst = (i * w + j) * 2;
			wg[ofs_dst + 0] = 255;
			wg[ofs_dst + 1] = r[ofs_src + p_ch];
		}
	}
	Ref<Image> img_g = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_g));
	r_font->set_texture_image(0, Vector2i(p_sz, p_ol), p_page, img_g);
}

void _convert_mono_4bit(Ref<FontData> &r_font, Ref<Image> &p_source, int p_page, int p_ch, int p_sz, int p_ol) {
	int w = p_source->get_width();
	int h = p_source->get_height();

	PackedByteArray imgdata = p_source->get_data();
	const uint8_t *r = imgdata.ptr();

	int size = 4;
	if (p_source->get_format() == Image::FORMAT_L8) {
		size = 1;
		p_ch = 0;
	}

	PackedByteArray imgdata_g;
	imgdata_g.resize(w * h * 2);
	uint8_t *wg = imgdata_g.ptrw();

	PackedByteArray imgdata_o;
	imgdata_o.resize(w * h * 2);
	uint8_t *wo = imgdata_o.ptrw();

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int ofs_src = (i * w + j) * size;
			int ofs_dst = (i * w + j) * 2;
			wg[ofs_dst + 0] = 255;
			wo[ofs_dst + 0] = 255;
			if (r[ofs_src + p_ch] > 0x7F) {
				wg[ofs_dst + 1] = r[ofs_src + p_ch];
				wo[ofs_dst + 1] = 0;
			} else {
				wg[ofs_dst + 1] = 0;
				wo[ofs_dst + 1] = r[ofs_src + p_ch] * 2;
			}
		}
	}
	Ref<Image> img_g = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_g));
	r_font->set_texture_image(0, Vector2i(p_sz, 0), p_page, img_g);

	Ref<Image> img_o = memnew(Image(w, h, 0, Image::FORMAT_LA8, imgdata_o));
	r_font->set_texture_image(0, Vector2i(p_sz, p_ol), p_page, img_o);
}

Error ResourceImporterBMFont::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	print_verbose("Importing BMFont font from: " + p_source_file);

	Ref<FontData> font;
	font.instantiate();
	font->set_antialiased(false);
	font->set_multichannel_signed_distance_field(false);
	font->set_force_autohinter(false);
	font->set_hinting(TextServer::HINTING_NONE);
	font->set_oversampling(1.0f);

	FileAccessRef f = FileAccess::open(p_source_file, FileAccess::READ);
	if (f == nullptr) {
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, TTR("Cannot open font from file ") + "\"" + p_source_file + "\".");
	}

	int base_size = 16;
	int height = 0;
	int ascent = 0;
	int outline = 0;
	uint32_t st_flags = 0;
	String font_name;

	bool packed = false;
	uint8_t ch[4] = { 0, 0, 0, 0 }; // RGBA
	int first_gl_ch = -1;
	int first_ol_ch = -1;
	int first_cm_ch = -1;

	unsigned char magic[4];
	f->get_buffer((unsigned char *)&magic, 4);
	if (magic[0] == 'B' && magic[1] == 'M' && magic[2] == 'F') {
		// Binary BMFont file.
		ERR_FAIL_COND_V_MSG(magic[3] != 3, ERR_CANT_CREATE, vformat(TTR("Version %d of BMFont is not supported."), (int)magic[3]));

		uint8_t block_type = f->get_8();
		uint32_t block_size = f->get_32();
		while (!f->eof_reached()) {
			uint64_t off = f->get_position();
			switch (block_type) {
				case 1: /* info */ {
					ERR_FAIL_COND_V_MSG(block_size < 15, ERR_CANT_CREATE, TTR("Invalid BMFont info block size."));
					base_size = f->get_16();
					uint8_t flags = f->get_8();
					ERR_FAIL_COND_V_MSG(flags & 0x02, ERR_CANT_CREATE, TTR("Non-unicode version of BMFont is not supported."));
					if (flags & (1 << 3)) {
						st_flags |= TextServer::FONT_BOLD;
					}
					if (flags & (1 << 2)) {
						st_flags |= TextServer::FONT_ITALIC;
					}
					f->get_8(); // non-unicode charset, skip
					f->get_16(); // stretch_h, skip
					f->get_8(); // aa, skip
					f->get_32(); // padding, skip
					f->get_16(); // spacing, skip
					outline = f->get_8();
					// font name
					PackedByteArray name_data;
					name_data.resize(block_size - 14);
					f->get_buffer(name_data.ptrw(), block_size - 14);
					font_name = String::utf8((const char *)name_data.ptr(), block_size - 14);
					font->set_fixed_size(base_size);
				} break;
				case 2: /* common */ {
					ERR_FAIL_COND_V_MSG(block_size != 15, ERR_CANT_CREATE, TTR("Invalid BMFont common block size."));
					height = f->get_16();
					ascent = f->get_16();
					f->get_32(); // scale, skip
					f->get_16(); // pages, skip
					uint8_t flags = f->get_8();
					packed = (flags & 0x01);
					ch[3] = f->get_8();
					ch[0] = f->get_8();
					ch[1] = f->get_8();
					ch[2] = f->get_8();
					for (int i = 0; i < 4; i++) {
						if (ch[i] == 0 && first_gl_ch == -1) {
							first_gl_ch = i;
						}
						if (ch[i] == 1 && first_ol_ch == -1) {
							first_ol_ch = i;
						}
						if (ch[i] == 2 && first_cm_ch == -1) {
							first_cm_ch = i;
						}
					}
				} break;
				case 3: /* pages */ {
					int page = 0;
					CharString cs;
					char32_t c = f->get_8();
					while (!f->eof_reached() && f->get_position() <= off + block_size) {
						if (c == '\0') {
							String base_dir = p_source_file.get_base_dir();
							String file = base_dir.plus_file(String::utf8(cs.ptr(), cs.length()));
							if (RenderingServer::get_singleton() != nullptr) {
								Ref<Image> img;
								img.instantiate();
								Error err = ImageLoader::load_image(file, img);
								ERR_FAIL_COND_V_MSG(err != OK, ERR_FILE_CANT_READ, TTR("Can't load font texture: ") + "\"" + file + "\".");

								if (packed) {
									if (ch[3] == 0) { // 4 x 8 bit monochrome, no outline
										outline = 0;
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
										_convert_packed_8bit(font, img, page, base_size);
									} else if ((ch[3] == 2) && (outline > 0)) { // 4 x 4 bit monochrome, gl + outline
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
										_convert_packed_4bit(font, img, page, base_size);
									} else {
										ERR_FAIL_V_MSG(ERR_CANT_CREATE, TTR("Unsupported BMFont texture format."));
									}
								} else {
									if ((ch[0] == 0) && (ch[1] == 0) && (ch[2] == 0) && (ch[3] == 0)) { // RGBA8 color, no outline
										outline = 0;
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
										font->set_texture_image(0, Vector2i(base_size, 0), page, img);
									} else if ((ch[0] == 2) && (ch[1] == 2) && (ch[2] == 2) && (ch[3] == 2) && (outline > 0)) { // RGBA4 color, gl + outline
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
										_convert_rgba_4bit(font, img, page, base_size);
									} else if ((first_gl_ch >= 0) && (first_ol_ch >= 0) && (outline > 0)) { // 1 x 8 bit monochrome, gl + outline
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
										_convert_mono_8bit(font, img, page, first_gl_ch, base_size, 0);
										_convert_mono_8bit(font, img, page, first_ol_ch, base_size, 1);
									} else if ((first_cm_ch >= 0) && (outline > 0)) { // 1 x 4 bit monochrome, gl + outline
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
										_convert_mono_4bit(font, img, page, first_cm_ch, base_size, 1);
									} else if (first_gl_ch >= 0) { // 1 x 8 bit monochrome, no outline
										outline = 0;
										ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
										_convert_mono_8bit(font, img, page, first_gl_ch, base_size, 0);
									} else {
										ERR_FAIL_V_MSG(ERR_CANT_CREATE, TTR("Unsupported BMFont texture format."));
									}
								}
							}
							page++;
							cs = "";
						} else {
							cs += c;
						}
						c = f->get_8();
					}
				} break;
				case 4: /* chars */ {
					int char_count = block_size / 20;
					for (int i = 0; i < char_count; i++) {
						Vector2 advance;
						Vector2 size;
						Vector2 offset;
						Rect2 uv_rect;

						char32_t idx = f->get_32();
						uv_rect.position.x = (int16_t)f->get_16();
						uv_rect.position.y = (int16_t)f->get_16();
						uv_rect.size.width = (int16_t)f->get_16();
						size.width = uv_rect.size.width;
						uv_rect.size.height = (int16_t)f->get_16();
						size.height = uv_rect.size.height;
						offset.x = (int16_t)f->get_16();
						offset.y = (int16_t)f->get_16() - ascent;
						advance.x = (int16_t)f->get_16();
						if (advance.x < 0) {
							advance.x = size.width + 1;
						}

						int texture_idx = f->get_8();
						uint8_t channel = f->get_8();

						ERR_FAIL_COND_V_MSG(!packed && channel != 15, ERR_CANT_CREATE, TTR("Invalid glyph channel."));
						int ch_off = 0;
						switch (channel) {
							case 1:
								ch_off = 2;
								break; // B
							case 2:
								ch_off = 1;
								break; // G
							case 4:
								ch_off = 0;
								break; // R
							case 8:
								ch_off = 3;
								break; // A
							default:
								ch_off = 0;
								break;
						}
						font->set_glyph_advance(0, base_size, idx, advance);
						font->set_glyph_offset(0, Vector2i(base_size, 0), idx, offset);
						font->set_glyph_size(0, Vector2i(base_size, 0), idx, size);
						font->set_glyph_uv_rect(0, Vector2i(base_size, 0), idx, uv_rect);
						font->set_glyph_texture_idx(0, Vector2i(base_size, 0), idx, texture_idx * (packed ? 4 : 1) + ch_off);
						if (outline > 0) {
							font->set_glyph_offset(0, Vector2i(base_size, 1), idx, offset);
							font->set_glyph_size(0, Vector2i(base_size, 1), idx, size);
							font->set_glyph_uv_rect(0, Vector2i(base_size, 1), idx, uv_rect);
							font->set_glyph_texture_idx(0, Vector2i(base_size, 1), idx, texture_idx * (packed ? 4 : 1) + ch_off);
						}
					}
				} break;
				case 5: /* kerning */ {
					int pair_count = block_size / 10;
					for (int i = 0; i < pair_count; i++) {
						Vector2i kpk;
						kpk.x = f->get_32();
						kpk.y = f->get_32();
						font->set_kerning(0, base_size, kpk, Vector2((int16_t)f->get_16(), 0));
					}
				} break;
				default: {
					ERR_FAIL_V_MSG(ERR_CANT_CREATE, TTR("Invalid BMFont block type."));
				} break;
			}
			f->seek(off + block_size);
			block_type = f->get_8();
			block_size = f->get_32();
		}

	} else {
		// Text BMFont file.
		f->seek(0);
		while (true) {
			String line = f->get_line();

			int delimiter = line.find(" ");
			String type = line.substr(0, delimiter);
			int pos = delimiter + 1;
			Map<String, String> keys;

			while (pos < line.size() && line[pos] == ' ') {
				pos++;
			}

			while (pos < line.size()) {
				int eq = line.find("=", pos);
				if (eq == -1) {
					break;
				}
				String key = line.substr(pos, eq - pos);
				int end = -1;
				String value;
				if (line[eq + 1] == '"') {
					end = line.find("\"", eq + 2);
					if (end == -1) {
						break;
					}
					value = line.substr(eq + 2, end - 1 - eq - 1);
					pos = end + 1;
				} else {
					end = line.find(" ", eq + 1);
					if (end == -1) {
						end = line.size();
					}
					value = line.substr(eq + 1, end - eq);
					pos = end;
				}

				while (pos < line.size() && line[pos] == ' ') {
					pos++;
				}

				keys[key] = value;
			}

			if (type == "info") {
				if (keys.has("size")) {
					base_size = keys["size"].to_int();
					font->set_fixed_size(base_size);
				}
				if (keys.has("outline")) {
					outline = keys["outline"].to_int();
				}
				if (keys.has("bold")) {
					if (keys["bold"].to_int()) {
						st_flags |= TextServer::FONT_BOLD;
					}
				}
				if (keys.has("italic")) {
					if (keys["italic"].to_int()) {
						st_flags |= TextServer::FONT_ITALIC;
					}
				}
				if (keys.has("face")) {
					font_name = keys["face"];
				}
				ERR_FAIL_COND_V_MSG((!keys.has("unicode") || keys["unicode"].to_int() != 1), ERR_CANT_CREATE, TTR("Non-unicode version of BMFont is not supported."));
			} else if (type == "common") {
				if (keys.has("lineHeight")) {
					height = keys["lineHeight"].to_int();
				}
				if (keys.has("base")) {
					ascent = keys["base"].to_int();
				}
				if (keys.has("packed")) {
					packed = (keys["packed"].to_int() == 1);
				}
				if (keys.has("alphaChnl")) {
					ch[3] = keys["alphaChnl"].to_int();
				}
				if (keys.has("redChnl")) {
					ch[0] = keys["redChnl"].to_int();
				}
				if (keys.has("greenChnl")) {
					ch[1] = keys["greenChnl"].to_int();
				}
				if (keys.has("blueChnl")) {
					ch[2] = keys["blueChnl"].to_int();
				}
				for (int i = 0; i < 4; i++) {
					if (ch[i] == 0 && first_gl_ch == -1) {
						first_gl_ch = i;
					}
					if (ch[i] == 1 && first_ol_ch == -1) {
						first_ol_ch = i;
					}
					if (ch[i] == 2 && first_cm_ch == -1) {
						first_cm_ch = i;
					}
				}
			} else if (type == "page") {
				int page = 0;
				if (keys.has("id")) {
					page = keys["id"].to_int();
				}
				if (keys.has("file")) {
					String base_dir = p_source_file.get_base_dir();
					String file = base_dir.plus_file(keys["file"]);
					if (RenderingServer::get_singleton() != nullptr) {
						Ref<Image> img;
						img.instantiate();
						Error err = ImageLoader::load_image(file, img);
						ERR_FAIL_COND_V_MSG(err != OK, ERR_FILE_CANT_READ, TTR("Can't load font texture: ") + "\"" + file + "\".");
						if (packed) {
							if (ch[3] == 0) { // 4 x 8 bit monochrome, no outline
								outline = 0;
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
								_convert_packed_8bit(font, img, page, base_size);
							} else if ((ch[3] == 2) && (outline > 0)) { // 4 x 4 bit monochrome, gl + outline
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
								_convert_packed_4bit(font, img, page, base_size);
							} else {
								ERR_FAIL_V_MSG(ERR_CANT_CREATE, TTR("Unsupported BMFont texture format."));
							}
						} else {
							if ((ch[0] == 0) && (ch[1] == 0) && (ch[2] == 0) && (ch[3] == 0)) { // RGBA8 color, no outline
								outline = 0;
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
								font->set_texture_image(0, Vector2i(base_size, 0), page, img);
							} else if ((ch[0] == 2) && (ch[1] == 2) && (ch[2] == 2) && (ch[3] == 2) && (outline > 0)) { // RGBA4 color, gl + outline
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
								_convert_rgba_4bit(font, img, page, base_size);
							} else if ((first_gl_ch >= 0) && (first_ol_ch >= 0) && (outline > 0)) { // 1 x 8 bit monochrome, gl + outline
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
								_convert_mono_8bit(font, img, page, first_gl_ch, base_size, 0);
								_convert_mono_8bit(font, img, page, first_ol_ch, base_size, 1);
							} else if ((first_cm_ch >= 0) && (outline > 0)) { // 1 x 4 bit monochrome, gl + outline
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
								_convert_mono_4bit(font, img, page, first_cm_ch, base_size, 1);
							} else if (first_gl_ch >= 0) { // 1 x 8 bit monochrome, no outline
								outline = 0;
								ERR_FAIL_COND_V_MSG(img->get_format() != Image::FORMAT_RGBA8 && img->get_format() != Image::FORMAT_L8, ERR_FILE_CANT_READ, TTR("Unsupported BMFont texture format."));
								_convert_mono_8bit(font, img, page, first_gl_ch, base_size, 0);
							} else {
								ERR_FAIL_V_MSG(ERR_CANT_CREATE, TTR("Unsupported BMFont texture format."));
							}
						}
					}
				}
			} else if (type == "char") {
				char32_t idx = 0;
				Vector2 advance;
				Vector2 size;
				Vector2 offset;
				Rect2 uv_rect;
				int texture_idx = -1;
				uint8_t channel = 15;

				if (keys.has("id")) {
					idx = keys["id"].to_int();
				}
				if (keys.has("x")) {
					uv_rect.position.x = keys["x"].to_int();
				}
				if (keys.has("y")) {
					uv_rect.position.y = keys["y"].to_int();
				}
				if (keys.has("width")) {
					uv_rect.size.width = keys["width"].to_int();
					size.width = keys["width"].to_int();
				}
				if (keys.has("height")) {
					uv_rect.size.height = keys["height"].to_int();
					size.height = keys["height"].to_int();
				}
				if (keys.has("xoffset")) {
					offset.x = keys["xoffset"].to_int();
				}
				if (keys.has("yoffset")) {
					offset.y = keys["yoffset"].to_int() - ascent;
				}
				if (keys.has("page")) {
					texture_idx = keys["page"].to_int();
				}
				if (keys.has("xadvance")) {
					advance.x = keys["xadvance"].to_int();
				}
				if (advance.x < 0) {
					advance.x = size.width + 1;
				}
				if (keys.has("chnl")) {
					channel = keys["chnl"].to_int();
				}

				ERR_FAIL_COND_V_MSG(!packed && channel != 15, ERR_CANT_CREATE, TTR("Invalid glyph channel."));
				int ch_off = 0;
				switch (channel) {
					case 1:
						ch_off = 2;
						break; // B
					case 2:
						ch_off = 1;
						break; // G
					case 4:
						ch_off = 0;
						break; // R
					case 8:
						ch_off = 3;
						break; // A
					default:
						ch_off = 0;
						break;
				}
				font->set_glyph_advance(0, base_size, idx, advance);
				font->set_glyph_offset(0, Vector2i(base_size, 0), idx, offset);
				font->set_glyph_size(0, Vector2i(base_size, 0), idx, size);
				font->set_glyph_uv_rect(0, Vector2i(base_size, 0), idx, uv_rect);
				font->set_glyph_texture_idx(0, Vector2i(base_size, 0), idx, texture_idx * (packed ? 4 : 1) + ch_off);
				if (outline > 0) {
					font->set_glyph_offset(0, Vector2i(base_size, 1), idx, offset);
					font->set_glyph_size(0, Vector2i(base_size, 1), idx, size);
					font->set_glyph_uv_rect(0, Vector2i(base_size, 1), idx, uv_rect);
					font->set_glyph_texture_idx(0, Vector2i(base_size, 1), idx, texture_idx * (packed ? 4 : 1) + ch_off);
				}
			} else if (type == "kerning") {
				Vector2i kpk;
				if (keys.has("first")) {
					kpk.x = keys["first"].to_int();
				}
				if (keys.has("second")) {
					kpk.y = keys["second"].to_int();
				}
				if (keys.has("amount")) {
					font->set_kerning(0, base_size, kpk, Vector2(keys["amount"].to_int(), 0));
				}
			}

			if (f->eof_reached()) {
				break;
			}
		}
	}

	font->set_font_name(font_name);
	font->set_font_style(st_flags);
	font->set_ascent(0, base_size, ascent);
	font->set_descent(0, base_size, height - ascent);

	int flg = ResourceSaver::SaverFlags::FLAG_BUNDLE_RESOURCES | ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS;
	if ((bool)p_options["compress"]) {
		flg |= ResourceSaver::SaverFlags::FLAG_COMPRESS;
	}

	print_verbose("Saving to: " + p_save_path + ".fontdata");
	Error err = ResourceSaver::save(p_save_path + ".fontdata", font, flg);
	ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot save font to file \"" + p_save_path + ".res\".");
	print_verbose("Done saving to: " + p_save_path + ".fontdata");
	return OK;
}

ResourceImporterBMFont::ResourceImporterBMFont() {
}
