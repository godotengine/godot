/*************************************************************************/
/*  text_server_fb.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "text_server_fb.h"

#include "core/error/error_macros.h"
#include "core/string/print_string.h"

#ifdef MODULE_MSDFGEN_ENABLED
#include "core/ShapeDistanceFinder.h"
#include "core/contour-combiners.h"
#include "core/edge-selectors.h"
#include "msdfgen.h"
#endif

/*************************************************************************/
/*  Character properties.                                                */
/*************************************************************************/

_FORCE_INLINE_ bool is_control(char32_t p_char) {
	return (p_char <= 0x001f) || (p_char >= 0x007f && p_char <= 0x009F);
}

_FORCE_INLINE_ bool is_whitespace(char32_t p_char) {
	return (p_char == 0x0020) || (p_char == 0x00A0) || (p_char == 0x1680) || (p_char >= 0x2000 && p_char <= 0x200a) || (p_char == 0x202f) || (p_char == 0x205f) || (p_char == 0x3000) || (p_char == 0x2028) || (p_char == 0x2029) || (p_char >= 0x0009 && p_char <= 0x000d) || (p_char == 0x0085);
}

_FORCE_INLINE_ bool is_linebreak(char32_t p_char) {
	return (p_char >= 0x000a && p_char <= 0x000d) || (p_char == 0x0085) || (p_char == 0x2028) || (p_char == 0x2029);
}

_FORCE_INLINE_ bool is_punct(char32_t p_char) {
	return (p_char >= 0x0020 && p_char <= 0x002F) || (p_char >= 0x003A && p_char <= 0x0040) || (p_char >= 0x005B && p_char <= 0x005E) || (p_char == 0x0060) || (p_char >= 0x007B && p_char <= 0x007E) || (p_char >= 0x2000 && p_char <= 0x206F) || (p_char >= 0x3000 && p_char <= 0x303F);
}

_FORCE_INLINE_ bool is_underscore(char32_t p_char) {
	return (p_char == 0x005F);
}

/*************************************************************************/

String TextServerFallback::interface_name = "Fallback";
uint32_t TextServerFallback::interface_features = 0; // Nothing is supported.

bool TextServerFallback::has_feature(Feature p_feature) const {
	return (interface_features & p_feature) == p_feature;
}

String TextServerFallback::get_name() const {
	return interface_name;
}

uint32_t TextServerFallback::get_features() const {
	return interface_features;
}

void TextServerFallback::free(RID p_rid) {
	_THREAD_SAFE_METHOD_
	if (font_owner.owns(p_rid)) {
		FontDataFallback *fd = font_owner.get_or_null(p_rid);
		font_owner.free(p_rid);
		memdelete(fd);
	} else if (shaped_owner.owns(p_rid)) {
		ShapedTextData *sd = shaped_owner.get_or_null(p_rid);
		shaped_owner.free(p_rid);
		memdelete(sd);
	}
}

bool TextServerFallback::has(RID p_rid) {
	_THREAD_SAFE_METHOD_
	return font_owner.owns(p_rid) || shaped_owner.owns(p_rid);
}

bool TextServerFallback::load_support_data(const String &p_filename) {
	return false; // No extra data used.
}

bool TextServerFallback::save_support_data(const String &p_filename) const {
	return false; // No extra data used.
}

bool TextServerFallback::is_locale_right_to_left(const String &p_locale) const {
	return false; // No RTL support.
}

void TextServerFallback::_insert_feature_sets() {
	// Registered OpenType variation tag.
	feature_sets.insert("italic", OT_TAG('i', 't', 'a', 'l'));
	feature_sets.insert("optical_size", OT_TAG('o', 'p', 's', 'z'));
	feature_sets.insert("slant", OT_TAG('s', 'l', 'n', 't'));
	feature_sets.insert("width", OT_TAG('w', 'd', 't', 'h'));
	feature_sets.insert("weight", OT_TAG('w', 'g', 'h', 't'));
}

_FORCE_INLINE_ int32_t ot_tag_from_string(const char *p_str, int p_len) {
	char tag[4];
	uint32_t i;

	if (!p_str || !p_len || !*p_str)
		return OT_TAG(0, 0, 0, 0);

	if (p_len < 0 || p_len > 4) {
		p_len = 4;
	}
	for (i = 0; i < (uint32_t)p_len && p_str[i]; i++) {
		tag[i] = p_str[i];
	}

	for (; i < 4; i++) {
		tag[i] = ' ';
	}

	return OT_TAG(tag[0], tag[1], tag[2], tag[3]);
}

int32_t TextServerFallback::name_to_tag(const String &p_name) const {
	if (feature_sets.has(p_name)) {
		return feature_sets[p_name];
	}

	// No readable name, use tag string.
	return ot_tag_from_string(p_name.replace("custom_", "").ascii().get_data(), -1);
}

_FORCE_INLINE_ void ot_tag_to_string(int32_t p_tag, char *p_buf) {
	p_buf[0] = (char)(uint8_t)(p_tag >> 24);
	p_buf[1] = (char)(uint8_t)(p_tag >> 16);
	p_buf[2] = (char)(uint8_t)(p_tag >> 8);
	p_buf[3] = (char)(uint8_t)(p_tag >> 0);
}

String TextServerFallback::tag_to_name(int32_t p_tag) const {
	for (const KeyValue<StringName, int32_t> &E : feature_sets) {
		if (E.value == p_tag) {
			return E.key;
		}
	}

	// No readable name, use tag string.
	char name[5];
	memset(name, 0, 5);
	ot_tag_to_string(p_tag, name);
	return String("custom_") + String(name);
}

/*************************************************************************/
/* Font Glyph Rendering                                                  */
/*************************************************************************/

_FORCE_INLINE_ TextServerFallback::FontTexturePosition TextServerFallback::find_texture_pos_for_glyph(FontDataForSizeFallback *p_data, int p_color_size, Image::Format p_image_format, int p_width, int p_height) const {
	FontTexturePosition ret;
	ret.index = -1;

	int mw = p_width;
	int mh = p_height;

	for (int i = 0; i < p_data->textures.size(); i++) {
		const FontTexture &ct = p_data->textures[i];

		if (RenderingServer::get_singleton() != nullptr) {
			if (ct.texture->get_format() != p_image_format) {
				continue;
			}
		}

		if (mw > ct.texture_w || mh > ct.texture_h) { // Too big for this texture.
			continue;
		}

		if (ct.offsets.size() < ct.texture_w) {
			continue;
		}

		ret.y = 0x7FFFFFFF;
		ret.x = 0;

		for (int j = 0; j < ct.texture_w - mw; j++) {
			int max_y = 0;

			for (int k = j; k < j + mw; k++) {
				int y = ct.offsets[k];
				if (y > max_y) {
					max_y = y;
				}
			}

			if (max_y < ret.y) {
				ret.y = max_y;
				ret.x = j;
			}
		}

		if (ret.y == 0x7FFFFFFF || ret.y + mh > ct.texture_h) {
			continue; // Fail, could not fit it here.
		}

		ret.index = i;
		break;
	}

	if (ret.index == -1) {
		// Could not find texture to fit, create one.
		ret.x = 0;
		ret.y = 0;

		int texsize = MAX(p_data->size.x * p_data->oversampling * 8, 256);
		if (mw > texsize) {
			texsize = mw; // Special case, adapt to it?
		}
		if (mh > texsize) {
			texsize = mh; // Special case, adapt to it?
		}

		texsize = next_power_of_2(texsize);

		texsize = MIN(texsize, 4096);

		FontTexture tex;
		tex.texture_w = texsize;
		tex.texture_h = texsize;
		tex.format = p_image_format;
		tex.imgdata.resize(texsize * texsize * p_color_size);

		{
			// Zero texture.
			uint8_t *w = tex.imgdata.ptrw();
			ERR_FAIL_COND_V(texsize * texsize * p_color_size > tex.imgdata.size(), ret);
			// Initialize the texture to all-white pixels to prevent artifacts when the
			// font is displayed at a non-default scale with filtering enabled.
			if (p_color_size == 2) {
				for (int i = 0; i < texsize * texsize * p_color_size; i += 2) { // FORMAT_LA8, BW font.
					w[i + 0] = 255;
					w[i + 1] = 0;
				}
			} else if (p_color_size == 4) {
				for (int i = 0; i < texsize * texsize * p_color_size; i += 4) { // FORMAT_RGBA8, Color font, Multichannel(+True) SDF.
					w[i + 0] = 255;
					w[i + 1] = 255;
					w[i + 2] = 255;
					w[i + 3] = 0;
				}
			} else {
				ERR_FAIL_V(ret);
			}
		}
		tex.offsets.resize(texsize);
		for (int i = 0; i < texsize; i++) { // Zero offsets.
			tex.offsets.write[i] = 0;
		}

		p_data->textures.push_back(tex);
		ret.index = p_data->textures.size() - 1;
	}

	return ret;
}

#ifdef MODULE_MSDFGEN_ENABLED

struct MSContext {
	msdfgen::Point2 position;
	msdfgen::Shape *shape;
	msdfgen::Contour *contour;
};

class DistancePixelConversion {
	double invRange;

public:
	_FORCE_INLINE_ explicit DistancePixelConversion(double range) :
			invRange(1 / range) {}
	_FORCE_INLINE_ void operator()(float *pixels, const msdfgen::MultiAndTrueDistance &distance) const {
		pixels[0] = float(invRange * distance.r + .5);
		pixels[1] = float(invRange * distance.g + .5);
		pixels[2] = float(invRange * distance.b + .5);
		pixels[3] = float(invRange * distance.a + .5);
	}
};

struct MSDFThreadData {
	msdfgen::Bitmap<float, 4> *output;
	msdfgen::Shape *shape;
	msdfgen::Projection *projection;
	DistancePixelConversion *distancePixelConversion;
};

static msdfgen::Point2 ft_point2(const FT_Vector &vector) {
	return msdfgen::Point2(vector.x / 60.0f, vector.y / 60.0f);
}

static int ft_move_to(const FT_Vector *to, void *user) {
	MSContext *context = reinterpret_cast<MSContext *>(user);
	if (!(context->contour && context->contour->edges.empty())) {
		context->contour = &context->shape->addContour();
	}
	context->position = ft_point2(*to);
	return 0;
}

static int ft_line_to(const FT_Vector *to, void *user) {
	MSContext *context = reinterpret_cast<MSContext *>(user);
	msdfgen::Point2 endpoint = ft_point2(*to);
	if (endpoint != context->position) {
		context->contour->addEdge(new msdfgen::LinearSegment(context->position, endpoint));
		context->position = endpoint;
	}
	return 0;
}

static int ft_conic_to(const FT_Vector *control, const FT_Vector *to, void *user) {
	MSContext *context = reinterpret_cast<MSContext *>(user);
	context->contour->addEdge(new msdfgen::QuadraticSegment(context->position, ft_point2(*control), ft_point2(*to)));
	context->position = ft_point2(*to);
	return 0;
}

static int ft_cubic_to(const FT_Vector *control1, const FT_Vector *control2, const FT_Vector *to, void *user) {
	MSContext *context = reinterpret_cast<MSContext *>(user);
	context->contour->addEdge(new msdfgen::CubicSegment(context->position, ft_point2(*control1), ft_point2(*control2), ft_point2(*to)));
	context->position = ft_point2(*to);
	return 0;
}

void TextServerFallback::_generateMTSDF_threaded(uint32_t y, void *p_td) const {
	MSDFThreadData *td = (MSDFThreadData *)p_td;

	msdfgen::ShapeDistanceFinder<msdfgen::OverlappingContourCombiner<msdfgen::MultiAndTrueDistanceSelector>> distanceFinder(*td->shape);
	int row = td->shape->inverseYAxis ? td->output->height() - y - 1 : y;
	for (int col = 0; col < td->output->width(); ++col) {
		int x = (y % 2) ? td->output->width() - col - 1 : col;
		msdfgen::Point2 p = td->projection->unproject(msdfgen::Point2(x + .5, y + .5));
		msdfgen::MultiAndTrueDistance distance = distanceFinder.distance(p);
		td->distancePixelConversion->operator()(td->output->operator()(x, row), distance);
	}
}

_FORCE_INLINE_ TextServerFallback::FontGlyph TextServerFallback::rasterize_msdf(FontDataFallback *p_font_data, FontDataForSizeFallback *p_data, int p_pixel_range, int p_rect_margin, FT_Outline *outline, const Vector2 &advance) const {
	msdfgen::Shape shape;

	shape.contours.clear();
	shape.inverseYAxis = false;

	MSContext context = {};
	context.shape = &shape;
	FT_Outline_Funcs ft_functions;
	ft_functions.move_to = &ft_move_to;
	ft_functions.line_to = &ft_line_to;
	ft_functions.conic_to = &ft_conic_to;
	ft_functions.cubic_to = &ft_cubic_to;
	ft_functions.shift = 0;
	ft_functions.delta = 0;

	int error = FT_Outline_Decompose(outline, &ft_functions, &context);
	ERR_FAIL_COND_V_MSG(error, FontGlyph(), "FreeType: Outline decomposition error: '" + String(FT_Error_String(error)) + "'.");
	if (!shape.contours.empty() && shape.contours.back().edges.empty()) {
		shape.contours.pop_back();
	}

	if (FT_Outline_Get_Orientation(outline) == 1) {
		for (int i = 0; i < (int)shape.contours.size(); ++i) {
			shape.contours[i].reverse();
		}
	}

	shape.inverseYAxis = true;
	shape.normalize();

	msdfgen::Shape::Bounds bounds = shape.getBounds(p_pixel_range);

	FontGlyph chr;
	chr.found = true;
	chr.advance = advance.round();

	if (shape.validate() && shape.contours.size() > 0) {
		int w = (bounds.r - bounds.l);
		int h = (bounds.t - bounds.b);

		int mw = w + p_rect_margin * 2;
		int mh = h + p_rect_margin * 2;

		ERR_FAIL_COND_V(mw > 4096, FontGlyph());
		ERR_FAIL_COND_V(mh > 4096, FontGlyph());

		FontTexturePosition tex_pos = find_texture_pos_for_glyph(p_data, 4, Image::FORMAT_RGBA8, mw, mh);
		ERR_FAIL_COND_V(tex_pos.index < 0, FontGlyph());
		FontTexture &tex = p_data->textures.write[tex_pos.index];

		edgeColoringSimple(shape, 3.0); // Max. angle.
		msdfgen::Bitmap<float, 4> image(w, h); // Texture size.

		DistancePixelConversion distancePixelConversion(p_pixel_range);
		msdfgen::Projection projection(msdfgen::Vector2(1.0, 1.0), msdfgen::Vector2(-bounds.l, -bounds.b));
		msdfgen::MSDFGeneratorConfig config(true, msdfgen::ErrorCorrectionConfig());

		MSDFThreadData td;
		td.output = &image;
		td.shape = &shape;
		td.projection = &projection;
		td.distancePixelConversion = &distancePixelConversion;

		if (p_font_data->work_pool.get_thread_count() == 0) {
			p_font_data->work_pool.init();
		}
		p_font_data->work_pool.do_work(h, this, &TextServerFallback::_generateMTSDF_threaded, &td);

		msdfgen::msdfErrorCorrection(image, shape, projection, p_pixel_range, config);

		{
			uint8_t *wr = tex.imgdata.ptrw();

			for (int i = 0; i < h; i++) {
				for (int j = 0; j < w; j++) {
					int ofs = ((i + tex_pos.y + p_rect_margin) * tex.texture_w + j + tex_pos.x + p_rect_margin) * 4;
					ERR_FAIL_COND_V(ofs >= tex.imgdata.size(), FontGlyph());
					wr[ofs + 0] = (uint8_t)(CLAMP(image(j, i)[0] * 256.f, 0.f, 255.f));
					wr[ofs + 1] = (uint8_t)(CLAMP(image(j, i)[1] * 256.f, 0.f, 255.f));
					wr[ofs + 2] = (uint8_t)(CLAMP(image(j, i)[2] * 256.f, 0.f, 255.f));
					wr[ofs + 3] = (uint8_t)(CLAMP(image(j, i)[3] * 256.f, 0.f, 255.f));
				}
			}
		}

		// Blit to image and texture.
		{
			if (RenderingServer::get_singleton() != nullptr) {
				Ref<Image> img = memnew(Image(tex.texture_w, tex.texture_h, 0, Image::FORMAT_RGBA8, tex.imgdata));
				if (tex.texture.is_null()) {
					tex.texture.instantiate();
					tex.texture->create_from_image(img);
				} else {
					tex.texture->update(img);
				}
			}
		}

		// Update height array.
		for (int k = tex_pos.x; k < tex_pos.x + mw; k++) {
			tex.offsets.write[k] = tex_pos.y + mh;
		}

		chr.texture_idx = tex_pos.index;

		chr.uv_rect = Rect2(tex_pos.x + p_rect_margin, tex_pos.y + p_rect_margin, w, h);
		chr.rect.position = Vector2(bounds.l, -bounds.t);
		chr.rect.size = chr.uv_rect.size;
	}
	return chr;
}
#endif

#ifdef MODULE_FREETYPE_ENABLED
_FORCE_INLINE_ TextServerFallback::FontGlyph TextServerFallback::rasterize_bitmap(FontDataForSizeFallback *p_data, int p_rect_margin, FT_Bitmap bitmap, int yofs, int xofs, const Vector2 &advance) const {
	int w = bitmap.width;
	int h = bitmap.rows;

	int mw = w + p_rect_margin * 2;
	int mh = h + p_rect_margin * 2;

	ERR_FAIL_COND_V(mw > 4096, FontGlyph());
	ERR_FAIL_COND_V(mh > 4096, FontGlyph());

	int color_size = bitmap.pixel_mode == FT_PIXEL_MODE_BGRA ? 4 : 2;
	Image::Format require_format = color_size == 4 ? Image::FORMAT_RGBA8 : Image::FORMAT_LA8;

	FontTexturePosition tex_pos = find_texture_pos_for_glyph(p_data, color_size, require_format, mw, mh);
	ERR_FAIL_COND_V(tex_pos.index < 0, FontGlyph());

	// Fit character in char texture.

	FontTexture &tex = p_data->textures.write[tex_pos.index];

	{
		uint8_t *wr = tex.imgdata.ptrw();

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				int ofs = ((i + tex_pos.y + p_rect_margin) * tex.texture_w + j + tex_pos.x + p_rect_margin) * color_size;
				ERR_FAIL_COND_V(ofs >= tex.imgdata.size(), FontGlyph());
				switch (bitmap.pixel_mode) {
					case FT_PIXEL_MODE_MONO: {
						int byte = i * bitmap.pitch + (j >> 3);
						int bit = 1 << (7 - (j % 8));
						wr[ofs + 0] = 255; // grayscale as 1
						wr[ofs + 1] = (bitmap.buffer[byte] & bit) ? 255 : 0;
					} break;
					case FT_PIXEL_MODE_GRAY:
						wr[ofs + 0] = 255; // grayscale as 1
						wr[ofs + 1] = bitmap.buffer[i * bitmap.pitch + j];
						break;
					case FT_PIXEL_MODE_BGRA: {
						int ofs_color = i * bitmap.pitch + (j << 2);
						wr[ofs + 2] = bitmap.buffer[ofs_color + 0];
						wr[ofs + 1] = bitmap.buffer[ofs_color + 1];
						wr[ofs + 0] = bitmap.buffer[ofs_color + 2];
						wr[ofs + 3] = bitmap.buffer[ofs_color + 3];
					} break;
					default:
						ERR_FAIL_V_MSG(FontGlyph(), "Font uses unsupported pixel format: " + itos(bitmap.pixel_mode) + ".");
						break;
				}
			}
		}
	}

	// Blit to image and texture.
	{
		if (RenderingServer::get_singleton() != nullptr) {
			Ref<Image> img = memnew(Image(tex.texture_w, tex.texture_h, 0, require_format, tex.imgdata));

			if (tex.texture.is_null()) {
				tex.texture.instantiate();
				tex.texture->create_from_image(img);
			} else {
				tex.texture->update(img);
			}
		}
	}

	// Update height array.
	for (int k = tex_pos.x; k < tex_pos.x + mw; k++) {
		tex.offsets.write[k] = tex_pos.y + mh;
	}

	FontGlyph chr;
	chr.advance = (advance * p_data->scale / p_data->oversampling).round();
	chr.texture_idx = tex_pos.index;
	chr.found = true;

	chr.uv_rect = Rect2(tex_pos.x + p_rect_margin, tex_pos.y + p_rect_margin, w, h);
	chr.rect.position = (Vector2(xofs, -yofs) * p_data->scale / p_data->oversampling).round();
	chr.rect.size = chr.uv_rect.size * p_data->scale / p_data->oversampling;
	return chr;
}
#endif

/*************************************************************************/
/* Font Cache                                                            */
/*************************************************************************/

_FORCE_INLINE_ bool TextServerFallback::_ensure_glyph(FontDataFallback *p_font_data, const Vector2i &p_size, int32_t p_glyph) const {
	ERR_FAIL_COND_V(!_ensure_cache_for_size(p_font_data, p_size), false);

	FontDataForSizeFallback *fd = p_font_data->cache[p_size];
	if (fd->glyph_map.has(p_glyph)) {
		return fd->glyph_map[p_glyph].found;
	}

	if (p_glyph == 0) { // Non graphical or invalid glyph, do not render.
		fd->glyph_map[p_glyph] = FontGlyph();
		return true;
	}

#ifdef MODULE_FREETYPE_ENABLED
	FontGlyph gl;
	if (fd->face) {
		FT_Int32 flags = FT_LOAD_DEFAULT;

		bool outline = p_size.y > 0;
		switch (p_font_data->hinting) {
			case TextServer::HINTING_NONE:
				flags |= FT_LOAD_NO_HINTING;
				break;
			case TextServer::HINTING_LIGHT:
				flags |= FT_LOAD_TARGET_LIGHT;
				break;
			default:
				flags |= FT_LOAD_TARGET_NORMAL;
				break;
		}
		if (p_font_data->force_autohinter) {
			flags |= FT_LOAD_FORCE_AUTOHINT;
		}
		if (outline) {
			flags |= FT_LOAD_NO_BITMAP;
		} else if (FT_HAS_COLOR(fd->face)) {
			flags |= FT_LOAD_COLOR;
		}

		int32_t glyph_index = FT_Get_Char_Index(fd->face, p_glyph);

		FT_Fixed v, h;
		FT_Get_Advance(fd->face, glyph_index, flags, &h);
		FT_Get_Advance(fd->face, glyph_index, flags | FT_LOAD_VERTICAL_LAYOUT, &v);

		int error = FT_Load_Glyph(fd->face, glyph_index, flags);
		if (error) {
			fd->glyph_map[p_glyph] = FontGlyph();
			return false;
		}

		if (!outline) {
			if (!p_font_data->msdf) {
				error = FT_Render_Glyph(fd->face->glyph, p_font_data->antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO);
			}
			FT_GlyphSlot slot = fd->face->glyph;
			if (!error) {
				if (p_font_data->msdf) {
#ifdef MODULE_MSDFGEN_ENABLED
					gl = rasterize_msdf(p_font_data, fd, p_font_data->msdf_range, rect_range, &slot->outline, Vector2((h + (1 << 9)) >> 10, (v + (1 << 9)) >> 10) / 64.0);
#else
					fd->glyph_map[p_glyph] = FontGlyph();
					ERR_FAIL_V_MSG(false, "Compiled without MSDFGEN support!");
#endif
				} else {
					gl = rasterize_bitmap(fd, rect_range, slot->bitmap, slot->bitmap_top, slot->bitmap_left, Vector2((h + (1 << 9)) >> 10, (v + (1 << 9)) >> 10) / 64.0);
				}
			}
		} else {
			FT_Stroker stroker;
			if (FT_Stroker_New(library, &stroker) != 0) {
				fd->glyph_map[p_glyph] = FontGlyph();
				ERR_FAIL_V_MSG(false, "FreeType: Failed to load glyph stroker.");
			}

			FT_Stroker_Set(stroker, (int)(fd->size.y * fd->oversampling * 16.0), FT_STROKER_LINECAP_BUTT, FT_STROKER_LINEJOIN_ROUND, 0);
			FT_Glyph glyph;
			FT_BitmapGlyph glyph_bitmap;

			if (FT_Get_Glyph(fd->face->glyph, &glyph) != 0) {
				goto cleanup_stroker;
			}
			if (FT_Glyph_Stroke(&glyph, stroker, 1) != 0) {
				goto cleanup_glyph;
			}
			if (FT_Glyph_To_Bitmap(&glyph, p_font_data->antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO, nullptr, 1) != 0) {
				goto cleanup_glyph;
			}
			glyph_bitmap = (FT_BitmapGlyph)glyph;
			gl = rasterize_bitmap(fd, rect_range, glyph_bitmap->bitmap, glyph_bitmap->top, glyph_bitmap->left, Vector2());

		cleanup_glyph:
			FT_Done_Glyph(glyph);
		cleanup_stroker:
			FT_Stroker_Done(stroker);
		}
		fd->glyph_map[p_glyph] = gl;
		return gl.found;
	}
#endif
	fd->glyph_map[p_glyph] = FontGlyph();
	return false;
}

_FORCE_INLINE_ bool TextServerFallback::_ensure_cache_for_size(FontDataFallback *p_font_data, const Vector2i &p_size) const {
	ERR_FAIL_COND_V(p_size.x <= 0, false);
	if (p_font_data->cache.has(p_size)) {
		return true;
	}

	FontDataForSizeFallback *fd = memnew(FontDataForSizeFallback);
	fd->size = p_size;
	if (p_font_data->data_ptr && (p_font_data->data_size > 0)) {
		// Init dynamic font.
#ifdef MODULE_FREETYPE_ENABLED
		int error = 0;
		if (!library) {
			error = FT_Init_FreeType(&library);
			ERR_FAIL_COND_V_MSG(error != 0, false, "FreeType: Error initializing library: '" + String(FT_Error_String(error)) + "'.");
		}

		memset(&fd->stream, 0, sizeof(FT_StreamRec));
		fd->stream.base = (unsigned char *)p_font_data->data_ptr;
		fd->stream.size = p_font_data->data_size;
		fd->stream.pos = 0;

		FT_Open_Args fargs;
		memset(&fargs, 0, sizeof(FT_Open_Args));
		fargs.memory_base = (unsigned char *)p_font_data->data_ptr;
		fargs.memory_size = p_font_data->data_size;
		fargs.flags = FT_OPEN_MEMORY;
		fargs.stream = &fd->stream;
		error = FT_Open_Face(library, &fargs, 0, &fd->face);
		if (error) {
			FT_Done_Face(fd->face);
			fd->face = nullptr;
			ERR_FAIL_V_MSG(false, "FreeType: Error loading font: '" + String(FT_Error_String(error)) + "'.");
		}

		if (p_font_data->msdf) {
			fd->oversampling = 1.0f;
			fd->size.x = p_font_data->msdf_source_size;
		} else if (p_font_data->oversampling <= 0.0f) {
			fd->oversampling = font_get_global_oversampling();
		} else {
			fd->oversampling = p_font_data->oversampling;
		}

		if (FT_HAS_COLOR(fd->face) && fd->face->num_fixed_sizes > 0) {
			int best_match = 0;
			int diff = ABS(fd->size.x - ((int64_t)fd->face->available_sizes[0].width));
			fd->scale = float(fd->size.x * fd->oversampling) / fd->face->available_sizes[0].width;
			for (int i = 1; i < fd->face->num_fixed_sizes; i++) {
				int ndiff = ABS(fd->size.x - ((int64_t)fd->face->available_sizes[i].width));
				if (ndiff < diff) {
					best_match = i;
					diff = ndiff;
					fd->scale = float(fd->size.x * fd->oversampling) / fd->face->available_sizes[i].width;
				}
			}
			FT_Select_Size(fd->face, best_match);
		} else {
			FT_Set_Pixel_Sizes(fd->face, 0, fd->size.x * fd->oversampling);
		}

		fd->ascent = (fd->face->size->metrics.ascender / 64.0) / fd->oversampling * fd->scale;
		fd->descent = (-fd->face->size->metrics.descender / 64.0) / fd->oversampling * fd->scale;
		fd->underline_position = (-FT_MulFix(fd->face->underline_position, fd->face->size->metrics.y_scale) / 64.0) / fd->oversampling * fd->scale;
		fd->underline_thickness = (FT_MulFix(fd->face->underline_thickness, fd->face->size->metrics.y_scale) / 64.0) / fd->oversampling * fd->scale;

		if (!p_font_data->face_init) {
			// Read OpenType variations.
			p_font_data->supported_varaitions.clear();
			if (fd->face->face_flags & FT_FACE_FLAG_MULTIPLE_MASTERS) {
				FT_MM_Var *amaster;
				FT_Get_MM_Var(fd->face, &amaster);
				for (FT_UInt i = 0; i < amaster->num_axis; i++) {
					p_font_data->supported_varaitions[(int32_t)amaster->axis[i].tag] = Vector3i(amaster->axis[i].minimum / 65536, amaster->axis[i].maximum / 65536, amaster->axis[i].def / 65536);
				}
				FT_Done_MM_Var(library, amaster);
			}
			p_font_data->face_init = true;
		}

		// Write variations.
		if (fd->face->face_flags & FT_FACE_FLAG_MULTIPLE_MASTERS) {
			FT_MM_Var *amaster;

			FT_Get_MM_Var(fd->face, &amaster);

			Vector<FT_Fixed> coords;
			coords.resize(amaster->num_axis);

			FT_Get_Var_Design_Coordinates(fd->face, coords.size(), coords.ptrw());

			for (FT_UInt i = 0; i < amaster->num_axis; i++) {
				// Reset to default.
				int32_t var_tag = amaster->axis[i].tag;
				float var_value = (double)amaster->axis[i].def / 65536.f;
				coords.write[i] = amaster->axis[i].def;

				if (p_font_data->variation_coordinates.has(var_tag)) {
					var_value = p_font_data->variation_coordinates[var_tag];
					coords.write[i] = CLAMP(var_value * 65536.f, amaster->axis[i].minimum, amaster->axis[i].maximum);
				}

				if (p_font_data->variation_coordinates.has(tag_to_name(var_tag))) {
					var_value = p_font_data->variation_coordinates[tag_to_name(var_tag)];
					coords.write[i] = CLAMP(var_value * 65536.f, amaster->axis[i].minimum, amaster->axis[i].maximum);
				}
			}

			FT_Set_Var_Design_Coordinates(fd->face, coords.size(), coords.ptrw());
			FT_Done_MM_Var(library, amaster);
		}
#else
		ERR_FAIL_V_MSG(false, "FreeType: Can't load dynamic font, engine is compiled without FreeType support!");
#endif
	}
	p_font_data->cache[p_size] = fd;
	return true;
}

_FORCE_INLINE_ void TextServerFallback::_font_clear_cache(FontDataFallback *p_font_data) {
	for (const KeyValue<Vector2i, FontDataForSizeFallback *> &E : p_font_data->cache) {
		memdelete(E.value);
	}

	p_font_data->cache.clear();
	p_font_data->face_init = false;
	p_font_data->supported_varaitions.clear();
}

RID TextServerFallback::create_font() {
	FontDataFallback *fd = memnew(FontDataFallback);

	return font_owner.make_rid(fd);
}

void TextServerFallback::font_set_data(RID p_font_rid, const PackedByteArray &p_data) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	_font_clear_cache(fd);
	fd->data = p_data;
	fd->data_ptr = fd->data.ptr();
	fd->data_size = fd->data.size();
}

void TextServerFallback::font_set_data_ptr(RID p_font_rid, const uint8_t *p_data_ptr, size_t p_data_size) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	_font_clear_cache(fd);
	fd->data.clear();
	fd->data_ptr = p_data_ptr;
	fd->data_size = p_data_size;
}

void TextServerFallback::font_set_antialiased(RID p_font_rid, bool p_antialiased) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	if (fd->antialiased != p_antialiased) {
		_font_clear_cache(fd);
		fd->antialiased = p_antialiased;
	}
}

bool TextServerFallback::font_is_antialiased(RID p_font_rid) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, false);

	MutexLock lock(fd->mutex);
	return fd->antialiased;
}

void TextServerFallback::font_set_multichannel_signed_distance_field(RID p_font_rid, bool p_msdf) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	if (fd->msdf != p_msdf) {
		_font_clear_cache(fd);
		fd->msdf = p_msdf;
	}
}

bool TextServerFallback::font_is_multichannel_signed_distance_field(RID p_font_rid) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, false);

	MutexLock lock(fd->mutex);
	return fd->msdf;
}

void TextServerFallback::font_set_msdf_pixel_range(RID p_font_rid, int p_msdf_pixel_range) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	if (fd->msdf_range != p_msdf_pixel_range) {
		_font_clear_cache(fd);
		fd->msdf_range = p_msdf_pixel_range;
	}
}

int TextServerFallback::font_get_msdf_pixel_range(RID p_font_rid) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, false);

	MutexLock lock(fd->mutex);
	return fd->msdf_range;
}

void TextServerFallback::font_set_msdf_size(RID p_font_rid, int p_msdf_size) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	if (fd->msdf_source_size != p_msdf_size) {
		_font_clear_cache(fd);
		fd->msdf_source_size = p_msdf_size;
	}
}

int TextServerFallback::font_get_msdf_size(RID p_font_rid) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, false);

	MutexLock lock(fd->mutex);
	return fd->msdf_source_size;
}

void TextServerFallback::font_set_fixed_size(RID p_font_rid, int p_fixed_size) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	if (fd->fixed_size != p_fixed_size) {
		fd->fixed_size = p_fixed_size;
	}
}

int TextServerFallback::font_get_fixed_size(RID p_font_rid) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, false);

	MutexLock lock(fd->mutex);
	return fd->fixed_size;
}

void TextServerFallback::font_set_force_autohinter(RID p_font_rid, bool p_force_autohinter) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	if (fd->force_autohinter != p_force_autohinter) {
		_font_clear_cache(fd);
		fd->force_autohinter = p_force_autohinter;
	}
}

bool TextServerFallback::font_is_force_autohinter(RID p_font_rid) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, false);

	MutexLock lock(fd->mutex);
	return fd->force_autohinter;
}

void TextServerFallback::font_set_hinting(RID p_font_rid, TextServer::Hinting p_hinting) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	if (fd->hinting != p_hinting) {
		_font_clear_cache(fd);
		fd->hinting = p_hinting;
	}
}

TextServer::Hinting TextServerFallback::font_get_hinting(RID p_font_rid) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, HINTING_NONE);

	MutexLock lock(fd->mutex);
	return fd->hinting;
}

void TextServerFallback::font_set_variation_coordinates(RID p_font_rid, const Dictionary &p_variation_coordinates) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	if (fd->variation_coordinates != p_variation_coordinates) {
		_font_clear_cache(fd);
		fd->variation_coordinates = p_variation_coordinates;
	}
}

Dictionary TextServerFallback::font_get_variation_coordinates(RID p_font_rid) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, Dictionary());

	MutexLock lock(fd->mutex);
	return fd->variation_coordinates;
}

void TextServerFallback::font_set_oversampling(RID p_font_rid, float p_oversampling) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	if (fd->oversampling != p_oversampling) {
		_font_clear_cache(fd);
		fd->oversampling = p_oversampling;
	}
}

float TextServerFallback::font_get_oversampling(RID p_font_rid) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, 0.f);

	MutexLock lock(fd->mutex);
	return fd->oversampling;
}

Array TextServerFallback::font_get_size_cache_list(RID p_font_rid) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, Array());

	MutexLock lock(fd->mutex);
	Array ret;
	for (const KeyValue<Vector2i, FontDataForSizeFallback *> &E : fd->cache) {
		ret.push_back(E.key);
	}
	return ret;
}

void TextServerFallback::font_clear_size_cache(RID p_font_rid) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	for (const KeyValue<Vector2i, FontDataForSizeFallback *> &E : fd->cache) {
		memdelete(E.value);
	}
	fd->cache.clear();
}

void TextServerFallback::font_remove_size_cache(RID p_font_rid, const Vector2i &p_size) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	if (fd->cache.has(p_size)) {
		memdelete(fd->cache[p_size]);
		fd->cache.erase(p_size);
	}
}

void TextServerFallback::font_set_ascent(RID p_font_rid, int p_size, float p_ascent) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	fd->cache[size]->ascent = p_ascent;
}

float TextServerFallback::font_get_ascent(RID p_font_rid, int p_size) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, 0.f);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), 0.f);

	if (fd->msdf) {
		return fd->cache[size]->ascent * (float)p_size / (float)fd->msdf_source_size;
	} else {
		return fd->cache[size]->ascent;
	}
}

void TextServerFallback::font_set_descent(RID p_font_rid, int p_size, float p_descent) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	fd->cache[size]->descent = p_descent;
}

float TextServerFallback::font_get_descent(RID p_font_rid, int p_size) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, 0.f);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), 0.f);

	if (fd->msdf) {
		return fd->cache[size]->descent * (float)p_size / (float)fd->msdf_source_size;
	} else {
		return fd->cache[size]->descent;
	}
}

void TextServerFallback::font_set_underline_position(RID p_font_rid, int p_size, float p_underline_position) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	fd->cache[size]->underline_position = p_underline_position;
}

float TextServerFallback::font_get_underline_position(RID p_font_rid, int p_size) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, 0.f);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), 0.f);

	if (fd->msdf) {
		return fd->cache[size]->underline_position * (float)p_size / (float)fd->msdf_source_size;
	} else {
		return fd->cache[size]->underline_position;
	}
}

void TextServerFallback::font_set_underline_thickness(RID p_font_rid, int p_size, float p_underline_thickness) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	fd->cache[size]->underline_thickness = p_underline_thickness;
}

float TextServerFallback::font_get_underline_thickness(RID p_font_rid, int p_size) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, 0.f);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), 0.f);

	if (fd->msdf) {
		return fd->cache[size]->underline_thickness * (float)p_size / (float)fd->msdf_source_size;
	} else {
		return fd->cache[size]->underline_thickness;
	}
}

void TextServerFallback::font_set_scale(RID p_font_rid, int p_size, float p_scale) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	fd->cache[size]->scale = p_scale;
}

float TextServerFallback::font_get_scale(RID p_font_rid, int p_size) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, 0.f);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), 0.f);

	if (fd->msdf) {
		return fd->cache[size]->scale * (float)p_size / (float)fd->msdf_source_size;
	} else {
		return fd->cache[size]->scale / fd->cache[size]->oversampling;
	}
}

void TextServerFallback::font_set_spacing(RID p_font_rid, int p_size, TextServer::SpacingType p_spacing, int p_value) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	switch (p_spacing) {
		case TextServer::SPACING_GLYPH: {
			fd->cache[size]->spacing_glyph = p_value;
		} break;
		case TextServer::SPACING_SPACE: {
			fd->cache[size]->spacing_space = p_value;
		} break;
		default: {
			ERR_FAIL_MSG("Invalid spacing type: " + itos(p_spacing));
		} break;
	}
}

int TextServerFallback::font_get_spacing(RID p_font_rid, int p_size, TextServer::SpacingType p_spacing) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, 0);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), 0);

	switch (p_spacing) {
		case TextServer::SPACING_GLYPH: {
			if (fd->msdf) {
				return fd->cache[size]->spacing_glyph * (float)p_size / (float)fd->msdf_source_size;
			} else {
				return fd->cache[size]->spacing_glyph;
			}
		} break;
		case TextServer::SPACING_SPACE: {
			if (fd->msdf) {
				return fd->cache[size]->spacing_space * (float)p_size / (float)fd->msdf_source_size;
			} else {
				return fd->cache[size]->spacing_space;
			}
		} break;
		default: {
			ERR_FAIL_V_MSG(0, "Invalid spacing type: " + itos(p_spacing));
		} break;
	}
	return 0;
}

int TextServerFallback::font_get_texture_count(RID p_font_rid, const Vector2i &p_size) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, 0);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), 0);

	return fd->cache[size]->textures.size();
}

void TextServerFallback::font_clear_textures(RID p_font_rid, const Vector2i &p_size) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);
	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	fd->cache[size]->textures.clear();
}

void TextServerFallback::font_remove_texture(RID p_font_rid, const Vector2i &p_size, int p_texture_index) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);
	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	ERR_FAIL_INDEX(p_texture_index, fd->cache[size]->textures.size());

	fd->cache[size]->textures.remove(p_texture_index);
}

void TextServerFallback::font_set_texture_image(RID p_font_rid, const Vector2i &p_size, int p_texture_index, const Ref<Image> &p_image) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);
	ERR_FAIL_COND(p_image.is_null());

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);
	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	ERR_FAIL_COND(p_texture_index < 0);
	if (p_texture_index >= fd->cache[size]->textures.size()) {
		fd->cache[size]->textures.resize(p_texture_index + 1);
	}

	FontTexture &tex = fd->cache[size]->textures.write[p_texture_index];

	tex.imgdata = p_image->get_data();
	tex.texture_w = p_image->get_width();
	tex.texture_h = p_image->get_height();
	tex.format = p_image->get_format();

	Ref<Image> img = memnew(Image(tex.texture_w, tex.texture_h, 0, tex.format, tex.imgdata));
	tex.texture = Ref<ImageTexture>();
	tex.texture.instantiate();
	tex.texture->create_from_image(img);
}

Ref<Image> TextServerFallback::font_get_texture_image(RID p_font_rid, const Vector2i &p_size, int p_texture_index) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, Ref<Image>());

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);
	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), Ref<Image>());
	ERR_FAIL_INDEX_V(p_texture_index, fd->cache[size]->textures.size(), Ref<Image>());

	const FontTexture &tex = fd->cache[size]->textures.write[p_texture_index];
	Ref<Image> img = memnew(Image(tex.texture_w, tex.texture_h, 0, tex.format, tex.imgdata));

	return img;
}

void TextServerFallback::font_set_texture_offsets(RID p_font_rid, const Vector2i &p_size, int p_texture_index, const PackedInt32Array &p_offset) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);
	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	ERR_FAIL_COND(p_texture_index < 0);
	if (p_texture_index >= fd->cache[size]->textures.size()) {
		fd->cache[size]->textures.resize(p_texture_index + 1);
	}

	FontTexture &tex = fd->cache[size]->textures.write[p_texture_index];
	tex.offsets = p_offset;
}

PackedInt32Array TextServerFallback::font_get_texture_offsets(RID p_font_rid, const Vector2i &p_size, int p_texture_index) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, PackedInt32Array());

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);
	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), PackedInt32Array());
	ERR_FAIL_INDEX_V(p_texture_index, fd->cache[size]->textures.size(), PackedInt32Array());

	const FontTexture &tex = fd->cache[size]->textures.write[p_texture_index];
	return tex.offsets;
}

Array TextServerFallback::font_get_glyph_list(RID p_font_rid, const Vector2i &p_size) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, Array());

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);
	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), Array());

	Array ret;
	const HashMap<int32_t, FontGlyph> &gl = fd->cache[size]->glyph_map;
	const int32_t *E = nullptr;
	while ((E = gl.next(E))) {
		ret.push_back(*E);
	}
	return ret;
}

void TextServerFallback::font_clear_glyphs(RID p_font_rid, const Vector2i &p_size) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);
	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));

	fd->cache[size]->glyph_map.clear();
}

void TextServerFallback::font_remove_glyph(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);
	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));

	fd->cache[size]->glyph_map.erase(p_glyph);
}

Vector2 TextServerFallback::font_get_glyph_advance(RID p_font_rid, int p_size, int32_t p_glyph) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, Vector2());

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), Vector2());
	if (!_ensure_glyph(fd, size, p_glyph)) {
		return Vector2(); // Invalid or non graphicl glyph, do not display errors.
	}

	const HashMap<int32_t, FontGlyph> &gl = fd->cache[size]->glyph_map;

	if (fd->msdf) {
		return gl[p_glyph].advance * (float)p_size / (float)fd->msdf_source_size;
	} else {
		return gl[p_glyph].advance;
	}
}

void TextServerFallback::font_set_glyph_advance(RID p_font_rid, int p_size, int32_t p_glyph, const Vector2 &p_advance) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));

	HashMap<int32_t, FontGlyph> &gl = fd->cache[size]->glyph_map;

	gl[p_glyph].advance = p_advance;
	gl[p_glyph].found = true;
}

Vector2 TextServerFallback::font_get_glyph_offset(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, Vector2());

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), Vector2());
	if (!_ensure_glyph(fd, size, p_glyph)) {
		return Vector2(); // Invalid or non graphicl glyph, do not display errors.
	}

	const HashMap<int32_t, FontGlyph> &gl = fd->cache[size]->glyph_map;

	if (fd->msdf) {
		return gl[p_glyph].rect.position * (float)p_size.x / (float)fd->msdf_source_size;
	} else {
		return gl[p_glyph].rect.position;
	}
}

void TextServerFallback::font_set_glyph_offset(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_offset) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));

	HashMap<int32_t, FontGlyph> &gl = fd->cache[size]->glyph_map;

	gl[p_glyph].rect.position = p_offset;
	gl[p_glyph].found = true;
}

Vector2 TextServerFallback::font_get_glyph_size(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, Vector2());

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), Vector2());
	if (!_ensure_glyph(fd, size, p_glyph)) {
		return Vector2(); // Invalid or non graphicl glyph, do not display errors.
	}

	const HashMap<int32_t, FontGlyph> &gl = fd->cache[size]->glyph_map;

	if (fd->msdf) {
		return gl[p_glyph].rect.size * (float)p_size.x / (float)fd->msdf_source_size;
	} else {
		return gl[p_glyph].rect.size;
	}
}

void TextServerFallback::font_set_glyph_size(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Vector2 &p_gl_size) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));

	HashMap<int32_t, FontGlyph> &gl = fd->cache[size]->glyph_map;

	gl[p_glyph].rect.size = p_gl_size;
	gl[p_glyph].found = true;
}

Rect2 TextServerFallback::font_get_glyph_uv_rect(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, Rect2());

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), Rect2());
	if (!_ensure_glyph(fd, size, p_glyph)) {
		return Rect2(); // Invalid or non graphicl glyph, do not display errors.
	}

	const HashMap<int32_t, FontGlyph> &gl = fd->cache[size]->glyph_map;
	return gl[p_glyph].uv_rect;
}

void TextServerFallback::font_set_glyph_uv_rect(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, const Rect2 &p_uv_rect) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));

	HashMap<int32_t, FontGlyph> &gl = fd->cache[size]->glyph_map;

	gl[p_glyph].uv_rect = p_uv_rect;
	gl[p_glyph].found = true;
}

int TextServerFallback::font_get_glyph_texture_idx(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, -1);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), -1);
	if (!_ensure_glyph(fd, size, p_glyph)) {
		return -1; // Invalid or non graphicl glyph, do not display errors.
	}

	const HashMap<int32_t, FontGlyph> &gl = fd->cache[size]->glyph_map;
	return gl[p_glyph].texture_idx;
}

void TextServerFallback::font_set_glyph_texture_idx(RID p_font_rid, const Vector2i &p_size, int32_t p_glyph, int p_texture_idx) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));

	HashMap<int32_t, FontGlyph> &gl = fd->cache[size]->glyph_map;

	gl[p_glyph].texture_idx = p_texture_idx;
	gl[p_glyph].found = true;
}

Dictionary TextServerFallback::font_get_glyph_contours(RID p_font_rid, int p_size, int32_t p_index) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, Dictionary());

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), Dictionary());

	Vector<Vector3> points;
	Vector<int32_t> contours;
	bool orientation;
#ifdef MODULE_FREETYPE_ENABLED
	int error = FT_Load_Glyph(fd->cache[size]->face, FT_Get_Char_Index(fd->cache[size]->face, p_index), FT_LOAD_NO_BITMAP | (fd->force_autohinter ? FT_LOAD_FORCE_AUTOHINT : 0));
	ERR_FAIL_COND_V(error, Dictionary());

	points.clear();
	contours.clear();

	float h = fd->cache[size]->ascent;
	float scale = (1.0 / 64.0) / fd->cache[size]->oversampling * fd->cache[size]->scale;
	if (fd->msdf) {
		scale = scale * (float)p_size / (float)fd->msdf_source_size;
	}
	for (short i = 0; i < fd->cache[size]->face->glyph->outline.n_points; i++) {
		points.push_back(Vector3(fd->cache[size]->face->glyph->outline.points[i].x * scale, h - fd->cache[size]->face->glyph->outline.points[i].y * scale, FT_CURVE_TAG(fd->cache[size]->face->glyph->outline.tags[i])));
	}
	for (short i = 0; i < fd->cache[size]->face->glyph->outline.n_contours; i++) {
		contours.push_back(fd->cache[size]->face->glyph->outline.contours[i]);
	}
	orientation = (FT_Outline_Get_Orientation(&fd->cache[size]->face->glyph->outline) == FT_ORIENTATION_FILL_RIGHT);
#else
	return Dictionary();
#endif

	Dictionary out;
	out["points"] = points;
	out["contours"] = contours;
	out["orientation"] = orientation;
	return out;
}

Array TextServerFallback::font_get_kerning_list(RID p_font_rid, int p_size) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, Array());

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), Array());

	Array ret;
	for (const KeyValue<Vector2i, Vector2> &E : fd->cache[size]->kerning_map) {
		ret.push_back(E.key);
	}
	return ret;
}

void TextServerFallback::font_clear_kerning_map(RID p_font_rid, int p_size) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	fd->cache[size]->kerning_map.clear();
}

void TextServerFallback::font_remove_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	fd->cache[size]->kerning_map.erase(p_glyph_pair);
}

void TextServerFallback::font_set_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair, const Vector2 &p_kerning) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	fd->cache[size]->kerning_map[p_glyph_pair] = p_kerning;
}

Vector2 TextServerFallback::font_get_kerning(RID p_font_rid, int p_size, const Vector2i &p_glyph_pair) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, Vector2());

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);

	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), Vector2());

	const Map<Vector2i, Vector2> &kern = fd->cache[size]->kerning_map;

	if (kern.has(p_glyph_pair)) {
		if (fd->msdf) {
			return kern[p_glyph_pair] * (float)p_size / (float)fd->msdf_source_size;
		} else {
			return kern[p_glyph_pair];
		}
	} else {
#ifdef MODULE_FREETYPE_ENABLED
		if (fd->cache[size]->face) {
			FT_Vector delta;
			int32_t glyph_a = FT_Get_Char_Index(fd->cache[size]->face, p_glyph_pair.x);
			int32_t glyph_b = FT_Get_Char_Index(fd->cache[size]->face, p_glyph_pair.y);
			FT_Get_Kerning(fd->cache[size]->face, glyph_a, glyph_b, FT_KERNING_DEFAULT, &delta);
			if (fd->msdf) {
				return Vector2(delta.x, delta.y) * (float)p_size / (float)fd->msdf_source_size;
			} else {
				return Vector2(delta.x, delta.y);
			}
		}
#endif
	}
	return Vector2();
}

int32_t TextServerFallback::font_get_glyph_index(RID p_font_rid, int p_size, char32_t p_char, char32_t p_variation_selector) const {
	ERR_FAIL_COND_V_MSG((p_char >= 0xd800 && p_char <= 0xdfff) || (p_char > 0x10ffff), 0, "Unicode parsing error: Invalid unicode codepoint " + String::num_int64(p_char, 16) + ".");
	return (int32_t)p_char;
}

bool TextServerFallback::font_has_char(RID p_font_rid, char32_t p_char) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, false);
	ERR_FAIL_COND_V_MSG((p_char >= 0xd800 && p_char <= 0xdfff) || (p_char > 0x10ffff), false, "Unicode parsing error: Invalid unicode codepoint " + String::num_int64(p_char, 16) + ".");

	MutexLock lock(fd->mutex);
	if (fd->cache.is_empty()) {
		ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, fd->msdf ? Vector2i(fd->msdf_source_size, 0) : Vector2i(16, 0)), false);
	}
	FontDataForSizeFallback *at_size = fd->cache.front()->get();

#ifdef MODULE_FREETYPE_ENABLED
	if (at_size && at_size->face) {
		return FT_Get_Char_Index(at_size->face, p_char) != 0;
	}
#endif
	return (at_size) ? at_size->glyph_map.has((int32_t)p_char) : false;
}

String TextServerFallback::font_get_supported_chars(RID p_font_rid) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, String());

	MutexLock lock(fd->mutex);
	if (fd->cache.is_empty()) {
		ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, fd->msdf ? Vector2i(fd->msdf_source_size, 0) : Vector2i(16, 0)), String());
	}
	FontDataForSizeFallback *at_size = fd->cache.front()->get();

	String chars;
#ifdef MODULE_FREETYPE_ENABLED
	if (at_size && at_size->face) {
		FT_UInt gindex;
		FT_ULong charcode = FT_Get_First_Char(at_size->face, &gindex);
		while (gindex != 0) {
			if (charcode != 0) {
				chars += char32_t(charcode);
			}
			charcode = FT_Get_Next_Char(at_size->face, charcode, &gindex);
		}
		return chars;
	}
#endif
	if (at_size) {
		const HashMap<int32_t, FontGlyph> &gl = at_size->glyph_map;
		const int32_t *E = nullptr;
		while ((E = gl.next(E))) {
			chars += char32_t(*E);
		}
	}
	return chars;
}

void TextServerFallback::font_render_range(RID p_font_rid, const Vector2i &p_size, char32_t p_start, char32_t p_end) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);
	ERR_FAIL_COND_MSG((p_start >= 0xd800 && p_start <= 0xdfff) || (p_start > 0x10ffff), "Unicode parsing error: Invalid unicode codepoint " + String::num_int64(p_start, 16) + ".");
	ERR_FAIL_COND_MSG((p_end >= 0xd800 && p_end <= 0xdfff) || (p_end > 0x10ffff), "Unicode parsing error: Invalid unicode codepoint " + String::num_int64(p_end, 16) + ".");

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);
	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	for (char32_t i = p_start; i <= p_end; i++) {
		_ensure_glyph(fd, size, (int32_t)i);
	}
}

void TextServerFallback::font_render_glyph(RID p_font_rid, const Vector2i &p_size, int32_t p_index) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, p_size);
	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	ERR_FAIL_COND(!_ensure_glyph(fd, size, p_index));
}

void TextServerFallback::font_draw_glyph(RID p_font_rid, RID p_canvas, int p_size, const Vector2 &p_pos, int32_t p_index, const Color &p_color) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, p_size);
	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	if (!_ensure_glyph(fd, size, p_index)) {
		return; // // Invalid or non graphicl glyph, do not display errors, nothing to draw.
	}

	const FontGlyph &gl = fd->cache[size]->glyph_map[p_index];
	if (gl.found) {
		ERR_FAIL_COND(gl.texture_idx < -1 || gl.texture_idx >= fd->cache[size]->textures.size());

		if (gl.texture_idx != -1) {
			Color modulate = p_color;
#ifdef MODULE_FREETYPE_ENABLED
			if (fd->cache[size]->face && FT_HAS_COLOR(fd->cache[size]->face)) {
				modulate.r = modulate.g = modulate.b = 1.0;
			}
#endif
			if (RenderingServer::get_singleton() != nullptr) {
				RID texture = fd->cache[size]->textures[gl.texture_idx].texture->get_rid();
				if (fd->msdf) {
					Point2 cpos = p_pos;
					cpos += gl.rect.position * (float)p_size / (float)fd->msdf_source_size;
					Size2 csize = gl.rect.size * (float)p_size / (float)fd->msdf_source_size;
					RenderingServer::get_singleton()->canvas_item_add_msdf_texture_rect_region(p_canvas, Rect2(cpos, csize), texture, gl.uv_rect, modulate, 0, fd->msdf_range);
				} else {
					Point2i cpos = p_pos;
					cpos += gl.rect.position;
					Size2i csize = gl.rect.size;
					RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, Rect2(cpos, csize), texture, gl.uv_rect, modulate, false, false);
				}
			}
		}
	}
}

void TextServerFallback::font_draw_glyph_outline(RID p_font_rid, RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, int32_t p_index, const Color &p_color) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size_outline(fd, Vector2i(p_size, p_outline_size));
	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	if (!_ensure_glyph(fd, size, p_index)) {
		return; // // Invalid or non graphicl glyph, do not display errors, nothing to draw.
	}

	const FontGlyph &gl = fd->cache[size]->glyph_map[p_index];
	if (gl.found) {
		ERR_FAIL_COND(gl.texture_idx < -1 || gl.texture_idx >= fd->cache[size]->textures.size());

		if (gl.texture_idx != -1) {
			Color modulate = p_color;
#ifdef MODULE_FREETYPE_ENABLED
			if (fd->cache[size]->face && FT_HAS_COLOR(fd->cache[size]->face)) {
				modulate.r = modulate.g = modulate.b = 1.0;
			}
#endif
			if (RenderingServer::get_singleton() != nullptr) {
				RID texture = fd->cache[size]->textures[gl.texture_idx].texture->get_rid();
				if (fd->msdf) {
					Point2 cpos = p_pos;
					cpos += gl.rect.position * (float)p_size / (float)fd->msdf_source_size;
					Size2 csize = gl.rect.size * (float)p_size / (float)fd->msdf_source_size;
					RenderingServer::get_singleton()->canvas_item_add_msdf_texture_rect_region(p_canvas, Rect2(cpos, csize), texture, gl.uv_rect, modulate, p_outline_size * 2, fd->msdf_range);
				} else {
					Point2i cpos = p_pos;
					cpos += gl.rect.position;
					Size2i csize = gl.rect.size;
					RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas, Rect2(cpos, csize), texture, gl.uv_rect, modulate, false, false);
				}
			}
		}
	}
}

bool TextServerFallback::font_is_language_supported(RID p_font_rid, const String &p_language) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, false);

	MutexLock lock(fd->mutex);
	if (fd->language_support_overrides.has(p_language)) {
		return fd->language_support_overrides[p_language];
	} else {
		return true;
	}
}

void TextServerFallback::font_set_language_support_override(RID p_font_rid, const String &p_language, bool p_supported) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	fd->language_support_overrides[p_language] = p_supported;
}

bool TextServerFallback::font_get_language_support_override(RID p_font_rid, const String &p_language) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, false);

	MutexLock lock(fd->mutex);
	return fd->language_support_overrides[p_language];
}

void TextServerFallback::font_remove_language_support_override(RID p_font_rid, const String &p_language) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	fd->language_support_overrides.erase(p_language);
}

Vector<String> TextServerFallback::font_get_language_support_overrides(RID p_font_rid) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, Vector<String>());

	MutexLock lock(fd->mutex);
	Vector<String> out;
	for (const KeyValue<String, bool> &E : fd->language_support_overrides) {
		out.push_back(E.key);
	}
	return out;
}

bool TextServerFallback::font_is_script_supported(RID p_font_rid, const String &p_script) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, false);

	MutexLock lock(fd->mutex);
	if (fd->script_support_overrides.has(p_script)) {
		return fd->script_support_overrides[p_script];
	} else {
		return true;
	}
}

void TextServerFallback::font_set_script_support_override(RID p_font_rid, const String &p_script, bool p_supported) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	fd->script_support_overrides[p_script] = p_supported;
}

bool TextServerFallback::font_get_script_support_override(RID p_font_rid, const String &p_script) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, false);

	MutexLock lock(fd->mutex);
	return fd->script_support_overrides[p_script];
}

void TextServerFallback::font_remove_script_support_override(RID p_font_rid, const String &p_script) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND(!fd);

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, 16);
	ERR_FAIL_COND(!_ensure_cache_for_size(fd, size));
	fd->script_support_overrides.erase(p_script);
}

Vector<String> TextServerFallback::font_get_script_support_overrides(RID p_font_rid) {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, Vector<String>());

	MutexLock lock(fd->mutex);
	Vector<String> out;
	for (const KeyValue<String, bool> &E : fd->script_support_overrides) {
		out.push_back(E.key);
	}
	return out;
}

Dictionary TextServerFallback::font_supported_feature_list(RID p_font_rid) const {
	return Dictionary();
}

Dictionary TextServerFallback::font_supported_variation_list(RID p_font_rid) const {
	FontDataFallback *fd = font_owner.get_or_null(p_font_rid);
	ERR_FAIL_COND_V(!fd, Dictionary());

	MutexLock lock(fd->mutex);
	Vector2i size = _get_size(fd, 16);
	ERR_FAIL_COND_V(!_ensure_cache_for_size(fd, size), Dictionary());
	return fd->supported_varaitions;
}

float TextServerFallback::font_get_global_oversampling() const {
	return oversampling;
}

void TextServerFallback::font_set_global_oversampling(float p_oversampling) {
	_THREAD_SAFE_METHOD_
	if (oversampling != p_oversampling) {
		oversampling = p_oversampling;
		List<RID> fonts;
		font_owner.get_owned_list(&fonts);
		bool font_cleared = false;
		for (const RID &E : fonts) {
			if (!font_is_multichannel_signed_distance_field(E) && font_get_oversampling(E) <= 0) {
				font_clear_size_cache(E);
				font_cleared = true;
			}
		}

		if (font_cleared) {
			List<RID> text_bufs;
			shaped_owner.get_owned_list(&text_bufs);
			for (const RID &E : text_bufs) {
				invalidate(shaped_owner.get_or_null(E));
			}
		}
	}
}

/*************************************************************************/
/* Shaped text buffer interface                                          */
/*************************************************************************/

void TextServerFallback::invalidate(ShapedTextData *p_shaped) {
	p_shaped->valid = false;
	p_shaped->sort_valid = false;
	p_shaped->line_breaks_valid = false;
	p_shaped->justification_ops_valid = false;
	p_shaped->ascent = 0.f;
	p_shaped->descent = 0.f;
	p_shaped->width = 0.f;
	p_shaped->upos = 0.f;
	p_shaped->uthk = 0.f;
	p_shaped->glyphs.clear();
	p_shaped->glyphs_logical.clear();
}

void TextServerFallback::full_copy(ShapedTextData *p_shaped) {
	ShapedTextData *parent = shaped_owner.get_or_null(p_shaped->parent);

	for (const KeyValue<Variant, ShapedTextData::EmbeddedObject> &E : parent->objects) {
		if (E.value.pos >= p_shaped->start && E.value.pos < p_shaped->end) {
			p_shaped->objects[E.key] = E.value;
		}
	}

	for (int k = 0; k < parent->spans.size(); k++) {
		ShapedTextData::Span span = parent->spans[k];
		if (span.start >= p_shaped->end || span.end <= p_shaped->start) {
			continue;
		}
		span.start = MAX(p_shaped->start, span.start);
		span.end = MIN(p_shaped->end, span.end);
		p_shaped->spans.push_back(span);
	}

	p_shaped->parent = RID();
}

RID TextServerFallback::create_shaped_text(TextServer::Direction p_direction, TextServer::Orientation p_orientation) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = memnew(ShapedTextData);
	sd->direction = p_direction;
	sd->orientation = p_orientation;

	return shaped_owner.make_rid(sd);
}

void TextServerFallback::shaped_text_clear(RID p_shaped) {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND(!sd);

	MutexLock lock(sd->mutex);
	sd->parent = RID();
	sd->start = 0;
	sd->end = 0;
	sd->text = String();
	sd->spans.clear();
	sd->objects.clear();
	invalidate(sd);
}

void TextServerFallback::shaped_text_set_direction(RID p_shaped, TextServer::Direction p_direction) {
	if (p_direction == DIRECTION_RTL) {
		ERR_PRINT_ONCE("Right-to-left layout is not supported by this text server.");
	}
}

TextServer::Direction TextServerFallback::shaped_text_get_direction(RID p_shaped) const {
	return TextServer::DIRECTION_LTR;
}

void TextServerFallback::shaped_text_set_orientation(RID p_shaped, TextServer::Orientation p_orientation) {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND(!sd);

	MutexLock lock(sd->mutex);
	if (sd->orientation != p_orientation) {
		if (sd->parent != RID()) {
			full_copy(sd);
		}
		sd->orientation = p_orientation;
		invalidate(sd);
	}
}

void TextServerFallback::shaped_text_set_bidi_override(RID p_shaped, const Array &p_override) {
	// No BiDi support, ignore.
}

TextServer::Orientation TextServerFallback::shaped_text_get_orientation(RID p_shaped) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, TextServer::ORIENTATION_HORIZONTAL);

	MutexLock lock(sd->mutex);
	return sd->orientation;
}

void TextServerFallback::shaped_text_set_preserve_invalid(RID p_shaped, bool p_enabled) {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);

	MutexLock lock(sd->mutex);
	ERR_FAIL_COND(!sd);
	if (sd->preserve_invalid != p_enabled) {
		if (sd->parent != RID()) {
			full_copy(sd);
		}
		sd->preserve_invalid = p_enabled;
		invalidate(sd);
	}
}

bool TextServerFallback::shaped_text_get_preserve_invalid(RID p_shaped) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, false);

	MutexLock lock(sd->mutex);
	return sd->preserve_invalid;
}

void TextServerFallback::shaped_text_set_preserve_control(RID p_shaped, bool p_enabled) {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND(!sd);

	MutexLock lock(sd->mutex);
	if (sd->preserve_control != p_enabled) {
		if (sd->parent != RID()) {
			full_copy(sd);
		}
		sd->preserve_control = p_enabled;
		invalidate(sd);
	}
}

bool TextServerFallback::shaped_text_get_preserve_control(RID p_shaped) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, false);

	MutexLock lock(sd->mutex);
	return sd->preserve_control;
}

bool TextServerFallback::shaped_text_add_string(RID p_shaped, const String &p_text, const Vector<RID> &p_fonts, int p_size, const Dictionary &p_opentype_features, const String &p_language) {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, false);

	MutexLock lock(sd->mutex);
	ERR_FAIL_COND_V(p_size <= 0, false);

	for (int i = 0; i < p_fonts.size(); i++) {
		ERR_FAIL_COND_V(!font_owner.get_or_null(p_fonts[i]), false);
	}

	if (p_text.is_empty()) {
		return true;
	}

	if (sd->parent != RID()) {
		full_copy(sd);
	}

	ShapedTextData::Span span;
	span.start = sd->text.length();
	span.end = span.start + p_text.length();

	// Pre-sort fonts, push fonts with the language support first.
	Vector<RID> fonts_no_match;
	int font_count = p_fonts.size();
	for (int i = 0; i < font_count; i++) {
		if (font_is_language_supported(p_fonts[i], p_language)) {
			span.fonts.push_back(p_fonts[i]);
		} else {
			fonts_no_match.push_back(p_fonts[i]);
		}
	}
	span.fonts.append_array(fonts_no_match);

	ERR_FAIL_COND_V(span.fonts.is_empty(), false);
	span.font_size = p_size;
	span.language = p_language;

	sd->spans.push_back(span);
	sd->text += p_text;
	sd->end += p_text.length();
	invalidate(sd);

	return true;
}

bool TextServerFallback::shaped_text_add_object(RID p_shaped, Variant p_key, const Size2 &p_size, InlineAlign p_inline_align, int p_length) {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, false);

	MutexLock lock(sd->mutex);
	ERR_FAIL_COND_V(p_key == Variant(), false);
	ERR_FAIL_COND_V(sd->objects.has(p_key), false);

	if (sd->parent != RID()) {
		full_copy(sd);
	}

	ShapedTextData::Span span;
	span.start = sd->text.length();
	span.end = span.start + p_length;
	span.embedded_key = p_key;

	ShapedTextData::EmbeddedObject obj;
	obj.inline_align = p_inline_align;
	obj.rect.size = p_size;
	obj.pos = span.start;

	sd->spans.push_back(span);
	sd->text += String::chr(0xfffc).repeat(p_length);
	sd->end += p_length;
	sd->objects[p_key] = obj;
	invalidate(sd);

	return true;
}

bool TextServerFallback::shaped_text_resize_object(RID p_shaped, Variant p_key, const Size2 &p_size, InlineAlign p_inline_align) {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, false);

	MutexLock lock(sd->mutex);
	ERR_FAIL_COND_V(!sd->objects.has(p_key), false);
	sd->objects[p_key].rect.size = p_size;
	sd->objects[p_key].inline_align = p_inline_align;
	if (sd->valid) {
		// Recalc string metrics.
		sd->ascent = 0;
		sd->descent = 0;
		sd->width = 0;
		sd->upos = 0;
		sd->uthk = 0;
		int sd_size = sd->glyphs.size();

		for (int i = 0; i < sd_size; i++) {
			Glyph gl = sd->glyphs[i];
			Variant key;
			if (gl.count == 1) {
				for (const KeyValue<Variant, ShapedTextData::EmbeddedObject> &E : sd->objects) {
					if (E.value.pos == gl.start) {
						key = E.key;
						break;
					}
				}
			}
			if (key != Variant()) {
				if (sd->orientation == ORIENTATION_HORIZONTAL) {
					sd->objects[key].rect.position.x = sd->width;
					sd->width += sd->objects[key].rect.size.x;
					sd->glyphs.write[i].advance = sd->objects[key].rect.size.x;
				} else {
					sd->objects[key].rect.position.y = sd->width;
					sd->width += sd->objects[key].rect.size.y;
					sd->glyphs.write[i].advance = sd->objects[key].rect.size.y;
				}
			} else {
				if (gl.font_rid.is_valid()) {
					if (sd->orientation == ORIENTATION_HORIZONTAL) {
						sd->ascent = MAX(sd->ascent, font_get_ascent(gl.font_rid, gl.font_size));
						sd->descent = MAX(sd->descent, font_get_descent(gl.font_rid, gl.font_size));
					} else {
						sd->ascent = MAX(sd->ascent, Math::round(font_get_glyph_advance(gl.font_rid, gl.font_size, gl.index).x * 0.5));
						sd->descent = MAX(sd->descent, Math::round(font_get_glyph_advance(gl.font_rid, gl.font_size, gl.index).x * 0.5));
					}
					sd->upos = MAX(sd->upos, font_get_underline_position(gl.font_rid, gl.font_size));
					sd->uthk = MAX(sd->uthk, font_get_underline_thickness(gl.font_rid, gl.font_size));
				} else if (sd->preserve_invalid || (sd->preserve_control && is_control(gl.index))) {
					// Glyph not found, replace with hex code box.
					if (sd->orientation == ORIENTATION_HORIZONTAL) {
						sd->ascent = MAX(sd->ascent, get_hex_code_box_size(gl.font_size, gl.index).y);
					} else {
						sd->ascent = MAX(sd->ascent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
						sd->descent = MAX(sd->descent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
					}
				}
				sd->width += gl.advance * gl.repeat;
			}
		}

		// Align embedded objects to baseline.
		float full_ascent = sd->ascent;
		float full_descent = sd->descent;
		for (KeyValue<Variant, ShapedTextData::EmbeddedObject> &E : sd->objects) {
			if ((E.value.pos >= sd->start) && (E.value.pos < sd->end)) {
				if (sd->orientation == ORIENTATION_HORIZONTAL) {
					switch (E.value.inline_align & INLINE_ALIGN_TEXT_MASK) {
						case INLINE_ALIGN_TO_TOP: {
							E.value.rect.position.y = -sd->ascent;
						} break;
						case INLINE_ALIGN_TO_CENTER: {
							E.value.rect.position.y = (-sd->ascent + sd->descent) / 2;
						} break;
						case INLINE_ALIGN_TO_BASELINE: {
							E.value.rect.position.y = 0;
						} break;
						case INLINE_ALIGN_TO_BOTTOM: {
							E.value.rect.position.y = sd->descent;
						} break;
					}
					switch (E.value.inline_align & INLINE_ALIGN_IMAGE_MASK) {
						case INLINE_ALIGN_BOTTOM_TO: {
							E.value.rect.position.y -= E.value.rect.size.y;
						} break;
						case INLINE_ALIGN_CENTER_TO: {
							E.value.rect.position.y -= E.value.rect.size.y / 2;
						} break;
						case INLINE_ALIGN_TOP_TO: {
							// NOP
						} break;
					}
					full_ascent = MAX(full_ascent, -E.value.rect.position.y);
					full_descent = MAX(full_descent, E.value.rect.position.y + E.value.rect.size.y);
				} else {
					switch (E.value.inline_align & INLINE_ALIGN_TEXT_MASK) {
						case INLINE_ALIGN_TO_TOP: {
							E.value.rect.position.x = -sd->ascent;
						} break;
						case INLINE_ALIGN_TO_CENTER: {
							E.value.rect.position.x = (-sd->ascent + sd->descent) / 2;
						} break;
						case INLINE_ALIGN_TO_BASELINE: {
							E.value.rect.position.x = 0;
						} break;
						case INLINE_ALIGN_TO_BOTTOM: {
							E.value.rect.position.x = sd->descent;
						} break;
					}
					switch (E.value.inline_align & INLINE_ALIGN_IMAGE_MASK) {
						case INLINE_ALIGN_BOTTOM_TO: {
							E.value.rect.position.x -= E.value.rect.size.x;
						} break;
						case INLINE_ALIGN_CENTER_TO: {
							E.value.rect.position.x -= E.value.rect.size.x / 2;
						} break;
						case INLINE_ALIGN_TOP_TO: {
							// NOP
						} break;
					}
					full_ascent = MAX(full_ascent, -E.value.rect.position.x);
					full_descent = MAX(full_descent, E.value.rect.position.x + E.value.rect.size.x);
				}
			}
		}
		sd->ascent = full_ascent;
		sd->descent = full_descent;
	}
	return true;
}

RID TextServerFallback::shaped_text_substr(RID p_shaped, int p_start, int p_length) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, RID());

	MutexLock lock(sd->mutex);
	if (sd->parent != RID()) {
		return shaped_text_substr(sd->parent, p_start, p_length);
	}
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	ERR_FAIL_COND_V(p_start < 0 || p_length < 0, RID());
	ERR_FAIL_COND_V(sd->start > p_start || sd->end < p_start, RID());
	ERR_FAIL_COND_V(sd->end < p_start + p_length, RID());

	ShapedTextData *new_sd = memnew(ShapedTextData);
	new_sd->parent = p_shaped;
	new_sd->start = p_start;
	new_sd->end = p_start + p_length;

	new_sd->orientation = sd->orientation;
	new_sd->direction = sd->direction;
	new_sd->para_direction = sd->para_direction;
	new_sd->line_breaks_valid = sd->line_breaks_valid;
	new_sd->justification_ops_valid = sd->justification_ops_valid;
	new_sd->sort_valid = false;
	new_sd->upos = sd->upos;
	new_sd->uthk = sd->uthk;

	if (p_length > 0) {
		new_sd->text = sd->text.substr(p_start, p_length);
		int sd_size = sd->glyphs.size();
		const Glyph *sd_glyphs = sd->glyphs.ptr();

		for (int i = 0; i < sd_size; i++) {
			if ((sd_glyphs[i].start >= new_sd->start) && (sd_glyphs[i].end <= new_sd->end)) {
				Glyph gl = sd_glyphs[i];
				Variant key;
				bool find_embedded = false;
				if (gl.count == 1) {
					for (const KeyValue<Variant, ShapedTextData::EmbeddedObject> &E : sd->objects) {
						if (E.value.pos == gl.start) {
							find_embedded = true;
							key = E.key;
							new_sd->objects[key] = E.value;
							break;
						}
					}
				}
				if (find_embedded) {
					if (new_sd->orientation == ORIENTATION_HORIZONTAL) {
						new_sd->objects[key].rect.position.x = new_sd->width;
						new_sd->width += new_sd->objects[key].rect.size.x;
					} else {
						new_sd->objects[key].rect.position.y = new_sd->width;
						new_sd->width += new_sd->objects[key].rect.size.y;
					}
				} else {
					if (gl.font_rid.is_valid()) {
						if (new_sd->orientation == ORIENTATION_HORIZONTAL) {
							new_sd->ascent = MAX(new_sd->ascent, font_get_ascent(gl.font_rid, gl.font_size));
							new_sd->descent = MAX(new_sd->descent, font_get_descent(gl.font_rid, gl.font_size));
						} else {
							new_sd->ascent = MAX(new_sd->ascent, Math::round(font_get_glyph_advance(gl.font_rid, gl.font_size, gl.index).x * 0.5));
							new_sd->descent = MAX(new_sd->descent, Math::round(font_get_glyph_advance(gl.font_rid, gl.font_size, gl.index).x * 0.5));
						}
					} else if (new_sd->preserve_invalid || (new_sd->preserve_control && is_control(gl.index))) {
						// Glyph not found, replace with hex code box.
						if (new_sd->orientation == ORIENTATION_HORIZONTAL) {
							new_sd->ascent = MAX(new_sd->ascent, get_hex_code_box_size(gl.font_size, gl.index).y);
						} else {
							new_sd->ascent = MAX(new_sd->ascent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
							new_sd->descent = MAX(new_sd->descent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
						}
					}
					new_sd->width += gl.advance * gl.repeat;
				}
				new_sd->glyphs.push_back(gl);
			}
		}

		// Align embedded objects to baseline.
		float full_ascent = new_sd->ascent;
		float full_descent = new_sd->descent;
		for (KeyValue<Variant, ShapedTextData::EmbeddedObject> &E : new_sd->objects) {
			if ((E.value.pos >= new_sd->start) && (E.value.pos < new_sd->end)) {
				if (sd->orientation == ORIENTATION_HORIZONTAL) {
					switch (E.value.inline_align & INLINE_ALIGN_TEXT_MASK) {
						case INLINE_ALIGN_TO_TOP: {
							E.value.rect.position.y = -new_sd->ascent;
						} break;
						case INLINE_ALIGN_TO_CENTER: {
							E.value.rect.position.y = (-new_sd->ascent + new_sd->descent) / 2;
						} break;
						case INLINE_ALIGN_TO_BASELINE: {
							E.value.rect.position.y = 0;
						} break;
						case INLINE_ALIGN_TO_BOTTOM: {
							E.value.rect.position.y = new_sd->descent;
						} break;
					}
					switch (E.value.inline_align & INLINE_ALIGN_IMAGE_MASK) {
						case INLINE_ALIGN_BOTTOM_TO: {
							E.value.rect.position.y -= E.value.rect.size.y;
						} break;
						case INLINE_ALIGN_CENTER_TO: {
							E.value.rect.position.y -= E.value.rect.size.y / 2;
						} break;
						case INLINE_ALIGN_TOP_TO: {
							// NOP
						} break;
					}
					full_ascent = MAX(full_ascent, -E.value.rect.position.y);
					full_descent = MAX(full_descent, E.value.rect.position.y + E.value.rect.size.y);
				} else {
					switch (E.value.inline_align & INLINE_ALIGN_TEXT_MASK) {
						case INLINE_ALIGN_TO_TOP: {
							E.value.rect.position.x = -new_sd->ascent;
						} break;
						case INLINE_ALIGN_TO_CENTER: {
							E.value.rect.position.x = (-new_sd->ascent + new_sd->descent) / 2;
						} break;
						case INLINE_ALIGN_TO_BASELINE: {
							E.value.rect.position.x = 0;
						} break;
						case INLINE_ALIGN_TO_BOTTOM: {
							E.value.rect.position.x = new_sd->descent;
						} break;
					}
					switch (E.value.inline_align & INLINE_ALIGN_IMAGE_MASK) {
						case INLINE_ALIGN_BOTTOM_TO: {
							E.value.rect.position.x -= E.value.rect.size.x;
						} break;
						case INLINE_ALIGN_CENTER_TO: {
							E.value.rect.position.x -= E.value.rect.size.x / 2;
						} break;
						case INLINE_ALIGN_TOP_TO: {
							// NOP
						} break;
					}
					full_ascent = MAX(full_ascent, -E.value.rect.position.x);
					full_descent = MAX(full_descent, E.value.rect.position.x + E.value.rect.size.x);
				}
			}
		}
		new_sd->ascent = full_ascent;
		new_sd->descent = full_descent;
	}
	new_sd->valid = true;

	return shaped_owner.make_rid(new_sd);
}

RID TextServerFallback::shaped_text_get_parent(RID p_shaped) const {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, RID());

	MutexLock lock(sd->mutex);
	return sd->parent;
}

float TextServerFallback::shaped_text_fit_to_width(RID p_shaped, float p_width, uint16_t /*JustificationFlag*/ p_jst_flags) {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);

	MutexLock lock(sd->mutex);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	if (!sd->justification_ops_valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_update_justification_ops(p_shaped);
	}

	int start_pos = 0;
	int end_pos = sd->glyphs.size() - 1;

	if ((p_jst_flags & JUSTIFICATION_AFTER_LAST_TAB) == JUSTIFICATION_AFTER_LAST_TAB) {
		int start, end, delta;
		if (sd->para_direction == DIRECTION_LTR) {
			start = sd->glyphs.size() - 1;
			end = -1;
			delta = -1;
		} else {
			start = 0;
			end = sd->glyphs.size();
			delta = +1;
		}

		for (int i = start; i != end; i += delta) {
			if ((sd->glyphs[i].flags & GRAPHEME_IS_TAB) == GRAPHEME_IS_TAB) {
				if (sd->para_direction == DIRECTION_LTR) {
					start_pos = i;
					break;
				} else {
					end_pos = i;
					break;
				}
			}
		}
	}

	if ((p_jst_flags & JUSTIFICATION_TRIM_EDGE_SPACES) == JUSTIFICATION_TRIM_EDGE_SPACES) {
		while ((start_pos < end_pos) && ((sd->glyphs[start_pos].flags & GRAPHEME_IS_SPACE) == GRAPHEME_IS_SPACE || (sd->glyphs[start_pos].flags & GRAPHEME_IS_BREAK_HARD) == GRAPHEME_IS_BREAK_HARD || (sd->glyphs[start_pos].flags & GRAPHEME_IS_BREAK_SOFT) == GRAPHEME_IS_BREAK_SOFT)) {
			sd->width -= sd->glyphs[start_pos].advance * sd->glyphs[start_pos].repeat;
			sd->glyphs.write[start_pos].advance = 0;
			start_pos += sd->glyphs[start_pos].count;
		}
		while ((start_pos < end_pos) && ((sd->glyphs[end_pos].flags & GRAPHEME_IS_SPACE) == GRAPHEME_IS_SPACE || (sd->glyphs[end_pos].flags & GRAPHEME_IS_BREAK_HARD) == GRAPHEME_IS_BREAK_HARD || (sd->glyphs[end_pos].flags & GRAPHEME_IS_BREAK_SOFT) == GRAPHEME_IS_BREAK_SOFT)) {
			sd->width -= sd->glyphs[end_pos].advance * sd->glyphs[end_pos].repeat;
			sd->glyphs.write[end_pos].advance = 0;
			end_pos -= sd->glyphs[end_pos].count;
		}
	}

	int space_count = 0;
	for (int i = start_pos; i <= end_pos; i++) {
		const Glyph &gl = sd->glyphs[i];
		if (gl.count > 0) {
			if ((gl.flags & GRAPHEME_IS_SPACE) == GRAPHEME_IS_SPACE) {
				space_count++;
			}
		}
	}

	if ((space_count > 0) && ((p_jst_flags & JUSTIFICATION_WORD_BOUND) == JUSTIFICATION_WORD_BOUND)) {
		float delta_width_per_space = (p_width - sd->width) / space_count;
		for (int i = start_pos; i <= end_pos; i++) {
			Glyph &gl = sd->glyphs.write[i];
			if (gl.count > 0) {
				if ((gl.flags & GRAPHEME_IS_SPACE) == GRAPHEME_IS_SPACE) {
					float old_adv = gl.advance;
					gl.advance = MAX(gl.advance + delta_width_per_space, Math::round(0.1 * gl.font_size));
					sd->width += (gl.advance - old_adv);
				}
			}
		}
	}

	return sd->width;
}

float TextServerFallback::shaped_text_tab_align(RID p_shaped, const PackedFloat32Array &p_tab_stops) {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);

	MutexLock lock(sd->mutex);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	if (!sd->line_breaks_valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_update_breaks(p_shaped);
	}

	int tab_index = 0;
	float off = 0.f;

	int start, end, delta;
	if (sd->para_direction == DIRECTION_LTR) {
		start = 0;
		end = sd->glyphs.size();
		delta = +1;
	} else {
		start = sd->glyphs.size() - 1;
		end = -1;
		delta = -1;
	}

	Glyph *gl = sd->glyphs.ptrw();

	for (int i = start; i != end; i += delta) {
		if ((gl[i].flags & GRAPHEME_IS_TAB) == GRAPHEME_IS_TAB) {
			float tab_off = 0.f;
			while (tab_off <= off) {
				tab_off += p_tab_stops[tab_index];
				tab_index++;
				if (tab_index >= p_tab_stops.size()) {
					tab_index = 0;
				}
			}
			float old_adv = gl[i].advance;
			gl[i].advance = tab_off - off;
			sd->width += gl[i].advance - old_adv;
			off = 0;
			continue;
		}
		off += gl[i].advance * gl[i].repeat;
	}

	return 0.f;
}

bool TextServerFallback::shaped_text_update_breaks(RID p_shaped) {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, false);

	MutexLock lock(sd->mutex);
	if (!sd->valid) {
		shaped_text_shape(p_shaped);
	}

	if (sd->line_breaks_valid) {
		return true; // Nothing to do.
	}

	int sd_size = sd->glyphs.size();
	for (int i = 0; i < sd_size; i++) {
		if (sd->glyphs[i].count > 0) {
			char32_t c = sd->text[sd->glyphs[i].start];
			if (is_punct(c)) {
				sd->glyphs.write[i].flags |= GRAPHEME_IS_PUNCTUATION;
			}
			if (is_underscore(c)) {
				sd->glyphs.write[i].flags |= GRAPHEME_IS_UNDERSCORE;
			}
			if (is_whitespace(c) && !is_linebreak(c)) {
				sd->glyphs.write[i].flags |= GRAPHEME_IS_SPACE;
				sd->glyphs.write[i].flags |= GRAPHEME_IS_BREAK_SOFT;
			}
			if (is_linebreak(c)) {
				sd->glyphs.write[i].flags |= GRAPHEME_IS_BREAK_HARD;
			}
			if (c == 0x0009 || c == 0x000b) {
				sd->glyphs.write[i].flags |= GRAPHEME_IS_TAB;
			}

			i += (sd->glyphs[i].count - 1);
		}
	}
	sd->line_breaks_valid = true;
	return sd->line_breaks_valid;
}

bool TextServerFallback::shaped_text_update_justification_ops(RID p_shaped) {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, false);

	MutexLock lock(sd->mutex);
	if (!sd->valid) {
		shaped_text_shape(p_shaped);
	}
	if (!sd->line_breaks_valid) {
		shaped_text_update_breaks(p_shaped);
	}

	sd->justification_ops_valid = true; // Not supported by fallback server.
	return true;
}

void TextServerFallback::shaped_text_overrun_trim_to_width(RID p_shaped_line, float p_width, uint16_t p_trim_flags) {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped_line);
	ERR_FAIL_COND_MSG(!sd, "ShapedTextDataFallback invalid.");

	MutexLock lock(sd->mutex);
	if (!sd->valid) {
		shaped_text_shape(p_shaped_line);
	}

	sd->text_trimmed = false;
	sd->overrun_trim_data.ellipsis_glyph_buf.clear();

	bool add_ellipsis = (p_trim_flags & OVERRUN_ADD_ELLIPSIS) == OVERRUN_ADD_ELLIPSIS;
	bool cut_per_word = (p_trim_flags & OVERRUN_TRIM_WORD_ONLY) == OVERRUN_TRIM_WORD_ONLY;
	bool enforce_ellipsis = (p_trim_flags & OVERRUN_ENFORCE_ELLIPSIS) == OVERRUN_ENFORCE_ELLIPSIS;
	bool justification_aware = (p_trim_flags & OVERRUN_JUSTIFICATION_AWARE) == OVERRUN_JUSTIFICATION_AWARE;

	Glyph *sd_glyphs = sd->glyphs.ptrw();

	if ((p_trim_flags & OVERRUN_TRIM) == OVERRUN_NO_TRIMMING || sd_glyphs == nullptr || p_width <= 0 || !(sd->width > p_width || enforce_ellipsis)) {
		sd->overrun_trim_data.trim_pos = -1;
		sd->overrun_trim_data.ellipsis_pos = -1;
		return;
	}

	if (justification_aware && !sd->fit_width_minimum_reached) {
		return;
	}

	int sd_size = sd->glyphs.size();
	RID last_gl_font_rid = sd_glyphs[sd_size - 1].font_rid;
	int last_gl_font_size = sd_glyphs[sd_size - 1].font_size;
	int32_t dot_gl_idx = font_get_glyph_index(last_gl_font_rid, '.', 0);
	Vector2 dot_adv = font_get_glyph_advance(last_gl_font_rid, last_gl_font_size, dot_gl_idx);
	int32_t whitespace_gl_idx = font_get_glyph_index(last_gl_font_rid, ' ', 0);
	Vector2 whitespace_adv = font_get_glyph_advance(last_gl_font_rid, last_gl_font_size, whitespace_gl_idx);

	int ellipsis_width = 0;
	if (add_ellipsis) {
		ellipsis_width = 3 * dot_adv.x + font_get_spacing(last_gl_font_rid, last_gl_font_size, TextServer::SPACING_GLYPH) + (cut_per_word ? whitespace_adv.x : 0);
	}

	int ell_min_characters = 6;
	float width = sd->width;

	int trim_pos = 0;
	int ellipsis_pos = (enforce_ellipsis) ? 0 : -1;

	int last_valid_cut = 0;
	bool found = false;

	for (int i = sd_size - 1; i != -1; i--) {
		width -= sd_glyphs[i].advance * sd_glyphs[i].repeat;

		if (sd_glyphs[i].count > 0) {
			bool above_min_char_treshold = (i >= ell_min_characters);

			if (width + (((above_min_char_treshold && add_ellipsis) || enforce_ellipsis) ? ellipsis_width : 0) <= p_width) {
				if (cut_per_word && above_min_char_treshold) {
					if ((sd_glyphs[i].flags & GRAPHEME_IS_BREAK_SOFT) == GRAPHEME_IS_BREAK_SOFT) {
						last_valid_cut = i;
						found = true;
					}
				} else {
					last_valid_cut = i;
					found = true;
				}
				if (found) {
					trim_pos = last_valid_cut;

					if (add_ellipsis && (above_min_char_treshold || enforce_ellipsis) && width - ellipsis_width <= p_width) {
						ellipsis_pos = trim_pos;
					}
					break;
				}
			}
		}
	}

	sd->overrun_trim_data.trim_pos = trim_pos;
	sd->overrun_trim_data.ellipsis_pos = ellipsis_pos;
	if (trim_pos == 0 && enforce_ellipsis && add_ellipsis) {
		sd->overrun_trim_data.ellipsis_pos = 0;
	}

	if ((trim_pos >= 0 && sd->width > p_width) || enforce_ellipsis) {
		if (add_ellipsis && (ellipsis_pos > 0 || enforce_ellipsis)) {
			// Insert an additional space when cutting word bound for aesthetics.
			if (cut_per_word && (ellipsis_pos > 0)) {
				Glyph gl;
				gl.count = 1;
				gl.advance = whitespace_adv.x;
				gl.index = whitespace_gl_idx;
				gl.font_rid = last_gl_font_rid;
				gl.font_size = last_gl_font_size;
				gl.flags = GRAPHEME_IS_SPACE | GRAPHEME_IS_BREAK_SOFT | GRAPHEME_IS_VIRTUAL;

				sd->overrun_trim_data.ellipsis_glyph_buf.append(gl);
			}
			// Add ellipsis dots.
			Glyph gl;
			gl.count = 1;
			gl.repeat = 3;
			gl.advance = dot_adv.x;
			gl.index = dot_gl_idx;
			gl.font_rid = last_gl_font_rid;
			gl.font_size = last_gl_font_size;
			gl.flags = GRAPHEME_IS_PUNCTUATION | GRAPHEME_IS_VIRTUAL;

			sd->overrun_trim_data.ellipsis_glyph_buf.append(gl);
		}

		sd->text_trimmed = true;
		sd->width_trimmed = width + ((ellipsis_pos != -1) ? ellipsis_width : 0);
	}
}

int TextServerFallback::shaped_text_get_trim_pos(RID p_shaped) const {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V_MSG(!sd, -1, "ShapedTextData invalid.");

	MutexLock lock(sd->mutex);
	return sd->overrun_trim_data.trim_pos;
}

int TextServerFallback::shaped_text_get_ellipsis_pos(RID p_shaped) const {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V_MSG(!sd, -1, "ShapedTextData invalid.");

	MutexLock lock(sd->mutex);
	return sd->overrun_trim_data.ellipsis_pos;
}

const Glyph *TextServerFallback::shaped_text_get_ellipsis_glyphs(RID p_shaped) const {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V_MSG(!sd, nullptr, "ShapedTextData invalid.");

	MutexLock lock(sd->mutex);
	return sd->overrun_trim_data.ellipsis_glyph_buf.ptr();
}

int TextServerFallback::shaped_text_get_ellipsis_glyph_count(RID p_shaped) const {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V_MSG(!sd, 0, "ShapedTextData invalid.");

	MutexLock lock(sd->mutex);
	return sd->overrun_trim_data.ellipsis_glyph_buf.size();
}

bool TextServerFallback::shaped_text_shape(RID p_shaped) {
	ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, false);

	MutexLock lock(sd->mutex);
	if (sd->valid) {
		return true;
	}

	if (sd->parent != RID()) {
		full_copy(sd);
	}

	// Cleanup.
	sd->justification_ops_valid = false;
	sd->line_breaks_valid = false;
	sd->ascent = 0.f;
	sd->descent = 0.f;
	sd->width = 0.f;
	sd->glyphs.clear();

	if (sd->text.length() == 0) {
		sd->valid = true;
		return true;
	}

	// "Shape" string.
	for (int i = 0; i < sd->spans.size(); i++) {
		const ShapedTextData::Span &span = sd->spans[i];
		if (span.embedded_key != Variant()) {
			// Embedded object.
			if (sd->orientation == ORIENTATION_HORIZONTAL) {
				sd->objects[span.embedded_key].rect.position.x = sd->width;
				sd->width += sd->objects[span.embedded_key].rect.size.x;
			} else {
				sd->objects[span.embedded_key].rect.position.y = sd->width;
				sd->width += sd->objects[span.embedded_key].rect.size.y;
			}
			Glyph gl;
			gl.start = span.start;
			gl.end = span.end;
			gl.count = 1;
			gl.index = 0;
			gl.flags = GRAPHEME_IS_VALID | GRAPHEME_IS_VIRTUAL;
			if (sd->orientation == ORIENTATION_HORIZONTAL) {
				gl.advance = sd->objects[span.embedded_key].rect.size.x;
			} else {
				gl.advance = sd->objects[span.embedded_key].rect.size.y;
			}
			sd->glyphs.push_back(gl);
		} else {
			// Text span.
			for (int j = span.start; j < span.end; j++) {
				Glyph gl;
				gl.start = j;
				gl.end = j + 1;
				gl.count = 1;
				gl.font_size = span.font_size;
				gl.index = (int32_t)sd->text[j]; // Use codepoint.
				if (gl.index == 0x0009 || gl.index == 0x000b) {
					gl.index = 0x0020;
				}
				if (!sd->preserve_control && is_control(gl.index)) {
					gl.index = 0x0020;
				}
				// Select first font which has character (font are already sorted by span language).
				for (int k = 0; k < span.fonts.size(); k++) {
					if (font_has_char(span.fonts[k], gl.index)) {
						gl.font_rid = span.fonts[k];
						break;
					}
				}

				if (gl.font_rid.is_valid()) {
					if (sd->text[j] != 0 && !is_linebreak(sd->text[j])) {
						if (sd->orientation == ORIENTATION_HORIZONTAL) {
							gl.advance = font_get_glyph_advance(gl.font_rid, gl.font_size, gl.index).x;
							gl.x_off = 0;
							gl.y_off = 0;
							sd->ascent = MAX(sd->ascent, font_get_ascent(gl.font_rid, gl.font_size));
							sd->descent = MAX(sd->descent, font_get_descent(gl.font_rid, gl.font_size));
						} else {
							gl.advance = font_get_glyph_advance(gl.font_rid, gl.font_size, gl.index).y;
							gl.x_off = -Math::round(font_get_glyph_advance(gl.font_rid, gl.font_size, gl.index).x * 0.5);
							gl.y_off = font_get_ascent(gl.font_rid, gl.font_size);
							sd->ascent = MAX(sd->ascent, Math::round(font_get_glyph_advance(gl.font_rid, gl.font_size, gl.index).x * 0.5));
							sd->descent = MAX(sd->descent, Math::round(font_get_glyph_advance(gl.font_rid, gl.font_size, gl.index).x * 0.5));
						}
					}
					if (font_get_spacing(gl.font_rid, gl.font_size, TextServer::SPACING_SPACE) && is_whitespace(sd->text[j])) {
						gl.advance += font_get_spacing(gl.font_rid, gl.font_size, TextServer::SPACING_SPACE);
					} else {
						gl.advance += font_get_spacing(gl.font_rid, gl.font_size, TextServer::SPACING_GLYPH);
					}
					sd->upos = MAX(sd->upos, font_get_underline_position(gl.font_rid, gl.font_size));
					sd->uthk = MAX(sd->uthk, font_get_underline_thickness(gl.font_rid, gl.font_size));

					// Add kerning to previous glyph.
					if (sd->glyphs.size() > 0) {
						Glyph &prev_gl = sd->glyphs.write[sd->glyphs.size() - 1];
						if (prev_gl.font_rid == gl.font_rid && prev_gl.font_size == gl.font_size) {
							if (sd->orientation == ORIENTATION_HORIZONTAL) {
								prev_gl.advance += font_get_kerning(gl.font_rid, gl.font_size, Vector2i(prev_gl.index, gl.index)).x;
							} else {
								prev_gl.advance += font_get_kerning(gl.font_rid, gl.font_size, Vector2i(prev_gl.index, gl.index)).y;
							}
						}
					}
				} else if (sd->preserve_invalid || (sd->preserve_control && is_control(gl.index))) {
					// Glyph not found, replace with hex code box.
					if (sd->orientation == ORIENTATION_HORIZONTAL) {
						gl.advance = get_hex_code_box_size(gl.font_size, gl.index).x;
						sd->ascent = MAX(sd->ascent, get_hex_code_box_size(gl.font_size, gl.index).y);
					} else {
						gl.advance = get_hex_code_box_size(gl.font_size, gl.index).y;
						sd->ascent = MAX(sd->ascent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
						sd->descent = MAX(sd->descent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
					}
				}
				sd->width += gl.advance;
				sd->glyphs.push_back(gl);
			}
		}
	}

	// Align embedded objects to baseline.
	float full_ascent = sd->ascent;
	float full_descent = sd->descent;
	for (KeyValue<Variant, ShapedTextData::EmbeddedObject> &E : sd->objects) {
		if (sd->orientation == ORIENTATION_HORIZONTAL) {
			switch (E.value.inline_align & INLINE_ALIGN_TEXT_MASK) {
				case INLINE_ALIGN_TO_TOP: {
					E.value.rect.position.y = -sd->ascent;
				} break;
				case INLINE_ALIGN_TO_CENTER: {
					E.value.rect.position.y = (-sd->ascent + sd->descent) / 2;
				} break;
				case INLINE_ALIGN_TO_BASELINE: {
					E.value.rect.position.y = 0;
				} break;
				case INLINE_ALIGN_TO_BOTTOM: {
					E.value.rect.position.y = sd->descent;
				} break;
			}
			switch (E.value.inline_align & INLINE_ALIGN_IMAGE_MASK) {
				case INLINE_ALIGN_BOTTOM_TO: {
					E.value.rect.position.y -= E.value.rect.size.y;
				} break;
				case INLINE_ALIGN_CENTER_TO: {
					E.value.rect.position.y -= E.value.rect.size.y / 2;
				} break;
				case INLINE_ALIGN_TOP_TO: {
					// NOP
				} break;
			}
			full_ascent = MAX(full_ascent, -E.value.rect.position.y);
			full_descent = MAX(full_descent, E.value.rect.position.y + E.value.rect.size.y);
		} else {
			switch (E.value.inline_align & INLINE_ALIGN_TEXT_MASK) {
				case INLINE_ALIGN_TO_TOP: {
					E.value.rect.position.x = -sd->ascent;
				} break;
				case INLINE_ALIGN_TO_CENTER: {
					E.value.rect.position.x = (-sd->ascent + sd->descent) / 2;
				} break;
				case INLINE_ALIGN_TO_BASELINE: {
					E.value.rect.position.x = 0;
				} break;
				case INLINE_ALIGN_TO_BOTTOM: {
					E.value.rect.position.x = sd->descent;
				} break;
			}
			switch (E.value.inline_align & INLINE_ALIGN_IMAGE_MASK) {
				case INLINE_ALIGN_BOTTOM_TO: {
					E.value.rect.position.x -= E.value.rect.size.x;
				} break;
				case INLINE_ALIGN_CENTER_TO: {
					E.value.rect.position.x -= E.value.rect.size.x / 2;
				} break;
				case INLINE_ALIGN_TOP_TO: {
					// NOP
				} break;
			}
			full_ascent = MAX(full_ascent, -E.value.rect.position.x);
			full_descent = MAX(full_descent, E.value.rect.position.x + E.value.rect.size.x);
		}
	}
	sd->ascent = full_ascent;
	sd->descent = full_descent;
	sd->valid = true;
	return sd->valid;
}

bool TextServerFallback::shaped_text_is_ready(RID p_shaped) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, false);

	MutexLock lock(sd->mutex);
	return sd->valid;
}

const Glyph *TextServerFallback::shaped_text_get_glyphs(RID p_shaped) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, nullptr);

	MutexLock lock(sd->mutex);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->glyphs.ptr();
}

int TextServerFallback::shaped_text_get_glyph_count(RID p_shaped) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, 0);

	MutexLock lock(sd->mutex);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->glyphs.size();
}

const Glyph *TextServerFallback::shaped_text_sort_logical(RID p_shaped) {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, nullptr);

	MutexLock lock(sd->mutex);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}

	return sd->glyphs.ptr(); // Already in the logical order, return as is.
}

Vector2i TextServerFallback::shaped_text_get_range(RID p_shaped) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, Vector2i());

	MutexLock lock(sd->mutex);
	return Vector2(sd->start, sd->end);
}

Array TextServerFallback::shaped_text_get_objects(RID p_shaped) const {
	Array ret;
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, ret);

	MutexLock lock(sd->mutex);
	for (const KeyValue<Variant, ShapedTextData::EmbeddedObject> &E : sd->objects) {
		ret.push_back(E.key);
	}

	return ret;
}

Rect2 TextServerFallback::shaped_text_get_object_rect(RID p_shaped, Variant p_key) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, Rect2());

	MutexLock lock(sd->mutex);
	ERR_FAIL_COND_V(!sd->objects.has(p_key), Rect2());
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->objects[p_key].rect;
}

Size2 TextServerFallback::shaped_text_get_size(RID p_shaped) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, Size2());

	MutexLock lock(sd->mutex);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	if (sd->orientation == TextServer::ORIENTATION_HORIZONTAL) {
		return Size2(sd->width, sd->ascent + sd->descent);
	} else {
		return Size2(sd->ascent + sd->descent, sd->width);
	}
}

float TextServerFallback::shaped_text_get_ascent(RID p_shaped) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);

	MutexLock lock(sd->mutex);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->ascent;
}

float TextServerFallback::shaped_text_get_descent(RID p_shaped) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);

	MutexLock lock(sd->mutex);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->descent;
}

float TextServerFallback::shaped_text_get_width(RID p_shaped) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);

	MutexLock lock(sd->mutex);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->width;
}

float TextServerFallback::shaped_text_get_underline_position(RID p_shaped) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);

	MutexLock lock(sd->mutex);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}

	return sd->upos;
}

float TextServerFallback::shaped_text_get_underline_thickness(RID p_shaped) const {
	const ShapedTextData *sd = shaped_owner.get_or_null(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);

	MutexLock lock(sd->mutex);
	if (!sd->valid) {
		const_cast<TextServerFallback *>(this)->shaped_text_shape(p_shaped);
	}

	return sd->uthk;
}

TextServerFallback::TextServerFallback() {
	_insert_feature_sets();
};

TextServerFallback::~TextServerFallback() {
	if (library != nullptr) {
		FT_Done_FreeType(library);
	}
};
