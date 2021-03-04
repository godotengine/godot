/*************************************************************************/
/*  msdf_loader.cpp                                                      */
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

#include "msdf_loader.h"

#include "modules/modules_enabled.gen.h"

#ifdef MODULE_SVG_ENABLED
#include <nanosvg.h>
#endif

void MSDFLoader::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_svg", "path", "scale"), &MSDFLoader::load_svg, DEFVAL(1.0));

	ClassDB::bind_method(D_METHOD("set_px_range", "range"), &MSDFLoader::set_px_range);
	ClassDB::bind_method(D_METHOD("get_px_range"), &MSDFLoader::get_px_range);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "px_range"), "set_px_range", "get_px_range");

	ClassDB::bind_method(D_METHOD("set_sdf_type", "type"), &MSDFLoader::set_sdf_type);
	ClassDB::bind_method(D_METHOD("get_sdf_type"), &MSDFLoader::get_sdf_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sdf_type", PROPERTY_HINT_ENUM, "True distance,Pseudo-distance,Multi-channel,Combined true and multi-channel"), "set_sdf_type", "get_sdf_type");

	ClassDB::bind_method(D_METHOD("clear_shape"), &MSDFLoader::clear_shape);
	ClassDB::bind_method(D_METHOD("move_to", "to"), &MSDFLoader::move_to);
	ClassDB::bind_method(D_METHOD("line_to", "to"), &MSDFLoader::line_to);
	ClassDB::bind_method(D_METHOD("conic_to", "ctrl", "to"), &MSDFLoader::conic_to);
	ClassDB::bind_method(D_METHOD("cubic_to", "ctrl1", "ctrl2", "to"), &MSDFLoader::cubic_to);

	ClassDB::bind_method(D_METHOD("get_data"), &MSDFLoader::get_data);

	BIND_ENUM_CONSTANT(SDF_TRUE);
	BIND_ENUM_CONSTANT(SDF_PSEUDO);
	BIND_ENUM_CONSTANT(SDF_MULTICHANNEL);
	BIND_ENUM_CONSTANT(SDF_COMBINED);
};

Error MSDFLoader::load_svg(const String &p_path, float p_scale) {
	clear_shape();

#ifdef MODULE_MSDFGEN_ENABLED
	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);
	if (!f) {
		ERR_PRINT("Error opening file '" + p_path + "'.");
		return err;
	}

	// Read to buffer.
	uint64_t size = f->get_length();
	Vector<uint8_t> src_image;
	src_image.resize(size + 1);
	uint8_t *src_w = src_image.ptrw();
	f->get_buffer(src_w, size);
	src_w[size] = '\0';
	f->close();
	memdelete(f);

	// Load SVG.
	const uint8_t *src_r = src_image.ptr();
	NSVGimage *image = nsvgParse((char *)src_r, "px", 96);
	if (image == nullptr) {
		ERR_PRINT("SVG Corrupted");
		return ERR_FILE_CORRUPT;
	}

	// Copy contours.
	for (const NSVGshape *svg_shape = image->shapes; svg_shape != nullptr; svg_shape = svg_shape->next) {
		for (const NSVGpath *path = svg_shape->paths; path != nullptr; path = path->next) {
			contour = &shape.addContour();
			for (int i = 0; i < path->npts - 1; i += 3) {
				const float *p = &path->pts[i * 2];
				contour->addEdge(new msdfgen::CubicSegment(msdfgen::Point2(p[0], p[1]) * p_scale, msdfgen::Point2(p[2], p[3]) * p_scale, msdfgen::Point2(p[4], p[5]) * p_scale, msdfgen::Point2(p[6], p[7]) * p_scale));
			}
		}
	}

	nsvgDelete(image);

	return OK;
#else
	ERR_PRINT("Compiled without NanoSVG support!");
	return ERR_CANT_CREATE;
#endif
}

void MSDFLoader::set_px_range(float p_range) {
	if (px_range != p_range) {
		px_range = p_range;
		dirty = true;
	}
}

float MSDFLoader::get_px_range() const {
	return px_range;
}

void MSDFLoader::set_sdf_type(int p_type) {
	if (sdf_type != (SDFType)p_type) {
		sdf_type = (SDFType)p_type;
		dirty = true;
	}
}

int MSDFLoader::get_sdf_type() const {
	return (int)sdf_type;
}

void MSDFLoader::clear_shape() {
	dirty = true;
	shape.contours.clear();
	contour = nullptr;
	position = msdfgen::Point2();
}

void MSDFLoader::move_to(const Vector2 &p_to) {
	if (!(contour && contour->edges.empty())) {
		contour = &shape.addContour();
	}
	position = msdfgen::Point2(p_to.x, p_to.y);
	dirty = true;
}

void MSDFLoader::line_to(const Vector2 &p_to) {
	ERR_FAIL_COND(!contour);
	msdfgen::Point2 endpoint = msdfgen::Point2(p_to.x, p_to.y);
	if (endpoint != position) {
		contour->addEdge(new msdfgen::LinearSegment(position, endpoint));
		position = endpoint;
		dirty = true;
	}
}

void MSDFLoader::conic_to(const Vector2 &p_ctrl, const Vector2 &p_to) {
	ERR_FAIL_COND(!contour);
	contour->addEdge(new msdfgen::QuadraticSegment(position, msdfgen::Point2(p_ctrl.x, p_ctrl.y), msdfgen::Point2(p_to.x, p_to.y)));
	position = msdfgen::Point2(p_to.x, p_to.y);
	dirty = true;
}

void MSDFLoader::cubic_to(const Vector2 &p_ctrl1, const Vector2 &p_ctrl2, const Vector2 &p_to) {
	ERR_FAIL_COND(!contour);
	contour->addEdge(new msdfgen::CubicSegment(position, msdfgen::Point2(p_ctrl1.x, p_ctrl1.y), msdfgen::Point2(p_ctrl2.x, p_ctrl2.y), msdfgen::Point2(p_to.x, p_to.y)));
	dirty = true;
}

Ref<Image> MSDFLoader::get_data() {
	if (dirty) {
		data = Ref<Image>();

		if (!shape.contours.empty() && shape.contours.back().edges.empty()) {
			shape.contours.pop_back();
		}
		shape.normalize();

		msdfgen::Shape::Bounds bounds = shape.getBounds(px_range);

		if (shape.validate() && shape.contours.size() > 0) {
			int w = (bounds.r - bounds.l);
			int h = (bounds.t - bounds.b);

			edgeColoringSimple(shape, 3.0); // Max. angle.
			switch (sdf_type) {
				case SDF_TRUE: {
					msdfgen::Bitmap<float, 1> image(w, h);
					msdfgen::generateSDF(image, shape, px_range, 1.0, msdfgen::Vector2(-bounds.l, -bounds.b)); // Range, scale, translation.
					Vector<uint8_t> imgdata;
					imgdata.resize(w * h);
					uint8_t *wr = imgdata.ptrw();
					for (int i = 0; i < h; i++) {
						for (int j = 0; j < w; j++) {
							int ofs = (i * w + j);
							wr[ofs + 0] = (uint8_t)(CLAMP(image(j, i)[0] * 256.f, 0.f, 255.f));
						}
					}
					data = Ref<Image>(memnew(Image(w, h, 0, Image::FORMAT_L8, imgdata)));
				} break;
				case SDF_PSEUDO: {
					msdfgen::Bitmap<float, 1> image(w, h);
					msdfgen::generatePseudoSDF(image, shape, px_range, 1.0, msdfgen::Vector2(-bounds.l, -bounds.b)); // Range, scale, translation.
					Vector<uint8_t> imgdata;
					imgdata.resize(w * h);
					uint8_t *wr = imgdata.ptrw();
					for (int i = 0; i < h; i++) {
						for (int j = 0; j < w; j++) {
							int ofs = (i * w + j);
							wr[ofs + 0] = (uint8_t)(CLAMP(image(j, i)[0] * 256.f, 0.f, 255.f));
						}
					}
					data = Ref<Image>(memnew(Image(w, h, 0, Image::FORMAT_L8, imgdata)));
				} break;
				case SDF_MULTICHANNEL: {
					msdfgen::Bitmap<float, 3> image(w, h);
					msdfgen::generateMSDF(image, shape, px_range, 1.0, msdfgen::Vector2(-bounds.l, -bounds.b)); // Range, scale, translation.
					Vector<uint8_t> imgdata;
					imgdata.resize(w * h * 3);
					uint8_t *wr = imgdata.ptrw();
					for (int i = 0; i < h; i++) {
						for (int j = 0; j < w; j++) {
							int ofs = (i * w + j) * 3;
							wr[ofs + 0] = (uint8_t)(CLAMP(image(j, i)[0] * 256.f, 0.f, 255.f));
							wr[ofs + 1] = (uint8_t)(CLAMP(image(j, i)[1] * 256.f, 0.f, 255.f));
							wr[ofs + 2] = (uint8_t)(CLAMP(image(j, i)[2] * 256.f, 0.f, 255.f));
						}
					}
					data = Ref<Image>(memnew(Image(w, h, 0, Image::FORMAT_RGB8, imgdata)));
				} break;
				case SDF_COMBINED: {
					msdfgen::Bitmap<float, 4> image(w, h);
					msdfgen::generateMTSDF(image, shape, px_range, 1.0, msdfgen::Vector2(-bounds.l, -bounds.b)); // Range, scale, translation.
					Vector<uint8_t> imgdata;
					imgdata.resize(w * h * 4);
					uint8_t *wr = imgdata.ptrw();
					for (int i = 0; i < h; i++) {
						for (int j = 0; j < w; j++) {
							int ofs = (i * w + j) * 4;
							wr[ofs + 0] = (uint8_t)(CLAMP(image(j, i)[0] * 256.f, 0.f, 255.f));
							wr[ofs + 1] = (uint8_t)(CLAMP(image(j, i)[1] * 256.f, 0.f, 255.f));
							wr[ofs + 2] = (uint8_t)(CLAMP(image(j, i)[2] * 256.f, 0.f, 255.f));
							wr[ofs + 3] = (uint8_t)(CLAMP(image(j, i)[3] * 256.f, 0.f, 255.f));
						}
					}
					data = Ref<Image>(memnew(Image(w, h, 0, Image::FORMAT_RGBA8, imgdata)));
				} break;
				default: {
					ERR_FAIL_V_MSG(Ref<Image>(), "Invalid SDF type");
				} break;
			}
			dirty = false;
		}
	}

	return data;
}
