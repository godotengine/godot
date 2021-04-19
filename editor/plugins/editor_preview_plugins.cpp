/*************************************************************************/
/*  editor_preview_plugins.cpp                                           */
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

#include "editor_preview_plugins.h"

#include "core/io/file_access_memory.h"
#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "scene/resources/bit_map.h"
#include "scene/resources/dynamic_font.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "servers/audio/audio_stream.h"

void post_process_preview(Ref<Image> p_image) {

	if (p_image->get_format() != Image::FORMAT_RGBA8)
		p_image->convert(Image::FORMAT_RGBA8);

	p_image->lock();

	const int w = p_image->get_width();
	const int h = p_image->get_height();

	const int r = MIN(w, h) / 32;
	const int r2 = r * r;
	Color transparent = Color(0, 0, 0, 0);

	for (int i = 0; i < r; i++) {
		for (int j = 0; j < r; j++) {
			int dx = i - r;
			int dy = j - r;
			if (dx * dx + dy * dy > r2) {
				p_image->set_pixel(i, j, transparent);
				p_image->set_pixel(w - 1 - i, j, transparent);
				p_image->set_pixel(w - 1 - i, h - 1 - j, transparent);
				p_image->set_pixel(i, h - 1 - j, transparent);
			} else {
				break;
			}
		}
	}

	p_image->unlock();
}

bool EditorTexturePreviewPlugin::handles(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "Texture");
}

bool EditorTexturePreviewPlugin::generate_small_preview_automatically() const {
	return true;
}

Ref<Texture> EditorTexturePreviewPlugin::generate(const RES &p_from, const Size2 &p_size) const {

	Ref<Image> img;
	Ref<AtlasTexture> atex = p_from;
	Ref<LargeTexture> ltex = p_from;
	if (atex.is_valid()) {
		Ref<Texture> tex = atex->get_atlas();
		if (!tex.is_valid()) {
			return Ref<Texture>();
		}

		Ref<Image> atlas = tex->get_data();
		if (!atlas.is_valid()) {
			return Ref<Texture>();
		}

		img = atlas->get_rect(atex->get_region());
	} else if (ltex.is_valid()) {
		img = ltex->to_image();
	} else {
		Ref<Texture> tex = p_from;
		if (tex.is_valid()) {
			img = tex->get_data();
			if (img.is_valid()) {
				img = img->duplicate();
			}
		}
	}

	if (img.is_null() || img->empty())
		return Ref<Texture>();

	img->clear_mipmaps();

	if (img->is_compressed()) {
		if (img->decompress() != OK)
			return Ref<Texture>();
	} else if (img->get_format() != Image::FORMAT_RGB8 && img->get_format() != Image::FORMAT_RGBA8) {
		img->convert(Image::FORMAT_RGBA8);
	}

	Vector2 new_size = img->get_size();
	if (new_size.x > p_size.x) {
		new_size = Vector2(p_size.x, new_size.y * p_size.x / new_size.x);
	}
	if (new_size.y > p_size.y) {
		new_size = Vector2(new_size.x * p_size.y / new_size.y, p_size.y);
	}
	Vector2i new_size_i(MAX(1, (int)new_size.x), MAX(1, (int)new_size.y));
	img->resize(new_size_i.x, new_size_i.y, Image::INTERPOLATE_CUBIC);

	post_process_preview(img);

	Ref<ImageTexture> ptex = Ref<ImageTexture>(memnew(ImageTexture));

	ptex->create_from_image(img, 0);
	return ptex;
}

EditorTexturePreviewPlugin::EditorTexturePreviewPlugin() {
}

////////////////////////////////////////////////////////////////////////////

bool EditorImagePreviewPlugin::handles(const String &p_type) const {

	return p_type == "Image";
}

Ref<Texture> EditorImagePreviewPlugin::generate(const RES &p_from, const Size2 &p_size) const {

	Ref<Image> img = p_from;

	if (img.is_null() || img->empty())
		return Ref<Image>();

	img = img->duplicate();
	img->clear_mipmaps();

	if (img->is_compressed()) {
		if (img->decompress() != OK)
			return Ref<Image>();
	} else if (img->get_format() != Image::FORMAT_RGB8 && img->get_format() != Image::FORMAT_RGBA8) {
		img->convert(Image::FORMAT_RGBA8);
	}

	Vector2 new_size = img->get_size();
	if (new_size.x > p_size.x) {
		new_size = Vector2(p_size.x, new_size.y * p_size.x / new_size.x);
	}
	if (new_size.y > p_size.y) {
		new_size = Vector2(new_size.x * p_size.y / new_size.y, p_size.y);
	}
	img->resize(new_size.x, new_size.y, Image::INTERPOLATE_CUBIC);

	post_process_preview(img);

	Ref<ImageTexture> ptex;
	ptex.instance();

	ptex->create_from_image(img, 0);
	return ptex;
}

EditorImagePreviewPlugin::EditorImagePreviewPlugin() {
}

bool EditorImagePreviewPlugin::generate_small_preview_automatically() const {
	return true;
}
////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////
bool EditorBitmapPreviewPlugin::handles(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "BitMap");
}

Ref<Texture> EditorBitmapPreviewPlugin::generate(const RES &p_from, const Size2 &p_size) const {

	Ref<BitMap> bm = p_from;

	if (bm->get_size() == Size2()) {
		return Ref<Texture>();
	}

	PoolVector<uint8_t> data;

	data.resize(bm->get_size().width * bm->get_size().height);

	{
		PoolVector<uint8_t>::Write w = data.write();

		for (int i = 0; i < bm->get_size().width; i++) {
			for (int j = 0; j < bm->get_size().height; j++) {
				if (bm->get_bit(Point2i(i, j))) {
					w[j * bm->get_size().width + i] = 255;
				} else {
					w[j * bm->get_size().width + i] = 0;
				}
			}
		}
	}

	Ref<Image> img;
	img.instance();
	img->create(bm->get_size().width, bm->get_size().height, 0, Image::FORMAT_L8, data);

	if (img->is_compressed()) {
		if (img->decompress() != OK)
			return Ref<Texture>();
	} else if (img->get_format() != Image::FORMAT_RGB8 && img->get_format() != Image::FORMAT_RGBA8) {
		img->convert(Image::FORMAT_RGBA8);
	}

	Vector2 new_size = img->get_size();
	if (new_size.x > p_size.x) {
		new_size = Vector2(p_size.x, new_size.y * p_size.x / new_size.x);
	}
	if (new_size.y > p_size.y) {
		new_size = Vector2(new_size.x * p_size.y / new_size.y, p_size.y);
	}
	img->resize(new_size.x, new_size.y, Image::INTERPOLATE_CUBIC);

	post_process_preview(img);

	Ref<ImageTexture> ptex = Ref<ImageTexture>(memnew(ImageTexture));

	ptex->create_from_image(img, 0);
	return ptex;
}

bool EditorBitmapPreviewPlugin::generate_small_preview_automatically() const {
	return true;
}

EditorBitmapPreviewPlugin::EditorBitmapPreviewPlugin() {
}

///////////////////////////////////////////////////////////////////////////

bool EditorPackedScenePreviewPlugin::handles(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "PackedScene");
}
Ref<Texture> EditorPackedScenePreviewPlugin::generate(const RES &p_from, const Size2 &p_size) const {

	return generate_from_path(p_from->get_path(), p_size);
}

Ref<Texture> EditorPackedScenePreviewPlugin::generate_from_path(const String &p_path, const Size2 &p_size) const {

	String temp_path = EditorSettings::get_singleton()->get_cache_dir();
	String cache_base = ProjectSettings::get_singleton()->globalize_path(p_path).md5_text();
	cache_base = temp_path.plus_file("resthumb-" + cache_base);

	//does not have it, try to load a cached thumbnail

	String path = cache_base + ".png";

	if (!FileAccess::exists(path))
		return Ref<Texture>();

	Ref<Image> img;
	img.instance();
	Error err = img->load(path);
	if (err == OK) {

		Ref<ImageTexture> ptex = Ref<ImageTexture>(memnew(ImageTexture));

		post_process_preview(img);
		ptex->create_from_image(img, 0);
		return ptex;

	} else {
		return Ref<Texture>();
	}
}

EditorPackedScenePreviewPlugin::EditorPackedScenePreviewPlugin() {
}

//////////////////////////////////////////////////////////////////

void EditorMaterialPreviewPlugin::_preview_done(const Variant &p_udata) {

	preview_done.set();
}

void EditorMaterialPreviewPlugin::_bind_methods() {

	ClassDB::bind_method("_preview_done", &EditorMaterialPreviewPlugin::_preview_done);
}

bool EditorMaterialPreviewPlugin::handles(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "Material"); //any material
}

bool EditorMaterialPreviewPlugin::generate_small_preview_automatically() const {
	return true;
}

Ref<Texture> EditorMaterialPreviewPlugin::generate(const RES &p_from, const Size2 &p_size) const {

	Ref<Material> material = p_from;
	ERR_FAIL_COND_V(material.is_null(), Ref<Texture>());

	if (material->get_shader_mode() == Shader::MODE_SPATIAL) {

		VS::get_singleton()->mesh_surface_set_material(sphere, 0, material->get_rid());

		VS::get_singleton()->viewport_set_update_mode(viewport, VS::VIEWPORT_UPDATE_ONCE); //once used for capture

		preview_done.clear();
		VS::get_singleton()->request_frame_drawn_callback(const_cast<EditorMaterialPreviewPlugin *>(this), "_preview_done", Variant());

		while (!preview_done.is_set()) {
			OS::get_singleton()->delay_usec(10);
		}

		Ref<Image> img = VS::get_singleton()->texture_get_data(viewport_texture);
		VS::get_singleton()->mesh_surface_set_material(sphere, 0, RID());

		ERR_FAIL_COND_V(!img.is_valid(), Ref<ImageTexture>());

		img->convert(Image::FORMAT_RGBA8);
		int thumbnail_size = MAX(p_size.x, p_size.y);
		img->resize(thumbnail_size, thumbnail_size, Image::INTERPOLATE_CUBIC);
		post_process_preview(img);
		Ref<ImageTexture> ptex = Ref<ImageTexture>(memnew(ImageTexture));
		ptex->create_from_image(img, 0);
		return ptex;
	}

	return Ref<Texture>();
}

EditorMaterialPreviewPlugin::EditorMaterialPreviewPlugin() {

	scenario = VS::get_singleton()->scenario_create();

	viewport = VS::get_singleton()->viewport_create();
	VS::get_singleton()->viewport_set_update_mode(viewport, VS::VIEWPORT_UPDATE_DISABLED);
	VS::get_singleton()->viewport_set_scenario(viewport, scenario);
	VS::get_singleton()->viewport_set_size(viewport, 128, 128);
	VS::get_singleton()->viewport_set_transparent_background(viewport, true);
	VS::get_singleton()->viewport_set_active(viewport, true);
	VS::get_singleton()->viewport_set_vflip(viewport, true);
	viewport_texture = VS::get_singleton()->viewport_get_texture(viewport);

	camera = VS::get_singleton()->camera_create();
	VS::get_singleton()->viewport_attach_camera(viewport, camera);
	VS::get_singleton()->camera_set_transform(camera, Transform(Basis(), Vector3(0, 0, 3)));
	VS::get_singleton()->camera_set_perspective(camera, 45, 0.1, 10);

	light = VS::get_singleton()->directional_light_create();
	light_instance = VS::get_singleton()->instance_create2(light, scenario);
	VS::get_singleton()->instance_set_transform(light_instance, Transform().looking_at(Vector3(-1, -1, -1), Vector3(0, 1, 0)));

	light2 = VS::get_singleton()->directional_light_create();
	VS::get_singleton()->light_set_color(light2, Color(0.7, 0.7, 0.7));
	//VS::get_singleton()->light_set_color(light2, Color(0.7, 0.7, 0.7));

	light_instance2 = VS::get_singleton()->instance_create2(light2, scenario);

	VS::get_singleton()->instance_set_transform(light_instance2, Transform().looking_at(Vector3(0, 1, 0), Vector3(0, 0, 1)));

	sphere = VS::get_singleton()->mesh_create();
	sphere_instance = VS::get_singleton()->instance_create2(sphere, scenario);

	int lats = 32;
	int lons = 32;
	float radius = 1.0;

	PoolVector<Vector3> vertices;
	PoolVector<Vector3> normals;
	PoolVector<Vector2> uvs;
	PoolVector<float> tangents;
	Basis tt = Basis(Vector3(0, 1, 0), Math_PI * 0.5);

	for (int i = 1; i <= lats; i++) {
		double lat0 = Math_PI * (-0.5 + (double)(i - 1) / lats);
		double z0 = Math::sin(lat0);
		double zr0 = Math::cos(lat0);

		double lat1 = Math_PI * (-0.5 + (double)i / lats);
		double z1 = Math::sin(lat1);
		double zr1 = Math::cos(lat1);

		for (int j = lons; j >= 1; j--) {

			double lng0 = 2 * Math_PI * (double)(j - 1) / lons;
			double x0 = Math::cos(lng0);
			double y0 = Math::sin(lng0);

			double lng1 = 2 * Math_PI * (double)(j) / lons;
			double x1 = Math::cos(lng1);
			double y1 = Math::sin(lng1);

			Vector3 v[4] = {
				Vector3(x1 * zr0, z0, y1 * zr0),
				Vector3(x1 * zr1, z1, y1 * zr1),
				Vector3(x0 * zr1, z1, y0 * zr1),
				Vector3(x0 * zr0, z0, y0 * zr0)
			};

#define ADD_POINT(m_idx)                                                                       \
	normals.push_back(v[m_idx]);                                                               \
	vertices.push_back(v[m_idx] * radius);                                                     \
	{                                                                                          \
		Vector2 uv(Math::atan2(v[m_idx].x, v[m_idx].z), Math::atan2(-v[m_idx].y, v[m_idx].z)); \
		uv /= Math_PI;                                                                         \
		uv *= 4.0;                                                                             \
		uv = uv * 0.5 + Vector2(0.5, 0.5);                                                     \
		uvs.push_back(uv);                                                                     \
	}                                                                                          \
	{                                                                                          \
		Vector3 t = tt.xform(v[m_idx]);                                                        \
		tangents.push_back(t.x);                                                               \
		tangents.push_back(t.y);                                                               \
		tangents.push_back(t.z);                                                               \
		tangents.push_back(1.0);                                                               \
	}

			ADD_POINT(0);
			ADD_POINT(1);
			ADD_POINT(2);

			ADD_POINT(2);
			ADD_POINT(3);
			ADD_POINT(0);
		}
	}

	Array arr;
	arr.resize(VS::ARRAY_MAX);
	arr[VS::ARRAY_VERTEX] = vertices;
	arr[VS::ARRAY_NORMAL] = normals;
	arr[VS::ARRAY_TANGENT] = tangents;
	arr[VS::ARRAY_TEX_UV] = uvs;
	VS::get_singleton()->mesh_add_surface_from_arrays(sphere, VS::PRIMITIVE_TRIANGLES, arr);
}

EditorMaterialPreviewPlugin::~EditorMaterialPreviewPlugin() {

	VS::get_singleton()->free(sphere);
	VS::get_singleton()->free(sphere_instance);
	VS::get_singleton()->free(viewport);
	VS::get_singleton()->free(light);
	VS::get_singleton()->free(light_instance);
	VS::get_singleton()->free(light2);
	VS::get_singleton()->free(light_instance2);
	VS::get_singleton()->free(camera);
	VS::get_singleton()->free(scenario);
}

///////////////////////////////////////////////////////////////////////////

static bool _is_text_char(CharType c) {

	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

bool EditorScriptPreviewPlugin::handles(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "Script");
}

Ref<Texture> EditorScriptPreviewPlugin::generate(const RES &p_from, const Size2 &p_size) const {

	Ref<Script> scr = p_from;
	if (scr.is_null())
		return Ref<Texture>();

	String code = scr->get_source_code().strip_edges();
	if (code == "")
		return Ref<Texture>();

	List<String> kwors;
	scr->get_language()->get_reserved_words(&kwors);

	Set<String> keywords;

	for (List<String>::Element *E = kwors.front(); E; E = E->next()) {

		keywords.insert(E->get());
	}

	int line = 0;
	int col = 0;
	Ref<Image> img;
	img.instance();
	int thumbnail_size = MAX(p_size.x, p_size.y);
	img->create(thumbnail_size, thumbnail_size, 0, Image::FORMAT_RGBA8);

	Color bg_color = EditorSettings::get_singleton()->get("text_editor/highlighting/background_color");
	Color keyword_color = EditorSettings::get_singleton()->get("text_editor/highlighting/keyword_color");
	Color text_color = EditorSettings::get_singleton()->get("text_editor/highlighting/text_color");
	Color symbol_color = EditorSettings::get_singleton()->get("text_editor/highlighting/symbol_color");

	img->lock();

	if (bg_color.a == 0)
		bg_color = Color(0, 0, 0, 0);
	bg_color.a = MAX(bg_color.a, 0.2); // some background

	for (int i = 0; i < thumbnail_size; i++) {
		for (int j = 0; j < thumbnail_size; j++) {
			img->set_pixel(i, j, bg_color);
		}
	}

	const int x0 = thumbnail_size / 8;
	const int y0 = thumbnail_size / 8;
	const int available_height = thumbnail_size - 2 * y0;
	col = x0;

	bool prev_is_text = false;
	bool in_keyword = false;
	for (int i = 0; i < code.length(); i++) {

		CharType c = code[i];
		if (c > 32) {
			if (col < thumbnail_size) {
				Color color = text_color;

				if (c != '_' && ((c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~') || c == '\t')) {
					//make symbol a little visible
					color = symbol_color;
					in_keyword = false;
				} else if (!prev_is_text && _is_text_char(c)) {
					int pos = i;

					while (_is_text_char(code[pos])) {
						pos++;
					}
					String word = code.substr(i, pos - i);
					if (keywords.has(word))
						in_keyword = true;

				} else if (!_is_text_char(c)) {
					in_keyword = false;
				}

				if (in_keyword)
					color = keyword_color;

				Color ul = color;
				ul.a *= 0.5;
				img->set_pixel(col, y0 + line * 2, bg_color.blend(ul));
				img->set_pixel(col, y0 + line * 2 + 1, color);

				prev_is_text = _is_text_char(c);
			}
		} else {

			prev_is_text = false;
			in_keyword = false;

			if (c == '\n') {
				col = x0;
				line++;
				if (line >= available_height / 2)
					break;
			} else if (c == '\t') {
				col += 3;
			}
		}
		col++;
	}

	img->unlock();

	post_process_preview(img);

	Ref<ImageTexture> ptex = Ref<ImageTexture>(memnew(ImageTexture));

	ptex->create_from_image(img, 0);
	return ptex;
}

EditorScriptPreviewPlugin::EditorScriptPreviewPlugin() {
}
///////////////////////////////////////////////////////////////////

bool EditorAudioStreamPreviewPlugin::handles(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "AudioStream");
}

Ref<Texture> EditorAudioStreamPreviewPlugin::generate(const RES &p_from, const Size2 &p_size) const {

	Ref<AudioStream> stream = p_from;
	ERR_FAIL_COND_V(stream.is_null(), Ref<Texture>());

	PoolVector<uint8_t> img;

	int w = p_size.x;
	int h = p_size.y;
	img.resize(w * h * 3);

	PoolVector<uint8_t>::Write imgdata = img.write();
	uint8_t *imgw = imgdata.ptr();

	Ref<AudioStreamPlayback> playback = stream->instance_playback();
	ERR_FAIL_COND_V(playback.is_null(), Ref<Texture>());

	float len_s = stream->get_length();
	if (len_s == 0) {
		len_s = 60; //one minute audio if no length specified
	}
	int frame_length = AudioServer::get_singleton()->get_mix_rate() * len_s;

	Vector<AudioFrame> frames;
	frames.resize(frame_length);

	playback->start();
	playback->mix(frames.ptrw(), 1, frames.size());
	playback->stop();

	for (int i = 0; i < w; i++) {

		float max = -1000;
		float min = 1000;
		int from = uint64_t(i) * frame_length / w;
		int to = (uint64_t(i) + 1) * frame_length / w;
		to = MIN(to, frame_length);
		from = MIN(from, frame_length - 1);
		if (to == from) {
			to = from + 1;
		}

		for (int j = from; j < to; j++) {

			max = MAX(max, frames[j].l);
			max = MAX(max, frames[j].r);

			min = MIN(min, frames[j].l);
			min = MIN(min, frames[j].r);
		}

		int pfrom = CLAMP((min * 0.5 + 0.5) * h / 2, 0, h / 2) + h / 4;
		int pto = CLAMP((max * 0.5 + 0.5) * h / 2, 0, h / 2) + h / 4;

		for (int j = 0; j < h; j++) {
			uint8_t *p = &imgw[(j * w + i) * 3];
			if (j < pfrom || j > pto) {
				p[0] = 100;
				p[1] = 100;
				p[2] = 100;
			} else {
				p[0] = 180;
				p[1] = 180;
				p[2] = 180;
			}
		}
	}

	imgdata.release();
	//post_process_preview(img);

	Ref<ImageTexture> ptex = Ref<ImageTexture>(memnew(ImageTexture));
	Ref<Image> image;
	image.instance();
	image->create(w, h, false, Image::FORMAT_RGB8, img);
	ptex->create_from_image(image, 0);
	return ptex;
}

EditorAudioStreamPreviewPlugin::EditorAudioStreamPreviewPlugin() {
}

///////////////////////////////////////////////////////////////////////////

void EditorMeshPreviewPlugin::_preview_done(const Variant &p_udata) {

	preview_done.set();
}

void EditorMeshPreviewPlugin::_bind_methods() {

	ClassDB::bind_method("_preview_done", &EditorMeshPreviewPlugin::_preview_done);
}
bool EditorMeshPreviewPlugin::handles(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "Mesh"); //any Mesh
}

Ref<Texture> EditorMeshPreviewPlugin::generate(const RES &p_from, const Size2 &p_size) const {

	Ref<Mesh> mesh = p_from;
	ERR_FAIL_COND_V(mesh.is_null(), Ref<Texture>());

	VS::get_singleton()->instance_set_base(mesh_instance, mesh->get_rid());

	AABB aabb = mesh->get_aabb();
	Vector3 ofs = aabb.position + aabb.size * 0.5;
	aabb.position -= ofs;
	Transform xform;
	xform.basis = Basis().rotated(Vector3(0, 1, 0), -Math_PI * 0.125);
	xform.basis = Basis().rotated(Vector3(1, 0, 0), Math_PI * 0.125) * xform.basis;
	AABB rot_aabb = xform.xform(aabb);
	float m = MAX(rot_aabb.size.x, rot_aabb.size.y) * 0.5;
	if (m == 0)
		return Ref<Texture>();
	m = 1.0 / m;
	m *= 0.5;
	xform.basis.scale(Vector3(m, m, m));
	xform.origin = -xform.basis.xform(ofs); //-ofs*m;
	xform.origin.z -= rot_aabb.size.z * 2;
	VS::get_singleton()->instance_set_transform(mesh_instance, xform);

	VS::get_singleton()->viewport_set_update_mode(viewport, VS::VIEWPORT_UPDATE_ONCE); //once used for capture

	preview_done.clear();
	VS::get_singleton()->request_frame_drawn_callback(const_cast<EditorMeshPreviewPlugin *>(this), "_preview_done", Variant());

	while (!preview_done.is_set()) {
		OS::get_singleton()->delay_usec(10);
	}

	Ref<Image> img = VS::get_singleton()->texture_get_data(viewport_texture);
	ERR_FAIL_COND_V(img.is_null(), Ref<ImageTexture>());

	VS::get_singleton()->instance_set_base(mesh_instance, RID());

	img->convert(Image::FORMAT_RGBA8);

	Vector2 new_size = img->get_size();
	if (new_size.x > p_size.x) {
		new_size = Vector2(p_size.x, new_size.y * p_size.x / new_size.x);
	}
	if (new_size.y > p_size.y) {
		new_size = Vector2(new_size.x * p_size.y / new_size.y, p_size.y);
	}
	img->resize(new_size.x, new_size.y, Image::INTERPOLATE_CUBIC);

	post_process_preview(img);

	Ref<ImageTexture> ptex = Ref<ImageTexture>(memnew(ImageTexture));
	ptex->create_from_image(img, 0);
	return ptex;
}

EditorMeshPreviewPlugin::EditorMeshPreviewPlugin() {

	scenario = VS::get_singleton()->scenario_create();

	viewport = VS::get_singleton()->viewport_create();
	VS::get_singleton()->viewport_set_update_mode(viewport, VS::VIEWPORT_UPDATE_DISABLED);
	VS::get_singleton()->viewport_set_vflip(viewport, true);
	VS::get_singleton()->viewport_set_scenario(viewport, scenario);
	VS::get_singleton()->viewport_set_size(viewport, 128, 128);
	VS::get_singleton()->viewport_set_transparent_background(viewport, true);
	VS::get_singleton()->viewport_set_active(viewport, true);
	viewport_texture = VS::get_singleton()->viewport_get_texture(viewport);

	camera = VS::get_singleton()->camera_create();
	VS::get_singleton()->viewport_attach_camera(viewport, camera);
	VS::get_singleton()->camera_set_transform(camera, Transform(Basis(), Vector3(0, 0, 3)));
	//VS::get_singleton()->camera_set_perspective(camera,45,0.1,10);
	VS::get_singleton()->camera_set_orthogonal(camera, 1.0, 0.01, 1000.0);

	light = VS::get_singleton()->directional_light_create();
	light_instance = VS::get_singleton()->instance_create2(light, scenario);
	VS::get_singleton()->instance_set_transform(light_instance, Transform().looking_at(Vector3(-1, -1, -1), Vector3(0, 1, 0)));

	light2 = VS::get_singleton()->directional_light_create();
	VS::get_singleton()->light_set_color(light2, Color(0.7, 0.7, 0.7));
	//VS::get_singleton()->light_set_color(light2, VS::LIGHT_COLOR_SPECULAR, Color(0.0, 0.0, 0.0));
	light_instance2 = VS::get_singleton()->instance_create2(light2, scenario);

	VS::get_singleton()->instance_set_transform(light_instance2, Transform().looking_at(Vector3(0, 1, 0), Vector3(0, 0, 1)));

	//sphere = VS::get_singleton()->mesh_create();
	mesh_instance = VS::get_singleton()->instance_create();
	VS::get_singleton()->instance_set_scenario(mesh_instance, scenario);
}

EditorMeshPreviewPlugin::~EditorMeshPreviewPlugin() {

	//VS::get_singleton()->free(sphere);
	VS::get_singleton()->free(mesh_instance);
	VS::get_singleton()->free(viewport);
	VS::get_singleton()->free(light);
	VS::get_singleton()->free(light_instance);
	VS::get_singleton()->free(light2);
	VS::get_singleton()->free(light_instance2);
	VS::get_singleton()->free(camera);
	VS::get_singleton()->free(scenario);
}

///////////////////////////////////////////////////////////////////////////

void EditorFontPreviewPlugin::_preview_done(const Variant &p_udata) {

	preview_done.set();
}

void EditorFontPreviewPlugin::_bind_methods() {

	ClassDB::bind_method("_preview_done", &EditorFontPreviewPlugin::_preview_done);
}

bool EditorFontPreviewPlugin::handles(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "DynamicFontData") || ClassDB::is_parent_class(p_type, "DynamicFont");
}

Ref<Texture> EditorFontPreviewPlugin::generate_from_path(const String &p_path, const Size2 &p_size) const {

	Ref<ResourceInteractiveLoader> ril = ResourceLoader::load_interactive(p_path);
	ril.ptr()->wait();
	RES res = ril.ptr()->get_resource();
	Ref<DynamicFont> sampled_font;
	if (res->is_class("DynamicFont")) {
		sampled_font = res->duplicate();
		if (sampled_font->get_outline_color() == Color(1, 1, 1, 1)) {
			sampled_font->set_outline_color(Color(0, 0, 0, 1));
		}
	} else if (res->is_class("DynamicFontData")) {
		sampled_font.instance();
		sampled_font->set_font_data(res);
	}
	sampled_font->set_size(50);

	String sampled_text = "Abg";
	Vector2 size = sampled_font->get_string_size(sampled_text);

	Vector2 pos;

	pos.x = 64 - size.x / 2;
	pos.y = 80;

	Ref<Font> font = sampled_font;

	font->draw(canvas_item, pos, sampled_text);

	preview_done.clear();
	VS::get_singleton()->viewport_set_update_mode(viewport, VS::VIEWPORT_UPDATE_ONCE); //once used for capture
	VS::get_singleton()->request_frame_drawn_callback(const_cast<EditorFontPreviewPlugin *>(this), "_preview_done", Variant());

	while (!preview_done.is_set()) {
		OS::get_singleton()->delay_usec(10);
	}

	VS::get_singleton()->canvas_item_clear(canvas_item);

	Ref<Image> img = VS::get_singleton()->texture_get_data(viewport_texture);
	ERR_FAIL_COND_V(img.is_null(), Ref<ImageTexture>());

	img->convert(Image::FORMAT_RGBA8);

	Vector2 new_size = img->get_size();
	if (new_size.x > p_size.x) {
		new_size = Vector2(p_size.x, new_size.y * p_size.x / new_size.x);
	}
	if (new_size.y > p_size.y) {
		new_size = Vector2(new_size.x * p_size.y / new_size.y, p_size.y);
	}
	img->resize(new_size.x, new_size.y, Image::INTERPOLATE_CUBIC);

	post_process_preview(img);

	Ref<ImageTexture> ptex = Ref<ImageTexture>(memnew(ImageTexture));
	ptex->create_from_image(img, 0);

	return ptex;
}

Ref<Texture> EditorFontPreviewPlugin::generate(const RES &p_from, const Size2 &p_size) const {

	String path = p_from->get_path();
	if (!FileAccess::exists(path)) {
		return Ref<Texture>();
	}
	return generate_from_path(path, p_size);
}

EditorFontPreviewPlugin::EditorFontPreviewPlugin() {

	viewport = VS::get_singleton()->viewport_create();
	VS::get_singleton()->viewport_set_update_mode(viewport, VS::VIEWPORT_UPDATE_DISABLED);
	VS::get_singleton()->viewport_set_vflip(viewport, true);
	VS::get_singleton()->viewport_set_size(viewport, 128, 128);
	VS::get_singleton()->viewport_set_active(viewport, true);
	viewport_texture = VS::get_singleton()->viewport_get_texture(viewport);

	canvas = VS::get_singleton()->canvas_create();
	canvas_item = VS::get_singleton()->canvas_item_create();

	VS::get_singleton()->viewport_attach_canvas(viewport, canvas);
	VS::get_singleton()->canvas_item_set_parent(canvas_item, canvas);
}

EditorFontPreviewPlugin::~EditorFontPreviewPlugin() {

	VS::get_singleton()->free(canvas_item);
	VS::get_singleton()->free(canvas);
	VS::get_singleton()->free(viewport);
}
