/*************************************************************************/
/*  editor_preview_plugins.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "editor_preview_plugins.h"

#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "io/file_access_memory.h"
#include "io/resource_loader.h"
#include "os/os.h"
#include "scene/resources/bit_mask.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"

bool EditorTexturePreviewPlugin::handles(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "Texture");
}

Ref<Texture> EditorTexturePreviewPlugin::generate(const RES &p_from) {

	Ref<Image> img;
	Ref<AtlasTexture> atex = p_from;
	if (atex.is_valid()) {
		Ref<Texture> tex = atex->get_atlas();
		if (!tex.is_valid()) {
			return Ref<Texture>();
		}
		Ref<Image> atlas = tex->get_data();
		img = atlas->get_rect(atex->get_region());
	} else {
		Ref<Texture> tex = p_from;
		img = tex->get_data();
	}

	if (img.is_null() || img->empty())
		return Ref<Texture>();

	img->clear_mipmaps();

	int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
	thumbnail_size *= EDSCALE;
	if (img->is_compressed()) {
		if (img->decompress() != OK)
			return Ref<Texture>();
	} else if (img->get_format() != Image::FORMAT_RGB8 && img->get_format() != Image::FORMAT_RGBA8) {
		img->convert(Image::FORMAT_RGBA8);
	}

	int width, height;
	if (img->get_width() > thumbnail_size && img->get_width() >= img->get_height()) {

		width = thumbnail_size;
		height = img->get_height() * thumbnail_size / img->get_width();
	} else if (img->get_height() > thumbnail_size && img->get_height() >= img->get_width()) {

		height = thumbnail_size;
		width = img->get_width() * thumbnail_size / img->get_height();
	} else {

		width = img->get_width();
		height = img->get_height();
	}

	img->resize(width, height);

	Ref<ImageTexture> ptex = Ref<ImageTexture>(memnew(ImageTexture));

	ptex->create_from_image(img, 0);
	return ptex;
}

EditorTexturePreviewPlugin::EditorTexturePreviewPlugin() {
}

////////////////////////////////////////////////////////////////////////////

bool EditorBitmapPreviewPlugin::handles(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "BitMap");
}

Ref<Texture> EditorBitmapPreviewPlugin::generate(const RES &p_from) {

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

	int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
	thumbnail_size *= EDSCALE;
	if (img->is_compressed()) {
		if (img->decompress() != OK)
			return Ref<Texture>();
	} else if (img->get_format() != Image::FORMAT_RGB8 && img->get_format() != Image::FORMAT_RGBA8) {
		img->convert(Image::FORMAT_RGBA8);
	}

	int width, height;
	if (img->get_width() > thumbnail_size && img->get_width() >= img->get_height()) {

		width = thumbnail_size;
		height = img->get_height() * thumbnail_size / img->get_width();
	} else if (img->get_height() > thumbnail_size && img->get_height() >= img->get_width()) {

		height = thumbnail_size;
		width = img->get_width() * thumbnail_size / img->get_height();
	} else {

		width = img->get_width();
		height = img->get_height();
	}

	img->resize(width, height);

	Ref<ImageTexture> ptex = Ref<ImageTexture>(memnew(ImageTexture));

	ptex->create_from_image(img, 0);
	return ptex;
}

EditorBitmapPreviewPlugin::EditorBitmapPreviewPlugin() {
}

///////////////////////////////////////////////////////////////////////////

bool EditorPackedScenePreviewPlugin::handles(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "PackedScene");
}
Ref<Texture> EditorPackedScenePreviewPlugin::generate(const RES &p_from) {

	return generate_from_path(p_from->get_path());
}

Ref<Texture> EditorPackedScenePreviewPlugin::generate_from_path(const String &p_path) {

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

	preview_done = true;
}

void EditorMaterialPreviewPlugin::_bind_methods() {

	ClassDB::bind_method("_preview_done", &EditorMaterialPreviewPlugin::_preview_done);
}

bool EditorMaterialPreviewPlugin::handles(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "Material"); //any material
}

Ref<Texture> EditorMaterialPreviewPlugin::generate(const RES &p_from) {

	Ref<Material> material = p_from;
	ERR_FAIL_COND_V(material.is_null(), Ref<Texture>());

	if (material->get_shader_mode() == Shader::MODE_SPATIAL) {

		VS::get_singleton()->mesh_surface_set_material(sphere, 0, material->get_rid());

		VS::get_singleton()->viewport_set_update_mode(viewport, VS::VIEWPORT_UPDATE_ONCE); //once used for capture

		preview_done = false;
		VS::get_singleton()->request_frame_drawn_callback(this, "_preview_done", Variant());

		while (!preview_done) {
			OS::get_singleton()->delay_usec(10);
		}

		Ref<Image> img = VS::get_singleton()->VS::get_singleton()->texture_get_data(viewport_texture);
		VS::get_singleton()->mesh_surface_set_material(sphere, 0, RID());

		ERR_FAIL_COND_V(!img.is_valid(), Ref<ImageTexture>());

		int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
		thumbnail_size *= EDSCALE;
		img->convert(Image::FORMAT_RGBA8);
		img->resize(thumbnail_size, thumbnail_size);
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

Ref<Texture> EditorScriptPreviewPlugin::generate(const RES &p_from) {

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
	int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
	thumbnail_size *= EDSCALE;
	Ref<Image> img;
	img.instance();
	img->create(thumbnail_size, thumbnail_size, 0, Image::FORMAT_RGBA8);

	Color bg_color = EditorSettings::get_singleton()->get("text_editor/highlighting/background_color");
	bg_color.a = 1.0;
	Color keyword_color = EditorSettings::get_singleton()->get("text_editor/highlighting/keyword_color");
	Color text_color = EditorSettings::get_singleton()->get("text_editor/highlighting/text_color");
	Color symbol_color = EditorSettings::get_singleton()->get("text_editor/highlighting/symbol_color");

	img->lock();

	for (int i = 0; i < thumbnail_size; i++) {
		for (int j = 0; j < thumbnail_size; j++) {
			img->set_pixel(i, j, bg_color);
		}
	}

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
				img->set_pixel(col, line * 2, bg_color.blend(ul));
				img->set_pixel(col, line * 2 + 1, color);

				prev_is_text = _is_text_char(c);
			}
		} else {

			prev_is_text = false;
			in_keyword = false;

			if (c == '\n') {
				col = 0;
				line++;
				if (line >= thumbnail_size / 2)
					break;
			} else if (c == '\t') {
				col += 3;
			}
		}
		col++;
	}

	img->unlock();

	Ref<ImageTexture> ptex = Ref<ImageTexture>(memnew(ImageTexture));

	ptex->create_from_image(img, 0);
	return ptex;
}

EditorScriptPreviewPlugin::EditorScriptPreviewPlugin() {
}
///////////////////////////////////////////////////////////////////

// FIXME: Needs to be rewritten for AudioStream in Godot 3.0+
#if 0
bool EditorSamplePreviewPlugin::handles(const String& p_type) const {

	return ClassDB::is_parent_class(p_type,"Sample");
}

Ref<Texture> EditorSamplePreviewPlugin::generate(const RES& p_from) {

	Ref<Sample> smp =p_from;
	ERR_FAIL_COND_V(smp.is_null(),Ref<Texture>());


	int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
	thumbnail_size*=EDSCALE;
	PoolVector<uint8_t> img;
	int w = thumbnail_size;
	int h = thumbnail_size;
	img.resize(w*h*3);

	PoolVector<uint8_t>::Write imgdata = img.write();
	uint8_t * imgw = imgdata.ptr();
	PoolVector<uint8_t> data = smp->get_data();
	PoolVector<uint8_t>::Read sampledata = data.read();
	const uint8_t *sdata=sampledata.ptr();

	bool stereo = smp->is_stereo();
	bool _16=smp->get_format()==Sample::FORMAT_PCM16;
	int len = smp->get_length();

	if (len<1)
		return Ref<Texture>();

	if (smp->get_format()==Sample::FORMAT_IMA_ADPCM) {

		struct IMA_ADPCM_State {

			int16_t step_index;
			int32_t predictor;
			/* values at loop point */
			int16_t loop_step_index;
			int32_t loop_predictor;
			int32_t last_nibble;
			int32_t loop_pos;
			int32_t window_ofs;
			const uint8_t *ptr;
		} ima_adpcm;

		ima_adpcm.step_index=0;
		ima_adpcm.predictor=0;
		ima_adpcm.loop_step_index=0;
		ima_adpcm.loop_predictor=0;
		ima_adpcm.last_nibble=-1;
		ima_adpcm.loop_pos=0x7FFFFFFF;
		ima_adpcm.window_ofs=0;
		ima_adpcm.ptr=NULL;


		for(int i=0;i<w;i++) {

			float max[2]={-1e10,-1e10};
			float min[2]={1e10,1e10};
			int from = i*len/w;
			int to = (i+1)*len/w;
			if (to>=len)
				to=len-1;

			for(int j=from;j<to;j++) {

				while(j>ima_adpcm.last_nibble) {

					static const int16_t _ima_adpcm_step_table[89] = {
						7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
						19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
						50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
						130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
						337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
						876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
						2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
						5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
						15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
					};

					static const int8_t _ima_adpcm_index_table[16] = {
						-1, -1, -1, -1, 2, 4, 6, 8,
						-1, -1, -1, -1, 2, 4, 6, 8
					};

					int16_t nibble,diff,step;

					ima_adpcm.last_nibble++;
					const uint8_t *src_ptr=sdata;

					int ofs = ima_adpcm.last_nibble>>1;

					if (stereo)
						ofs*=2;


					nibble = (ima_adpcm.last_nibble&1)?
							(src_ptr[ofs]>>4):(src_ptr[ofs]&0xF);
					step=_ima_adpcm_step_table[ima_adpcm.step_index];

					ima_adpcm.step_index += _ima_adpcm_index_table[nibble];
					if (ima_adpcm.step_index<0)
						ima_adpcm.step_index=0;
					if (ima_adpcm.step_index>88)
						ima_adpcm.step_index=88;

					diff = step >> 3 ;
					if (nibble & 1)
						diff += step >> 2 ;
					if (nibble & 2)
						diff += step >> 1 ;
					if (nibble & 4)
						diff += step ;
					if (nibble & 8)
						diff = -diff ;

					ima_adpcm.predictor+=diff;
					if (ima_adpcm.predictor<-0x8000)
						ima_adpcm.predictor=-0x8000;
					else if (ima_adpcm.predictor>0x7FFF)
						ima_adpcm.predictor=0x7FFF;


					/* store loop if there */
					if (ima_adpcm.last_nibble==ima_adpcm.loop_pos) {

						ima_adpcm.loop_step_index = ima_adpcm.step_index;
						ima_adpcm.loop_predictor = ima_adpcm.predictor;
					}

				}

				float v=ima_adpcm.predictor/32767.0;
				if (v>max[0])
					max[0]=v;
				if (v<min[0])
					min[0]=v;
			}
			max[0]*=0.8;
			min[0]*=0.8;

			for(int j=0;j<h;j++) {
				float v = (j/(float)h) * 2.0 - 1.0;
				uint8_t* imgofs = &imgw[(uint64_t(j)*w+i)*3];
				if (v>min[0] && v<max[0]) {
					imgofs[0]=255;
					imgofs[1]=150;
					imgofs[2]=80;
				} else {
					imgofs[0]=0;
					imgofs[1]=0;
					imgofs[2]=0;
				}
			}
		}
	} else {
		for(int i=0;i<w;i++) {
			// i trust gcc will optimize this loop
			float max[2]={-1e10,-1e10};
			float min[2]={1e10,1e10};
			int c=stereo?2:1;
			int from = uint64_t(i)*len/w;
			int to = (uint64_t(i)+1)*len/w;
			if (to>=len)
				to=len-1;

			if (_16) {
				const int16_t*src =(const int16_t*)sdata;

				for(int j=0;j<c;j++) {

					for(int k=from;k<=to;k++) {

						float v = src[uint64_t(k)*c+j]/32768.0;
						if (v>max[j])
							max[j]=v;
						if (v<min[j])
							min[j]=v;
					}

				}
			} else {

				const int8_t*src =(const int8_t*)sdata;

				for(int j=0;j<c;j++) {

					for(int k=from;k<=to;k++) {

						float v = src[uint64_t(k)*c+j]/128.0;
						if (v>max[j])
							max[j]=v;
						if (v<min[j])
							min[j]=v;
					}

				}
			}

			max[0]*=0.8;
			max[1]*=0.8;
			min[0]*=0.8;
			min[1]*=0.8;

			if (!stereo) {
				for(int j=0;j<h;j++) {
					float v = (j/(float)h) * 2.0 - 1.0;
					uint8_t* imgofs = &imgw[(j*w+i)*3];
					if (v>min[0] && v<max[0]) {
						imgofs[0]=255;
						imgofs[1]=150;
						imgofs[2]=80;
					} else {
						imgofs[0]=0;
						imgofs[1]=0;
						imgofs[2]=0;
					}
				}
			} else {

				for(int j=0;j<h;j++) {

					int half;
					float v;
					if (j<(h/2)) {
						half=0;
						v = (j/(float)(h/2)) * 2.0 - 1.0;
					} else {
						half=1;
						if( (float)(h/2) != 0 ) {
							v = ((j-(h/2))/(float)(h/2)) * 2.0 - 1.0;
						} else {
							v = ((j-(h/2))/(float)(1/2)) * 2.0 - 1.0;
						}
					}

					uint8_t* imgofs = &imgw[(j*w+i)*3];
					if (v>min[half] && v<max[half]) {
						imgofs[0]=255;
						imgofs[1]=150;
						imgofs[2]=80;
					} else {
						imgofs[0]=0;
						imgofs[1]=0;
						imgofs[2]=0;
					}
				}

			}

		}
	}

	imgdata = PoolVector<uint8_t>::Write();

	Ref<ImageTexture> ptex = Ref<ImageTexture>( memnew( ImageTexture));
	ptex->create_from_image(Image(w,h,0,Image::FORMAT_RGB8,img),0);
	return ptex;

}

EditorSamplePreviewPlugin::EditorSamplePreviewPlugin() {
}
#endif

///////////////////////////////////////////////////////////////////////////

void EditorMeshPreviewPlugin::_preview_done(const Variant &p_udata) {

	preview_done = true;
}

void EditorMeshPreviewPlugin::_bind_methods() {

	ClassDB::bind_method("_preview_done", &EditorMeshPreviewPlugin::_preview_done);
}
bool EditorMeshPreviewPlugin::handles(const String &p_type) const {

	return ClassDB::is_parent_class(p_type, "Mesh"); //any Mesh
}

Ref<Texture> EditorMeshPreviewPlugin::generate(const RES &p_from) {

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

	preview_done = false;
	VS::get_singleton()->request_frame_drawn_callback(this, "_preview_done", Variant());

	while (!preview_done) {
		OS::get_singleton()->delay_usec(10);
	}

	Ref<Image> img = VS::get_singleton()->VS::get_singleton()->texture_get_data(viewport_texture);
	ERR_FAIL_COND_V(img.is_null(), Ref<ImageTexture>());

	VS::get_singleton()->instance_set_base(mesh_instance, RID());

	int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
	thumbnail_size *= EDSCALE;
	img->convert(Image::FORMAT_RGBA8);
	img->resize(thumbnail_size, thumbnail_size);

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
