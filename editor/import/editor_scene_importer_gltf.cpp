#include "editor_scene_importer_gltf.h"
#include "io/json.h"
#include "math_defs.h"
#include "os/file_access.h"
#include "os/os.h"
#include "scene/3d/camera.h"
#include "scene/3d/mesh_instance.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/surface_tool.h"
#include "thirdparty/misc/base64.h"

uint32_t EditorSceneImporterGLTF::get_import_flags() const {

	return IMPORT_SCENE | IMPORT_ANIMATION;
}
void EditorSceneImporterGLTF::get_extensions(List<String> *r_extensions) const {

	r_extensions->push_back("gltf");
	r_extensions->push_back("glb");
}

Error EditorSceneImporterGLTF::_parse_json(const String &p_path, GLTFState &state) {

	Error err;
	FileAccessRef f = FileAccess::open(p_path, FileAccess::READ, &err);
	if (!f) {
		return err;
	}

	Vector<uint8_t> array;
	array.resize(f->get_len());
	f->get_buffer(array.ptrw(), array.size());
	String text;
	text.parse_utf8((const char *)array.ptr(), array.size());

	String err_txt;
	int err_line;
	Variant v;
	err = JSON::parse(text, v, err_txt, err_line);
	if (err != OK) {
		_err_print_error("", p_path.utf8().get_data(), err_line, err_txt.utf8().get_data(), ERR_HANDLER_SCRIPT);
		return err;
	}
	state.json = v;

	return OK;
}

Error EditorSceneImporterGLTF::_parse_glb(const String &p_path, GLTFState &state) {

	Error err;
	FileAccessRef f = FileAccess::open(p_path, FileAccess::READ, &err);
	if (!f) {
		return err;
	}

	uint32_t magic = f->get_32();
	ERR_FAIL_COND_V(magic != 0x46546C67, ERR_FILE_UNRECOGNIZED); //glTF
	f->get_32(); // version
	f->get_32(); // length

	uint32_t chunk_length = f->get_32();
	uint32_t chunk_type = f->get_32();

	ERR_FAIL_COND_V(chunk_type != 0x4E4F534A, ERR_PARSE_ERROR); //JSON
	Vector<uint8_t> json_data;
	json_data.resize(chunk_length);
	uint32_t len = f->get_buffer(json_data.ptrw(), chunk_length);
	ERR_FAIL_COND_V(len != chunk_length, ERR_FILE_CORRUPT);

	String text;
	text.parse_utf8((const char *)json_data.ptr(), json_data.size());

	String err_txt;
	int err_line;
	Variant v;
	err = JSON::parse(text, v, err_txt, err_line);
	if (err != OK) {
		_err_print_error("", p_path.utf8().get_data(), err_line, err_txt.utf8().get_data(), ERR_HANDLER_SCRIPT);
		return err;
	}

	state.json = v;

	//data?

	chunk_length = f->get_32();
	chunk_type = f->get_32();

	if (f->eof_reached()) {
		return OK; //all good
	}

	ERR_FAIL_COND_V(chunk_type != 0x004E4942, ERR_PARSE_ERROR); //BIN

	state.glb_data.resize(chunk_length);
	len = f->get_buffer(state.glb_data.ptrw(), chunk_length);
	ERR_FAIL_COND_V(len != chunk_length, ERR_FILE_CORRUPT);

	return OK;
}

static Vector3 _arr_to_vec3(const Array &p_array) {
	ERR_FAIL_COND_V(p_array.size() != 3, Vector3());
	return Vector3(p_array[0], p_array[1], p_array[2]);
}

static Quat _arr_to_quat(const Array &p_array) {
	ERR_FAIL_COND_V(p_array.size() != 4, Quat());
	return Quat(p_array[0], p_array[1], p_array[2], p_array[3]);
}

static Transform _arr_to_xform(const Array &p_array) {
	ERR_FAIL_COND_V(p_array.size() != 16, Transform());

	Transform xform;
	xform.basis.set_axis(Vector3::AXIS_X, Vector3(p_array[0], p_array[1], p_array[2]));
	xform.basis.set_axis(Vector3::AXIS_Y, Vector3(p_array[4], p_array[5], p_array[6]));
	xform.basis.set_axis(Vector3::AXIS_Z, Vector3(p_array[8], p_array[9], p_array[10]));
	xform.set_origin(Vector3(p_array[12], p_array[13], p_array[14]));

	return xform;
}

String EditorSceneImporterGLTF::_gen_unique_name(GLTFState &state, const String &p_name) {

	int index = 1;

	String name;
	while (true) {

		name = p_name;
		if (index > 1) {
			name += " " + itos(index);
		}
		if (!state.unique_names.has(name)) {
			break;
		}
		index++;
	}

	state.unique_names.insert(name);

	return name;
}

Error EditorSceneImporterGLTF::_parse_scenes(GLTFState &state) {

	ERR_FAIL_COND_V(!state.json.has("scenes"), ERR_FILE_CORRUPT);
	Array scenes = state.json["scenes"];
	for (int i = 0; i < 1; i++) { //only first scene is imported
		Dictionary s = scenes[i];
		ERR_FAIL_COND_V(!s.has("nodes"), ERR_UNAVAILABLE);
		Array nodes = s["nodes"];
		for (int j = 0; j < nodes.size(); j++) {
			state.root_nodes.push_back(nodes[j]);
		}

		if (s.has("name")) {
			state.scene_name = s["name"];
		}
	}

	return OK;
}

Error EditorSceneImporterGLTF::_parse_nodes(GLTFState &state) {

	ERR_FAIL_COND_V(!state.json.has("nodes"), ERR_FILE_CORRUPT);
	Array nodes = state.json["nodes"];
	for (int i = 0; i < nodes.size(); i++) {

		GLTFNode *node = memnew(GLTFNode);
		Dictionary n = nodes[i];

		print_line("node " + itos(i) + ": " + String(Variant(n)));
		if (n.has("name")) {
			node->name = n["name"];
		}
		if (n.has("camera")) {
			node->camera = n["camera"];
		}
		if (n.has("mesh")) {
			node->mesh = n["mesh"];
		}
		if (n.has("skin")) {
			node->skin = n["skin"];
			if (!state.skin_users.has(node->skin)) {
				state.skin_users[node->skin] = Vector<int>();
			}

			state.skin_users[node->skin].push_back(i);
		}
		if (n.has("matrix")) {
			node->xform = _arr_to_xform(n["matrix"]);

		} else {

			if (n.has("translation")) {
				node->translation = _arr_to_vec3(n["translation"]);
			}
			if (n.has("rotation")) {
				node->rotation = _arr_to_quat(n["rotation"]);
			}
			if (n.has("scale")) {
				node->scale = _arr_to_vec3(n["scale"]);
			}

			node->xform.basis = Basis(node->rotation);
			node->xform.basis.scale(node->scale);
			node->xform.origin = node->translation;
		}

		if (n.has("children")) {
			Array children = n["children"];
			for (int i = 0; i < children.size(); i++) {
				node->children.push_back(children[i]);
			}
		}

		state.nodes.push_back(node);
	}

	//build the hierarchy

	for (int i = 0; i < state.nodes.size(); i++) {

		for (int j = 0; j < state.nodes[i]->children.size(); j++) {
			int child = state.nodes[i]->children[j];
			ERR_FAIL_INDEX_V(child, state.nodes.size(), ERR_FILE_CORRUPT);
			ERR_CONTINUE(state.nodes[child]->parent != -1); //node already has a parent, wtf.

			state.nodes[child]->parent = i;
		}
	}

	return OK;
}

static Vector<uint8_t> _parse_base64_uri(const String &uri) {

	int start = uri.find(",");
	ERR_FAIL_COND_V(start == -1, Vector<uint8_t>());

	CharString substr = uri.right(start + 1).ascii();

	int strlen = substr.length();

	Vector<uint8_t> buf;
	buf.resize(strlen / 4 * 3 + 1 + 1);

	int len = base64_decode((char *)buf.ptr(), (char *)substr.get_data(), strlen);

	buf.resize(len);

	return buf;
}

Error EditorSceneImporterGLTF::_parse_buffers(GLTFState &state, const String &p_base_path) {

	if (!state.json.has("buffers"))
		return OK;

	Array buffers = state.json["buffers"];
	for (int i = 0; i < buffers.size(); i++) {

		if (i == 0 && state.glb_data.size()) {
			state.buffers.push_back(state.glb_data);

		} else {
			Dictionary buffer = buffers[i];
			if (buffer.has("uri")) {

				Vector<uint8_t> buffer_data;
				String uri = buffer["uri"];

				if (uri.findn("data:application/octet-stream;base64") == 0) {
					//embedded data
					buffer_data = _parse_base64_uri(uri);
				} else {

					uri = p_base_path.plus_file(uri).replace("\\", "/"); //fix for windows
					buffer_data = FileAccess::get_file_as_array(uri);
					ERR_FAIL_COND_V(buffer.size() == 0, ERR_PARSE_ERROR);
				}

				ERR_FAIL_COND_V(!buffer.has("byteLength"), ERR_PARSE_ERROR);
				int byteLength = buffer["byteLength"];
				ERR_FAIL_COND_V(byteLength < buffer_data.size(), ERR_PARSE_ERROR);
				state.buffers.push_back(buffer_data);
			}
		}
	}

	print_line("total buffers: " + itos(state.buffers.size()));

	return OK;
}

Error EditorSceneImporterGLTF::_parse_buffer_views(GLTFState &state) {

	ERR_FAIL_COND_V(!state.json.has("bufferViews"), ERR_FILE_CORRUPT);
	Array buffers = state.json["bufferViews"];
	for (int i = 0; i < buffers.size(); i++) {

		Dictionary d = buffers[i];

		GLTFBufferView buffer_view;

		ERR_FAIL_COND_V(!d.has("buffer"), ERR_PARSE_ERROR);
		buffer_view.buffer = d["buffer"];
		ERR_FAIL_COND_V(!d.has("byteLength"), ERR_PARSE_ERROR);
		buffer_view.byte_length = d["byteLength"];

		if (d.has("byteOffset")) {
			buffer_view.byte_offset = d["byteOffset"];
		}

		if (d.has("byteStride")) {
			buffer_view.byte_stride = d["byteStride"];
		}

		if (d.has("target")) {
			int target = d["target"];
			buffer_view.indices = target == ELEMENT_ARRAY_BUFFER;
		}

		state.buffer_views.push_back(buffer_view);
	}

	print_line("total buffer views: " + itos(state.buffer_views.size()));

	return OK;
}

EditorSceneImporterGLTF::GLTFType EditorSceneImporterGLTF::_get_type_from_str(const String &p_string) {

	if (p_string == "SCALAR")
		return TYPE_SCALAR;

	if (p_string == "VEC2")
		return TYPE_VEC2;
	if (p_string == "VEC3")
		return TYPE_VEC3;
	if (p_string == "VEC4")
		return TYPE_VEC4;

	if (p_string == "MAT2")
		return TYPE_MAT2;
	if (p_string == "MAT3")
		return TYPE_MAT3;
	if (p_string == "MAT4")
		return TYPE_MAT4;

	ERR_FAIL_V(TYPE_SCALAR);
}

Error EditorSceneImporterGLTF::_parse_accessors(GLTFState &state) {

	ERR_FAIL_COND_V(!state.json.has("accessors"), ERR_FILE_CORRUPT);
	Array accessors = state.json["accessors"];
	for (int i = 0; i < accessors.size(); i++) {

		Dictionary d = accessors[i];

		GLTFAccessor accessor;

		ERR_FAIL_COND_V(!d.has("componentType"), ERR_PARSE_ERROR);
		accessor.component_type = d["componentType"];
		ERR_FAIL_COND_V(!d.has("count"), ERR_PARSE_ERROR);
		accessor.count = d["count"];
		ERR_FAIL_COND_V(!d.has("type"), ERR_PARSE_ERROR);
		accessor.type = _get_type_from_str(d["type"]);

		if (d.has("bufferView")) {
			accessor.buffer_view = d["bufferView"]; //optional because it may be sparse...
		}

		if (d.has("byteOffset")) {
			accessor.byte_offset = d["byteOffset"];
		}

		if (d.has("max")) {
			accessor.max = d["max"];
		}

		if (d.has("min")) {
			accessor.min = d["min"];
		}

		if (d.has("sparse")) {
			//eeh..

			Dictionary s = d["sparse"];

			ERR_FAIL_COND_V(!d.has("count"), ERR_PARSE_ERROR);
			accessor.sparse_count = d["count"];
			ERR_FAIL_COND_V(!d.has("indices"), ERR_PARSE_ERROR);
			Dictionary si = d["indices"];

			ERR_FAIL_COND_V(!si.has("bufferView"), ERR_PARSE_ERROR);
			accessor.sparse_indices_buffer_view = si["bufferView"];
			ERR_FAIL_COND_V(!si.has("componentType"), ERR_PARSE_ERROR);
			accessor.sparse_indices_component_type = si["componentType"];

			if (si.has("byteOffset")) {
				accessor.sparse_indices_byte_offset = si["byteOffset"];
			}

			ERR_FAIL_COND_V(!d.has("values"), ERR_PARSE_ERROR);
			Dictionary sv = d["values"];

			ERR_FAIL_COND_V(!sv.has("bufferView"), ERR_PARSE_ERROR);
			accessor.sparse_values_buffer_view = sv["bufferView"];
			if (sv.has("byteOffset")) {
				accessor.sparse_values_byte_offset = sv["byteOffset"];
			}
		}

		state.accessors.push_back(accessor);
	}

	print_line("total accessors: " + itos(state.accessors.size()));

	return OK;
}

String EditorSceneImporterGLTF::_get_component_type_name(uint32_t p_component) {

	switch (p_component) {
		case COMPONENT_TYPE_BYTE: return "Byte";
		case COMPONENT_TYPE_UNSIGNED_BYTE: return "UByte";
		case COMPONENT_TYPE_SHORT: return "Short";
		case COMPONENT_TYPE_UNSIGNED_SHORT: return "UShort";
		case COMPONENT_TYPE_INT: return "Int";
		case COMPONENT_TYPE_FLOAT: return "Float";
	}

	return "<Error>";
}

String EditorSceneImporterGLTF::_get_type_name(GLTFType p_component) {

	static const char *names[] = {
		"float",
		"vec2",
		"vec3",
		"vec4",
		"mat2",
		"mat3",
		"mat4"
	};

	return names[p_component];
}

Error EditorSceneImporterGLTF::_decode_buffer_view(GLTFState &state, int p_buffer_view, double *dst, int skip_every, int skip_bytes, int element_size, int count, GLTFType type, int component_count, int component_type, int component_size, bool normalized, int byte_offset, bool for_vertex) {

	const GLTFBufferView &bv = state.buffer_views[p_buffer_view];

	int stride = bv.byte_stride ? bv.byte_stride : element_size;
	if (for_vertex && stride % 4) {
		stride += 4 - (stride % 4); //according to spec must be multiple of 4
	}

	ERR_FAIL_INDEX_V(bv.buffer, state.buffers.size(), ERR_PARSE_ERROR);

	uint32_t offset = bv.byte_offset + byte_offset;
	Vector<uint8_t> buffer = state.buffers[bv.buffer]; //copy on write, so no performance hit
	const uint8_t *bufptr = buffer.ptr();

	//use to debug
	//print_line("type " + _get_type_name(type) + " component type: " + _get_component_type_name(component_type) + " stride: " + itos(stride) + " amount " + itos(count));
	print_line("accessor offset" + itos(byte_offset) + " view offset: " + itos(bv.byte_offset) + " total buffer len: " + itos(buffer.size()) + " view len " + itos(bv.byte_length));

	int buffer_end = (stride * (count - 1)) + element_size;
	ERR_FAIL_COND_V(buffer_end > bv.byte_length, ERR_PARSE_ERROR);

	ERR_FAIL_COND_V((int)(offset + buffer_end) > buffer.size(), ERR_PARSE_ERROR);

	//fill everything as doubles

	for (int i = 0; i < count; i++) {

		const uint8_t *src = &bufptr[offset + i * stride];

		for (int j = 0; j < component_count; j++) {

			if (skip_every && j > 0 && (j % skip_every) == 0) {
				src += skip_bytes;
			}

			double d = 0;

			switch (component_type) {
				case COMPONENT_TYPE_BYTE: {
					int8_t b = int8_t(*src);
					if (normalized) {
						d = (double(b) / 128.0);
					} else {
						d = double(b);
					}
				} break;
				case COMPONENT_TYPE_UNSIGNED_BYTE: {
					uint8_t b = *src;
					if (normalized) {
						d = (double(b) / 255.0);
					} else {
						d = double(b);
					}
				} break;
				case COMPONENT_TYPE_SHORT: {
					int16_t s = *(int16_t *)src;
					if (normalized) {
						d = (double(s) / 32768.0);
					} else {
						d = double(s);
					}
				} break;
				case COMPONENT_TYPE_UNSIGNED_SHORT: {
					uint16_t s = *(uint16_t *)src;
					if (normalized) {
						d = (double(s) / 65535.0);
					} else {
						d = double(s);
					}

				} break;
				case COMPONENT_TYPE_INT: {
					d = *(int *)src;
				} break;
				case COMPONENT_TYPE_FLOAT: {
					d = *(float *)src;
				} break;
			}

			*dst++ = d;
			src += component_size;
		}
	}

	return OK;
}

int EditorSceneImporterGLTF::_get_component_type_size(int component_type) {

	switch (component_type) {
		case COMPONENT_TYPE_BYTE: return 1; break;
		case COMPONENT_TYPE_UNSIGNED_BYTE: return 1; break;
		case COMPONENT_TYPE_SHORT: return 2; break;
		case COMPONENT_TYPE_UNSIGNED_SHORT: return 2; break;
		case COMPONENT_TYPE_INT: return 4; break;
		case COMPONENT_TYPE_FLOAT: return 4; break;
		default: { ERR_FAIL_V(0); }
	}
	return 0;
}

Vector<double> EditorSceneImporterGLTF::_decode_accessor(GLTFState &state, int p_accessor, bool p_for_vertex) {

	//spec, for reference:
	//https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#data-alignment

	ERR_FAIL_INDEX_V(p_accessor, state.accessors.size(), Vector<double>());

	const GLTFAccessor &a = state.accessors[p_accessor];

	int component_count_for_type[7] = {
		1, 2, 3, 4, 4, 9, 16
	};

	int component_count = component_count_for_type[a.type];
	int component_size = _get_component_type_size(a.component_type);
	ERR_FAIL_COND_V(component_size == 0, Vector<double>());
	int element_size = component_count * component_size;

	int skip_every = 0;
	int skip_bytes = 0;
	//special case of alignments, as described in spec
	switch (a.component_type) {
		case COMPONENT_TYPE_BYTE:
		case COMPONENT_TYPE_UNSIGNED_BYTE: {

			if (a.type == TYPE_MAT2) {
				skip_every = 2;
				skip_bytes = 2;
				element_size = 8; //override for this case
			}
			if (a.type == TYPE_MAT3) {
				skip_every = 3;
				skip_bytes = 1;
				element_size = 12; //override for this case
			}

		} break;
		case COMPONENT_TYPE_SHORT:
		case COMPONENT_TYPE_UNSIGNED_SHORT: {
			if (a.type == TYPE_MAT3) {
				skip_every = 6;
				skip_bytes = 4;
				element_size = 16; //override for this case
			}
		} break;
		default: {}
	}

	Vector<double> dst_buffer;
	dst_buffer.resize(component_count * a.count);
	double *dst = dst_buffer.ptrw();

	if (a.buffer_view >= 0) {

		ERR_FAIL_INDEX_V(a.buffer_view, state.buffer_views.size(), Vector<double>());

		Error err = _decode_buffer_view(state, a.buffer_view, dst, skip_every, skip_bytes, element_size, a.count, a.type, component_count, a.component_type, component_size, a.normalized, a.byte_offset, p_for_vertex);
		if (err != OK)
			return Vector<double>();

	} else {
		//fill with zeros, as bufferview is not defined.
		for (int i = 0; i < (a.count * component_count); i++) {
			dst_buffer[i] = 0;
		}
	}

	if (a.sparse_count > 0) {
		// I could not find any file using this, so this code is so far untested
		Vector<double> indices;
		indices.resize(a.sparse_count);
		int indices_component_size = _get_component_type_size(a.sparse_indices_component_type);

		Error err = _decode_buffer_view(state, a.sparse_indices_buffer_view, indices.ptrw(), 0, 0, indices_component_size, a.sparse_count, TYPE_SCALAR, 1, a.sparse_indices_component_type, indices_component_size, false, a.sparse_indices_byte_offset, false);
		if (err != OK)
			return Vector<double>();

		Vector<double> data;
		data.resize(component_count * a.sparse_count);
		err = _decode_buffer_view(state, a.sparse_values_buffer_view, data.ptrw(), skip_every, skip_bytes, element_size, a.sparse_count, a.type, component_count, a.component_type, component_size, a.normalized, a.sparse_values_byte_offset, p_for_vertex);
		if (err != OK)
			return Vector<double>();

		for (int i = 0; i < indices.size(); i++) {
			int write_offset = int(indices[i]) * component_count;

			for (int j = 0; j < component_count; j++) {
				dst[write_offset + j] = data[i * component_count + j];
			}
		}
	}

	return dst_buffer;
}

PoolVector<int> EditorSceneImporterGLTF::_decode_accessor_as_ints(GLTFState &state, int p_accessor, bool p_for_vertex) {

	Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	PoolVector<int> ret;
	if (attribs.size() == 0)
		return ret;
	const double *attribs_ptr = attribs.ptr();
	int ret_size = attribs.size();
	ret.resize(ret_size);
	{
		PoolVector<int>::Write w = ret.write();
		for (int i = 0; i < ret_size; i++) {
			w[i] = int(attribs_ptr[i]);
		}
	}
	return ret;
}

PoolVector<float> EditorSceneImporterGLTF::_decode_accessor_as_floats(GLTFState &state, int p_accessor, bool p_for_vertex) {

	Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	PoolVector<float> ret;
	if (attribs.size() == 0)
		return ret;
	const double *attribs_ptr = attribs.ptr();
	int ret_size = attribs.size();
	ret.resize(ret_size);
	{
		PoolVector<float>::Write w = ret.write();
		for (int i = 0; i < ret_size; i++) {
			w[i] = float(attribs_ptr[i]);
		}
	}
	return ret;
}

PoolVector<Vector2> EditorSceneImporterGLTF::_decode_accessor_as_vec2(GLTFState &state, int p_accessor, bool p_for_vertex) {

	Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	PoolVector<Vector2> ret;
	if (attribs.size() == 0)
		return ret;
	ERR_FAIL_COND_V(attribs.size() % 2 != 0, ret);
	const double *attribs_ptr = attribs.ptr();
	int ret_size = attribs.size() / 2;
	ret.resize(ret_size);
	{
		PoolVector<Vector2>::Write w = ret.write();
		for (int i = 0; i < ret_size; i++) {
			w[i] = Vector2(attribs_ptr[i * 2 + 0], attribs_ptr[i * 2 + 1]);
		}
	}
	return ret;
}

PoolVector<Vector3> EditorSceneImporterGLTF::_decode_accessor_as_vec3(GLTFState &state, int p_accessor, bool p_for_vertex) {

	Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	PoolVector<Vector3> ret;
	if (attribs.size() == 0)
		return ret;
	ERR_FAIL_COND_V(attribs.size() % 3 != 0, ret);
	const double *attribs_ptr = attribs.ptr();
	int ret_size = attribs.size() / 3;
	ret.resize(ret_size);
	{
		PoolVector<Vector3>::Write w = ret.write();
		for (int i = 0; i < ret_size; i++) {
			w[i] = Vector3(attribs_ptr[i * 3 + 0], attribs_ptr[i * 3 + 1], attribs_ptr[i * 3 + 2]);
		}
	}
	return ret;
}
PoolVector<Color> EditorSceneImporterGLTF::_decode_accessor_as_color(GLTFState &state, int p_accessor, bool p_for_vertex) {

	Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	PoolVector<Color> ret;
	if (attribs.size() == 0)
		return ret;
	ERR_FAIL_COND_V(attribs.size() % 4 != 0, ret);
	const double *attribs_ptr = attribs.ptr();
	int ret_size = attribs.size() / 4;
	ret.resize(ret_size);
	{
		PoolVector<Color>::Write w = ret.write();
		for (int i = 0; i < ret_size; i++) {
			w[i] = Color(attribs_ptr[i * 4 + 0], attribs_ptr[i * 4 + 1], attribs_ptr[i * 4 + 2], attribs_ptr[i * 4 + 3]);
		}
	}
	return ret;
}
Vector<Quat> EditorSceneImporterGLTF::_decode_accessor_as_quat(GLTFState &state, int p_accessor, bool p_for_vertex) {

	Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	Vector<Quat> ret;
	if (attribs.size() == 0)
		return ret;
	ERR_FAIL_COND_V(attribs.size() % 4 != 0, ret);
	const double *attribs_ptr = attribs.ptr();
	int ret_size = attribs.size() / 4;
	ret.resize(ret_size);
	{
		for (int i = 0; i < ret_size; i++) {
			ret[i] = Quat(attribs_ptr[i * 4 + 0], attribs_ptr[i * 4 + 1], attribs_ptr[i * 4 + 2], attribs_ptr[i * 4 + 3]);
		}
	}
	return ret;
}
Vector<Transform2D> EditorSceneImporterGLTF::_decode_accessor_as_xform2d(GLTFState &state, int p_accessor, bool p_for_vertex) {

	Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	Vector<Transform2D> ret;
	if (attribs.size() == 0)
		return ret;
	ERR_FAIL_COND_V(attribs.size() % 4 != 0, ret);
	ret.resize(attribs.size() / 4);
	for (int i = 0; i < ret.size(); i++) {
		ret[i][0] = Vector2(attribs[i * 4 + 0], attribs[i * 4 + 1]);
		ret[i][1] = Vector2(attribs[i * 4 + 2], attribs[i * 4 + 3]);
	}
	return ret;
}

Vector<Basis> EditorSceneImporterGLTF::_decode_accessor_as_basis(GLTFState &state, int p_accessor, bool p_for_vertex) {

	Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	Vector<Basis> ret;
	if (attribs.size() == 0)
		return ret;
	ERR_FAIL_COND_V(attribs.size() % 9 != 0, ret);
	ret.resize(attribs.size() / 9);
	for (int i = 0; i < ret.size(); i++) {
		ret[i].set_axis(0, Vector3(attribs[i * 9 + 0], attribs[i * 9 + 1], attribs[i * 9 + 2]));
		ret[i].set_axis(1, Vector3(attribs[i * 9 + 3], attribs[i * 9 + 4], attribs[i * 9 + 5]));
		ret[i].set_axis(2, Vector3(attribs[i * 9 + 6], attribs[i * 9 + 7], attribs[i * 9 + 8]));
	}
	return ret;
}
Vector<Transform> EditorSceneImporterGLTF::_decode_accessor_as_xform(GLTFState &state, int p_accessor, bool p_for_vertex) {

	Vector<double> attribs = _decode_accessor(state, p_accessor, p_for_vertex);
	Vector<Transform> ret;
	if (attribs.size() == 0)
		return ret;
	ERR_FAIL_COND_V(attribs.size() % 16 != 0, ret);
	ret.resize(attribs.size() / 16);
	for (int i = 0; i < ret.size(); i++) {
		ret[i].basis.set_axis(0, Vector3(attribs[i * 16 + 0], attribs[i * 16 + 1], attribs[i * 16 + 2]));
		ret[i].basis.set_axis(1, Vector3(attribs[i * 16 + 4], attribs[i * 16 + 5], attribs[i * 16 + 6]));
		ret[i].basis.set_axis(2, Vector3(attribs[i * 16 + 8], attribs[i * 16 + 9], attribs[i * 16 + 10]));
		ret[i].set_origin(Vector3(attribs[i * 16 + 12], attribs[i * 16 + 13], attribs[i * 16 + 14]));
	}
	return ret;
}

Error EditorSceneImporterGLTF::_parse_meshes(GLTFState &state) {

	if (!state.json.has("meshes"))
		return OK;

	Array meshes = state.json["meshes"];
	for (int i = 0; i < meshes.size(); i++) {

		print_line("on mesh: " + itos(i));
		Dictionary d = meshes[i];

		GLTFMesh mesh;
		mesh.mesh.instance();

		ERR_FAIL_COND_V(!d.has("primitives"), ERR_PARSE_ERROR);

		Array primitives = d["primitives"];

		for (int j = 0; j < primitives.size(); j++) {

			Dictionary p = primitives[j];

			Array array;
			array.resize(Mesh::ARRAY_MAX);

			ERR_FAIL_COND_V(!p.has("attributes"), ERR_PARSE_ERROR);

			Dictionary a = p["attributes"];

			Mesh::PrimitiveType primitive = Mesh::PRIMITIVE_TRIANGLES;
			if (p.has("mode")) {
				int mode = p["mode"];
				ERR_FAIL_INDEX_V(mode, 7, ERR_FILE_CORRUPT);
				static const Mesh::PrimitiveType primitives[7] = {
					Mesh::PRIMITIVE_POINTS,
					Mesh::PRIMITIVE_LINES,
					Mesh::PRIMITIVE_LINE_LOOP,
					Mesh::PRIMITIVE_LINE_STRIP,
					Mesh::PRIMITIVE_TRIANGLES,
					Mesh::PRIMITIVE_TRIANGLE_STRIP,
					Mesh::PRIMITIVE_TRIANGLE_FAN,
				};

				primitive = primitives[mode];
			}

			if (a.has("POSITION")) {
				array[Mesh::ARRAY_VERTEX] = _decode_accessor_as_vec3(state, a["POSITION"], true);
			}
			if (a.has("NORMAL")) {
				array[Mesh::ARRAY_NORMAL] = _decode_accessor_as_vec3(state, a["NORMAL"], true);
			}
			if (a.has("TANGENT")) {
				array[Mesh::ARRAY_TANGENT] = _decode_accessor_as_floats(state, a["TANGENT"], true);
			}
			if (a.has("TEXCOORD_0")) {
				array[Mesh::ARRAY_TEX_UV] = _decode_accessor_as_vec2(state, a["TEXCOORD_0"], true);
			}
			if (a.has("TEXCOORD_1")) {
				array[Mesh::ARRAY_TEX_UV2] = _decode_accessor_as_vec2(state, a["TEXCOORD_1"], true);
			}
			if (a.has("COLOR_0")) {
				array[Mesh::ARRAY_COLOR] = _decode_accessor_as_color(state, a["COLOR_0"], true);
			}
			if (a.has("JOINTS_0")) {
				array[Mesh::ARRAY_BONES] = _decode_accessor_as_ints(state, a["JOINTS_0"], true);
			}
			if (a.has("WEIGHTS_0")) {
				PoolVector<float> weights = _decode_accessor_as_floats(state, a["WEIGHTS_0"], true);
				{ //gltf does not seem to normalize the weights for some reason..
					int wc = weights.size();
					PoolVector<float>::Write w = weights.write();
					for (int i = 0; i < wc; i += 4) {
						float total = 0.0;
						total += w[i + 0];
						total += w[i + 1];
						total += w[i + 2];
						total += w[i + 3];
						if (total > 0.0) {
							w[i + 0] /= total;
							w[i + 1] /= total;
							w[i + 2] /= total;
							w[i + 3] /= total;
						}
					}
				}
				array[Mesh::ARRAY_WEIGHTS] = weights;
			}

			if (p.has("indices")) {

				PoolVector<int> indices = _decode_accessor_as_ints(state, p["indices"], false);

				if (primitive == Mesh::PRIMITIVE_TRIANGLES) {
					//swap around indices, convert ccw to cw for front face

					int is = indices.size();
					PoolVector<int>::Write w = indices.write();
					for (int i = 0; i < is; i += 3) {
						SWAP(w[i + 1], w[i + 2]);
					}
				}
				array[Mesh::ARRAY_INDEX] = indices;
			} else if (primitive == Mesh::PRIMITIVE_TRIANGLES) {
				//generate indices because they need to be swapped for CW/CCW
				PoolVector<Vector3> vertices = array[Mesh::ARRAY_VERTEX];
				ERR_FAIL_COND_V(vertices.size() == 0, ERR_PARSE_ERROR);
				PoolVector<int> indices;
				int vs = vertices.size();
				indices.resize(vs);
				{
					PoolVector<int>::Write w = indices.write();
					for (int i = 0; i < vs; i += 3) {
						w[i] = i;
						w[i + 1] = i + 2;
						w[i + 2] = i + 1;
					}
				}
				array[Mesh::ARRAY_INDEX] = indices;
			}

			bool generated_tangents = false;
			Variant erased_indices;

			if (primitive == Mesh::PRIMITIVE_TRIANGLES && !a.has("TANGENT") && a.has("TEXCOORD_0") && a.has("NORMAL")) {
				//must generate mikktspace tangents.. ergh..
				Ref<SurfaceTool> st;
				st.instance();
				st->create_from_triangle_arrays(array);
				if (p.has("targets")) {
					//morph targets should not be reindexed, as array size might differ
					//removing indices is the best bet here
					st->deindex();
					erased_indices = a[Mesh::ARRAY_INDEX];
					a[Mesh::ARRAY_INDEX] = Variant();
				}
				st->generate_tangents();
				array = st->commit_to_arrays();
				generated_tangents = true;
			}

			Array morphs;
			//blend shapes
			if (p.has("targets")) {
				print_line("has targets!");
				Array targets = p["targets"];

				if (j == 0) {
					for (int k = 0; k < targets.size(); k++) {
						mesh.mesh->add_blend_shape(String("morph_") + itos(k));
					}
				}

				for (int k = 0; k < targets.size(); k++) {

					Dictionary t = targets[k];

					Array array_copy;
					array_copy.resize(Mesh::ARRAY_MAX);

					for (int l = 0; l < Mesh::ARRAY_MAX; l++) {
						array_copy[l] = array[l];
					}

					array_copy[Mesh::ARRAY_INDEX] = Variant();

					if (t.has("POSITION")) {
						array_copy[Mesh::ARRAY_VERTEX] = _decode_accessor_as_vec3(state, t["POSITION"], true);
					}
					if (t.has("NORMAL")) {
						array_copy[Mesh::ARRAY_NORMAL] = _decode_accessor_as_vec3(state, t["NORMAL"], true);
					}
					if (t.has("TANGENT")) {
						PoolVector<Vector3> tangents_v3 = _decode_accessor_as_vec3(state, t["TANGENT"], true);
						PoolVector<float> tangents_v4;
						PoolVector<float> src_tangents = array[Mesh::ARRAY_TANGENT];
						ERR_FAIL_COND_V(src_tangents.size() == 0, ERR_PARSE_ERROR);

						{

							int size4 = src_tangents.size();
							tangents_v4.resize(size4);
							PoolVector<float>::Write w4 = tangents_v4.write();

							PoolVector<Vector3>::Read r3 = tangents_v3.read();
							PoolVector<float>::Read r4 = src_tangents.read();

							for (int l = 0; l < size4 / 4; l++) {

								w4[l * 4 + 0] = r3[l].x;
								w4[l * 4 + 1] = r3[l].y;
								w4[l * 4 + 2] = r3[l].z;
								w4[l * 4 + 3] = r4[l * 4 + 3]; //copy flip value
							}
						}

						array_copy[Mesh::ARRAY_TANGENT] = tangents_v4;
					}

					if (generated_tangents) {
						Ref<SurfaceTool> st;
						st.instance();
						array_copy[Mesh::ARRAY_INDEX] = erased_indices; //needed for tangent generation, erased by deindex
						st->create_from_triangle_arrays(array_copy);
						st->deindex();
						st->generate_tangents();
						array_copy = st->commit_to_arrays();
					}

					morphs.push_back(array_copy);
				}
			}

			//just add it
			mesh.mesh->add_surface_from_arrays(primitive, array, morphs);

			if (p.has("material")) {
				int material = p["material"];
				ERR_FAIL_INDEX_V(material, state.materials.size(), ERR_FILE_CORRUPT);
				Ref<Material> mat = state.materials[material];

				mesh.mesh->surface_set_material(mesh.mesh->get_surface_count() - 1, mat);
			}
		}

		if (d.has("weights")) {
			Array weights = d["weights"];
			ERR_FAIL_COND_V(mesh.mesh->get_blend_shape_count() != weights.size(), ERR_PARSE_ERROR);
			mesh.blend_weights.resize(weights.size());
			for (int j = 0; j < weights.size(); j++) {
				mesh.blend_weights[j] = weights[j];
			}
		}

		state.meshes.push_back(mesh);
	}

	print_line("total meshes: " + itos(state.meshes.size()));

	return OK;
}

Error EditorSceneImporterGLTF::_parse_images(GLTFState &state, const String &p_base_path) {

	if (!state.json.has("images"))
		return OK;

	Array images = state.json["images"];
	for (int i = 0; i < images.size(); i++) {

		Dictionary d = images[i];

		String mimetype;
		if (d.has("mimeType")) {
			mimetype = d["mimeType"];
		}

		Vector<uint8_t> data;
		const uint8_t *data_ptr = NULL;
		int data_size = 0;

		if (d.has("uri")) {
			String uri = d["uri"];

			if (uri.findn("data:application/octet-stream;base64") == 0) {
				//embedded data
				data = _parse_base64_uri(uri);
				data_ptr = data.ptr();
				data_size = data.size();
			} else {

				uri = p_base_path.plus_file(uri).replace("\\", "/"); //fix for windows
				Ref<Texture> texture = ResourceLoader::load(uri);
				state.images.push_back(texture);
				continue;
			}
		}

		if (d.has("bufferView")) {
			int bvi = d["bufferView"];

			ERR_FAIL_INDEX_V(bvi, state.buffer_views.size(), ERR_PARAMETER_RANGE_ERROR);

			GLTFBufferView &bv = state.buffer_views[bvi];

			int bi = bv.buffer;
			ERR_FAIL_INDEX_V(bi, state.buffers.size(), ERR_PARAMETER_RANGE_ERROR);

			ERR_FAIL_COND_V(bv.byte_offset + bv.byte_length > state.buffers[bi].size(), ERR_FILE_CORRUPT);

			data_ptr = &state.buffers[bi][bv.byte_offset];
			data_size = bv.byte_length;
		}

		ERR_FAIL_COND_V(mimetype == "", ERR_FILE_CORRUPT);

		if (mimetype.findn("png") != -1) {
			//is a png
			Ref<Image> img = Image::_png_mem_loader_func(data_ptr, data_size);

			ERR_FAIL_COND_V(img.is_null(), ERR_FILE_CORRUPT);

			Ref<ImageTexture> t;
			t.instance();
			t->create_from_image(img);

			state.images.push_back(t);
			continue;
		}

		if (mimetype.findn("jpg") != -1) {
			//is a jpg
			Ref<Image> img = Image::_jpg_mem_loader_func(data_ptr, data_size);

			ERR_FAIL_COND_V(img.is_null(), ERR_FILE_CORRUPT);

			Ref<ImageTexture> t;
			t.instance();
			t->create_from_image(img);

			state.images.push_back(t);

			continue;
		}

		ERR_FAIL_V(ERR_FILE_CORRUPT);
	}

	print_line("total images: " + itos(state.images.size()));

	return OK;
}

Error EditorSceneImporterGLTF::_parse_textures(GLTFState &state) {

	if (!state.json.has("textures"))
		return OK;

	Array textures = state.json["textures"];
	for (int i = 0; i < textures.size(); i++) {

		Dictionary d = textures[i];

		ERR_FAIL_COND_V(!d.has("source"), ERR_PARSE_ERROR);

		GLTFTexture t;
		t.src_image = d["source"];
		state.textures.push_back(t);
	}

	return OK;
}

Ref<Texture> EditorSceneImporterGLTF::_get_texture(GLTFState &state, int p_texture) {
	ERR_FAIL_INDEX_V(p_texture, state.textures.size(), Ref<Texture>());
	int image = state.textures[p_texture].src_image;

	ERR_FAIL_INDEX_V(image, state.images.size(), Ref<Texture>());

	return state.images[image];
}

Error EditorSceneImporterGLTF::_parse_materials(GLTFState &state) {

	if (!state.json.has("materials"))
		return OK;

	Array materials = state.json["materials"];
	for (int i = 0; i < materials.size(); i++) {

		Dictionary d = materials[i];

		Ref<SpatialMaterial> material;
		material.instance();
		if (d.has("name")) {
			material->set_name(d["name"]);
		}

		if (d.has("pbrMetallicRoughness")) {

			Dictionary mr = d["pbrMetallicRoughness"];
			if (mr.has("baseColorFactor")) {
				Array arr = mr["baseColorFactor"];
				ERR_FAIL_COND_V(arr.size() != 4, ERR_PARSE_ERROR);
				Color c = Color(arr[0], arr[1], arr[2], arr[3]).to_srgb();

				material->set_albedo(c);
			}

			if (mr.has("baseColorTexture")) {
				Dictionary bct = mr["baseColorTexture"];
				if (bct.has("index")) {
					material->set_texture(SpatialMaterial::TEXTURE_ALBEDO, _get_texture(state, bct["index"]));
				}
				if (!mr.has("baseColorFactor")) {
					material->set_albedo(Color(1, 1, 1));
				}
			}

			if (mr.has("metallicFactor")) {

				material->set_metallic(mr["metallicFactor"]);
			}
			if (mr.has("roughnessFactor")) {

				material->set_roughness(mr["roughnessFactor"]);
			}

			if (mr.has("metallicRoughnessTexture")) {
				Dictionary bct = mr["metallicRoughnessTexture"];
				if (bct.has("index")) {
					Ref<Texture> t = _get_texture(state, bct["index"]);
					material->set_texture(SpatialMaterial::TEXTURE_METALLIC, t);
					material->set_metallic_texture_channel(SpatialMaterial::TEXTURE_CHANNEL_BLUE);
					material->set_texture(SpatialMaterial::TEXTURE_ROUGHNESS, t);
					material->set_roughness_texture_channel(SpatialMaterial::TEXTURE_CHANNEL_GREEN);
					if (!mr.has("metallicFactor")) {
						material->set_metallic(1);
					}
					if (!mr.has("roughnessFactor")) {
						material->set_roughness(1);
					}
				}
			}
		}

		if (d.has("normalTexture")) {
			Dictionary bct = d["normalTexture"];
			if (bct.has("index")) {
				material->set_texture(SpatialMaterial::TEXTURE_NORMAL, _get_texture(state, bct["index"]));
				material->set_feature(SpatialMaterial::FEATURE_NORMAL_MAPPING, true);
			}
			if (bct.has("scale")) {
				material->set_normal_scale(bct["scale"]);
			}
		}
		if (d.has("occlusionTexture")) {
			Dictionary bct = d["occlusionTexture"];
			if (bct.has("index")) {
				material->set_texture(SpatialMaterial::TEXTURE_AMBIENT_OCCLUSION, _get_texture(state, bct["index"]));
				material->set_ao_texture_channel(SpatialMaterial::TEXTURE_CHANNEL_RED);
				material->set_feature(SpatialMaterial::FEATURE_AMBIENT_OCCLUSION, true);
			}
		}

		if (d.has("emissiveFactor")) {
			Array arr = d["emissiveFactor"];
			ERR_FAIL_COND_V(arr.size() != 3, ERR_PARSE_ERROR);
			Color c = Color(arr[0], arr[1], arr[2]).to_srgb();
			material->set_feature(SpatialMaterial::FEATURE_EMISSION, true);

			material->set_emission(c);
		}

		if (d.has("emissiveTexture")) {
			Dictionary bct = d["emissiveTexture"];
			if (bct.has("index")) {
				material->set_texture(SpatialMaterial::TEXTURE_EMISSION, _get_texture(state, bct["index"]));
				material->set_feature(SpatialMaterial::FEATURE_EMISSION, true);
				material->set_emission(Color(0, 0, 0));
			}
		}

		if (d.has("doubleSided")) {
			bool ds = d["doubleSided"];
			if (ds) {
				material->set_cull_mode(SpatialMaterial::CULL_DISABLED);
			}
		}

		if (d.has("alphaMode")) {
			String am = d["alphaMode"];
			if (am != "OPAQUE") {
				material->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
			}
		}

		state.materials.push_back(material);
	}

	print_line("total materials: " + itos(state.materials.size()));

	return OK;
}

Error EditorSceneImporterGLTF::_parse_skins(GLTFState &state) {

	if (!state.json.has("skins"))
		return OK;

	Array skins = state.json["skins"];
	for (int i = 0; i < skins.size(); i++) {

		Dictionary d = skins[i];

		GLTFSkin skin;

		ERR_FAIL_COND_V(!d.has("joints"), ERR_PARSE_ERROR);

		Array joints = d["joints"];
		Vector<Transform> bind_matrices;

		if (d.has("inverseBindMatrices")) {
			bind_matrices = _decode_accessor_as_xform(state, d["inverseBindMatrices"], false);
			ERR_FAIL_COND_V(bind_matrices.size() != joints.size(), ERR_PARSE_ERROR);
		}

		for (int j = 0; j < joints.size(); j++) {
			int index = joints[j];
			ERR_FAIL_INDEX_V(index, state.nodes.size(), ERR_PARSE_ERROR);
			state.nodes[index]->joint_skin = state.skins.size();
			state.nodes[index]->joint_bone = j;
			GLTFSkin::Bone bone;
			bone.node = index;
			if (bind_matrices.size()) {
				bone.inverse_bind = bind_matrices[j];
			}

			skin.bones.push_back(bone);
		}

		print_line("skin has skeleton? " + itos(d.has("skeleton")));
		if (d.has("skeleton")) {
			int skeleton = d["skeleton"];
			ERR_FAIL_INDEX_V(skeleton, state.nodes.size(), ERR_PARSE_ERROR);
			state.nodes[skeleton]->skeleton_skin = state.skins.size();
			print_line("setting skeleton skin to" + itos(skeleton));
			skin.skeleton = skeleton;
		}

		if (d.has("name")) {
			skin.name = d["name"];
		}

		//locate the right place to put a Skeleton node

		if (state.skin_users.has(i)) {
			Vector<int> users = state.skin_users[i];
			int skin_node = -1;
			for (int j = 0; j < users.size(); j++) {
				int user = state.nodes[users[j]]->parent; //always go from parent
				if (j == 0) {
					skin_node = user;
				} else if (skin_node != -1) {
					bool found = false;
					while (skin_node >= 0) {

						int cuser = user;
						while (cuser != -1) {
							if (cuser == skin_node) {
								found = true;
								break;
							}
							cuser = state.nodes[skin_node]->parent;
						}
						if (found)
							break;
						skin_node = state.nodes[skin_node]->parent;
					}

					if (!found) {
						skin_node = -1; //just leave where it is
					}

					//find a common parent
				}
			}

			if (skin_node != -1) {
				for (int j = 0; j < users.size(); j++) {
					state.nodes[users[j]]->child_of_skeleton = i;
				}

				state.nodes[skin_node]->skeleton_children.push_back(i);
			}
		}
		state.skins.push_back(skin);
	}
	print_line("total skins: " + itos(state.skins.size()));

	//now

	return OK;
}

Error EditorSceneImporterGLTF::_parse_cameras(GLTFState &state) {

	if (!state.json.has("cameras"))
		return OK;

	Array cameras = state.json["cameras"];

	for (int i = 0; i < cameras.size(); i++) {

		Dictionary d = cameras[i];

		GLTFCamera camera;
		ERR_FAIL_COND_V(!d.has("type"), ERR_PARSE_ERROR);
		String type = d["type"];
		if (type == "orthographic") {

			camera.perspective = false;
			if (d.has("orthographic")) {
				Dictionary og = d["orthographic"];
				camera.fov_size = og["ymag"];
				camera.zfar = og["zfar"];
				camera.znear = og["znear"];
			} else {
				camera.fov_size = 10;
			}

		} else if (type == "perspective") {

			camera.perspective = true;
			if (d.has("perspective")) {
				Dictionary ppt = d["perspective"];
				// GLTF spec is in radians, Godot's camera is in degrees.
				camera.fov_size = (double)ppt["yfov"] * 180.0 / Math_PI;
				camera.zfar = ppt["zfar"];
				camera.znear = ppt["znear"];
			} else {
				camera.fov_size = 10;
			}
		} else {
			ERR_EXPLAIN("Camera should be in 'orthographic' or 'perspective'");
			ERR_FAIL_V(ERR_PARSE_ERROR);
		}

		state.cameras.push_back(camera);
	}

	print_line("total cameras: " + itos(state.cameras.size()));

	return OK;
}

Error EditorSceneImporterGLTF::_parse_animations(GLTFState &state) {

	if (!state.json.has("animations"))
		return OK;

	Array animations = state.json["animations"];

	for (int i = 0; i < animations.size(); i++) {

		Dictionary d = animations[i];

		GLTFAnimation animation;

		if (!d.has("channels") || !d.has("samplers"))
			continue;

		Array channels = d["channels"];
		Array samplers = d["samplers"];

		if (d.has("name")) {
			animation.name = d["name"];
		}

		for (int j = 0; j < channels.size(); j++) {

			Dictionary c = channels[j];
			if (!c.has("target"))
				continue;

			Dictionary t = c["target"];
			if (!t.has("node") || !t.has("path")) {
				continue;
			}

			ERR_FAIL_COND_V(!c.has("sampler"), ERR_PARSE_ERROR);
			int sampler = c["sampler"];
			ERR_FAIL_INDEX_V(sampler, samplers.size(), ERR_PARSE_ERROR);

			int node = t["node"];
			String path = t["path"];

			ERR_FAIL_INDEX_V(node, state.nodes.size(), ERR_PARSE_ERROR);

			GLTFAnimation::Track *track = NULL;

			if (!animation.tracks.has(node)) {
				animation.tracks[node] = GLTFAnimation::Track();
			}

			track = &animation.tracks[node];

			Dictionary s = samplers[sampler];

			ERR_FAIL_COND_V(!s.has("input"), ERR_PARSE_ERROR);
			ERR_FAIL_COND_V(!s.has("output"), ERR_PARSE_ERROR);

			int input = s["input"];
			int output = s["output"];

			GLTFAnimation::Interpolation interp = GLTFAnimation::INTERP_LINEAR;
			if (s.has("interpolation")) {
				String in = s["interpolation"];
				if (in == "STEP") {
					interp = GLTFAnimation::INTERP_STEP;
				} else if (in == "LINEAR") {
					interp = GLTFAnimation::INTERP_LINEAR;
				} else if (in == "CATMULLROMSPLINE") {
					interp = GLTFAnimation::INTERP_CATMULLROMSPLINE;
				} else if (in == "CUBICSPLINE") {
					interp = GLTFAnimation::INTERP_CUBIC_SPLINE;
				}
			}

			print_line("path: " + path);
			PoolVector<float> times = _decode_accessor_as_floats(state, input, false);
			if (path == "translation") {
				PoolVector<Vector3> translations = _decode_accessor_as_vec3(state, output, false);
				track->translation_track.interpolation = interp;
				track->translation_track.times = Variant(times); //convert via variant
				track->translation_track.values = Variant(translations); //convert via variant
			} else if (path == "rotation") {
				Vector<Quat> rotations = _decode_accessor_as_quat(state, output, false);
				track->rotation_track.interpolation = interp;
				track->rotation_track.times = Variant(times); //convert via variant
				track->rotation_track.values = rotations; //convert via variant
			} else if (path == "scale") {
				PoolVector<Vector3> scales = _decode_accessor_as_vec3(state, output, false);
				track->scale_track.interpolation = interp;
				track->scale_track.times = Variant(times); //convert via variant
				track->scale_track.values = Variant(scales); //convert via variant
			} else if (path == "weights") {
				PoolVector<float> weights = _decode_accessor_as_floats(state, output, false);

				ERR_FAIL_INDEX_V(state.nodes[node]->mesh, state.meshes.size(), ERR_PARSE_ERROR);
				GLTFMesh *mesh = &state.meshes[state.nodes[node]->mesh];
				ERR_FAIL_COND_V(mesh->blend_weights.size() == 0, ERR_PARSE_ERROR);
				int wc = mesh->blend_weights.size();

				track->weight_tracks.resize(wc);

				int wlen = weights.size() / wc;
				PoolVector<float>::Read r = weights.read();
				for (int k = 0; k < wc; k++) { //separate tracks, having them together is not such a good idea
					GLTFAnimation::Channel<float> cf;
					cf.interpolation = interp;
					cf.times = Variant(times);
					Vector<float> wdata;
					wdata.resize(wlen);
					for (int l = 0; l < wlen; l++) {
						wdata[l] = r[l * wc + k];
					}

					cf.values = wdata;
					track->weight_tracks[k] = cf;
				}
			} else {
				WARN_PRINTS("Invalid path: " + path);
			}
		}

		state.animations.push_back(animation);
	}

	print_line("total animations: " + itos(state.animations.size()));

	return OK;
}

void EditorSceneImporterGLTF::_assign_scene_names(GLTFState &state) {

	for (int i = 0; i < state.nodes.size(); i++) {
		GLTFNode *n = state.nodes[i];
		if (n->name == "") {
			if (n->mesh >= 0) {
				n->name = "Mesh";
			} else if (n->joint_skin >= 0) {
				n->name = "Bone";
			} else {
				n->name = "Node";
			}
		}

		n->name = _gen_unique_name(state, n->name);
	}
}

void EditorSceneImporterGLTF::_generate_node(GLTFState &state, int p_node, Node *p_parent, Node *p_owner, Vector<Skeleton *> &skeletons) {
	ERR_FAIL_INDEX(p_node, state.nodes.size());

	GLTFNode *n = state.nodes[p_node];
	Spatial *node;

	if (n->mesh >= 0) {
		ERR_FAIL_INDEX(n->mesh, state.meshes.size());
		MeshInstance *mi = memnew(MeshInstance);
		GLTFMesh &mesh = state.meshes[n->mesh];
		mi->set_mesh(mesh.mesh);
		if (mesh.mesh->get_name() == "") {
			mesh.mesh->set_name(n->name);
		}
		for (int i = 0; i < mesh.blend_weights.size(); i++) {
			mi->set("blend_shapes/" + mesh.mesh->get_blend_shape_name(i), mesh.blend_weights[i]);
		}

		node = mi;
	} else if (n->camera >= 0) {
		ERR_FAIL_INDEX(n->camera, state.cameras.size());
		Camera *camera = memnew(Camera);

		const GLTFCamera &c = state.cameras[n->camera];
		if (c.perspective) {
			camera->set_perspective(c.fov_size, c.znear, c.znear);
		} else {
			camera->set_orthogonal(c.fov_size, c.znear, c.znear);
		}

		node = camera;
	} else {
		node = memnew(Spatial);
	}

	node->set_name(n->name);

	if (n->child_of_skeleton >= 0) {
		//move skeleton around and place it on node, as the node _is_ a skeleton.
		Skeleton *s = skeletons[n->child_of_skeleton];
		p_parent = s;
	}

	p_parent->add_child(node);
	node->set_owner(p_owner);
	node->set_transform(n->xform);

	n->godot_node = node;

	for (int i = 0; i < n->skeleton_children.size(); i++) {

		Skeleton *s = skeletons[n->skeleton_children[i]];
		s->get_parent()->remove_child(s);
		node->add_child(s);
		s->set_owner(p_owner);
	}

	for (int i = 0; i < n->children.size(); i++) {
		if (state.nodes[n->children[i]]->joint_skin >= 0) {
			_generate_bone(state, n->children[i], skeletons, -1);
		} else {
			_generate_node(state, n->children[i], node, p_owner, skeletons);
		}
	}
}

void EditorSceneImporterGLTF::_generate_bone(GLTFState &state, int p_node, Vector<Skeleton *> &skeletons, int p_parent_bone) {
	ERR_FAIL_INDEX(p_node, state.nodes.size());

	GLTFNode *n = state.nodes[p_node];

	ERR_FAIL_COND(n->joint_skin < 0);

	int bone_index = skeletons[n->joint_skin]->get_bone_count();
	skeletons[n->joint_skin]->add_bone(n->name);
	if (p_parent_bone >= 0) {
		skeletons[n->joint_skin]->set_bone_parent(bone_index, p_parent_bone);
	}
	skeletons[n->joint_skin]->set_bone_rest(bone_index, state.skins[n->joint_skin].bones[n->joint_bone].inverse_bind.affine_inverse());

	n->godot_node = skeletons[n->joint_skin];
	n->godot_bone_index = bone_index;

	for (int i = 0; i < n->children.size(); i++) {
		ERR_CONTINUE(state.nodes[n->children[i]]->joint_skin < 0);
		_generate_bone(state, n->children[i], skeletons, bone_index);
	}
}

template <class T>
struct EditorSceneImporterGLTFInterpolate {

	T lerp(const T &a, const T &b, float c) const {

		return a + (b - a) * c;
	}

	T catmull_rom(const T &p0, const T &p1, const T &p2, const T &p3, float t) {

		float t2 = t * t;
		float t3 = t2 * t;

		return 0.5f * ((2.0f * p1) + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4 * p2 - p3) * t2 + (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
	}

	T bezier(T start, T control_1, T control_2, T end, float t) {
		/* Formula from Wikipedia article on Bezier curves. */
		real_t omt = (1.0 - t);
		real_t omt2 = omt * omt;
		real_t omt3 = omt2 * omt;
		real_t t2 = t * t;
		real_t t3 = t2 * t;

		return start * omt3 + control_1 * omt2 * t * 3.0 + control_2 * omt * t2 * 3.0 + end * t3;
	}
};

//thank you for existing, partial specialization
template <>
struct EditorSceneImporterGLTFInterpolate<Quat> {

	Quat lerp(const Quat &a, const Quat &b, float c) const {

		return a.slerp(b, c);
	}

	Quat catmull_rom(const Quat &p0, const Quat &p1, const Quat &p2, const Quat &p3, float c) {

		return p1.slerp(p2, c);
	}

	Quat bezier(Quat start, Quat control_1, Quat control_2, Quat end, float t) {
		return start.slerp(end, t);
	}
};

template <class T>
T EditorSceneImporterGLTF::_interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values, float p_time, GLTFAnimation::Interpolation p_interp) {

	//could use binary search, worth it?
	int idx = -1;
	for (int i = 0; i < p_times.size(); i++) {
		if (p_times[i] > p_time)
			break;
		idx++;
	}

	EditorSceneImporterGLTFInterpolate<T> interp;

	switch (p_interp) {
		case GLTFAnimation::INTERP_LINEAR: {

			if (idx == -1) {
				return p_values[0];
			} else if (idx >= p_times.size() - 1) {
				return p_values[p_times.size() - 1];
			}

			float c = (p_time - p_times[idx]) / (p_times[idx + 1] - p_times[idx]);

			return interp.lerp(p_values[idx], p_values[idx + 1], c);

		} break;
		case GLTFAnimation::INTERP_STEP: {

			if (idx == -1) {
				return p_values[0];
			} else if (idx >= p_times.size() - 1) {
				return p_values[p_times.size() - 1];
			}

			return p_values[idx];

		} break;
		case GLTFAnimation::INTERP_CATMULLROMSPLINE: {

			if (idx == -1) {
				return p_values[1];
			} else if (idx >= p_times.size() - 1) {
				return p_values[1 + p_times.size() - 1];
			}

			float c = (p_time - p_times[idx]) / (p_times[idx + 1] - p_times[idx]);

			return interp.catmull_rom(p_values[idx - 1], p_values[idx], p_values[idx + 1], p_values[idx + 3], c);

		} break;
		case GLTFAnimation::INTERP_CUBIC_SPLINE: {

			if (idx == -1) {
				return p_values[1];
			} else if (idx >= p_times.size() - 1) {
				return p_values[(p_times.size() - 1) * 3 + 1];
			}

			float c = (p_time - p_times[idx]) / (p_times[idx + 1] - p_times[idx]);

			T from = p_values[idx * 3 + 1];
			T c1 = from + p_values[idx * 3 + 0];
			T to = p_values[idx * 3 + 3];
			T c2 = to + p_values[idx * 3 + 2];

			return interp.bezier(from, c1, c2, to, c);

		} break;
	}

	ERR_FAIL_V(p_values[0]);
}

void EditorSceneImporterGLTF::_import_animation(GLTFState &state, AnimationPlayer *ap, int index, int bake_fps, Vector<Skeleton *> skeletons) {

	const GLTFAnimation &anim = state.animations[index];

	String name = anim.name;
	if (name == "") {
		name = _gen_unique_name(state, "Animation");
	}

	Ref<Animation> animation;
	animation.instance();
	animation->set_name(name);

	for (Map<int, GLTFAnimation::Track>::Element *E = anim.tracks.front(); E; E = E->next()) {

		const GLTFAnimation::Track &track = E->get();
		//need to find the path
		NodePath node_path;

		GLTFNode *node = state.nodes[E->key()];
		ERR_CONTINUE(!node->godot_node);

		if (node->godot_bone_index >= 0) {
			Skeleton *sk = (Skeleton *)node->godot_node;
			String path = ap->get_parent()->get_path_to(sk);
			String bone = sk->get_bone_name(node->godot_bone_index);
			node_path = path + ":" + bone;
		} else {
			node_path = ap->get_parent()->get_path_to(node->godot_node);
		}

		float length = 0;

		for (int i = 0; i < track.rotation_track.times.size(); i++) {
			length = MAX(length, track.rotation_track.times[i]);
		}
		for (int i = 0; i < track.translation_track.times.size(); i++) {
			length = MAX(length, track.translation_track.times[i]);
		}
		for (int i = 0; i < track.scale_track.times.size(); i++) {
			length = MAX(length, track.scale_track.times[i]);
		}

		for (int i = 0; i < track.weight_tracks.size(); i++) {
			for (int j = 0; j < track.weight_tracks[i].times.size(); j++) {
				length = MAX(length, track.weight_tracks[i].times[j]);
			}
		}

		animation->set_length(length);

		if (track.rotation_track.values.size() || track.translation_track.values.size() || track.scale_track.values.size()) {
			//make transform track
			int track_idx = animation->get_track_count();
			animation->add_track(Animation::TYPE_TRANSFORM);
			animation->track_set_path(track_idx, node_path);
			//first determine animation length

			float increment = 1.0 / float(bake_fps);
			float time = 0.0;

			Vector3 base_pos;
			Quat base_rot;
			Vector3 base_scale = Vector3(1, 1, 1);

			if (!track.rotation_track.values.size()) {
				base_rot = state.nodes[E->key()]->rotation;
			}

			if (!track.translation_track.values.size()) {
				base_pos = state.nodes[E->key()]->translation;
			}

			if (!track.scale_track.values.size()) {
				base_scale = state.nodes[E->key()]->scale;
			}

			bool last = false;
			while (true) {

				Vector3 pos = base_pos;
				Quat rot = base_rot;
				Vector3 scale = base_scale;

				if (track.translation_track.times.size()) {

					pos = _interpolate_track<Vector3>(track.translation_track.times, track.translation_track.values, time, track.translation_track.interpolation);
				}

				if (track.rotation_track.times.size()) {

					rot = _interpolate_track<Quat>(track.rotation_track.times, track.rotation_track.values, time, track.rotation_track.interpolation);
				}

				if (track.scale_track.times.size()) {

					scale = _interpolate_track<Vector3>(track.scale_track.times, track.scale_track.values, time, track.scale_track.interpolation);
				}

				if (node->godot_bone_index >= 0) {

					Transform xform;
					xform.basis = Basis(rot);
					xform.basis.scale(scale);
					xform.origin = pos;

					Skeleton *skeleton = skeletons[node->joint_skin];
					int bone = node->godot_bone_index;
					xform = skeleton->get_bone_rest(bone).affine_inverse() * xform;

					rot = xform.basis;
					rot.normalize();
					scale = xform.basis.get_scale();
					pos = xform.origin;
				}

				animation->transform_track_insert_key(track_idx, time, pos, rot, scale);

				if (last) {
					break;
				}
				time += increment;
				if (time >= length) {
					last = true;
					time = length;
				}
			}
		}

		for (int i = 0; i < track.weight_tracks.size(); i++) {
			ERR_CONTINUE(node->mesh < 0 || node->mesh >= state.meshes.size());
			const GLTFMesh &mesh = state.meshes[node->mesh];
			String prop = "blend_shapes/" + mesh.mesh->get_blend_shape_name(i);
			node_path = String(node_path) + ":" + prop;

			int track_idx = animation->get_track_count();
			animation->add_track(Animation::TYPE_VALUE);
			animation->track_set_path(track_idx, node_path);

			if (track.weight_tracks[i].interpolation <= GLTFAnimation::INTERP_STEP) {
				animation->track_set_interpolation_type(track_idx, track.weight_tracks[i].interpolation == GLTFAnimation::INTERP_STEP ? Animation::INTERPOLATION_NEAREST : Animation::INTERPOLATION_NEAREST);
				for (int j = 0; j < track.weight_tracks[i].times.size(); j++) {
					float t = track.weight_tracks[i].times[j];
					float w = track.weight_tracks[i].values[j];
					animation->track_insert_key(track_idx, t, w);
				}
			} else {
				//must bake, apologies.
				float increment = 1.0 / float(bake_fps);
				float time = 0.0;

				bool last = false;
				while (true) {

					_interpolate_track<float>(track.weight_tracks[i].times, track.weight_tracks[i].values, time, track.weight_tracks[i].interpolation);
					if (last) {
						break;
					}
					time += increment;
					if (time >= length) {
						last = true;
						time = length;
					}
				}
			}
		}
	}

	ap->add_animation(name, animation);
}

Spatial *EditorSceneImporterGLTF::_generate_scene(GLTFState &state, int p_bake_fps) {

	Spatial *root = memnew(Spatial);
	root->set_name(state.scene_name);
	//generate skeletons
	Vector<Skeleton *> skeletons;
	for (int i = 0; i < state.skins.size(); i++) {
		Skeleton *s = memnew(Skeleton);
		String name = state.skins[i].name;
		if (name == "") {
			name = _gen_unique_name(state, "Skeleton");
		}
		s->set_name(name);
		root->add_child(s);
		s->set_owner(root);
		skeletons.push_back(s);
	}
	for (int i = 0; i < state.root_nodes.size(); i++) {
		if (state.nodes[state.root_nodes[i]]->joint_skin >= 0) {
			_generate_bone(state, state.root_nodes[i], skeletons, -1);
		} else {
			_generate_node(state, state.root_nodes[i], root, root, skeletons);
		}
	}

	for (int i = 0; i < skeletons.size(); i++) {
		skeletons[i]->localize_rests();
	}

	if (state.animations.size()) {
		AnimationPlayer *ap = memnew(AnimationPlayer);
		ap->set_name("AnimationPlayer");
		root->add_child(ap);
		ap->set_owner(root);

		for (int i = 0; i < state.animations.size(); i++) {
			_import_animation(state, ap, i, p_bake_fps, skeletons);
		}
	}

	return root;
}

Node *EditorSceneImporterGLTF::import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err) {

	GLTFState state;

	if (p_path.to_lower().ends_with("glb")) {
		//binary file
		//text file
		Error err = _parse_glb(p_path, state);
		if (err)
			return NULL;
	} else {
		//text file
		Error err = _parse_json(p_path, state);
		if (err)
			return NULL;
	}

	ERR_FAIL_COND_V(!state.json.has("asset"), NULL);

	Dictionary asset = state.json["asset"];

	ERR_FAIL_COND_V(!asset.has("version"), NULL);

	String version = asset["version"];

	state.major_version = version.get_slice(".", 0).to_int();
	state.minor_version = version.get_slice(".", 1).to_int();

	/* STEP 0 PARSE SCENE */
	Error err = _parse_scenes(state);
	if (err != OK)
		return NULL;

	/* STEP 1 PARSE NODES */
	err = _parse_nodes(state);
	if (err != OK)
		return NULL;

	/* STEP 2 PARSE BUFFERS */
	err = _parse_buffers(state, p_path.get_base_dir());
	if (err != OK)
		return NULL;

	/* STEP 3 PARSE BUFFER VIEWS */
	err = _parse_buffer_views(state);
	if (err != OK)
		return NULL;

	/* STEP 4 PARSE ACCESSORS */
	err = _parse_accessors(state);
	if (err != OK)
		return NULL;

	/* STEP 5 PARSE IMAGES */
	err = _parse_images(state, p_path.get_base_dir());
	if (err != OK)
		return NULL;

	/* STEP 6 PARSE TEXTURES */
	err = _parse_textures(state);
	if (err != OK)
		return NULL;

	/* STEP 7 PARSE TEXTURES */
	err = _parse_materials(state);
	if (err != OK)
		return NULL;

	/* STEP 8 PARSE MESHES (we have enough info now) */
	err = _parse_meshes(state);
	if (err != OK)
		return NULL;

	/* STEP 9 PARSE SKINS */
	err = _parse_skins(state);
	if (err != OK)
		return NULL;

	/* STEP 10 PARSE CAMERAS */
	err = _parse_cameras(state);
	if (err != OK)
		return NULL;

	/* STEP 11 PARSE ANIMATIONS */
	err = _parse_animations(state);
	if (err != OK)
		return NULL;

	/* STEP 12 ASSIGN SCENE NAMES */
	_assign_scene_names(state);

	/* STEP 13 MAKE SCENE! */
	Spatial *scene = _generate_scene(state, p_bake_fps);

	return scene;
}

Ref<Animation> EditorSceneImporterGLTF::import_animation(const String &p_path, uint32_t p_flags, int p_bake_fps) {

	return Ref<Animation>();
}

EditorSceneImporterGLTF::EditorSceneImporterGLTF() {
}
