/**************************************************************************/
/*  gltf_document.compat.inc                                              */
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

#ifndef DISABLE_DEPRECATED

Error GLTFDocument::append_from_file_81746(String p_path, Ref<GLTFState> p_state, uint32_t p_flags, String p_base_path) {
	return append_data_from_file(p_path, p_state, p_flags, p_base_path);
}

Error GLTFDocument::append_from_buffer_81746(PackedByteArray p_bytes, String p_base_path, Ref<GLTFState> p_state, uint32_t p_flags) {
	return append_data_from_buffer(p_bytes, p_base_path, p_state, p_flags);
}

Error GLTFDocument::append_from_scene_81746(Node *p_node, Ref<GLTFState> p_state, uint32_t p_flags) {
	return append_data_from_scene(p_node, p_state, p_flags);
}

Node *GLTFDocument::generate_scene_81746(Ref<GLTFState> p_state, float p_bake_fps, bool p_trimming, bool p_remove_immutable_tracks) {
	return generate_scene_from_data(p_state, p_bake_fps, p_trimming, p_remove_immutable_tracks);
}

PackedByteArray GLTFDocument::generate_buffer_81746(Ref<GLTFState> p_state) {
	return generate_buffer_from_data(p_state);
}

Error GLTFDocument::write_to_filesystem_81746(Ref<GLTFState> p_state, const String &p_path) {
	return write_to_filesystem_from_data(p_state, p_path);
}

void GLTFDocument::_bind_compatibility_methods() {
	ClassDB::bind_method(D_METHOD("append_from_file", "path", "state", "flags", "base_path"),
			&GLTFDocument::append_from_file_81746, DEFVAL(0), DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("append_from_buffer", "bytes", "base_path", "state", "flags"),
			&GLTFDocument::append_from_buffer_81746, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("append_from_scene", "node", "state", "flags"),
			&GLTFDocument::append_from_scene_81746, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("generate_scene", "state", "bake_fps", "trimming", "remove_immutable_tracks"),
			&GLTFDocument::generate_scene_81746, DEFVAL(30), DEFVAL(false), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("generate_buffer", "state"),
			&GLTFDocument::generate_buffer_81746);
	ClassDB::bind_method(D_METHOD("write_to_filesystem", "state", "path"),
			&GLTFDocument::write_to_filesystem_81746);
}

#endif // DISABLE_DEPRECATED
