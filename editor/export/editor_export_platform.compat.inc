/**************************************************************************/
/*  editor_export_platform.compat.inc                                     */
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

Vector<String> EditorExportPlatform::_get_forced_export_files_bind_compat_71542() {
	return get_forced_export_files(Ref<EditorExportPreset>());
}

Error EditorExportPlatform::export_pack_bind_compat_XXXXX(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	return export_pack(p_preset, p_debug, p_path, p_flags, false);
}

Error EditorExportPlatform::_export_project_files_bind_compat_XXXXX(const Ref<EditorExportPreset> &p_preset, bool p_debug, const Callable &p_save_func, const Callable &p_so_func) {
	return _export_project_files(p_preset, p_debug, p_save_func, p_so_func, false);
}

Dictionary EditorExportPlatform::_save_pack_bind_compat_XXXXX(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, bool p_embed) {
	return _save_pack(p_preset, p_debug, p_path, p_embed, false);
}

Error EditorExportPlatform::export_pack_patch_bind_compat_XXXXX(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, const Vector<String> &p_patches, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	return export_pack_patch(p_preset, p_debug, p_path, p_patches, p_flags, false);
}

Dictionary EditorExportPlatform::_save_pack_patch_bind_compat_XXXXX(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path) {
	return _save_pack_patch(p_preset, p_debug, p_path, false);
}

void EditorExportPlatform::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_static_method("EditorExportPlatform", D_METHOD("get_forced_export_files"), &EditorExportPlatform::_get_forced_export_files_bind_compat_71542);
	ClassDB::bind_compatibility_method(D_METHOD("export_pack", "preset", "debug", "path", "flags"), &EditorExportPlatform::export_pack_bind_compat_XXXXX, DEFVAL(0));
	ClassDB::bind_compatibility_method(D_METHOD("export_project_files", "preset", "debug", "save_cb", "shared_cb"), &EditorExportPlatform::_export_project_files_bind_compat_XXXXX, DEFVAL(Callable()));
	ClassDB::bind_compatibility_method(D_METHOD("save_pack", "preset", "debug", "path", "embed"), &EditorExportPlatform::_save_pack_bind_compat_XXXXX, DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("export_pack_patch", "preset", "debug", "path", "patches", "flags"), &EditorExportPlatform::export_pack_patch_bind_compat_XXXXX, DEFVAL(PackedStringArray()), DEFVAL(0));
	ClassDB::bind_compatibility_method(D_METHOD("save_pack_patch", "preset", "debug", "path"), &EditorExportPlatform::_save_pack_patch_bind_compat_XXXXX);
}

#endif
