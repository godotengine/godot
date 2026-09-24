/**************************************************************************/
/*  slime_ai_code_reader.cpp                                              */
/**************************************************************************/

#include "slime_ai_code_reader.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "editor/script/script_editor_plugin.h"
#include "scene/gui/text_edit.h"

#include <filesystem>

namespace SlimeAI {

static constexpr uint64_t MAX_DOCUMENT_BYTES = 256 * 1024;
static constexpr int MAX_RETURNED_CHARS = 16 * 1024;

static Dictionary _error(const String &p_code) {
	Dictionary result;
	result["error"] = p_code;
	return result;
}

static bool _safe_segment(const String &p_segment) {
	const String lower = p_segment.to_lower();
	if (p_segment.is_empty() || p_segment == "." || p_segment == ".." || p_segment.begins_with(".")) {
		return false;
	}
	if (lower == "node_modules" || lower == "credentials" || lower == "secrets" || lower == "private" || lower == "keychain") {
		return false;
	}
	if (lower.contains("credential") || lower.contains("secret") || lower.contains("token") || lower.contains("password") || lower.contains("private_key") || lower.contains("id_rsa") || lower.contains("id_ed25519")) {
		return false;
	}
	return true;
}

static bool _contains_sensitive_material(const String &p_text) {
	const String lower = p_text.to_lower();
	return lower.contains("-----begin private key") || lower.contains("-----begin openssh private key") || lower.contains("api_key=") || lower.contains("api_key:") || lower.contains("authorization: bearer ") || lower.contains("password=") || lower.contains("client_secret=");
}

bool CodeReader::resolve_path(const String &p_project_root, const String &p_relative_path, String &r_absolute_path, String &r_error) {
	r_absolute_path.clear();
	r_error = "SCOPE_DENIED";
	String relative = p_relative_path;
	if (relative.begins_with("res://")) {
		relative = relative.substr(6);
	}
	if (relative.is_empty() || relative.is_absolute_path() || relative.contains("\\") || relative.contains(":") || relative.contains("//")) {
		return false;
	}
	const PackedStringArray segments = relative.split("/", true);
	for (const String &segment : segments) {
		if (!_safe_segment(segment)) {
			return false;
		}
	}
	const String extension = relative.get_extension().to_lower();
	if (extension != "gd" && extension != "cs" && extension != "txt" && extension != "md" && extension != "json" && extension != "tscn" && extension != "tres") {
		return false;
	}
	if (relative.get_file().to_lower() == "project.godot" || relative.get_file().to_lower() == "package-lock.json") {
		return false;
	}
	std::error_code error;
	const std::filesystem::path root = std::filesystem::canonical(std::filesystem::u8path(p_project_root.utf8().get_data()), error);
	if (error || !std::filesystem::is_directory(root)) {
		r_error = "CAPABILITY_UNAVAILABLE";
		return false;
	}
	const std::filesystem::path candidate = std::filesystem::canonical(root / std::filesystem::u8path(relative.utf8().get_data()), error);
	if (error || !std::filesystem::is_regular_file(candidate)) {
		r_error = "FILE_UNAVAILABLE";
		return false;
	}
	const std::filesystem::path inside = candidate.lexically_relative(root);
	if (inside.empty() || inside.is_absolute() || *inside.begin() == "..") {
		return false;
	}
	const String resolved_relative = String::utf8(inside.generic_u8string().c_str());
	for (const String &segment : resolved_relative.split("/", true)) {
		if (!_safe_segment(segment)) {
			return false;
		}
	}
	// Open the resolved target, avoiding a second traversal through a mutable link.
	r_absolute_path = String::utf8(candidate.generic_u8string().c_str());
	r_error.clear();
	return true;
}

Dictionary CodeReader::load_document(const String &p_project_root, const String &p_relative_path) {
	String path;
	String error;
	if (!resolve_path(p_project_root, p_relative_path, path, error)) {
		return _error(error);
	}
	Dictionary result;
	String relative = p_relative_path.begins_with("res://") ? p_relative_path.substr(6) : p_relative_path;
	result["path"] = "res://" + relative;
	result["disk_revision"] = FileAccess::get_sha256(path);
	if (ScriptEditor *script_editor = ScriptEditor::get_singleton()) {
		const Vector<Ref<Script>> scripts = script_editor->get_open_scripts();
		for (const Ref<Script> &script : scripts) {
			if (script.is_null() || script->get_path() != "res://" + relative) {
				continue;
			}
			ScriptEditorBase *editor = script_editor->get_resource_editor(script);
			if (editor && editor->is_unsaved()) {
				TextEdit *text_editor = Object::cast_to<TextEdit>(editor->get_base_editor());
				if (!text_editor) {
					return _error("UNSAVED_BUFFER_UNAVAILABLE");
				}
				const String contents = text_editor->get_text();
				if (contents.utf8().length() > MAX_DOCUMENT_BYTES) {
					return _error("FILE_TOO_LARGE");
				}
				if (_contains_sensitive_material(contents)) {
					return _error("SENSITIVE_CONTENT");
				}
				result["text"] = contents;
				result["source"] = "unsaved_editor_buffer";
				result["revision"] = contents.sha256_text();
				return result;
			}
		}
		const PackedStringArray unsaved_paths = script_editor->get_unsaved_files();
		for (const String &unsaved_path : unsaved_paths) {
			if (unsaved_path == "res://" + relative) {
				return _error("UNSAVED_BUFFER_UNAVAILABLE");
			}
		}
	}
	Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
	if (file.is_null()) {
		return _error("FILE_UNAVAILABLE");
	}
	if (file->get_length() > MAX_DOCUMENT_BYTES) {
		return _error("FILE_TOO_LARGE");
	}
	const PackedByteArray bytes = file->get_buffer(file->get_length());
	if (bytes.size() != file->get_length()) {
		return _error("FILE_UNAVAILABLE");
	}
	String contents;
	if (contents.append_utf8((const char *)bytes.ptr(), bytes.size()) != OK) {
		return _error("INVALID_UTF8");
	}
	if (_contains_sensitive_material(contents)) {
		return _error("SENSITIVE_CONTENT");
	}
	result["text"] = contents;
	result["source"] = "saved_file";
	result["revision"] = result["disk_revision"];
	return result;
}

Dictionary CodeReader::read(const String &p_project_root, const String &p_relative_path, int p_start_line, int p_line_limit) {
	Dictionary result = load_document(p_project_root, p_relative_path);
	if (result.has("error")) {
		return result;
	}
	const String text = result["text"];
	result.erase("text");
	const PackedStringArray lines = text.split("\n", true);
	const int start = CLAMP(p_start_line, 1, MAX(1, lines.size()));
	const int limit = CLAMP(p_line_limit, 1, 200);
	String excerpt;
	int end = start - 1;
	for (int i = start - 1; i < MIN(lines.size(), start - 1 + limit); i++) {
		const String line = lines[i];
		if (excerpt.length() + line.length() + 1 > MAX_RETURNED_CHARS) {
			break;
		}
		excerpt += line;
		if (i + 1 < lines.size()) {
			excerpt += "\n";
		}
		end = i + 1;
	}
	result["start_line"] = start;
	result["end_line"] = end;
	result["line_count"] = lines.size();
	result["text"] = excerpt;
	result["next_line"] = end < lines.size() ? end + 1 : -1;
	result["truncated"] = end < lines.size();
	return result;
}

Dictionary CodeReader::read(const String &p_relative_path, int p_start_line, int p_line_limit) {
	ProjectSettings *settings = ProjectSettings::get_singleton();
	if (!settings) {
		return _error("CAPABILITY_UNAVAILABLE");
	}
	return read(settings->get_resource_path(), p_relative_path, p_start_line, p_line_limit);
}

} // namespace SlimeAI
