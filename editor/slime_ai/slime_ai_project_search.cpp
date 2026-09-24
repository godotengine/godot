/**************************************************************************/
/*  slime_ai_project_search.cpp                                           */
/**************************************************************************/

#include "slime_ai_project_search.h"

#include "core/config/project_settings.h"
#include "editor/slime_ai/slime_ai_code_reader.h"

#include <algorithm>
#include <filesystem>
#include <vector>

namespace SlimeAI {

static constexpr int MAX_SCAN_FILES = 512;
static constexpr uint64_t MAX_SCAN_BYTES = 4 * 1024 * 1024;
static constexpr int MAX_MATCHES = 1024;

static bool _skip_segment(const String &p_segment) {
	const String lower = p_segment.to_lower();
	return p_segment.begins_with(".") || lower == "node_modules" || lower == "private" || lower == "secrets" || lower == "credentials" || lower.contains("secret") || lower.contains("token") || lower.contains("password") || lower.contains("credential");
}

Dictionary ProjectSearch::search(const String &p_project_root, const String &p_query, int p_offset, int p_limit) {
	Dictionary result;
	if (p_query.is_empty() || p_query.length() > 128) {
		result["error"] = "INVALID_QUERY";
		return result;
	}
	std::error_code error;
	const std::filesystem::path root = std::filesystem::canonical(std::filesystem::u8path(p_project_root.utf8().get_data()), error);
	if (error || !std::filesystem::is_directory(root)) {
		result["error"] = "CAPABILITY_UNAVAILABLE";
		return result;
	}
	std::vector<String> candidates;
	std::filesystem::recursive_directory_iterator it(root, std::filesystem::directory_options::skip_permission_denied, error);
	const std::filesystem::recursive_directory_iterator end;
	int visited = 0;
	bool scan_truncated = false;
	for (; !error && it != end; it.increment(error)) {
		if (++visited > MAX_SCAN_FILES * 4) {
			scan_truncated = true;
			break;
		}
		const std::filesystem::path relative_fs = it->path().lexically_relative(root);
		const String relative = String::utf8(relative_fs.generic_u8string().c_str());
		if (_skip_segment(String::utf8(it->path().filename().u8string().c_str()))) {
			it.disable_recursion_pending();
			continue;
		}
		const std::filesystem::path canonical = std::filesystem::canonical(it->path(), error);
		if (error) {
			error.clear();
			it.disable_recursion_pending();
			continue;
		}
		const std::filesystem::path inside = canonical.lexically_relative(root);
		if (inside.empty() || inside.is_absolute() || *inside.begin() == "..") {
			it.disable_recursion_pending();
			continue;
		}
		if (it->is_directory(error)) {
			if (relative.count("/") >= 8) {
				it.disable_recursion_pending();
				scan_truncated = true;
			}
			continue;
		}
		if (!error && it->is_regular_file(error)) {
			String absolute;
			String scope_error;
			if (CodeReader::resolve_path(p_project_root, relative, absolute, scope_error)) {
				candidates.push_back(relative);
				if (candidates.size() >= MAX_SCAN_FILES) {
					scan_truncated = true;
					break;
				}
			}
		}
		error.clear();
	}
	if (error) {
		scan_truncated = true;
	}
	std::sort(candidates.begin(), candidates.end(), [](const String &a, const String &b) { return a < b; });
	const int offset = CLAMP(p_offset, 0, 1000);
	const int limit = CLAMP(p_limit, 1, 50);
	Array matches;
	int match_count = 0;
	uint64_t scanned_bytes = 0;
	for (const String &relative : candidates) {
		if (match_count >= MAX_MATCHES) {
			scan_truncated = true;
			break;
		}
		const Dictionary doc = CodeReader::load_document(p_project_root, relative);
		if (doc.has("error")) {
			continue;
		}
		const String text = doc["text"];
		scanned_bytes += text.utf8().length();
		if (scanned_bytes > MAX_SCAN_BYTES) {
			scan_truncated = true;
			break;
		}
		const bool filename_hit = relative.get_file().find(p_query) >= 0;
		const PackedStringArray lines = text.split("\n", true);
		for (int i = filename_hit ? -1 : 0; i < lines.size(); i++) {
			if (i >= 0 && lines[i].find(p_query) < 0) {
				continue;
			}
			if (match_count >= offset && matches.size() < limit) {
				Dictionary hit;
				hit["path"] = "res://" + relative;
				hit["source"] = doc["source"];
				hit["revision"] = doc["revision"];
				hit["disk_revision"] = doc["disk_revision"];
				hit["match_kind"] = i < 0 ? "filename" : "exact_text";
				hit["line"] = i < 0 ? 0 : i + 1;
				hit["excerpt"] = i < 0 ? String() : lines[i].substr(0, 240);
				matches.push_back(hit);
			}
			match_count++;
			if (match_count >= MAX_MATCHES) {
				scan_truncated = true;
				break;
			}
		}
	}
	result["query"] = p_query;
	result["matches"] = matches;
	result["offset"] = offset;
	result["limit"] = limit;
	result["next_offset"] = match_count > offset + matches.size() ? offset + matches.size() : -1;
	result["truncated"] = scan_truncated || match_count > offset + matches.size();
	result["scan_truncated"] = scan_truncated;
	result["scanned_files"] = candidates.size();
	result["search_kind"] = "filename_and_exact_text";
	return result;
}

Dictionary ProjectSearch::search(const String &p_query, int p_offset, int p_limit) {
	ProjectSettings *settings = ProjectSettings::get_singleton();
	if (!settings) {
		Dictionary result;
		result["error"] = "CAPABILITY_UNAVAILABLE";
		return result;
	}
	return search(settings->get_resource_path(), p_query, p_offset, p_limit);
}

} // namespace SlimeAI
