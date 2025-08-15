/**************************************************************************/
/*  os_openharmony.cpp                                                    */
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

#include "os_openharmony.h"

#include "dir_access_openharmony.h"
#include "display_server_openharmony.h"
#include "file_access_openharmony.h"

#include "main/main.h"
#include "scene/main/scene_tree.h"

#include <hilog/log.h>
#include <native_drawing/drawing_text_font_descriptor.h>
#include <native_drawing/drawing_text_typography.h>

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x3200
#define LOG_TAG "LIB_GODOT"

const char *OS_OpenHarmony::EXEC_PATH = "template";
const char *OS_OpenHarmony::BUNDLE_RESOURCE_DIR = "/data/storage/el1/bundle/resources/rawfile/";
const char *OS_OpenHarmony::USER_DATA_DIR = "/data/storage/el2/base/files/";

OS_OpenHarmony *OS_OpenHarmony::get_singleton() {
	return static_cast<OS_OpenHarmony *>(OS::get_singleton());
}

OS_OpenHarmony::OS_OpenHarmony() {
	Vector<Logger *> loggers;
	Logger_OpenHarmony *logger = memnew(Logger_OpenHarmony);
	loggers.push_back(logger);
	_set_logger(memnew(CompositeLogger(loggers)));

	AudioDriverManager::add_driver(&audio_driver);
	DisplayServerOpenHarmony::register_openharmony_driver();
}

void OS_OpenHarmony::set_native_window(OHNativeWindow *p_native_window) {
	native_window = p_native_window;
}

OHNativeWindow *OS_OpenHarmony::get_native_window() const {
	return native_window;
}

void OS_OpenHarmony::set_window_id(int32_t p_window_id) {
	window_id = p_window_id;
}

int32_t OS_OpenHarmony::get_window_id() const {
	return window_id;
}

void OS_OpenHarmony::set_display_size(const Size2i &p_size) {
	display_size = p_size;
}

Size2i OS_OpenHarmony::get_display_size() const {
	return display_size;
}

void OS_OpenHarmony::set_allowed_permissions(const char *p_allowed_permissions) {
	String permissions = p_allowed_permissions;
	for (const String &permission : permissions.split(",")) {
		allowed_permissions.insert(permission);
	}
}

bool OS_OpenHarmony::request_permission(const String &p_name) {
	return allowed_permissions.has(p_name);
}

bool OS_OpenHarmony::request_permissions() {
	return false;
}

void OS_OpenHarmony::initialize() {
	OS_Unix::initialize_core();

	FileAccess::make_default<FileAccessOpenHarmony>(FileAccess::ACCESS_FILESYSTEM);
	DirAccess::make_default<DirAccessOpenHarmony>(DirAccess::ACCESS_FILESYSTEM);
}

void OS_OpenHarmony::initialize_joypads() {
}

void OS_OpenHarmony::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
}

MainLoop *OS_OpenHarmony::get_main_loop() const {
	return main_loop;
}

void OS_OpenHarmony::delete_main_loop() {
}

void OS_OpenHarmony::finalize() {
}

bool OS_OpenHarmony::_check_internal_feature_support(const String &p_feature) {
	if (p_feature == "system_fonts") {
		return true;
	}
	if (p_feature == "mobile") {
		return true;
	}
	return false;
}

String OS_OpenHarmony::get_user_data_dir(const String &p_user_dir) const {
	return OS_OpenHarmony::USER_DATA_DIR;
}

String OS_OpenHarmony::get_bundle_resource_dir() const {
	return OS_OpenHarmony::BUNDLE_RESOURCE_DIR;
}

String OS_OpenHarmony::get_executable_path() const {
	return OS_OpenHarmony::EXEC_PATH;
}

void OS_OpenHarmony::_load_system_font_config() const {
	font_config_loaded = false;
	font_aliases.clear();
	fonts.clear();
	font_names.clear();

	OH_Drawing_FontConfigInfoErrorCode error_code;
	OH_Drawing_FontConfigInfo *font_config_info = OH_Drawing_GetSystemFontConfigInfo(&error_code);
	ERR_FAIL_COND_MSG(error_code != SUCCESS_FONT_CONFIG_INFO, vformat("Failed to load system font config: %d.", error_code));

	HashSet<String> generic_font_names;
	for (int i = 0; i < font_config_info->fontGenericInfoSize; i++) {
		OH_Drawing_FontGenericInfo &info = font_config_info->fontGenericInfoSet[i];
		String font_name = String(info.familyName).to_lower();
		for (size_t j = 0; j < info.aliasInfoSize; j++) {
			String alias_name = String(info.aliasInfoSet[j].familyName).to_lower();
			font_aliases[alias_name] = font_name;
			generic_font_names.insert(font_name);
			generic_font_names.insert(alias_name);
		}
	}

	HashMap<String, Vector<String>> font_languages;
	for (int i = 0; i < font_config_info->fallbackGroupSize; i++) {
		OH_Drawing_FontFallbackGroup &group = font_config_info->fallbackGroupSet[i];
		for (size_t j = 0; j < group.fallbackInfoSize; j++) {
			OH_Drawing_FontFallbackInfo &info = group.fallbackInfoSet[j];
			font_languages[String(info.familyName).to_lower()].push_back(info.language);
		}
	}
	OH_Drawing_DestroySystemFontConfigInfo(font_config_info);

	OH_Drawing_Array *names = OH_Drawing_GetSystemFontFullNamesByType(OH_Drawing_SystemFontType::GENERIC);
	for (size_t i = 0;; i++) {
		const OH_Drawing_String *name = OH_Drawing_GetSystemFontFullNameByIndex(names, i);
		if (!name) {
			break;
		}
		OH_Drawing_FontDescriptor *descriptor = OH_Drawing_GetFontDescriptorByFullName(name, OH_Drawing_SystemFontType::GENERIC);
		String font_name = String(descriptor->fontFamily).to_lower();
		FontInfo fi;
		if (!generic_font_names.has(font_name)) {
			fi.priority = 2;
		}
		if (font_name.ends_with("-condensed")) {
			font_name = font_name.trim_suffix("-condensed");
			fi.stretch = 75;
			fi.font_name = font_name;
		}
		fi.font_name = font_name;
		fi.weight = descriptor->weight;
		fi.italic = descriptor->italic;
		fi.path = String(descriptor->path);
		fi.descriptor = descriptor;
		Vector<String> lang_codes = font_languages[font_name];
		if (lang_codes.is_empty()) {
			lang_codes.push_back("en");
		}
		for (int i = 0; i < lang_codes.size(); i++) {
			Vector<String> lang_code_elements = lang_codes[i].split("-");
			if (lang_code_elements.size() >= 1 && lang_code_elements[0] != "und") {
				// Add missing script codes.
				if (lang_code_elements[0] == "ko") {
					fi.script.insert("Hani");
					fi.script.insert("Hang");
				}
				if (lang_code_elements[0] == "ja") {
					fi.script.insert("Hani");
					fi.script.insert("Kana");
					fi.script.insert("Hira");
				}
				if (!lang_code_elements[0].is_empty()) {
					fi.lang.insert(lang_code_elements[0]);
				}
			}
			if (lang_code_elements.size() >= 2) {
				// Add common codes for variants and remove variants not supported by HarfBuzz/ICU.
				if (lang_code_elements[1] == "Aran") {
					fi.script.insert("Arab");
				}
				if (lang_code_elements[1] == "Cyrs") {
					fi.script.insert("Cyrl");
				}
				if (lang_code_elements[1] == "Hanb") {
					fi.script.insert("Hani");
					fi.script.insert("Bopo");
				}
				if (lang_code_elements[1] == "Hans" || lang_code_elements[1] == "Hant") {
					fi.script.insert("Hani");
				}
				if (lang_code_elements[1] == "Syrj" || lang_code_elements[1] == "Syre" || lang_code_elements[1] == "Syrn") {
					fi.script.insert("Syrc");
				}
				if (!lang_code_elements[1].is_empty() && lang_code_elements[1] != "Zsym" && lang_code_elements[1] != "Zsye" && lang_code_elements[1] != "Zmth") {
					fi.script.insert(lang_code_elements[1]);
				}
			}
		}
		fonts.push_back(fi);
		font_names.insert(font_name);
	}
	OH_Drawing_DestroySystemFontFullNames(names);
	font_config_loaded = true;
}

Vector<String> OS_OpenHarmony::get_system_fonts() const {
	if (!font_config_loaded) {
		_load_system_font_config();
	}
	Vector<String> ret;
	for (const String &E : font_names) {
		ret.push_back(E);
	}
	return ret;
}

String OS_OpenHarmony::get_system_font_path(const String &p_font_name, int p_weight, int p_stretch, bool p_italic) const {
	if (!font_config_loaded) {
		_load_system_font_config();
	}
	String font_name = p_font_name.to_lower();
	if (font_aliases.has(font_name)) {
		font_name = font_aliases[font_name];
	}

	int best_score = 0;
	const List<FontInfo>::Element *best_match = nullptr;

	for (const List<FontInfo>::Element *E = fonts.front(); E; E = E->next()) {
		int score = 0;
		if (E->get().font_name == font_name) {
			score += (65 - E->get().priority);
		}
		score += (20 - Math::abs(E->get().weight - p_weight) / 50);
		score += (20 - Math::abs(E->get().stretch - p_stretch) / 10);
		if (E->get().italic == p_italic) {
			score += 30;
		}
		if (score >= 60 && score > best_score) {
			best_score = score;
			best_match = E;
		}
		if (score >= 140) {
			break; // Perfect match.
		}
	}
	if (best_match) {
		return best_match->get().path;
	}
	return String();
}

Vector<String> OS_OpenHarmony::get_system_font_path_for_text(const String &p_font_name, const String &p_text, const String &p_locale, const String &p_script, int p_weight, int p_stretch, bool p_italic) const {
	if (!font_config_loaded) {
		_load_system_font_config();
	}
	String font_name = p_font_name.to_lower();
	if (font_aliases.has(font_name)) {
		font_name = font_aliases[font_name];
	}
	String lang_prefix = p_locale.get_slicec('_', 0);
	Vector<String> ret;
	int best_score = 0;
	for (const List<FontInfo>::Element *E = fonts.front(); E; E = E->next()) {
		int score = 0;
		if (!E->get().script.is_empty() && !p_script.is_empty() && !E->get().script.has(p_script)) {
			continue;
		}
		float sim = E->get().font_name.similarity(font_name);
		if (sim > 0.0) {
			score += (60 * sim + 5 - E->get().priority);
		}
		if (E->get().lang.has(p_locale)) {
			score += 120;
		} else if (E->get().lang.has(lang_prefix)) {
			score += 115;
		}
		if (E->get().script.has(p_script)) {
			score += 240;
		}
		score += (20 - Math::abs(E->get().weight - p_weight) / 50);
		score += (20 - Math::abs(E->get().stretch - p_stretch) / 10);
		if (E->get().italic == p_italic) {
			score += 30;
		}
		if (score > best_score) {
			best_score = score;
			if (!ret.has(E->get().path)) {
				ret.insert(0, E->get().path);
			}
		} else if (score == best_score || E->get().script.is_empty()) {
			if (!ret.has(E->get().path)) {
				ret.push_back(E->get().path);
			}
		}
		if (score >= 490) {
			break; // Perfect match.
		}
	}

	return ret;
}

String OS_OpenHarmony::get_system_ca_certificates() {
	String certfile;
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	if (da->file_exists("/etc/ssl/certs/cacert.pem")) {
		certfile = "/etc/ssl/certs/cacert.pem";
	}

	if (certfile.is_empty()) {
		return "";
	}

	Ref<FileAccess> f = FileAccess::open(certfile, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(f.is_null(), String(), vformat(R"(Failed to open system CA certificates file: "%s".)", certfile));

	String data = f->get_as_text();

	return data;
}

void OS_OpenHarmony::main_loop_begin() {
	if (main_loop) {
		main_loop->initialize();
	}
}

bool OS_OpenHarmony::main_loop_iterate() {
	if (!main_loop) {
		return false;
	}
	DisplayServerOpenHarmony::get_singleton()->process_events();
	return Main::iteration();
}

void OS_OpenHarmony::main_loop_end() {
	if (main_loop) {
		SceneTree *scene_tree = Object::cast_to<SceneTree>(main_loop);
		if (scene_tree) {
			scene_tree->quit();
		}
		main_loop->finalize();
	}
}

void OS_OpenHarmony::on_focus_out() {
	if (is_focused) {
		is_focused = false;

		if (DisplayServerOpenHarmony::get_singleton()) {
			DisplayServerOpenHarmony::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_FOCUS_OUT);
		}

		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_OUT);
		}

		audio_driver.set_pause(true);
	}
}

void OS_OpenHarmony::on_focus_in() {
	if (!is_focused) {
		is_focused = true;

		if (DisplayServerOpenHarmony::get_singleton()) {
			DisplayServerOpenHarmony::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_FOCUS_IN);
		}

		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_IN);
		}

		audio_driver.set_pause(false);
	}
}

void OS_OpenHarmony::on_enter_background() {
	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_PAUSED);
	}

	on_focus_out();
}

void OS_OpenHarmony::on_exit_background() {
	if (!is_focused) {
		on_focus_in();

		if (OS::get_singleton()->get_main_loop()) {
			OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_RESUMED);
		}
	}
}

void Logger_OpenHarmony::logv(const char *p_format, va_list p_list, bool p_err) {
	if (!should_log(p_err)) {
		return;
	}

	char buffer[4096];
	vsnprintf(&buffer[0], sizeof(buffer) - 1, p_format, p_list);

	if (p_err) {
		OH_LOG_ERROR(LOG_APP, "%{public}s", &buffer[0]);
	} else {
		OH_LOG_INFO(LOG_APP, "%{public}s", &buffer[0]);
	}
}
