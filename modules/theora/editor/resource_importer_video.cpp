/**************************************************************************/
/*  resource_importer_video.cpp                                           */
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

#include "resource_importer_video.h"

#include "core/config/project_settings.h"
#include "core/io/resource_saver.h"
#include "editor/settings/editor_settings.h"
#include "modules/theora/video_stream_theora.h"

String ResourceImporterVideo::get_importer_name() const {
	return "video";
}

String ResourceImporterVideo::get_visible_name() const {
	return "Video";
}

void ResourceImporterVideo::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("mp4");
	p_extensions->push_back("webm");
	p_extensions->push_back("mpg");
	p_extensions->push_back("mpeg");
	p_extensions->push_back("mkv");
	p_extensions->push_back("avi");
	p_extensions->push_back("mov");
	p_extensions->push_back("wmv");
}

String ResourceImporterVideo::get_save_extension() const {
	return "ogv";
}

String ResourceImporterVideo::get_resource_type() const {
	return "VideoStreamTheora";
}

bool ResourceImporterVideo::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return true;
}

int ResourceImporterVideo::get_preset_count() const {
	return 0;
}

String ResourceImporterVideo::get_preset_name(int p_idx) const {
	return String();
}

void ResourceImporterVideo::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::COLOR, "alpha_to_color", PROPERTY_HINT_COLOR_NO_ALPHA, ""), Color()));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "scale_width", PROPERTY_HINT_NONE, ""), -1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "scale_height", PROPERTY_HINT_NONE, ""), -1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "video_quality", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), 0.75));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "audio_quality", PROPERTY_HINT_RANGE, "-0.1,1.0,0.01"), 0.5));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "keyframe_interval", PROPERTY_HINT_RANGE, "1,1024,1"), 64));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "encoding_speed", PROPERTY_HINT_ENUM, "Fastest (Lowest Efficiency):4,Fast (Low Efficiency):3,Slow (High Efficiency):2,Slowest (Highest Efficiency):1,"), 4));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "additional_arguments", PROPERTY_HINT_NONE, ""), ""));
}

bool ResourceImporterVideo::check_ffmpeg_version() {
	List<String> args;

	args.push_back("-loglevel");
	args.push_back("quiet");
	args.push_back("-h");
	args.push_back("encoder=libtheora");

	String str;
	int exitcode = 0;

	String ffmpeg_path = EDITOR_GET("filesystem/import/video/ffmpeg_path");
	Error err = OS::get_singleton()->execute(ffmpeg_path, args, &str, &exitcode, true);

	if (err != OK || exitcode != 0) {
		print_verbose(str);
		return false;
	}

	return str.contains("-speed_level");
}

Error ResourceImporterVideo::import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	List<String> args;
	String ffmpeg_path = EDITOR_GET("filesystem/import/video/ffmpeg_path");

	if (ffmpeg_path.is_empty() || !FileAccess::exists(ffmpeg_path)) {
		ERR_FAIL_V_MSG(ERR_UNCONFIGURED, vformat("Failed to import %s, set the path to the FFmpeg binary in the editor setting `filesystem/import/video/ffmpeg_path`.", p_source_file));
	}

	if (int(p_options["encoding_speed"]) != 1) {
		if (!check_ffmpeg_version()) {
			WARN_PRINT(vformat("The configured FFmpeg version doesn't support changing the encoding speed. Update FFmpeg or use Slowest to disable this warning. Resource: %s", p_source_file));
		}
	}

	args.push_back("-loglevel");
	args.push_back("warning");

	args.push_back("-i");
	args.push_back(ProjectSettings::get_singleton()->globalize_path(p_source_file));

	if (!String(p_options["additional_arguments"]).is_empty()) {
		Vector<String> additional_arguments = String(p_options["additional_arguments"]).split(" ", false);
		for (const String &additional_argument : additional_arguments) {
			args.push_back(additional_argument);
		}
	}

	int scale_width = p_options["scale_width"];
	int scale_height = p_options["scale_height"];
	String filter = "";
	if (scale_width > 0 || scale_height > 0) {
		filter += "[0:v]scale=" + String::num_int64(scale_width) + ":" + String::num_int64(scale_height);
	}

	if (p_options["alpha_to_color"] != Color()) {
		if (!filter.is_empty()) {
			filter += "[s];[s]";
		}
		filter += "split=2[bg][fg];[bg]drawbox=c=#" + Color(p_options["alpha_to_color"]).to_html(false) + ":replace=1:t=fill[bg];[bg][fg]overlay=format=auto";
	}

	if (!filter.is_empty()) {
		args.push_back("-filter_complex");
		args.push_back(filter);
	}

	args.push_back("-q:v");
	args.push_back(Variant((float)(p_options["video_quality"]) * 10.0));

	args.push_back("-q:a");
	args.push_back(Variant((float)(p_options["audio_quality"]) * 10.0));

	args.push_back("-g:v");
	args.push_back(p_options["keyframe_interval"]);

	args.push_back("-speed_level");
	args.push_back(p_options["encoding_speed"]);

	args.push_back("-y");

	args.push_back(ProjectSettings::get_singleton()->globalize_path(p_save_path + ".ogv"));

	String str;
	int exitcode = 0;

	Error err = OS::get_singleton()->execute(ffmpeg_path, args, &str, &exitcode, true);

	if (err != OK || exitcode != 0) {
		print_verbose(str);
		return FAILED;
	}

	return OK;
}
