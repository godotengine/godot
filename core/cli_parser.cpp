/*************************************************************************/
/*  cli_parser.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "cli_parser.h"
#include "core/os/os.h"
#include "core/translation.h"
#include "core/version.h"
#include "core/version_hash.gen.h"
#include "main/tests/test_main.h"

#define COMMAND_SEPARATOR "--"

// Checkers

struct CommandFlag::CommandChecker {

	virtual bool check(const String &p_arg) const = 0;

	String error_msg;

	virtual ~CommandChecker() {}
};

struct CommandFlag::FunctionChecker : public CommandFlag::CommandChecker {

	bool check(const String &p_arg) const override {
		return function(p_arg);
	}

	CommandFlag::check_function function;
};

struct CommandFlag::ObjectChecker : public CommandFlag::CommandChecker {

	bool check(const String &p_arg) const override {
		Variant::CallError ce;
		Variant v_arg = (Variant)p_arg;
		const Variant *arg[1] = { &v_arg };

		bool res = obj->call(func, arg, 1, ce);
		if (ce.error != Variant::CallError::CALL_OK) {
			res = false;
		}
		return res;
	}

	StringName func;
	Object *obj;
};

// CommandFlag

void CommandFlag::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_argument_name", "argument_name"), &CommandFlag::set_argument_name);
	ClassDB::bind_method(D_METHOD("get_argument_name"), &CommandFlag::get_argument_name);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "argument_name"), "set_argument_name", "get_argument_name");
	ClassDB::bind_method(D_METHOD("set_description", "description"), &CommandFlag::set_description);
	ClassDB::bind_method(D_METHOD("get_description"), &CommandFlag::get_description);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "description"), "set_description", "get_description");
	ClassDB::bind_method(D_METHOD("set_category", "category"), &CommandFlag::set_category);
	ClassDB::bind_method(D_METHOD("get_category"), &CommandFlag::get_category);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "category"), "set_category", "get_category");
	ClassDB::bind_method(D_METHOD("set_flag", "flag"), &CommandFlag::set_flag);
	ClassDB::bind_method(D_METHOD("get_flag"), &CommandFlag::get_flag);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "flag"), "set_flag", "get_flag");
	ClassDB::bind_method(D_METHOD("set_short_flag", "args"), &CommandFlag::set_short_flag);
	ClassDB::bind_method(D_METHOD("get_short_flag"), &CommandFlag::get_short_flag);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "short_flag"), "set_short_flag", "get_short_flag");
	ClassDB::bind_method(D_METHOD("set_show_in_help", "show_in_help"), &CommandFlag::set_show_in_help);
	ClassDB::bind_method(D_METHOD("get_show_in_help"), &CommandFlag::get_show_in_help);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "show_in_help"), "set_show_in_help", "get_show_in_help");

	ClassDB::bind_method(D_METHOD("needs_argument"), &CommandFlag::needs_argument);
	ClassDB::bind_method(D_METHOD("set_flags", "flag", "short_flag"), &CommandFlag::set_flags);
	ClassDB::bind_method(D_METHOD("set_data", "description", "category"), &CommandFlag::set_data);
	ClassDB::bind_method(D_METHOD("add_object_checker", "obj", "func", "err_msg"), &CommandFlag::add_object_checker);
	ClassDB::bind_method(D_METHOD("clear_checkers"), &CommandFlag::clear_checkers);
}

bool CommandFlag::locale_checker(const String &p_arg) {
	String univ_locale = TranslationServer::standardize_locale(p_arg);
	return TranslationServer::is_locale_valid(univ_locale);
}

bool CommandFlag::numeric_checker(const String &p_arg) {
	return p_arg.is_numeric();
}

bool CommandFlag::resolution_checker(const String &p_arg) {
	bool res = (p_arg.find("x") != -1) &&
			   (p_arg.get_slice("x", 0).to_int() <= 0) &&
			   (p_arg.get_slice("x", 1).to_int() <= 0);
	return res;
}

bool CommandFlag::position_checker(const String &p_arg) {
	Vector<String> coords = p_arg.split(",");
	return coords.size() == 2 && coords[0].is_numeric() && coords[1].is_numeric();
}

bool CommandFlag::host_address_checker(const String &p_arg) {
	return p_arg.find(":") != -1;
}

bool CommandFlag::audio_driver_checker(const String &p_arg) {
	bool res = false;
	for (int i = 0; i < OS::get_singleton()->get_audio_driver_count(); i++) {
		res = (OS::get_singleton()->get_audio_driver_name(i) == p_arg);
		if (res)
			break;
	}
	return res;
}

bool CommandFlag::video_driver_checker(const String &p_arg) {
	bool res = false;
	for (int i = 0; i < OS::get_singleton()->get_video_driver_count(); i++) {
		res = (OS::get_singleton()->get_video_driver_name(i) == p_arg);
		if (res)
			break;
	}
	return res;
}

bool CommandFlag::render_thread_checker(const String &p_arg) {
	return p_arg == "unsafe" || p_arg == "safe" || p_arg == "separate";
}

void CommandFlag::set_argument_name(const String &p_arg_name) {

	_arg_name = p_arg_name;
}

String CommandFlag::get_argument_name() const {
	return _arg_name;
}

void CommandFlag::set_description(const String &p_description) {
	_description = p_description;
}

String CommandFlag::get_description() const {
	return _description;
}

void CommandFlag::set_category(const String &p_category) {
	_category = p_category;
}

String CommandFlag::get_category() const {
	return _category;
}

void CommandFlag::set_flag(const String &p_flag) {
	_flag = p_flag;
}

String CommandFlag::get_flag() const {
	return _flag;
}

void CommandFlag::set_short_flag(const String &p_short_flag) {
	_short_flag = p_short_flag;
}

String CommandFlag::get_short_flag() const {
	return _short_flag;
}

void CommandFlag::set_show_in_help(const bool p_enabled) {
	_show_in_help = p_enabled;
}

bool CommandFlag::get_show_in_help() const {
	return _show_in_help;
}

void CommandFlag::set_flags(const String &p_flag, const String &p_short_flag) {

	_flag = p_flag;
	_short_flag = p_short_flag;
}

void CommandFlag::set_data(const String &p_description, const String &p_category) {

	_description = p_description;
	_category = p_category;
}

void CommandFlag::add_checker(check_function p_f, const String &p_error_msg) {
	FunctionChecker *check = memnew(FunctionChecker);
	check->function = p_f;
	check->error_msg = p_error_msg;
	check_list.push_back(check);
}

void CommandFlag::add_object_checker(Object *p_obj, const StringName &p_function, const String &p_error_msg) {
	ObjectChecker *check = memnew(ObjectChecker);
	check->error_msg = p_error_msg;
	check->obj = p_obj;
	check->func = p_function;
	check_list.push_back(check);
}

void CommandFlag::clear_checkers() {
	for (int i = 0; i < check_list.size(); i++) {
		memdelete(check_list[i]);
	}
	check_list.clear();
}

bool CommandFlag::needs_argument() const {

	return _arg_name.length() > 0;
}

CommandFlag::CommandFlag() :
		_show_in_help(true) {
}

CommandFlag::CommandFlag(const String &p_flag) :
		_show_in_help(true) {
	_flag = p_flag;
}

CommandFlag::CommandFlag(const String &p_flag, const String &p_short_flag) :
		_show_in_help(true) {
	_flag = p_flag;
	_short_flag = p_short_flag;
}

CommandFlag::~CommandFlag() {
	clear_checkers();
}

// CommandParser

void CommandParser::_bind_methods() {

	ClassDB::bind_method(D_METHOD("parse_arguments", "args"), &CommandParser::parse_arguments);
	ClassDB::bind_method(D_METHOD("print_help"), &CommandParser::print_help);
	ClassDB::bind_method(D_METHOD("print_version"), &CommandParser::print_version);
	ClassDB::bind_method(D_METHOD("add_command", "command"), &CommandParser::add_command);
	ClassDB::bind_method(D_METHOD("is_argument_set", "flag"), &CommandParser::is_argument_set);
	ClassDB::bind_method(D_METHOD("needs_argument", "flag"), &CommandParser::needs_argument);
	ClassDB::bind_method(D_METHOD("get_argument", "flag"), &CommandParser::get_argument);
	ClassDB::bind_method(D_METHOD("has_scene_defined"), &CommandParser::has_scene_defined);
	ClassDB::bind_method(D_METHOD("has_project_defined"), &CommandParser::has_project_defined);
	ClassDB::bind_method(D_METHOD("has_script_defined"), &CommandParser::has_script_defined);
	ClassDB::bind_method(D_METHOD("has_shader_defined"), &CommandParser::has_shader_defined);
	ClassDB::bind_method(D_METHOD("clear"), &CommandParser::clear);
	ClassDB::bind_method(D_METHOD("check_command_flag_collision"), &CommandParser::check_command_flag_collision);

	ClassDB::bind_method(D_METHOD("set_search_project_file", "enable"), &CommandParser::set_search_project_file);
	ClassDB::bind_method(D_METHOD("get_search_project_file"), &CommandParser::get_search_project_file);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "search_project_file"), "set_search_project_file", "get_search_project_file");
	ClassDB::bind_method(D_METHOD("set_help_header", "help_header"), &CommandParser::set_help_header);
	ClassDB::bind_method(D_METHOD("get_help_header"), &CommandParser::get_help_header);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "help_header"), "set_help_header", "get_help_header");
	ClassDB::bind_method(D_METHOD("set_help_footer", "help_footer"), &CommandParser::set_help_footer);
	ClassDB::bind_method(D_METHOD("get_help_footer"), &CommandParser::get_help_footer);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "help_footer"), "set_help_footer", "get_help_footer");
	ClassDB::bind_method(D_METHOD("set_version", "version"), &CommandParser::set_version);
	ClassDB::bind_method(D_METHOD("get_version"), &CommandParser::get_version);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "version"), "set_version", "get_version");
}

void CommandParser::init_engine_defaults() {

	_separator_enabled = true;
	_search_project_file = true;

	_version = get_full_version_string();
	_help_header = String(VERSION_NAME) + " v" + _version + " - https://godotengine.org\n" +
				   "(c) 2007-2019 Juan Linietsky, Ariel Manzur.\n" +
				   "(c) 2014-2019 Godot Engine contributors.\n\n" +
				   "Usage: " + OS::get_singleton()->get_executable_path().utf8().get_data() +
				   " [options] [path to scene or 'project.godot' file] " + String(COMMAND_SEPARATOR) + " [project options]\n";
	// General options
	Ref<CommandFlag> help_command = memnew(CommandFlag("--help", "-h"));
	help_command->set_data("Display this help message.", "General options");
	add_command(help_command);

	Ref<CommandFlag> version_command = memnew(CommandFlag("--version"));
	version_command->set_data("Display the version string.", "General options");
	add_command(version_command);

	Ref<CommandFlag> verbose_command = memnew(CommandFlag("--verbose", "-v"));
	verbose_command->set_data("Use verbose stdout mode.", "General options");
	add_command(verbose_command);

	Ref<CommandFlag> quiet_command = memnew(CommandFlag("--quiet"));
	quiet_command->set_data("Quiet mode, silences stdout messages. Errors are still displayed.", "General options");
	add_command(quiet_command);

	// Run options
	Ref<CommandFlag> editor_command = memnew(CommandFlag("--editor", "-e"));
	editor_command->set_data("Start the editor instead of running the scene.", "Run options");
	add_command(editor_command);

	Ref<CommandFlag> project_manager_command = memnew(CommandFlag("--project-manager", "-p"));
	project_manager_command->set_data("Start the project manager, even if a project is auto-detected.", "Run options");
	add_command(project_manager_command);

	Ref<CommandFlag> quit_command = memnew(CommandFlag("--quit", "-q"));
	quit_command->set_data("Quit after the first iteration.", "Run options");
	add_command(quit_command);

	Ref<CommandFlag> language_command = memnew(CommandFlag("--language", "-l"));
	language_command->set_data("Use a specific locale (<locale> being a two-letter code).", "Run options");
	language_command->set_argument_name("<locale>");
	language_command->add_checker(CommandFlag::locale_checker, "Invalid locale.");
	add_command(language_command);

	Ref<CommandFlag> path_command = memnew(CommandFlag("--path"));
	path_command->set_data("Path to a project (<directory> must contain a 'project.godot' file).", "Run options");
	path_command->set_argument_name("<directory>");
	add_command(path_command);

	Ref<CommandFlag> upwards_command = memnew(CommandFlag("--upwards", "-u"));
	upwards_command->set_data("Scan folders upwards for project.godot file.", "Run options");
	add_command(upwards_command);

	Ref<CommandFlag> main_pack_command = memnew(CommandFlag("--main-pack"));
	main_pack_command->set_data("Path to a pack (.pck) file to load.", "Run options");
	main_pack_command->set_argument_name("<file>");
	add_command(main_pack_command);

	Ref<CommandFlag> render_thread_command = memnew(CommandFlag("--render-thread"));
	render_thread_command->set_data("Render thread mode ('unsafe', 'safe', 'separate').", "Run options");
	render_thread_command->set_argument_name("<mode>");
	render_thread_command->add_checker(CommandFlag::render_thread_checker, "The argument must be a valid render-thread.");
	add_command(render_thread_command);

	Ref<CommandFlag> remote_fs_command = memnew(CommandFlag("--remote-fs"));
	remote_fs_command->set_data("Remote filesystem (<host/IP>[:<port>] address).", "Run options");
	remote_fs_command->set_argument_name("<address>");
	add_command(remote_fs_command);

	Ref<CommandFlag> remote_fs_password_command = memnew(CommandFlag("--remote-fs-password"));
	remote_fs_password_command->set_data("Password for remote filesystem.", "Run options");
	remote_fs_password_command->set_argument_name("<password>");
	add_command(remote_fs_password_command);

	Ref<CommandFlag> audio_driver_command = memnew(CommandFlag("--audio-driver"));
	String audio_driver_description = "Audio driver (";
	for (int i = 0; i < OS::get_singleton()->get_audio_driver_count(); i++) {
		if (i != 0)
			audio_driver_description += ", ";
		audio_driver_description += OS::get_singleton()->get_audio_driver_name(i);
	}
	audio_driver_description += ").";
	audio_driver_command->set_data(audio_driver_description, "Run options");
	audio_driver_command->add_checker(CommandFlag::audio_driver_checker, "The argument is an invalid audio driver.");
	add_command(audio_driver_command);

	Ref<CommandFlag> video_driver_command = memnew(CommandFlag("--video-driver"));
	String video_driver_description = "Video driver (";
	for (int i = 0; i < OS::get_singleton()->get_video_driver_count(); i++) {
		if (i != 0)
			video_driver_description += ", ";
		video_driver_description += OS::get_singleton()->get_video_driver_name(i);
	}
	video_driver_description += ").";
	video_driver_command->set_data(video_driver_description, "Run options");
	video_driver_command->add_checker(CommandFlag::video_driver_checker, "The argument is an invalid video driver.");
	add_command(video_driver_command);

	// Display options
#ifndef SERVER_ENABLED
	Ref<CommandFlag> fullscreen_command = memnew(CommandFlag("--fullscreen", "-f"));
	fullscreen_command->set_data("Request fullscreen mode.", "Display options");
	add_command(fullscreen_command);

	Ref<CommandFlag> maximized_command = memnew(CommandFlag("--maximized", "-m"));
	maximized_command->set_data("Request a maximized window.", "Display options");
	add_command(maximized_command);

	Ref<CommandFlag> windowed_command = memnew(CommandFlag("--windowed", "-w"));
	windowed_command->set_data("Request windowed mode.", "Display options");
	add_command(windowed_command);

	Ref<CommandFlag> always_on_top_command = memnew(CommandFlag("--always-on-top", "-t"));
	always_on_top_command->set_data("Request an always-on-top window.", "Display options");
	add_command(always_on_top_command);

	Ref<CommandFlag> resolution_command = memnew(CommandFlag("--resolution"));
	resolution_command->set_data("Request window resolution.", "Display options");
	resolution_command->set_argument_name("<W>x<H>");
	resolution_command->add_checker(CommandFlag::resolution_checker, "Invalid resolution, it should be e.g. '1280x720' and the values must be above 0.");
	add_command(resolution_command);

	Ref<CommandFlag> position_command = memnew(CommandFlag("--position"));
	position_command->set_data("Request window position.", "Display options");
	position_command->set_argument_name("<X>,<Y>");
	position_command->add_checker(CommandFlag::position_checker, "Invalid position, it should be e.g. '80,128'.");
	add_command(position_command);

	Ref<CommandFlag> low_dpi_command = memnew(CommandFlag("--low-dpi"));
	low_dpi_command->set_data("Force low-DPI mode (macOS and Windows only).", "Display options");
	add_command(low_dpi_command);

	Ref<CommandFlag> no_window_command = memnew(CommandFlag("--no-window"));
	no_window_command->set_data("Disable window creation (Windows only). Useful together with --script.", "Display options");
	add_command(no_window_command);

	Ref<CommandFlag> no_window_command = memnew(CommandFlag("--enable-vsync-via-compositor"));
	no_window_command->set_data("When vsync is enabled, vsync via the OS' window compositor (Windows only).", "Display options");
	add_command(no_window_command);

	Ref<CommandFlag> no_window_command = memnew(CommandFlag("--disable-vsync-via-compositor"));
	no_window_command->set_data("Disable vsync via the OS' window compositor (Windows only).", "Display options");
	add_command(no_window_command);
#endif

	// Debug options
	Ref<CommandFlag> steal_pid_command = memnew(CommandFlag("--allow_focus_steal_pid"));
	steal_pid_command->set_data("Allow set window to the foreground.", "Debug options");
	steal_pid_command->set_argument_name("<PID>");
	add_command(steal_pid_command);

	Ref<CommandFlag> debug_command = memnew(CommandFlag("--debug", "-d"));
	debug_command->set_data("Debug (local stdout debugger).", "Debug options");
	add_command(debug_command);

	Ref<CommandFlag> breakpoints_command = memnew(CommandFlag("--breakpoints", "-b"));
	breakpoints_command->set_data("Breakpoint list as source::line comma-separated pairs, no spaces (use %%20 instead).", "Debug options");
	breakpoints_command->set_argument_name("<pairs>");
	add_command(breakpoints_command);

	Ref<CommandFlag> profiling_command = memnew(CommandFlag("--profiling"));
	profiling_command->set_data("Enable profiling in the script debugger.", "Debug options");
	add_command(profiling_command);

	Ref<CommandFlag> remote_debug_command = memnew(CommandFlag("--remote-debug"));
	remote_debug_command->set_data("Remote debug (<host/IP>:<port> address).", "Debug options");
	remote_debug_command->set_argument_name("<address>");
	remote_debug_command->add_checker(CommandFlag::host_address_checker, "Invalid debug host address, it should be of the form <host/IP>:<port>.");
	add_command(remote_debug_command);

#if defined(DEBUG_ENABLED) && !defined(SERVER_ENABLED)
	Ref<CommandFlag> debug_collisions_command = memnew(CommandFlag("--debug-collisions"));
	debug_collisions_command->set_data("Show collisions shapes when running the scene.", "Debug options");
	add_command(debug_collisions_command);

	Ref<CommandFlag> debug_navigation_command = memnew(CommandFlag("--debug-navigation"));
	debug_navigation_command->set_data("Show navigation polygons when running the scene.", "Debug options");
	add_command(debug_navigation_command);
#endif

	Ref<CommandFlag> frame_delay_command = memnew(CommandFlag("--frame-delay"));
	frame_delay_command->set_data("Simulate high CPU load (delay each frame by <ms> milliseconds).", "Debug options");
	frame_delay_command->set_argument_name("<ms>");
	frame_delay_command->add_checker(CommandFlag::numeric_checker, "Invalid delay value. Must be numeric above 0");
	add_command(frame_delay_command);

	Ref<CommandFlag> time_scale_command = memnew(CommandFlag("--time-scale"));
	time_scale_command->set_data("Force time scale (higher values are faster, 1.0 is normal speed).", "Debug options");
	time_scale_command->set_argument_name("<scale>");
	time_scale_command->add_checker(CommandFlag::numeric_checker, "Invalid scale value. Must be numeric above 0");
	add_command(time_scale_command);

	Ref<CommandFlag> disable_render_loop_command = memnew(CommandFlag("--disable-render-loop"));
	disable_render_loop_command->set_data("Disable render loop so rendering only occurs when called explicitly from script.", "Debug options");
	add_command(disable_render_loop_command);

	Ref<CommandFlag> disable_crash_handler_command = memnew(CommandFlag("--disable-crash-handler"));
	disable_crash_handler_command->set_data("Disable crash handler when supported by the platform code.", "Debug options");
	add_command(disable_crash_handler_command);

	Ref<CommandFlag> skip_breakpoints_command = memnew(CommandFlag("--skip-breakpoints"));
	skip_breakpoints_command->set_data("Disable breakpoints.", "Debug options");
	add_command(skip_breakpoints_command);

	Ref<CommandFlag> fixed_fps_command = memnew(CommandFlag("--fixed-fps"));
	fixed_fps_command->set_data("Force a fixed number of frames per second. This setting disables real-time synchronization.", "Debug options");
	fixed_fps_command->set_argument_name("<fps>");
	fixed_fps_command->add_checker(CommandFlag::numeric_checker, "Invalid fps value. Must be numeric above 0.");
	add_command(fixed_fps_command);

	Ref<CommandFlag> print_fps_command = memnew(CommandFlag("--print-fps"));
	print_fps_command->set_data("Print the frames per second to the stdout.", "Debug options");
	add_command(print_fps_command);

	// Standalone tools
	Ref<CommandFlag> script_command = memnew(CommandFlag("--script", "-s"));
	script_command->set_data("Run a script.", "Standalone options");
	script_command->set_argument_name("<script>");
	add_command(script_command);

	Ref<CommandFlag> check_only_command = memnew(CommandFlag("--check-only"));
	check_only_command->set_data("Only parse for errors and quit (use with --script).", "Standalone options");
	add_command(check_only_command);

#ifdef TOOLS_ENABLED
	Ref<CommandFlag> export_command = memnew(CommandFlag("--export"));
	export_command->set_data("Export the project using the given export target. Export only main pack if path ends with .pck or .zip.", "Standalone options");
	export_command->set_argument_name("<target>");
	add_command(export_command);

	Ref<CommandFlag> export_debug_command = memnew(CommandFlag("--export-debug"));
	export_debug_command->set_data("Like --export, but use debug template.", "Standalone options");
	export_debug_command->set_argument_name("<target>");
	add_command(export_debug_command);

	Ref<CommandFlag> doctool_command = memnew(CommandFlag("--doctool"));
	doctool_command->set_data("Dump the engine API reference to the given <path> in XML format, merging if existing files are found.", "Standalone options");
	doctool_command->set_argument_name("<path>");
	add_command(doctool_command);

	Ref<CommandFlag> no_docbase_command = memnew(CommandFlag("--no-docbase"));
	no_docbase_command->set_data("Disallow dumping the base types (used with --doctool).", "Standalone options");
	add_command(no_docbase_command);

	Ref<CommandFlag> build_solutions_command = memnew(CommandFlag("--build-solutions"));
	build_solutions_command->set_data("Build the scripting solutions (e.g. for C# projects).", "Standalone options");
	add_command(build_solutions_command);

#ifdef DEBUG_METHODS_ENABLED
	Ref<CommandFlag> gdnative_generate_json_api_command = memnew(CommandFlag("--gdnative-generate-json-api"));
	gdnative_generate_json_api_command->set_data("Generate JSON dump of the Godot API for GDNative bindings.", "Standalone options");
	add_command(gdnative_generate_json_api_command);
#endif
	Ref<CommandFlag> test_command = memnew(CommandFlag("--test"));
	String test_description = "Run a unit test (";
	const char **test_names = tests_get_names();
	const char *comma = "";
	while (*test_names) {
		test_description += comma;
		test_description += *test_names;
		test_names++;
		comma = ", ";
	}
	test_description += ").";
	test_command->set_data(test_description, "Standalone options");
	test_command->set_argument_name("<test>");
	add_command(test_command);
#endif
}

Error CommandParser::parse() {

	_data_found.clear();
	_game_args.clear();
	_project_file.clear();

	int args_flag_pos = _args.size();

	for (int i = 0; i < _args.size(); i++) {

		// Split engine commands from project commands
		if (_separator_enabled && _args[i] == COMMAND_SEPARATOR) {
			args_flag_pos = i;
		}
	}

	// We need COMMAND_SEPARATOR and an actual command at the end to have
	// content.
	if (args_flag_pos + 1 < _args.size()) {

		for (int i = args_flag_pos + 1; i < _args.size(); i++) {
			_game_args.push_back(_args[i]);
		}
	}

	if (args_flag_pos == 0) {
		return OK;
	}

	if (_search_project_file) {
		// Project file defined: scene, script, path, etc.
		// It is viable to check if we have a file or path a the end if we only
		// have one engine arg or if the previous command doesn't require argument.
		const bool file_viable = (args_flag_pos == 1) || (args_flag_pos > 1 && !needs_argument(_args[args_flag_pos - 2]));
		const String &last_engine_arg = _args[args_flag_pos - 1];

		if (file_viable && !last_engine_arg.begins_with("-")) {

			_project_file = last_engine_arg;
			args_flag_pos--;

			if (args_flag_pos == 0) {
				return OK;
			}
		}
	}

	Vector<String> engine_args;
	engine_args.resize(args_flag_pos);
	for (int i = 0; i < args_flag_pos; i++) {
		engine_args.write[i] = _args[i];
	}

	for (int i = 0; i < engine_args.size(); i++) {
		String arg = engine_args[i];
		String value = "";

		const CommandFlag *command = find_command(arg);
		if (!command) {
			print_line("'" + arg + "' is not a valid command. Use '" + OS::get_singleton()->get_executable_path() + " --help' to get help.");
			// Try to suggest the correct command.
			command = find_most_similar_command(arg);
			if (command) {
				print_line("  Maybe you wanted to use: '" + command->_flag + "'.");
			}
			return ERR_INVALID_DATA;
		}

		if (command->needs_argument()) {

			i++;
			if (i == engine_args.size()) {
				print_command_error(arg, "Missing argument.");
				return ERR_INVALID_DATA;
			}

			value = engine_args[i];

			for (int j = 0; j < command->check_list.size(); j++) {

				if (!command->check_list[j]->check(value)) {
					print_command_error(arg + " " + value, command->check_list[j]->error_msg);
					return ERR_INVALID_DATA;
				}
			}
		}
		// Register 2 to be able to search result by short and normal form.
		_data_found[command->_flag] = value;
		_data_found[command->_short_flag] = value;
	}
	return OK;
}

Error CommandParser::parse_arguments(const PoolStringArray &p_args) {

	List<String> args;
	PoolStringArray::Read r = p_args.read();
	for (int i = 0; i < p_args.size(); i++) {
		args.push_back(r[i]);
	}
	set_cmdline_args(args);
	return parse();
}

void CommandParser::print_help() const {

	int longest_length = 0;

	Vector<String> tmp_lines;
	tmp_lines.resize(_commands.size());

	// Build the formated " -x, --xxxxx" and save the longest size
	// to align the descriptions.
	for (int i = 0; i < _commands.size(); i++) {

		const CommandFlag *command = _commands[i].ptr();
		if (!command->_show_in_help) {
			continue;
		}
		String line = "  ";
		if (!command->_short_flag.empty()) {
			line += command->_short_flag + ", ";
		}
		line += command->_flag;
		if (!command->_arg_name.empty()) {
			line += " " + command->_arg_name + "  ";
		}

		longest_length = MAX(longest_length, line.size());
		tmp_lines.write[i] = line;
	}

	HashMap<String, Vector<String> > category_data;
	Vector<String> ordered_categories;

	// Fill category_data and ordered_categories
	for (int i = 0; i < _commands.size(); i++) {

		String line = tmp_lines[i].rpad(longest_length);
		const String description = _commands[i]->_description;
		line += description;

		const String &category = _commands[i]->_category;

		if (category_data.has(category)) {

			category_data[category].push_back(line);
		} else {

			Vector<String> cat_lines;
			cat_lines.push_back(line);
			category_data[category] = cat_lines;

			ordered_categories.push_back(category);
		}
	}

	ordered_categories.sort();

	// Start printing
	print_line(_help_header);

	for (int i = 0; i < ordered_categories.size(); i++) {

		const String &category = ordered_categories[i];

		const Vector<String> &lines = category_data[category];

		print_line(" "); //add a blank line for readability
		if (!category.empty()) {
			print_line(category + ":");
		}
		for (int j = 0; j < lines.size(); j++) {
			print_line(lines[j]);
		}
	}

	print_line(_help_footer);
	print_line(" "); //add a blank line for readability
}

void CommandParser::print_version() const {
	print_line(_version);
}

void CommandParser::add_command(const Ref<CommandFlag> &p_command) {

	_commands.push_back(p_command);
}

bool CommandParser::is_argument_set(const String &p_flag) const {

	return _data_found.has(p_flag);
}

bool CommandParser::needs_argument(const String &p_flag) const {

	const CommandFlag *command = find_command(p_flag);
	return command && command->needs_argument();
}

String CommandParser::get_argument(const String &p_flag) const {

	const String *result = _data_found.getptr(p_flag);
	if (!result) {
		return "";
	}

	return *result;
}

String CommandParser::get_defined_project_file() const {
	return _project_file;
}

bool CommandParser::has_scene_defined() const {

	return _project_file.ends_with(".tscn") || _project_file.ends_with(".scn") || _project_file.ends_with(".escn");
}

bool CommandParser::has_project_defined() const {

	return _project_file.ends_with("project.godot");
}

bool CommandParser::has_script_defined() const {

	return _project_file.ends_with(".gd") || _project_file.ends_with(".gdc");
}

bool CommandParser::has_shader_defined() const {

	return _project_file.ends_with(".shader");
}

List<String> CommandParser::get_project_args() const {
	List<String> res;
	for (int i = 0; i < _game_args.size(); i++) { // TODOcli
		res.push_back(_game_args[i]);
	}
	return res;
}

List<String> CommandParser::get_args() const {
	List<String> res;
	for (int i = 0; i < _args.size(); i++) { // TODOcli
		res.push_back(_args[i]);
	}
	return res;
}

void CommandParser::set_cmdline_args(const List<String> &p_args) {
	_game_args.clear();
	_args.clear();

	for (const List<String>::Element *E = p_args.front(); E; E = E->next()) {
		// Unescape cmd commands
		_args.push_back(E->get().strip_edges().replace("%20", " "));
	}
}

void CommandParser::clear() {
	_commands.clear();
	_game_args.clear();
	_data_found.clear();
	_project_file.clear();
}

void CommandParser::set_help_header(const String &p_help_header) {
	_help_header = p_help_header;
}

String CommandParser::get_help_header() const {
	return _help_header;
}

void CommandParser::set_help_footer(const String &p_help_footer) {
	_help_footer = p_help_footer;
}

String CommandParser::get_help_footer() const {
	return _help_footer;
}

void CommandParser::set_version(const String &p_version) {
	_version = p_version;
}

String CommandParser::get_version() const {
	return _version;
}

void CommandParser::set_search_project_file(const bool p_enable) {
	_search_project_file = p_enable;
}

bool CommandParser::get_search_project_file() const {
	return _search_project_file;
}

bool CommandParser::check_command_flag_collision() const {
	Set<String> ocurrences;
	Vector<String> repeated;

	for (int i = 0; i < _commands.size(); i++) {

		if (ocurrences.has(_commands[i]->_flag)) {
			repeated.push_back(_commands[i]->_flag);
		} else {
			ocurrences.insert(_commands[i]->_flag);
		}
		if (ocurrences.has(_commands[i]->_short_flag)) {
			repeated.push_back(_commands[i]->_short_flag);
		} else {
			ocurrences.insert(_commands[i]->_short_flag);
		}
	}

	bool has_collision = repeated.size() != 0;

	if (has_collision) {

		String msg = "Repeated flags: " + repeated[0];
		for (int i = 1; i < repeated.size(); i++) {
			msg += ", " + repeated[i];
		}
		msg += ".";

		print_line(msg);
	}

	return has_collision;
}

String CommandParser::get_full_version_string() const {

	String version(VERSION_FULL_BUILD);

	String hash = String(VERSION_HASH);
	if (hash.length() != 0) {
		hash = "." + hash.left(9);
	}
	return version + hash;
}

const CommandFlag *CommandParser::find_command(const String &p_flag) const {

	const CommandFlag *res = NULL;

	for (int i = 0; i < _commands.size(); i++) {

		if (_commands[i]->_flag == p_flag || _commands[i]->_short_flag == p_flag) {
			res = _commands[i].ptr();
		}
	}
	return res;
}

const CommandFlag *CommandParser::find_most_similar_command(const String &p_flag) const {

	const CommandFlag *res = NULL;

	float max_similarity = 0.0;
	for (int i = 0; i < _commands.size(); i++) {

		float similarity = MAX(_commands[i]->_flag.similarity(p_flag), _commands[i]->_short_flag.similarity(p_flag));

		if (max_similarity < similarity) {
			res = _commands[i].ptr();
			max_similarity = similarity;
		}
	}
	if (max_similarity < 0.5) { // Don't return unrelated commands
		res = NULL;
	}
	return res;
}

void CommandParser::print_command_error(const String &cause, const String &msg) const {

	print_line("Error in '" + cause + "': " + msg);
}

CommandParser::CommandParser() :
		_separator_enabled(false),
		_search_project_file(false) {
}

CommandParser::~CommandParser() {
}