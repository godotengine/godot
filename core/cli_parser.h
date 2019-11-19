/*************************************************************************/
/*  cli_parser.h                                                        */
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

#ifndef CLIPARSER_H
#define CLIPARSER_H

#include "core/object.h"
#include "core/reference.h"

// Types

class CommandParser;

class CommandFlag : public Reference {

	GDCLASS(CommandFlag, Reference);

protected:
	static void _bind_methods();

public:
	// Use functions in the format bool check(const String &data)
	typedef bool (*check_function)(const String &);

	// Checkers
	// They are exposed so they can be reused in other parts of the engine
	static bool locale_checker(const String &p_arg);
	static bool numeric_checker(const String &p_arg);
	static bool resolution_checker(const String &p_arg);
	static bool position_checker(const String &p_arg);
	static bool host_address_checker(const String &p_arg);

private:
	// Private checkers
	static bool audio_driver_checker(const String &p_arg);
	static bool video_driver_checker(const String &p_arg);
	static bool render_thread_checker(const String &p_arg);

	friend class CommandParser;

	// Flags for the command. e.g help, h, version, v.
	String _flag;
	String _short_flag;
	// Name of the argument passed to the flag, empty if none.
	String _arg_name;
	// Command description
	String _description;
	// Command category, commands sharing the same category are grouped
	// together in the help message.
	String _category;

	bool _show_in_help;

	struct CommandChecker;
	struct FunctionChecker;
	struct ObjectChecker;

	Vector<CommandChecker *> check_list;

public:
	void set_argument_name(const String &p_arg_name);
	String get_argument_name() const;
	void set_description(const String &p_description);
	String get_description() const;
	void set_category(const String &p_category);
	String get_category() const;
	void set_flag(const String &p_flag);
	String get_flag() const;
	void set_short_flag(const String &p_short_flag);
	String get_short_flag() const;
	void set_show_in_help(const bool p_enabled);
	bool get_show_in_help() const;

	void set_flags(const String &p_flag, const String &p_short_flag = "");
	void set_data(const String &p_description, const String &p_category = "");
	void add_checker(check_function p_f, const String &p_error_msg = "");
	void add_object_checker(Object *p_obj, const StringName &p_function, const String &p_error_msg);
	void clear_checkers();
	bool needs_argument() const;

	CommandFlag();
	CommandFlag(const String &p_flag);
	CommandFlag(const String &p_flag, const String &p_short_flag);
	virtual ~CommandFlag();
};

class CommandParser : public Object {

	GDCLASS(CommandParser, Object);

protected:
	static void _bind_methods();

private:
	Vector<Ref<CommandFlag> > _commands;

	Vector<String> _game_args;
	Vector<String> _args;

	// Ocurrences of the parsed data and their values.
	HashMap<String, String> _data_found;

	String _project_file;

	String _help_header;
	String _help_footer;
	String _version;

	// Utility
	String get_full_version_string() const;
	const CommandFlag *find_command(const String &p_flag) const;
	const CommandFlag *find_most_similar_command(const String &p_flag) const;
	void print_command_error(const String &cause, const String &msg) const;

	bool _separator_enabled;
	bool _search_project_file;

public:
	void init_engine_defaults();

	Error parse();
	// Method to expose to scriping API
	Error parse_arguments(const PoolStringArray &p_args);

	void print_help() const;
	void print_version() const;

	void add_command(const Ref<CommandFlag> &p_command);

	bool is_argument_set(const String &p_flag) const;
	bool needs_argument(const String &p_flag) const;
	String get_argument(const String &p_flag) const;

	String get_defined_project_file() const;
	bool has_scene_defined() const;
	bool has_project_defined() const;
	bool has_script_defined() const;
	bool has_shader_defined() const;
	List<String> get_project_args() const;
	List<String> get_args() const;
	void set_cmdline_args(const List<String> &p_args);

	void clear();

	void set_help_header(const String &p_help_header);
	String get_help_header() const;
	void set_help_footer(const String &p_help_footer);
	String get_help_footer() const;
	void set_version(const String &p_version);
	String get_version() const;
	void set_search_project_file(const bool p_enable);
	bool get_search_project_file() const;

	bool check_command_flag_collision() const;

	CommandParser();
	virtual ~CommandParser();
};

#endif // CLIPARSER_H
