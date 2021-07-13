/*************************************************************************/
/*  command_line_parser.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef COMMAND_LINE_PARSER_H
#define COMMAND_LINE_PARSER_H

#include "core/object/reference.h"
#include "core/templates/ordered_hash_map.h"

class CommandLineOption : public Reference {
	GDCLASS(CommandLineOption, Reference);

	struct ArgumentChecker;
	struct FunctionChecker;
	struct CallableChecker;

	// Names for the options. e.g --help or -h.
	PackedStringArray _names;
	// List of default values for each argument, empty if any value is allowed.
	PackedStringArray _default_args;
	// List of allowed values for each argument, empty if any value is allowed.
	PackedStringArray _allowed_args;
	// Option description that will be displayed in help.
	String _description;
	// Option category, options sharing the same category are grouped together in the help text.
	String _category;
	// Name for the option arguments that will be displayed in help text.
	String _arg_text = RTR("<arg>");
	// Make the option visible in help.
	bool _hidden = false;
	// If true, arguments can be specified without option name.
	bool _positional = false;
	// If true, argument always should be provided.
	bool _required = false;
	// If true, the option can be specified several times.
	bool _multitoken = false;
	// Arguments count required for the option, -1 for all arguments left.
	int _arg_count = 1;
	// Checker for each argument, nullptr if not check specified.
	ArgumentChecker *_checker = nullptr;

protected:
	static void _bind_methods();

public:
	using CheckFunction = bool (*)(const String &);

	void set_names(const PackedStringArray &p_names);
	PackedStringArray get_names() const;
	void set_default_args(const PackedStringArray &p_args);
	PackedStringArray get_default_args() const;
	void set_allowed_args(const PackedStringArray &p_args);
	PackedStringArray get_allowed_args() const;
	void set_description(const String &p_description);
	String get_description() const;
	void set_category(const String &p_category);
	String get_category() const;
	void set_arg_text(const String &p_arg_text);
	String get_arg_text() const;
	void set_arg_count(int p_count);
	int get_arg_count() const;
	void set_hidden(bool p_hidden);
	bool is_hidden() const;
	void set_positional(bool p_positional);
	bool is_positional() const;
	void set_required(bool p_required);
	bool is_required() const;
	void set_multitoken(bool p_multitoken);
	bool is_multitoken() const;

	void set_static_checker(CheckFunction p_function, const String &p_error_msg);
	Error set_checker(const Callable &p_callable, const String &p_error_msg);
	const ArgumentChecker *get_checker() const;
	void remove_checker();

	CommandLineOption() = default;
	explicit CommandLineOption(const PackedStringArray &p_names, int p_arg_count = 1);
	~CommandLineOption() override;
};

class CommandLineHelpFormat : public Reference {
	GDCLASS(CommandLineHelpFormat, Reference);

	String _help_header;
	String _help_footer;
	String _usage_title;

	int _left_help_pad = 2;
	int _right_help_pad = 4;
	int _help_line_length = 80;
	int _min_description_length = _help_line_length / 2;

	bool _autogenerate_usage = true;

protected:
	static void _bind_methods();

public:
	void set_header(const String &p_header);
	String get_header() const;
	void set_footer(const String &p_footer);
	String get_footer() const;
	void set_usage_title(const String &p_title);
	String get_usage_title() const;

	void set_left_pad(int p_size);
	int get_left_pad() const;
	void set_right_pad(int p_size);
	int get_right_pad() const;
	void set_line_length(int p_length);
	int get_line_length() const;
	void set_min_description_length(int p_length);
	int get_min_description_length() const;

	void set_autogenerate_usage(bool p_generate);
	bool is_autogenerate_usage() const;

	CommandLineHelpFormat() = default;
};

class CommandLineParser : public Reference {
	GDCLASS(CommandLineParser, Reference);

	struct ParsedPrefix;

	Vector<Ref<CommandLineOption>> _options;

	PackedStringArray _args;
	PackedStringArray _forwarded_args;

	PackedStringArray _long_prefixes = sarray("--");
	PackedStringArray _short_prefixes = sarray("-");

	Map<const CommandLineOption *, PackedStringArray> _parsed_values;
	Map<const CommandLineOption *, PackedStringArray> _parsed_prefixes;
	Map<const CommandLineOption *, int> _parsed_count;

	String _error;

	float _similarity_bias = 0.3;

	bool _allow_forwarding_args = false;
	bool _allow_adjacent = true;
	bool _allow_sticky = true;
	bool _allow_compound = true;

	// Parser main helpers
	bool _is_options_valid() const;
	void _read_default_args();
	int _validate_arguments(int p_current_idx); // Returns number of arguments taken, -1 on validation error.

	// Helpers for the function above that parse a specific case
	int _validate_positional(const String &p_arg, int p_current_idx);
	int _validate_adjacent(const String &p_arg, const String &p_prefix, int p_separator);
	int _validate_short(const String &p_arg, const String &p_prefix, int p_current_idx);
	int _validate_long(const String &p_arg, const String &p_prefix, int p_current_idx);

	// Validation helpers
	const CommandLineOption *_validate_option(const String &p_name, const String &p_prefix);
	int _validate_option_args(const CommandLineOption *p_option, const String &p_display_name, int p_current_idx, bool p_skip_first = false);
	bool _validate_option_arg(const CommandLineOption *p_option, const String &p_display_name, const String &p_arg);

	// Save information about parsed option
	void _save_parsed_option(const CommandLineOption *p_option, const String &p_prefix, int p_idx, int p_arg_count, const String &p_additional_value = String());
	void _save_parsed_option(const CommandLineOption *p_option, const String &p_prefix, const String &p_value = String());
	void _save_parsed_option(const CommandLineOption *p_option, int p_idx, int p_arg_count);

	// Help text printers
	String _get_usage(const Vector<Pair<const CommandLineOption *, String>> &p_printable_options, const String &p_title) const;
	String _get_options_description(const OrderedHashMap<String, PackedStringArray> &p_categories_data) const;

	// Other utilies
	String _to_string(const PackedStringArray &p_names) const; // Returns all option names separated by commas with all prefix variants.
	String _get_prefixed_longest_name(const PackedStringArray &p_names) const; // Returns longest name with first available prefix (long or short).
	ParsedPrefix _parse_prefix(const String &p_arg) const;
	String _find_most_similar(const String &p_name) const;
	static bool _contains_optional_options(const Vector<Pair<const CommandLineOption *, String>> &p_printable_options);

protected:
	static void _bind_methods();

public:
	Error parse_args(const PackedStringArray &p_args);
	Error validate();

	void add_option(const Ref<CommandLineOption> &p_option);
	int get_option_count() const;
	Ref<CommandLineOption> get_option(int p_idx) const;
	void set_option(int p_idx, const Ref<CommandLineOption> &p_option);
	void remove_option(int p_idx);
	Ref<CommandLineOption> find_option(const String &p_name) const;

	Ref<CommandLineOption> add_help_option();
	Ref<CommandLineOption> add_version_option();

	bool is_set(const Ref<CommandLineOption> &p_option) const;
	String get_value(const Ref<CommandLineOption> &p_option) const;
	PackedStringArray get_values(const Ref<CommandLineOption> &p_option) const;

	String get_prefix(const Ref<CommandLineOption> &p_option) const;
	PackedStringArray get_prefixes(const Ref<CommandLineOption> &p_option) const;

	int get_occurence_count(const Ref<CommandLineOption> &p_option) const;

	PackedStringArray get_forwarded_args() const;
	PackedStringArray get_args() const;
	String get_help_text(const Ref<CommandLineHelpFormat> &p_format) const;
	String get_error() const;
	void clear();

	void set_long_prefixes(const PackedStringArray &p_prefixes);
	PackedStringArray get_long_prefixes() const;
	void set_short_prefixes(const PackedStringArray &p_prefixes);
	PackedStringArray get_short_prefixes() const;
	void set_similarity_bias(float p_bias);
	float get_similarity_bias() const;
	void set_allow_forwarding_args(bool p_allow);
	bool is_allow_forwarding_args() const;
	void set_allow_adjacent(bool p_allow);
	bool is_allow_adjacent() const;
	void set_allow_sticky(bool p_allow);
	bool is_allow_sticky() const;
	void set_allow_compound(bool p_allow);
	bool is_allow_compound() const;

	CommandLineParser() = default;
};

#endif // COMMAND_LINE_PARSER_H
