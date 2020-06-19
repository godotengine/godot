/*************************************************************************/
/*  pot_generator.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef POT_GENERATOR_H
#define POT_GENERATOR_H

#include "core/ordered_hash_map.h"
#include "core/set.h"
#include "modules/regex/regex.h"
#include "scene/resources/packed_scene.h"

class POTGenerator {
	static POTGenerator *singleton;
	// Stores all translatable strings and the source files containing them.
	OrderedHashMap<String, Set<String>> all_translation_strings;

	// Scene Node's properties that contain translation strings.
	Set<String> lookup_properties;

	// Regex and search patterns that are used to match translation strings in scripts.
	const String text = "((?:[^\"\\\\]|\\\\[\\s\\S])*(?:\"[\\s\\\\]*\\+[\\s\\\\]*\"(?:[^\"\\\\]|\\\\[\\s\\S])*)*)";
	RegEx regex;
	Vector<String> patterns;
	Vector<String> file_dialog_patterns;

	Vector<String> _parse_scene(const Ref<SceneState> &p_state);
	Vector<String> _parse_script(const String &p_source_code);
	void _parse_file_dialog(const String &p_source_code, Vector<String> *r_output);
	void _get_captured_strings(const Array &p_results, Vector<String> *r_output);
	void _write_to_pot(const String &p_file);

public:
	static POTGenerator *get_singleton();
	void generate_pot(const String &p_file);

	POTGenerator();
	~POTGenerator();
};

#endif // POT_GENERATOR_H
