/**************************************************************************/
/*  gdscript_preprocessor.cpp                                             */
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

#include "gdscript_preprocessor.h"

const StringName GDScriptPreprocessor::PREP_IF = "~if";
const StringName GDScriptPreprocessor::PREP_ENDIF = "~endif";

GDScriptPreprocessor::GDScriptPreprocessor() {
}

GDScriptPreprocessor::~GDScriptPreprocessor() {
}

GDScriptPreprocessor::ParserError GDScriptPreprocessor::read_source(const String &p_source_code, String &p_new_source_code) {
	ParserError error = { "" };

	Vector<String> lines = p_source_code.split("\n");
	int line = 1;
	data_if = LocalVector<DataIf>();
	data_endif = LocalVector<DataEndIf>();

	// Get all preprocessors data.
	for (int i = 0; i < lines.size(); i++) {
		String text = lines[i] + "\n";
		DataIf _if;
		DataEndIf _endif;

		_if.line = line;
		_endif.line = line;

		if (find_preprocessor_if(text, _if)) {
			data_if.push_back(_if);
		} else if (find_preprocessor_endif(text, _endif)) {
			data_endif.push_back(_endif);
		}

		line++;
	}

	// if (data_if.size() == 0 && data_endif.size() == 0) return p_source_code;
	// if (data_if.size() == 0 && data_endif.size() > 0) {
	//     parser.push_error("FIXME",0,0);
	//     return "";
	// }
	// if (data_if.size() > 0 && data_endif.size() == 0) {
	//     parser.push_error("FIXME", 0,0);
	//     return "";
	// }

	error = validate();
	if (error.message != "")
		return error;

	//TODO: get features

	// keep only code which has active features
	features.clear();
	features.push_back("DEBUG");
	features.push_back("TEST");

	struct SourceRemove {
		int line_a;
		int line_b;
		bool from_a_to_b;
	};

	LocalVector<SourceRemove> removes;

	for (const DataIf &if_data : data_if) {
		SourceRemove sr;
		sr.line_a = if_data.line;
		sr.line_b = if_data.matching_endif->line;
		if (is_active_feature(if_data.feature)) {
			sr.from_a_to_b = false;
		} else {
			sr.from_a_to_b = true;
		}
		removes.push_back(sr);
	}

	for (const SourceRemove &rm : removes) {
		print_line("=====");
		print_line("line_a: " + itos(rm.line_a));
		print_line("line_b: " + itos(rm.line_b));
		print_line("from_a_to_b: " + String(Variant(rm.from_a_to_b)));
		print_line("=====");
	}

	String source = "";
	line = 0;
	for (int index = 0; index < lines.size(); index++) {
		line++;
		String text = lines[index] + "\n";
		bool removed_line = false;
		for (unsigned int j = 0; j < removes.size(); j++) {
			if (removed_line)
				continue;
			const SourceRemove &sr = removes[j];

			if (sr.from_a_to_b == false) {
				if (line == sr.line_a) {
					source += "";
					removed_line = true;
					continue;
				}

				if (line == sr.line_b) {
					source += "";
					removed_line = true;
					continue;
				};

			} else {
				while (line >= sr.line_a && line <= sr.line_b) {
					source += "";
					removed_line = true;
					index++;
					line++;
					if (line > sr.line_b)
						break;
				}
			}
		}

		if (removed_line == false) {
			source += text;
		}
	}

	print_line("=============");
	print_line(source);
	print_line("=============");

	p_new_source_code = source;

	return error;
}

GDScriptPreprocessor::ParserError GDScriptPreprocessor::validate() {
	ParserError err = { "", 0, 0 };

	for (unsigned int i = 0; i < data_if.size(); i++) {
		DataIf *_if = &data_if[i];

		//Find first endif
		const DataEndIf *first_endif = nullptr;
		int first_endif_index = -1;

		for (const DataEndIf &_endif : data_endif) {
			first_endif_index++;
			if (_endif.line > _if->line) {
				first_endif = &_endif;
				break;
			}
		}

		if (first_endif == nullptr) {
			err.message = "No matching \"" + String(PREP_ENDIF) + "\" found for: " + String(PREP_IF) + " \"" + String(_if->feature.c_escape()) + "\".";
			err.line = _if->line;
			err.column = _if->column;
			return err;
		}

		// Get how many ~if after _if
		int found_ifs = 0;
		for (const DataIf &target_if : data_if) {
			if (target_if.line <= _if->line)
				continue;
			if (target_if.line > first_endif->line)
				break;

			found_ifs++;
		}
		int target_endif = found_ifs + first_endif_index;
		//Fill the structs for later checking
		DataEndIf *e = &data_endif[target_endif];
		_if->matching_endif = e;
		e->matching_if = _if;
	}

	print_line("DONE VALIDATING");

	return err;
}

bool GDScriptPreprocessor::match(const String &p_search, const String &p_target, int p_at_index) {
	int index = 0;

	while (index < p_search.length() && p_at_index < p_target.length()) {
		if (p_search[index] != p_target[p_at_index]) {
			return false;
		}
		index++;
		p_at_index++;
	}

	return true;
}

bool GDScriptPreprocessor::fast_check(const char32_t &p_first_letter, const String &p_text, int &p_index, int &p_ident_level, char &p_c) {
	bool found = false;
	// Fast check for "~" keyword.
	while (p_index < p_text.length()) {
		p_c = p_text[p_index];

		if (p_c == ' ' || p_c == '\t') {
			p_ident_level++;
		} else if (p_c == '~') {
			if (p_text[p_index + 1] == p_first_letter)
				found = true;
			break;
		} else
			break;

		p_index++;
	}
	return found;
}

bool GDScriptPreprocessor::check(const StringName &p_PREP, int &p_index, const String &p_text, char &p_c) {
	bool found = false;
	while (p_index < p_text.length()) {
		p_c = p_text[p_index];
		switch (p_c) {
			case '~':
				found = match(p_PREP, p_text, p_index);
				break;
		}
		p_index++;
	}

	return found;
}

bool GDScriptPreprocessor::find_preprocessor_if(const String &p_text, DataIf &p_data) {
	int index = 0;
	int ident_level = 0;
	char c = p_text[index];

	bool found = fast_check('i', p_text, index, ident_level, c);

	if (!found)
		return false;

	found = check(PREP_IF, index, p_text, c);

	p_data.ident_level = ident_level;
	p_data.column = ident_level + 4;
	p_data.feature = p_text.substr(p_data.column); //TODO: Improve because its getting comments and stuff
	p_data.feature = p_data.feature.strip_edges();
	return found;
}

bool GDScriptPreprocessor::find_preprocessor_endif(const String &p_text, DataEndIf &p_data) {
	int index = 0;
	int ident_level = 0;
	char c = p_text[index];
	bool found = fast_check('e', p_text, index, ident_level, c);

	if (!found)
		return false;

	found = check(PREP_ENDIF, index, p_text, c);

	return found;
}

bool GDScriptPreprocessor::is_active_feature(const String &p_feature) {
	for (const String &def_feature : features) {
		if (p_feature == def_feature) {
			print_line("\"" + p_feature.c_escape() + "\" is active");
			return true;
		}
		print_line("\"" + p_feature.c_escape() + "\" is not active");
	}
	return false;
}
