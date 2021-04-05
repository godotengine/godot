/*************************************************************************/
/*  validation_tools.h                                                   */
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

#ifndef FBX_VALIDATION_TOOLS_H
#define FBX_VALIDATION_TOOLS_H

#ifdef TOOLS_ENABLED

#include "core/io/json.h"
#include "core/os/file_access.h"
#include "core/string/ustring.h"
#include "core/templates/local_vector.h"
#include "core/templates/map.h"

class ValidationTracker {
protected:
	struct Entries {
		Map<String, LocalVector<String>> validation_entries = Map<String, LocalVector<String>>();

		// for printing our CSV to dump validation problems of files
		// later we can make some agnostic tooling for this but this is fine for the time being.
		void add_validation_error(String asset_path, String message);
		void print_to_csv() {
			print_verbose("Exporting assset validation log please wait");
			String massive_log_file;

			String csv_header = "file_path, error message, extra data\n";
			massive_log_file += csv_header;

			for (Map<String, LocalVector<String>>::Element *element = validation_entries.front(); element; element = element->next()) {
				for (unsigned int x = 0; x < element->value().size(); x++) {
					const String &line_entry = element->key() + ", " + element->value()[x].c_escape() + "\n";
					massive_log_file += line_entry;
				}
			}

			String path = "asset_validation_errors.csv";
			Error err;
			FileAccess *file = FileAccess::open(path, FileAccess::WRITE, &err);
			if (!file || err) {
				if (file) {
					memdelete(file);
				}
				print_error("ValidationTracker Error - failed to create file - path: %s\n" + path);
				return;
			}

			file->store_string(massive_log_file);
			if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
				print_error("ValidationTracker Error - failed to write to file - path: %s\n" + path);
			}
			file->close();
			memdelete(file);
		}
	};
	// asset path, error messages
	static Entries *entries_singleton;

public:
	static Entries *get_singleton() {
		return entries_singleton;
	}
};

#endif // TOOLS_ENABLED
#endif // FBX_VALIDATION_TOOLS_H
