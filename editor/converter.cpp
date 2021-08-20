/*************************************************************************/
/*  converter.cpp                                                        */
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

#ifndef CONVERTER_H
#define CONVERTER_H

#include "core/core_bind.h"
#include "core/io/file_access.h"
#include "core/string/print_string.cpp"
#include "core/string/ustring.h"

//#include <filesystem>
//#include <fstream>
// #include <string>

// ALL FILES - tscn, gd, tres

// Usage
// Checking only gd files
// Use class.USED_ENUM
// Use class.get(USED_ENUM) - TO CHECK THIS
static const char *enum_renames[][2] = {
	{ "BEFORE", "AFTER" }

};

static void rename_enums(String &file_content) {
	file_content = file_content.replace(enum_renames[0][0], enum_renames[0][1]);
	print_line("Starting converting enums.");
};
static void rename_classes() {
	print_line("Starting converting classes.");
};
static void rename_functions() {
	print_line("Starting converting functions.");
};
static void rename_properties() {
	print_line("Starting converting properties.");
};

// Collect files which will be checking, it will not touch txt, mp4, wav etc. files
static Vector<String> check_for_files() {
	Vector<String> collected_files = Vector<String>();

	Vector<String> directories_to_check = Vector<String>();
	directories_to_check.push_back("res://");

	core_bind::Directory dir = core_bind::Directory();
	while (!directories_to_check.is_empty()) {
		String path = directories_to_check.get(directories_to_check.size() - 1); // Is there any pop_back function?
		directories_to_check.resize(directories_to_check.size() - 1); // Remove last element
		if (dir.open(path) == OK) {
			dir.list_dir_begin();
			String current_dir = dir.get_current_dir();
			String file_name = dir.get_next();

			while (file_name != "") {
				// Node2D.gd is only for tests
				if (file_name == ".." || file_name == "." || file_name == "." || file_name == "Node2D.gd") {
					file_name = dir.get_next();
					continue;
				}
				if (dir.current_is_dir()) {
					directories_to_check.append(current_dir + file_name + "/");
				} else {
					bool proper_extension = false;
					// TODO only gd files ar checked
					if (file_name.ends_with(".gd"))
						proper_extension = true;

					if (proper_extension) {
						collected_files.append(current_dir + file_name);
					}
				}
				file_name = dir.get_next();
			}
		} else {
			print_verbose("Failed to open " + path);
		}
	}
	return collected_files;
}

static void converter() {
	print_line("Starting Converting.");

	//	FileAccess* project_file_to_check  = FileAccess::open("project.godot", FileAccess::ModeFlags::READ);
	if (!FileAccess::exists("project.godot")) {
		print_line("Current directory doesn't contains any Godot 3 project");
		return;
	}
	// TODO Add comment to project.godot, to save users from using this tool more than once on project
	// Probably comment at the begining of project.godot - # This project was updated to Godot 4.0, should be enough

	Vector<String> collected_files = check_for_files();
	{
		print_line("Collected Files:");
		for (int i = 0; i < collected_files.size(); i++) {
			print_line(collected_files[i]);
		}
	}

	// // Only can search files
	// if (std::filesystem::exists("project.godot")) {
	// 	// Iterate over all gd, tscn etc. files, probably better is to collect all file names into vector and then use them in loop
	// 	while (true) {
	// 		String name = "test.gd";

	// 		file_handler.open("test.gd");

	// 		if (file_handler.is_open()) {
	// 			String file_content = "";

	// 			print_line("Opened file test.gd");

	// 			while (file_handler) {
	// 				file_content += file_handler.get();
	// 			}

	// 			print_line(file_content);

	// 			if (name.ends_with(".gd")) {
	// 				rename_enums(file_content);
	// 			} else if (name.ends_with(".tscn")) {
	// 			}

	// 			file_handler.open("test.gd", std::ios::trunc | std::ios::out | std::ios::in);
	// 			file_handler.
	// 		} else {
	// 			print_line("Failed to open file " + name);
	// 		}

	// 		break;
	// 	}
	// }

	// print_line("Ending Converting.");
};

#endif // CONVERTER_H
