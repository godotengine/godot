/*************************************************************************/
/*  dotnet_solution.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "dotnet_solution.h"

#include "core/os/dir_access.h"
#include "core/os/file_access.h"

#include "../utils/path_utils.h"
#include "../utils/string_utils.h"
#include "csharp_project.h"

#define SOLUTION_TEMPLATE                                             \
	"Microsoft Visual Studio Solution File, Format Version 12.00\n"   \
	"# Visual Studio 2012\n"                                          \
	"%0\n"                                                            \
	"Global\n"                                                        \
	"\tGlobalSection(SolutionConfigurationPlatforms) = preSolution\n" \
	"%1\n"                                                            \
	"\tEndGlobalSection\n"                                            \
	"\tGlobalSection(ProjectConfigurationPlatforms) = postSolution\n" \
	"%2\n"                                                            \
	"\tEndGlobalSection\n"                                            \
	"EndGlobal\n"

#define PROJECT_DECLARATION "Project(\"{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}\") = \"%0\", \"%1\", \"{%2}\"\nEndProject"

#define SOLUTION_PLATFORMS_CONFIG "\t%0|Any CPU = %0|Any CPU"

#define PROJECT_PLATFORMS_CONFIG                   \
	"\t\t{%0}.%1|Any CPU.ActiveCfg = %1|Any CPU\n" \
	"\t\t{%0}.%1|Any CPU.Build.0 = %1|Any CPU"

void DotNetSolution::add_new_project(const String &p_name, const ProjectInfo &p_project_info) {
	projects[p_name] = p_project_info;
}

bool DotNetSolution::has_project(const String &p_name) const {
	return projects.find(p_name) != NULL;
}

const DotNetSolution::ProjectInfo &DotNetSolution::get_project_info(const String &p_name) const {
	return projects[p_name];
}

bool DotNetSolution::remove_project(const String &p_name) {
	return projects.erase(p_name);
}

Error DotNetSolution::save() {
	bool dir_exists = DirAccess::exists(path);
	ERR_EXPLAIN("The directory does not exist.");
	ERR_FAIL_COND_V(!dir_exists, ERR_FILE_NOT_FOUND);

	String projs_decl;
	String sln_platform_cfg;
	String proj_platform_cfg;

	for (Map<String, ProjectInfo>::Element *E = projects.front(); E; E = E->next()) {
		const String &name = E->key();
		const ProjectInfo &proj_info = E->value();

		bool is_front = E == projects.front();

		if (!is_front)
			projs_decl += "\n";

		projs_decl += sformat(PROJECT_DECLARATION, name, proj_info.relpath.replace("/", "\\"), proj_info.guid);

		for (int i = 0; i < proj_info.configs.size(); i++) {
			const String &config = proj_info.configs[i];

			if (i != 0 || !is_front) {
				sln_platform_cfg += "\n";
				proj_platform_cfg += "\n";
			}

			sln_platform_cfg += sformat(SOLUTION_PLATFORMS_CONFIG, config);
			proj_platform_cfg += sformat(PROJECT_PLATFORMS_CONFIG, proj_info.guid, config);
		}
	}

	String content = sformat(SOLUTION_TEMPLATE, projs_decl, sln_platform_cfg, proj_platform_cfg);

	FileAccess *file = FileAccess::open(path_join(path, name + ".sln"), FileAccess::WRITE);
	ERR_FAIL_NULL_V(file, ERR_FILE_CANT_WRITE);
	file->store_string(content);
	file->close();
	memdelete(file);

	return OK;
}

bool DotNetSolution::set_path(const String &p_existing_path) {
	if (p_existing_path.is_abs_path()) {
		path = p_existing_path;
	} else {
		String abspath;
		if (!rel_path_to_abs(p_existing_path, abspath))
			return false;
		path = abspath;
	}

	return true;
}

String DotNetSolution::get_path() {
	return path;
}

DotNetSolution::DotNetSolution(const String &p_name) {
	name = p_name;
}
