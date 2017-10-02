/*************************************************************************/
/*  net_solution.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "net_solution.h"

#include "os/dir_access.h"
#include "os/file_access.h"

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

#define SOLUTION_PLATFORMS_CONFIG "\t\%0|Any CPU = %0|Any CPU"

#define PROJECT_PLATFORMS_CONFIG                   \
	"\t\t{%0}.%1|Any CPU.ActiveCfg = %1|Any CPU\n" \
	"\t\t{%0}.%1|Any CPU.Build.0 = %1|Any CPU"

void NETSolution::add_new_project(const String &p_name, const String &p_guid, const Vector<String> &p_extra_configs) {
	if (projects.has(p_name))
		WARN_PRINT("Overriding existing project.");

	ProjectInfo procinfo;
	procinfo.guid = p_guid;

	procinfo.configs.push_back("Debug");
	procinfo.configs.push_back("Release");

	for (int i = 0; i < p_extra_configs.size(); i++) {
		procinfo.configs.push_back(p_extra_configs[i]);
	}

	projects[p_name] = procinfo;
}

Error NETSolution::save() {
	bool dir_exists = DirAccess::exists(path);
	ERR_EXPLAIN("The directory does not exist.");
	ERR_FAIL_COND_V(!dir_exists, ERR_FILE_BAD_PATH);

	String projs_decl;
	String sln_platform_cfg;
	String proj_platform_cfg;

	for (Map<String, ProjectInfo>::Element *E = projects.front(); E; E = E->next()) {
		const String &name = E->key();
		const ProjectInfo &procinfo = E->value();

		projs_decl += sformat(PROJECT_DECLARATION, name, name + ".csproj", procinfo.guid);

		for (int i = 0; i < procinfo.configs.size(); i++) {
			const String &config = procinfo.configs[i];

			if (i != 0) {
				sln_platform_cfg += "\n";
				proj_platform_cfg += "\n";
			}

			sln_platform_cfg += sformat(SOLUTION_PLATFORMS_CONFIG, config);
			proj_platform_cfg += sformat(PROJECT_PLATFORMS_CONFIG, procinfo.guid, config);
		}
	}

	String content = sformat(SOLUTION_TEMPLATE, projs_decl, sln_platform_cfg, proj_platform_cfg);

	FileAccessRef file = FileAccess::open(path_join(path, name + ".sln"), FileAccess::WRITE);
	ERR_FAIL_COND_V(!file, ERR_FILE_CANT_WRITE);
	file->store_string(content);
	file->close();

	return OK;
}

bool NETSolution::set_path(const String &p_existing_path) {
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

NETSolution::NETSolution(const String &p_name) {
	name = p_name;
}
