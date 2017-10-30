/*************************************************************************/
/*  path_utils.cpp                                                       */
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
#include "path_utils.h"

#include "os/dir_access.h"
#include "os/file_access.h"
#include "os/os.h"
#include "project_settings.h"

#ifdef WINDOWS_ENABLED
#define ENV_PATH_SEP ";"
#else
#define ENV_PATH_SEP ":"
#include <limits.h>
#endif

#include <stdlib.h>

String path_which(const String &p_name) {

#ifdef WINDOWS_ENABLED
	Vector<String> exts = OS::get_singleton()->get_environment("PATHEXT").split(ENV_PATH_SEP, false);
#endif
	Vector<String> env_path = OS::get_singleton()->get_environment("PATH").split(ENV_PATH_SEP, false);

	if (env_path.empty())
		return String();

	for (int i = 0; i < env_path.size(); i++) {
		String p = path_join(env_path[i], p_name);

#ifdef WINDOWS_ENABLED
		for (int j = 0; j < exts.size(); j++) {
			String p2 = p + exts[j];

			if (FileAccess::exists(p2))
				return p2;
		}
#else
		if (FileAccess::exists(p))
			return p;
#endif
	}

	return String();
}

void fix_path(const String &p_path, String &r_out) {
	r_out = p_path.replace("\\", "/");

	while (true) { // in case of using 2 or more slash
		String compare = r_out.replace("//", "/");
		if (r_out == compare)
			break;
		else
			r_out = compare;
	}
}

bool rel_path_to_abs(const String &p_existing_path, String &r_abs_path) {
#ifdef WINDOWS_ENABLED
	CharType ret[_MAX_PATH];
	if (_wfullpath(ret, p_existing_path.c_str(), _MAX_PATH)) {
		String abspath = String(ret).replace("\\", "/");
		int pos = abspath.find(":/");
		if (pos != -1) {
			r_abs_path = abspath.substr(pos - 1, abspath.length());
		} else {
			r_abs_path = abspath;
		}
		return true;
	}
#else
	char ret[PATH_MAX];
	if (realpath(p_existing_path.utf8().get_data(), ret)) {
		String retstr;
		if (!retstr.parse_utf8(ret)) {
			r_abs_path = retstr;
			return true;
		}
	}
#endif
	return false;
}
