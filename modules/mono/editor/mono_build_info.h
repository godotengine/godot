/*************************************************************************/
/*  mono_build_info.h                                                    */
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
#ifndef MONO_BUILD_INFO_H
#define MONO_BUILD_INFO_H

#include "../mono_gd/gd_mono_utils.h"

struct MonoBuildInfo {

	struct Hasher {
		static _FORCE_INLINE_ uint32_t hash(const MonoBuildInfo &p_key) {
			uint32_t hash = 0;

			GDMonoUtils::hash_combine(hash, p_key.solution.hash());
			GDMonoUtils::hash_combine(hash, p_key.configuration.hash());

			return hash;
		}
	};

	String solution;
	String configuration;
	Vector<String> custom_props;

	MonoBuildInfo() {}

	MonoBuildInfo(const String &p_solution, const String &p_config) {
		solution = p_solution;
		configuration = p_config;
	}

	bool operator==(const MonoBuildInfo &p_b) const {
		return p_b.solution == solution && p_b.configuration == configuration;
	}
};

#endif // MONO_BUILD_INFO_H
