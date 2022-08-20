/*************************************************************************/
/*  Dictionary.java                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

package org.godotengine.godot;

import java.util.HashMap;
import java.util.Set;

public class Dictionary extends HashMap<String, Object> {
	protected String[] keys_cache;

	public String[] get_keys() {
		String[] ret = new String[size()];
		int i = 0;
		Set<String> keys = keySet();
		for (String key : keys) {
			ret[i] = key;
			i++;
		}

		return ret;
	}

	public Object[] get_values() {
		Object[] ret = new Object[size()];
		int i = 0;
		Set<String> keys = keySet();
		for (String key : keys) {
			ret[i] = get(key);
			i++;
		}

		return ret;
	}

	public void set_keys(String[] keys) {
		keys_cache = keys;
	}

	public void set_values(Object[] vals) {
		int i = 0;
		for (String key : keys_cache) {
			put(key, vals[i]);
			i++;
		}
		keys_cache = null;
	}
}
