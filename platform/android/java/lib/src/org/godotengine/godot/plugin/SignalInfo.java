/*************************************************************************/
/*  SignalInfo.java                                                      */
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

package org.godotengine.godot.plugin;

import android.text.TextUtils;

import androidx.annotation.NonNull;

import java.util.Arrays;

/**
 * Store information about a {@link GodotPlugin}'s signal.
 */
public final class SignalInfo {
	private final String name;
	private final Class<?>[] paramTypes;
	private final String[] paramTypesNames;

	public SignalInfo(@NonNull String signalName, Class<?>... paramTypes) {
		if (TextUtils.isEmpty(signalName)) {
			throw new IllegalArgumentException("Invalid signal name: " + signalName);
		}

		this.name = signalName;
		this.paramTypes = paramTypes == null ? new Class<?>[ 0 ] : paramTypes;
		this.paramTypesNames = new String[this.paramTypes.length];
		for (int i = 0; i < this.paramTypes.length; i++) {
			this.paramTypesNames[i] = this.paramTypes[i].getName();
		}
	}

	public String getName() {
		return name;
	}

	Class<?>[] getParamTypes() {
		return paramTypes;
	}

	String[] getParamTypesNames() {
		return paramTypesNames;
	}

	@Override
	public String toString() {
		return "SignalInfo{"
				+
				"name='" + name + '\'' +
				", paramsTypes=" + Arrays.toString(paramTypes) +
				'}';
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) {
			return true;
		}
		if (!(o instanceof SignalInfo)) {
			return false;
		}

		SignalInfo that = (SignalInfo)o;

		return name.equals(that.name);
	}

	@Override
	public int hashCode() {
		return name.hashCode();
	}
}
