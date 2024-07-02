/**************************************************************************/
/*  platform_text.h                                                       */
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

#ifndef PLATFORM_TEXT_H
#define PLATFORM_TEXT_H

#include "core/object/object.h"
#include "core/os/os.h"
#include "core/variant/binder_common.h"

enum class PlatformTextID {
};
VARIANT_ENUM_CAST(PlatformTextID);

enum class PlatformTextEditorID {
};
VARIANT_ENUM_CAST(PlatformTextEditorID);

class PlatformTextImplementation {
protected:
	bool _is_editor() const;

public:
	virtual String get_string(PlatformTextID p_id) const;
	virtual String get_editor_string(PlatformTextEditorID p_id) const;

	PlatformTextImplementation() {}
	virtual ~PlatformTextImplementation() {}
};

//////////////////////////

class PlatformText : public Object {
	GDCLASS(PlatformText, Object);

	static PlatformText *singleton;
	PlatformTextImplementation *implementation = nullptr;

protected:
	static void _bind_methods();

public:
	static PlatformText *get_singleton() { return singleton; }

	void set_implementation(PlatformTextImplementation *p_implementation) {
		implementation = p_implementation;
	}

	virtual String get_string(PlatformTextID p_id) const;
	virtual String get_editor_string(PlatformTextEditorID p_id) const;

	PlatformText() {
		singleton = this;
	}
	virtual ~PlatformText() {}
};

#endif // PLATFORM_TEXT_H
