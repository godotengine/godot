/**************************************************************************/
/*  import_pipeline_step.h                                                */
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

#ifndef IMPORT_PIPELINE_STEP_H
#define IMPORT_PIPELINE_STEP_H

#include "core/error/error_macros.h"
#include "core/io/resource_importer.h"
#include "core/variant/dictionary.h"
#include "scene/resources/packed_scene.h"

class ImportPipelineStep : public RefCounted {
	GDCLASS(ImportPipelineStep, RefCounted);

	Ref<Resource> _source;
	String step_name;
	String category_name;

protected:
	GDVIRTUAL0(_update)
	GDVIRTUAL0(_source_changed)
	GDVIRTUAL0R(PackedStringArray, _get_inputs)
	GDVIRTUAL0R(PackedStringArray, _get_outputs)
	GDVIRTUAL0R(Node *, _get_tree)

	static void _bind_methods();

public:
	enum StepType {
		STEP_IMPORTER,
		STEP_LOADER,
		STEP_OVERWRITER,
		STEP_SAVER,
		STEP_DEFAULT,
	};

	String get_step_name() { return step_name; }
	void set_step_name(const String &p_step_name) { step_name = p_step_name; }
	String get_category_name() { return category_name; }
	void set_category_name(const String &p_category_name) { category_name = p_category_name; }

	virtual void update();
	virtual void source_changed();
	virtual PackedStringArray get_inputs();
	virtual PackedStringArray get_outputs();
	virtual Node *get_tree();

	ImportPipelineStep() {}
};

#endif // IMPORT_PIPELINE_STEP_H
