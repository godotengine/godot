/**************************************************************************/
/*  import_pipeline.h                                                     */
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

#ifndef IMPORT_PIPELINE_H
#define IMPORT_PIPELINE_H

#include "core/error/error_macros.h"
#include "core/io/resource_importer.h"
#include "core/variant/dictionary.h"
#include "editor/import/import_pipeline_step.h"
#include "scene/resources/packed_scene.h"

class ImportPipeline : public Resource {
	GDCLASS(ImportPipeline, Resource);
	RES_BASE_EXTENSION("pipeline");

	TypedArray<Dictionary> steps;
	TypedArray<Dictionary> connections;

protected:
	static void _bind_methods();

public:
	TypedArray<Dictionary> get_steps() { return steps; }
	void set_steps(TypedArray<Dictionary> p_steps) { steps = p_steps; }
	TypedArray<Dictionary> get_connections() { return connections; }
	void set_connections(TypedArray<Dictionary> p_connections) { connections = p_connections; }

	Ref<Resource> execute(Ref<Resource> p_source, const String &p_path);

	ImportPipeline() {}
};

#endif // IMPORT_PIPELINE_H
