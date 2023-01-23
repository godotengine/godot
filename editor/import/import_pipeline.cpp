/**************************************************************************/
/*  import_pipeline.cpp                                                   */
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

#include "import_pipeline.h"

#include "core/error/error_macros.h"
#include "editor/editor_node.h"
#include "editor/import/import_pipeline_plugin.h"
#include "editor/import/import_pipeline_step.h"

void ImportPipeline::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_steps"), &ImportPipeline::get_steps);
	ClassDB::bind_method(D_METHOD("set_steps", "steps"), &ImportPipeline::set_steps);
	ClassDB::bind_method(D_METHOD("get_connections"), &ImportPipeline::get_connections);
	ClassDB::bind_method(D_METHOD("set_connections", "connections"), &ImportPipeline::set_connections);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "steps", PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_READ_ONLY | PROPERTY_USAGE_STORAGE), "set_steps", "get_steps");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "connections", PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_READ_ONLY | PROPERTY_USAGE_STORAGE), "set_connections", "get_connections");
}

Ref<Resource> ImportPipeline::execute(Ref<Resource> p_source, const String &p_path) {
	if (steps.size() == 0) {
		return p_source;
	}

	struct Source {
		int from;
		int port;
	};

	struct Result {
		Ref<ImportPipelineStep> step;
		bool valid;
		Vector<Ref<Resource>> resources;
		Vector<Source> sources;
	};

	Vector<Result> results;
	results.resize(steps.size());

	//Create the steps
	for (int i = 0; i < steps.size(); i++) {
		Dictionary step_data = steps[i];
		Result result;
		switch (ImportPipelineStep::StepType(int(step_data["type"]))) {
			case ImportPipelineStep::STEP_IMPORTER: {
				result.valid = true;
				result.resources.push_back(p_source);
				result.sources.resize(0);
			} break;
			case ImportPipelineStep::STEP_OVERWRITER: {
				result.valid = true;
				result.sources.resize(1);
			} break;
			case ImportPipelineStep::STEP_LOADER: {
				result.valid = true;
				result.resources.push_back(ResourceLoader::load(step_data["path"]));
				result.sources.resize(0);
			} break;
			case ImportPipelineStep::STEP_SAVER: {
				result.valid = true;
				result.sources.resize(1);
			} break;
			case ImportPipelineStep::STEP_DEFAULT: {
				Ref<ImportPipelineStep> step = ImportPipelinePlugins::get_singleton()->create_step(step_data["category"], step_data["name"]);
				if (step_data.has("script")) {
					step->set_script(step_data["script"]);
				}
				Dictionary parameters = step_data["parameters"];
				Array keys = parameters.keys();
				for (int j = 0; j < keys.size(); j++) {
					String key = keys[j];
					step->set(key, parameters[key]);
				}
				result.step = step;
				result.valid = false;
				Vector<Source> sources;
				sources.resize(step->get_inputs().size());
				for (int j = 0; j < sources.size(); j++) {
					Source source;
					source.from = -1;
					source.port = -1;
					sources.set(j, source);
				}
				result.sources = sources;
			} break;
		}
		results.set(i, result);
	}

	//Create the connections
	for (int i = 0; i < connections.size(); i++) {
		Dictionary connection = connections[i];
		int to = connection["to"];
		int to_port = connection["to_port"];
		Result res = results[to];
		Source source;
		source.from = connection["from"];
		source.port = connection["from_port"];
		res.sources.set(to_port, source);
		results.set(to, res);
	}

	//Calculate the results
	for (int iterations = 0; iterations < steps.size(); iterations++) {
		bool all_valid = true;
		for (int i = 0; i < steps.size(); i++) {
			if (results[i].valid) {
				continue;
			}
			bool sources_valid = true;
			Result res = results[i];
			for (int j = 0; sources_valid && j < res.sources.size(); j++) {
				int from = res.sources[j].from;
				sources_valid &= (from < 0) || results[res.sources[j].from].valid;
			}
			if (!sources_valid) {
				all_valid = false;
				continue;
			}
			Ref<ImportPipelineStep> step = res.step;
			PackedStringArray inputs = step->get_inputs();
			for (int j = 0; j < inputs.size(); j++) {
				Source source = res.sources[j];
				if (source.from < 0) {
					continue;
				}
				step->set(inputs[j], results[source.from].resources[source.port]);
			}
			step->source_changed();
			step->update();
			PackedStringArray outputs = step->get_outputs();
			res.resources.resize(outputs.size());
			for (int j = 0; j < outputs.size(); j++) {
				res.resources.set(j, step->get(outputs[j]));
			}
			res.valid = true;
			results.set(i, res);
		}
		if (all_valid) {
			goto cont;
		}
	}
	ERR_FAIL_V_MSG(p_source, "Pipeline contains cycles.");
cont:

	Ref<Resource> result;
	//Save the results
	for (int i = 0; i < steps.size(); i++) {
		Dictionary step_data = steps[i];
		switch (ImportPipelineStep::StepType(int(step_data["type"]))) {
			case ImportPipelineStep::STEP_IMPORTER: {
			} break;
			case ImportPipelineStep::STEP_OVERWRITER: {
				Source source = results[i].sources[0];
				if (source.from < 0) {
					result = p_source;
				} else {
					result = results[source.from].resources[source.port];
				}
			} break;
			case ImportPipelineStep::STEP_LOADER: {
			} break;
			case ImportPipelineStep::STEP_SAVER: {
				Source source = results[i].sources[0];
				if (source.from < 0) {
					continue;
				}
				Ref<Resource> resource = results[source.from].resources[source.port];
				if (resource.is_valid()) {
					String path = p_path.get_basename() + "-" + String(step_data["name"]) + ".res";
					resource->set_path(path, true);
					EditorNode::get_singleton()->save_resource(resource);
				}
			} break;
			case ImportPipelineStep::STEP_DEFAULT: {
			} break;
		}
	}

	return result;
}
