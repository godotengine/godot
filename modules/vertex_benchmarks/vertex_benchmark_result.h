/**************************************************************************/
/*  vertex_benchmark_result.h                                            */
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

#pragma once

#include "core/object/ref_counted.h"

// One benchmark result: per-workload samples and aggregates.
class VertexBenchmarkResult : public RefCounted {
	GDCLASS(VertexBenchmarkResult, RefCounted)

private:
	Dictionary samples; // workload -> Array of per-frame dictionaries.
	Dictionary aggregates; // workload -> aggregate dictionary.

public:
	void add_sample(const String &p_workload, const Dictionary &p_sample) {
		if (!samples.has(p_workload)) {
			samples[p_workload] = Array();
		}
		Array arr = samples[p_workload];
		arr.push_back(p_sample);
		samples[p_workload] = arr;
	}
	void set_aggregate(const String &p_workload, const Dictionary &p_agg) { aggregates[p_workload] = p_agg; }
	Dictionary get_aggregate(const String &p_workload) const { return aggregates.has(p_workload) ? Dictionary(aggregates[p_workload]) : Dictionary(); }
	Dictionary get_samples() const { return samples; }
	Dictionary get_aggregates() const { return aggregates; }

	Dictionary to_dictionary() const {
		Dictionary d;
		d["samples"] = samples;
		d["aggregates"] = aggregates;
		return d;
	}

protected:
	static void _bind_methods();
};
