/**************************************************************************/
/*  vertex_optimization_report.h                                         */
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

// Result of a Vertex optimization pass. Contains measured metrics and a list of
// recommendations, each tagged with a severity and an optional safe action.
class VertexOptimizationReport : public RefCounted {
	GDCLASS(VertexOptimizationReport, RefCounted)

public:
	enum Severity {
		SEVERITY_INFO,
		SEVERITY_WARNING,
		SEVERITY_CRITICAL,
	};

private:
	Dictionary metrics;
	Vector<Dictionary> recommendations;

public:
	void set_metrics(const Dictionary &p_metrics) { metrics = p_metrics; }
	Dictionary get_metrics() const { return metrics; }

	void add_recommendation(Severity p_severity, const String &p_message, const String &p_action = String(), const String &p_target = String());

	int get_recommendation_count() const { return recommendations.size(); }
	Dictionary get_recommendation(int p_index) const;
	Vector<Dictionary> get_recommendations() const { return recommendations; }

	int get_critical_count() const;
	int get_warning_count() const;

	Dictionary to_dictionary() const;

protected:
	static void _bind_methods();
};

VARIANT_ENUM_CAST(VertexOptimizationReport::Severity);
