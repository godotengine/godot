// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2021-2023 Arm Limited
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
// ----------------------------------------------------------------------------

/**
 * @brief Functions for the library entrypoint.
 */

#if defined(ASTCENC_DIAGNOSTICS)

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cmath>
#include <limits>
#include <string>

#include "astcenc_diagnostic_trace.h"

/** @brief The global trace logger. */
static TraceLog* g_TraceLog = nullptr;

/** @brief The JSON indentation level. */
static const size_t g_trace_indent = 2;

TraceLog::TraceLog(
	const char* file_name):
	m_file(file_name, std::ofstream::out | std::ofstream::binary)
{
	assert(!g_TraceLog);
	g_TraceLog = this;
	m_root = new TraceNode("root");
}

/* See header for documentation. */
TraceNode* TraceLog::get_current_leaf()
{
	if (m_stack.size())
	{
		return m_stack.back();
	}

	return nullptr;
}

/* See header for documentation. */
size_t TraceLog::get_depth()
{
	return m_stack.size();
}

/* See header for documentation. */
TraceLog::~TraceLog()
{
	assert(g_TraceLog == this);
	delete m_root;
	g_TraceLog = nullptr;
}

/* See header for documentation. */
TraceNode::TraceNode(
	const char* format,
	...
) {
	// Format the name string
	constexpr size_t bufsz = 256;
	char buffer[bufsz];

	va_list args;
	va_start (args, format);
	vsnprintf (buffer, bufsz, format, args);
	va_end (args);

	// Guarantee there is a nul terminator
	buffer[bufsz - 1] = 0;

	// Generate the node
	TraceNode* parent = g_TraceLog->get_current_leaf();
	size_t depth = g_TraceLog->get_depth();
	g_TraceLog->m_stack.push_back(this);

	bool comma = parent && parent->m_attrib_count;
	auto& out = g_TraceLog->m_file;

	if (parent)
	{
		parent->m_attrib_count++;
	}

	if (comma)
	{
		out << ',';
	}

	if (depth)
	{
		out << '\n';
	}

	size_t out_indent = (depth * 2) * g_trace_indent;
	size_t in_indent = (depth * 2 + 1) * g_trace_indent;

	std::string out_indents("");
	if (out_indent)
	{
		out_indents = std::string(out_indent, ' ');
	}

	std::string in_indents(in_indent, ' ');

	out << out_indents << "[ \"node\", \"" << buffer << "\",\n";
	out << in_indents << "[";
}

/* See header for documentation. */
void TraceNode::add_attrib(
	std::string type,
	std::string key,
	std::string value
) {
	(void)type;

	size_t depth = g_TraceLog->get_depth();
	size_t indent = (depth * 2) * g_trace_indent;
	auto& out = g_TraceLog->m_file;
	bool comma = m_attrib_count;
	m_attrib_count++;

	if (comma)
	{
		out << ',';
	}

	out << '\n';
	out << std::string(indent, ' ') << "[ "
	                                << "\"" << key << "\", "
	                                << value << " ]";
}

/* See header for documentation. */
TraceNode::~TraceNode()
{
	g_TraceLog->m_stack.pop_back();

	auto& out = g_TraceLog->m_file;
	size_t depth = g_TraceLog->get_depth();
	size_t out_indent = (depth * 2) * g_trace_indent;
	size_t in_indent = (depth * 2 + 1) * g_trace_indent;

	std::string out_indents("");
	if (out_indent)
	{
		out_indents = std::string(out_indent, ' ');
	}

	std::string in_indents(in_indent, ' ');

	if (m_attrib_count)
	{
		out << "\n" << in_indents;
	}
	out << "]\n";

	out << out_indents << "]";
}

/* See header for documentation. */
void trace_add_data(
	const char* key,
	const char* format,
	...
) {
	constexpr size_t bufsz = 256;
	char buffer[bufsz];

	va_list args;
	va_start (args, format);
	vsnprintf (buffer, bufsz, format, args);
	va_end (args);

	// Guarantee there is a nul terminator
	buffer[bufsz - 1] = 0;

	std::string value = "\"" + std::string(buffer) + "\"";

	TraceNode* node = g_TraceLog->get_current_leaf();
	node->add_attrib("str", key, value);
}

/* See header for documentation. */
void trace_add_data(
	const char* key,
	float value
) {
	// Turn infinities into parseable values
	if (std::isinf(value))
	{
		if (value > 0.0f)
		{
			value = std::numeric_limits<float>::max();
		}
		else
		{
			value = -std::numeric_limits<float>::max();
		}
	}

	char buffer[256];
	sprintf(buffer, "%.20g", (double)value);
	TraceNode* node = g_TraceLog->get_current_leaf();
	node->add_attrib("float", key, buffer);
}

/* See header for documentation. */
void trace_add_data(
	const char* key,
	int value
) {
	TraceNode* node = g_TraceLog->get_current_leaf();
	node->add_attrib("int", key, std::to_string(value));
}

/* See header for documentation. */
void trace_add_data(
	const char* key,
	unsigned int value
) {
	TraceNode* node = g_TraceLog->get_current_leaf();
	node->add_attrib("int", key, std::to_string(value));
}

#endif
