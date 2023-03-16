// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2021-2022 Arm Limited
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
 * @brief This module provides a set of diagnostic tracing utilities.
 *
 * Overview
 * ========
 *
 * The built-in diagnostic trace tool generates a hierarchical JSON tree structure. The tree
 * hierarchy contains three levels:
 *
 *    - block
 *        - pass
 *           - candidate
 *
 * One block node exists for each compressed block in the image. One pass node exists for each major
 * pass (N partition, M planes, O components) applied to a block. One candidate node exists for each
 * encoding candidate trialed for a pass.
 *
 * Each node contains both the hierarchy but also a number of attributes which explain the behavior.
 * For example, the block node contains the block coordinates in the image, the pass explains the
 * pass configuration, and the candidate will explain the candidate encoding such as weight
 * decimation, refinement error, etc.
 *
 * Trace Nodes are designed as scope-managed C++ objects with stack-like push/pop behavior.
 * Constructing a trace node on the stack will automatically add it to the current node as a child,
 * and then make it the current node. Destroying the current node will pop the stack and set the
 * parent to the current node. This provides a robust mechanism for ensuring reliable nesting in the
 * tree structure.
 *
 * A set of utility macros are provided to add attribute annotations to the current trace node.
 *
 * Usage
 * =====
 *
 * Create Trace Nodes on the stack using the @c TRACE_NODE() macro. This will compile-out completely
 * in builds with diagnostics disabled.
 *
 * Add annotations to the current trace node using the @c trace_add_data() macro. This will
 * similarly compile out completely in builds with diagnostics disabled.
 *
 * If you need to add additional code to support diagnostics-only behavior wrap
 * it in preprocessor guards:
 *
 *     #if defined(ASTCENC_DIAGNOSTICS)
 *     #endif
 */

#ifndef ASTCENC_DIAGNOSTIC_TRACE_INCLUDED
#define ASTCENC_DIAGNOSTIC_TRACE_INCLUDED

#if defined(ASTCENC_DIAGNOSTICS)

#include <iostream>
#include <fstream>
#include <vector>

/**
 * @brief Class representing a single node in the trace hierarchy.
 */
class TraceNode
{
public:
	/**
	 * @brief Construct a new node.
	 *
	 * Constructing a node will push to the the top of the stack, automatically making it a child of
	 * the current node, and then setting it to become the current node.
	 *
	 * @param format   The format template for the node name.
	 * @param ...      The format parameters.
	 */
	TraceNode(const char* format, ...);

	/**
	 * @brief Add an attribute to this node.
	 *
	 * Note that no quoting is applied to the @c value, so if quoting is needed it must be done by
	 * the caller.
	 *
	 * @param type    The type of the attribute.
	 * @param key     The key of the attribute.
	 * @param value   The value of the attribute.
	 */
	void add_attrib(std::string type, std::string key, std::string value);

	/**
	 * @brief Destroy this node.
	 *
	 * Destroying a node will pop it from the top of the stack, making its parent the current node.
	 * It is invalid behavior to destroy a node that is not the current node; usage must conform to
	 * stack push-pop semantics.
	 */
	~TraceNode();

	/**
	 * @brief The number of attributes and child nodes in this node.
	 */
	unsigned int m_attrib_count { 0 };
};

/**
 * @brief Class representing the trace log file being written.
 */
class TraceLog
{
public:
	/**
	 * @brief Create a new trace log.
	 *
	 * The trace log is global; there can be only one at a time.
	 *
	 * @param file_name   The name of the file to write.
	 */
	TraceLog(const char* file_name);

	/**
	 * @brief Detroy the trace log.
	 *
	 * Trace logs MUST be cleanly destroyed to ensure the file gets written.
	 */
	~TraceLog();

	/**
	 * @brief Get the current child node.
	 *
	 * @return The current leaf node.
	 */
	TraceNode* get_current_leaf();

	/**
	 * @brief Get the stack depth of the current child node.
	 *
	 * @return The current leaf node stack depth.
	 */
	size_t get_depth();

	/**
	 * @brief The file stream to write to.
	 */
	std::ofstream m_file;

	/**
	 * @brief The stack of nodes (newest at the back).
	 */
	std::vector<TraceNode*> m_stack;

private:
	/**
	 * @brief The root node in the JSON file.
	 */
	TraceNode* m_root;
};

/**
 * @brief Utility macro to create a trace node on the stack.
 *
 * @param name     The variable name to use.
 * @param ...      The name template and format parameters.
 */
#define TRACE_NODE(name, ...) TraceNode name(__VA_ARGS__);

/**
 * @brief Add a string annotation to the current node.
 *
 * @param key      The name of the attribute.
 * @param format   The format template for the attribute value.
 * @param ...      The format parameters.
 */
void trace_add_data(const char* key, const char* format, ...);

/**
 * @brief Add a float annotation to the current node.
 *
 * @param key     The name of the attribute.
 * @param value   The value of the attribute.
 */
void trace_add_data(const char* key, float value);

/**
 * @brief Add an integer annotation to the current node.
 *
 * @param key     The name of the attribute.
 * @param value   The value of the attribute.
 */
void trace_add_data(const char* key, int value);

/**
 * @brief Add an unsigned integer annotation to the current node.
 *
 * @param key     The name of the attribute.
 * @param value   The value of the attribute.
 */
void trace_add_data(const char* key, unsigned int value);

#else

#define TRACE_NODE(name, ...)

#define trace_add_data(...)

#endif

#endif
