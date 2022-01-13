/*************************************************************************/
/*  octree.h                                                             */
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

#ifndef OCTREE_H
#define OCTREE_H

#define OCTREE_ELEMENT_INVALID_ID 0
#define OCTREE_SIZE_LIMIT 1e15
#define OCTREE_DEFAULT_OCTANT_LIMIT 0

// We want 2 versions of the octree, Octree
// and Octree_CL which uses cached lists (optimized).
// we don't want to use the extra memory of cached lists on
// the non cached list version, so we use macros
// to avoid duplicating the code which is in octree_definition.
// The name of the class is overridden and the changes with the define
// OCTREE_USE_CACHED_LISTS.

// The two classes can be used identically but one contains the cached
// list optimization.

// standard octree
#define OCTREE_CLASS_NAME Octree
#undef OCTREE_USE_CACHED_LISTS
#include "octree_definition.inc"
#undef OCTREE_CLASS_NAME

// cached lists octree
#define OCTREE_CLASS_NAME Octree_CL
#define OCTREE_USE_CACHED_LISTS
#include "octree_definition.inc"
#undef OCTREE_CLASS_NAME

#endif // OCTREE_H
