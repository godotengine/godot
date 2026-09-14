/**************************************************************************/
/*  soft_body_capability.h                                                */
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

#include "core/string/string_name.h"
#include "core/templates/local_vector.h"
#include "core/templates/span.h"
#include "core/variant/variant.h"

#include <Jolt/Jolt.h>

#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/SoftBody/SoftBodyCreationSettings.h>
#include <Jolt/Physics/SoftBody/SoftBodySharedSettings.h>

class JoltSoftBodyCapState;
namespace JPH {
class TempAllocator;
}

// REPLACE supplies vertices; NONE maps mesh indices through mesh_to_physics.
// EXTEND is reserved.
enum class GeometryRole {
	NONE,
	EXTEND,
	REPLACE,
};

// Live reads require in_space() and disable storage. Stored keys remain readable
// without a live body.
struct CapProperty {
	const char *name;
	Variant::Type type;
	bool live_read;
	bool stored = true;
	bool clearable = false;
	bool rebuild_on_write = true;
};

struct CapabilitySpec {
	const char *prefix;

	// Only the required /config key activates a production capability.
	const char *required_key;

	Span<CapProperty> props;
	GeometryRole geometry;

	// Validate the full candidate before committing state or rebuilding.
	bool (*validate)(const JoltSoftBodyCapState &p_state, String *r_err);

	// Use total p_mass to match the vertex inverse masses written by _update_mass().
	void (*contribute)(const JoltSoftBodyCapState &p_state, JPH::SoftBodySharedSettings &r_settings, const LocalVector<int> &p_mesh_to_physics, float p_mass);

	// Physics-vertex pins (mInvMass = 0), shared by contribute and the re-pin
	// after _update_mass() overwrites all inverse masses.
	void (*pinned_vertices)(const JoltSoftBodyCapState &p_state, LocalVector<int> &r_indices);

	void (*apply_body)(const JoltSoftBodyCapState &p_state, JPH::SoftBodyCreationSettings &r_settings);
	void (*derive)(JPH::SoftBodySharedSettings &r_settings);

	// Dispatcher passes a body only for live reads, after checking in_space().
	// Stored keys receive nullptr and remain readable outside a space.
	Variant (*get)(const JoltSoftBodyCapState &p_state, const JPH::Body *p_body, const StringName &p_name);

	bool (*mesh_attributes)(const JoltSoftBodyCapState &, const LocalVector<int> &, LocalVector<JPH::SoftBodySharedSettings::VertexAttributes> &, JPH::SoftBodySharedSettings::EBendType &, String *) = nullptr;
	void (*mesh_pinned_vertices)(const JoltSoftBodyCapState &, const LocalVector<int> &, LocalVector<int> &) = nullptr;
	bool (*validate_build)(const JoltSoftBodyCapState &, const JPH::SoftBodySharedSettings &, const LocalVector<int> &, String *) = nullptr;
	bool (*pre_step)(const JoltSoftBodyCapState &, JPH::Body &, float, JPH::TempAllocator &, String *) = nullptr;
};
