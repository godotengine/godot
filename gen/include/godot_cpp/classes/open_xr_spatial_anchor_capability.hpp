/**************************************************************************/
/*  open_xr_spatial_anchor_capability.hpp                                 */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/open_xr_extension_wrapper.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/rid.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class OpenXRAnchorTracker;
class OpenXRFutureResult;
struct Transform3D;

class OpenXRSpatialAnchorCapability : public OpenXRExtensionWrapper {
	GDEXTENSION_CLASS(OpenXRSpatialAnchorCapability, OpenXRExtensionWrapper)

public:
	enum PersistenceScope {
		PERSISTENCE_SCOPE_SYSTEM_MANAGED = 1,
		PERSISTENCE_SCOPE_LOCAL_ANCHORS = 1000781000,
	};

	bool is_spatial_anchor_supported();
	bool is_spatial_persistence_supported();
	bool is_persistence_scope_supported(OpenXRSpatialAnchorCapability::PersistenceScope p_scope);
	Ref<OpenXRFutureResult> create_persistence_context(OpenXRSpatialAnchorCapability::PersistenceScope p_scope, const Callable &p_user_callback = Callable());
	uint64_t get_persistence_context_handle(const RID &p_persistence_context) const;
	void free_persistence_context(const RID &p_persistence_context);
	Ref<OpenXRAnchorTracker> create_new_anchor(const Transform3D &p_transform, const RID &p_spatial_context = RID());
	void remove_anchor(const Ref<OpenXRAnchorTracker> &p_anchor_tracker);
	Ref<OpenXRFutureResult> persist_anchor(const Ref<OpenXRAnchorTracker> &p_anchor_tracker, const RID &p_persistence_context = RID(), const Callable &p_user_callback = Callable());
	Ref<OpenXRFutureResult> unpersist_anchor(const Ref<OpenXRAnchorTracker> &p_anchor_tracker, const RID &p_persistence_context = RID(), const Callable &p_user_callback = Callable());

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		OpenXRExtensionWrapper::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(OpenXRSpatialAnchorCapability::PersistenceScope);

