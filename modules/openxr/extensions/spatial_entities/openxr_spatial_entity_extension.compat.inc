/**************************************************************************/
/*  openxr_spatial_entity_extension.compat.inc                            */
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

#ifndef DISABLE_DEPRECATED

#include "openxr_spatial_entity_extension.h"

#include "core/object/class_db.h"

void OpenXRSpatialEntityExtension::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("create_spatial_context", "capability_configurations", "next", "user_callback"), &OpenXRSpatialEntityExtension::_create_spatial_context_bind_compat_121123, DEFVAL(Variant()), DEFVAL(Callable()));
}

Ref<OpenXRFutureResult> OpenXRSpatialEntityExtension::_create_spatial_context_bind_compat_121123(const TypedArray<OpenXRSpatialCapabilityConfigurationBaseHeader> &p_capability_configurations, Ref<OpenXRStructureBase> p_next, const Callable &p_user_callback) {
	return create_spatial_context(p_capability_configurations, p_next, p_user_callback, Callable());
}

#endif // DISABLE_DEPRECATED
