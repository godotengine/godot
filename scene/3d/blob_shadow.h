/**************************************************************************/
/*  blob_shadow.h                                                         */
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

#ifndef BLOB_SHADOW_H
#define BLOB_SHADOW_H

#include "scene/3d/spatial.h"

class BlobShadow : public Spatial {
	GDCLASS(BlobShadow, Spatial);

public:
	enum BlobShadowType {
		BLOB_SHADOW_SPHERE,
		BLOB_SHADOW_CAPSULE,
	};

private:
	struct Data {
		BlobShadowType type = BLOB_SHADOW_SPHERE;

		RID blob;
		RID capsule;

		// Radius of sphere / capsules.
		real_t radius[2];

		// Offset to second sphere and second sphere radius
		// when using a capsule.
		Vector3 offset;

		Vector3 prev_pos;
		Transform prev_xform;

		Data() {
			radius[0] = 1;
			radius[1] = 1;
		}
	} data;

	void _refresh_visibility(bool p_in_tree);
	void _update_server(bool p_force_update);

protected:
	static void _bind_methods();
	void _notification(int p_what);
	void _validate_property(PropertyInfo &property) const;
	virtual void fti_update_servers_xform();
	virtual void _physics_interpolated_changed();

public:
	void set_radius(int p_index, real_t p_radius);
	real_t get_radius(int p_index = 0) const;

	void set_offset(const Vector3 &p_offset);
	Vector3 get_offset() const { return data.offset; }

	void set_shadow_type(BlobShadowType p_type);
	BlobShadowType get_shadow_type() const { return data.type; }

	BlobShadow();
	~BlobShadow();
};

VARIANT_ENUM_CAST(BlobShadow::BlobShadowType);

#endif // BLOB_SHADOW_H
