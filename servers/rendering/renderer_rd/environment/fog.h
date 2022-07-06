/*************************************************************************/
/*  fog.h                                                                */
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

#ifndef FOG_RD_H
#define FOG_RD_H

#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"
#include "servers/rendering/environment/renderer_fog.h"
#include "servers/rendering/storage/utilities.h"

namespace RendererRD {

class Fog : public RendererFog {
public:
	struct FogVolume {
		RID material;
		Vector3 extents = Vector3(1, 1, 1);

		RS::FogVolumeShape shape = RS::FOG_VOLUME_SHAPE_BOX;

		Dependency dependency;
	};

private:
	static Fog *singleton;

	mutable RID_Owner<FogVolume, true> fog_volume_owner;

public:
	static Fog *get_singleton() { return singleton; }

	Fog();
	~Fog();

	/* FOG VOLUMES */

	FogVolume *get_fog_volume(RID p_rid) { return fog_volume_owner.get_or_null(p_rid); };
	bool owns_fog_volume(RID p_rid) { return fog_volume_owner.owns(p_rid); };

	virtual RID fog_volume_allocate() override;
	virtual void fog_volume_initialize(RID p_rid) override;
	virtual void fog_free(RID p_rid) override;

	virtual void fog_volume_set_shape(RID p_fog_volume, RS::FogVolumeShape p_shape) override;
	virtual void fog_volume_set_extents(RID p_fog_volume, const Vector3 &p_extents) override;
	virtual void fog_volume_set_material(RID p_fog_volume, RID p_material) override;
	virtual RS::FogVolumeShape fog_volume_get_shape(RID p_fog_volume) const override;
	RID fog_volume_get_material(RID p_fog_volume) const;
	virtual AABB fog_volume_get_aabb(RID p_fog_volume) const override;
	Vector3 fog_volume_get_extents(RID p_fog_volume) const;
};

} // namespace RendererRD

#endif // !FOG_RD_H
