/**************************************************************************/
/*  light_storage.cpp                                                     */
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

#include "light_storage.h"

using namespace RendererDummy;

LightStorage *LightStorage::singleton = nullptr;

LightStorage *LightStorage::get_singleton() {
	return singleton;
}

LightStorage::LightStorage() {
	singleton = this;
}

LightStorage::~LightStorage() {
	singleton = nullptr;
}

bool LightStorage::free(RID p_rid) {
	if (owns_lightmap(p_rid)) {
		lightmap_free(p_rid);
		return true;
	} else if (owns_lightmap_instance(p_rid)) {
		lightmap_instance_free(p_rid);
		return true;
	}

	return false;
}

/* LIGHTMAP API */

RID LightStorage::lightmap_allocate() {
	return lightmap_owner.allocate_rid();
}

void LightStorage::lightmap_initialize(RID p_lightmap) {
	lightmap_owner.initialize_rid(p_lightmap, Lightmap());
}

void LightStorage::lightmap_free(RID p_rid) {
	lightmap_set_textures(p_rid, RID(), false);
	lightmap_owner.free(p_rid);
}

/* LIGHTMAP INSTANCE */

RID LightStorage::lightmap_instance_create(RID p_lightmap) {
	LightmapInstance li;
	li.lightmap = p_lightmap;
	return lightmap_instance_owner.make_rid(li);
}

void LightStorage::lightmap_instance_free(RID p_lightmap) {
	lightmap_instance_owner.free(p_lightmap);
}
