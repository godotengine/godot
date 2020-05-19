/*************************************************************************/
/*  rasterizer.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "rasterizer.h"

#include "core/os/os.h"
#include "core/print_string.h"

Rasterizer *(*Rasterizer::_create_func)() = nullptr;

void RasterizerScene::InstanceDependency::instance_notify_changed(bool p_aabb, bool p_dependencies) {
	for (Map<InstanceBase *, uint32_t>::Element *E = instances.front(); E; E = E->next()) {
		E->key()->dependency_changed(p_aabb, p_dependencies);
	}
}

void RasterizerScene::InstanceDependency::instance_notify_deleted(RID p_deleted) {
	for (Map<InstanceBase *, uint32_t>::Element *E = instances.front(); E; E = E->next()) {
		E->key()->dependency_deleted(p_deleted);
	}
	for (Map<InstanceBase *, uint32_t>::Element *E = instances.front(); E; E = E->next()) {
		E->key()->dependencies.erase(this);
	}

	instances.clear();
}

RasterizerScene::InstanceDependency::~InstanceDependency() {
#ifdef DEBUG_ENABLED
	if (instances.size()) {
		WARN_PRINT("Leaked instance dependency: Bug - did not call instance_notify_deleted when freeing.");
		for (Map<InstanceBase *, uint32_t>::Element *E = instances.front(); E; E = E->next()) {
			E->key()->dependencies.erase(this);
		}
	}
#endif
}

Rasterizer *Rasterizer::create() {
	return _create_func();
}

RasterizerCanvas *RasterizerCanvas::singleton = nullptr;

RasterizerStorage *RasterizerStorage::base_singleton = nullptr;

RasterizerStorage::RasterizerStorage() {
	base_singleton = this;
}
