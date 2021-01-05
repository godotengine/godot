/*************************************************************************/
/*  renderer_storage.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "renderer_storage.h"

RendererStorage *RendererStorage::base_singleton = nullptr;

void RendererStorage::Dependency::changed_notify(DependencyChangedNotification p_notification) {
	for (Map<DependencyTracker *, uint32_t>::Element *E = instances.front(); E; E = E->next()) {
		if (E->key()->changed_callback) {
			E->key()->changed_callback(p_notification, E->key());
		}
	}
}

void RendererStorage::Dependency::deleted_notify(const RID &p_rid) {
	for (Map<DependencyTracker *, uint32_t>::Element *E = instances.front(); E; E = E->next()) {
		if (E->key()->deleted_callback) {
			E->key()->deleted_callback(p_rid, E->key());
		}
	}
	for (Map<DependencyTracker *, uint32_t>::Element *E = instances.front(); E; E = E->next()) {
		E->key()->dependencies.erase(this);
	}
	instances.clear();
}

RendererStorage::Dependency::~Dependency() {
#ifdef DEBUG_ENABLED
	if (instances.size()) {
		WARN_PRINT("Leaked instance dependency: Bug - did not call instance_notify_deleted when freeing.");
		for (Map<DependencyTracker *, uint32_t>::Element *E = instances.front(); E; E = E->next()) {
			E->key()->dependencies.erase(this);
		}
	}
#endif
}

RendererStorage::RendererStorage() {
	base_singleton = this;
}
