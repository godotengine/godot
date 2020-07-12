/*************************************************************************/
/*  world_2d.cpp                                                         */
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

#include "world_2d.h"

#include "core/project_settings.h"
#include "scene/2d/camera_2d.h"
#include "scene/2d/visibility_notifier_2d.h"
#include "scene/main/window.h"
#include "servers/physics_server_2d.h"
#include "servers/rendering_server.h"

struct SpatialIndexer2D {
	struct CellRef {
		int ref;

		_FORCE_INLINE_ int inc() {
			ref++;
			return ref;
		}
		_FORCE_INLINE_ int dec() {
			ref--;
			return ref;
		}

		_FORCE_INLINE_ CellRef() {
			ref = 0;
		}
	};

	struct CellKey {
		union {
			struct {
				int32_t x;
				int32_t y;
			};
			uint64_t key;
		};

		bool operator==(const CellKey &p_key) const { return key == p_key.key; }
		_FORCE_INLINE_ bool operator<(const CellKey &p_key) const {
			return key < p_key.key;
		}
	};

	struct CellData {
		Map<VisibilityNotifier2D *, CellRef> notifiers;
	};

	Map<CellKey, CellData> cells;
	int cell_size;

	Map<VisibilityNotifier2D *, Rect2> notifiers;

	struct ViewportData {
		Map<VisibilityNotifier2D *, uint64_t> notifiers;
		Rect2 rect;
	};

	Map<Viewport *, ViewportData> viewports;

	bool changed;

	uint64_t pass;

	void _notifier_update_cells(VisibilityNotifier2D *p_notifier, const Rect2 &p_rect, bool p_add) {
		Point2i begin = p_rect.position;
		begin /= cell_size;
		Point2i end = p_rect.position + p_rect.size;
		end /= cell_size;
		for (int i = begin.x; i <= end.x; i++) {
			for (int j = begin.y; j <= end.y; j++) {
				CellKey ck;
				ck.x = i;
				ck.y = j;
				Map<CellKey, CellData>::Element *E = cells.find(ck);

				if (p_add) {
					if (!E) {
						E = cells.insert(ck, CellData());
					}
					E->get().notifiers[p_notifier].inc();
				} else {
					ERR_CONTINUE(!E);
					if (E->get().notifiers[p_notifier].dec() == 0) {
						E->get().notifiers.erase(p_notifier);
						if (E->get().notifiers.empty()) {
							cells.erase(E);
						}
					}
				}
			}
		}
	}

	void _notifier_add(VisibilityNotifier2D *p_notifier, const Rect2 &p_rect) {
		ERR_FAIL_COND(notifiers.has(p_notifier));
		notifiers[p_notifier] = p_rect;
		_notifier_update_cells(p_notifier, p_rect, true);
		changed = true;
	}

	void _notifier_update(VisibilityNotifier2D *p_notifier, const Rect2 &p_rect) {
		Map<VisibilityNotifier2D *, Rect2>::Element *E = notifiers.find(p_notifier);
		ERR_FAIL_COND(!E);
		if (E->get() == p_rect) {
			return;
		}

		_notifier_update_cells(p_notifier, p_rect, true);
		_notifier_update_cells(p_notifier, E->get(), false);
		E->get() = p_rect;
		changed = true;
	}

	void _notifier_remove(VisibilityNotifier2D *p_notifier) {
		Map<VisibilityNotifier2D *, Rect2>::Element *E = notifiers.find(p_notifier);
		ERR_FAIL_COND(!E);
		_notifier_update_cells(p_notifier, E->get(), false);
		notifiers.erase(p_notifier);

		List<Viewport *> removed;
		for (Map<Viewport *, ViewportData>::Element *F = viewports.front(); F; F = F->next()) {
			Map<VisibilityNotifier2D *, uint64_t>::Element *G = F->get().notifiers.find(p_notifier);

			if (G) {
				F->get().notifiers.erase(G);
				removed.push_back(F->key());
			}
		}

		while (!removed.empty()) {
			p_notifier->_exit_viewport(removed.front()->get());
			removed.pop_front();
		}

		changed = true;
	}

	void _add_viewport(Viewport *p_viewport, const Rect2 &p_rect) {
		ERR_FAIL_COND(viewports.has(p_viewport));
		ViewportData vd;
		vd.rect = p_rect;
		viewports[p_viewport] = vd;
		changed = true;
	}

	void _update_viewport(Viewport *p_viewport, const Rect2 &p_rect) {
		Map<Viewport *, ViewportData>::Element *E = viewports.find(p_viewport);
		ERR_FAIL_COND(!E);
		if (E->get().rect == p_rect) {
			return;
		}
		E->get().rect = p_rect;
		changed = true;
	}

	void _remove_viewport(Viewport *p_viewport) {
		ERR_FAIL_COND(!viewports.has(p_viewport));
		List<VisibilityNotifier2D *> removed;
		for (Map<VisibilityNotifier2D *, uint64_t>::Element *E = viewports[p_viewport].notifiers.front(); E; E = E->next()) {
			removed.push_back(E->key());
		}

		while (!removed.empty()) {
			removed.front()->get()->_exit_viewport(p_viewport);
			removed.pop_front();
		}

		viewports.erase(p_viewport);
	}

	void _update() {
		if (!changed) {
			return;
		}

		for (Map<Viewport *, ViewportData>::Element *E = viewports.front(); E; E = E->next()) {
			Point2i begin = E->get().rect.position;
			begin /= cell_size;
			Point2i end = E->get().rect.position + E->get().rect.size;
			end /= cell_size;
			pass++;
			List<VisibilityNotifier2D *> added;
			List<VisibilityNotifier2D *> removed;

			uint64_t visible_cells = (uint64_t)(end.x - begin.x) * (uint64_t)(end.y - begin.y);

			if (visible_cells > 10000) {
				//well you zoomed out a lot, it's your problem. To avoid freezing in the for loops below, we'll manually check cell by cell

				for (Map<CellKey, CellData>::Element *F = cells.front(); F; F = F->next()) {
					const CellKey &ck = F->key();

					if (ck.x < begin.x || ck.x > end.x) {
						continue;
					}
					if (ck.y < begin.y || ck.y > end.y) {
						continue;
					}

					//notifiers in cell
					for (Map<VisibilityNotifier2D *, CellRef>::Element *G = F->get().notifiers.front(); G; G = G->next()) {
						Map<VisibilityNotifier2D *, uint64_t>::Element *H = E->get().notifiers.find(G->key());
						if (!H) {
							H = E->get().notifiers.insert(G->key(), pass);
							added.push_back(G->key());
						} else {
							H->get() = pass;
						}
					}
				}

			} else {
				//check cells in grid fashion
				for (int i = begin.x; i <= end.x; i++) {
					for (int j = begin.y; j <= end.y; j++) {
						CellKey ck;
						ck.x = i;
						ck.y = j;

						Map<CellKey, CellData>::Element *F = cells.find(ck);
						if (!F) {
							continue;
						}

						//notifiers in cell
						for (Map<VisibilityNotifier2D *, CellRef>::Element *G = F->get().notifiers.front(); G; G = G->next()) {
							Map<VisibilityNotifier2D *, uint64_t>::Element *H = E->get().notifiers.find(G->key());
							if (!H) {
								H = E->get().notifiers.insert(G->key(), pass);
								added.push_back(G->key());
							} else {
								H->get() = pass;
							}
						}
					}
				}
			}

			for (Map<VisibilityNotifier2D *, uint64_t>::Element *F = E->get().notifiers.front(); F; F = F->next()) {
				if (F->get() != pass) {
					removed.push_back(F->key());
				}
			}

			while (!added.empty()) {
				added.front()->get()->_enter_viewport(E->key());
				added.pop_front();
			}

			while (!removed.empty()) {
				E->get().notifiers.erase(removed.front()->get());
				removed.front()->get()->_exit_viewport(E->key());
				removed.pop_front();
			}
		}

		changed = false;
	}

	SpatialIndexer2D() {
		pass = 0;
		changed = false;
		cell_size = GLOBAL_DEF("world/2d/cell_size", 100);
	}
};

void World2D::_register_viewport(Viewport *p_viewport, const Rect2 &p_rect) {
	indexer->_add_viewport(p_viewport, p_rect);
}

void World2D::_update_viewport(Viewport *p_viewport, const Rect2 &p_rect) {
	indexer->_update_viewport(p_viewport, p_rect);
}

void World2D::_remove_viewport(Viewport *p_viewport) {
	indexer->_remove_viewport(p_viewport);
}

void World2D::_register_notifier(VisibilityNotifier2D *p_notifier, const Rect2 &p_rect) {
	indexer->_notifier_add(p_notifier, p_rect);
}

void World2D::_update_notifier(VisibilityNotifier2D *p_notifier, const Rect2 &p_rect) {
	indexer->_notifier_update(p_notifier, p_rect);
}

void World2D::_remove_notifier(VisibilityNotifier2D *p_notifier) {
	indexer->_notifier_remove(p_notifier);
}

void World2D::_update() {
	indexer->_update();
}

RID World2D::get_canvas() {
	return canvas;
}

RID World2D::get_space() {
	return space;
}

void World2D::get_viewport_list(List<Viewport *> *r_viewports) {
	for (Map<Viewport *, SpatialIndexer2D::ViewportData>::Element *E = indexer->viewports.front(); E; E = E->next()) {
		r_viewports->push_back(E->key());
	}
}

void World2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_canvas"), &World2D::get_canvas);
	ClassDB::bind_method(D_METHOD("get_space"), &World2D::get_space);

	ClassDB::bind_method(D_METHOD("get_direct_space_state"), &World2D::get_direct_space_state);

	ADD_PROPERTY(PropertyInfo(Variant::_RID, "canvas", PROPERTY_HINT_NONE, "", 0), "", "get_canvas");
	ADD_PROPERTY(PropertyInfo(Variant::_RID, "space", PROPERTY_HINT_NONE, "", 0), "", "get_space");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "direct_space_state", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsDirectSpaceState2D", 0), "", "get_direct_space_state");
}

PhysicsDirectSpaceState2D *World2D::get_direct_space_state() {
	return PhysicsServer2D::get_singleton()->space_get_direct_state(space);
}

World2D::World2D() {
	canvas = RenderingServer::get_singleton()->canvas_create();
	space = PhysicsServer2D::get_singleton()->space_create();

	//set space2D to be more friendly with pixels than meters, by adjusting some constants
	PhysicsServer2D::get_singleton()->space_set_active(space, true);
	PhysicsServer2D::get_singleton()->area_set_param(space, PhysicsServer2D::AREA_PARAM_GRAVITY, GLOBAL_DEF("physics/2d/default_gravity", 98));
	PhysicsServer2D::get_singleton()->area_set_param(space, PhysicsServer2D::AREA_PARAM_GRAVITY_VECTOR, GLOBAL_DEF("physics/2d/default_gravity_vector", Vector2(0, 1)));
	PhysicsServer2D::get_singleton()->area_set_param(space, PhysicsServer2D::AREA_PARAM_LINEAR_DAMP, GLOBAL_DEF("physics/2d/default_linear_damp", 0.1));
	ProjectSettings::get_singleton()->set_custom_property_info("physics/2d/default_linear_damp", PropertyInfo(Variant::FLOAT, "physics/2d/default_linear_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"));
	PhysicsServer2D::get_singleton()->area_set_param(space, PhysicsServer2D::AREA_PARAM_ANGULAR_DAMP, GLOBAL_DEF("physics/2d/default_angular_damp", 1.0));
	ProjectSettings::get_singleton()->set_custom_property_info("physics/2d/default_angular_damp", PropertyInfo(Variant::FLOAT, "physics/2d/default_angular_damp", PROPERTY_HINT_RANGE, "-1,100,0.001,or_greater"));
	indexer = memnew(SpatialIndexer2D);
}

World2D::~World2D() {
	RenderingServer::get_singleton()->free(canvas);
	PhysicsServer2D::get_singleton()->free(space);
	memdelete(indexer);
}
