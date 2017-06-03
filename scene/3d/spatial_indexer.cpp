/*************************************************************************/
/*  spatial_indexer.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "spatial_indexer.h"

#if 0

#include "camera.h"
#include "proximity_area.h"
#include "scene/scene_string_names.h"

void SpatialIndexer::add_camera(Camera* p_camera) {

	cameras.insert(p_camera);
}

void SpatialIndexer::remove_camera(Camera* p_camera){

	for (Set<ProximityArea*>::Element *F=proximity_areas.front();F;F=F->next()) {

		ProximityArea *prox = F->get();
		TK<Camera> k;
		k.against=p_camera;
		k.area=prox;
		if (camera_pairs.has(k)) {
			camera_pairs.erase(k);
			prox->area_exit(ProximityArea::TRACK_CAMERAS,p_camera);
		}
	}
	cameras.erase(p_camera);

}

void SpatialIndexer::update_camera(Camera* p_camera) {


	_request_update();
}

void SpatialIndexer::_update_pairs() {

	// brute force interseciton code, no broadphase
	// will implement broadphase in the future

	for (Set<Camera*>::Element *E=cameras.front();E;E=E->next()) {

		Camera *cam = E->get();
		Vector<Plane> cplanes = cam->get_frustum();

		for (Set<ProximityArea*>::Element *F=proximity_areas.front();F;F=F->next()) {

			ProximityArea *prox = F->get();

			bool inters=false;

			if (prox->get_track_flag(ProximityArea::TRACK_CAMERAS)) {

				AABB aabb = prox->get_global_transform().xform(prox->get_aabb());
				if (aabb.intersects_convex_shape(cplanes.ptr(),cplanes.size()))
					inters=true;
			}

			TK<Camera> k;
			k.against=cam;
			k.area=prox;

			bool has = camera_pairs.has(k);

			if (inters==has)
				continue;

			if (inters) {
				camera_pairs.insert(k);
				prox->area_enter(ProximityArea::TRACK_CAMERAS,cam);
			} else {

				camera_pairs.erase(k);
				prox->area_exit(ProximityArea::TRACK_CAMERAS,cam);
			}
		}

	}

	pending_update=false;
}

void SpatialIndexer::_bind_methods() {


	ClassDB::bind_method(D_METHOD("_update_pairs"),&SpatialIndexer::_update_pairs);
}


void SpatialIndexer::add_proximity_area(ProximityArea* p_area) {

	proximity_areas.insert(p_area);

}

void SpatialIndexer::remove_proximity_area(ProximityArea* p_area) {

	for (Set<Camera*>::Element *E=cameras.front();E;E=E->next()) {

		Camera *cam = E->get();
		TK<Camera> k;
		k.against=cam;
		k.area=p_area;
		if (camera_pairs.has(k)) {
			camera_pairs.erase(k);
			p_area->area_exit(ProximityArea::TRACK_CAMERAS,cam);
		}
	}
	proximity_areas.erase(p_area);

}

void SpatialIndexer::_request_update() {

	if (pending_update)
		return;
	pending_update=true;
	call_deferred(SceneStringNames::get_singleton()->_update_pairs);

}

void SpatialIndexer::update_proximity_area_transform(ProximityArea* p_area) {

	_request_update();
}

void SpatialIndexer::update_proximity_area_flags(ProximityArea* p_area) {

	_request_update();
}

SpatialIndexer::SpatialIndexer() {

	pending_update=false;
}
#endif
