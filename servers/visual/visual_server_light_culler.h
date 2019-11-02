/*************************************************************************/
/*  visual_server_light_culler.h                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef VISUALSERVERLIGHTCULLER_H
#define VISUALSERVERLIGHTCULLER_H

#include "core/math/plane.h"
#include "core/math/vector3.h"
#include "visual_server_scene.h"

struct CameraMatrix;
class Transform;

/*
For testing performance improvements from the LightCuller:
Uncomment LIGHT_CULLER_DEBUG_FLASH and it will turn the culler
on and off every LIGHT_CULLER_DEBUG_FLASH_FREQUENCY camera prepares.
Uncomment LIGHT_CULLER_DEBUG_LOGGING to get period print of the number of casters culled before / after.
*/

//#define LIGHT_CULLER_DEBUG_LOGGING
//#define LIGHT_CULLER_DEBUG_FLASH
#define LIGHT_CULLER_DEBUG_FLASH_FREQUENCY 256

// Culls shadow casters that can't cast shadows into the camera frustum.
class VisualServerLightCuller {
public:
	VisualServerLightCuller();

private:
	class LightSource {
	public:
		enum eSourceType {
			ST_UNKNOWN,
			ST_DIRECTIONAL,
			ST_SPOTLIGHT,
			ST_OMNI,
		};

		LightSource() {
			etype = ST_UNKNOWN;
			angle = 0.0f;
			range = FLT_MAX;
		}

		// all in world space, culling done in world space
		Vector3 pos;
		Vector3 dir;
		eSourceType etype;

		float angle; // for spotlight
		float range;
	};

	// same order as godot
	enum ePlane {
		P_NEAR,
		P_FAR,
		P_LEFT,
		P_TOP,
		P_RIGHT,
		P_BOTTOM,
		P_TOTAL,
	};

	// same order as godot
	enum ePoint {
		PT_FAR_LEFT_TOP,
		PT_FAR_LEFT_BOTTOM,
		PT_FAR_RIGHT_TOP,
		PT_FAR_RIGHT_BOTTOM,
		PT_NEAR_LEFT_TOP,
		PT_NEAR_LEFT_BOTTOM,
		PT_NEAR_RIGHT_TOP,
		PT_NEAR_RIGHT_BOTTOM,
	};

	// 6 bits, 6 planes
	enum { NUM_CAM_PLANES = 6,
		NUM_CAM_POINTS = 8,
		MAX_CULL_PLANES = 16,
		LUT_SIZE = 64,
	};

public:
	// before each pass with a different camera, you must call this so the culler can pre-create
	// the camera frustum planes and corner points in world space which are used for the culling
	bool prepare_camera(const Transform &p_cam_transform, const CameraMatrix &p_cam_matrix);

	// returns false if the entire light is culled (i.e. there is no intersection between the light and the view frustum)
	bool prepare_light(const VisualServerScene::Instance &p_instance);

	// cull according to the planes that were setup in the previous call to prepare_light
	int cull(int count, VisualServerScene::Instance **ppInstances);

	// can turn on and off from the engine if desired
	void set_active(bool p_active) { bactive = p_active; }

private:
	// internal version uses LightSource
	bool _add_light_camera_planes(const LightSource &LightSource);

	// directional light gives parallel culling planes (as opposed to point lights)
	bool add_light_camera_planes_directional(const LightSource &LightSource);

	// is the light culler active? maybe not in the editor
	bool is_active() const { return bactive; }

	// do we want to log some debug output?
	bool is_logging() const { return debug_count == 0; }

	// camera frustum planes (world space) - order ePlane
	Vector<Plane> frustum_planes;

	// camera frustum corners (world space) - order ePoint
	Vector3 frustum_points[NUM_CAM_POINTS];

	// culling planes
	void add_cull_plane(const Plane &p);

	// we are storing cull planes in a ye olde style array to prevent needless allocations
	Plane cull_planes[MAX_CULL_PLANES];
	int num_cull_planes;

	// the whole light can be out of range of the view frustum, in which case all casters should be culled
	bool out_of_range;

#ifdef LIGHT_CULLER_DEBUG_LOGGING
	String plane_bitfield_to_string(unsigned int BF);
	// names of the plane and point enums, useful for debugging
	static const char *string_planes[];
	static const char *string_points[];
#endif

	// precalculated LUT
	static uint8_t LUT_entry_sizes[LUT_SIZE];
	static uint8_t LUT_entries[LUT_SIZE][8];

	bool bactive;

	// ideally a frame counter, but for ease of implementation
	// this is just incremented on each prepare_camera.
	// used to turn on and off debugging features.
	int debug_count;
};

#endif // VISUALSERVERLIGHTCULLER_H
