/*************************************************************************/
/*  gltf_animation.h                                                     */
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

#ifndef GLTF_ANIMATION_H
#define GLTF_ANIMATION_H

#include "core/io/resource.h"

class GLTFAnimation : public Resource {
	GDCLASS(GLTFAnimation, Resource);

protected:
	static void _bind_methods();

public:
	enum Interpolation {
		INTERP_LINEAR,
		INTERP_STEP,
		INTERP_CATMULLROMSPLINE,
		INTERP_CUBIC_SPLINE,
	};

	template <class T>
	struct Channel {
		Interpolation interpolation;
		Vector<real_t> times;
		Vector<T> values;
	};

	struct Track {
		Channel<Vector3> position_track;
		Channel<Quaternion> rotation_track;
		Channel<Vector3> scale_track;
		Vector<Channel<real_t>> weight_tracks;
	};

public:
	bool get_loop() const;
	void set_loop(bool p_val);
	Map<int, GLTFAnimation::Track> &get_tracks();
	GLTFAnimation();

private:
	bool loop = false;
	Map<int, Track> tracks;
};
#endif // GLTF_ANIMATION_H
