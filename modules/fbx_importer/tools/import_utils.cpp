/*************************************************************************/
/*  import_utils.cpp					                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "import_utils.h"
#include "data/pivot_transform.h"

Quat AssimpUtils::EulerToQuaternion(Assimp::FBX::Model::RotOrder mode, const Vector3 &p_rotation) {
	Vector3 rotation = p_rotation;
	// Returns a quaternion that will perform a rotation specified by Euler angles (in the YXZ convention: first Z, then X, and Y last), given in the vector format as (X-angle, Y-angle, Z-angle).
	// Godot uses ZXY convention

	if (Math::is_equal_approx(0, rotation.x)) {
		rotation.x = 0;
	}

	if (Math::is_equal_approx(0, rotation.y)) {
		rotation.y = 0;
	}

	if (Math::is_equal_approx(0, rotation.z)) {
		rotation.z = 0;
	}

	print_verbose("[euler->quat] rotation input: " + rotation);
	if(rotation.x == 0 && rotation.y == 0 && rotation.z == 0)
	{
		return Quat();
	}

	// we want to convert from rot order to ZXY
	const Quat x = Quat(Vector3(Math::deg2rad(rotation.x), 0, 0));
	const Quat y = Quat(Vector3(0, Math::deg2rad(rotation.y), 0));
	const Quat z = Quat(Vector3(0, 0, Math::deg2rad(rotation.z)));


	Quat result;
	// So we can theoretically convert calls
	switch (mode) {
		case Assimp::FBX::Model::RotOrder_EulerXYZ:
			//print_verbose("rot order: x y z");
			result = z * y * x;
			break;
		case Assimp::FBX::Model::RotOrder_EulerXZY:
			//print_verbose("rot order: x z y");
			result = y * z * x;
			break;
		case Assimp::FBX::Model::RotOrder_EulerYZX:
			//print_verbose("rot order: y z x");
			result = x * z * y;
			break;
		case Assimp::FBX::Model::RotOrder_EulerYXZ:
			//print_verbose("rot order: y x z");
			result = z * x * y;
			break;
		case Assimp::FBX::Model::RotOrder_EulerZXY:
			//print_verbose("rot order: z x y");
			result = y * x * z;
			break;
		case Assimp::FBX::Model::RotOrder_EulerZYX:
			//print_verbose("rot order: z x y");
			result = y * x * z;
			break;
		case Assimp::FBX::Model::RotOrder_SphericXYZ:
			//print_error("unsupported rotation order, defaulted to xyz");
			result = z * y * x;
			break;
		default:
			result = z * y * x;
			break;
	}
	//print_verbose("euler input data:" + rotation);
	print_verbose("euler to quaternion: " + (result.get_euler() * (180 / Math_PI)));

	return result;
}