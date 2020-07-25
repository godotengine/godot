/*************************************************************************/
/*  test_fbx.cpp                                                         */
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

#include "core/math/random_number_generator.h"
#include "core/os/os.h"
#include "core/ustring.h"
#include <stdio.h>
#include <wchar.h>

#include "test_fbx.h"

#include <modules/fbx/editor_scene_importer_fbx.h>

namespace TestFBX {

bool test_rotation(Vector3 deg_original_euler, Assimp::FBX::Model::RotOrder rot_order) {
	// This test:
	// 1. Converts the rotation vector from deg to rad.
	// 2. Converts euler to basis.
	// 3. Converts the above basis back into euler.
	// 4. Converts the above euler into basis again.
	// 5. Compares the basis obtained in step 2 with the basis of step 4
	//
	// The conversion "basis to euler", done in the step 3, may be different from
	// the original euler, even if the final rotation are the same.
	// This happens because there are more ways to represents the same rotation,
	// both valid, using eulers.
	// For this reason is necessary to convert that euler back to basis and finally
	// compares it.
	//
	// In this way we can assert that both functions: basis to euler / euler to basis
	// are correct.

	bool pass = true;

	// Euler to rotation
	const Vector3 original_euler = ImportUtils::deg2rad(deg_original_euler);
	const Basis to_rotation = ImportUtils::EulerToBasis(rot_order, original_euler);

	// Euler from rotation
	const Vector3 euler_from_rotation = ImportUtils::BasisToEuler(rot_order, to_rotation);
	const Basis rotation_from_computed_euler = ImportUtils::EulerToBasis(rot_order, euler_from_rotation);

	Basis res = to_rotation.inverse() * rotation_from_computed_euler;

	if ((res.get_axis(0) - Vector3(1.0, 0.0, 0.0)).length() > 0.1) {
		OS::get_singleton()->print("Fail due to X %ls\n", String(res.get_axis(0)).c_str());
		pass = false;
	}
	if ((res.get_axis(1) - Vector3(0.0, 1.0, 0.0)).length() > 0.1) {
		OS::get_singleton()->print("Fail due to Y %ls\n", String(res.get_axis(1)).c_str());
		pass = false;
	}
	if ((res.get_axis(2) - Vector3(0.0, 0.0, 1.0)).length() > 0.1) {
		OS::get_singleton()->print("Fail due to Z %ls\n", String(res.get_axis(2)).c_str());
		pass = false;
	}

	if (pass) {
		// Double check that is pass even with another rotation order.
		const Vector3 euler_xyz_from_rotation = to_rotation.get_euler_xyz();
		Basis rotation_from_xyz_computed_euler;
		rotation_from_xyz_computed_euler.set_euler_xyz(euler_xyz_from_rotation);

		res = to_rotation.inverse() * rotation_from_xyz_computed_euler;

		if ((res.get_axis(0) - Vector3(1.0, 0.0, 0.0)).length() > 0.1) {
			OS::get_singleton()->print("Double check with XYZ rot order failed, due to X %ls\n", String(res.get_axis(0)).c_str());
			pass = false;
		}
		if ((res.get_axis(1) - Vector3(0.0, 1.0, 0.0)).length() > 0.1) {
			OS::get_singleton()->print("Double check with XYZ rot order failed, due to Y %ls\n", String(res.get_axis(1)).c_str());
			pass = false;
		}
		if ((res.get_axis(2) - Vector3(0.0, 0.0, 1.0)).length() > 0.1) {
			OS::get_singleton()->print("Double check with XYZ rot order failed, due to Z %ls\n", String(res.get_axis(2)).c_str());
			pass = false;
		}
	}

	if (pass == false) {
		// Print phase only if not pass.
		OS *os = OS::get_singleton();
		switch (rot_order) {
			case Assimp::FBX::Model::RotOrder_EulerXYZ:
				os->print("Rotation order XYZ.");
				break;
			case Assimp::FBX::Model::RotOrder_EulerXZY:
				os->print("Rotation order XZY.");
				break;
			case Assimp::FBX::Model::RotOrder_EulerYZX:
				os->print("Rotation order YZX.");
				break;
			case Assimp::FBX::Model::RotOrder_EulerYXZ:
				os->print("Rotation order YXZ.");
				break;
			case Assimp::FBX::Model::RotOrder_EulerZXY:
				os->print("Rotation order ZXY.");
				break;
			case Assimp::FBX::Model::RotOrder_EulerZYX:
				os->print("Rotation order ZYX.");
				break;
			case Assimp::FBX::Model::RotOrder_SphericXYZ:
				os->print("Rotation order SphericXYZ.");
				break;
			default:
				os->print("Rotation order not supported!");
				return false;
		}
		os->print("\n");
		os->print("Original Rotation: %ls\n", String(deg_original_euler).c_str());
		os->print("Quaternion to rotation order: %ls\n", String(ImportUtils::rad2deg(euler_from_rotation)).c_str());
	}
	return pass;
}

bool test_1() {
	Vector<Assimp::FBX::Model::RotOrder> rotorder_to_test;
	rotorder_to_test.push_back(Assimp::FBX::Model::RotOrder_EulerXYZ);
	rotorder_to_test.push_back(Assimp::FBX::Model::RotOrder_EulerXZY);
	rotorder_to_test.push_back(Assimp::FBX::Model::RotOrder_EulerYZX);
	rotorder_to_test.push_back(Assimp::FBX::Model::RotOrder_EulerYXZ);
	rotorder_to_test.push_back(Assimp::FBX::Model::RotOrder_EulerZXY);
	rotorder_to_test.push_back(Assimp::FBX::Model::RotOrder_EulerZYX);
	//rotorder_to_test.push_back(Assimp::FBX::Model::RotOrder_SphericXYZ);

	Vector<Vector3> vectors_to_test;

	// Test the special cases.
	vectors_to_test.push_back(Vector3(0.0, 0.0, 0.0));
	vectors_to_test.push_back(Vector3(0.5, 0.5, 0.5));
	vectors_to_test.push_back(Vector3(-0.5, -0.5, -0.5));
	vectors_to_test.push_back(Vector3(40.0, 40.0, 40.0));
	vectors_to_test.push_back(Vector3(-40.0, -40.0, -40.0));
	vectors_to_test.push_back(Vector3(0.0, 0.0, -90.0));
	vectors_to_test.push_back(Vector3(0.0, -90.0, 0.0));
	vectors_to_test.push_back(Vector3(-90.0, 0.0, 0.0));
	vectors_to_test.push_back(Vector3(0.0, 0.0, 90.0));
	vectors_to_test.push_back(Vector3(0.0, 90.0, 0.0));
	vectors_to_test.push_back(Vector3(90.0, 0.0, 0.0));
	vectors_to_test.push_back(Vector3(0.0, 0.0, -30.0));
	vectors_to_test.push_back(Vector3(0.0, -30.0, 0.0));
	vectors_to_test.push_back(Vector3(-30.0, 0.0, 0.0));
	vectors_to_test.push_back(Vector3(0.0, 0.0, 30.0));
	vectors_to_test.push_back(Vector3(0.0, 30.0, 0.0));
	vectors_to_test.push_back(Vector3(30.0, 0.0, 0.0));
	vectors_to_test.push_back(Vector3(0.5, 50.0, 20.0));
	vectors_to_test.push_back(Vector3(-0.5, -50.0, -20.0));
	vectors_to_test.push_back(Vector3(0.5, 0.0, 90.0));
	vectors_to_test.push_back(Vector3(0.5, 0.0, -90.0));
	vectors_to_test.push_back(Vector3(360.0, 360.0, 360.0));
	vectors_to_test.push_back(Vector3(-360.0, -360.0, -360.0));
	vectors_to_test.push_back(Vector3(-90.0, 60.0, -90.0));
	vectors_to_test.push_back(Vector3(90.0, 60.0, -90.0));
	vectors_to_test.push_back(Vector3(90.0, -60.0, -90.0));
	vectors_to_test.push_back(Vector3(-90.0, -60.0, -90.0));
	vectors_to_test.push_back(Vector3(-90.0, 60.0, 90.0));
	vectors_to_test.push_back(Vector3(90.0, 60.0, 90.0));
	vectors_to_test.push_back(Vector3(90.0, -60.0, 90.0));
	vectors_to_test.push_back(Vector3(-90.0, -60.0, 90.0));
	vectors_to_test.push_back(Vector3(60.0, 90.0, -40.0));
	vectors_to_test.push_back(Vector3(60.0, -90.0, -40.0));
	vectors_to_test.push_back(Vector3(-60.0, -90.0, -40.0));
	vectors_to_test.push_back(Vector3(-60.0, 90.0, 40.0));
	vectors_to_test.push_back(Vector3(60.0, 90.0, 40.0));
	vectors_to_test.push_back(Vector3(60.0, -90.0, 40.0));
	vectors_to_test.push_back(Vector3(-60.0, -90.0, 40.0));
	vectors_to_test.push_back(Vector3(-90.0, 90.0, -90.0));
	vectors_to_test.push_back(Vector3(90.0, 90.0, -90.0));
	vectors_to_test.push_back(Vector3(90.0, -90.0, -90.0));
	vectors_to_test.push_back(Vector3(-90.0, -90.0, -90.0));
	vectors_to_test.push_back(Vector3(-90.0, 90.0, 90.0));
	vectors_to_test.push_back(Vector3(90.0, 90.0, 90.0));
	vectors_to_test.push_back(Vector3(90.0, -90.0, 90.0));
	vectors_to_test.push_back(Vector3(20.0, 150.0, 30.0));
	vectors_to_test.push_back(Vector3(20.0, -150.0, 30.0));
	vectors_to_test.push_back(Vector3(-120.0, -150.0, 30.0));
	vectors_to_test.push_back(Vector3(-120.0, -150.0, -130.0));
	vectors_to_test.push_back(Vector3(120.0, -150.0, -130.0));
	vectors_to_test.push_back(Vector3(120.0, 150.0, -130.0));
	vectors_to_test.push_back(Vector3(120.0, 150.0, 130.0));

	// Add 1000 random vectors with weirds numbers.
	RandomNumberGenerator rng;
	for (int _ = 0; _ < 1000; _ += 1) {
		vectors_to_test.push_back(Vector3(
				rng.randf_range(-1800, 1800),
				rng.randf_range(-1800, 1800),
				rng.randf_range(-1800, 1800)));
	}

	int passed = 0;
	int failed = 0;
	for (int h = 0; h < rotorder_to_test.size(); h += 1) {
		for (int i = 0; i < vectors_to_test.size(); i += 1) {
			if (test_rotation(vectors_to_test[i], rotorder_to_test[h])) {
				//OS::get_singleton()->print("Success. \n\n");
				passed += 1;
			} else {
				OS::get_singleton()->print("FAILED                   FAILED                        FAILED. \n\n");
				OS::get_singleton()->print("------------>\n");
				OS::get_singleton()->print("------------>\n");
				failed += 1;
			}
		}
	}

	OS::get_singleton()->print("Tests passed: %i\n", passed);
	OS::get_singleton()->print("Tests failed: %i\n", failed);

	return failed == 0;
}

typedef bool (*TestFunc)(void);

TestFunc test_funcs[] = {
	test_1,
	// End
	0
};

MainLoop *test() {
	/** A character length != wchar_t may be forced, so the tests won't work */

	ERR_FAIL_COND_V(sizeof(CharType) != sizeof(wchar_t), NULL);

	int count = 0;
	int passed = 0;

	while (true) {
		if (!test_funcs[count])
			break;

		OS::get_singleton()->print("\n---------------------------------------------\n");
		OS::get_singleton()->print("[fbx] running test: %d\n", count + 1);
		bool pass = test_funcs[count]();
		if (pass)
			passed++;
		OS::get_singleton()->print("\t%s\n", pass ? "PASS" : "FAILED");
		count++;
	}

	OS::get_singleton()->print("\n\n\n");
	OS::get_singleton()->print("*************\n");
	OS::get_singleton()->print("***TOTALS!***\n");
	OS::get_singleton()->print("*************\n");

	OS::get_singleton()->print("Passed %i of %i tests\n", passed, count);

	return NULL;
}
} // namespace TestFBX
