// Code sample from Filippo Ramaciotti

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_cross_product.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <cstdio>
#include <vector>
#include <utility>

namespace test_eulerAngleX
{
	int test()
	{
		int Error = 0;

		float const Angle(glm::pi<float>() * 0.5f);
		glm::vec3 const X(1.0f, 0.0f, 0.0f);

		glm::vec4 const Y(0.0f, 1.0f, 0.0f, 1.0f);
		glm::vec4 const Y1 = glm::rotate(glm::mat4(1.0f), Angle, X) * Y;
		glm::vec4 const Y2 = glm::eulerAngleX(Angle) * Y;
		glm::vec4 const Y3 = glm::eulerAngleXY(Angle, 0.0f) * Y;
		glm::vec4 const Y4 = glm::eulerAngleYX(0.0f, Angle) * Y;
		glm::vec4 const Y5 = glm::eulerAngleXZ(Angle, 0.0f) * Y;
		glm::vec4 const Y6 = glm::eulerAngleZX(0.0f, Angle) * Y;
		glm::vec4 const Y7 = glm::eulerAngleYXZ(0.0f, Angle, 0.0f) * Y;
		Error += glm::all(glm::epsilonEqual(Y1, Y2, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Y1, Y3, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Y1, Y4, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Y1, Y5, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Y1, Y6, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Y1, Y7, 0.00001f)) ? 0 : 1;

		glm::vec4 const Z(0.0f, 0.0f, 1.0f, 1.0f);
		glm::vec4 const Z1 = glm::rotate(glm::mat4(1.0f), Angle, X) * Z;
		glm::vec4 const Z2 = glm::eulerAngleX(Angle) * Z;
		glm::vec4 const Z3 = glm::eulerAngleXY(Angle, 0.0f) * Z;
		glm::vec4 const Z4 = glm::eulerAngleYX(0.0f, Angle) * Z;
		glm::vec4 const Z5 = glm::eulerAngleXZ(Angle, 0.0f) * Z;
		glm::vec4 const Z6 = glm::eulerAngleZX(0.0f, Angle) * Z;
		glm::vec4 const Z7 = glm::eulerAngleYXZ(0.0f, Angle, 0.0f) * Z;
		Error += glm::all(glm::epsilonEqual(Z1, Z2, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z3, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z4, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z5, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z6, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z7, 0.00001f)) ? 0 : 1;

		return Error;
	}
}//namespace test_eulerAngleX

namespace test_eulerAngleY
{
	int test()
	{
		int Error = 0;

		float const Angle(glm::pi<float>() * 0.5f);
		glm::vec3 const Y(0.0f, 1.0f, 0.0f);

		glm::vec4 const X(1.0f, 0.0f, 0.0f, 1.0f);
		glm::vec4 const X1 = glm::rotate(glm::mat4(1.0f), Angle, Y) * X;
		glm::vec4 const X2 = glm::eulerAngleY(Angle) * X;
		glm::vec4 const X3 = glm::eulerAngleYX(Angle, 0.0f) * X;
		glm::vec4 const X4 = glm::eulerAngleXY(0.0f, Angle) * X;
		glm::vec4 const X5 = glm::eulerAngleYZ(Angle, 0.0f) * X;
		glm::vec4 const X6 = glm::eulerAngleZY(0.0f, Angle) * X;
		glm::vec4 const X7 = glm::eulerAngleYXZ(Angle, 0.0f, 0.0f) * X;
		Error += glm::all(glm::epsilonEqual(X1, X2, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(X1, X3, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(X1, X4, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(X1, X5, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(X1, X6, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(X1, X7, 0.00001f)) ? 0 : 1;

		glm::vec4 const Z(0.0f, 0.0f, 1.0f, 1.0f);
		glm::vec4 const Z1 = glm::eulerAngleY(Angle) * Z;
		glm::vec4 const Z2 = glm::rotate(glm::mat4(1.0f), Angle, Y) * Z;
		glm::vec4 const Z3 = glm::eulerAngleYX(Angle, 0.0f) * Z;
		glm::vec4 const Z4 = glm::eulerAngleXY(0.0f, Angle) * Z;
		glm::vec4 const Z5 = glm::eulerAngleYZ(Angle, 0.0f) * Z;
		glm::vec4 const Z6 = glm::eulerAngleZY(0.0f, Angle) * Z;
		glm::vec4 const Z7 = glm::eulerAngleYXZ(Angle, 0.0f, 0.0f) * Z;
		Error += glm::all(glm::epsilonEqual(Z1, Z2, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z3, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z4, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z5, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z6, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z7, 0.00001f)) ? 0 : 1;

		return Error;
	}
}//namespace test_eulerAngleY

namespace test_eulerAngleZ
{
	int test()
	{
		int Error = 0;

		float const Angle(glm::pi<float>() * 0.5f);
		glm::vec3 const Z(0.0f, 0.0f, 1.0f);

		glm::vec4 const X(1.0f, 0.0f, 0.0f, 1.0f);
		glm::vec4 const X1 = glm::rotate(glm::mat4(1.0f), Angle, Z) * X;
		glm::vec4 const X2 = glm::eulerAngleZ(Angle) * X;
		glm::vec4 const X3 = glm::eulerAngleZX(Angle, 0.0f) * X;
		glm::vec4 const X4 = glm::eulerAngleXZ(0.0f, Angle) * X;
		glm::vec4 const X5 = glm::eulerAngleZY(Angle, 0.0f) * X;
		glm::vec4 const X6 = glm::eulerAngleYZ(0.0f, Angle) * X;
		glm::vec4 const X7 = glm::eulerAngleYXZ(0.0f, 0.0f, Angle) * X;
		Error += glm::all(glm::epsilonEqual(X1, X2, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(X1, X3, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(X1, X4, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(X1, X5, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(X1, X6, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(X1, X7, 0.00001f)) ? 0 : 1;

		glm::vec4 const Y(1.0f, 0.0f, 0.0f, 1.0f);
		glm::vec4 const Z1 = glm::rotate(glm::mat4(1.0f), Angle, Z) * Y;
		glm::vec4 const Z2 = glm::eulerAngleZ(Angle) * Y;
		glm::vec4 const Z3 = glm::eulerAngleZX(Angle, 0.0f) * Y;
		glm::vec4 const Z4 = glm::eulerAngleXZ(0.0f, Angle) * Y;
		glm::vec4 const Z5 = glm::eulerAngleZY(Angle, 0.0f) * Y;
		glm::vec4 const Z6 = glm::eulerAngleYZ(0.0f, Angle) * Y;
		glm::vec4 const Z7 = glm::eulerAngleYXZ(0.0f, 0.0f, Angle) * Y;
		Error += glm::all(glm::epsilonEqual(Z1, Z2, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z3, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z4, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z5, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z6, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(Z1, Z7, 0.00001f)) ? 0 : 1;

		return Error;
	}
}//namespace test_eulerAngleZ

namespace test_derivedEulerAngles
{
	bool epsilonEqual(glm::mat4 const& mat1, glm::mat4 const& mat2, glm::mat4::value_type const& epsilon)
	{
		return glm::all(glm::epsilonEqual(mat1[0], mat2[0], epsilon)) ?
				(
					glm::all(glm::epsilonEqual(mat1[1], mat2[1], epsilon)) ?
					(
						glm::all(glm::epsilonEqual(mat1[2], mat2[2], epsilon)) ?
						(
							glm::all(glm::epsilonEqual(mat1[3], mat2[3], epsilon)) ? true : false
						) : false
					) : false
				) : false;
	}

	template<typename RotationFunc, typename TestDerivedFunc>
	int test(RotationFunc rotationFunc, TestDerivedFunc testDerivedFunc, const glm::vec3& basis)
	{
		int Error = 0;

		typedef glm::vec3::value_type value;
		value const zeroAngle(0.0f);
		value const Angle(glm::pi<float>() * 0.75f);
		value const negativeAngle(-Angle);
		value const zeroAngleVelocity(0.0f);
		value const AngleVelocity(glm::pi<float>() * 0.27f);
		value const negativeAngleVelocity(-AngleVelocity);

		typedef std::pair<value,value> AngleAndAngleVelocity;
		std::vector<AngleAndAngleVelocity> testPairs;
		testPairs.push_back(AngleAndAngleVelocity(zeroAngle, zeroAngleVelocity));
		testPairs.push_back(AngleAndAngleVelocity(zeroAngle, AngleVelocity));
		testPairs.push_back(AngleAndAngleVelocity(zeroAngle, negativeAngleVelocity));
		testPairs.push_back(AngleAndAngleVelocity(Angle, zeroAngleVelocity));
		testPairs.push_back(AngleAndAngleVelocity(Angle, AngleVelocity));
		testPairs.push_back(AngleAndAngleVelocity(Angle, negativeAngleVelocity));
		testPairs.push_back(AngleAndAngleVelocity(negativeAngle, zeroAngleVelocity));
		testPairs.push_back(AngleAndAngleVelocity(negativeAngle, AngleVelocity));
		testPairs.push_back(AngleAndAngleVelocity(negativeAngle, negativeAngleVelocity));

		for (size_t i = 0, size = testPairs.size(); i < size; ++i)
		{
			AngleAndAngleVelocity const& pair = testPairs.at(i);

			glm::mat4 const W = glm::matrixCross4(basis * pair.second);
			glm::mat4 const rotMt = glm::transpose(rotationFunc(pair.first));
			glm::mat4 const derivedRotM = testDerivedFunc(pair.first, pair.second);

			Error += epsilonEqual(W, derivedRotM * rotMt, 0.00001f) ? 0 : 1;
		}

		return Error;
	}
}//namespace test_derivedEulerAngles

namespace test_eulerAngleXY
{
	int test()
	{
		int Error = 0;

		glm::vec4 const V(1.0f);

		float const AngleX(glm::pi<float>() * 0.5f);
		float const AngleY(glm::pi<float>() * 0.25f);

		glm::vec3 const axisX(1.0f, 0.0f, 0.0f);
		glm::vec3 const axisY(0.0f, 1.0f, 0.0f);

		glm::vec4 const V1 = (glm::rotate(glm::mat4(1.0f), AngleX, axisX) * glm::rotate(glm::mat4(1.0f), AngleY, axisY)) * V;
		glm::vec4 const V2 = glm::eulerAngleXY(AngleX, AngleY) * V;
		glm::vec4 const V3 = glm::eulerAngleX(AngleX) * glm::eulerAngleY(AngleY) * V;
		Error += glm::all(glm::epsilonEqual(V1, V2, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(V1, V3, 0.00001f)) ? 0 : 1;

		return Error;
	}
}//namespace test_eulerAngleXY

namespace test_eulerAngleYX
{
	int test()
	{
		int Error = 0;

		glm::vec4 const V(1.0f);

		float const AngleX(glm::pi<float>() * 0.5f);
		float const AngleY(glm::pi<float>() * 0.25f);

		glm::vec3 const axisX(1.0f, 0.0f, 0.0f);
		glm::vec3 const axisY(0.0f, 1.0f, 0.0f);

		glm::vec4 const V1 = (glm::rotate(glm::mat4(1.0f), AngleY, axisY) * glm::rotate(glm::mat4(1.0f), AngleX, axisX)) * V;
		glm::vec4 const V2 = glm::eulerAngleYX(AngleY, AngleX) * V;
		glm::vec4 const V3 = glm::eulerAngleY(AngleY) * glm::eulerAngleX(AngleX) * V;
		Error += glm::all(glm::epsilonEqual(V1, V2, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(V1, V3, 0.00001f)) ? 0 : 1;

		return Error;
	}
}//namespace test_eulerAngleYX

namespace test_eulerAngleXZ
{
	int test()
	{
		int Error = 0;

		glm::vec4 const V(1.0f);

		float const AngleX(glm::pi<float>() * 0.5f);
		float const AngleZ(glm::pi<float>() * 0.25f);

		glm::vec3 const axisX(1.0f, 0.0f, 0.0f);
		glm::vec3 const axisZ(0.0f, 0.0f, 1.0f);

		glm::vec4 const V1 = (glm::rotate(glm::mat4(1.0f), AngleX, axisX) * glm::rotate(glm::mat4(1.0f), AngleZ, axisZ)) * V;
		glm::vec4 const V2 = glm::eulerAngleXZ(AngleX, AngleZ) * V;
		glm::vec4 const V3 = glm::eulerAngleX(AngleX) * glm::eulerAngleZ(AngleZ) * V;
		Error += glm::all(glm::epsilonEqual(V1, V2, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(V1, V3, 0.00001f)) ? 0 : 1;

		return Error;
	}
}//namespace test_eulerAngleXZ

namespace test_eulerAngleZX
{
	int test()
	{
		int Error = 0;

		glm::vec4 const V(1.0f);

		float const AngleX(glm::pi<float>() * 0.5f);
		float const AngleZ(glm::pi<float>() * 0.25f);

		glm::vec3 const axisX(1.0f, 0.0f, 0.0f);
		glm::vec3 const axisZ(0.0f, 0.0f, 1.0f);

		glm::vec4 const V1 = (glm::rotate(glm::mat4(1.0f), AngleZ, axisZ) * glm::rotate(glm::mat4(1.0f), AngleX, axisX)) * V;
		glm::vec4 const V2 = glm::eulerAngleZX(AngleZ, AngleX) * V;
		glm::vec4 const V3 = glm::eulerAngleZ(AngleZ) * glm::eulerAngleX(AngleX) * V;
		Error += glm::all(glm::epsilonEqual(V1, V2, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(V1, V3, 0.00001f)) ? 0 : 1;

		return Error;
	}
}//namespace test_eulerAngleZX

namespace test_eulerAngleYZ
{
	int test()
	{
		int Error = 0;

		glm::vec4 const V(1.0f);

		float const AngleY(glm::pi<float>() * 0.5f);
		float const AngleZ(glm::pi<float>() * 0.25f);

		glm::vec3 const axisX(1.0f, 0.0f, 0.0f);
		glm::vec3 const axisY(0.0f, 1.0f, 0.0f);
		glm::vec3 const axisZ(0.0f, 0.0f, 1.0f);

		glm::vec4 const V1 = (glm::rotate(glm::mat4(1.0f), AngleY, axisY) * glm::rotate(glm::mat4(1.0f), AngleZ, axisZ)) * V;
		glm::vec4 const V2 = glm::eulerAngleYZ(AngleY, AngleZ) * V;
		glm::vec4 const V3 = glm::eulerAngleY(AngleY) * glm::eulerAngleZ(AngleZ) * V;
		Error += glm::all(glm::epsilonEqual(V1, V2, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(V1, V3, 0.00001f)) ? 0 : 1;

		return Error;
	}
}//namespace test_eulerAngleYZ

namespace test_eulerAngleZY
{
	int test()
	{
		int Error = 0;

		glm::vec4 const V(1.0f);

		float const AngleY(glm::pi<float>() * 0.5f);
		float const AngleZ(glm::pi<float>() * 0.25f);

		glm::vec3 const axisX(1.0f, 0.0f, 0.0f);
		glm::vec3 const axisY(0.0f, 1.0f, 0.0f);
		glm::vec3 const axisZ(0.0f, 0.0f, 1.0f);

		glm::vec4 const V1 = (glm::rotate(glm::mat4(1.0f), AngleZ, axisZ) * glm::rotate(glm::mat4(1.0f), AngleY, axisY)) * V;
		glm::vec4 const V2 = glm::eulerAngleZY(AngleZ, AngleY) * V;
		glm::vec4 const V3 = glm::eulerAngleZ(AngleZ) * glm::eulerAngleY(AngleY) * V;
		Error += glm::all(glm::epsilonEqual(V1, V2, 0.00001f)) ? 0 : 1;
		Error += glm::all(glm::epsilonEqual(V1, V3, 0.00001f)) ? 0 : 1;

		return Error;
	}
}//namespace test_eulerAngleZY

namespace test_eulerAngleYXZ
{
	int test()
	{
		glm::f32 first =  1.046f;
		glm::f32 second = 0.52f;
		glm::f32 third = -0.785f;

		glm::fmat4 rotationEuler = glm::eulerAngleYXZ(first, second, third); 

		glm::fmat4 rotationInvertedY  = glm::eulerAngleY(-1.f*first) * glm::eulerAngleX(second) * glm::eulerAngleZ(third); 
		glm::fmat4 rotationDumb = glm::fmat4(); 
		rotationDumb = glm::rotate(rotationDumb, first, glm::fvec3(0,1,0));
		rotationDumb = glm::rotate(rotationDumb, second, glm::fvec3(1,0,0));
		rotationDumb = glm::rotate(rotationDumb, third, glm::fvec3(0,0,1));

		std::printf("%s\n", glm::to_string(glm::fmat3(rotationEuler)).c_str());
		std::printf("%s\n", glm::to_string(glm::fmat3(rotationDumb)).c_str());
		std::printf("%s\n", glm::to_string(glm::fmat3(rotationInvertedY)).c_str());

		std::printf("\nRESIDUAL\n");
		std::printf("%s\n", glm::to_string(glm::fmat3(rotationEuler-(rotationDumb))).c_str());
		std::printf("%s\n", glm::to_string(glm::fmat3(rotationEuler-(rotationInvertedY))).c_str());

		return 0;
	}
}//namespace eulerAngleYXZ

namespace test_eulerAngles
{
	template<typename TestRotationFunc>
	int test(TestRotationFunc testRotationFunc, glm::vec3 const& I, glm::vec3 const& J, glm::vec3 const& K)
	{
		int Error = 0;

		typedef glm::mat4::value_type value;
		value const minAngle(-glm::pi<value>());
		value const maxAngle(glm::pi<value>());
		value const maxAngleWithDelta(maxAngle - 0.0000001f);
		value const minMidAngle(-glm::pi<value>() * 0.5f);
		value const maxMidAngle(glm::pi<value>() * 0.5f);

		std::vector<glm::vec3> testEulerAngles;
		testEulerAngles.push_back(glm::vec3(1.046f, 0.52f, -0.785f));
		testEulerAngles.push_back(glm::vec3(minAngle, minMidAngle, minAngle));
		testEulerAngles.push_back(glm::vec3(minAngle, minMidAngle, maxAngle));
		testEulerAngles.push_back(glm::vec3(minAngle, minMidAngle, maxAngleWithDelta));
		testEulerAngles.push_back(glm::vec3(minAngle, maxMidAngle, minAngle));
		testEulerAngles.push_back(glm::vec3(minAngle, maxMidAngle, maxAngle));
		testEulerAngles.push_back(glm::vec3(minAngle, maxMidAngle, maxAngleWithDelta));
		testEulerAngles.push_back(glm::vec3(maxAngle, minMidAngle, minAngle));
		testEulerAngles.push_back(glm::vec3(maxAngle, minMidAngle, maxAngle));
		testEulerAngles.push_back(glm::vec3(maxAngle, minMidAngle, maxAngleWithDelta));
		testEulerAngles.push_back(glm::vec3(maxAngleWithDelta, minMidAngle, maxAngle));
		testEulerAngles.push_back(glm::vec3(maxAngleWithDelta, minMidAngle, maxAngleWithDelta));
		testEulerAngles.push_back(glm::vec3(maxAngle, maxMidAngle, minAngle));
		testEulerAngles.push_back(glm::vec3(maxAngleWithDelta, maxMidAngle, minAngle));
		testEulerAngles.push_back(glm::vec3(maxAngle, maxMidAngle, maxAngle));
		testEulerAngles.push_back(glm::vec3(maxAngle, maxMidAngle, maxAngleWithDelta));
		testEulerAngles.push_back(glm::vec3(maxAngleWithDelta, maxMidAngle, maxAngle));
		testEulerAngles.push_back(glm::vec3(maxAngleWithDelta, maxMidAngle, maxAngleWithDelta));
		testEulerAngles.push_back(glm::vec3(minAngle, 0.0f, minAngle));
		testEulerAngles.push_back(glm::vec3(minAngle, 0.0f, maxAngle));
		testEulerAngles.push_back(glm::vec3(maxAngle, maxAngle, minAngle));
		testEulerAngles.push_back(glm::vec3(maxAngle, maxAngle, maxAngle));

		for (size_t i = 0, size = testEulerAngles.size(); i < size; ++i)
		{
			glm::vec3 const& angles = testEulerAngles.at(i);
			glm::mat4 const rotationEuler = testRotationFunc(angles.x, angles.y, angles.z);

			glm::mat4 rotationDumb = glm::diagonal4x4(glm::mat4::col_type(1.0f));
			rotationDumb = glm::rotate(rotationDumb, angles.x, I);
			rotationDumb = glm::rotate(rotationDumb, angles.y, J);
			rotationDumb = glm::rotate(rotationDumb, angles.z, K);

			glm::vec4 const V(1.0f,1.0f,1.0f,1.0f);
			glm::vec4 const V1 = rotationEuler * V;
			glm::vec4 const V2 = rotationDumb * V;

			Error += glm::all(glm::epsilonEqual(V1, V2, 0.00001f)) ? 0 : 1;
		}

		return Error;
	}
}//namespace test_extractsEulerAngles

namespace test_extractsEulerAngles
{
	template<typename RotationFunc, typename TestExtractionFunc>
	int test(RotationFunc rotationFunc, TestExtractionFunc testExtractionFunc)
	{
		int Error = 0;

		typedef glm::mat4::value_type value;
		value const minAngle(-glm::pi<value>());
		value const maxAngle(glm::pi<value>());
		value const maxAngleWithDelta(maxAngle - 0.0000001f);
		value const minMidAngle(-glm::pi<value>() * 0.5f);
		value const maxMidAngle(glm::pi<value>() * 0.5f);

		std::vector<glm::vec3> testEulerAngles;
		testEulerAngles.push_back(glm::vec3(1.046f, 0.52f, -0.785f));
		testEulerAngles.push_back(glm::vec3(minAngle, minMidAngle, minAngle));
		testEulerAngles.push_back(glm::vec3(minAngle, minMidAngle, maxAngle));
		testEulerAngles.push_back(glm::vec3(minAngle, minMidAngle, maxAngleWithDelta));
		testEulerAngles.push_back(glm::vec3(minAngle, maxMidAngle, minAngle));
		testEulerAngles.push_back(glm::vec3(minAngle, maxMidAngle, maxAngle));
		testEulerAngles.push_back(glm::vec3(minAngle, maxMidAngle, maxAngleWithDelta));
		testEulerAngles.push_back(glm::vec3(maxAngle, minMidAngle, minAngle));
		testEulerAngles.push_back(glm::vec3(maxAngle, minMidAngle, maxAngle));
		testEulerAngles.push_back(glm::vec3(maxAngle, minMidAngle, maxAngleWithDelta));
		testEulerAngles.push_back(glm::vec3(maxAngleWithDelta, minMidAngle, maxAngle));
		testEulerAngles.push_back(glm::vec3(maxAngleWithDelta, minMidAngle, maxAngleWithDelta));
		testEulerAngles.push_back(glm::vec3(maxAngle, maxMidAngle, minAngle));
		testEulerAngles.push_back(glm::vec3(maxAngleWithDelta, maxMidAngle, minAngle));
		testEulerAngles.push_back(glm::vec3(maxAngle, maxMidAngle, maxAngle));
		testEulerAngles.push_back(glm::vec3(maxAngle, maxMidAngle, maxAngleWithDelta));
		testEulerAngles.push_back(glm::vec3(maxAngleWithDelta, maxMidAngle, maxAngle));
		testEulerAngles.push_back(glm::vec3(maxAngleWithDelta, maxMidAngle, maxAngleWithDelta));
		testEulerAngles.push_back(glm::vec3(minAngle, 0.0f, minAngle));
		testEulerAngles.push_back(glm::vec3(minAngle, 0.0f, maxAngle));
		testEulerAngles.push_back(glm::vec3(maxAngle, maxAngle, minAngle));
		testEulerAngles.push_back(glm::vec3(maxAngle, maxAngle, maxAngle));

		for (size_t i = 0, size = testEulerAngles.size(); i < size; ++i)
		{
			glm::vec3 const& angles = testEulerAngles.at(i);
			glm::mat4 const rotation = rotationFunc(angles.x, angles.y, angles.z);

			glm::vec3 extractedEulerAngles(0.0f);
			testExtractionFunc(rotation, extractedEulerAngles.x, extractedEulerAngles.y, extractedEulerAngles.z);
			glm::mat4 const extractedRotation = rotationFunc(extractedEulerAngles.x, extractedEulerAngles.y, extractedEulerAngles.z);

			glm::vec4 const V(1.0f,1.0f,1.0f,1.0f);
			glm::vec4 const V1 = rotation * V;
			glm::vec4 const V2 = extractedRotation * V;

			Error += glm::all(glm::epsilonEqual(V1, V2, 0.00001f)) ? 0 : 1;
		}

		return Error;
	}
}//namespace test_extractsEulerAngles

int main()
{ 
	int Error = 0;

	typedef glm::mat4::value_type value;
	glm::vec3 const X(1.0f, 0.0f, 0.0f);
	glm::vec3 const Y(0.0f, 1.0f, 0.0f);
	glm::vec3 const Z(0.0f, 0.0f, 1.0f);

	Error += test_eulerAngleX::test();
	Error += test_eulerAngleY::test();
	Error += test_eulerAngleZ::test();

	Error += test_derivedEulerAngles::test(glm::eulerAngleX<value>, glm::derivedEulerAngleX<value>, X);
	Error += test_derivedEulerAngles::test(glm::eulerAngleY<value>, glm::derivedEulerAngleY<value>, Y);
	Error += test_derivedEulerAngles::test(glm::eulerAngleZ<value>, glm::derivedEulerAngleZ<value>, Z);

	Error += test_eulerAngleXY::test();
	Error += test_eulerAngleYX::test();
	Error += test_eulerAngleXZ::test();
	Error += test_eulerAngleZX::test();
	Error += test_eulerAngleYZ::test();
	Error += test_eulerAngleZY::test();
	Error += test_eulerAngleYXZ::test();

	Error += test_eulerAngles::test(glm::eulerAngleXZX<value>, X, Z, X);
	Error += test_eulerAngles::test(glm::eulerAngleXYX<value>, X, Y, X);
	Error += test_eulerAngles::test(glm::eulerAngleYXY<value>, Y, X, Y);
	Error += test_eulerAngles::test(glm::eulerAngleYZY<value>, Y, Z, Y);
	Error += test_eulerAngles::test(glm::eulerAngleZYZ<value>, Z, Y, Z);
	Error += test_eulerAngles::test(glm::eulerAngleZXZ<value>, Z, X, Z);
	Error += test_eulerAngles::test(glm::eulerAngleXZY<value>, X, Z, Y);
	Error += test_eulerAngles::test(glm::eulerAngleYZX<value>, Y, Z, X);
	Error += test_eulerAngles::test(glm::eulerAngleZYX<value>, Z, Y, X);
	Error += test_eulerAngles::test(glm::eulerAngleZXY<value>, Z, X, Y);

	Error += test_extractsEulerAngles::test(glm::eulerAngleYXZ<value>, glm::extractEulerAngleYXZ<value>);
	Error += test_extractsEulerAngles::test(glm::eulerAngleXZX<value>, glm::extractEulerAngleXZX<value>);
	Error += test_extractsEulerAngles::test(glm::eulerAngleXYX<value>, glm::extractEulerAngleXYX<value>);
	Error += test_extractsEulerAngles::test(glm::eulerAngleYXY<value>, glm::extractEulerAngleYXY<value>);
	Error += test_extractsEulerAngles::test(glm::eulerAngleYZY<value>, glm::extractEulerAngleYZY<value>);
	Error += test_extractsEulerAngles::test(glm::eulerAngleZYZ<value>, glm::extractEulerAngleZYZ<value>);
	Error += test_extractsEulerAngles::test(glm::eulerAngleZXZ<value>, glm::extractEulerAngleZXZ<value>);
	Error += test_extractsEulerAngles::test(glm::eulerAngleXZY<value>, glm::extractEulerAngleXZY<value>);
	Error += test_extractsEulerAngles::test(glm::eulerAngleYZX<value>, glm::extractEulerAngleYZX<value>);
	Error += test_extractsEulerAngles::test(glm::eulerAngleZYX<value>, glm::extractEulerAngleZYX<value>);
	Error += test_extractsEulerAngles::test(glm::eulerAngleZXY<value>, glm::extractEulerAngleZXY<value>);

	return Error; 
}
