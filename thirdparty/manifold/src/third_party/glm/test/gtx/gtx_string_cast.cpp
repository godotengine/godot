#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <limits>

int test_string_cast_vector()
{
	int Error = 0;

	{
		glm::vec2 A1(1, 2);
		std::string A2 = glm::to_string(A1);
		Error += A2 != std::string("vec2(1.000000, 2.000000)") ? 1 : 0;
	
		glm::vec3 B1(1, 2, 3);
		std::string B2 = glm::to_string(B1);
		Error += B2 != std::string("vec3(1.000000, 2.000000, 3.000000)") ? 1 : 0;

		glm::vec4 C1(1, 2, 3, 4);
		std::string C2 = glm::to_string(C1);
		Error += C2 != std::string("vec4(1.000000, 2.000000, 3.000000, 4.000000)") ? 1 : 0;
	
		glm::dvec2 J1(1, 2);
		std::string J2 = glm::to_string(J1);
		Error += J2 != std::string("dvec2(1.000000, 2.000000)") ? 1 : 0;
	
		glm::dvec3 K1(1, 2, 3);
		std::string K2 = glm::to_string(K1);
		Error += K2 != std::string("dvec3(1.000000, 2.000000, 3.000000)") ? 1 : 0;
	
		glm::dvec4 L1(1, 2, 3, 4);
		std::string L2 = glm::to_string(L1);
		Error += L2 != std::string("dvec4(1.000000, 2.000000, 3.000000, 4.000000)") ? 1 : 0;
	}

	{
		glm::bvec2 M1(false, true);
		std::string M2 = glm::to_string(M1);
		Error += M2 != std::string("bvec2(false, true)") ? 1 : 0;
	
		glm::bvec3 O1(false, true, false);
		std::string O2 = glm::to_string(O1);
		Error += O2 != std::string("bvec3(false, true, false)") ? 1 : 0;

		glm::bvec4 P1(false, true, false, true);
		std::string P2 = glm::to_string(P1);
		Error += P2 != std::string("bvec4(false, true, false, true)") ? 1 : 0;
	}

	{
		glm::ivec2 D1(1, 2);
		std::string D2 = glm::to_string(D1);
		Error += D2 != std::string("ivec2(1, 2)") ? 1 : 0;
	
		glm::ivec3 E1(1, 2, 3);
		std::string E2 = glm::to_string(E1);
		Error += E2 != std::string("ivec3(1, 2, 3)") ? 1 : 0;
	
		glm::ivec4 F1(1, 2, 3, 4);
		std::string F2 = glm::to_string(F1);
		Error += F2 != std::string("ivec4(1, 2, 3, 4)") ? 1 : 0;
	}

	{
		glm::i8vec2 D1(1, 2);
		std::string D2 = glm::to_string(D1);
		Error += D2 != std::string("i8vec2(1, 2)") ? 1 : 0;
	
		glm::i8vec3 E1(1, 2, 3);
		std::string E2 = glm::to_string(E1);
		Error += E2 != std::string("i8vec3(1, 2, 3)") ? 1 : 0;
	
		glm::i8vec4 F1(1, 2, 3, 4);
		std::string F2 = glm::to_string(F1);
		Error += F2 != std::string("i8vec4(1, 2, 3, 4)") ? 1 : 0;
	}

	{
		glm::i16vec2 D1(1, 2);
		std::string D2 = glm::to_string(D1);
		Error += D2 != std::string("i16vec2(1, 2)") ? 1 : 0;
	
		glm::i16vec3 E1(1, 2, 3);
		std::string E2 = glm::to_string(E1);
		Error += E2 != std::string("i16vec3(1, 2, 3)") ? 1 : 0;
	
		glm::i16vec4 F1(1, 2, 3, 4);
		std::string F2 = glm::to_string(F1);
		Error += F2 != std::string("i16vec4(1, 2, 3, 4)") ? 1 : 0;
	}

	{
		glm::i64vec2 D1(1, 2);
		std::string D2 = glm::to_string(D1);
		Error += D2 != std::string("i64vec2(1, 2)") ? 1 : 0;
	
		glm::i64vec3 E1(1, 2, 3);
		std::string E2 = glm::to_string(E1);
		Error += E2 != std::string("i64vec3(1, 2, 3)") ? 1 : 0;
	
		glm::i64vec4 F1(1, 2, 3, 4);
		std::string F2 = glm::to_string(F1);
		Error += F2 != std::string("i64vec4(1, 2, 3, 4)") ? 1 : 0;
	}

	return Error;
}

int test_string_cast_matrix()
{
	int Error = 0;
	
	glm::mat2x2 A1(1.000000, 2.000000, 3.000000, 4.000000);
	std::string A2 = glm::to_string(A1);
	Error += A2 != std::string("mat2x2((1.000000, 2.000000), (3.000000, 4.000000))") ? 1 : 0;

	return Error;
}

int test_string_cast_quaternion()
{
	int Error = 0;

	glm::quat Q0 = glm::quat(1.0f, 2.0f, 3.0f, 4.0f);
	std::string S0 = glm::to_string(Q0);
	Error += S0 != std::string("quat(1.000000, {2.000000, 3.000000, 4.000000})") ? 1 : 0;

	return Error;

}

int test_string_cast_dual_quaternion()
{
	int Error = 0;

	glm::dualquat Q0 = glm::dualquat(glm::quat(1.0f, 2.0f, 3.0f, 4.0f), glm::quat(5.0f, 6.0f, 7.0f, 8.0f));
	std::string S0 = glm::to_string(Q0);
	Error += S0 != std::string("dualquat((1.000000, {2.000000, 3.000000, 4.000000}), (5.000000, {6.000000, 7.000000, 8.000000}))") ? 1 : 0;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_string_cast_vector();
	Error += test_string_cast_matrix();
	Error += test_string_cast_quaternion();
	Error += test_string_cast_dual_quaternion();

	return Error;
}


