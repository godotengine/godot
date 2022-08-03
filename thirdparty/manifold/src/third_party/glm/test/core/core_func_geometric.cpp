#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <glm/ext/scalar_relational.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/vector_float1.hpp>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_double2.hpp>
#include <glm/ext/vector_double3.hpp>
#include <glm/ext/vector_double4.hpp>
#include <limits>

namespace length
{
	int test()
	{
		float Length1 = glm::length(glm::vec1(1));
		float Length2 = glm::length(glm::vec2(1, 0));
		float Length3 = glm::length(glm::vec3(1, 0, 0));
		float Length4 = glm::length(glm::vec4(1, 0, 0, 0));

		int Error = 0;

		Error += glm::abs(Length1 - 1.0f) < std::numeric_limits<float>::epsilon() ? 0 : 1;
		Error += glm::abs(Length2 - 1.0f) < std::numeric_limits<float>::epsilon() ? 0 : 1;
		Error += glm::abs(Length3 - 1.0f) < std::numeric_limits<float>::epsilon() ? 0 : 1;
		Error += glm::abs(Length4 - 1.0f) < std::numeric_limits<float>::epsilon() ? 0 : 1;

		return Error;
	}
}//namespace length

namespace distance
{
	int test()
	{
		float Distance1 = glm::distance(glm::vec1(1), glm::vec1(1));
		float Distance2 = glm::distance(glm::vec2(1, 0), glm::vec2(1, 0));
		float Distance3 = glm::distance(glm::vec3(1, 0, 0), glm::vec3(1, 0, 0));
		float Distance4 = glm::distance(glm::vec4(1, 0, 0, 0), glm::vec4(1, 0, 0, 0));

		int Error = 0;

		Error += glm::abs(Distance1) < std::numeric_limits<float>::epsilon() ? 0 : 1;
		Error += glm::abs(Distance2) < std::numeric_limits<float>::epsilon() ? 0 : 1;
		Error += glm::abs(Distance3) < std::numeric_limits<float>::epsilon() ? 0 : 1;
		Error += glm::abs(Distance4) < std::numeric_limits<float>::epsilon() ? 0 : 1;

		return Error;
	}
}//namespace distance

namespace dot
{
	int test()
	{
		float Dot1 = glm::dot(glm::vec1(1), glm::vec1(1));
		float Dot2 = glm::dot(glm::vec2(1), glm::vec2(1));
		float Dot3 = glm::dot(glm::vec3(1), glm::vec3(1));
		float Dot4 = glm::dot(glm::vec4(1), glm::vec4(1));

		int Error = 0;

		Error += glm::abs(Dot1 - 1.0f) < std::numeric_limits<float>::epsilon() ? 0 : 1;
		Error += glm::abs(Dot2 - 2.0f) < std::numeric_limits<float>::epsilon() ? 0 : 1;
		Error += glm::abs(Dot3 - 3.0f) < std::numeric_limits<float>::epsilon() ? 0 : 1;
		Error += glm::abs(Dot4 - 4.0f) < std::numeric_limits<float>::epsilon() ? 0 : 1;

		return Error;
	}
}//namespace dot

namespace cross
{
	int test()
	{
		glm::vec3 Cross1 = glm::cross(glm::vec3(1, 0, 0), glm::vec3(0, 1, 0));
		glm::vec3 Cross2 = glm::cross(glm::vec3(0, 1, 0), glm::vec3(1, 0, 0));

		int Error = 0;

		Error += glm::all(glm::lessThan(glm::abs(Cross1 - glm::vec3(0, 0, 1)), glm::vec3(std::numeric_limits<float>::epsilon()))) ? 0 : 1;
		Error += glm::all(glm::lessThan(glm::abs(Cross2 - glm::vec3(0, 0,-1)), glm::vec3(std::numeric_limits<float>::epsilon()))) ? 0 : 1;

		return Error;
	}
}//namespace cross

namespace normalize
{
	int test()
	{
		glm::vec3 Normalize1 = glm::normalize(glm::vec3(1, 0, 0));
		glm::vec3 Normalize2 = glm::normalize(glm::vec3(2, 0, 0));

		glm::vec3 Normalize3 = glm::normalize(glm::vec3(-0.6, 0.7, -0.5));

		glm::vec3 ro = glm::vec3(glm::cos(5.f) * 3.f, 2.f, glm::sin(5.f) * 3.f);
		glm::vec3 w = glm::normalize(glm::vec3(0, -0.2f, 0) - ro);
		glm::vec3 u = glm::normalize(glm::cross(w, glm::vec3(0, 1, 0)));
		glm::vec3 v = glm::cross(u, w);

		int Error = 0;

		Error += glm::all(glm::lessThan(glm::abs(Normalize1 - glm::vec3(1, 0, 0)), glm::vec3(std::numeric_limits<float>::epsilon()))) ? 0 : 1;
		Error += glm::all(glm::lessThan(glm::abs(Normalize2 - glm::vec3(1, 0, 0)), glm::vec3(std::numeric_limits<float>::epsilon()))) ? 0 : 1;

		return Error;
	}
}//namespace normalize

namespace faceforward
{
	int test()
	{
		int Error = 0;

		{
			glm::vec3 N(0.0f, 0.0f, 1.0f);
			glm::vec3 I(1.0f, 0.0f, 1.0f);
			glm::vec3 Nref(0.0f, 0.0f, 1.0f);
			glm::vec3 F = glm::faceforward(N, I, Nref);
		}

		return Error;
	}
}//namespace faceforward

namespace reflect
{
	int test()
	{
		int Error = 0;

		{
			glm::vec2 A(1.0f,-1.0f);
			glm::vec2 B(0.0f, 1.0f);
			glm::vec2 C = glm::reflect(A, B);
			Error += glm::all(glm::equal(C, glm::vec2(1.0, 1.0), 0.0001f)) ? 0 : 1;
		}

		{
			glm::dvec2 A(1.0f,-1.0f);
			glm::dvec2 B(0.0f, 1.0f);
			glm::dvec2 C = glm::reflect(A, B);
			Error += glm::all(glm::equal(C, glm::dvec2(1.0, 1.0), 0.0001)) ? 0 : 1;
		}

		return Error;
	}
}//namespace reflect

namespace refract
{
	int test()
	{
		int Error = 0;

		{
			float A(-1.0f);
			float B(1.0f);
			float C = glm::refract(A, B, 0.5f);
			Error += glm::equal(C, -1.0f, 0.0001f) ? 0 : 1;
		}

		{
			glm::vec2 A(0.0f,-1.0f);
			glm::vec2 B(0.0f, 1.0f);
			glm::vec2 C = glm::refract(A, B, 0.5f);
			Error += glm::all(glm::equal(C, glm::vec2(0.0, -1.0), 0.0001f)) ? 0 : 1;
		}

		{
			glm::dvec2 A(0.0f,-1.0f);
			glm::dvec2 B(0.0f, 1.0f);
			glm::dvec2 C = glm::refract(A, B, 0.5);
			Error += glm::all(glm::equal(C, glm::dvec2(0.0, -1.0), 0.0001)) ? 0 : 1;
		}

		return Error;
	}
}//namespace refract

int main()
{
	int Error(0);

	Error += length::test();
	Error += distance::test();
	Error += dot::test();
	Error += cross::test();
	Error += normalize::test();
	Error += faceforward::test();
	Error += reflect::test();
	Error += refract::test();

	return Error;
}

