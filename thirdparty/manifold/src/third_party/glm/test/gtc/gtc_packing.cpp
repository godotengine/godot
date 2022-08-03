#include <glm/packing.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/ext/vector_relational.hpp>
#include <cstdio>
#include <vector>

void print_bits(float const& s)
{
	union
	{
		float f;
		unsigned int i;
	} uif;

	uif.f = s;

	std::printf("f32: ");
	for(std::size_t j = sizeof(s) * 8; j > 0; --j)
	{
		if(j == 23 || j == 31)
			std::printf(" ");
		std::printf("%d", (uif.i & (1 << (j - 1))) ? 1 : 0);
	}
}

void print_10bits(glm::uint const& s)
{
	std::printf("10b: ");
	for(std::size_t j = 10; j > 0; --j)
	{
		if(j == 5)
			std::printf(" ");
		std::printf("%d", (s & (1 << (j - 1))) ? 1 : 0);
	}
}

void print_11bits(glm::uint const& s)
{
	std::printf("11b: ");
	for(std::size_t j = 11; j > 0; --j)
	{
		if(j == 6)
			std::printf(" ");
		std::printf("%d", (s & (1 << (j - 1))) ? 1 : 0);
	}
}

void print_value(float const& s)
{
	std::printf("%2.5f, ", static_cast<double>(s));
	print_bits(s);
	std::printf(", ");
//	print_11bits(detail::floatTo11bit(s));
//	std::printf(", ");
//	print_10bits(detail::floatTo10bit(s));
	std::printf("\n");
}

int test_Half1x16()
{
	int Error = 0;

	std::vector<float> Tests;
	Tests.push_back(0.0f);
	Tests.push_back(1.0f);
	Tests.push_back(-1.0f);
	Tests.push_back(2.0f);
	Tests.push_back(-2.0f);
	Tests.push_back(1.9f);

	for(std::size_t i = 0; i < Tests.size(); ++i)
	{
		glm::uint16 p0 = glm::packHalf1x16(Tests[i]);
		float v0 = glm::unpackHalf1x16(p0);
		glm::uint16 p1 = glm::packHalf1x16(v0);
		float v1 = glm::unpackHalf1x16(p1);
		Error += glm::epsilonEqual(v0, v1, glm::epsilon<float>()) ? 0 : 1;
	}

	return Error;
}

int test_Half4x16()
{
	int Error = 0;

	std::vector<glm::vec4> Tests;
	Tests.push_back(glm::vec4(1.0f));
	Tests.push_back(glm::vec4(0.0f));
	Tests.push_back(glm::vec4(2.0f));
	Tests.push_back(glm::vec4(0.1f));
	Tests.push_back(glm::vec4(0.5f));
	Tests.push_back(glm::vec4(-0.9f));

	for(std::size_t i = 0; i < Tests.size(); ++i)
	{
		glm::uint64 p0 = glm::packHalf4x16(Tests[i]);
		glm::vec4 v0 = glm::unpackHalf4x16(p0);
		glm::uint64 p1 = glm::packHalf4x16(v0);
		glm::vec4 v1 = glm::unpackHalf4x16(p1);
		glm::u16vec4 p2 = glm::packHalf(v0);
		glm::vec4 v2 = glm::unpackHalf(p2);

		Error += glm::all(glm::equal(v0, v1, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(v0, v2, glm::epsilon<float>())) ? 0 : 1;
	}

	return Error;
}

int test_I3x10_1x2()
{
	int Error = 0;

	std::vector<glm::ivec4> Tests;
	Tests.push_back(glm::ivec4(0));
	Tests.push_back(glm::ivec4(1));
	Tests.push_back(glm::ivec4(-1));
	Tests.push_back(glm::ivec4(2));
	Tests.push_back(glm::ivec4(-2));
	Tests.push_back(glm::ivec4(3));

	for(std::size_t i = 0; i < Tests.size(); ++i)
	{
		glm::uint32 p0 = glm::packI3x10_1x2(Tests[i]);
		glm::ivec4 v0 = glm::unpackI3x10_1x2(p0);
		glm::uint32 p1 = glm::packI3x10_1x2(v0);
		glm::ivec4 v1 = glm::unpackI3x10_1x2(p1);
		Error += glm::all(glm::equal(v0, v1)) ? 0 : 1;
	}

	return Error;
}

int test_U3x10_1x2()
{
	int Error = 0;

	std::vector<glm::uvec4> Tests;
	Tests.push_back(glm::uvec4(0));
	Tests.push_back(glm::uvec4(1));
	Tests.push_back(glm::uvec4(2));
	Tests.push_back(glm::uvec4(3));
	Tests.push_back(glm::uvec4(4));
	Tests.push_back(glm::uvec4(5));

	for(std::size_t i = 0; i < Tests.size(); ++i)
	{
		glm::uint32 p0 = glm::packU3x10_1x2(Tests[i]);
		glm::uvec4 v0 = glm::unpackU3x10_1x2(p0);
		glm::uint32 p1 = glm::packU3x10_1x2(v0);
		glm::uvec4 v1 = glm::unpackU3x10_1x2(p1);
		Error += glm::all(glm::equal(v0, v1)) ? 0 : 1;
	}

	glm::u8vec4 const v0(0xff, 0x77, 0x0, 0x33);
	glm::uint32 const p0 = *reinterpret_cast<glm::uint32 const*>(&v0[0]);
	glm::uint32 const r0 = 0x330077ff;

	Error += p0 == r0 ? 0 : 1;

	glm::uvec4 const v1(0xff, 0x77, 0x0, 0x33);
	glm::uint32 const p1 = glm::packU3x10_1x2(v1);
	glm::uint32 const r1 = 0xc001dcff;

	Error += p1 == r1 ? 0 : 1;

	return Error;
}

int test_Snorm3x10_1x2()
{
	int Error = 0;

	std::vector<glm::vec4> Tests;
	Tests.push_back(glm::vec4(1.0f));
	Tests.push_back(glm::vec4(0.0f));
	Tests.push_back(glm::vec4(2.0f));
	Tests.push_back(glm::vec4(0.1f));
	Tests.push_back(glm::vec4(0.5f));
	Tests.push_back(glm::vec4(0.9f));

	for(std::size_t i = 0; i < Tests.size(); ++i)
	{
		glm::uint32 p0 = glm::packSnorm3x10_1x2(Tests[i]);
		glm::vec4 v0 = glm::unpackSnorm3x10_1x2(p0);
		glm::uint32 p1 = glm::packSnorm3x10_1x2(v0);
		glm::vec4 v1 = glm::unpackSnorm3x10_1x2(p1);

		Error += glm::all(glm::epsilonEqual(v0, v1, 0.01f)) ? 0 : 1;
	}

	return Error;
}

int test_Unorm3x10_1x2()
{
	int Error = 0;

	std::vector<glm::vec4> Tests;
	Tests.push_back(glm::vec4(1.0f));
	Tests.push_back(glm::vec4(0.0f));
	Tests.push_back(glm::vec4(2.0f));
	Tests.push_back(glm::vec4(0.1f));
	Tests.push_back(glm::vec4(0.5f));
	Tests.push_back(glm::vec4(0.9f));

	for(std::size_t i = 0; i < Tests.size(); ++i)
	{
		glm::uint32 p0 = glm::packUnorm3x10_1x2(Tests[i]);
		glm::vec4 v0 = glm::unpackUnorm3x10_1x2(p0);
		glm::uint32 p1 = glm::packUnorm3x10_1x2(v0);
		glm::vec4 v1 = glm::unpackUnorm3x10_1x2(p1);

		Error += glm::all(glm::epsilonEqual(v0, v1, 0.001f)) ? 0 : 1;
	}

	return Error;
}

int test_F2x11_1x10()
{
	int Error = 0;

	std::vector<glm::vec3> Tests;
	Tests.push_back(glm::vec3(1.0f));
	Tests.push_back(glm::vec3(0.0f));
	Tests.push_back(glm::vec3(2.0f));
	Tests.push_back(glm::vec3(0.1f));
	Tests.push_back(glm::vec3(0.5f));
	Tests.push_back(glm::vec3(0.9f));

	for(std::size_t i = 0; i < Tests.size(); ++i)
	{
		glm::uint32 p0 = glm::packF2x11_1x10(Tests[i]);
		glm::vec3 v0 = glm::unpackF2x11_1x10(p0);
		glm::uint32 p1 = glm::packF2x11_1x10(v0);
		glm::vec3 v1 = glm::unpackF2x11_1x10(p1);
		Error += glm::all(glm::equal(v0, v1, glm::epsilon<float>())) ? 0 : 1;
	}

	return Error;
}

int test_F3x9_E1x5()
{
	int Error = 0;

	std::vector<glm::vec3> Tests;
	Tests.push_back(glm::vec3(1.0f));
	Tests.push_back(glm::vec3(0.0f));
	Tests.push_back(glm::vec3(2.0f));
	Tests.push_back(glm::vec3(0.1f));
	Tests.push_back(glm::vec3(0.5f));
	Tests.push_back(glm::vec3(0.9f));

	for(std::size_t i = 0; i < Tests.size(); ++i)
	{
		glm::uint32 p0 = glm::packF3x9_E1x5(Tests[i]);
		glm::vec3 v0 = glm::unpackF3x9_E1x5(p0);
		glm::uint32 p1 = glm::packF3x9_E1x5(v0);
		glm::vec3 v1 = glm::unpackF3x9_E1x5(p1);
		Error += glm::all(glm::equal(v0, v1, glm::epsilon<float>())) ? 0 : 1;
	}

	return Error;
}

int test_RGBM()
{
	int Error = 0;

	for(std::size_t i = 0; i < 1024; ++i)
	{
		glm::vec3 const Color(static_cast<float>(i));
		glm::vec4 const RGBM = glm::packRGBM(Color);
		glm::vec3 const Result= glm::unpackRGBM(RGBM);

		Error += glm::all(glm::equal(Color, Result, 0.01f)) ? 0 : 1;
	}

	return Error;
}

int test_packUnorm1x16()
{
	int Error = 0;

	std::vector<glm::vec1> A;
	A.push_back(glm::vec1(1.0f));
	A.push_back(glm::vec1(0.5f));
	A.push_back(glm::vec1(0.1f));
	A.push_back(glm::vec1(0.0f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec1 B(A[i]);
		glm::uint16 C = glm::packUnorm1x16(B.x);
		glm::vec1 D(glm::unpackUnorm1x16(C));
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 65535.f)) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

int test_packSnorm1x16()
{
	int Error = 0;

	std::vector<glm::vec1> A;
	A.push_back(glm::vec1( 1.0f));
	A.push_back(glm::vec1( 0.0f));
	A.push_back(glm::vec1(-0.5f));
	A.push_back(glm::vec1(-0.1f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec1 B(A[i]);
		glm::uint16 C = glm::packSnorm1x16(B.x);
		glm::vec1 D(glm::unpackSnorm1x16(C));
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 32767.0f * 2.0f)) ? 0 : 1;
	}
	
	return Error;
}

int test_packUnorm2x16()
{
	int Error = 0;

	std::vector<glm::vec2> A;
	A.push_back(glm::vec2(1.0f, 0.0f));
	A.push_back(glm::vec2(0.5f, 0.7f));
	A.push_back(glm::vec2(0.1f, 0.2f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec2 B(A[i]);
		glm::uint32 C = glm::packUnorm2x16(B);
		glm::vec2 D = glm::unpackUnorm2x16(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 65535.f)) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

int test_packSnorm2x16()
{
	int Error = 0;

	std::vector<glm::vec2> A;
	A.push_back(glm::vec2( 1.0f, 0.0f));
	A.push_back(glm::vec2(-0.5f,-0.7f));
	A.push_back(glm::vec2(-0.1f, 0.1f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec2 B(A[i]);
		glm::uint32 C = glm::packSnorm2x16(B);
		glm::vec2 D = glm::unpackSnorm2x16(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 32767.0f * 2.0f)) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

int test_packUnorm4x16()
{
	int Error = 0;

	std::vector<glm::vec4> A;
	A.push_back(glm::vec4(1.0f));
	A.push_back(glm::vec4(0.5f));
	A.push_back(glm::vec4(0.1f));
	A.push_back(glm::vec4(0.0f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec4 B(A[i]);
		glm::uint64 C = glm::packUnorm4x16(B);
		glm::vec4 D(glm::unpackUnorm4x16(C));
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 65535.f)) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

int test_packSnorm4x16()
{
	int Error = 0;

	std::vector<glm::vec4> A;
	A.push_back(glm::vec4( 1.0f, 0.0f, -0.5f, 0.5f));
	A.push_back(glm::vec4(-0.3f,-0.7f,  0.3f, 0.7f));
	A.push_back(glm::vec4(-0.1f, 0.1f, -0.2f, 0.2f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec4 B(A[i]);
		glm::uint64 C = glm::packSnorm4x16(B);
		glm::vec4 D(glm::unpackSnorm4x16(C));
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 32767.0f * 2.0f)) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

int test_packUnorm1x8()
{
	int Error = 0;
	
	std::vector<glm::vec1> A;
	A.push_back(glm::vec1(1.0f));
	A.push_back(glm::vec1(0.5f));
	A.push_back(glm::vec1(0.0f));
	
	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec1 B(A[i]);
		glm::uint8 C = glm::packUnorm1x8(B.x);
		glm::vec1 D(glm::unpackUnorm1x8(C));
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 255.f)) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

int test_packSnorm1x8()
{
	int Error = 0;
	
	std::vector<glm::vec1> A;
	A.push_back(glm::vec1( 1.0f));
	A.push_back(glm::vec1(-0.7f));
	A.push_back(glm::vec1(-1.0f));
	
	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec1 B(A[i]);
		glm::uint8 C = glm::packSnorm1x8(B.x);
		glm::vec1 D(glm::unpackSnorm1x8(C));
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 127.f)) ? 0 : 1;
	}
	
	return Error;
}

int test_packUnorm2x8()
{
	int Error = 0;
	
	std::vector<glm::vec2> A;
	A.push_back(glm::vec2(1.0f, 0.7f));
	A.push_back(glm::vec2(0.5f, 0.1f));
	
	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec2 B(A[i]);
		glm::uint16 C = glm::packUnorm2x8(B);
		glm::vec2 D = glm::unpackUnorm2x8(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 255.f)) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

int test_packSnorm2x8()
{
	int Error = 0;
	
	std::vector<glm::vec2> A;
	A.push_back(glm::vec2( 1.0f, 0.0f));
	A.push_back(glm::vec2(-0.7f,-0.1f));
	
	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec2 B(A[i]);
		glm::uint16 C = glm::packSnorm2x8(B);
		glm::vec2 D = glm::unpackSnorm2x8(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 127.f)) ? 0 : 1;
	}
	
	return Error;
}

int test_packUnorm4x8()
{
	int Error = 0;
	
	std::vector<glm::vec4> A;
	A.push_back(glm::vec4(1.0f, 0.7f, 0.3f, 0.0f));
	A.push_back(glm::vec4(0.5f, 0.1f, 0.2f, 0.3f));
	
	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec4 B(A[i]);
		glm::uint32 C = glm::packUnorm4x8(B);
		glm::vec4 D = glm::unpackUnorm4x8(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 255.f)) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

int test_packSnorm4x8()
{
	int Error = 0;
	
	std::vector<glm::vec4> A;
	A.push_back(glm::vec4( 1.0f, 0.0f,-0.5f,-1.0f));
	A.push_back(glm::vec4(-0.7f,-0.1f, 0.1f, 0.7f));
	
	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec4 B(A[i]);
		glm::uint32 C = glm::packSnorm4x8(B);
		glm::vec4 D = glm::unpackSnorm4x8(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 127.f)) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

int test_packUnorm()
{
	int Error = 0;

	std::vector<glm::vec2> A;
	A.push_back(glm::vec2(1.0f, 0.7f));
	A.push_back(glm::vec2(0.5f, 0.1f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec2 B(A[i]);
		glm::u16vec2 C = glm::packUnorm<glm::uint16>(B);
		glm::vec2 D = glm::unpackUnorm<float>(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 255.f)) ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int test_packSnorm()
{
	int Error = 0;

	std::vector<glm::vec2> A;
	A.push_back(glm::vec2( 1.0f, 0.0f));
	A.push_back(glm::vec2(-0.5f,-0.7f));
	A.push_back(glm::vec2(-0.1f, 0.1f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec2 B(A[i]);
		glm::i16vec2 C = glm::packSnorm<glm::int16>(B);
		glm::vec2 D = glm::unpackSnorm<float>(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 32767.0f * 2.0f)) ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int test_packUnorm2x4()
{
	int Error = 0;

	std::vector<glm::vec2> A;
	A.push_back(glm::vec2(1.0f, 0.7f));
	A.push_back(glm::vec2(0.5f, 0.0f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec2 B(A[i]);
		glm::uint8 C = glm::packUnorm2x4(B);
		glm::vec2 D = glm::unpackUnorm2x4(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 15.f)) ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int test_packUnorm4x4()
{
	int Error = 0;

	std::vector<glm::vec4> A;
	A.push_back(glm::vec4(1.0f, 0.7f, 0.5f, 0.0f));
	A.push_back(glm::vec4(0.5f, 0.1f, 0.0f, 1.0f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec4 B(A[i]);
		glm::uint16 C = glm::packUnorm4x4(B);
		glm::vec4 D = glm::unpackUnorm4x4(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 15.f)) ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int test_packUnorm3x5_1x1()
{
	int Error = 0;

	std::vector<glm::vec4> A;
	A.push_back(glm::vec4(1.0f, 0.7f, 0.5f, 0.0f));
	A.push_back(glm::vec4(0.5f, 0.1f, 0.0f, 1.0f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec4 B(A[i]);
		glm::uint16 C = glm::packUnorm3x5_1x1(B);
		glm::vec4 D = glm::unpackUnorm3x5_1x1(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 15.f)) ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int test_packUnorm1x5_1x6_1x5()
{
	int Error = 0;

	std::vector<glm::vec3> A;
	A.push_back(glm::vec3(1.0f, 0.7f, 0.5f));
	A.push_back(glm::vec3(0.5f, 0.1f, 0.0f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec3 B(A[i]);
		glm::uint16 C = glm::packUnorm1x5_1x6_1x5(B);
		glm::vec3 D = glm::unpackUnorm1x5_1x6_1x5(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 15.f)) ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int test_packUnorm2x3_1x2()
{
	int Error = 0;

	std::vector<glm::vec3> A;
	A.push_back(glm::vec3(1.0f, 0.7f, 0.5f));
	A.push_back(glm::vec3(0.5f, 0.1f, 0.0f));

	for(std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec3 B(A[i]);
		glm::uint8 C = glm::packUnorm2x3_1x2(B);
		glm::vec3 D = glm::unpackUnorm2x3_1x2(C);
		Error += glm::all(glm::epsilonEqual(B, D, 1.0f / 3.f)) ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int test_packUint2x8()
{
	int Error = 0;

	glm::u8vec2 const Source(1, 2);

	glm::uint16 const Packed = glm::packUint2x8(Source);
	Error += Packed != 0 ? 0 : 1;

	glm::u8vec2 const Unpacked = glm::unpackUint2x8(Packed);
	Error += Source == Unpacked ? 0 : 1;

	return Error;
}

int test_packUint4x8()
{
	int Error = 0;

	glm::u8vec4 const Source(1, 2, 3, 4);

	glm::uint32 const Packed = glm::packUint4x8(Source);
	Error += Packed != 0 ? 0 : 1;

	glm::u8vec4 const Unpacked = glm::unpackUint4x8(Packed);
	Error += Source == Unpacked ? 0 : 1;

	return Error;
}

int test_packUint2x16()
{
	int Error = 0;

	glm::u16vec2 const Source(1, 2);

	glm::uint32 const Packed = glm::packUint2x16(Source);
	Error += Packed != 0 ? 0 : 1;

	glm::u16vec2 const Unpacked = glm::unpackUint2x16(Packed);
	Error += Source == Unpacked ? 0 : 1;

	return Error;
}

int test_packUint4x16()
{
	int Error = 0;

	glm::u16vec4 const Source(1, 2, 3, 4);

	glm::uint64 const Packed = glm::packUint4x16(Source);
	Error += Packed != 0 ? 0 : 1;

	glm::u16vec4 const Unpacked = glm::unpackUint4x16(Packed);
	Error += Source == Unpacked ? 0 : 1;

	return Error;
}

int test_packUint2x32()
{
	int Error = 0;

	glm::u32vec2 const Source(1, 2);

	glm::uint64 const Packed = glm::packUint2x32(Source);
	Error += Packed != 0 ? 0 : 1;

	glm::u32vec2 const Unpacked = glm::unpackUint2x32(Packed);
	Error += Source == Unpacked ? 0 : 1;

	return Error;
}

int test_packInt2x8()
{
	int Error = 0;

	glm::i8vec2 const Source(1, 2);

	glm::int16 const Packed = glm::packInt2x8(Source);
	Error += Packed != 0 ? 0 : 1;

	glm::i8vec2 const Unpacked = glm::unpackInt2x8(Packed);
	Error += Source == Unpacked ? 0 : 1;

	return Error;
}

int test_packInt4x8()
{
	int Error = 0;

	glm::i8vec4 const Source(1, 2, 3, 4);

	glm::int32 const Packed = glm::packInt4x8(Source);
	Error += Packed != 0 ? 0 : 1;

	glm::i8vec4 const Unpacked = glm::unpackInt4x8(Packed);
	Error += Source == Unpacked ? 0 : 1;

	return Error;
}

int test_packInt2x16()
{
	int Error = 0;

	glm::i16vec2 const Source(1, 2);

	glm::int32 const Packed = glm::packInt2x16(Source);
	Error += Packed != 0 ? 0 : 1;

	glm::i16vec2 const Unpacked = glm::unpackInt2x16(Packed);
	Error += Source == Unpacked ? 0 : 1;

	return Error;
}

int test_packInt4x16()
{
	int Error = 0;

	glm::i16vec4 const Source(1, 2, 3, 4);

	glm::int64 const Packed = glm::packInt4x16(Source);
	Error += Packed != 0 ? 0 : 1;

	glm::i16vec4 const Unpacked = glm::unpackInt4x16(Packed);
	Error += Source == Unpacked ? 0 : 1;

	return Error;
}

int test_packInt2x32()
{
	int Error = 0;

	glm::i32vec2 const Source(1, 2);

	glm::int64 const Packed = glm::packInt2x32(Source);
	Error += Packed != 0 ? 0 : 1;

	glm::i32vec2 const Unpacked = glm::unpackInt2x32(Packed);
	Error += Source == Unpacked ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_packUnorm();
	Error += test_packSnorm();

	Error += test_packSnorm1x16();
	Error += test_packSnorm2x16();
	Error += test_packSnorm4x16();

	Error += test_packSnorm1x8();
	Error += test_packSnorm2x8();
	Error += test_packSnorm4x8();

	Error += test_packUnorm1x16();
	Error += test_packUnorm2x16();
	Error += test_packUnorm4x16();

	Error += test_packUnorm1x8();
	Error += test_packUnorm2x8();
	Error += test_packUnorm4x8();

	Error += test_packUnorm2x4();
	Error += test_packUnorm4x4();
	Error += test_packUnorm3x5_1x1();
	Error += test_packUnorm1x5_1x6_1x5();
	Error += test_packUnorm2x3_1x2();

	Error += test_packUint2x8();
	Error += test_packUint4x8();
	Error += test_packUint2x16();
	Error += test_packUint4x16();
	Error += test_packUint2x32();

	Error += test_packInt2x8();
	Error += test_packInt4x8();
	Error += test_packInt2x16();
	Error += test_packInt4x16();
	Error += test_packInt2x32();

	Error += test_F2x11_1x10();
	Error += test_F3x9_E1x5();
	Error += test_RGBM();
	Error += test_Unorm3x10_1x2();
	Error += test_Snorm3x10_1x2();

	Error += test_I3x10_1x2();
	Error += test_U3x10_1x2();
	Error += test_Half1x16();
	Error += test_Half4x16();

	return Error;
}
