///////////////////////////////////////////////////////////////////////////////////
/// OpenGL Mathematics (glm.g-truc.net)
///
/// Copyright (c) 2005 - 2012 G-Truc Creation (www.g-truc.net)
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
/// 
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// 
/// Restrictions:
///		By making use of the Software for military purposes, you choose to make
///		a Bunny unhappy.
/// 
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.
///
/// @file test/gtx/gtx_simd_mat4.cpp
/// @date 2010-09-16 / 2014-11-25
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtx/simd_vec4.hpp>
#include <glm/gtx/simd_mat4.hpp>
#include <cstdio>
#include <ctime>
#include <vector>

#if(GLM_ARCH != GLM_ARCH_PURE)

std::vector<float> test_detA(std::vector<glm::mat4> const & Data)
{
	std::vector<float> Test(Data.size());

	std::clock_t TimeStart = clock();

	for(std::size_t i = 0; i < Test.size() - 1; ++i)
		Test[i] = glm::determinant(Data[i]);

	std::clock_t TimeEnd = clock();
	printf("Det A: %ld\n", TimeEnd - TimeStart);

	return Test;
}

std::vector<float> test_detB(std::vector<glm::mat4> const & Data)
{
	std::vector<float> Test(Data.size());

	std::clock_t TimeStart = clock();

	for(std::size_t i = 0; i < Test.size() - 1; ++i)
	{
		_mm_prefetch((char*)&Data[i + 1], _MM_HINT_T0);
		glm::simdMat4 m(Data[i]);
		glm::simdVec4 d(glm::detail::sse_slow_det_ps((__m128 const * const)&m)); 
		glm::vec4 v;//(d);
		Test[i] = v.x;
	}

	std::clock_t TimeEnd = clock();
	printf("Det B: %ld\n", TimeEnd - TimeStart);

	return Test;
}

std::vector<float> test_detC(std::vector<glm::mat4> const & Data)
{
	std::vector<float> Test(Data.size());

	std::clock_t TimeStart = clock();

	for(std::size_t i = 0; i < Test.size() - 1; ++i)
	{
		_mm_prefetch((char*)&Data[i + 1], _MM_HINT_T0);
		glm::simdMat4 m(Data[i]);
		glm::simdVec4 d(glm::detail::sse_det_ps((__m128 const * const)&m));
		glm::vec4 v;//(d);
		Test[i] = v.x;
	}

	std::clock_t TimeEnd = clock();
	printf("Det C: %ld\n", TimeEnd - TimeStart);

	return Test;
}

std::vector<float> test_detD(std::vector<glm::mat4> const & Data)
{
	std::vector<float> Test(Data.size());

	std::clock_t TimeStart = clock();

	for(std::size_t i = 0; i < Test.size() - 1; ++i)
	{
		_mm_prefetch((char*)&Data[i + 1], _MM_HINT_T0);
		glm::simdMat4 m(Data[i]);
		glm::simdVec4 d(glm::detail::sse_detd_ps((__m128 const * const)&m));
		glm::vec4 v;//(d); 
		Test[i] = v.x;
	}

	std::clock_t TimeEnd = clock();
	printf("Det D: %ld\n", TimeEnd - TimeStart);

	return Test;
}

void test_invA(std::vector<glm::mat4> const & Data, std::vector<glm::mat4> & Out)
{
	//std::vector<float> Test(Data.size());
	Out.resize(Data.size());

	std::clock_t TimeStart = clock();

	for(std::size_t i = 0; i < Out.size() - 1; ++i)
	{
		Out[i] = glm::inverse(Data[i]);
	}

	std::clock_t TimeEnd = clock();
	printf("Inv A: %ld\n", TimeEnd - TimeStart);
}

void test_invC(std::vector<glm::mat4> const & Data, std::vector<glm::mat4> & Out)
{
	//std::vector<float> Test(Data.size());
	Out.resize(Data.size());

	std::clock_t TimeStart = clock();

	for(std::size_t i = 0; i < Out.size() - 1; ++i)
	{
		_mm_prefetch((char*)&Data[i + 1], _MM_HINT_T0);
		glm::simdMat4 m(Data[i]);
		glm::simdMat4 o;
		glm::detail::sse_inverse_fast_ps((__m128 const * const)&m, (__m128 *)&o);
		Out[i] = *(glm::mat4*)&o;
	}

	std::clock_t TimeEnd = clock();
	printf("Inv C: %ld\n", TimeEnd - TimeStart);
}

void test_invD(std::vector<glm::mat4> const & Data, std::vector<glm::mat4> & Out)
{
	//std::vector<float> Test(Data.size());
	Out.resize(Data.size());

	std::clock_t TimeStart = clock();

	for(std::size_t i = 0; i < Out.size() - 1; ++i)
	{
		_mm_prefetch((char*)&Data[i + 1], _MM_HINT_T0);
		glm::simdMat4 m(Data[i]);
		glm::simdMat4 o;
		glm::detail::sse_inverse_ps((__m128 const * const)&m, (__m128 *)&o);
		Out[i] = *(glm::mat4*)&o;
	}

	std::clock_t TimeEnd = clock();
	printf("Inv D: %ld\n", TimeEnd - TimeStart);
}

void test_mulA(std::vector<glm::mat4> const & Data, std::vector<glm::mat4> & Out)
{
	//std::vector<float> Test(Data.size());
	Out.resize(Data.size());

	std::clock_t TimeStart = clock();

	for(std::size_t i = 0; i < Out.size() - 1; ++i)
	{
		Out[i] = Data[i] * Data[i];
	}

	std::clock_t TimeEnd = clock();
	printf("Mul A: %ld\n", TimeEnd - TimeStart);
}

void test_mulD(std::vector<glm::mat4> const & Data, std::vector<glm::mat4> & Out)
{
	//std::vector<float> Test(Data.size());
	Out.resize(Data.size());

	std::clock_t TimeStart = clock();

	for(std::size_t i = 0; i < Out.size() - 1; ++i)
	{
		_mm_prefetch((char*)&Data[i + 1], _MM_HINT_T0);
		glm::simdMat4 m(Data[i]);
		glm::simdMat4 o;
		glm::detail::sse_mul_ps((__m128 const * const)&m, (__m128 const * const)&m, (__m128*)&o);
		Out[i] = *(glm::mat4*)&o;
	}

	std::clock_t TimeEnd = clock();
	printf("Mul D: %ld\n", TimeEnd - TimeStart);
}

int test_compute_glm()
{
	return 0;
}

int test_compute_gtx()
{
	std::vector<glm::vec4> Output(1000000);

	std::clock_t TimeStart = clock();

	for(std::size_t k = 0; k < Output.size(); ++k)
	{
		float i = float(k) / 1000.f + 0.001f;
		glm::vec3 A = glm::normalize(glm::vec3(i));
		glm::vec3 B = glm::cross(A, glm::normalize(glm::vec3(1, 1, 2)));
		glm::mat4 C = glm::rotate(glm::mat4(1.0f), i, B);
		glm::mat4 D = glm::scale(C, glm::vec3(0.8f, 1.0f, 1.2f));
		glm::mat4 E = glm::translate(D, glm::vec3(1.4f, 1.2f, 1.1f));
		glm::mat4 F = glm::perspective(i, 1.5f, 0.1f, 1000.f);
		glm::mat4 G = glm::inverse(F * E);
		glm::vec3 H = glm::unProject(glm::vec3(i), G, F, E[3]);
		glm::vec3 I = glm::any(glm::isnan(glm::project(H, G, F, E[3]))) ? glm::vec3(2) : glm::vec3(1);
		glm::mat4 J = glm::lookAt(glm::normalize(glm::max(B, glm::vec3(0.001f))), H, I);
		glm::mat4 K = glm::transpose(J);
		glm::quat L = glm::normalize(glm::quat_cast(K));
		glm::vec4 M = L * glm::smoothstep(K[3], J[3], glm::vec4(i));
		glm::mat4 N = glm::mat4(glm::normalize(glm::max(M, glm::vec4(0.001f))), K[3], J[3], glm::vec4(i));
		glm::mat4 O = N * glm::inverse(N);
		glm::vec4 P = O * glm::reflect(N[3], glm::vec4(A, 1.0f));
		glm::vec4 Q = glm::vec4(glm::dot(M, P));
		glm::vec4 R = glm::quat(Q.w, glm::vec3(Q)) * P;
		Output[k] = R;
	}

	std::clock_t TimeEnd = clock();
	printf("test_compute_gtx: %ld\n", TimeEnd - TimeStart);

	return 0;
}

int main()
{
	int Error = 0;

	std::vector<glm::mat4> Data(64 * 64 * 1);
	for(std::size_t i = 0; i < Data.size(); ++i)
		Data[i] = glm::mat4(
			glm::vec4(glm::linearRand(glm::vec4(-2.0f), glm::vec4(2.0f))),
			glm::vec4(glm::linearRand(glm::vec4(-2.0f), glm::vec4(2.0f))),
			glm::vec4(glm::linearRand(glm::vec4(-2.0f), glm::vec4(2.0f))),
			glm::vec4(glm::linearRand(glm::vec4(-2.0f), glm::vec4(2.0f))));

	{
		std::vector<glm::mat4> TestInvA;
		test_invA(Data, TestInvA);
	}
	{
		std::vector<glm::mat4> TestInvC;
		test_invC(Data, TestInvC);
	}
	{
		std::vector<glm::mat4> TestInvD;
		test_invD(Data, TestInvD);
	}

	{
		std::vector<glm::mat4> TestA;
		test_mulA(Data, TestA);
	}
	{
		std::vector<glm::mat4> TestD;
		test_mulD(Data, TestD);
	}

	{
		std::vector<float> TestDetA = test_detA(Data);
		std::vector<float> TestDetB = test_detB(Data);
		std::vector<float> TestDetD = test_detD(Data);
		std::vector<float> TestDetC = test_detC(Data);

		for(std::size_t i = 0; i < TestDetA.size(); ++i)
			if(TestDetA[i] != TestDetB[i] && TestDetC[i] != TestDetB[i] && TestDetC[i] != TestDetD[i])
				return 1;
	}

	// shuffle test
	glm::simdVec4 A(1.0f, 2.0f, 3.0f, 4.0f);
	glm::simdVec4 B(5.0f, 6.0f, 7.0f, 8.0f);
	//__m128 C = _mm_shuffle_ps(A.Data, B.Data, _MM_SHUFFLE(1, 0, 1, 0));

	Error += test_compute_glm();
	Error += test_compute_gtx();
	float Det = glm::determinant(glm::simdMat4(1.0));
	Error += Det == 1.0f ? 0 : 1;
	
	glm::simdMat4 D = glm::matrixCompMult(glm::simdMat4(1.0), glm::simdMat4(1.0));

	return Error;
}

#else

int main()
{
	int Error = 0;

	return Error;
}

#endif//(GLM_ARCH != GLM_ARCH_PURE)
