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
/// @file test/gtx/gtx_simd_vec4.cpp
/// @date 2010-09-16 / 2014-11-25
/// @author Christophe Riccio
///////////////////////////////////////////////////////////////////////////////////

#include <glm/glm.hpp>
#include <glm/gtx/simd_vec4.hpp>
#include <cstdio>

#if(GLM_ARCH != GLM_ARCH_PURE)

int main()
{
	glm::simdVec4 A1(0.0f, 0.1f, 0.2f, 0.3f);
	glm::simdVec4 B1(0.4f, 0.5f, 0.6f, 0.7f);
	glm::simdVec4 C1 = A1 + B1;
	glm::simdVec4 D1 = A1.swizzle<glm::X, glm::Z, glm::Y, glm::W>();
	glm::simdVec4 E1(glm::vec4(1.0f));
	glm::vec4 F1 = glm::vec4_cast(E1);
	//glm::vec4 G1(E1);

	//printf("A1(%2.3f, %2.3f, %2.3f, %2.3f)\n", A1.x, A1.y, A1.z, A1.w);
	//printf("B1(%2.3f, %2.3f, %2.3f, %2.3f)\n", B1.x, B1.y, B1.z, B1.w);
	//printf("C1(%2.3f, %2.3f, %2.3f, %2.3f)\n", C1.x, C1.y, C1.z, C1.w);
	//printf("D1(%2.3f, %2.3f, %2.3f, %2.3f)\n", D1.x, D1.y, D1.z, D1.w);

	__m128 value = _mm_set1_ps(0.0f);
	__m128 data = _mm_cmpeq_ps(value, value);
	__m128 add0 = _mm_add_ps(data, data);

	glm::simdVec4 GNI(add0);

	return 0;
}

#else

int main()
{
	int Error = 0;

	return Error;
}

#endif//(GLM_ARCH != GLM_ARCH_PURE)
