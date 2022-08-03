///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2012 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2011-05-31
// Updated : 2011-05-31
// Licence : This source is under MIT licence
// File    : test/gtx/random.cpp
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <glm/glm.hpp>
#include <glm/gtx/random.hpp>
#include <glm/gtx/epsilon.hpp>
#include <iostream>

int test_signedRand1()
{
	int Error = 0;

	{
		float ResultFloat = 0.0f;
		double ResultDouble = 0.0f;
		for(std::size_t i = 0; i < 100000; ++i)
		{
			ResultFloat += glm::signedRand1<float>();
			ResultDouble += glm::signedRand1<double>();
		}

		Error += glm::equalEpsilon(ResultFloat, 0.0f, 0.0001f);
		Error += glm::equalEpsilon(ResultDouble, 0.0, 0.0001);
	}

	return Error;
}

int test_normalizedRand2()
{
	int Error = 0;

	{
		std::size_t Max = 100000;
		float ResultFloat = 0.0f;
		double ResultDouble = 0.0f;
		for(std::size_t i = 0; i < Max; ++i)
		{
			ResultFloat += glm::length(glm::normalizedRand2<float>());
			ResultDouble += glm::length(glm::normalizedRand2<double>());
		}

		Error += glm::equalEpsilon(ResultFloat, float(Max), 0.000001f) ? 0 : 1;
		Error += glm::equalEpsilon(ResultDouble, double(Max), 0.000001) ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int test_normalizedRand3()
{
	int Error = 0;

	{
		std::size_t Max = 100000;
		float ResultFloatA = 0.0f;
		float ResultFloatB = 0.0f;
		float ResultFloatC = 0.0f;
		double ResultDoubleA = 0.0f;
		double ResultDoubleB = 0.0f;
		double ResultDoubleC = 0.0f;
		for(std::size_t i = 0; i < Max; ++i)
		{
			ResultFloatA += glm::length(glm::normalizedRand3<float>());
			ResultDoubleA += glm::length(glm::normalizedRand3<double>());
			ResultFloatB += glm::length(glm::normalizedRand3(2.0f, 2.0f));
			ResultDoubleB += glm::length(glm::normalizedRand3(2.0, 2.0));
			ResultFloatC += glm::length(glm::normalizedRand3(1.0f, 3.0f));
			ResultDoubleC += glm::length(glm::normalizedRand3(1.0, 3.0));
		}

		Error += glm::equalEpsilon(ResultFloatA, float(Max), 0.0001f) ? 0 : 1;
		Error += glm::equalEpsilon(ResultDoubleA, double(Max), 0.0001) ? 0 : 1;
		Error += glm::equalEpsilon(ResultFloatB, float(Max * 2), 0.0001f) ? 0 : 1;
		Error += glm::equalEpsilon(ResultDoubleB, double(Max * 2), 0.0001) ? 0 : 1;
		Error += (ResultFloatC >= float(Max) && ResultFloatC <= float(Max * 3)) ? 0 : 1;
		Error += (ResultDoubleC >= double(Max) && ResultDoubleC <= double(Max * 3)) ? 0 : 1;
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_signedRand1();
	Error += test_normalizedRand2();
	Error += test_normalizedRand3();

	return Error;
}
