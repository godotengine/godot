#pragma once
#include <random>
namespace Mtree
{
	class RandomGenerator
	{
	private:
		std::mt19937 engine;
		std::uniform_real_distribution<float> uniform;
	public:
		RandomGenerator() { uniform = std::uniform_real_distribution<float>{ 0.0f,1.0f }; };
		void set_seed(int seed) { engine = std::mt19937{ (unsigned int) seed }; srand(seed); };
		float get_0_1() { return uniform(engine); };
	};
}