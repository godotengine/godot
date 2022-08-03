#define GLM_FORCE_MESSAGES
#include <glm/vec3.hpp>
#include <cstdio>

int test_compiler()
{
	int Error(0);
	
	if(GLM_COMPILER & GLM_COMPILER_VC)
	{
		switch(GLM_COMPILER)
		{
		case GLM_COMPILER_VC12:
			std::printf("Visual C++ 12 - 2013\n");
			break;
		case GLM_COMPILER_VC14:
			std::printf("Visual C++ 14 - 2015\n");
			break;
		case GLM_COMPILER_VC15:
			std::printf("Visual C++ 15 - 2017\n");
			break;
		case GLM_COMPILER_VC15_3:
			std::printf("Visual C++ 15.3 - 2017\n");
			break;
		case GLM_COMPILER_VC15_5:
			std::printf("Visual C++ 15.5 - 2017\n");
			break;
		case GLM_COMPILER_VC15_6:
			std::printf("Visual C++ 15.6 - 2017\n");
			break;
		case GLM_COMPILER_VC15_7:
			std::printf("Visual C++ 15.7 - 2017\n");
			break;
		case GLM_COMPILER_VC15_8:
			std::printf("Visual C++ 15.8 - 2017\n");
			break;
		case GLM_COMPILER_VC15_9:
			std::printf("Visual C++ 15.9 - 2017\n");
			break;
		case GLM_COMPILER_VC16:
			std::printf("Visual C++ 16 - 2019\n");
			break;
		default:
			std::printf("Visual C++ version not detected\n");
			Error += 1;
			break;
		}
	}
	else if(GLM_COMPILER & GLM_COMPILER_GCC)
	{
		switch(GLM_COMPILER)
		{
		case GLM_COMPILER_GCC46:
			std::printf("GCC 4.6\n");
			break;
		case GLM_COMPILER_GCC47:
			std::printf("GCC 4.7\n");
			break;
		case GLM_COMPILER_GCC48:
			std::printf("GCC 4.8\n");
			break;
		case GLM_COMPILER_GCC49:
			std::printf("GCC 4.9\n");
			break;
		case GLM_COMPILER_GCC5:
			std::printf("GCC 5\n");
			break;
		case GLM_COMPILER_GCC6:
			std::printf("GCC 6\n");
			break;
		case GLM_COMPILER_GCC7:
			std::printf("GCC 7\n");
			break;
		case GLM_COMPILER_GCC8:
			std::printf("GCC 8\n");
			break;
		default:
			std::printf("GCC version not detected\n");
			Error += 1;
			break;
		}
	}
	else if(GLM_COMPILER & GLM_COMPILER_CUDA)
	{
		std::printf("CUDA\n");
	}
	else if(GLM_COMPILER & GLM_COMPILER_CLANG)
	{
		switch(GLM_COMPILER)
		{
		case GLM_COMPILER_CLANG34:
			std::printf("Clang 3.4\n");
			break;
		case GLM_COMPILER_CLANG35:
			std::printf("Clang 3.5\n");
			break;
		case GLM_COMPILER_CLANG36:
			std::printf("Clang 3.6\n");
			break;
		case GLM_COMPILER_CLANG37:
			std::printf("Clang 3.7\n");
			break;
		case GLM_COMPILER_CLANG38:
			std::printf("Clang 3.8\n");
			break;
		case GLM_COMPILER_CLANG39:
			std::printf("Clang 3.9\n");
			break;
		case GLM_COMPILER_CLANG40:
			std::printf("Clang 4.0\n");
			break;
		case GLM_COMPILER_CLANG41:
			std::printf("Clang 4.1\n");
			break;
		case GLM_COMPILER_CLANG42:
			std::printf("Clang 4.2\n");
			break;
		default:
			std::printf("LLVM version not detected\n");
			break;
		}
	}
	else if(GLM_COMPILER & GLM_COMPILER_INTEL)
	{
		switch(GLM_COMPILER)
		{
		case GLM_COMPILER_INTEL14:
			std::printf("ICC 14 - 2013 SP1\n");
			break;
		case GLM_COMPILER_INTEL15:
			std::printf("ICC 15 - 2015\n");
			break;
		case GLM_COMPILER_INTEL16:
			std::printf("ICC 16 - 2017\n");
			break;
		case GLM_COMPILER_INTEL17:
			std::printf("ICC 17 - 20XX\n");
			break;
		default:
			std::printf("Intel compiler version not detected\n");
			Error += 1;
			break;
		}
	}
	else
	{
		std::printf("Undetected compiler\n");
		Error += 1;
	}
	
	return Error;
}

int test_model()
{
	int Error = 0;
	
	Error += ((sizeof(void*) == 4) && (GLM_MODEL == GLM_MODEL_32)) || ((sizeof(void*) == 8) && (GLM_MODEL == GLM_MODEL_64)) ? 0 : 1;
	
	if(GLM_MODEL == GLM_MODEL_32)
		std::printf("GLM_MODEL_32\n");
	else if(GLM_MODEL == GLM_MODEL_64)
		std::printf("GLM_MODEL_64\n");
	
	return Error;
}

int test_instruction_set()
{
	int Error = 0;

	std::printf("GLM_ARCH: ");

	if(GLM_ARCH & GLM_ARCH_ARM_BIT)
		std::printf("ARM ");
	if(GLM_ARCH & GLM_ARCH_NEON_BIT)
		std::printf("NEON ");
	if(GLM_ARCH & GLM_ARCH_AVX2_BIT)
		std::printf("AVX2 ");
	if(GLM_ARCH & GLM_ARCH_AVX_BIT)
		std::printf("AVX ");
	if(GLM_ARCH & GLM_ARCH_SSE42_BIT)
		std::printf("SSE4.2 ");
	if(GLM_ARCH & GLM_ARCH_SSE41_BIT)
		std::printf("SSE4.1 ");
	if(GLM_ARCH & GLM_ARCH_SSSE3_BIT)
		std::printf("SSSE3 ");
	if(GLM_ARCH & GLM_ARCH_SSE3_BIT)
		std::printf("SSE3 ");
	if(GLM_ARCH & GLM_ARCH_SSE2_BIT)
		std::printf("SSE2 ");

	std::printf("\n");

	return Error;
}

int test_cpp_version()
{
	std::printf("__cplusplus: %d\n", static_cast<int>(__cplusplus));
	
	return 0;
}

int test_operators()
{
	glm::ivec3 A(1);
	glm::ivec3 B(1);
	bool R = A != B;
	bool S = A == B;

	return (S && !R) ? 0 : 1;
}

int main()
{
	int Error = 0;

#	if !defined(GLM_FORCE_PLATFORM_UNKNOWN) && !defined(GLM_FORCE_COMPILER_UNKNOWN) && !defined(GLM_FORCE_ARCH_UNKNOWN) && !defined(GLM_FORCE_CXX_UNKNOWN)
		
		Error += test_cpp_version();
		Error += test_compiler();
		Error += test_model();
		Error += test_instruction_set();
		Error += test_operators();

#	endif
	
	return Error;
}
