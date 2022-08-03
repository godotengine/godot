#include <glm/ext/vector_integer.hpp>
#include <glm/ext/vector_int1.hpp>
#include <glm/ext/vector_int1_sized.hpp>
#include <glm/ext/vector_uint1.hpp>
#include <glm/ext/vector_uint1_sized.hpp>

template <typename genType>
static int test_operators()
{
	int Error = 0;

	{
		genType const A(1);
		genType const B(1);

		bool const R = A != B;
		bool const S = A == B;
		Error += (S && !R) ? 0 : 1;
	}

	{
		genType const A(1);
		genType const B(1);

		genType const C = A + B;
		Error += C == genType(2) ? 0 : 1;

		genType const D = A - B;
		Error += D == genType(0) ? 0 : 1;

		genType const E = A * B;
		Error += E == genType(1) ? 0 : 1;

		genType const F = A / B;
		Error += F == genType(1) ? 0 : 1;
	}

	{
		genType const A(3);
		genType const B(2);

		genType const C = A % B;
		Error += C == genType(1) ? 0 : 1;
	}

	{
		genType const A(1);
		genType const B(1);
		genType const C(0);

		genType const I = A & B;
		Error += I == genType(1) ? 0 : 1;
		genType const D = A & C;
		Error += D == genType(0) ? 0 : 1;

		genType const E = A | B;
		Error += E == genType(1) ? 0 : 1;
		genType const F = A | C;
		Error += F == genType(1) ? 0 : 1;

		genType const G = A ^ B;
		Error += G == genType(0) ? 0 : 1;
		genType const H = A ^ C;
		Error += H == genType(1) ? 0 : 1;
	}

	{
		genType const A(0);
		genType const B(1);
		genType const C(2);

		genType const D = B << B;
		Error += D == genType(2) ? 0 : 1;
		genType const E = C >> B;
		Error += E == genType(1) ? 0 : 1;
	}

	return Error;
}

template <typename genType>
static int test_ctor()
{
	typedef typename genType::value_type T;
	
	int Error = 0;

	genType const A = genType(1);

	genType const E(genType(1));
	Error += A == E ? 0 : 1;

	genType const F(E);
	Error += A == F ? 0 : 1;

	genType const B = genType(1);
	genType const G(glm::vec<2, T>(1));
	Error += B == G ? 0 : 1;

	genType const H(glm::vec<3, T>(1));
	Error += B == H ? 0 : 1;

	genType const I(glm::vec<4, T>(1));
	Error += B == I ? 0 : 1;

	return Error;
}

template <typename genType>
static int test_size()
{
	int Error = 0;

	Error += sizeof(typename genType::value_type) == sizeof(genType) ? 0 : 1;
	Error += genType().length() == 1 ? 0 : 1;
	Error += genType::length() == 1 ? 0 : 1;

	return Error;
}

template <typename genType>
static int test_relational()
{
	int Error = 0;

	genType const A(1);
	genType const B(1);
	genType const C(0);

	Error += A == B ? 0 : 1;
	Error += A != C ? 0 : 1;
	Error += all(equal(A, B)) ? 0 : 1;
	Error += any(notEqual(A, C)) ? 0 : 1;

	return Error;
}

template <typename genType>
static int test_constexpr()
{
#	if GLM_CONFIG_CONSTEXP == GLM_ENABLE
		static_assert(genType::length() == 1, "GLM: Failed constexpr");
		static_assert(genType(1)[0] == 1, "GLM: Failed constexpr");
		static_assert(genType(1) == genType(1), "GLM: Failed constexpr");
		static_assert(genType(1) != genType(0), "GLM: Failed constexpr");
#	endif

	return 0;
}

int main()
{
	int Error = 0;

	Error += test_operators<glm::ivec1>();
	Error += test_operators<glm::i8vec1>();
	Error += test_operators<glm::i16vec1>();
	Error += test_operators<glm::i32vec1>();
	Error += test_operators<glm::i64vec1>();

	Error += test_ctor<glm::ivec1>();
	Error += test_ctor<glm::i8vec1>();
	Error += test_ctor<glm::i16vec1>();
	Error += test_ctor<glm::i32vec1>();
	Error += test_ctor<glm::i64vec1>();

	Error += test_size<glm::ivec1>();
	Error += test_size<glm::i8vec1>();
	Error += test_size<glm::i16vec1>();
	Error += test_size<glm::i32vec1>();
	Error += test_size<glm::i64vec1>();

	Error += test_relational<glm::ivec1>();
	Error += test_relational<glm::i8vec1>();
	Error += test_relational<glm::i16vec1>();
	Error += test_relational<glm::i32vec1>();
	Error += test_relational<glm::i64vec1>();

	Error += test_constexpr<glm::ivec1>();
	Error += test_constexpr<glm::i8vec1>();
	Error += test_constexpr<glm::i16vec1>();
	Error += test_constexpr<glm::i32vec1>();
	Error += test_constexpr<glm::i64vec1>();

	Error += test_operators<glm::uvec1>();
	Error += test_operators<glm::u8vec1>();
	Error += test_operators<glm::u16vec1>();
	Error += test_operators<glm::u32vec1>();
	Error += test_operators<glm::u64vec1>();
	
	Error += test_ctor<glm::uvec1>();
	Error += test_ctor<glm::u8vec1>();
	Error += test_ctor<glm::u16vec1>();
	Error += test_ctor<glm::u32vec1>();
	Error += test_ctor<glm::u64vec1>();
	
	Error += test_size<glm::uvec1>();
	Error += test_size<glm::u8vec1>();
	Error += test_size<glm::u16vec1>();
	Error += test_size<glm::u32vec1>();
	Error += test_size<glm::u64vec1>();
	
	Error += test_relational<glm::uvec1>();
	Error += test_relational<glm::u8vec1>();
	Error += test_relational<glm::u16vec1>();
	Error += test_relational<glm::u32vec1>();
	Error += test_relational<glm::u64vec1>();
	
	Error += test_constexpr<glm::uvec1>();
	Error += test_constexpr<glm::u8vec1>();
	Error += test_constexpr<glm::u16vec1>();
	Error += test_constexpr<glm::u32vec1>();
	Error += test_constexpr<glm::u64vec1>();
	
	return Error;
}
