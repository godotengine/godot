#include <glm/glm.hpp>
#include <glm/ext/scalar_int_sized.hpp>

static int test_bit_operator()
{
	int Error = 0;

	glm::ivec4 const a(1);
	glm::ivec4 const b = ~a;
	Error += glm::all(glm::equal(b, glm::ivec4(-2))) ? 0 : 1;

	glm::int32 const c(1);
	glm::int32 const d = ~c;
	Error += d == -2 ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_bit_operator();

	return Error;
}
