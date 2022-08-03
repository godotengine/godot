#include <glm/ext/scalar_reciprocal.hpp>
#include <glm/ext/scalar_relational.hpp>
#include <glm/ext/scalar_constants.hpp>

static int test_sec()
{
	int Error = 0;
	
	Error += glm::equal(glm::sec(0.0), 1.0, 0.01) ? 0 : 1;
	Error += glm::equal(glm::sec(glm::pi<double>() * 2.0), 1.0, 0.01) ? 0 : 1;
	Error += glm::equal(glm::sec(glm::pi<double>() * -2.0), 1.0, 0.01) ? 0 : 1;
	Error += glm::equal(glm::sec(glm::pi<double>() * 1.0), -1.0, 0.01) ? 0 : 1;
	Error += glm::equal(glm::sec(glm::pi<double>() * -1.0), -1.0, 0.01) ? 0 : 1;

	return Error;
}

static int test_csc()
{
	int Error = 0;
	
	double const a = glm::csc(glm::pi<double>() * 0.5);
	Error += glm::equal(a, 1.0, 0.01) ? 0 : 1;
	double const b = glm::csc(glm::pi<double>() * -0.5);
	Error += glm::equal(b, -1.0, 0.01) ? 0 : 1;

	return Error;
}

static int test_cot()
{
	int Error = 0;
	
	double const a = glm::cot(glm::pi<double>() * 0.5);
	Error += glm::equal(a, 0.0, 0.01) ? 0 : 1;
	double const b = glm::cot(glm::pi<double>() * -0.5);
	Error += glm::equal(b, 0.0, 0.01) ? 0 : 1;

	return Error;
}

static int test_asec()
{
	int Error = 0;
	
	Error += glm::equal(glm::asec(100000.0), glm::pi<double>() * 0.5, 0.01) ? 0 : 1;
	Error += glm::equal(glm::asec(-100000.0), glm::pi<double>() * 0.5, 0.01) ? 0 : 1;

	return Error;
}

static int test_acsc()
{
	int Error = 0;
	
	Error += glm::equal(glm::acsc(100000.0), 0.0, 0.01) ? 0 : 1;
	Error += glm::equal(glm::acsc(-100000.0), 0.0, 0.01) ? 0 : 1;

	return Error;
}

static int test_acot()
{
	int Error = 0;
	
	Error += glm::equal(glm::acot(100000.0), 0.0, 0.01) ? 0 : 1;
	Error += glm::equal(glm::acot(-100000.0), glm::pi<double>(), 0.01) ? 0 : 1;
	Error += glm::equal(glm::acot(0.0), glm::pi<double>() * 0.5, 0.01) ? 0 : 1;

	return Error;
}

static int test_sech()
{
	int Error = 0;
	
	Error += glm::equal(glm::sech(100000.0), 0.0, 0.01) ? 0 : 1;
	Error += glm::equal(glm::sech(-100000.0), 0.0, 0.01) ? 0 : 1;
	Error += glm::equal(glm::sech(0.0), 1.0, 0.01) ? 0 : 1;

	return Error;
}

static int test_csch()
{
	int Error = 0;
	
	Error += glm::equal(glm::csch(100000.0), 0.0, 0.01) ? 0 : 1;
	Error += glm::equal(glm::csch(-100000.0), 0.0, 0.01) ? 0 : 1;

	return Error;
}

static int test_coth()
{
	int Error = 0;
	
	double const a = glm::coth(100.0);
	Error += glm::equal(a, 1.0, 0.01) ? 0 : 1;
	
	double const b = glm::coth(-100.0);
	Error += glm::equal(b, -1.0, 0.01) ? 0 : 1;

	return Error;
}

static int test_asech()
{
	int Error = 0;
	
	double const a = glm::asech(1.0);
	Error += glm::equal(a, 0.0, 0.01) ? 0 : 1;

	return Error;
}

static int test_acsch()
{
	int Error = 0;
	
	Error += glm::acsch(0.01) > 1.0 ? 0 : 1;
	Error += glm::acsch(-0.01) < -1.0 ? 0 : 1;

	Error += glm::equal(glm::acsch(100.0), 0.0, 0.01) ? 0 : 1;
	Error += glm::equal(glm::acsch(-100.0), 0.0, 0.01) ? 0 : 1;

	return Error;
}

static int test_acoth()
{
	int Error = 0;
	
	double const a = glm::acoth(1.00001);
	Error += a > 6.0 ? 0 : 1;
	
	double const b = glm::acoth(-1.00001);
	Error += b < -6.0 ? 0 : 1;

	double const c = glm::acoth(10000.0);
	Error += glm::equal(c, 0.0, 0.01) ? 0 : 1;
	
	double const d = glm::acoth(-10000.0);
	Error += glm::equal(d, 0.0, 0.01) ? 0 : 1;

	return Error;
}


int main()
{
	int Error = 0;
	
	Error += test_sec();
	Error += test_csc();
	Error += test_cot();

	Error += test_asec();
	Error += test_acsc();
	Error += test_acot();

	Error += test_sech();
	Error += test_csch();
	Error += test_coth();

	Error += test_asech();
	Error += test_acsch();
	Error += test_acoth();

	return Error;
}
