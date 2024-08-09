/// @ref gtx_integer

namespace glm
{
	// pow
	GLM_FUNC_QUALIFIER int pow(int x, uint y)
	{
		if(y == 0)
			return x >= 0 ? 1 : -1;

		int result = x;
		for(uint i = 1; i < y; ++i)
			result *= x;
		return result;
	}

	// sqrt: From Christopher J. Musial, An integer square root, Graphics Gems, 1990, page 387
	GLM_FUNC_QUALIFIER int sqrt(int x)
	{
		if(x <= 1) return x;

		int NextTrial = x >> 1;
		int CurrentAnswer;

		do
		{
			CurrentAnswer = NextTrial;
			NextTrial = (NextTrial + x / NextTrial) >> 1;
		} while(NextTrial < CurrentAnswer);

		return CurrentAnswer;
	}

// Henry Gordon Dietz: http://aggregate.org/MAGIC/
namespace detail
{
	GLM_FUNC_QUALIFIER unsigned int ones32(unsigned int x)
	{
		/* 32-bit recursive reduction using SWAR...
		but first step is mapping 2-bit values
		into sum of 2 1-bit values in sneaky way
		*/
		x -= ((x >> 1) & 0x55555555);
		x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
		x = (((x >> 4) + x) & 0x0f0f0f0f);
		x += (x >> 8);
		x += (x >> 16);
		return(x & 0x0000003f);
	}
}//namespace detail

	// Henry Gordon Dietz: http://aggregate.org/MAGIC/
/*
	GLM_FUNC_QUALIFIER unsigned int floor_log2(unsigned int x)
	{
		x |= (x >> 1);
		x |= (x >> 2);
		x |= (x >> 4);
		x |= (x >> 8);
		x |= (x >> 16);

		return _detail::ones32(x) >> 1;
	}
*/
	// mod
	GLM_FUNC_QUALIFIER int mod(int x, int y)
	{
		return ((x % y) + y) % y;
	}

	// factorial (!12 max, integer only)
	template<typename genType>
	GLM_FUNC_QUALIFIER genType factorial(genType const& x)
	{
		genType Temp = x;
		genType Result;
		for(Result = 1; Temp > 1; --Temp)
			Result *= Temp;
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<2, T, Q> factorial(
		vec<2, T, Q> const& x)
	{
		return vec<2, T, Q>(
			factorial(x.x),
			factorial(x.y));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, T, Q> factorial(
		vec<3, T, Q> const& x)
	{
		return vec<3, T, Q>(
			factorial(x.x),
			factorial(x.y),
			factorial(x.z));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<4, T, Q> factorial(
		vec<4, T, Q> const& x)
	{
		return vec<4, T, Q>(
			factorial(x.x),
			factorial(x.y),
			factorial(x.z),
			factorial(x.w));
	}

	GLM_FUNC_QUALIFIER uint pow(uint x, uint y)
	{
		if (y == 0)
			return 1u;

		uint result = x;
		for(uint i = 1; i < y; ++i)
			result *= x;
		return result;
	}

	GLM_FUNC_QUALIFIER uint sqrt(uint x)
	{
		if(x <= 1) return x;

		uint NextTrial = x >> 1;
		uint CurrentAnswer;

		do
		{
			CurrentAnswer = NextTrial;
			NextTrial = (NextTrial + x / NextTrial) >> 1;
		} while(NextTrial < CurrentAnswer);

		return CurrentAnswer;
	}

	GLM_FUNC_QUALIFIER uint mod(uint x, uint y)
	{
		return x - y * (x / y);
	}

#if(GLM_COMPILER & (GLM_COMPILER_VC | GLM_COMPILER_GCC))

	GLM_FUNC_QUALIFIER unsigned int nlz(unsigned int x)
	{
		return 31u - findMSB(x);
	}

#else

	// Hackers Delight: http://www.hackersdelight.org/HDcode/nlz.c.txt
	GLM_FUNC_QUALIFIER unsigned int nlz(unsigned int x)
	{
		int y, m, n;

		y = -int(x >> 16);      // If left half of x is 0,
		m = (y >> 16) & 16;  // set n = 16.  If left half
		n = 16 - m;          // is nonzero, set n = 0 and
		x = x >> m;          // shift x right 16.
							// Now x is of the form 0000xxxx.
		y = x - 0x100;       // If positions 8-15 are 0,
		m = (y >> 16) & 8;   // add 8 to n and shift x left 8.
		n = n + m;
		x = x << m;

		y = x - 0x1000;      // If positions 12-15 are 0,
		m = (y >> 16) & 4;   // add 4 to n and shift x left 4.
		n = n + m;
		x = x << m;

		y = x - 0x4000;      // If positions 14-15 are 0,
		m = (y >> 16) & 2;   // add 2 to n and shift x left 2.
		n = n + m;
		x = x << m;

		y = x >> 14;         // Set y = 0, 1, 2, or 3.
		m = y & ~(y >> 1);   // Set m = 0, 1, 2, or 2 resp.
		return unsigned(n + 2 - m);
	}

#endif//(GLM_COMPILER)

}//namespace glm
