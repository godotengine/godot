#include "rtti.hpp"

int Base::sync()
{
	return 0;
}

int Test::sync()
{
	buffer[0] = '\n';
	buffer[1] = 0;
	return 666;
}
