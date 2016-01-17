/*************************************************************************/
/* Copyright (c) 2015 dx, http://kaimi.ru                                */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person           */
/* obtaining a copy of this software and associated documentation        */
/* files (the "Software"), to deal in the Software without               */
/* restriction, including without limitation the rights to use,          */
/* copy, modify, merge, publish, distribute, sublicense, and/or          */
/* sell copies of the Software, and to permit persons to whom the        */
/* Software is furnished to do so, subject to the following conditions:  */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#pragma once
#include <istream>
#include "pe_base.h"

namespace pe_bliss
{
class entropy_calculator
{
public:
	//Calculates entropy for PE image section
	static double calculate_entropy(const section& s);

	//Calculates entropy for istream (from current position of stream)
	static double calculate_entropy(std::istream& file);

	//Calculates entropy for data block
	static double calculate_entropy(const char* data, size_t length);

	//Calculates entropy for this PE file (only section data)
	static double calculate_entropy(const pe_base& pe);

private:
	entropy_calculator();
	entropy_calculator(const entropy_calculator&);
	entropy_calculator& operator=(const entropy_calculator&);

	//Calculates entropy from bytes count
	static double calculate_entropy(const uint32_t byte_count[256], std::streamoff total_length);
};
}
