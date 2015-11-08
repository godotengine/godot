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
