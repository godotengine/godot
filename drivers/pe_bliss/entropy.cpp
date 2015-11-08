#include <cmath>
#include "entropy.h"
#include "utils.h"

namespace pe_bliss
{
//Calculates entropy for PE image section
double entropy_calculator::calculate_entropy(const section& s)
{
	if(s.get_raw_data().empty()) //Don't count entropy for empty sections
		throw pe_exception("Section is empty", pe_exception::section_is_empty);

	return calculate_entropy(s.get_raw_data().data(), s.get_raw_data().length());
}

//Calculates entropy for istream (from current position of stream)
double entropy_calculator::calculate_entropy(std::istream& file)
{
	uint32_t byte_count[256] = {0}; //Byte count for each of 255 bytes

	if(file.bad())
		throw pe_exception("Stream is bad", pe_exception::stream_is_bad);

	std::streamoff pos = file.tellg();

	std::streamoff length = pe_utils::get_file_size(file);
	length -= file.tellg();

	if(!length) //Don't calculate entropy for empty buffers
		throw pe_exception("Data length is zero", pe_exception::data_is_empty);

	//Count bytes
	for(std::streamoff i = 0; i != length; ++i)
		++byte_count[static_cast<unsigned char>(file.get())];

	file.seekg(pos);

	return calculate_entropy(byte_count, length);
}

//Calculates entropy for data block
double entropy_calculator::calculate_entropy(const char* data, size_t length)
{
	uint32_t byte_count[256] = {0}; //Byte count for each of 255 bytes

	if(!length) //Don't calculate entropy for empty buffers
		throw pe_exception("Data length is zero", pe_exception::data_is_empty);

	//Count bytes
	for(size_t i = 0; i != length; ++i)
		++byte_count[static_cast<unsigned char>(data[i])];

	return calculate_entropy(byte_count, length);
}

//Calculates entropy for this PE file (only section data)
double entropy_calculator::calculate_entropy(const pe_base& pe)
{
	uint32_t byte_count[256] = {0}; //Byte count for each of 255 bytes

	size_t total_data_length = 0;

	//Count bytes for each section
	for(section_list::const_iterator it = pe.get_image_sections().begin(); it != pe.get_image_sections().end(); ++it)
	{
		const std::string& data = (*it).get_raw_data();
		size_t length = data.length();
		total_data_length += length;
		for(size_t i = 0; i != length; ++i)
			++byte_count[static_cast<unsigned char>(data[i])];
	}

	return calculate_entropy(byte_count, total_data_length);
}

//Calculates entropy from bytes count
double entropy_calculator::calculate_entropy(const uint32_t byte_count[256], std::streamoff total_length)
{
	double entropy = 0.; //Entropy result value
	//Calculate entropy
	for(uint32_t i = 0; i < 256; ++i)
	{
		double temp = static_cast<double>(byte_count[i]) / total_length;
		if(temp > 0.)
			entropy += std::abs(temp * (std::log(temp) * pe_utils::log_2));
	}

	return entropy;
}
}
