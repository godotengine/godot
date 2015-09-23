#include "pe_checksum.h"
#include "pe_structures.h"
#include "pe_base.h"

namespace pe_bliss
{
using namespace pe_win;

//Calculate checksum of image
uint32_t calculate_checksum(std::istream& file)
{
	//Save istream state
	std::ios_base::iostate state = file.exceptions();
	std::streamoff old_offset = file.tellg();

	//Checksum value
	unsigned long long checksum = 0;

	try
	{
		image_dos_header header;

		file.exceptions(std::ios::goodbit);

		//Read DOS header
		pe_base::read_dos_header(file, header);

		//Calculate PE checksum
		file.seekg(0);
		unsigned long long top = 0xFFFFFFFF;
		top++;

		//"CheckSum" field position in optional PE headers - it's always 64 for PE and PE+
		static const unsigned long checksum_pos_in_optional_headers = 64;
		//Calculate real PE headers "CheckSum" field position
		//Sum is safe here
		unsigned long pe_checksum_pos = header.e_lfanew + sizeof(image_file_header) + sizeof(uint32_t) + checksum_pos_in_optional_headers;

		//Calculate checksum for each byte of file
		std::streamoff filesize = pe_utils::get_file_size(file);
		for(long long i = 0; i < filesize; i += 4)
		{
			unsigned long dw = 0;

			//Read DWORD from file
			file.read(reinterpret_cast<char*>(&dw), sizeof(unsigned long));
			//Skip "CheckSum" DWORD
			if(i == pe_checksum_pos)
				continue;

			//Calculate checksum
			checksum = (checksum & 0xffffffff) + dw + (checksum >> 32);
			if(checksum > top)
				checksum = (checksum & 0xffffffff) + (checksum >> 32);
		}

		//Finish checksum
		checksum = (checksum & 0xffff) + (checksum >> 16);
		checksum = (checksum) + (checksum >> 16);
		checksum = checksum & 0xffff;

		checksum += static_cast<unsigned long>(filesize);
	}
	catch(const std::exception&)
	{
		//If something went wrong, restore istream state
		file.exceptions(state);
		file.seekg(old_offset);
		file.clear();
		//Rethrow
		throw;
	}

	//Restore istream state
	file.exceptions(state);
	file.seekg(old_offset);
	file.clear();

	//Return checksum
	return static_cast<uint32_t>(checksum);	
}
}
