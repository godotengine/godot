#include "pe_rich_data.h"

namespace pe_bliss
{
//STUB OVERLAY
//Default constructor
rich_data::rich_data()
	:number_(0), version_(0), times_(0)
{}

//Who knows, what these fields mean...
uint32_t rich_data::get_number() const
{
	return number_;
}

uint32_t rich_data::get_version() const
{
	return version_;
}

uint32_t rich_data::get_times() const
{
	return times_;
}

void rich_data::set_number(uint32_t number)
{
	number_ = number;
}

void rich_data::set_version(uint32_t version)
{
	version_ = version;
}

void rich_data::set_times(uint32_t times)
{
	times_ = times;
}

//Returns MSVC rich data
const rich_data_list get_rich_data(const pe_base& pe)
{
	//Returned value
	rich_data_list ret;

	const std::string& rich_overlay = pe.get_stub_overlay();

	//If there's no rich overlay, return empty vector
	if(rich_overlay.size() < sizeof(uint32_t))
		return ret;

	//True if rich data was found
	bool found = false;

	//Rich overlay ID ("Rich" word)
	static const uint32_t rich_overlay_id = 0x68636952;

	//Search for rich data overlay ID
	const char* begin = &rich_overlay[0];
	const char* end = begin + rich_overlay.length();
	for(; begin != end; ++begin)
	{
		if(*reinterpret_cast<const uint32_t*>(begin) == rich_overlay_id)
		{
			found = true; //We've found it!
			break;
		}
	}

	//If we found it
	if(found)
	{
		//Check remaining length
		if(static_cast<size_t>(end - begin) < sizeof(uint32_t))
			return ret;

		//The XOR key is after "Rich" word, we should get it
		uint32_t xorkey = *reinterpret_cast<const uint32_t*>(begin + sizeof(uint32_t));

		//True if rich data was found
		found = false;

		//Second search for signature "DanS"
		begin = &rich_overlay[0];
		for(; begin != end; ++begin)
		{
			if((*reinterpret_cast<const uint32_t*>(begin) ^ xorkey) == 0x536e6144) //"DanS"
			{
				found = true;
				break;
			}
		}

		//If second signature is found
		if(found)
		{
			begin += sizeof(uint32_t) * 3;
			//List all rich data structures
			while(begin < end)
			{
				begin += sizeof(uint32_t);
				if(begin >= end)
					break;

				//Check for rich overlay data end ("Rich" word reached)
				if(*reinterpret_cast<const uint32_t*>(begin) == rich_overlay_id)
					break;

				//Create rich_data structure
				rich_data data;
				data.set_number((*reinterpret_cast<const uint32_t*>(begin) ^ xorkey) >> 16);
				data.set_version((*reinterpret_cast<const uint32_t*>(begin) ^ xorkey) & 0xFFFF);

				begin += sizeof(uint32_t);
				if(begin >= end)
					break;

				data.set_times(*reinterpret_cast<const uint32_t*>(begin) ^ xorkey);

				//Save rich data structure
				ret.push_back(data);
			}
		}
	}

	//Return rich data structures list
	return ret;
}
}
