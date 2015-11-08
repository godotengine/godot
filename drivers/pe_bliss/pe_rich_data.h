#pragma once
#include <vector>
#include "pe_structures.h"
#include "pe_base.h"

namespace pe_bliss
{
//Rich data overlay class of Microsoft Visual Studio
class rich_data
{
public:
	//Default constructor
	rich_data();

public: //Getters
	//Who knows, what these fields mean...
	uint32_t get_number() const;
	uint32_t get_version() const;
	uint32_t get_times() const;

public: //Setters, used by PE library only
	void set_number(uint32_t number);
	void set_version(uint32_t version);
	void set_times(uint32_t times);

private:
	uint32_t number_;
	uint32_t version_;
	uint32_t times_;
};

//Rich data list typedef
typedef std::vector<rich_data> rich_data_list;

//Returns a vector with rich data (stub overlay)
const rich_data_list get_rich_data(const pe_base& pe);
}
