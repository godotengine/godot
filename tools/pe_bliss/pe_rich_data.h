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
