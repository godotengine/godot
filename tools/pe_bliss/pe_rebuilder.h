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
#include <ostream>
#include <fstream>

namespace pe_bliss
{
class pe_base;
//Rebuilds PE image, writes resulting image to ostream "out". If strip_dos_header == true, DOS header will be stripped a little
//If change_size_of_headers == true, SizeOfHeaders will be recalculated automatically
//If save_bound_import == true, existing bound import directory will be saved correctly (because some compilers and bind.exe put it to PE headers)
void rebuild_pe(pe_base& pe, std::ostream& out, bool strip_dos_header = false, bool change_size_of_headers = true, bool save_bound_import = true);

//Rebuild PE image and write it to "out" file
//If strip_dos_header is true, DOS headers partially will be used for PE headers
//If change_size_of_headers == true, SizeOfHeaders will be recalculated automatically
//If save_bound_import == true, existing bound import directory will be saved correctly (because some compilers and bind.exe put it to PE headers)
void rebuild_pe(pe_base& pe, const char* out, bool strip_dos_header = false, bool change_size_of_headers = true, bool save_bound_import = true);

}
