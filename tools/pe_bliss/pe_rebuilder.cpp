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
#include "pe_rebuilder.h"
#include "pe_base.h"
#include "pe_structures.h"
#include "pe_exception.h"

namespace pe_bliss
{
using namespace pe_win;

//Rebuilds PE image headers
//If strip_dos_header is true, DOS headers partially will be used for PE headers
//If change_size_of_headers == true, SizeOfHeaders will be recalculated automatically
//If save_bound_import == true, existing bound import directory will be saved correctly (because some compilers and bind.exe put it to PE headers)
void rebuild_pe(pe_base& pe, image_dos_header& dos_header, bool strip_dos_header, bool change_size_of_headers, bool save_bound_import)
{
	dos_header = pe.get_dos_header();

	if(strip_dos_header)
	{
		//Strip stub overlay
		pe.strip_stub_overlay();
		//BaseOfCode NT Headers field now overlaps
		//e_lfanew field, so we're acrually setting
		//e_lfanew with this call
		pe.set_base_of_code(8 * sizeof(uint16_t));
	}
	else
	{
		//Set start of PE headers
		dos_header.e_lfanew = sizeof(image_dos_header)
			+ pe_utils::align_up(static_cast<uint32_t>(pe.get_stub_overlay().size()), sizeof(uint32_t));
	}

	section_list& sections = pe.get_image_sections();

	//Calculate pointer to section data
	size_t ptr_to_section_data = (strip_dos_header ? 8 * sizeof(uint16_t) : sizeof(image_dos_header)) + pe.get_sizeof_nt_header()
		+ pe_utils::align_up(pe.get_stub_overlay().size(), sizeof(uint32_t))
		- sizeof(image_data_directory) * (image_numberof_directory_entries - pe.get_number_of_rvas_and_sizes())
		+ sections.size() * sizeof(image_section_header);

	if(save_bound_import && pe.has_bound_import())
	{
		//It will be aligned to DWORD, because we're aligning to DWORD everything above it
		pe.set_directory_rva(image_directory_entry_bound_import, static_cast<uint32_t>(ptr_to_section_data));
		ptr_to_section_data += pe.get_directory_size(image_directory_entry_bound_import);	
	}
	
	ptr_to_section_data = pe_utils::align_up(ptr_to_section_data, pe.get_file_alignment());

	//Set size of headers and size of optional header
	if(change_size_of_headers)
	{
		if(!pe.get_image_sections().empty())
		{
			if(static_cast<uint32_t>(ptr_to_section_data) > (*sections.begin()).get_virtual_address())
				throw pe_exception("Headers of PE file are too long. Try to strip STUB or don't build bound import", pe_exception::cannot_rebuild_image);
		}

		pe.set_size_of_headers(static_cast<uint32_t>(ptr_to_section_data));
	}

	//Set number of sections in PE header
	pe.update_number_of_sections();

	pe.update_image_size();

	pe.set_size_of_optional_header(static_cast<uint16_t>(pe.get_sizeof_opt_headers()
		- sizeof(image_data_directory) * (image_numberof_directory_entries - pe.get_number_of_rvas_and_sizes())));

	//Recalculate pointer to raw data according to section list
	for(section_list::iterator it = sections.begin(); it != sections.end(); ++it)
	{
		//Save section headers PointerToRawData
		(*it).set_pointer_to_raw_data(static_cast<uint32_t>(ptr_to_section_data));
		ptr_to_section_data += (*it).get_aligned_raw_size(pe.get_file_alignment());
	}
}

//Rebuild PE image and write it to "out" ostream
//If strip_dos_header is true, DOS headers partially will be used for PE headers
//If change_size_of_headers == true, SizeOfHeaders will be recalculated automatically
//If save_bound_import == true, existing bound import directory will be saved correctly (because some compilers and bind.exe put it to PE headers)
void rebuild_pe(pe_base& pe, std::ostream& out, bool strip_dos_header, bool change_size_of_headers, bool save_bound_import)
{
	if(out.bad())
		throw pe_exception("Stream is bad", pe_exception::stream_is_bad);

	if(save_bound_import && pe.has_bound_import())
	{
		if(pe.section_data_length_from_rva(pe.get_directory_rva(image_directory_entry_bound_import), pe.get_directory_rva(image_directory_entry_bound_import), section_data_raw, true)
			< pe.get_directory_size(image_directory_entry_bound_import))
			throw pe_exception("Incorrect bound import directory", pe_exception::incorrect_bound_import_directory);
	}

	//Change ostream state
	out.exceptions(std::ios::goodbit);
	out.clear();
	
	uint32_t original_bound_import_rva = pe.has_bound_import() ? pe.get_directory_rva(image_directory_entry_bound_import) : 0;
	if(original_bound_import_rva && original_bound_import_rva > pe.get_size_of_headers())
	{
		//No need to do anything with bound import directory
		//if it is placed inside of any section, not headers
		original_bound_import_rva = 0;
		save_bound_import = false;
	}

	{
		image_dos_header dos_header;

		//Rebuild PE image headers
		rebuild_pe(pe, dos_header, strip_dos_header, change_size_of_headers, save_bound_import);

		//Write DOS header
		out.write(reinterpret_cast<const char*>(&dos_header), strip_dos_header ? 8 * sizeof(uint16_t) : sizeof(image_dos_header));
	}

	//If we have stub overlay, write it too
	{
		const std::string& stub = pe.get_stub_overlay();
		if(stub.size())
		{
			out.write(stub.data(), stub.size());
			size_t aligned_size = pe_utils::align_up(stub.size(), sizeof(uint32_t));
			//Align PE header, which is right after rich overlay
			while(aligned_size > stub.size())
			{
				out.put('\0');
				--aligned_size;
			}
		}
	}
	
	//Write NT headers
	out.write(static_cast<const pe_base&>(pe).get_nt_headers_ptr(), pe.get_sizeof_nt_header()
		- sizeof(image_data_directory) * (image_numberof_directory_entries - pe.get_number_of_rvas_and_sizes()));

	//Write section headers
	const section_list& sections = pe.get_image_sections();
	for(section_list::const_iterator it = sections.begin(); it != sections.end(); ++it)
	{
		if(it == sections.end() - 1) //If last section encountered
		{
			image_section_header header((*it).get_raw_header());
			header.SizeOfRawData = static_cast<uint32_t>((*it).get_raw_data().length()); //Set non-aligned actual data length for it
			out.write(reinterpret_cast<const char*>(&header), sizeof(image_section_header));
		}
		else
		{
			out.write(reinterpret_cast<const char*>(&(*it).get_raw_header()), sizeof(image_section_header));
		}
	}

	//Write bound import data if requested
	if(save_bound_import && pe.has_bound_import())
	{
		out.write(pe.section_data_from_rva(original_bound_import_rva, section_data_raw, true),
			pe.get_directory_size(image_directory_entry_bound_import));
	}

	//Write section data finally
	for(section_list::const_iterator it = sections.begin(); it != sections.end(); ++it)
	{
		const section& s = *it;

		std::streamoff wpos = out.tellp();

		//Fill unused overlay data between sections with null bytes
		for(unsigned int i = 0; i < s.get_pointer_to_raw_data() - wpos; i++)
			out.put(0);

		//Write raw section data
		out.write(s.get_raw_data().data(), s.get_raw_data().length());
	}
}

//Rebuild PE image and write it to "out" file
//If strip_dos_header is true, DOS headers partially will be used for PE headers
//If change_size_of_headers == true, SizeOfHeaders will be recalculated automatically
//If save_bound_import == true, existing bound import directory will be saved correctly (because some compilers and bind.exe put it to PE headers)
void rebuild_pe(pe_base& pe, const char* out, bool strip_dos_header, bool change_size_of_headers, bool save_bound_import)
{
	std::ofstream pe_file(out, std::ios::out | std::ios::binary | std::ios::trunc);
	if(!pe_file)
	{
		throw pe_exception("Error in open file.", pe_exception::stream_is_bad);
	}
	rebuild_pe(pe, pe_file, strip_dos_header, change_size_of_headers, save_bound_import);
}


}
