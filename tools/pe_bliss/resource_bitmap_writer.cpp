#include "resource_bitmap_writer.h"
#include "pe_resource_manager.h"
#include "pe_structures.h"

namespace pe_bliss
{
using namespace pe_win;

resource_bitmap_writer::resource_bitmap_writer(pe_resource_manager& res)
	:res_(res)
{}

//Adds bitmap from bitmap file data. If bitmap already exists, replaces it
//timestamp will be used for directories that will be added
void resource_bitmap_writer::add_bitmap(const std::string& bitmap_file, uint32_t id, uint32_t language, uint32_t codepage, uint32_t timestamp)
{
	//Check bitmap data a little
	if(bitmap_file.length() < sizeof(bitmapfileheader))
		throw pe_exception("Incorrect resource bitmap", pe_exception::resource_incorrect_bitmap);

	resource_directory_entry new_entry;
	new_entry.set_id(id);

	//Add bitmap
	res_.add_resource(bitmap_file.substr(sizeof(bitmapfileheader)), pe_resource_viewer::resource_bitmap, new_entry, resource_directory::entry_finder(id), language, codepage, timestamp);
}

//Adds bitmap from bitmap file data. If bitmap already exists, replaces it
//timestamp will be used for directories that will be added
void resource_bitmap_writer::add_bitmap(const std::string& bitmap_file, const std::wstring& name, uint32_t language, uint32_t codepage, uint32_t timestamp)
{
	//Check bitmap data a little
	if(bitmap_file.length() < sizeof(bitmapfileheader))
		throw pe_exception("Incorrect resource bitmap", pe_exception::resource_incorrect_bitmap);

	resource_directory_entry new_entry;
	new_entry.set_name(name);

	//Add bitmap
	res_.add_resource(bitmap_file.substr(sizeof(bitmapfileheader)), pe_resource_viewer::resource_bitmap, new_entry, resource_directory::entry_finder(name), language, codepage, timestamp);
}

//Removes bitmap by name/ID and language
bool resource_bitmap_writer::remove_bitmap(const std::wstring& name, uint32_t language)
{
	return res_.remove_resource(pe_resource_viewer::resource_bitmap, name, language);
}

//Removes bitmap by name/ID and language
bool resource_bitmap_writer::remove_bitmap(uint32_t id, uint32_t language)
{
	return res_.remove_resource(pe_resource_viewer::resource_bitmap, id, language);
}
}
