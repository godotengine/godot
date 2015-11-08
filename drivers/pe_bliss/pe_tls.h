#pragma once
#include <memory>
#include <istream>
#include "pe_base.h"
#include "pe_directory.h"

namespace pe_bliss
{
//Class representing TLS info
//We use "DWORD" type to represent RVAs, because RVA is
//always 32bit even in PE+
class tls_info
{
public:
	typedef std::vector<uint32_t> tls_callback_list;

public:
	//Default constructor
	tls_info();

	//Returns start RVA of TLS raw data
	uint32_t get_raw_data_start_rva() const;
	//Returns end RVA of TLS raw data
	uint32_t get_raw_data_end_rva() const;
	//Returns TLS index RVA
	uint32_t get_index_rva() const;
	//Returns TLS callbacks RVA
	uint32_t get_callbacks_rva() const;
	//Returns size of zero fill
	uint32_t get_size_of_zero_fill() const;
	//Returns characteristics
	uint32_t get_characteristics() const;
	//Returns raw TLS data
	const std::string& get_raw_data() const;
	//Returns TLS callbacks addresses
	const tls_callback_list& get_tls_callbacks() const;

public: //These functions do not change everything inside image, they are used by PE class
	//You can also use them to rebuild TLS directory

	//Sets start RVA of TLS raw data
	void set_raw_data_start_rva(uint32_t rva);
	//Sets end RVA of TLS raw data
	void set_raw_data_end_rva(uint32_t rva);
	//Sets TLS index RVA
	void set_index_rva(uint32_t rva);
	//Sets TLS callbacks RVA
	void set_callbacks_rva(uint32_t rva);
	//Sets size of zero fill
	void set_size_of_zero_fill(uint32_t size);
	//Sets characteristics
	void set_characteristics(uint32_t characteristics);
	//Sets raw TLS data
	void set_raw_data(const std::string& data);
	//Returns TLS callbacks addresses
	tls_callback_list& get_tls_callbacks();
	//Adds TLS callback
	void add_tls_callback(uint32_t rva);
	//Clears TLS callbacks list
	void clear_tls_callbacks();
	//Recalculates end address of raw TLS data
	void recalc_raw_data_end_rva();

private:
	uint32_t start_rva_, end_rva_, index_rva_, callbacks_rva_;
	uint32_t size_of_zero_fill_, characteristics_;

	//Raw TLS data
	std::string raw_data_;

	//TLS callback RVAs
	tls_callback_list callbacks_;
};

//Represents type of expanding of TLS section containing raw data
//(Works only if you are writing TLS raw data to tls_section and it is the last one in the PE image on the moment of TLS rebuild)
enum tls_data_expand_type
{
	tls_data_expand_raw, //If there is not enough raw space for raw TLS data, it can be expanded
	tls_data_expand_virtual //If there is not enough virtual place for raw TLS data, it can be expanded
};


//Get TLS info
//If image does not have TLS, throws an exception
const tls_info get_tls_info(const pe_base& pe);

template<typename PEClassType>
const tls_info get_tls_info_base(const pe_base& pe);
	
//Rebuilder of TLS structures
//If write_tls_callbacks = true, TLS callbacks VAs will be written to their place
//If write_tls_data = true, TLS data will be written to its place
//If you have chosen to rewrite raw data, only (EndAddressOfRawData - StartAddressOfRawData) bytes will be written, not the full length of string
//representing raw data content
//auto_strip_last_section - if true and TLS are placed in the last section, it will be automatically stripped
const image_directory rebuild_tls(pe_base& pe, const tls_info& info, section& tls_section, uint32_t offset_from_section_start = 0, bool write_tls_callbacks = true, bool write_tls_data = true, tls_data_expand_type expand = tls_data_expand_raw, bool save_to_pe_header = true, bool auto_strip_last_section = true);

template<typename PEClassType>
const image_directory rebuild_tls_base(pe_base& pe, const tls_info& info, section& tls_section, uint32_t offset_from_section_start = 0, bool write_tls_callbacks = true, bool write_tls_data = true, tls_data_expand_type expand = tls_data_expand_raw, bool save_to_pe_header = true, bool auto_strip_last_section = true);
}
