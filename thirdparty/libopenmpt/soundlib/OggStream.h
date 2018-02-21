/*
 * OggStream.h
 * -----------
 * Purpose: Basic Ogg stream parsing functionality
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "../common/Endianness.h"
#include "../common/mptIO.h"

#include "../common/FileReaderFwd.h"


OPENMPT_NAMESPACE_BEGIN


namespace Ogg
{

struct PageHeader
{
	char     capture_pattern[4]; // "OggS"
	uint8le  version;
	uint8le  header_type;
	uint64le granule_position;
	uint32le bitstream_serial_number;
	uint32le page_seqauence_number;
	uint32le CRC_checksum;
	uint8le  page_segments;
};

} // namespace Ogg
MPT_BINARY_STRUCT(Ogg::PageHeader, 27)
namespace Ogg {


struct PageInfo
{
	PageHeader header;
	uint8 segment_table[255];
	PageInfo()
	{
		MemsetZero(header);
		MemsetZero(segment_table);
	}
	uint16 GetPagePhysicalSize() const;
	uint16 GetPageDataSize() const;
};


// returns false on EOF
bool AdvanceToPageMagic(FileReader &file);

bool ReadPage(FileReader &file, PageInfo &pageInfo, std::vector<uint8> &pageData);

bool ReadPageAndSkipJunk(FileReader &file, PageInfo &pageInfo, std::vector<uint8> &pageData);


bool UpdatePageCRC(PageInfo &pageInfo, const std::vector<uint8> &pageData);


template <typename Tfile>
bool WritePage(Tfile & f, const PageInfo &pageInfo, const std::vector<uint8> &pageData)
{
	if(!mpt::IO::Write(f, pageInfo.header))
	{
		return false;
	}
	if(!mpt::IO::WriteRaw(f, pageInfo.segment_table, pageInfo.header.page_segments))
	{
		return false;
	}
	if(!mpt::IO::WriteRaw(f, pageData.data(), pageData.size()))
	{
		return false;
	}
	return true;
}


} // namespace Ogg


OPENMPT_NAMESPACE_END
