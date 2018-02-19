/*
 * OggStream.cpp
 * -------------
 * Purpose: Basic Ogg stream parsing functionality
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#include "stdafx.h"
#include "OggStream.h"
#include "../common/FileReader.h"
#include "../common/mptCRC.h"
#include "../common/mptIO.h"


OPENMPT_NAMESPACE_BEGIN


namespace Ogg
{


uint16 PageInfo::GetPagePhysicalSize() const
{
	uint16 size = 0;
	size += sizeof(PageHeader);
	size += header.page_segments;
	for(uint8 segment = 0; segment < header.page_segments; ++segment)
	{
		size += segment_table[segment];
	}
	return size;
}


uint16 PageInfo::GetPageDataSize() const
{
	uint16 size = 0;
	for(uint8 segment = 0; segment < header.page_segments; ++segment)
	{
		size += segment_table[segment];
	}
	return size;
}


bool AdvanceToPageMagic(FileReader &file)
{
#if MPT_COMPILER_MSVC
#pragma warning(push)
#pragma warning(disable:4127) // conditional expression is constant
#endif // MPT_COMPILER_MSVC
	while(true)
#if MPT_COMPILER_MSVC
#pragma warning(pop)
#endif // MPT_COMPILER_MSVC
	{
		if(!file.CanRead(4))
		{
			return false;
		}
		if(file.ReadMagic("OggS"))
		{
			file.SkipBack(4);
			return true;
		}
		file.Skip(1);
	}
}


bool ReadPage(FileReader &file, PageInfo &pageInfo, std::vector<uint8> &pageData)
{
	pageInfo = PageInfo();
	pageData.clear();
	if(!file.ReadMagic("OggS"))
	{
		return false;
	}
	file.SkipBack(4);
	FileReader filePageReader = file; // do not modify original file read position
	if(!filePageReader.ReadStruct(pageInfo.header))
	{
		return false;
	}
	if(!filePageReader.CanRead(pageInfo.header.page_segments))
	{
		return false;
	}
	uint16 pageDataSize = 0;
	for(uint8 segment = 0; segment < pageInfo.header.page_segments; ++segment)
	{
		pageInfo.segment_table[segment] = filePageReader.ReadIntLE<uint8>();
		pageDataSize += pageInfo.segment_table[segment];
	}
	if(!filePageReader.CanRead(pageDataSize))
	{
		return false;
	}
	filePageReader.ReadVector(pageData, pageDataSize);
	filePageReader.SkipBack(pageInfo.GetPagePhysicalSize());
	{
		mpt::crc32_ogg calculatedCRC;
		uint8 rawHeader[sizeof(PageHeader)];
		MemsetZero(rawHeader);
		filePageReader.ReadArray(rawHeader);
		std::memset(rawHeader + 22, 0, 4); // clear out old crc
		calculatedCRC.process(rawHeader, rawHeader + sizeof(rawHeader));
		calculatedCRC.process(pageInfo.segment_table, pageInfo.segment_table + pageInfo.header.page_segments);
		calculatedCRC.process(pageData);
		if(calculatedCRC != pageInfo.header.CRC_checksum)
		{
			return false;
		}
	}
	file.Skip(pageInfo.GetPagePhysicalSize());
	return true;
}


bool ReadPageAndSkipJunk(FileReader &file, PageInfo &pageInfo, std::vector<uint8> &pageData)
{
	pageInfo = PageInfo();
	pageData.clear();
#if MPT_COMPILER_MSVC
#pragma warning(push)
#pragma warning(disable:4127) // conditional expression is constant
#endif // MPT_COMPILER_MSVC
	while(true)
#if MPT_COMPILER_MSVC
#pragma warning(pop)
#endif // MPT_COMPILER_MSVC
	{
		if(!AdvanceToPageMagic(file))
		{
			return false;
		}
		if(ReadPage(file, pageInfo, pageData))
		{
			return true;
		} else
		{
			pageInfo = PageInfo();
			pageData.clear();
		}
		file.Skip(1);
	}
}


bool UpdatePageCRC(PageInfo &pageInfo, const std::vector<uint8> &pageData)
{
	if(pageData.size() != pageInfo.GetPageDataSize())
	{
		return false;
	}
	mpt::crc32_ogg crc;
	pageInfo.header.CRC_checksum = 0;
	char rawHeader[sizeof(PageHeader)];
	std::memcpy(rawHeader, &pageInfo.header, sizeof(PageHeader));
	crc.process(rawHeader, rawHeader + sizeof(PageHeader));
	crc.process(pageInfo.segment_table, pageInfo.segment_table + pageInfo.header.page_segments);
	crc.process(pageData);
	pageInfo.header.CRC_checksum = crc;
	return true;
}


} // namespace Ogg


OPENMPT_NAMESPACE_END
