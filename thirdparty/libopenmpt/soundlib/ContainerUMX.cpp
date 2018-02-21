/*
 * ContainerUMX.cpp
 * ----------------
 * Purpose: UMX (Unreal Music) module ripper
 * Notes  : Obviously, this code only rips modules from older Unreal Engine games, such as Unreal 1, Unreal Tournament 1 and Deus Ex.
 * Authors: Johannes Schultz (inspired by code from http://wiki.beyondunreal.com/Legacy:Package_File_Format)
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"
#include "UMXTools.h"
#include "Container.h"
#include "Sndfile.h"


OPENMPT_NAMESPACE_BEGIN


static bool ValidateHeader(const UMXFileHeader &fileHeader)
{
	if(std::memcmp(fileHeader.magic, "\xC1\x83\x2A\x9E", 4)
		|| fileHeader.nameCount == 0
		|| fileHeader.exportCount == 0
		|| fileHeader.importCount == 0
		)
	{
		return false;
	}
	return true;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderUMX(MemoryFileReader file, const uint64 *pfilesize)
{
	UMXFileHeader fileHeader;
	if(!file.ReadStruct(fileHeader))
	{
		return ProbeWantMoreData;
	}
	if(!ValidateHeader(fileHeader))
	{
		return ProbeFailure;
	}
	if(!FindUMXNameTableEntryMemory(file, fileHeader, "music"))
	{
		return ProbeFailure;
	}
	MPT_UNREFERENCED_PARAMETER(pfilesize);
	return ProbeSuccess;
}


bool UnpackUMX(std::vector<ContainerItem> &containerItems, FileReader &file, ContainerLoadingFlags loadFlags)
{
	file.Rewind();
	containerItems.clear();

	UMXFileHeader fileHeader;
	if(!file.ReadStruct(fileHeader))
	{
		return false;
	}
	if(!ValidateHeader(fileHeader))
	{
		return false;
	}

	// Note that this can be a false positive, e.g. Unreal maps will have music and sound
	// in their name table because they usually import such files. However, it spares us
	// from wildly seeking through the file, as the name table is usually right at the
	// start of the file, so it is hopefully a good enough heuristic for our purposes.
	if(!FindUMXNameTableEntry(file, fileHeader, "music"))
	{
		return false;
	}

	if(loadFlags == ContainerOnlyVerifyHeader)
	{
		return true;
	}

	// Read name table
	std::vector<std::string> names = ReadUMXNameTable(file, fileHeader);

	// Read import table
	if(!file.Seek(fileHeader.importOffset))
	{
		return false;
	}

	std::vector<int32> classes;
	classes.reserve(fileHeader.importCount);
	for(uint32 i = 0; i < fileHeader.importCount && file.CanRead(4); i++)
	{
		int32 objName = ReadUMXImportTableEntry(file, fileHeader.packageVersion);
		if(static_cast<size_t>(objName) < names.size())
		{
			classes.push_back(objName);
		}
	}

	// Read export table
	if(!file.Seek(fileHeader.exportOffset))
	{
		return false;
	}

	// Now we can be pretty sure that we're doing the right thing.
	
	for(uint32 i = 0; i < fileHeader.exportCount && file.CanRead(4); i++)
	{
		int32 objClass, objOffset, objSize, objName;
		ReadUMXExportTableEntry(file, objClass, objOffset, objSize, objName, fileHeader.packageVersion);

		if(objSize <= 0 || objClass >= 0)
		{
			continue;
		}

		// Look up object class name (we only want music).
		objClass = -objClass - 1;
		bool isMusic = false;
		if(static_cast<size_t>(objClass) < classes.size())
		{
			isMusic = (names[classes[objClass]] == "music");
		}
		if(!isMusic)
		{
			continue;
		}

		FileReader chunk = file.GetChunkAt(objOffset, objSize);

		if(chunk.IsValid())
		{
			if(fileHeader.packageVersion < 40)
			{
				chunk.Skip(8); // 00 00 00 00 00 00 00 00
			}
			if(fileHeader.packageVersion < 60)
			{
				chunk.Skip(16); // 81 00 00 00 00 00 FF FF FF FF FF FF FF FF 00 00
			}
			// Read object properties
#if 0
			size_t propertyName = static_cast<size_t>(ReadUMXIndex(chunk));
			if(propertyName >= names.size() || names[propertyName] != "none")
			{
				// Can't bother to implement property reading, as no UMX files I've seen so far use properties for the relevant objects,
				// and only the UAX files in the Unreal 1997/98 beta seem to use this and still load just fine when ignoring it.
				// If it should be necessary to implement this, check CUnProperty.cpp in http://ut-files.com/index.php?dir=Utilities/&file=utcms_source.zip
				MPT_ASSERT_NOTREACHED();
				continue;
			}
#else
			ReadUMXIndex(chunk);
#endif

			if(fileHeader.packageVersion >= 120)
			{
				// UT2003 Packages
				ReadUMXIndex(chunk);
				chunk.Skip(8);
			} else if(fileHeader.packageVersion >= 100)
			{
				// AAO Packages
				chunk.Skip(4);
				ReadUMXIndex(chunk);
				chunk.Skip(4);
			} else if(fileHeader.packageVersion >= 62)
			{
				// UT Packages
				// Mech8.umx and a few other UT tunes have packageVersion = 62.
				// In CUnSound.cpp, the condition above reads "packageVersion >= 63" but if that is used, those tunes won't load properly.
				ReadUMXIndex(chunk);
				chunk.Skip(4);
			} else
			{
				// Old Unreal Packagaes
				ReadUMXIndex(chunk);
			}

			int32 size = ReadUMXIndex(chunk);

			ContainerItem item;

			if(objName >= 0 && static_cast<std::size_t>(objName) < names.size())
			{
				item.name = mpt::ToUnicode(mpt::CharsetISO8859_1, names[objName]);
			}

			item.file = chunk.ReadChunk(size);

			containerItems.push_back(std::move(item));

		}
	}

	return !containerItems.empty();
}


OPENMPT_NAMESPACE_END
