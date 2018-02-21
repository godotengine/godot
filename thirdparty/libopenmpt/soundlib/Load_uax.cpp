/*
 * Load_uax.cpp
 * ------------
 * Purpose: UAX (Unreal Sounds) module ripper
 * Notes  : The sounds are read into module sample slots.
 * Authors: Johannes Schultz (inspired by code from http://wiki.beyondunreal.com/Legacy:Package_File_Format)
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"
#include "UMXTools.h"


OPENMPT_NAMESPACE_BEGIN


bool CSoundFile::ReadUAX(FileReader &file, ModLoadingFlags loadFlags)
{
	file.Rewind();
	UMXFileHeader fileHeader;
	if(!file.ReadStruct(fileHeader)
		|| memcmp(fileHeader.magic, "\xC1\x83\x2A\x9E", 4)
		|| fileHeader.nameCount == 0
		|| fileHeader.exportCount == 0
		|| fileHeader.importCount == 0
		)
	{
		return false;
	}

	// Note that this can be a false positive, e.g. Unreal maps will have music and sound
	// in their name table because they usually import such files. However, it spares us
	// from wildly seeking through the file, as the name table is usually right at the
	// start of the file, so it is hopefully a good enough heuristic for our purposes.
	if(!FindUMXNameTableEntry(file, fileHeader, "sound"))
	{
		return false;
	}

	if(loadFlags == onlyVerifyHeader)
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
	InitializeGlobals();
	m_madeWithTracker = mpt::format(MPT_USTRING("Unreal Package v%1"))(fileHeader.packageVersion);
	
	for(uint32 i = 0; i < fileHeader.exportCount && file.CanRead(4); i++)
	{
		int32 objClass, objOffset, objSize, objName;
		ReadUMXExportTableEntry(file, objClass, objOffset, objSize, objName, fileHeader.packageVersion);

		if(objSize <= 0 || objClass >= 0)
		{
			continue;
		}

		// Look up object class name (we only want sounds).
		objClass = -objClass - 1;
		bool isSound = false;
		if(static_cast<size_t>(objClass) < classes.size())
		{
			isSound = (names[classes[objClass]] == "sound");
		}
		if(!isSound)
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

			FileReader fileChunk = chunk.ReadChunk(size);

			if(GetNumSamples() < MAX_SAMPLES - 1)
			{
				// Read as sample
				if(ReadSampleFromFile(GetNumSamples() + 1, fileChunk, true))
				{
					if(static_cast<size_t>(objName) < names.size())
					{
						mpt::String::Copy(m_szNames[GetNumSamples()], names[objName]);
					}
				}
			}
		}
	}

	if(m_nSamples != 0)
	{
		InitializeChannels();
		SetType(MOD_TYPE_MPT);
		m_ContainerType = MOD_CONTAINERTYPE_UAX;
		m_nChannels = 4;
		Patterns.Insert(0, 64);
		Order().assign(1, 0);
		return true;
	} else
	{
		return false;
	}
}


OPENMPT_NAMESPACE_END
