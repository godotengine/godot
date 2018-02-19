/*
* UMXTools.h
* ------------
* Purpose: UMX/UAX (Unreal) helper functions
* Notes  : None.
* Authors: Johannes Schultz (inspired by code from http://wiki.beyondunreal.com/Legacy:Package_File_Format)
* The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
*/


#include "stdafx.h"
#include "Loaders.h"
#include "UMXTools.h"


OPENMPT_NAMESPACE_BEGIN


// Read compressed unreal integers - similar to MIDI integers, but signed values are possible.
template <typename Tfile>
static int32 ReadUMXIndexImpl(Tfile &chunk)
{
	enum
	{
		signMask		= 0x80,	// Highest bit of first byte indicates if value is signed
		valueMask1		= 0x3F,	// Low 6 bits of first byte are actual value
		continueMask1	= 0x40,	// Second-highest bit of first byte indicates if further bytes follow
		valueMask		= 0x7F,	// Low 7 bits of following bytes are actual value
		continueMask	= 0x80,	// Highest bit of following bytes indicates if further bytes follow
	};

	// Read first byte
	uint8 b = chunk.ReadUint8();
	bool isSigned = (b & signMask) != 0;
	int32 result = (b & valueMask1);
	int shift = 6;

	if(b & continueMask1)
	{
		// Read remaining bytes
		do
		{
			b = chunk.ReadUint8();
			int32 data = static_cast<int32>(b) & valueMask;
			data <<= shift;
			result |= data;
			shift += 7;
		} while((b & continueMask) != 0 && (shift < 32));
	}

	if(isSigned)
	{
		result = -result;
	}
	return result;
}

int32 ReadUMXIndex(FileReader &chunk)
{
	return ReadUMXIndexImpl(chunk);
}


// Returns true if the given nme exists in the name table.
template <typename TFile>
static bool FindUMXNameTableEntryImpl(TFile &file, const UMXFileHeader &fileHeader, const char *name)
{
	if(!name)
	{
		return false;
	}
	std::size_t name_len = std::strlen(name);
	if(name_len == 0)
	{
		return false;
	}
	bool result = false;
	const FileReader::off_t oldpos = file.GetPosition();
	if(file.Seek(fileHeader.nameOffset))
	{
		for(uint32 i = 0; i < fileHeader.nameCount && file.CanRead(4); i++)
		{
			if(fileHeader.packageVersion >= 64)
			{
				int32 length = ReadUMXIndexImpl(file);
				if(length <= 0)
				{
					continue;
				}
			}
			bool match = true;
			std::size_t pos = 0;
			char c = 0;
			while((c = file.ReadUint8()) != 0)
			{
				c = mpt::ToLowerCaseAscii(c);
				if(pos < name_len)
				{
					match = match && (c == name[pos]);
				}
				pos++;
			}
			if(pos != name_len)
			{
				match = false;
			}
			if(match)
			{
				result = true;
			}
			file.Skip(4);  // Object flags
		}
	}
	file.Seek(oldpos);
	return result;
}

bool FindUMXNameTableEntry(FileReader &file, const UMXFileHeader &fileHeader, const char *name)
{
	return FindUMXNameTableEntryImpl(file, fileHeader, name);
}

bool FindUMXNameTableEntryMemory(MemoryFileReader &file, const UMXFileHeader &fileHeader, const char *name)
{
	return FindUMXNameTableEntryImpl(file, fileHeader, name);
}


// Read an entry from the name table.
std::string ReadUMXNameTableEntry(FileReader &chunk, uint16 packageVersion)
{
	std::string name;
	if(packageVersion >= 64)
	{
		// String length
		int32 length = ReadUMXIndex(chunk);
		if(length <= 0)
		{
			return "";
		}
		name.reserve(length);
	}

	// Simple zero-terminated string
	uint8 chr;
	while((chr = chunk.ReadUint8()) != 0)
	{
		// Convert string to lower case
		if(chr >= 'A' && chr <= 'Z')
		{
			chr = chr - 'A' + 'a';
		}
		name.append(1, static_cast<char>(chr));
	}

	chunk.Skip(4);	// Object flags
	return name;
}


// Read complete name table.
std::vector<std::string> ReadUMXNameTable(FileReader &file, const UMXFileHeader &fileHeader)
{
	std::vector<std::string> names;
	if(!file.Seek(fileHeader.nameOffset))
	{
		return names;
	}
	names.reserve(fileHeader.nameCount);
	for(uint32 i = 0; i < fileHeader.nameCount && file.CanRead(4); i++)
	{
		names.push_back(ReadUMXNameTableEntry(file, fileHeader.packageVersion));
	}
	return names;
}


// Read an entry from the import table.
int32 ReadUMXImportTableEntry(FileReader &chunk, uint16 packageVersion)
{
	ReadUMXIndex(chunk);		// Class package
	ReadUMXIndex(chunk);		// Class name
	if(packageVersion >= 60)
	{
		chunk.Skip(4); // Package
	} else
	{
		ReadUMXIndex(chunk); // ??
	}
	return ReadUMXIndex(chunk);	// Object name (offset into the name table)
}


// Read an entry from the export table.
void ReadUMXExportTableEntry(FileReader &chunk, int32 &objClass, int32 &objOffset, int32 &objSize, int32 &objName, uint16 packageVersion)
{
	objClass = ReadUMXIndex(chunk);	// Object class
	ReadUMXIndex(chunk);			// Object parent
	if(packageVersion >= 60)
	{
		chunk.Skip(4);				// Internal package / group of the object
	}
	objName = ReadUMXIndex(chunk);	// Object name (offset into the name table)
	chunk.Skip(4);					// Object flags
	objSize = ReadUMXIndex(chunk);
	if(objSize > 0)
	{
		objOffset = ReadUMXIndex(chunk);
	}
}


OPENMPT_NAMESPACE_END
