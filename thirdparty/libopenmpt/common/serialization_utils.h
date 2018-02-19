/*
 * serialization_utils.h
 * ---------------------
 * Purpose: Serializing data to and from MPTM files.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "../common/typedefs.h"
#include "../common/mptTypeTraits.h"
#include "../common/mptIO.h"
#include "../common/Endianness.h"

#include <algorithm>
#include <bitset>
#include <ios>
#include <iosfwd>
#include <limits>
#include <string>
#include <vector>

#include <istream>
#include <ostream>

#include <cstring>

OPENMPT_NAMESPACE_BEGIN

namespace srlztn //SeRiaLiZaTioN
{

typedef std::ios::off_type Offtype;
typedef Offtype Postype;

typedef uintptr_t	DataSize;	// Data size type.
typedef uintptr_t	RposType;	// Relative position type.
typedef uintptr_t	NumType;	// Entry count type.

const DataSize invalidDatasize = DataSize(-1);

enum 
{
	SNT_PROGRESS =		0x08000000, // = 1 << 27
	SNT_FAILURE =		0x40000000, // = 1 << 30
	SNT_NOTE =			0x20000000, // = 1 << 29
	SNT_WARNING =		0x10000000, // = 1 << 28
	SNT_NONE = 0,

	SNRW_BADGIVEN_STREAM =								1	| SNT_FAILURE,

	// Read failures.
	SNR_BADSTREAM_AFTER_MAPHEADERSEEK =					2	| SNT_FAILURE,
	SNR_STARTBYTE_MISMATCH =							3	| SNT_FAILURE,
	SNR_BADSTREAM_AT_MAP_READ =							4	| SNT_FAILURE,
	SNR_INSUFFICIENT_STREAM_OFFTYPE =					5	| SNT_FAILURE,
	SNR_OBJECTCLASS_IDMISMATCH =						6	| SNT_FAILURE,
	SNR_TOO_MANY_ENTRIES_TO_READ =						7	| SNT_FAILURE,
	SNR_INSUFFICIENT_RPOSTYPE =							8	| SNT_FAILURE,

	// Read notes and warnings.
	SNR_ZEROENTRYCOUNT =								0x80	| SNT_NOTE, // 0x80 == 1 << 7
	SNR_NO_ENTRYIDS_WITH_CUSTOMID_DEFINED =				0x100	| SNT_NOTE,
	SNR_LOADING_OBJECT_WITH_LARGER_VERSION =			0x200	| SNT_NOTE,
	
	// Write failures.
	SNW_INSUFFICIENT_FIXEDSIZE =						(0x10)	| SNT_FAILURE,
	SNW_CHANGING_IDSIZE_WITH_FIXED_IDSIZESETTING =		(0x11)	| SNT_FAILURE,
	SNW_DATASIZETYPE_OVERFLOW =							(0x13)	| SNT_FAILURE,
	SNW_MAX_WRITE_COUNT_REACHED =						(0x14)	| SNT_FAILURE,
	SNW_INSUFFICIENT_DATASIZETYPE =						(0x16)	| SNT_FAILURE
};


enum
{
	IdSizeVariable = uint16_max,
	IdSizeMaxFixedSize = (uint8_max >> 1)
};

typedef int32 SsbStatus;


struct ReadEntry
{
	ReadEntry() : nIdpos(0), rposStart(0), nSize(invalidDatasize), nIdLength(0) {}

	uintptr_t nIdpos;	// Index of id start in ID array.
	RposType rposStart;	// Entry start position.
	DataSize nSize;		// Entry size.
	uint16 nIdLength;	// Length of id.
};


enum Rwf
{
	RwfWMapStartPosEntry,	// Write. True to include data start pos entry to map.
	RwfWMapSizeEntry,		// Write. True to include data size entry to map.
	RwfWMapDescEntry,		// Write. True to include description entry to map.
	RwfWVersionNum,			// Write. True to include version numeric.
	RwfRMapCached,			// Read. True if map has been cached.
	RwfRMapHasId,			// Read. True if map has IDs
	RwfRMapHasStartpos,		// Read. True if map data start pos.
	RwfRMapHasSize,			// Read. True if map has entry size.
	RwfRMapHasDesc,			// Read. True if map has entry description.
	RwfRTwoBytesDescChar,	// Read. True if map description characters are two bytes.
	RwfRHeaderIsRead,		// Read. True when header is read.
	RwfRwHasMap,			// Read/write. True if map exists.
	RwfNumFlags
};


template<class T>
inline void Binarywrite(std::ostream& oStrm, const T& data)
{
	mpt::IO::WriteIntLE(oStrm, data);
}

template<>
inline void Binarywrite(std::ostream& oStrm, const float& data)
{
	IEEE754binary32LE tmp = IEEE754binary32LE(data);
	mpt::IO::Write(oStrm, tmp);
}

template<>
inline void Binarywrite(std::ostream& oStrm, const double& data)
{
	IEEE754binary64LE tmp = IEEE754binary64LE(data);
	mpt::IO::Write(oStrm, tmp);
}

template <class T>
inline void WriteItem(std::ostream& oStrm, const T& data)
{
	static_assert(std::is_trivial<T>::value == true, "");
	Binarywrite(oStrm, data);
}

void WriteItemString(std::ostream& oStrm, const std::string &str);

template <>
inline void WriteItem<std::string>(std::ostream& oStrm, const std::string& str) {WriteItemString(oStrm, str);}


template<class T>
inline void Binaryread(std::istream& iStrm, T& data)
{
	mpt::IO::ReadIntLE(iStrm, data);
}

template<>
inline void Binaryread(std::istream& iStrm, float& data)
{
	IEEE754binary32LE tmp = IEEE754binary32LE(0.0f);
	mpt::IO::Read(iStrm, tmp);
	data = tmp;
}

template<>
inline void Binaryread(std::istream& iStrm, double& data)
{
	IEEE754binary64LE tmp = IEEE754binary64LE(0.0);
	mpt::IO::Read(iStrm, tmp);
	data = tmp;
}

//Read only given number of bytes to the beginning of data; data bytes are memset to 0 before reading.
template <class T>
inline void Binaryread(std::istream& iStrm, T& data, const Offtype bytecount)
{
	mpt::IO::ReadBinaryTruncatedLE(iStrm, data, static_cast<std::size_t>(bytecount));
}

template <>
inline void Binaryread<float>(std::istream& iStrm, float& data, const Offtype bytecount)
{
	typedef IEEE754binary32LE T;
	mpt::byte bytes[sizeof(T)];
	std::memset(bytes, 0, sizeof(T));
	mpt::IO::ReadRaw(iStrm, bytes, std::min(static_cast<std::size_t>(bytecount), sizeof(T)));
	// There is not much we can sanely do for truncated floats,
	// thus we ignore what we just read and return 0.
	data = 0.0f;
}

template <>
inline void Binaryread<double>(std::istream& iStrm, double& data, const Offtype bytecount)
{
	typedef IEEE754binary64LE T;
	mpt::byte bytes[sizeof(T)];
	std::memset(bytes, 0, sizeof(T));
	mpt::IO::ReadRaw(iStrm, bytes, std::min(static_cast<std::size_t>(bytecount), sizeof(T)));
	// There is not much we can sanely do for truncated floats,
	// thus we ignore what we just read and return 0.
	data = 0.0;
}


template <class T>
inline void ReadItem(std::istream& iStrm, T& data, const DataSize nSize)
{
	static_assert(std::is_trivial<T>::value == true, "");
	if (nSize == sizeof(T) || nSize == invalidDatasize)
		Binaryread(iStrm, data);
	else
		Binaryread(iStrm, data, nSize);
}

void ReadItemString(std::istream& iStrm, std::string& str, const DataSize);

template <>
inline void ReadItem<std::string>(std::istream& iStrm, std::string& str, const DataSize nSize)
{
	ReadItemString(iStrm, str, nSize);
}



class ID
{
private:
	std::string m_ID; // NOTE: can contain null characters ('\0')
public:
	ID() { }
	ID(const std::string &id) : m_ID(id) { }
	ID(const char *beg, const char *end) : m_ID(beg, end) { }
	ID(const char *id) : m_ID(id?id:"") { }
	ID(const char * str, std::size_t len) : m_ID(str, str + len) { }
	template <typename T>
	static ID FromInt(const T &val)
	{
		STATIC_ASSERT(std::numeric_limits<T>::is_integer);
		typename mpt::make_le<T>::type valle;
		valle = val;
		return ID(std::string(mpt::as_raw_memory(valle), mpt::as_raw_memory(valle) + sizeof(valle)));
	}
	bool IsPrintable() const;
	mpt::ustring AsString() const;
	const char *GetBytes() const { return m_ID.c_str(); }
	std::size_t GetSize() const { return m_ID.length(); }
	bool operator == (const ID &other) const { return m_ID == other.m_ID; }
	bool operator != (const ID &other) const { return m_ID != other.m_ID; }
};



class Ssb
{

protected:

	Ssb();

public:

	SsbStatus GetStatus() const
	{
		return m_Status;
	}

protected:

	// When writing, returns the number of entries written.
	// When reading, returns the number of entries read not including unrecognized entries.
	NumType GetCounter() const {return m_nCounter;}

	void SetFlag(Rwf flag, bool val) {m_Flags.set(flag, val);}
	bool GetFlag(Rwf flag) const {return m_Flags[flag];}

protected:

	SsbStatus m_Status;

	uint32 m_nFixedEntrySize;			// Read/write: If > 0, data entries have given fixed size.

	Postype m_posStart;					// Read/write: Stream position at the beginning of object.

	uint16 m_nIdbytes;					// Read/Write: Tells map ID entry size in bytes. If size is variable, value is IdSizeVariable.
	NumType m_nCounter;					// Read/write: Keeps count of entries written/read.

	std::bitset<RwfNumFlags> m_Flags;	// Read/write: Various flags.

protected:

	static const uint8 s_DefaultFlagbyte = 0;
	static const char s_EntryID[3];

};



class SsbRead
	: public Ssb
{

public:

	enum ReadRv // Read return value.
	{
		EntryRead,
		EntryNotFound
	};
	enum IdMatchStatus
	{
		IdMatch, IdMismatch
	};
	typedef std::vector<ReadEntry>::const_iterator ReadIterator;

	SsbRead(std::istream& iStrm);

	// Call this to begin reading: must be called before other read functions.
	void BeginRead(const ID &id, const uint64& nVersion);

	// After calling BeginRead(), this returns number of entries in the file.
	NumType GetNumEntries() const {return m_nReadEntrycount;}

	// Returns read iterator to the beginning of entries.
	// The behaviour of read iterators is undefined if map doesn't
	// contain entry ids or data begin positions.
	ReadIterator GetReadBegin();

	// Returns read iterator to the end(one past last) of entries.
	ReadIterator GetReadEnd();

	// Compares given id with read entry id 
	IdMatchStatus CompareId(const ReadIterator& iter, const ID &id);

	uint64 GetReadVersion() {return m_nReadVersion;}

	// Read item using default read implementation.
	template <class T>
	ReadRv ReadItem(T& obj, const ID &id) {return ReadItem(obj, id, srlztn::ReadItem<T>);}

	// Read item using given function.
	template <class T, class FuncObj>
	ReadRv ReadItem(T& obj, const ID &id, FuncObj);

	// Read item using read iterator.
	template <class T>
	ReadRv ReadIterItem(const ReadIterator& iter, T& obj) {return ReadIterItem(iter, obj, srlztn::ReadItem<T>);}
	template <class T, class FuncObj>
	ReadRv ReadIterItem(const ReadIterator& iter, T& obj, FuncObj func);

private:

	// Reads map to cache.
	void CacheMap();

	// Searches for entry with given ID. If found, returns pointer to corresponding entry, else
	// returns nullptr.
	const ReadEntry* Find(const ID &id);

	// Called after reading an object.
	ReadRv OnReadEntry(const ReadEntry* pE, const ID &id, const Postype& posReadBegin);

	void AddReadNote(const SsbStatus s);

	// Called after reading entry. pRe is a pointer to associated map entry if exists.
	void AddReadNote(const ReadEntry* const pRe, const NumType nNum);

	void ResetReadstatus();

private:

	//  mapData is a cache that facilitates faster access to the stored data
	// without having to reparse on every access.
	//  Iterator invalidation in CacheMap() is not a problem because every code
	// path that ever returns an iterator into mapData does CacheMap exactly once
	// beforehand. Following calls use this already cached map. As the data is
	// immutable when reading, there is no need to ever invalidate the cache and
	// redo CacheMap().

	std::istream* m_pIstrm;					// Read: Pointer to read stream.

	std::vector<char> m_Idarray;		// Read: Holds entry ids.

	std::vector<ReadEntry> mapData;		// Read: Contains map information.
	uint64 m_nReadVersion;				// Read: Version is placed here when reading.
	RposType m_rposMapBegin;			// Read: If map exists, rpos of map begin, else m_rposEndofHdrData.
	Postype m_posMapEnd;				// Read: If map exists, map end position, else pos of end of hdrData.
	Postype m_posDataBegin;				// Read: Data begin position.
	RposType m_rposEndofHdrData;		// Read: rpos of end of header data.
	NumType m_nReadEntrycount;			// Read: Number of entries.

	NumType m_nNextReadHint;			// Read: Hint where to start looking for the next read entry.

};



class SsbWrite
	: public Ssb
{

public:

	SsbWrite(std::ostream& oStrm);

	// Write header
	void BeginWrite(const ID &id, const uint64& nVersion);

	// Write item using default write implementation.
	template <class T>
	void WriteItem(const T& obj, const ID &id) {WriteItem(obj, id, &srlztn::WriteItem<T>);}

	// Write item using given function.
	template <class T, class FuncObj>
	void WriteItem(const T& obj, const ID &id, FuncObj);

	// Writes mapping.
	void FinishWrite();

private:

	// Called after writing an item.
	void OnWroteItem(const ID &id, const Postype& posBeforeWrite);

	void AddWriteNote(const SsbStatus s);
	void AddWriteNote(const ID &id,
		const NumType nEntryNum,
		const DataSize nBytecount,
		const RposType rposStart);

	// Writes mapping item to mapstream.
	void WriteMapItem(const ID &id,
		const RposType& rposDataStart,
		const DataSize& nDatasize,
		const char* pszDesc);

	void ResetWritestatus() {m_Status = SNT_NONE;}

	void IncrementWriteCounter();

private:

	std::ostream* m_pOstrm;				// Write: Pointer to write stream.

	Postype m_posEntrycount;			// Write: Pos of entrycount field. 
	Postype m_posMapPosField;			// Write: Pos of map position field.
	std::string m_MapStreamString;				// Write: Map stream string.

};


template <class T, class FuncObj>
void SsbWrite::WriteItem(const T& obj, const ID &id, FuncObj Func)
{
	const Postype pos = m_pOstrm->tellp();
	Func(*m_pOstrm, obj);
	OnWroteItem(id, pos);
}

template <class T, class FuncObj>
SsbRead::ReadRv SsbRead::ReadItem(T& obj, const ID &id, FuncObj Func)
{
	const ReadEntry* pE = Find(id);
	const Postype pos = m_pIstrm->tellg();
	if (pE != nullptr || GetFlag(RwfRMapHasId) == false)
		Func(*m_pIstrm, obj, (pE) ? (pE->nSize) : invalidDatasize);
	return OnReadEntry(pE, id, pos);
}


template <class T, class FuncObj>
SsbRead::ReadRv SsbRead::ReadIterItem(const ReadIterator& iter, T& obj, FuncObj func)
{
	m_pIstrm->clear();
	if (iter->rposStart != 0)
		m_pIstrm->seekg(m_posStart + Postype(iter->rposStart));
	const Postype pos = m_pIstrm->tellg();
	func(*m_pIstrm, obj, iter->nSize);
	return OnReadEntry(&(*iter), ID(&m_Idarray[iter->nIdpos], iter->nIdLength), pos);
}


inline SsbRead::IdMatchStatus SsbRead::CompareId(const ReadIterator& iter, const ID &id)
{
	if(iter->nIdpos >= m_Idarray.size()) return IdMismatch;
	return (id == ID(&m_Idarray[iter->nIdpos], iter->nIdLength)) ? IdMatch : IdMismatch;
}


inline SsbRead::ReadIterator SsbRead::GetReadBegin()
{
	MPT_ASSERT(GetFlag(RwfRMapHasId) && (GetFlag(RwfRMapHasStartpos) || GetFlag(RwfRMapHasSize) || m_nFixedEntrySize > 0));
	if (GetFlag(RwfRMapCached) == false)
		CacheMap();
	return mapData.begin();
}


inline SsbRead::ReadIterator SsbRead::GetReadEnd()
{
	if (GetFlag(RwfRMapCached) == false)
		CacheMap();
	return mapData.end();
}


template <class T>
struct VectorWriter
{
	VectorWriter(size_t nCount) : m_nCount(nCount) {}
	void operator()(std::ostream &oStrm, const std::vector<T> &vec)
	{
		for(size_t i = 0; i < m_nCount; i++)
		{
			Binarywrite(oStrm, vec[i]);
		}
	}
	size_t m_nCount;
};

template <class T>
struct VectorReader
{
	VectorReader(size_t nCount) : m_nCount(nCount) {}
	void operator()(std::istream& iStrm, std::vector<T> &vec, const size_t)
	{
		vec.resize(m_nCount);
		for(std::size_t i = 0; i < m_nCount; ++i)
		{
			Binaryread(iStrm, vec[i]);
		}
	}
	size_t m_nCount;
};

template <class T>
struct ArrayReader
{
	ArrayReader(size_t nCount) : m_nCount(nCount) {}
	void operator()(std::istream& iStrm, T* pData, const size_t)
	{
		for(std::size_t i=0; i<m_nCount; ++i)
		{
			Binaryread(iStrm, pData[i]);
		}
	} 
	size_t m_nCount;
};



} //namespace srlztn.


OPENMPT_NAMESPACE_END
