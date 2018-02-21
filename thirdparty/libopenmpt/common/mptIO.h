/*
 * mptIO.h
 * -------
 * Purpose: Basic functions for reading/writing binary and endian safe data to/from files/streams.
 * Notes  : This is work-in-progress.
 *          Some useful functions for reading and writing are still missing.
 * Authors: Joern Heusipp
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


#include "../common/typedefs.h"
#include "../common/mptTypeTraits.h"
#include "../common/Endianness.h"
#include <algorithm>
#include <iosfwd>
#include <limits>
#include <cstring>

#if defined(MPT_ENABLE_FILEIO_STDIO)
#include <cstdio>
#include <stdio.h>
#endif // MPT_ENABLE_FILEIO_STDIO


OPENMPT_NAMESPACE_BEGIN


namespace mpt {

namespace IO {

typedef int64 Offset;

static const std::size_t BUFFERSIZE_TINY   =    1 * 1024; // on stack usage
static const std::size_t BUFFERSIZE_SMALL  =    4 * 1024; // on heap
static const std::size_t BUFFERSIZE_NORMAL =   64 * 1024; // FILE I/O
static const std::size_t BUFFERSIZE_LARGE  = 1024 * 1024;



// Returns true iff 'off' fits into 'Toff'.
template < typename Toff >
inline bool OffsetFits(IO::Offset off)
{
	return (static_cast<IO::Offset>(mpt::saturate_cast<Toff>(off)) == off);
}



bool IsValid(std::ostream & f);
bool IsValid(std::istream & f);
bool IsValid(std::iostream & f);
IO::Offset TellRead(std::istream & f);
IO::Offset TellWrite(std::ostream & f);
bool SeekBegin(std::ostream & f);
bool SeekBegin(std::istream & f);
bool SeekBegin(std::iostream & f);
bool SeekEnd(std::ostream & f);
bool SeekEnd(std::istream & f);
bool SeekEnd(std::iostream & f);
bool SeekAbsolute(std::ostream & f, IO::Offset pos);
bool SeekAbsolute(std::istream & f, IO::Offset pos);
bool SeekAbsolute(std::iostream & f, IO::Offset pos);
bool SeekRelative(std::ostream & f, IO::Offset off);
bool SeekRelative(std::istream & f, IO::Offset off);
bool SeekRelative(std::iostream & f, IO::Offset off);
IO::Offset ReadRawImpl(std::istream & f, mpt::byte * data, std::size_t size);
bool WriteRawImpl(std::ostream & f, const mpt::byte * data, std::size_t size);
bool IsEof(std::istream & f);
bool Flush(std::ostream & f);



#if defined(MPT_ENABLE_FILEIO_STDIO)

bool IsValid(FILE* & f);
IO::Offset TellRead(FILE* & f);
IO::Offset TellWrite(FILE* & f);
bool SeekBegin(FILE* & f);
bool SeekEnd(FILE* & f);
bool SeekAbsolute(FILE* & f, IO::Offset pos);
bool SeekRelative(FILE* & f, IO::Offset off);
IO::Offset ReadRawImpl(FILE * & f, mpt::byte * data, std::size_t size);
bool WriteRawImpl(FILE* & f, const mpt::byte * data, std::size_t size);
bool IsEof(FILE * & f);
bool Flush(FILE* & f);

#endif // MPT_ENABLE_FILEIO_STDIO



template <typename Tbyte> bool IsValid(std::pair<mpt::span<Tbyte>, IO::Offset> & f)
{
	return (f.second >= 0);
}
template <typename Tbyte> IO::Offset TellRead(std::pair<mpt::span<Tbyte>, IO::Offset> & f)
{
	return f.second;
}
template <typename Tbyte> IO::Offset TellWrite(std::pair<mpt::span<Tbyte>, IO::Offset> & f)
{
	return f.second;
}
template <typename Tbyte> bool SeekBegin(std::pair<mpt::span<Tbyte>, IO::Offset> & f)
{
	f.second = 0;
	return true;
}
template <typename Tbyte> bool SeekEnd(std::pair<mpt::span<Tbyte>, IO::Offset> & f)
{
	f.second = f.first.size();
	return true;
}
template <typename Tbyte> bool SeekAbsolute(std::pair<mpt::span<Tbyte>, IO::Offset> & f, IO::Offset pos)
{
	f.second = pos;
	return true;
}
template <typename Tbyte> bool SeekRelative(std::pair<mpt::span<Tbyte>, IO::Offset> & f, IO::Offset off)
{
	if(f.second < 0)
	{
		return false;
	}
	f.second += off;
	return true;
}
template <typename Tbyte> IO::Offset ReadRawImpl(std::pair<mpt::span<Tbyte>, IO::Offset> & f, mpt::byte * data, std::size_t size)
{
	if(f.second < 0)
	{
		return 0;
	}
	if(f.second >= static_cast<IO::Offset>(f.first.size()))
	{
		return 0;
	}
	std::size_t num = mpt::saturate_cast<std::size_t>(std::min<IO::Offset>(f.first.size() - f.second, size));
	std::copy(mpt::byte_cast<const mpt::byte*>(f.first.data() + f.second), mpt::byte_cast<const mpt::byte*>(f.first.data() + f.second + num), data);
	f.second += num;
	return num;
}
template <typename Tbyte> bool WriteRawImpl(std::pair<mpt::span<Tbyte>, IO::Offset> & f, const mpt::byte * data, std::size_t size)
{
	if(f.second < 0)
	{
		return false;
	}
	if(f.second >= static_cast<IO::Offset>(f.first.size()))
	{
		return false;
	}
	std::size_t num = mpt::saturate_cast<std::size_t>(std::min<IO::Offset>(f.first.size() - f.second, size));
	if(num != size)
	{
		return false;
	}
	std::copy(data, data + num, mpt::byte_cast<mpt::byte*>(f.first.data() + f.second));
	f.second += num;
	return true;
}
template <typename Tbyte> bool IsEof(std::pair<mpt::span<Tbyte>, IO::Offset> & f)
{
	return (f.second >= static_cast<IO::Offset>(f.first.size()));
}
template <typename Tbyte> bool Flush(std::pair<mpt::span<Tbyte>, IO::Offset> & f)
{
	MPT_UNREFERENCED_PARAMTER(f);
	return true;
}



template <typename Tbyte, typename Tfile>
inline IO::Offset ReadRaw(Tfile & f, Tbyte * data, std::size_t size)
{
	return IO::ReadRawImpl(f, mpt::byte_cast<mpt::byte*>(data), size);
}

template <typename Tbyte, typename Tfile>
inline bool WriteRaw(Tfile & f, const Tbyte * data, std::size_t size)
{
	return IO::WriteRawImpl(f, mpt::byte_cast<const mpt::byte*>(data), size);
}

template <typename Tbinary, typename Tfile>
inline bool Read(Tfile & f, Tbinary & v)
{
	return IO::ReadRaw(f, mpt::as_raw_memory(v), sizeof(Tbinary)) == sizeof(Tbinary);
}

template <typename Tbinary, typename Tfile>
inline bool Write(Tfile & f, const Tbinary & v)
{
	return IO::WriteRaw(f, mpt::as_raw_memory(v), sizeof(Tbinary));
}

template <typename T, typename Tfile>
inline bool WritePartial(Tfile & f, const T & v, size_t size = sizeof(T))
{
	MPT_ASSERT(size <= sizeof(T));
	return IO::WriteRaw(f, mpt::as_raw_memory(v), size);
}

template <typename T, typename Tfile>
inline bool ReadBinaryTruncatedLE(Tfile & f, T & v, std::size_t size)
{
	bool result = false;
	MPT_STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	mpt::byte bytes[sizeof(T)];
	std::memset(bytes, 0, sizeof(T));
	const IO::Offset readResult = IO::ReadRaw(f, bytes, std::min(size, sizeof(T)));
	if(readResult < 0)
	{
		result = false;
	} else
	{
		result = (static_cast<uint64>(readResult) == std::min(size, sizeof(T)));
	}
	#ifdef MPT_PLATFORM_BIG_ENDIAN
		std::reverse(bytes, bytes + sizeof(T));
	#endif
	std::memcpy(&v, bytes, sizeof(T));
	return result;
}

template <typename T, typename Tfile>
inline bool ReadIntLE(Tfile & f, T & v)
{
	bool result = false;
	STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	mpt::byte bytes[sizeof(T)];
	std::memset(bytes, 0, sizeof(T));
	const IO::Offset readResult = IO::ReadRaw(f, bytes, sizeof(T));
	if(readResult < 0)
	{
		result = false;
	} else
	{
		result = (static_cast<uint64>(readResult) == sizeof(T));
	}
	T val = 0;
	std::memcpy(&val, bytes, sizeof(T));
	v = SwapBytesLE(val);
	return result;
}

template <typename T, typename Tfile>
inline bool ReadIntBE(Tfile & f, T & v)
{
	bool result = false;
	STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	mpt::byte bytes[sizeof(T)];
	std::memset(bytes, 0, sizeof(T));
	const IO::Offset readResult = IO::ReadRaw(f, bytes, sizeof(T));
	if(readResult < 0)
	{
		result = false;
	} else
	{
		result = (static_cast<uint64>(readResult) == sizeof(T));
	}
	T val = 0;
	std::memcpy(&val, bytes, sizeof(T));
	v = SwapBytesBE(val);
	return result;
}

template <typename Tfile>
inline bool ReadAdaptiveInt16LE(Tfile & f, uint16 & v)
{
	bool result = true;
	mpt::byte byte = 0;
	std::size_t additionalBytes = 0;
	v = 0;
	byte = 0;
	if(!IO::ReadIntLE<mpt::byte>(f, byte)) result = false;
	additionalBytes = (byte & 0x01);
	v = byte >> 1;
	for(std::size_t i = 0; i < additionalBytes; ++i)
	{
		byte = 0;
		if(!IO::ReadIntLE<mpt::byte>(f, byte)) result = false;
		v |= (static_cast<uint16>(byte) << (((i+1)*8) - 1));
	}
	return result;
}

template <typename Tfile>
inline bool ReadAdaptiveInt32LE(Tfile & f, uint32 & v)
{
	bool result = true;
	mpt::byte byte = 0;
	std::size_t additionalBytes = 0;
	v = 0;
	byte = 0;
	if(!IO::ReadIntLE<mpt::byte>(f, byte)) result = false;
	additionalBytes = (byte & 0x03);
	v = byte >> 2;
	for(std::size_t i = 0; i < additionalBytes; ++i)
	{
		byte = 0;
		if(!IO::ReadIntLE<mpt::byte>(f, byte)) result = false;
		v |= (static_cast<uint32>(byte) << (((i+1)*8) - 2));
	}
	return result;
}

template <typename Tfile>
inline bool ReadAdaptiveInt64LE(Tfile & f, uint64 & v)
{
	bool result = true;
	mpt::byte byte = 0;
	std::size_t additionalBytes = 0;
	v = 0;
	byte = 0;
	if(!IO::ReadIntLE<mpt::byte>(f, byte)) result = false;
	additionalBytes = (1 << (byte & 0x03)) - 1;
	v = byte >> 2;
	for(std::size_t i = 0; i < additionalBytes; ++i)
	{
		byte = 0;
		if(!IO::ReadIntLE<mpt::byte>(f, byte)) result = false;
		v |= (static_cast<uint64>(byte) << (((i+1)*8) - 2));
	}
	return result;
}

template <typename Tsize, typename Tfile>
inline bool ReadSizedStringLE(Tfile & f, std::string & str, Tsize maxSize = std::numeric_limits<Tsize>::max())
{
	STATIC_ASSERT(std::numeric_limits<Tsize>::is_integer);
	str.clear();
	Tsize size = 0;
	if(!mpt::IO::ReadIntLE(f, size))
	{
		return false;
	}
	if(size > maxSize)
	{
		return false;
	}
	for(Tsize i = 0; i != size; ++i)
	{
		char c = '\0';
		if(!mpt::IO::ReadIntLE(f, c))
		{
			return false;
		}
		str.push_back(c);
	}
	return true;
}


template <typename T, typename Tfile>
inline bool WriteIntLE(Tfile & f, const T v)
{
	STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	const T val = SwapBytesLE(v);
	mpt::byte bytes[sizeof(T)];
	std::memcpy(bytes, &val, sizeof(T));
	return IO::WriteRaw(f, bytes, sizeof(T));
}

template <typename T, typename Tfile>
inline bool WriteIntBE(Tfile & f, const T v)
{
	STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	const T val = SwapBytesBE(v);
	mpt::byte bytes[sizeof(T)];
	std::memcpy(bytes, &val, sizeof(T));
	return IO::WriteRaw(f, bytes, sizeof(T));
}

template <typename Tfile>
inline bool WriteAdaptiveInt16LE(Tfile & f, const uint16 v, std::size_t minSize = 0, std::size_t maxSize = 0)
{
	MPT_ASSERT(minSize == 0 || minSize == 1 || minSize == 2);
	MPT_ASSERT(maxSize == 0 || maxSize == 1 || maxSize == 2);
	MPT_ASSERT(maxSize == 0 || maxSize >= minSize);
	if(v < 0x80 && minSize <= 1 && (1 <= maxSize || maxSize == 0))
	{
		return IO::WriteIntLE<uint8>(f, static_cast<uint8>(v << 1) | 0x00);
	} else if(v < 0x8000 && minSize <= 2 && (2 <= maxSize || maxSize == 0))
	{
		return IO::WriteIntLE<uint16>(f, static_cast<uint16>(v << 1) | 0x01);
	} else
	{
		MPT_ASSERT_NOTREACHED();
		return false;
	}
}

template <typename Tfile>
inline bool WriteAdaptiveInt32LE(Tfile & f, const uint32 v, std::size_t minSize = 0, std::size_t maxSize = 0)
{
	MPT_ASSERT(minSize == 0 || minSize == 1 || minSize == 2 || minSize == 3 || minSize == 4);
	MPT_ASSERT(maxSize == 0 || maxSize == 1 || maxSize == 2 || maxSize == 3 || maxSize == 4);
	MPT_ASSERT(maxSize == 0 || maxSize >= minSize);
	if(v < 0x40 && minSize <= 1 && (1 <= maxSize || maxSize == 0))
	{
		return IO::WriteIntLE<uint8>(f, static_cast<uint8>(v << 2) | 0x00);
	} else if(v < 0x4000 && minSize <= 2 && (2 <= maxSize || maxSize == 0))
	{
		return IO::WriteIntLE<uint16>(f, static_cast<uint16>(v << 2) | 0x01);
	} else if(v < 0x400000 && minSize <= 3 && (3 <= maxSize || maxSize == 0))
	{
		uint32 value = static_cast<uint32>(v << 2) | 0x02;
		mpt::byte bytes[3];
		bytes[0] = static_cast<mpt::byte>(value >>  0);
		bytes[1] = static_cast<mpt::byte>(value >>  8);
		bytes[2] = static_cast<mpt::byte>(value >> 16);
		return IO::WriteRaw(f, bytes, 3);
	} else if(v < 0x40000000 && minSize <= 4 && (4 <= maxSize || maxSize == 0))
	{
		return IO::WriteIntLE<uint32>(f, static_cast<uint32>(v << 2) | 0x03);
	} else
	{
		MPT_ASSERT_NOTREACHED();
		return false;
	}
}

template <typename Tfile>
inline bool WriteAdaptiveInt64LE(Tfile & f, const uint64 v, std::size_t minSize = 0, std::size_t maxSize = 0)
{
	MPT_ASSERT(minSize == 0 || minSize == 1 || minSize == 2 || minSize == 4 || minSize == 8);
	MPT_ASSERT(maxSize == 0 || maxSize == 1 || maxSize == 2 || maxSize == 4 || maxSize == 8);
	MPT_ASSERT(maxSize == 0 || maxSize >= minSize);
	if(v < 0x40 && minSize <= 1 && (1 <= maxSize || maxSize == 0))
	{
		return IO::WriteIntLE<uint8>(f, static_cast<uint8>(v << 2) | 0x00);
	} else if(v < 0x4000 && minSize <= 2 && (2 <= maxSize || maxSize == 0))
	{
		return IO::WriteIntLE<uint16>(f, static_cast<uint16>(v << 2) | 0x01);
	} else if(v < 0x40000000 && minSize <= 4 && (4 <= maxSize || maxSize == 0))
	{
		return IO::WriteIntLE<uint32>(f, static_cast<uint32>(v << 2) | 0x02);
	} else if(v < 0x4000000000000000ull && minSize <= 8 && (8 <= maxSize || maxSize == 0))
	{
		return IO::WriteIntLE<uint64>(f, static_cast<uint64>(v << 2) | 0x03);
	} else
	{
		MPT_ASSERT_NOTREACHED();
		return false;
	}
}

// Write a variable-length integer, as found in MIDI files. The number of written bytes is placed in the bytesWritten parameter.
template <typename Tfile, typename T>
bool WriteVarInt(Tfile & f, const T v, size_t *bytesWritten = nullptr)
{
	STATIC_ASSERT(std::numeric_limits<T>::is_integer);
	STATIC_ASSERT(!std::numeric_limits<T>::is_signed);
	mpt::byte out[(sizeof(T) * 8 + 6) / 7];
	size_t numBytes = 0;
	for(uint32 n = (sizeof(T) * 8) / 7; n > 0; n--)
	{
		if(v >= (static_cast<T>(1) << (n * 7u)))
		{
			out[numBytes++] = static_cast<mpt::byte>(((v >> (n * 7u)) & 0x7F) | 0x80);
		}
	}
	out[numBytes++] = static_cast<mpt::byte>(v & 0x7F);
	MPT_ASSERT(numBytes <= mpt::size(out));
	if(bytesWritten != nullptr) *bytesWritten = numBytes;
	return mpt::IO::WriteRaw(f, out, numBytes);
}

template <typename Tsize, typename Tfile>
inline bool WriteSizedStringLE(Tfile & f, const std::string & str)
{
	STATIC_ASSERT(std::numeric_limits<Tsize>::is_integer);
	if(str.size() > std::numeric_limits<Tsize>::max())
	{
		return false;
	}
	Tsize size = static_cast<Tsize>(str.size());
	if(!mpt::IO::WriteIntLE(f, size))
	{
		return false;
	}
	if(!mpt::IO::WriteRaw(f, str.data(), str.size()))
	{
		return false;
	}
	return true;
}

template <typename Tfile>
inline bool WriteText(Tfile &f, const std::string &s)
{
	return mpt::IO::WriteRaw(f, s.data(), s.size());
}

template <typename Tfile>
inline bool WriteTextCRLF(Tfile &f)
{
	return mpt::IO::WriteText(f, "\r\n");
}

template <typename Tfile>
inline bool WriteTextLF(Tfile &f)
{
	return mpt::IO::WriteText(f, "\n");
}

template <typename Tfile>
inline bool WriteTextCRLF(Tfile &f, const std::string &s)
{
	return mpt::IO::WriteText(f, s) && mpt::IO::WriteTextCRLF(f);
}

template <typename Tfile>
inline bool WriteTextLF(Tfile &f, const std::string &s)
{
	return mpt::IO::WriteText(f, s) && mpt::IO::WriteTextLF(f);
}

} // namespace IO


} // namespace mpt



#if defined(MPT_FILEREADER_STD_ISTREAM)

class IFileDataContainer {
public:
	typedef std::size_t off_t;
protected:
	IFileDataContainer() { }
public:
	virtual ~IFileDataContainer() { }
public:
	virtual bool IsValid() const = 0;
	virtual bool HasFastGetLength() const = 0;
	virtual bool HasPinnedView() const = 0;
	virtual const mpt::byte *GetRawData() const = 0;
	virtual off_t GetLength() const = 0;
	virtual off_t Read(mpt::byte *dst, off_t pos, off_t count) const = 0;

	virtual bool CanRead(off_t pos, off_t length) const
	{
		off_t dataLength = GetLength();
		if((pos == dataLength) && (length == 0))
		{
			return true;
		}
		if(pos >= dataLength)
		{
			return false;
		}
		return length <= dataLength - pos;
	}

	virtual off_t GetReadableLength(off_t pos, off_t length) const
	{
		off_t dataLength = GetLength();
		if(pos >= dataLength)
		{
			return 0;
		}
		return std::min<off_t>(length, dataLength - pos);
	}
};


class FileDataContainerDummy : public IFileDataContainer {
public:
	FileDataContainerDummy() { }
	virtual ~FileDataContainerDummy() { }
public:
	bool IsValid() const
	{
		return false;
	}

	bool HasFastGetLength() const
	{
		return true;
	}

	bool HasPinnedView() const
	{
		return true;
	}

	const mpt::byte *GetRawData() const
	{
		return nullptr;
	}

	off_t GetLength() const
	{
		return 0;
	}
	off_t Read(mpt::byte * /*dst*/, off_t /*pos*/, off_t /*count*/) const
	{
		return 0;
	}
};


class FileDataContainerWindow : public IFileDataContainer
{
private:
	std::shared_ptr<const IFileDataContainer> data;
	const off_t dataOffset;
	const off_t dataLength;
public:
	FileDataContainerWindow(std::shared_ptr<const IFileDataContainer> src, off_t off, off_t len) : data(src), dataOffset(off), dataLength(len) { }
	virtual ~FileDataContainerWindow() { }

	bool IsValid() const
	{
		return data->IsValid();
	}
	bool HasFastGetLength() const
	{
		return data->HasFastGetLength();
	}
	bool HasPinnedView() const
	{
		return data->HasPinnedView();
	}
	const mpt::byte *GetRawData() const {
		return data->GetRawData() + dataOffset;
	}
	off_t GetLength() const {
		return dataLength;
	}
	off_t Read(mpt::byte *dst, off_t pos, off_t count) const
	{
		if(pos >= dataLength)
		{
			return 0;
		}
		return data->Read(dst, dataOffset + pos, std::min(count, dataLength - pos));
	}
	bool CanRead(off_t pos, off_t length) const {
		if((pos == dataLength) && (length == 0))
		{
			return true;
		}
		if(pos >= dataLength)
		{
			return false;
		}
		return (length <= dataLength - pos);
	}
	off_t GetReadableLength(off_t pos, off_t length) const
	{
		if(pos >= dataLength)
		{
			return 0;
		}
		return std::min(length, dataLength - pos);
	}
};


class FileDataContainerSeekable : public IFileDataContainer {

private:

	off_t streamLength;

	mutable bool cached;
	mutable std::vector<mpt::byte> cache;

protected:

	FileDataContainerSeekable(off_t length);
	virtual ~FileDataContainerSeekable();

private:
	
	void CacheStream() const;

public:

	bool IsValid() const;
	bool HasFastGetLength() const;
	bool HasPinnedView() const;
	const mpt::byte *GetRawData() const;
	off_t GetLength() const;
	off_t Read(mpt::byte *dst, off_t pos, off_t count) const;

private:

	virtual off_t InternalRead(mpt::byte *dst, off_t pos, off_t count) const = 0;

};


class FileDataContainerStdStreamSeekable : public FileDataContainerSeekable {

private:

	std::istream *stream;

public:

	FileDataContainerStdStreamSeekable(std::istream *s);
	virtual ~FileDataContainerStdStreamSeekable();

	static bool IsSeekable(std::istream *stream);
	static off_t GetLength(std::istream *stream);

private:

	off_t InternalRead(mpt::byte *dst, off_t pos, off_t count) const;

};


class FileDataContainerUnseekable : public IFileDataContainer {

private:

	mutable std::vector<mpt::byte> cache;
	mutable std::size_t cachesize;
	mutable bool streamFullyCached;

protected:

	FileDataContainerUnseekable();
	virtual ~FileDataContainerUnseekable();

private:

	static const std::size_t QUANTUM_SIZE = mpt::IO::BUFFERSIZE_SMALL;
	static const std::size_t BUFFER_SIZE = mpt::IO::BUFFERSIZE_NORMAL;

	void EnsureCacheBuffer(std::size_t requiredbuffersize) const;
	void CacheStream() const;
	void CacheStreamUpTo(off_t pos, off_t length) const;

private:

	void ReadCached(mpt::byte *dst, off_t pos, off_t count) const;

public:

	bool IsValid() const;
	bool HasFastGetLength() const;
	bool HasPinnedView() const;
	const mpt::byte *GetRawData() const;
	off_t GetLength() const;
	off_t Read(mpt::byte *dst, off_t pos, off_t count) const;
	bool CanRead(off_t pos, off_t length) const;
	off_t GetReadableLength(off_t pos, off_t length) const;

private:

	virtual bool InternalEof() const = 0;
	virtual off_t InternalRead(mpt::byte *dst, off_t count) const = 0;

};


class FileDataContainerStdStream : public FileDataContainerUnseekable {

private:

	std::istream *stream;

public:

	FileDataContainerStdStream(std::istream *s);
	virtual ~FileDataContainerStdStream();

private:

	bool InternalEof() const;
	off_t InternalRead(mpt::byte *dst, off_t count) const;

};


#if defined(MPT_FILEREADER_CALLBACK_STREAM)


struct CallbackStream
{
	static const int SeekSet = 0;
	static const int SeekCur = 1;
	static const int SeekEnd = 2;
	void *stream;
	std::size_t (*read)( void * stream, void * dst, std::size_t bytes );
	int (*seek)( void * stream, int64 offset, int whence );
	int64 (*tell)( void * stream );
};


class FileDataContainerCallbackStreamSeekable : public FileDataContainerSeekable
{
private:
	CallbackStream stream;
public:
	static bool IsSeekable(CallbackStream stream);
	static off_t GetLength(CallbackStream stream);
	FileDataContainerCallbackStreamSeekable(CallbackStream s);
	virtual ~FileDataContainerCallbackStreamSeekable();
private:
	off_t InternalRead(mpt::byte *dst, off_t pos, off_t count) const;
};


class FileDataContainerCallbackStream : public FileDataContainerUnseekable
{
private:
	CallbackStream stream;
	mutable bool eof_reached;
public:
	FileDataContainerCallbackStream(CallbackStream s);
	virtual ~FileDataContainerCallbackStream();
private:
	bool InternalEof() const;
	off_t InternalRead(mpt::byte *dst, off_t count) const;
};


#endif // MPT_FILEREADER_CALLBACK_STREAM


#endif


class FileDataContainerMemory
#if defined(MPT_FILEREADER_STD_ISTREAM)
	: public IFileDataContainer
#endif
{

#if !defined(MPT_FILEREADER_STD_ISTREAM)
public:
	typedef std::size_t off_t;
#endif

private:

	const mpt::byte *streamData;	// Pointer to memory-mapped file
	off_t streamLength;		// Size of memory-mapped file in bytes

public:
	FileDataContainerMemory() : streamData(nullptr), streamLength(0) { }
	FileDataContainerMemory(mpt::const_byte_span data) : streamData(data.data()), streamLength(data.size()) { }
#if defined(MPT_FILEREADER_STD_ISTREAM)
	virtual
#endif
		~FileDataContainerMemory() { }

public:

	bool IsValid() const
	{
		return streamData != nullptr;
	}

	bool HasFastGetLength() const
	{
		return true;
	}

	bool HasPinnedView() const
	{
		return true;
	}

	const mpt::byte *GetRawData() const
	{
		return streamData;
	}

	off_t GetLength() const
	{
		return streamLength;
	}

	off_t Read(mpt::byte *dst, off_t pos, off_t count) const
	{
		if(pos >= streamLength)
		{
			return 0;
		}
		off_t avail = std::min<off_t>(streamLength - pos, count);
		std::copy(streamData + pos, streamData + pos + avail, dst);
		return avail;
	}

	bool CanRead(off_t pos, off_t length) const
	{
		if((pos == streamLength) && (length == 0))
		{
			return true;
		}
		if(pos >= streamLength)
		{
			return false;
		}
		return (length <= streamLength - pos);
	}

	off_t GetReadableLength(off_t pos, off_t length) const
	{
		if(pos >= streamLength)
		{
			return 0;
		}
		return std::min<off_t>(length, streamLength - pos);
	}

};



OPENMPT_NAMESPACE_END
