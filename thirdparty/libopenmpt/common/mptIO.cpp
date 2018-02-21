/*
 * mptIO.cpp
 * ---------
 * Purpose: Basic functions for reading/writing binary and endian safe data to/from files/streams.
 * Notes  : This is work-in-progress.
 *          Some useful functions for reading and writing are still missing.
 * Authors: Joern Heusipp
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#include "mptIO.h"

#include <ios>
#include <istream>
#include <ostream>
#include <sstream>
#if MPT_COMPILER_MSVC
#include <typeinfo>
#endif // MPT_COMPILER_MSVC

#if defined(MPT_ENABLE_FILEIO_STDIO)
#include <cstdio>
#include <stdio.h>
#endif // MPT_ENABLE_FILEIO_STDIO


OPENMPT_NAMESPACE_BEGIN


namespace mpt {

namespace IO {


#if MPT_COMPILER_MSVC

// MSVC std::stringbuf (and thereby std::ostringstream, std::istringstream and
// std::stringstream) fail seekoff() when the stringbuf is currently empty.
// seekoff() can get called via tell*() or seek*() iostream members. tell*() has
// been special cased from VS2010 onwards to handle this specific case and
// changed to not fail when the stringbuf is empty.
// In addition to using out own wrapper around std::stringstream and
// std::stringbuf, we also work-around the plain native type's problem in case
// we get handed such an object from third party code. This mitigation of course
// requires using our consolidated and normalized IO functions.
// We use the following work-around strategy:
//  *  If the stream is already in failed state, we do not do any work-around
//     and bail out early.
//  *  If the underlying streambuf is not a std::stringbuf, the work-around is
//     not necessary and we skip it.
//  *  If querying the current position does not fail and returns a
//     position > 0, the underlying stringbuf is not empty and we also bail out.
//  *  Otherwise, we actually query the string contained in the stringbuf to be
//     empty. This operation is slow as it has to copy the string into a
//     temporary.
//     Note, however, that this is only ever necessary if the current position
//     is 0. If it always has been 0, the stringbuf will be empty anyway and the
//     copy does not cost anything measurable. If it got seeked to position 0,
//     we have to pay the price. However, this should be relatively uncommmon in
//     pratice.
//  *  The actual work-around consists of performing or emulating the requested
//     operation and resetting the failed state afterwards.

static bool StreamIsStringStreamAndValidAndEmpty(std::ostream & f)
{
	if(f.fail() || !f.rdbuf())
	{ // failed
		return false;
	}
	if(!dynamic_cast<std::stringbuf*>(f.rdbuf()) || (typeid(*(f.rdbuf())) != typeid(std::stringbuf)))
	{ // no stringbuf
		return false;
	}
	std::streampos pos = f.tellp();
	f.clear(f.rdstate() & ~std::ios::failbit);
	if(pos != std::streampos(-1) && pos > 0)
	{ // if the position is not 0, the streambuf is not empty
		return false;
	}
	return dynamic_cast<std::stringbuf*>(f.rdbuf())->str().empty(); // slow
}

static bool StreamIsStringStreamAndValidAndEmpty(std::istream & f)
{
	if(f.fail() || !f.rdbuf())
	{ // failed
		return false;
	}
	if(!dynamic_cast<std::stringbuf*>(f.rdbuf()) || (typeid(*(f.rdbuf())) != typeid(std::stringbuf)))
	{ // no stringbuf
		return false;
	}
	std::streampos pos = f.tellg();
	f.clear(f.rdstate() & ~std::ios::failbit);
	if(pos != std::streampos(-1) && pos > 0)
	{ // if the position is not 0, the streambuf is not empty
		return false;
	}
	return dynamic_cast<std::stringbuf*>(f.rdbuf())->str().empty(); // slow
}

static bool StreamIsStringStreamAndValidAndEmpty(std::iostream & f)
{
	if(f.fail() || !f.rdbuf())
	{ // failed
		return false;
	}
	if(!dynamic_cast<std::stringbuf*>(f.rdbuf()) || (typeid(*(f.rdbuf())) != typeid(std::stringbuf)))
	{ // no stringbuf
		return false;
	}
	std::streampos ipos = f.tellg();
	f.clear(f.rdstate() & ~std::ios::failbit);
	std::streampos opos = f.tellp();
	f.clear(f.rdstate() & ~std::ios::failbit);
	if((ipos != std::streampos(-1) && ipos > 0) || (opos != std::streampos(-1) && opos > 0))
	{ // if the position is not 0, the streambuf is not empty
		return false;
	}
	return dynamic_cast<std::stringbuf*>(f.rdbuf())->str().empty(); // slow
}

#endif // MPT_COMPILER_MSVC

//STATIC_ASSERT(sizeof(std::streamoff) == 8); // Assert 64bit file support.
bool IsValid(std::ostream & f) { return !f.fail(); }
bool IsValid(std::istream & f) { return !f.fail(); }
bool IsValid(std::iostream & f) { return !f.fail(); }
IO::Offset TellRead(std::istream & f)
{
	return f.tellg();
}
IO::Offset TellWrite(std::ostream & f)
{
	return f.tellp();
}
bool SeekBegin(std::ostream & f)
{
	#if MPT_COMPILER_MSVC
		if(StreamIsStringStreamAndValidAndEmpty(f))
		{ // VS std::stringbuf fail seek when the internal buffer is empty. Work-around it in case the stream is not already in failed state.
			f.seekp(0); f.clear(f.rdstate() & ~std::ios::failbit); return true;
		}
	#endif
	f.seekp(0); return !f.fail();
}
bool SeekBegin(std::istream & f)
{
	#if MPT_COMPILER_MSVC
		if(StreamIsStringStreamAndValidAndEmpty(f))
		{
			f.seekg(0); f.clear(f.rdstate() & ~std::ios::failbit); return true;
		}
	#endif
	f.seekg(0); return !f.fail();
}
bool SeekBegin(std::iostream & f)
{
	#if MPT_COMPILER_MSVC
		if(StreamIsStringStreamAndValidAndEmpty(f))
		{
			f.seekg(0); f.clear(f.rdstate() & ~std::ios::failbit); f.seekp(0); f.clear(f.rdstate() & ~std::ios::failbit); return true;
		}
	#endif
	f.seekg(0); f.seekp(0); return !f.fail();
}
bool SeekEnd(std::ostream & f)
{
	#if MPT_COMPILER_MSVC
		if(StreamIsStringStreamAndValidAndEmpty(f))
		{
			f.seekp(0); f.clear(f.rdstate() & ~std::ios::failbit); return true;
		}
	#endif
	f.seekp(0, std::ios::end); return !f.fail();
}
bool SeekEnd(std::istream & f)
{
	#if MPT_COMPILER_MSVC
		if(StreamIsStringStreamAndValidAndEmpty(f))
		{
			f.seekg(0); f.clear(f.rdstate() & ~std::ios::failbit); return true;
		}
	#endif
	f.seekg(0, std::ios::end); return !f.fail();
}
bool SeekEnd(std::iostream & f)
{
	#if MPT_COMPILER_MSVC
		if(StreamIsStringStreamAndValidAndEmpty(f))
		{
			f.seekg(0); f.clear(f.rdstate() & ~std::ios::failbit);  f.seekp(0); f.clear(f.rdstate() & ~std::ios::failbit); return true;
		}
	#endif
	f.seekg(0, std::ios::end); f.seekp(0, std::ios::end); return !f.fail();
}
bool SeekAbsolute(std::ostream & f, IO::Offset pos)
{
	if(!OffsetFits<std::streamoff>(pos)) { return false; }
	#if MPT_COMPILER_MSVC
		if(StreamIsStringStreamAndValidAndEmpty(f))
		{
			if(pos == 0)
			{
				f.seekp(static_cast<std::streamoff>(pos), std::ios::beg); f.clear(f.rdstate() & ~std::ios::failbit); return true;
			}
		}
	#endif
	f.seekp(static_cast<std::streamoff>(pos), std::ios::beg); return !f.fail();
}
bool SeekAbsolute(std::istream & f, IO::Offset pos)
{
	if(!OffsetFits<std::streamoff>(pos)) { return false; }
	#if MPT_COMPILER_MSVC
		if(StreamIsStringStreamAndValidAndEmpty(f))
		{
			if(pos == 0)
			{
				f.seekg(static_cast<std::streamoff>(pos), std::ios::beg); f.clear(f.rdstate() & ~std::ios::failbit); return true;
			}
		}
	#endif
	f.seekg(static_cast<std::streamoff>(pos), std::ios::beg); return !f.fail();
}
bool SeekAbsolute(std::iostream & f, IO::Offset pos)
{
	if(!OffsetFits<std::streamoff>(pos)) { return false; }
	#if MPT_COMPILER_MSVC
		if(StreamIsStringStreamAndValidAndEmpty(f))
		{
			if(pos == 0)
			{
				f.seekg(static_cast<std::streamoff>(pos), std::ios::beg); f.clear(f.rdstate() & ~std::ios::failbit); f.seekp(static_cast<std::streamoff>(pos), std::ios::beg); f.clear(f.rdstate() & ~std::ios::failbit); return true;
			}
		}
	#endif
	f.seekg(static_cast<std::streamoff>(pos), std::ios::beg); f.seekp(static_cast<std::streamoff>(pos), std::ios::beg); return !f.fail();
}
bool SeekRelative(std::ostream & f, IO::Offset off)
{
	if(!OffsetFits<std::streamoff>(off)) { return false; }
	#if MPT_COMPILER_MSVC
		if(StreamIsStringStreamAndValidAndEmpty(f))
		{
			if(off == 0)
			{
				f.seekp(static_cast<std::streamoff>(off), std::ios::cur); f.clear(f.rdstate() & ~std::ios::failbit); return true;
			}
		}
	#endif
	f.seekp(static_cast<std::streamoff>(off), std::ios::cur); return !f.fail();
}
bool SeekRelative(std::istream & f, IO::Offset off)
{
	if(!OffsetFits<std::streamoff>(off)) { return false; }
	#if MPT_COMPILER_MSVC
		if(StreamIsStringStreamAndValidAndEmpty(f))
		{
			if(off == 0)
			{
				f.seekg(static_cast<std::streamoff>(off), std::ios::cur); f.clear(f.rdstate() & ~std::ios::failbit); return true;
			}
		}
	#endif
	f.seekg(static_cast<std::streamoff>(off), std::ios::cur); return !f.fail();
}
bool SeekRelative(std::iostream & f, IO::Offset off)
{
	if(!OffsetFits<std::streamoff>(off)) { return false; }
	#if MPT_COMPILER_MSVC
		if(StreamIsStringStreamAndValidAndEmpty(f))
		{
			if(off == 0)
			{
				f.seekg(static_cast<std::streamoff>(off), std::ios::cur); f.clear(f.rdstate() & ~std::ios::failbit); f.seekp(static_cast<std::streamoff>(off), std::ios::cur); f.clear(f.rdstate() & ~std::ios::failbit); return true;
			}
		}
	#endif
	f.seekg(static_cast<std::streamoff>(off), std::ios::cur); f.seekp(static_cast<std::streamoff>(off), std::ios::cur); return !f.fail();
}
IO::Offset ReadRawImpl(std::istream & f, mpt::byte * data, std::size_t size) { return f.read(mpt::byte_cast<char *>(data), size) ? f.gcount() : std::streamsize(0); }
bool WriteRawImpl(std::ostream & f, const mpt::byte * data, std::size_t size) { f.write(mpt::byte_cast<const char *>(data), size); return !f.fail(); }
bool IsEof(std::istream & f) { return f.eof(); }
bool Flush(std::ostream & f) { f.flush(); return !f.fail(); }



#if defined(MPT_ENABLE_FILEIO_STDIO)

bool IsValid(FILE* & f) { return f != NULL; }

#if MPT_COMPILER_MSVC

IO::Offset TellRead(FILE* & f) { return _ftelli64(f); }
IO::Offset TellWrite(FILE* & f) { return _ftelli64(f); }
bool SeekBegin(FILE* & f) { return _fseeki64(f, 0, SEEK_SET) == 0; }
bool SeekEnd(FILE* & f) { return _fseeki64(f, 0, SEEK_END) == 0; }
bool SeekAbsolute(FILE* & f, IO::Offset pos) { return _fseeki64(f, pos, SEEK_SET) == 0; }
bool SeekRelative(FILE* & f, IO::Offset off) { return _fseeki64(f, off, SEEK_CUR) == 0; }

#elif defined(_POSIX_SOURCE) && (_POSIX_SOURCE > 0) 

//STATIC_ASSERT(sizeof(off_t) == 8);
IO::Offset TellRead(FILE* & f) { return ftello(f); }
IO::Offset TellWrite(FILE* & f) { return ftello(f); }
bool SeekBegin(FILE* & f) { return fseeko(f, 0, SEEK_SET) == 0; }
bool SeekEnd(FILE* & f) { return fseeko(f, 0, SEEK_END) == 0; }
bool SeekAbsolute(FILE* & f, IO::Offset pos) { return OffsetFits<off_t>(pos) && (fseek(f, mpt::saturate_cast<off_t>(pos), SEEK_SET) == 0); }
bool SeekRelative(FILE* & f, IO::Offset off) { return OffsetFits<off_t>(off) && (fseek(f, mpt::saturate_cast<off_t>(off), SEEK_CUR) == 0); }

#else

//STATIC_ASSERT(sizeof(long) == 8); // Fails on 32bit non-POSIX systems for now.
IO::Offset TellRead(FILE* & f) { return ftell(f); }
IO::Offset TellWrite(FILE* & f) { return ftell(f); }
bool SeekBegin(FILE* & f) { return fseek(f, 0, SEEK_SET) == 0; }
bool SeekEnd(FILE* & f) { return fseek(f, 0, SEEK_END) == 0; }
bool SeekAbsolute(FILE* & f, IO::Offset pos) { return OffsetFits<long>(pos) && (fseek(f, mpt::saturate_cast<long>(pos), SEEK_SET) == 0); }
bool SeekRelative(FILE* & f, IO::Offset off) { return OffsetFits<long>(off) && (fseek(f, mpt::saturate_cast<long>(off), SEEK_CUR) == 0); }

#endif

IO::Offset ReadRawImpl(FILE * & f, mpt::byte * data, std::size_t size) { return fread(mpt::void_cast<void*>(data), 1, size, f); }
bool WriteRawImpl(FILE* & f, const mpt::byte * data, std::size_t size) { return fwrite(mpt::void_cast<const void*>(data), 1, size, f) == size; }
bool IsEof(FILE * & f) { return feof(f) != 0; }
bool Flush(FILE* & f) { return fflush(f) == 0; }

#endif // MPT_ENABLE_FILEIO_STDIO



} // namespace IO

} // namespace mpt



#if defined(MPT_FILEREADER_STD_ISTREAM)



FileDataContainerSeekable::FileDataContainerSeekable(off_t streamLength)
	: streamLength(streamLength)
	, cached(false)
{
	return;
}

FileDataContainerSeekable::~FileDataContainerSeekable()
{
	return;
}
void FileDataContainerSeekable::CacheStream() const
{
	if(cached)
	{
		return;
	}
	cache.resize(streamLength);
	InternalRead(cache.data(), 0, streamLength);
	cached = true;
}

bool FileDataContainerSeekable::IsValid() const
{
	return true;
}

bool FileDataContainerSeekable::HasFastGetLength() const
{
	return true;
}

bool FileDataContainerSeekable::HasPinnedView() const
{
	return cached;
}

const mpt::byte *FileDataContainerSeekable::GetRawData() const
{
	CacheStream();
	return cache.data();
}

IFileDataContainer::off_t FileDataContainerSeekable::GetLength() const
{
	return streamLength;
}

IFileDataContainer::off_t FileDataContainerSeekable::Read(mpt::byte *dst, IFileDataContainer::off_t pos, IFileDataContainer::off_t count) const
{
	if(cached)
	{
		IFileDataContainer::off_t cache_avail = std::min<IFileDataContainer::off_t>(IFileDataContainer::off_t(cache.size()) - pos, count);
		std::copy(cache.begin() + pos, cache.begin() + pos + cache_avail, dst);
		return cache_avail;
	} else
	{
		return InternalRead(dst, pos, count);
	}
}



bool FileDataContainerStdStreamSeekable::IsSeekable(std::istream *stream)
{
	stream->clear();
	std::streampos oldpos = stream->tellg();
	if(stream->fail() || oldpos == std::streampos(-1))
	{
		stream->clear();
		return false;
	}
	stream->seekg(0, std::ios::beg);
	if(stream->fail())
	{
		stream->clear();
		stream->seekg(oldpos);
		stream->clear();
		return false;
	}
	stream->seekg(0, std::ios::end);
	if(stream->fail())
	{
		stream->clear();
		stream->seekg(oldpos);
		stream->clear();
		return false;
	}
	std::streampos length = stream->tellg();
	if(stream->fail() || length == std::streampos(-1))
	{
		stream->clear();
		stream->seekg(oldpos);
		stream->clear();
		return false;
	}
	stream->seekg(oldpos);
	stream->clear();
	return true;
}

IFileDataContainer::off_t FileDataContainerStdStreamSeekable::GetLength(std::istream *stream)
{
	stream->clear();
	std::streampos oldpos = stream->tellg();
	stream->seekg(0, std::ios::end);
	std::streampos length = stream->tellg();
	stream->seekg(oldpos);
	return mpt::saturate_cast<IFileDataContainer::off_t>(static_cast<int64>(length));
}

FileDataContainerStdStreamSeekable::FileDataContainerStdStreamSeekable(std::istream *s)
	: FileDataContainerSeekable(GetLength(s))
	, stream(s)
{
	return;
}

FileDataContainerStdStreamSeekable::~FileDataContainerStdStreamSeekable()
{
	return;
}

IFileDataContainer::off_t FileDataContainerStdStreamSeekable::InternalRead(mpt::byte *dst, off_t pos, off_t count) const
{
	stream->clear(); // tellg needs eof and fail bits unset
	std::streampos currentpos = stream->tellg();
	if(currentpos == std::streampos(-1) || static_cast<int64>(pos) != currentpos)
	{ // inefficient istream implementations might invalidate their buffer when seeking, even when seeking to the current position
		stream->seekg(pos);
	}
	stream->read(mpt::byte_cast<char*>(dst), count);
	return static_cast<IFileDataContainer::off_t>(stream->gcount());
}


FileDataContainerUnseekable::FileDataContainerUnseekable()
	: cachesize(0), streamFullyCached(false)
{
	return;
}

FileDataContainerUnseekable::~FileDataContainerUnseekable()
{
	return;
}

void FileDataContainerUnseekable::EnsureCacheBuffer(std::size_t requiredbuffersize) const
{
	if(cache.size() >= cachesize + requiredbuffersize)
	{
		return;
	}
	if(cache.size() == 0)
	{
		cache.resize(Util::AlignUp<std::size_t>(cachesize + requiredbuffersize, BUFFER_SIZE));
	} else if(Util::ExponentialGrow(cache.size()) < cachesize + requiredbuffersize)
	{
		cache.resize(Util::AlignUp<std::size_t>(cachesize + requiredbuffersize, BUFFER_SIZE));
	} else
	{
		cache.resize(Util::ExponentialGrow(cache.size()));
	}
}

void FileDataContainerUnseekable::CacheStream() const
{
	if(streamFullyCached)
	{
		return;
	}
	while(!InternalEof())
	{
		EnsureCacheBuffer(BUFFER_SIZE);
		std::size_t readcount = InternalRead(&cache[cachesize], BUFFER_SIZE);
		cachesize += readcount;
	}
	streamFullyCached = true;
}

void FileDataContainerUnseekable::CacheStreamUpTo(off_t pos, off_t length) const
{
	if(streamFullyCached)
	{
		return;
	}
	if(length > std::numeric_limits<off_t>::max() - pos)
	{
		length = std::numeric_limits<off_t>::max() - pos;
	}
	std::size_t target = mpt::saturate_cast<std::size_t>(pos + length);
	if(target <= cachesize)
	{
		return;
	}
	std::size_t alignedpos = Util::AlignUp<std::size_t>(target, QUANTUM_SIZE);
	std::size_t needcount = alignedpos - cachesize;
	EnsureCacheBuffer(needcount);
	std::size_t readcount = InternalRead(&cache[cachesize], alignedpos - cachesize);
	cachesize += readcount;
	if(!InternalEof())
	{
		// can read further
		return;
	}
	streamFullyCached = true;
}

void FileDataContainerUnseekable::ReadCached(mpt::byte *dst, IFileDataContainer::off_t pos, IFileDataContainer::off_t count) const
{
	std::copy(cache.begin() + pos, cache.begin() + pos + count, dst);
}

bool FileDataContainerUnseekable::IsValid() const
{
	return true;
}

bool FileDataContainerUnseekable::HasFastGetLength() const
{
	return false;
}

bool FileDataContainerUnseekable::HasPinnedView() const
{
	return true; // we have the cache which is required for seeking anyway
}

const mpt::byte *FileDataContainerUnseekable::GetRawData() const
{
	CacheStream();
	return cache.data();
}

IFileDataContainer::off_t FileDataContainerUnseekable::GetLength() const
{
	CacheStream();
	return cachesize;
}

IFileDataContainer::off_t FileDataContainerUnseekable::Read(mpt::byte *dst, IFileDataContainer::off_t pos, IFileDataContainer::off_t count) const
{
	CacheStreamUpTo(pos, count);
	if(pos >= IFileDataContainer::off_t(cachesize))
	{
		return 0;
	}
	IFileDataContainer::off_t cache_avail = std::min<IFileDataContainer::off_t>(IFileDataContainer::off_t(cachesize) - pos, count);
	ReadCached(dst, pos, cache_avail);
	return cache_avail;
}

bool FileDataContainerUnseekable::CanRead(IFileDataContainer::off_t pos, IFileDataContainer::off_t length) const
{
	CacheStreamUpTo(pos, length);
	if((pos == IFileDataContainer::off_t(cachesize)) && (length == 0))
	{
		return true;
	}
	if(pos >= IFileDataContainer::off_t(cachesize))
	{
		return false;
	}
	return length <= IFileDataContainer::off_t(cachesize) - pos;
}

IFileDataContainer::off_t FileDataContainerUnseekable::GetReadableLength(IFileDataContainer::off_t pos, IFileDataContainer::off_t length) const
{
	CacheStreamUpTo(pos, length);
	if(pos >= cachesize)
	{
		return 0;
	}
	return std::min<IFileDataContainer::off_t>(cachesize - pos, length);
}



FileDataContainerStdStream::FileDataContainerStdStream(std::istream *s)
	: stream(s)
{
	return;
}

FileDataContainerStdStream::~FileDataContainerStdStream()
{
	return;
}

bool FileDataContainerStdStream::InternalEof() const
{
	if(*stream)
	{
		return false;
	} else
	{
		return true;
	}
}

IFileDataContainer::off_t FileDataContainerStdStream::InternalRead(mpt::byte *dst, off_t count) const
{
	stream->read(mpt::byte_cast<char*>(dst), count);
	return static_cast<std::size_t>(stream->gcount());
}



#if defined(MPT_FILEREADER_CALLBACK_STREAM)


bool FileDataContainerCallbackStreamSeekable::IsSeekable(CallbackStream stream)
{
	if(!stream.stream)
	{
		return false;
	}
	if(!stream.seek)
	{
		return false;
	}
	if(!stream.tell)
	{
		return false;
	}
	int64 oldpos = stream.tell(stream.stream);
	if(oldpos < 0)
	{
		return false;
	}
	if(stream.seek(stream.stream, 0, CallbackStream::SeekSet) < 0)
	{
		stream.seek(stream.stream, oldpos, CallbackStream::SeekSet);
		return false;
	}
	if(stream.seek(stream.stream, 0, CallbackStream::SeekEnd) < 0)
	{
		stream.seek(stream.stream, oldpos, CallbackStream::SeekSet);
		return false;
	}
	int64 length = stream.tell(stream.stream);
	if(length < 0)
	{
		stream.seek(stream.stream, oldpos, CallbackStream::SeekSet);
		return false;
	}
	stream.seek(stream.stream, oldpos, CallbackStream::SeekSet);
	return true;
}

IFileDataContainer::off_t FileDataContainerCallbackStreamSeekable::GetLength(CallbackStream stream)
{
	if(!stream.stream)
	{
		return 0;
	}
	if(!stream.seek)
	{
		return false;
	}
	if(!stream.tell)
	{
		return false;
	}
	int64 oldpos = stream.tell(stream.stream);
	if(oldpos < 0)
	{
		return 0;
	}
	if(stream.seek(stream.stream, 0, CallbackStream::SeekSet) < 0)
	{
		stream.seek(stream.stream, oldpos, CallbackStream::SeekSet);
		return 0;
	}
	if(stream.seek(stream.stream, 0, CallbackStream::SeekEnd) < 0)
	{
		stream.seek(stream.stream, oldpos, CallbackStream::SeekSet);
		return 0;
	}
	int64 length = stream.tell(stream.stream);
	if(length < 0)
	{
		stream.seek(stream.stream, oldpos, CallbackStream::SeekSet);
		return 0;
	}
	stream.seek(stream.stream, oldpos, CallbackStream::SeekSet);
	return mpt::saturate_cast<IFileDataContainer::off_t>(length);
}

FileDataContainerCallbackStreamSeekable::FileDataContainerCallbackStreamSeekable(CallbackStream s)
	: FileDataContainerSeekable(GetLength(s))
	, stream(s)
{
	return;
}

FileDataContainerCallbackStreamSeekable::~FileDataContainerCallbackStreamSeekable()
{
	return;
}

IFileDataContainer::off_t FileDataContainerCallbackStreamSeekable::InternalRead(mpt::byte *dst, off_t pos, off_t count) const
{
	if(!stream.read)
	{
		return 0;
	}
	if(stream.seek(stream.stream, pos, CallbackStream::SeekSet) < 0)
	{
		return 0;
	}
	int64 totalread = 0;
	while(count > 0)
	{
		int64 readcount = stream.read(stream.stream, dst, count);
		if(readcount <= 0)
		{
			break;
		}
		dst += static_cast<std::size_t>(readcount);
		count -= static_cast<IFileDataContainer::off_t>(readcount);
		totalread += readcount;
	}
	return static_cast<IFileDataContainer::off_t>(totalread);
}



FileDataContainerCallbackStream::FileDataContainerCallbackStream(CallbackStream s)
	: FileDataContainerUnseekable()
	, stream(s)
	, eof_reached(false)
{
	return;
}

FileDataContainerCallbackStream::~FileDataContainerCallbackStream()
{
	return;
}

bool FileDataContainerCallbackStream::InternalEof() const
{
	return eof_reached;
}

IFileDataContainer::off_t FileDataContainerCallbackStream::InternalRead(mpt::byte *dst, off_t count) const
{
	if(eof_reached)
	{
		return 0;
	}
	if(!stream.read)
	{
		eof_reached = true;
		return 0;
	}
	int64 totalread = 0;
	while(count > 0)
	{
		int64 readcount = stream.read(stream.stream, dst, count);
		if(readcount <= 0)
		{
			eof_reached = true;
			break;
		}
		dst += static_cast<std::size_t>(readcount);
		count -= static_cast<IFileDataContainer::off_t>(readcount);
		totalread += readcount;
	}
	return static_cast<IFileDataContainer::off_t>(totalread);
}


#endif // MPT_FILEREADER_CALLBACK_STREAM


#endif



OPENMPT_NAMESPACE_END
