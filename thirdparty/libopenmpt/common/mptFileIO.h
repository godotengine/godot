/*
 * mptFileIO.h
 * -----------
 * Purpose: A wrapper around std::fstream, fixing VS2008 charset conversion braindamage, and enforcing usage of mpt::PathString.
 * Notes  : You should only ever use these wrappers instead of plain std::fstream classes.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once

#if defined(MPT_ENABLE_FILEIO)

#include "../common/mptString.h"
#include "../common/mptPathString.h"
#include "../common/mptIO.h"

#include <fstream>
#include <ios>
#include <ostream>
#include <streambuf>
#include <utility>

#endif // MPT_ENABLE_FILEIO


OPENMPT_NAMESPACE_BEGIN


#if defined(MPT_ENABLE_FILEIO)


#if defined(MPT_ENABLE_FILEIO_STDIO)

static inline FILE * mpt_fopen(const mpt::PathString &filename, const char *mode)
{
	#if MPT_OS_WINDOWS
		return _wfopen(filename.AsNativePrefixed().c_str(), mode ? mpt::ToWide(mpt::CharsetASCII, mode).c_str() : L"");
	#else // !MPT_OS_WINDOWS
		return fopen(filename.AsNative().c_str(), mode);
	#endif // MPT_OS_WINDOWS
}

#endif // MPT_ENABLE_FILEIO_STDIO


// Sets the NTFS compression attribute on the file or directory.
// Requires read and write permissions for already opened files.
// Returns true if the attribute has been set.
// In almost all cases, the return value should be ignored because most filesystems other than NTFS do not support compression.
#ifdef MODPLUG_TRACKER
#if MPT_OS_WINDOWS
bool SetFilesystemCompression(HANDLE hFile);
bool SetFilesystemCompression(int fd);
#if defined(MPT_ENABLE_FILEIO_STDIO)
bool SetFilesystemCompression(FILE *file);
#endif // MPT_ENABLE_FILEIO_STDIO
bool SetFilesystemCompression(const mpt::PathString &filename);
#endif // MPT_OS_WINDOWS
#endif // MODPLUG_TRACKER


namespace mpt
{

#if MPT_COMPILER_GCC

#if MPT_OS_WINDOWS
// GCC C++ library has no wchar_t overloads
#define MPT_FSTREAM_DO_CONVERSIONS
#define MPT_FSTREAM_DO_CONVERSIONS_ANSI
#endif

#endif

#ifdef MPT_FSTREAM_DO_CONVERSIONS
#define MPT_FSTREAM_OPEN(filename, mode) detail::fstream_open<Tbase>(*this, (filename), (mode))
#else
#define MPT_FSTREAM_OPEN(filename, mode) Tbase::open((filename), (mode))
#endif

namespace detail
{

template<typename Tbase>
inline void fstream_open(Tbase & base, const mpt::PathString & filename, std::ios_base::openmode mode)
{
#if defined(MPT_FSTREAM_DO_CONVERSIONS_ANSI)
	base.open(mpt::ToCharset(mpt::CharsetLocale, filename.AsNative()).c_str(), mode);
#else
	base.open(filename.AsNativePrefixed().c_str(), mode);
#endif
}

#ifdef MPT_FSTREAM_DO_CONVERSIONS

template<typename Tbase>
inline void fstream_open(Tbase & base, const std::wstring & filename, std::ios_base::openmode mode)
{
	detail::fstream_open<Tbase>(base, mpt::PathString::FromWide(filename), mode);
}

template<typename Tbase>
inline void fstream_open(Tbase & base, const wchar_t * filename, std::ios_base::openmode mode)
{
	detail::fstream_open<Tbase>(base, mpt::PathString::FromWide(filename ? std::wstring(filename) : std::wstring()), mode);
}

template<typename Tbase>
inline void fstream_open(Tbase & base, const std::string & filename, std::ios_base::openmode mode)
{
	detail::fstream_open<Tbase>(base, mpt::PathString::FromWide(mpt::ToWide(mpt::CharsetLocale, filename)), mode);
}

template<typename Tbase>
inline void fstream_open(Tbase & base, const char * filename, std::ios_base::openmode mode)
{
	detail::fstream_open<Tbase>(base, mpt::PathString::FromWide(mpt::ToWide(mpt::CharsetLocale, filename ? std::string(filename) : std::string())), mode);
}

#endif

} // namespace detail

class fstream
	: public std::fstream
{
private:
	typedef std::fstream Tbase;
public:
	fstream() {}
	fstream(const mpt::PathString & filename, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
	{
		detail::fstream_open<Tbase>(*this, filename, mode);
	}
	void open(const mpt::PathString & filename, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
	{
		detail::fstream_open<Tbase>(*this, filename, mode);
	}
	MPT_DEPRECATED_PATH void open(const char * filename, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
	{
		MPT_FSTREAM_OPEN(filename, mode);
	}
	MPT_DEPRECATED_PATH void open(const std::string & filename, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
	{
		MPT_FSTREAM_OPEN(filename.c_str(), mode);
	}
#if MPT_OS_WINDOWS
	MPT_DEPRECATED_PATH void open(const wchar_t * filename, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
	{
		MPT_FSTREAM_OPEN(filename, mode);
	}
	MPT_DEPRECATED_PATH void open(const std::wstring & filename, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
	{
		MPT_FSTREAM_OPEN(filename.c_str(), mode);
	}
#endif
};

class ifstream
	: public std::ifstream
{
private:
	typedef std::ifstream Tbase;
public:
	ifstream() {}
	ifstream(const mpt::PathString & filename, std::ios_base::openmode mode = std::ios_base::in)
	{
		detail::fstream_open<Tbase>(*this, filename, mode);
	}
	void open(const mpt::PathString & filename, std::ios_base::openmode mode = std::ios_base::in)
	{
		detail::fstream_open<Tbase>(*this, filename, mode);
	}
	MPT_DEPRECATED_PATH void open(const char * filename, std::ios_base::openmode mode = std::ios_base::in)
	{
		MPT_FSTREAM_OPEN(filename, mode);
	}
	MPT_DEPRECATED_PATH void open(const std::string & filename, std::ios_base::openmode mode = std::ios_base::in)
	{
		MPT_FSTREAM_OPEN(filename.c_str(), mode);
	}
#if MPT_OS_WINDOWS
	MPT_DEPRECATED_PATH void open(const wchar_t * filename, std::ios_base::openmode mode = std::ios_base::in)
	{
		MPT_FSTREAM_OPEN(filename, mode);
	}
	MPT_DEPRECATED_PATH void open(const std::wstring & filename, std::ios_base::openmode mode = std::ios_base::in)
	{
		MPT_FSTREAM_OPEN(filename.c_str(), mode);
	}
#endif
};

class ofstream
	: public std::ofstream
{
private:
	typedef std::ofstream Tbase;
public:
	ofstream() {}
	ofstream(const mpt::PathString & filename, std::ios_base::openmode mode = std::ios_base::out)
	{
		detail::fstream_open<Tbase>(*this, filename, mode);
	}
	void open(const mpt::PathString & filename, std::ios_base::openmode mode = std::ios_base::out)
	{
		detail::fstream_open<Tbase>(*this, filename, mode);
	}
	MPT_DEPRECATED_PATH void open(const char * filename, std::ios_base::openmode mode = std::ios_base::out)
	{
		MPT_FSTREAM_OPEN(filename, mode);
	}
	MPT_DEPRECATED_PATH void open(const std::string & filename, std::ios_base::openmode mode = std::ios_base::out)
	{
		MPT_FSTREAM_OPEN(filename.c_str(), mode);
	}
#if MPT_OS_WINDOWS
	MPT_DEPRECATED_PATH void open(const wchar_t * filename, std::ios_base::openmode mode = std::ios_base::out)
	{
		MPT_FSTREAM_OPEN(filename, mode);
	}
	MPT_DEPRECATED_PATH void open(const std::wstring & filename, std::ios_base::openmode mode = std::ios_base::out)
	{
		MPT_FSTREAM_OPEN(filename.c_str(), mode);
	}
#endif
};

#undef MPT_FSTREAM_OPEN



// LazyFileRef is a simple reference to an on-disk file by the means of a
// filename which allows easy assignment of the whole file contents to and from
// byte buffers.
class LazyFileRef {
private:
	const mpt::PathString m_Filename;
public:
	LazyFileRef(const mpt::PathString &filename)
		: m_Filename(filename)
	{
		return;
	}
public:
	LazyFileRef & operator = (const std::vector<mpt::byte> &data);
	LazyFileRef & operator = (const std::vector<char> &data);
	LazyFileRef & operator = (const std::string &data);
	operator std::vector<mpt::byte> () const;
	operator std::vector<char> () const;
	operator std::string () const;
};



#if defined(MPT_ENABLE_FILEIO_STDIO)

// class FILE_ostream, FILE_output_streambuf and FILE_output_buffered_streambuf
//  provide a portable way of wrapping a std::ostream around an FILE* opened for output.
// They offer similar functionality to the badly documented
//  MSVC std::fstream(FILE*) constructor or GCC libstdc++  __gnu_cxx::stdio_sync_filebuf class,
//  and, for other compilers, provide a race-free alternative to
//  closing the FILE* and opening it again as a std::ofstream.
//
// Only output functionality is implemented because we have no need for an input wrapper.
//
// During the whole lifetime of the iostream wrappers, the FILE* object is assumend to be
//  either
//   - NULL
//  or
//   - valid
//   - opened for writing in non-append mode
//   - opened in binary mode
//   - seekable
// Some of these preconditions cannot be verified,
//  and even the others do not get verified.
// Behaviour in case of any unmet preconditions is undefined.
//
// The buffered streambuf and the ostream use a buffer of 64KiB by default.
//
// For FILE_output_streambuf, coherency with the underlying FILE* is always guaranteed.
// For FILE_ostream and FILE_output_buffered_streambuf, coherence is only
//  guaranteed when flush() or pubsync() get called.
// The constructors and destructors take care to not violate coherency.
// When mixing FILE* and iostream I/O during the lifetime of the iostream objects,
//  the user is responsible for providing coherency via the appropriate
//  flush and sync functions.
// Behaviour in case of incoherent access is undefined.


class FILE_output_streambuf : public std::streambuf
{
public:
	typedef std::streambuf::char_type char_type;
	typedef std::streambuf::traits_type traits_type;
	typedef traits_type::int_type int_type;
	typedef traits_type::pos_type pos_type;
	typedef traits_type::off_type off_type;
protected:
	FILE *f;
public:
	FILE_output_streambuf(FILE *f)
		: f(f)
	{
		return;
	}
	~FILE_output_streambuf()
	{
		return;
	}
protected:
	virtual int_type overflow(int_type ch)
	{
		if(!mpt::IO::IsValid(f))
		{
			return traits_type::eof();
		}
		if(traits_type::eq_int_type(ch, traits_type::eof()))
		{
			return traits_type::eof();
		}
		char_type c = traits_type::to_char_type(ch);
		if(!mpt::IO::WriteRaw(f, &c, 1))
		{
			return traits_type::eof();
		}
		return ch;
	}
	virtual int sync()
	{
		if(!mpt::IO::IsValid(f))
		{
			return -1;
		}
		if(!mpt::IO::Flush(f))
		{
			return -1;
		}
		return 0;
	}
	virtual pos_type seekpos(pos_type pos, std::ios_base::openmode which)
	{
		if(!mpt::IO::IsValid(f))
		{
			return pos_type(off_type(-1));
		}
		return seekoff(pos, std::ios_base::beg, which);
	}
	virtual pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which)
	{
		if(!mpt::IO::IsValid(f))
		{
			return pos_type(off_type(-1));
		}
		if(which & std::ios_base::in)
		{
			return pos_type(off_type(-1));
		}
		if(!(which & std::ios_base::out))
		{
			return pos_type(off_type(-1));
		}
		mpt::IO::Offset oldpos = mpt::IO::TellWrite(f);
		if(dir == std::ios_base::beg)
		{
			if(!mpt::IO::SeekAbsolute(f, off))
			{
				mpt::IO::SeekAbsolute(f, oldpos);
				return pos_type(off_type(-1));
			}
		} else if(dir == std::ios_base::cur)
		{
			if(!mpt::IO::SeekRelative(f, off))
			{
				mpt::IO::SeekAbsolute(f, oldpos);
				return pos_type(off_type(-1));
			}
		} else if(dir == std::ios_base::end)
		{
			if(!(mpt::IO::SeekEnd(f) && mpt::IO::SeekRelative(f, off)))
			{
				mpt::IO::SeekAbsolute(f, oldpos);
				return pos_type(off_type(-1));
			}
		} else
		{
			return pos_type(off_type(-1));
		}
		mpt::IO::Offset newpos = mpt::IO::TellWrite(f);
		if(!mpt::IO::OffsetFits<off_type>(newpos))
		{
			mpt::IO::SeekAbsolute(f, oldpos);
			return pos_type(off_type(-1));
		}
		return pos_type(static_cast<off_type>(newpos));
	}
}; // class FILE_output_streambuf


class FILE_output_buffered_streambuf : public FILE_output_streambuf
{
public:
	typedef std::streambuf::char_type char_type;
	typedef std::streambuf::traits_type traits_type;
	typedef traits_type::int_type int_type;
	typedef traits_type::pos_type pos_type;
	typedef traits_type::off_type off_type;
private:
	typedef FILE_output_streambuf Tparent;
	std::vector<char_type> buf;
public:
	FILE_output_buffered_streambuf(FILE *f, std::size_t bufSize = 64*1024)
		: FILE_output_streambuf(f)
		, buf((bufSize > 0) ? bufSize : 1)
	{
		setp(buf.data(), buf.data() + buf.size());
	}
	~FILE_output_buffered_streambuf()
	{
		if(!mpt::IO::IsValid(f))
		{
			return;
		}
		WriteOut();
	}
private:
	bool IsDirty() const
	{
		return ((pptr() - pbase()) > 0);
	}
	bool WriteOut()
	{
		std::ptrdiff_t n = pptr() - pbase();
		std::ptrdiff_t left = n;
		while(left > 0)
		{
			int backchunk = mpt::saturate_cast<int>(-left);
			pbump(backchunk);
			left += backchunk;
		}
		return mpt::IO::WriteRaw(f, pbase(), n);
	}
protected:
	virtual int_type overflow(int_type ch)
	{
		if(!mpt::IO::IsValid(f))
		{
			return traits_type::eof();
		}
		if(traits_type::eq_int_type(ch, traits_type::eof()))
		{
			return traits_type::eof();
		}
		if(!WriteOut())
		{
			return traits_type::eof();
		}
		char_type c = traits_type::to_char_type(ch);
		*pptr() = c;
		pbump(1);
		return ch;
	}
	virtual int sync()
	{
		if(!mpt::IO::IsValid(f))
		{
			return -1;
		}
		if(!WriteOut())
		{
			return -1;
		}
		return Tparent::sync();
	}
	virtual pos_type seekpos(pos_type pos, std::ios_base::openmode which)
	{
		if(!mpt::IO::IsValid(f))
		{
			return pos_type(off_type(-1));
		}
		if(!WriteOut())
		{
			return pos_type(off_type(-1));
		}
		return Tparent::seekpos(pos, which);
	}
	virtual pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which)
	{
		if(!mpt::IO::IsValid(f))
		{
			return pos_type(off_type(-1));
		}
		if(!WriteOut())
		{
			return pos_type(off_type(-1));
		}
		return Tparent::seekoff(off, dir, which);
	}
}; // class FILE_output_buffered_streambuf


class FILE_ostream : public std::ostream {
private:
	FILE *f;
	FILE_output_buffered_streambuf buf;
public:
	FILE_ostream(FILE *f, std::size_t bufSize = 64*1024)
		: std::ostream(&buf)
		, f(f)
		, buf(f, bufSize)
	{
		if(mpt::IO::IsValid(f)) mpt::IO::Flush(f);
	}
	~FILE_ostream()
	{
		flush();
		buf.pubsync();
		if(mpt::IO::IsValid(f)) mpt::IO::Flush(f);
	}
}; // class FILE_ostream                                                                                        

#endif // MPT_ENABLE_FILEIO_STDIO


} // namespace mpt



#ifdef MODPLUG_TRACKER
#if MPT_OS_WINDOWS
class CMappedFile
{
protected:
	HANDLE m_hFile;
	HANDLE m_hFMap;
	void *m_pData;
	mpt::PathString m_FileName;

public:
	CMappedFile() : m_hFile(nullptr), m_hFMap(nullptr), m_pData(nullptr) { }
	~CMappedFile();

public:
	bool Open(const mpt::PathString &filename);
	bool IsOpen() const { return m_hFile != NULL && m_hFile != INVALID_HANDLE_VALUE; }
	const mpt::PathString * GetpFilename() const { return &m_FileName; }
	void Close();
	size_t GetLength();
	const mpt::byte *Lock();
};
#endif // MPT_OS_WINDOWS
#endif // MODPLUG_TRACKER


class InputFile
{
private:
	mpt::PathString m_Filename;
	#ifdef MPT_FILEREADER_STD_ISTREAM
		mpt::ifstream m_File;
	#else
		CMappedFile m_File;
	#endif
public:
	InputFile();
	InputFile(const mpt::PathString &filename);
	~InputFile();
	bool Open(const mpt::PathString &filename);
	bool IsValid() const;
#if defined(MPT_FILEREADER_STD_ISTREAM)
	typedef std::pair<std::istream*, const mpt::PathString*> ContentsRef;
#else
	struct Data
	{
		const mpt::byte *data;
		std::size_t size;
	};
	typedef std::pair<InputFile::Data, const mpt::PathString*> ContentsRef;
#endif
	InputFile::ContentsRef Get();
};


#endif // MPT_ENABLE_FILEIO


OPENMPT_NAMESPACE_END

