/*
 * mptBufferIO.h
 * -------------
 * Purpose: A wrapper around std::stringstream, fixing MSVC tell/seek problems with empty streams.
 * Notes  : You should only ever use these wrappers instead of plain std::stringstream classes.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once

#include <ios>
#include <istream>
#include <ostream>
#include <sstream>
#include <streambuf>

OPENMPT_NAMESPACE_BEGIN


// MSVC std::stringbuf (and thereby std::ostringstream, std::istringstream and
// std::stringstream) fail seekpos() and seekoff() when the stringbuf is
// currently empty.
// seekpos() and seekoff() can get called via tell*() or seek*() iostream
// members. seekoff() (and thereby tell*()), but not seekpos(), has been fixed
// from VS2010 onwards to handle this specific case and changed to not fail
// when the stringbuf is empty.
// Work-around strategy:
// As re-implementing or duplicating the whole std::stringbuf semantics would be
// rather convoluted, we make use of the knowledge of specific inner workings of
// the MSVC implementation here and just fix-up where it causes problems. This
// keeps the additional code at a minimum size.


namespace mpt
{

#if MPT_COMPILER_MSVC

class stringbuf
	: public std::stringbuf
{
public:
	stringbuf(std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
		: std::stringbuf(mode)
	{
		return;
	}
	stringbuf(const std::string &str, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
		: std::stringbuf(str, mode)
	{
		return;
	}
public:
	virtual pos_type seekoff(off_type off, std::ios_base::seekdir way, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out)
	{
		pos_type result = std::stringbuf::seekoff(off, way, which);
		if(result == pos_type(-1))
		{
			if((which & std::ios_base::in) || (which & std::ios_base::out))
			{
				if(off == 0)
				{
					result = 0;
				}
			}
		}
		return result;
	}
	virtual pos_type seekpos(pos_type ptr, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
	{
		pos_type result = std::stringbuf::seekpos(ptr, mode);
		if(result == pos_type(-1))
		{
			if((mode & std::ios_base::in) || (mode & std::ios_base::out))
			{
				if(static_cast<std::streamoff>(ptr) == 0)
				{
					result = 0;
				}
			}
		}
		return result;
	}
};

class istringstream
	: public std::basic_istream<char>
{
private:
	mpt::stringbuf buf;
public:
	istringstream(std::ios_base::openmode mode = std::ios_base::in)
		: std::basic_istream<char>(&buf)
		, buf(mode | std::ios_base::in)
	{
	}
	istringstream(const std::string &str, std::ios_base::openmode mode = std::ios_base::in)
		: std::basic_istream<char>(&buf)
		, buf(str, mode | std::ios_base::in)
	{
	}
	~istringstream()
	{
	}
public:
	mpt::stringbuf *rdbuf() const { return const_cast<mpt::stringbuf*>(&buf); }
	std::string str() const { return buf.str(); }
	void str(const std::string &str) { buf.str(str); } 
};

class ostringstream
	: public std::basic_ostream<char>
{
private:
	mpt::stringbuf buf;
public:
	ostringstream(std::ios_base::openmode mode = std::ios_base::out)
		: std::basic_ostream<char>(&buf)
		, buf(mode | std::ios_base::out)
	{
	}
	ostringstream(const std::string &str, std::ios_base::openmode mode = std::ios_base::out)
		: std::basic_ostream<char>(&buf)
		, buf(str, mode | std::ios_base::out)
	{
	}
	~ostringstream()
	{
	}
public:
	mpt::stringbuf *rdbuf() const { return const_cast<mpt::stringbuf*>(&buf); }
	std::string str() const { return buf.str(); }
	void str(const std::string &str) { buf.str(str); } 
};

class stringstream
	: public std::basic_iostream<char>
{
private:
	mpt::stringbuf buf;
public:
	stringstream(std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
		: std::basic_iostream<char>(&buf)
		, buf(mode | std::ios_base::in | std::ios_base::out)
	{
	}
	stringstream(const std::string &str, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
		: std::basic_iostream<char>(&buf)
		, buf(str, mode | std::ios_base::in | std::ios_base::out)
	{
	}
	~stringstream()
	{
	}
public:
	mpt::stringbuf *rdbuf() const { return const_cast<mpt::stringbuf*>(&buf); }
	std::string str() const { return buf.str(); }
	void str(const std::string &str) { buf.str(str); } 
};

#else

typedef std::stringbuf stringbuf;
typedef std::istringstream istringstream;
typedef std::ostringstream ostringstream;
typedef std::stringstream stringstream;

#endif

} // namespace mpt


OPENMPT_NAMESPACE_END

