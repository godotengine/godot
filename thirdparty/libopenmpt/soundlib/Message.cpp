/*
 * Message.cpp
 * -----------
 * Purpose: Various functions for processing song messages (allocating, reading from file...)
 * Notes  : Those functions should offer a rather high level of abstraction compared to
 *          previous ways of reading the song messages. There are still many things to do,
 *          though. Future versions of ReadMessage() could e.g. offer charset conversion
 *          and the code is not yet ready for unicode.
 *          Some functions for preparing the message text to be written to a file would
 *          also be handy.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Message.h"
#include "../common/FileReader.h"

OPENMPT_NAMESPACE_BEGIN

// Read song message from a mapped file.
// [in]  data: pointer to the data in memory that is going to be read
// [in]  length: number of characters that should be read, not including a possible trailing null terminator (it is automatically appended).
// [in]  lineEnding: line ending formatting of the text in memory.
// [out] returns true on success.
bool SongMessage::Read(const mpt::byte *data, size_t length, LineEnding lineEnding)
{
	const char *str = mpt::byte_cast<const char *>(data);
	while(length != 0 && str[length - 1] == '\0')
	{
		// Ignore trailing null character.
		length--;
	}

	// Simple line-ending detection algorithm. VERY simple.
	if(lineEnding == leAutodetect)
	{
		char cprev = 0;
		size_t nCR = 0, nLF = 0, nCRLF = 0;
		// find CRs, LFs and CRLFs
		for(size_t i = 0; i < length; i++)
		{
			char c = str[i];

			if(c == '\r') nCR++;
			else if(c == '\n') nLF++;

			if(i && cprev == '\r' && c == '\n') nCRLF++;
			cprev = c;
		}
		// evaluate findings
		if(nCR == nLF && nCR == nCRLF)
			lineEnding = leCRLF;
		else if(nCR && !nLF)
			lineEnding = leCR;
		else if(!nCR && nLF)
			lineEnding = leLF;
		else
			lineEnding = leMixed;
	}

	size_t finalLength = 0;
	// calculate the final amount of characters to be allocated.
	for(size_t i = 0; i < length; i++)
	{
		char c = str[i];

		if(c != '\n' || lineEnding != leCRLF)
			finalLength++;
	}

	resize(finalLength);

	size_t cpos = 0;
	for(size_t i = 0; i < length; i++, cpos++)
	{
		char c = str[i];

		switch(c)
		{
		case '\r':
			if(lineEnding != leLF)
				at(cpos) = InternalLineEnding;
			else
				at(cpos) = ' ';
			if(lineEnding == leCRLF) i++;	// skip the LF
			break;
		case '\n':
			if(lineEnding != leCR && lineEnding != leCRLF)
				at(cpos) = InternalLineEnding;
			else
				at(cpos) = ' ';
			break;
		case '\0':
			at(cpos) = ' ';
			break;
		default:
			at(cpos) = c;
			break;
		}
	}

	return true;
}


bool SongMessage::Read(FileReader &file, const size_t length, LineEnding lineEnding)
{
	FileReader::off_t readLength = std::min(static_cast<FileReader::off_t>(length), file.BytesLeft());
	FileReader::PinnedRawDataView fileView = file.ReadPinnedRawDataView(readLength);
	bool success = Read(fileView.data(), fileView.size(), lineEnding);
	return success;
}


// Read comments with fixed line length from a mapped file.
// [in]  data: pointer to the data in memory that is going to be read
// [in]  length: number of characters that should be read, not including a possible trailing null terminator (it is automatically appended).
// [in]  lineLength: The fixed length of a line.
// [in]  lineEndingLength: The padding space between two fixed lines. (there could for example be a null char after every line)
// [out] returns true on success.
bool SongMessage::ReadFixedLineLength(const mpt::byte *data, const size_t length, const size_t lineLength, const size_t lineEndingLength)
{
	if(lineLength == 0)
		return false;
	clear();
	reserve(length);

	size_t readPos = 0, writePos = 0;
	while(readPos < length)
	{
		size_t thisLineLength = std::min(lineLength, length - readPos);
		append(mpt::byte_cast<const char *>(data) + readPos, thisLineLength);
		append(1, InternalLineEnding);

		// Fix weird chars
		for(size_t pos = writePos; pos < writePos + thisLineLength; pos++)
		{
			switch(at(pos))
			{
			case '\0':
			case '\n':
			case '\r':
				at(pos) = ' ';
				break;
			}

		}

		readPos += thisLineLength + std::min(lineEndingLength, length - readPos - thisLineLength);
		writePos += thisLineLength + 1;
	}
	return true;
}


bool SongMessage::ReadFixedLineLength(FileReader &file, const size_t length, const size_t lineLength, const size_t lineEndingLength)
{
	FileReader::off_t readLength = std::min(static_cast<FileReader::off_t>(length), file.BytesLeft());
	FileReader::PinnedRawDataView fileView = file.ReadPinnedRawDataView(readLength);
	bool success = ReadFixedLineLength(fileView.data(), fileView.size(), lineLength, lineEndingLength);
	return success;
}


// Retrieve song message.
// [in]  lineEnding: line ending formatting of the text in memory.
// [out] returns formatted song message.
std::string SongMessage::GetFormatted(const LineEnding lineEnding) const
{
	std::string comments;
	comments.reserve(length());
	for(auto c : *this)
	{
		if(c == InternalLineEnding)
		{
			switch(lineEnding)
			{
			case leCR:
				comments.push_back('\r');
				break;
			case leCRLF:
				comments.push_back('\r');
				comments.push_back('\n');
				break;
			case leLF:
				comments.push_back('\n');
				break;
			default:
				comments.push_back('\r');
				break;
			}
		} else
		{
			comments.push_back(c);
		}
	}
	return comments;
}


bool SongMessage::SetFormatted(std::string message, LineEnding lineEnding)
{
	MPT_ASSERT(lineEnding == leLF || lineEnding == leCR || lineEnding == leCRLF);
	switch (lineEnding)
	{
	case leLF:
		message = mpt::String::Replace(message, "\n", std::string(1, InternalLineEnding));
		break;
	case leCR:
		message = mpt::String::Replace(message, "\r", std::string(1, InternalLineEnding));
		break;
	case leCRLF:
		message = mpt::String::Replace(message, "\r\n", std::string(1, InternalLineEnding));
		break;
	default:
		MPT_ASSERT_NOTREACHED();
		break;
	}
	if(message == *this)
	{
		return false;
	}
	assign(message);
	return true;
}


OPENMPT_NAMESPACE_END
