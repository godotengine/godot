/*
 * Message.h
 * ---------
 * Purpose: Various functions for processing song messages (allocating, reading from file...)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include <string>

#include "../common/FileReaderFwd.h"

OPENMPT_NAMESPACE_BEGIN

class SongMessage : public std::string
{
public:

	// Line ending types (for reading song messages from module files)
	enum LineEnding
	{
		leCR,			// Carriage Return (0x0D, \r)
		leLF,			// Line Feed (0x0A \n)
		leCRLF,			// Carriage Return, Line Feed (0x0D0A, \r\n)
		leMixed,		// It is not defined whether Carriage Return or Line Feed is the actual line ending. Both are accepted.
		leAutodetect,	// Detect suitable line ending
	};

	enum
	{
		InternalLineEnding	= '\r',	// The character that represents line endings internally
	};

	// Read song message from a mapped file.
	// [in]  data: pointer to the data in memory that is going to be read
	// [in]  length: number of characters that should be read, not including a possible trailing null terminator (it is automatically appended).
	// [in]  lineEnding: line ending formatting of the text in memory.
	// [out] returns true on success.
	bool Read(const mpt::byte *data, const size_t length, LineEnding lineEnding);
	bool Read(FileReader &file, const size_t length, LineEnding lineEnding);

	// Read comments with fixed line length from a mapped file.
	// [in]  data: pointer to the data in memory that is going to be read
	// [in]  length: number of characters that should be read, not including a possible trailing null terminator (it is automatically appended).
	// [in]  lineLength: The fixed length of a line.
	// [in]  lineEndingLength: The padding space between two fixed lines. (there could for example be a null char after every line)
	// [out] returns true on success.
	bool ReadFixedLineLength(const mpt::byte *data, const size_t length, const size_t lineLength, const size_t lineEndingLength);
	bool ReadFixedLineLength(FileReader &file, const size_t length, const size_t lineLength, const size_t lineEndingLength);

	// Retrieve song message.
	// [in]  lineEnding: line ending formatting of the text in memory.
	// [out] returns formatted song message.
	std::string GetFormatted(const LineEnding lineEnding) const;

	// Set song message.
	// [in]  lineEnding: line ending formatting of the text in memory. Must be leCR or leLF or leCRLF,
	// [out] returns true if the message has been changed.
	bool SetFormatted(std::string message, LineEnding lineEnding);

};

OPENMPT_NAMESPACE_END
