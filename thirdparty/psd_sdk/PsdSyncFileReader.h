// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

class File;


/// \ingroup Files
/// \brief Synchronous file wrapper using an arbitrary \ref File implementation for sequential reads.
/// \details In certain situations, working with synchronous read operations is much easier than having to deal with a number
/// of asynchronous reads, keeping track of individual read operations. This is especially true when parsing a file sequentially,
/// where different read operations depend on previous ones.
/// \sa File
class SyncFileReader
{
public:
	/// Constructor initializing the internal read position to zero.
	/// \remark The given \a file must already be open.
	explicit SyncFileReader(File* file);

	/// Reads \a count bytes into \a buffer synchronously, incrementing the internal read position.
	void Read(void* buffer, uint32_t count);

	/// Skips \a count bytes.
	void Skip(uint64_t count);

	/// Sets the internal read position for the next call to Read().
	void SetPosition(uint64_t position);

	/// Returns the internal read position.
	uint64_t GetPosition(void) const;

private:
	File* m_file;
	uint64_t m_position;
};

PSD_NAMESPACE_END
