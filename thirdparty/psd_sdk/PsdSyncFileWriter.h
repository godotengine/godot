// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

class File;


/// \ingroup Files
/// \brief Synchronous file wrapper using an arbitrary \ref File implementation for sequential writes.
/// \details In certain situations, working with synchronous write operations is much easier than having to deal with a number
/// of asynchronous writes, keeping track of individual write operations. This is especially true when e.g. writing header information.
/// \sa File
class SyncFileWriter
{
public:
	/// Constructor initializing the internal write position to zero.
	/// \remark The given \a file must already be open.
	explicit SyncFileWriter(File* file);

	/// Writes \a count bytes from \a buffer synchronously, incrementing the internal write position.
	void Write(const void* buffer, uint32_t count);

	/// Returns the internal write position.
	uint64_t GetPosition(void) const;

private:
	File* m_file;
	uint64_t m_position;
};

PSD_NAMESPACE_END
