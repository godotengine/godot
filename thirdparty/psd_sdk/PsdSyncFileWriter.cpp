// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdSyncFileWriter.h"

#include "PsdFile.h"


PSD_NAMESPACE_BEGIN

// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
SyncFileWriter::SyncFileWriter(File* file)
	: m_file(file)
	, m_position(0ull)
{
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void SyncFileWriter::Write(const void* buffer, uint32_t count)
{
	// do an asynchronous write, wait until it's finished, and update the file position
	File::WriteOperation op = m_file->Write(buffer, count, m_position);
	m_file->WaitForWrite(op);

	m_position += count;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
uint64_t SyncFileWriter::GetPosition(void) const
{
	return m_position;
}

PSD_NAMESPACE_END
