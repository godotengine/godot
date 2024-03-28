// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdSyncFileReader.h"

#include "PsdFile.h"


PSD_NAMESPACE_BEGIN

// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
SyncFileReader::SyncFileReader(File* file)
	: m_file(file)
	, m_position(0ull)
{
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void SyncFileReader::Read(void* buffer, uint32_t count)
{
	// do an asynchronous read, wait until it's finished, and update the file position
	File::ReadOperation op = m_file->Read(buffer, count, m_position);
	m_file->WaitForRead(op);

	m_position += count;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void SyncFileReader::Skip(uint64_t count)
{
	m_position += count;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void SyncFileReader::SetPosition(uint64_t position)
{
	m_position = position;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
uint64_t SyncFileReader::GetPosition(void) const
{
	return m_position;
}

PSD_NAMESPACE_END
