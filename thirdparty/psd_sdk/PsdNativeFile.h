// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once

#include "PsdFile.h"


PSD_NAMESPACE_BEGIN

/// \ingroup Files
/// \brief Simple file implementation that uses Windows' native file functions internally.
/// \sa File
class NativeFile : public File
{
public:
	/// Constructor.
	explicit NativeFile(Allocator* allocator);

private:
	virtual bool DoOpenRead(const wchar_t* filename) PSD_OVERRIDE;
	virtual bool DoOpenWrite(const wchar_t* filename) PSD_OVERRIDE;
	virtual bool DoClose(void) PSD_OVERRIDE;

	virtual File::ReadOperation DoRead(void* buffer, uint32_t count, uint64_t position) PSD_OVERRIDE;
	virtual bool DoWaitForRead(File::ReadOperation& operation) PSD_OVERRIDE;

	virtual File::WriteOperation DoWrite(const void* buffer, uint32_t count, uint64_t position) PSD_OVERRIDE;
	virtual bool DoWaitForWrite(File::WriteOperation& operation) PSD_OVERRIDE;

	virtual uint64_t DoGetSize(void) const PSD_OVERRIDE;

	// internally, this is a HANDLE, but we don't want to include Windows.h just for that
	void* m_file;
};

PSD_NAMESPACE_END
