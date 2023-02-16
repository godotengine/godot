//Linux 
#pragma once

#include "PsdPch.h"
#include "PsdAllocator.h"
#include "PsdFile.h"
#include "PsdNamespace.h"

PSD_NAMESPACE_BEGIN

/// \ingroup Files
/// \brief Simple file implementation that uses Posix asio internally.
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
	
	int m_fd;
};


PSD_NAMESPACE_END

