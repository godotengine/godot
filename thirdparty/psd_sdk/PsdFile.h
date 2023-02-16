// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

class Allocator;


/// \ingroup Interfaces
/// \ingroup Files
/// \brief Base class for all files.
/// \details Files are used throughout the library to fully control all file I/O operations. This makes it possible to use
/// native (platform- and OS-provided) functions for e.g. asynchronous I/O.
/// \remark Note that the interface only offers asynchronous read operations. The reason for this is that asynchronous reads
/// allow for parallelizing file accesses to the same file, while still being able to add synchronous reads as a wrapper on top.
/// \sa NativeFile SyncFileReader
class File
{
public:
	/// A type representing an object associated with a read operation.
	typedef void* ReadOperation;

	/// A type representing an object associated with a write operation.
	typedef void* WriteOperation;

	/// Constructor.
	explicit File(Allocator* allocator);

	/// Empty destructor.
	virtual ~File(void);

	/// Tries to open a file for reading, and returns whether the operation was successful.
	bool OpenRead(const wchar_t* filename);

	/// Tries to open a file for writing, and returns whether the operation was successful.
	bool OpenWrite(const wchar_t* filename);

	/// Tries to close a file, and returns whether the operation was successful.
	bool Close(void);

	/// Asynchronously loads count bytes into the buffer, reading from position in the file.
	/// The returned ReadOperation must be used in a call to WaitForRead() in order to free resources associated with it.
	ReadOperation Read(void* buffer, uint32_t count, uint64_t position);

	/// Waits until the read operation associated with the given object is finished, and deletes its internal resources.
	bool WaitForRead(ReadOperation& operation);

	/// Asynchronously writes count bytes from the buffer, writing to position in the file.
	/// The returned WriteOperation must be used in a call to WaitForWrite() in order to free resources associated with it.
	WriteOperation Write(const void* buffer, uint32_t count, uint64_t position);

	/// Waits until the write operation associated with the given object is finished, and deletes its internal resources.
	bool WaitForWrite(WriteOperation& operation);

	/// Returns the size of the file. Calling this method is only valid on a file that has successfully been opened by a call to Open() previously.
	/// If the function fails, 0 will be returned.
	uint64_t GetSize(void) const;

protected:
	Allocator* m_allocator;

private:
	virtual bool DoOpenRead(const wchar_t* filename) PSD_ABSTRACT;
	virtual bool DoOpenWrite(const wchar_t* filename) PSD_ABSTRACT;
	virtual bool DoClose(void) PSD_ABSTRACT;

	virtual ReadOperation DoRead(void* buffer, uint32_t count, uint64_t position) PSD_ABSTRACT;
	virtual bool DoWaitForRead(ReadOperation& operation) PSD_ABSTRACT;

	virtual WriteOperation DoWrite(const void* buffer, uint32_t count, uint64_t position) PSD_ABSTRACT;
	virtual bool DoWaitForWrite(WriteOperation& operation) PSD_ABSTRACT;

	virtual uint64_t DoGetSize(void) const PSD_ABSTRACT;
};

PSD_NAMESPACE_END
