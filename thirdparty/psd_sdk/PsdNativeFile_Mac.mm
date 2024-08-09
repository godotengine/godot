//
//  PsdNativeFile_Mac.mm
//  Contributed to psd_sdk
//
//  Created by Oluseyi Sonaiya on 3/29/20.
//  Copyright © 2020 Oluseyi Sonaiya. All rights reserved.
//
// psd_sdk Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include <wchar.h>
#include <codecvt>
#include <locale>
#include <string>

#include "PsdPch.h"
#include "PsdNativeFile_Mac.h"

#include "PsdAllocator.h"
#include "PsdPlatform.h"
#include "PsdMemoryUtil.h"
#include "PsdLog.h"
#include "Psdinttypes.h"


PSD_NAMESPACE_BEGIN

typedef void (^DispatchIOHandler)(dispatch_data_t data, int error);

struct DispatchReadOperation
{
    void* dataReadBuffer;
    uint32_t length;
    uint64_t offset;
    DispatchIOHandler ioHandler;
    dispatch_semaphore_t semaphore;
};

struct DispatchWriteOperation
{
    dispatch_data_t dataToWrite;
    size_t bytesWritten;
    uint32_t length;
    uint64_t offset;
    DispatchIOHandler ioHandler;
    dispatch_semaphore_t semaphore;
};


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
NativeFile::NativeFile(Allocator* allocator)
    : File(allocator)
    , m_fileDescriptor(0)
{
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
bool NativeFile::DoOpenRead(const wchar_t* filename)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>,wchar_t> convert;
    std::string s = convert.to_bytes(filename);
    char const *cs = s.c_str();
    m_fileDescriptor = open(cs, O_RDONLY);
    if (m_fileDescriptor == -1)
    {
        PSD_ERROR("NativeFile", "Cannot obtain handle for file \"%ls\".", filename);
        return false;
    }

    return true;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
bool NativeFile::DoOpenWrite(const wchar_t* filename)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>,wchar_t> convert;
    std::string s = convert.to_bytes(filename);
    char const *cs = s.c_str();
    m_fileDescriptor = open(cs, O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP);
    if (m_fileDescriptor == -1)
    {
        PSD_ERROR("NativeFile", "Cannot obtain handle for file \"%ls\".", filename);
        return false;
    }

    return true;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
bool NativeFile::DoClose(void)
{
    if (m_fileDescriptor == -1)
        return false;
    
    const int success = close(m_fileDescriptor);
    if  (success == -1)
    {
        PSD_ERROR("NativeFile", "Cannot close handle.");
        return false;
    }

    m_fileDescriptor = -1;
    return true;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
File::ReadOperation NativeFile::DoRead(void* buffer, uint32_t count, uint64_t position)
{
    DispatchReadOperation *operation = memoryUtil::Allocate<DispatchReadOperation>(m_allocator);
    operation->dataReadBuffer = buffer;
    operation->length = count;
    operation->offset = position;
    operation->ioHandler = ^(dispatch_data_t data, int error)
    {
        dispatch_data_apply(data, ^bool(dispatch_data_t  _Nonnull region, size_t offset, const void * _Nonnull buffer, size_t size)
            {
            // TODO: make sure this doesn't get called because PSD file is loaded as multiple data regions
                memcpy(operation->dataReadBuffer, buffer, size);
                dispatch_semaphore_signal(operation->semaphore);
                return true;
            });

        size_t bytesRead = dispatch_data_get_size(data);
        if (bytesRead < operation->length)
        {
            PSD_ERROR("NativeFile", "Cannot read %u bytes from file position %" PRIu64 " asynchronously.", count, position);
        }
    };
    return static_cast<File::ReadOperation>(operation);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
bool NativeFile::DoWaitForRead(File::ReadOperation& operation)
{
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    
    DispatchReadOperation *op = static_cast<DispatchReadOperation *>(operation);
    lseek(m_fileDescriptor, op->offset, SEEK_SET);
    op->semaphore = dispatch_semaphore_create(0);
    dispatch_read(m_fileDescriptor, op->length, queue, op->ioHandler);
    
    dispatch_semaphore_wait(op->semaphore, DISPATCH_TIME_FOREVER);
    if (op->dataReadBuffer == nil)
    {
        PSD_ERROR("NativeFile", "Failed to wait for previous asynchronous read operation.");
        return false;
    }

    return true;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
File::WriteOperation NativeFile::DoWrite(const void* buffer, uint32_t count, uint64_t position)
{
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

    DispatchWriteOperation *operation = memoryUtil::Allocate<DispatchWriteOperation>(m_allocator);
    operation->length = count;
    operation->offset = position;
    operation->dataToWrite = dispatch_data_create(buffer, count, queue, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    operation->ioHandler = ^(dispatch_data_t d, int error)
    {
        if (d != NULL || error != 0 )
        {
            PSD_ERROR("NativeFile", "Cannot write %u bytes to file position %" PRIu64 " asynchronously.", count, position);
        }
        else
        {
            operation->bytesWritten = operation->length;
        }
        dispatch_semaphore_signal(operation->semaphore);
    };
    return static_cast<File::ReadOperation>(operation);
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
bool NativeFile::DoWaitForWrite(File::WriteOperation& operation)
{
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    
    DispatchWriteOperation *op = static_cast<DispatchWriteOperation *>(operation);
    lseek(m_fileDescriptor, op->offset, SEEK_SET);
    op->semaphore = dispatch_semaphore_create(0);
    dispatch_write(m_fileDescriptor, op->dataToWrite, queue, op->ioHandler);
    
    dispatch_semaphore_wait(op->semaphore, DISPATCH_TIME_FOREVER);
    if (op->bytesWritten < op->length)
    {
        PSD_ERROR("NativeFile", "Failed to wait for previous asynchronous write operation.");
        return false;
    }

    return true;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
uint64_t NativeFile::DoGetSize(void) const
{
// fstat

    return 0;
}

PSD_NAMESPACE_END
