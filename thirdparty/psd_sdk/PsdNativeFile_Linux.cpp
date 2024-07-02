
#include "PsdPch.h"

#include "PsdLog.h"
#include "PsdFile.h"
#include "PsdAllocator.h"
#include "PsdNamespace.h"
#include "PsdMemoryUtil.h"
#include "PsdStringUtil.h"
#include "PsdNativeFile_Linux.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <aio.h>

#include <cstring>
#include <cstdlib>
#include <cwchar>
#include <string>


namespace
{
	//Wait for R/W
	namespace _Psd = PSD_NAMESPACE_NAME;
	static bool generic_wait(aiocb *operation,_Psd::Allocator *alloc){
		//Wait for it
		if(aio_suspend(&operation,1,nullptr) == -1)
		{
			PSD_ERROR("NativeFile","aio_suspend() => %s",strerror(errno));
			_Psd::memoryUtil::Free(alloc,operation);
			return false;
		}
		//Get status
		ssize_t ret = aio_return(operation);
		int errcode = aio_error(operation);

		if(ret == -1)
		{
			PSD_ERROR("NativeFile","aio_error() %d => %s",errcode,strerror(errcode));
		}
		_Psd::memoryUtil::Free(alloc,operation);
		return ret != -1;
	}
}


PSD_NAMESPACE_BEGIN

NativeFile::NativeFile(Allocator *alloc):
	File(alloc),
	m_fd(-1)
{

}

//Convert wchar to char and open
bool NativeFile::DoOpenRead(const wchar_t* filename)
{
	char *name = stringUtil::ConvertWString(filename,m_allocator);
	m_fd = open(name,O_RDONLY);
	if(m_fd == -1)
	{
		PSD_ERROR("NativeFile","open(%s) => %s",name,strerror(errno));
		m_allocator->Free(name);
		return false;
	}
	m_allocator->Free(name);
	return true;
}
bool NativeFile::DoOpenWrite(const wchar_t* filename){
	char *name = stringUtil::ConvertWString(filename,m_allocator);
	//Create a new file
	m_fd = open(name,O_WRONLY | O_CREAT,S_IRUSR | S_IWUSR);
	if(m_fd == -1)
	{
		PSD_ERROR("NativeFile","open(%s) => %s",name,strerror(errno));
		m_allocator->Free(name);
		return false;
	}
	m_allocator->Free(name);
	return true;
}
bool NativeFile::DoClose()
{
	int ret = close(m_fd);
	m_fd = -1;
	return ret == 0;
}

//Wrtie / Read

File::ReadOperation NativeFile::DoRead(void* buffer, uint32_t count, uint64_t position)
{
	aiocb *operation = memoryUtil::Allocate<aiocb>(m_allocator);
	std::memset(operation,0,sizeof(aiocb));

	operation->aio_buf = buffer;
	operation->aio_fildes = m_fd;
	operation->aio_lio_opcode = LIO_READ;//Do read
	operation->aio_nbytes = count;
	operation->aio_offset = position;
	operation->aio_reqprio = 0;
	operation->aio_sigevent.sigev_notify = SIGEV_NONE;//No signal will be send 

	//OK Execute it
	if(aio_read(operation) == -1)
	{
		//Has Error
		PSD_ERROR("NativeFile","On DoRead aio_read(m_fd:%d) => %s",m_fd,strerror(errno));
		memoryUtil::Free(m_allocator,operation);
		return nullptr;
	}
	return operation;
}
File::ReadOperation NativeFile::DoWrite(const void* buffer, uint32_t count, uint64_t position)
{
	aiocb *operation = memoryUtil::Allocate<aiocb>(m_allocator);
	std::memset(operation,0,sizeof(aiocb));
	
	operation->aio_buf = const_cast<void*>(buffer);
	operation->aio_fildes = m_fd;
	operation->aio_lio_opcode = LIO_WRITE;//Do Write
	operation->aio_nbytes = count;
	operation->aio_offset = position;
	operation->aio_reqprio = 0;
	operation->aio_sigevent.sigev_notify = SIGEV_NONE;//No signal will be send 

	//OK Execute it
	if(aio_write(operation) == -1)
	{
		//Has Error
		PSD_ERROR("NativeFile","On DoWrite aio_write(m_fd:%d) => %s",m_fd,strerror(errno));
		memoryUtil::Free(m_allocator,operation);
		return nullptr;
	}
	return operation;
}


bool NativeFile::DoWaitForRead(ReadOperation &_operation)
{
	aiocb *operation = static_cast<aiocb*>(_operation);
	return generic_wait(operation,m_allocator);
}
bool NativeFile::DoWaitForWrite(ReadOperation &_operation)
{
	aiocb *operation = static_cast<aiocb*>(_operation);
	return generic_wait(operation,m_allocator);
}


uint64_t NativeFile::DoGetSize() const
{
	struct stat s;
	if(fstat(m_fd,&s) == -1){
		PSD_ERROR("NativeFile","fstat(%d) => %s",m_fd,strerror(errno));
		//Emm,return 0 on error
		return 0;
	}
	return s.st_size;
}

PSD_NAMESPACE_END
