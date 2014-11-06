#ifndef BYTESBUFFER_H
#define BYTESBUFFER_H
#include "dvector.h"
#include "file_access.h"
class BytesBuffer : public FileAccess
{
private:
    DVector<uint8_t> data;
    mutable size_t position;
public:
    BytesBuffer();
    ~BytesBuffer();

protected:
    virtual Error _open(const String& p_path, int p_mode_flags); ///< open a file
    virtual uint64_t _get_modified_time(const String& p_file);
public:
    virtual void close(); ///< close a file
    virtual bool is_open() const; ///< true when file is open
    virtual void seek(size_t p_position); ///< seek to a given position
    virtual void seek_end(int64_t p_position=0); ///< seek from the end of file
    virtual size_t get_pos() const; ///< get position in the file
    virtual size_t get_len() const; ///< get size of the file
    virtual bool eof_reached() const; ///< reading passed EOF
    virtual uint8_t get_8() const; ///< get a byte
    virtual Error get_error() const; ///< get last error
    virtual void store_8(uint8_t p_dest); ///< store a byte
    virtual bool file_exists(const String& p_name); ///< return true if a file exists
};

#endif // BYTESBUFFER_H
