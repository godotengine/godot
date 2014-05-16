#include "bytesbuffer.h"
BytesBuffer::BytesBuffer()
{
    position = 0;
}

Error BytesBuffer::_open(const String &p_path, int p_mode_flags)
{
    return FAILED;
}

uint64_t BytesBuffer::_get_modified_time(const String &p_file)
{
    return 0;
}

void BytesBuffer::close()
{
    data.resize(0);
    position = 0;
}

bool BytesBuffer::is_open() const
{
    return true;
}

void BytesBuffer::seek(size_t p_position)
{
    if(p_position>0 && data.size()>0){
        if(p_position>=data.size())p_position=data.size()-1;
    }
    position = p_position;
}

void BytesBuffer::seek_end(int64_t p_position)
{
    if(p_position==0){
        p_position=data.size();
        if(p_position>0)p_position-=1;
    }else{
        if(p_position>0 && data.size()>0){
            if(p_position>=data.size())p_position=data.size()-1;
        }
    }
    position = p_position;
}

size_t BytesBuffer::get_pos() const
{
    return position;
}

size_t BytesBuffer::get_len() const
{
    return data.size();
}

bool BytesBuffer::eof_reached() const
{
    return position>=data.size()?true:false;
}

uint8_t BytesBuffer::get_8() const
{
    ERR_FAIL_COND_V(position>=data.size(),0);
    uint8_t v = data.get(position);
    ++position;
    return v;
}

Error BytesBuffer::get_error() const
{
    return OK;
}

void BytesBuffer::store_8(uint8_t p_dest)
{
    data.push_back(p_dest);
}

bool BytesBuffer::file_exists(const String &p_name)
{
    return true;
}
 BytesBuffer::~BytesBuffer()
 {
    if(position>0 || data.size()>0)close();
 }
