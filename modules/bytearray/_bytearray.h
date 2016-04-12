#ifndef _BYTEARRAY_H
#define _BYTEARRAY_H
#include "reference.h"
#include "core/io/marshalls.h"

class _ByteArray: public Reference {

    OBJ_TYPE(_ByteArray,Reference);
private:
    DVector<uint8_t> data;
    bool endian_swap;
    bool real_is_double;
    mutable int position;
protected:
     static void _bind_methods();
public:
    void seek(int64_t p_position);
    void clear();
    int64_t get_pos() const; ///< get position in the file
    int64_t get_len() const;

    bool eof_reached() const; ///< reading passed EOF

    int8_t read_8() const; ///< get a byte
    int16_t read_16() const; ///< get 16 bits uint
    int32_t read_32() const; ///< get 32 bits uint
    int64_t read_64() const; ///< get 64 bits uint
    uint8_t read_u8() const; ///< get a byte
    uint16_t read_u16() const; ///< get 16 bits uint
    uint32_t read_u32() const; ///< get 32 bits uint
    uint64_t read_u64() const; ///< get 64 bits uint

    float read_float() const;
    double read_double() const;
    DVector<uint8_t> read_buffer(int p_length) const;
    int _read_buffer(uint8_t *p_dst,int p_length) const; ///< get an array of bytes
    String read_line() const;
    Vector<String> read_csv_line() const;
    String read_pascal_string() const;
    String read_as_text() const;
    /**< use this for files WRITTEN in _big_ endian machines (ie, amiga/mac)
     * It's not about the current CPU type but file formats.
     * this flags get reset to false (little endian) on each open
     */

    void set_endian_swap(bool p_swap) { endian_swap=p_swap; }
    bool get_endian_swap() const { return endian_swap; }

    void write_u8(uint8_t p_dest); ///< store a byte
    void write_u16(uint16_t p_dest); ///< store 16 bits uint
    void write_u32(uint32_t p_dest); ///< store 32 bits uint
    void write_u64(uint64_t p_dest); ///< store 64 bits uint

    void write_8(int8_t p_dest); ///< store a byte
    void write_16(int16_t p_dest); ///< store 16 bits uint
    void write_32(int32_t p_dest); ///< store 32 bits uint
    void write_64(int64_t p_dest); ///< store 64 bits uint

    void write_float(float p_dest);
    void write_double(double p_dest);
    void write_real(real_t p_real);

    void write_string(const String& p_string);
    void write_line(const String& p_string);

    void write_pascal_string(const String& p_string);
    String get_pascal_string();

    void _write_buffer(const uint8_t *p_src,int p_length); ///< store an array of bytes
    void write_buffer(const DVector<uint8_t>& p_buffer);

public:
    _ByteArray();
    ~_ByteArray();
};

#endif // _BYTEARRAY_H
