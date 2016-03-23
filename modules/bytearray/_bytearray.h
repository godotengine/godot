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

    uint8_t get_8() const; ///< get a byte
    uint16_t get_16() const; ///< get 16 bits uint
    uint32_t get_32() const; ///< get 32 bits uint
    uint64_t get_64() const; ///< get 64 bits uint

    float get_float() const;
    double get_double() const;
    real_t get_real() const;
    DVector<uint8_t> get_buffer(int p_length) const;
    int _get_buffer(uint8_t *p_dst,int p_length) const; ///< get an array of bytes
    String get_line() const;
    Vector<String> get_csv_line() const;
    String get_as_text() const;
    /**< use this for files WRITTEN in _big_ endian machines (ie, amiga/mac)
     * It's not about the current CPU type but file formats.
     * this flags get reset to false (little endian) on each open
     */

    void set_endian_swap(bool p_swap) { endian_swap=p_swap; }
    bool get_endian_swap() const { return endian_swap; }

    void store_8(uint8_t p_dest); ///< store a byte
    void store_16(uint16_t p_dest); ///< store 16 bits uint
    void store_32(uint32_t p_dest); ///< store 32 bits uint
    void store_64(uint64_t p_dest); ///< store 64 bits uint

    void store_float(float p_dest);
    void store_double(double p_dest);
    void store_real(real_t p_real);

    void store_string(const String& p_string);
    void store_line(const String& p_string);

    void store_pascal_string(const String& p_string);
    String get_pascal_string();

    void _store_buffer(const uint8_t *p_src,int p_length); ///< store an array of bytes
    void store_buffer(const DVector<uint8_t>& p_buffer);

public:
    _ByteArray();
    ~_ByteArray();
};

#endif // _BYTEARRAY_H
