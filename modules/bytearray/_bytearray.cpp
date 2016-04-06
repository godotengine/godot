#include "_bytearray.h"

_ByteArray::_ByteArray()
{

}
void _ByteArray::_bind_methods() {


    ObjectTypeDB::bind_method(_MD("clear"),&_ByteArray::clear);
    ObjectTypeDB::bind_method(_MD("seek","pos"),&_ByteArray::seek);

    ObjectTypeDB::bind_method(_MD("get_pos"),&_ByteArray::get_pos);
    ObjectTypeDB::bind_method(_MD("get_len"),&_ByteArray::get_len);
    ObjectTypeDB::bind_method(_MD("eof_reached"),&_ByteArray::eof_reached);

    ObjectTypeDB::bind_method(_MD("read_8"),&_ByteArray::read_8);
    ObjectTypeDB::bind_method(_MD("read_16"),&_ByteArray::read_16);
    ObjectTypeDB::bind_method(_MD("read_32"),&_ByteArray::read_32);
    ObjectTypeDB::bind_method(_MD("read_64"),&_ByteArray::read_64);
    ObjectTypeDB::bind_method(_MD("read_u8"),&_ByteArray::read_u8);
    ObjectTypeDB::bind_method(_MD("read_u16"),&_ByteArray::read_u16);
    ObjectTypeDB::bind_method(_MD("read_u32"),&_ByteArray::read_u32);
    ObjectTypeDB::bind_method(_MD("read_u64"),&_ByteArray::read_u64);
    ObjectTypeDB::bind_method(_MD("read_float"),&_ByteArray::read_float);
    ObjectTypeDB::bind_method(_MD("read_double"),&_ByteArray::read_double);
    ObjectTypeDB::bind_method(_MD("read_buffer","len"),&_ByteArray::read_buffer);
    ObjectTypeDB::bind_method(_MD("read_line"),&_ByteArray::read_line);
    ObjectTypeDB::bind_method(_MD("read_as_text"),&_ByteArray::read_as_text);
    ObjectTypeDB::bind_method(_MD("read_csv_line"),&_ByteArray::read_csv_line);
    ObjectTypeDB::bind_method(_MD("read_pascal_string"),&_ByteArray::read_pascal_string);

    ObjectTypeDB::bind_method(_MD("get_endian_swap"),&_ByteArray::get_endian_swap);
    ObjectTypeDB::bind_method(_MD("set_endian_swap","enable"),&_ByteArray::set_endian_swap);

    ObjectTypeDB::bind_method(_MD("write_8","value"),&_ByteArray::write_8);
    ObjectTypeDB::bind_method(_MD("write_16","value"),&_ByteArray::write_16);
    ObjectTypeDB::bind_method(_MD("write_32","value"),&_ByteArray::write_32);
    ObjectTypeDB::bind_method(_MD("write_64","value"),&_ByteArray::write_64);
    ObjectTypeDB::bind_method(_MD("write_u8","value"),&_ByteArray::write_u8);
    ObjectTypeDB::bind_method(_MD("write_u16","value"),&_ByteArray::write_u16);
    ObjectTypeDB::bind_method(_MD("write_u32","value"),&_ByteArray::write_u32);
    ObjectTypeDB::bind_method(_MD("write_u64","value"),&_ByteArray::write_u64);
    ObjectTypeDB::bind_method(_MD("write_float","value"),&_ByteArray::write_float);
    ObjectTypeDB::bind_method(_MD("write_double","value"),&_ByteArray::write_double);
    ObjectTypeDB::bind_method(_MD("write_buffer","buffer"),&_ByteArray::write_buffer);
    ObjectTypeDB::bind_method(_MD("write_line","line"),&_ByteArray::write_line);
    ObjectTypeDB::bind_method(_MD("write_string","string"),&_ByteArray::write_string);
    ObjectTypeDB::bind_method(_MD("write_pascal_string","string"),&_ByteArray::write_pascal_string);



}
void _ByteArray::clear()
{
    data.resize(0);
    position=0;
}

void _ByteArray::seek(int64_t p_position)
{
    position=p_position;
}

int64_t _ByteArray::get_pos() const
{
    return position;
}

int64_t _ByteArray::get_len() const
{
    return data.size();
}
bool _ByteArray::eof_reached() const
{
    if(position>=data.size()){
        return true;
    }
    return false;
}
uint8_t _ByteArray::read_u8() const {
    ERR_FAIL_COND_V(position>=data.size(),0);
    uint8_t v = data.get(position);
    ++position;
    return v;
}



uint16_t _ByteArray::read_u16()const {

    uint16_t res;
    uint8_t a,b;

    a=read_u8();
    b=read_u8();

    if (endian_swap) {

        SWAP( a,b );
    }

    res=b;
    res<<=8;
    res|=a;

    return res;
}
uint32_t _ByteArray::read_u32() const{

    uint32_t res;
    uint16_t a,b;

    a=read_u16();
    b=read_u16();

    if (endian_swap) {

        SWAP( a,b );
    }

    res=b;
    res<<=16;
    res|=a;

    return res;
}
uint64_t _ByteArray::read_u64()const {

    uint64_t res;
    uint32_t a,b;

    a=read_u32();
    b=read_u32();

    if (endian_swap) {

        SWAP( a,b );
    }

    res=b;
    res<<=32;
    res|=a;

    return res;

}
int8_t _ByteArray::read_8() const {

    return read_u8();
}

int16_t _ByteArray::read_16()const {

    return read_u16();
}
int32_t _ByteArray::read_32() const{

    return read_u32();

}
int64_t _ByteArray::read_64()const {

    return read_u64();

}
float _ByteArray::read_float() const {

    MarshallFloat m;
    m.i = read_u32();
    return m.f;
}

double _ByteArray::read_double() const {

    MarshallDouble m;
    m.l = read_u64();
    return m.d;
}

Vector<String> _ByteArray::read_csv_line() const {

    String l;
    int qc=0;
    do {
        l+=read_line();
        qc=0;
        for(int i=0;i<l.length();i++) {

            if (l[i]=='"')
                qc++;
        }


    } while (qc%2);

    Vector<String> strings;

    bool in_quote=false;
    String current;
    for(int i=0;i<l.length();i++) {

        CharType c = l[i];
        CharType s[2]={0,0};


        if (!in_quote && c==',') {
            strings.push_back(current);
            current=String();
        } else if (c=='"') {
            if (l[i+1]=='"') {
                s[0]='"';
                current+=s;
                i++;
            } else {

                in_quote=!in_quote;
            }
        } else {
            s[0]=c;
            current+=s;
        }

    }

    strings.push_back(current);

    return strings;
}
String _ByteArray::read_as_text() const
{
    String text;
    String l = "";
    while(!eof_reached()) {
        l = read_line();
        text+=l+"\n";
    }
    return text;
}
String _ByteArray::read_pascal_string() const {

    uint32_t sl = read_u16();
    CharString cs;
    cs.resize(sl+1);
    _read_buffer((uint8_t*)cs.ptr(),sl);
    cs[sl]=0;

    String ret;
    ret.parse_utf8(cs.ptr());

    return ret;
}

String _ByteArray::read_line() const {

    CharString line;

    CharType c=read_u8();

    while(!eof_reached()) {

        if (c=='\n' || c=='\0') {
            line.push_back(0);
            return String::utf8(line.get_data());
        } else if (c!='\r')
            line.push_back(c);

        c=read_u8();
    }
    line.push_back(0);
    return String::utf8(line.get_data());
}

int _ByteArray::_read_buffer(uint8_t *p_dst,int p_length) const{

    int i=0;
    for (i=0; i<p_length && !eof_reached(); i++)
        p_dst[i]=read_u8();

    return i;
}

DVector<uint8_t> _ByteArray::read_buffer(int p_length) const{

    DVector<uint8_t> db;
    ERR_FAIL_COND_V(p_length<0,db);
    if (p_length==0)
        return db;
    Error err = db.resize(p_length);
    ERR_FAIL_COND_V(err!=OK,db);
    DVector<uint8_t>::Write w = db.write();
    int len = _read_buffer(&w[0],p_length);
    ERR_FAIL_COND_V( len < 0 , DVector<uint8_t>());
    w = DVector<uint8_t>::Write();

    if (len < p_length)
        db.resize(p_length);

    return db;

}

void _ByteArray::write_u8(uint8_t p_byte) {

    data.push_back(p_byte);
}

void _ByteArray::write_u16(uint16_t p_dest) {

    uint8_t a,b;

    a=p_dest&0xFF;
    b=p_dest>>8;

    if (endian_swap) {

        SWAP( a,b );
    }

    write_u8(a);
    write_u8(b);

}
void _ByteArray::write_u32(uint32_t p_dest) {


    uint16_t a,b;

    a=p_dest&0xFFFF;
    b=p_dest>>16;

    if (endian_swap) {

        SWAP( a,b );
    }

    write_u16(a);
    write_u16(b);

}
void _ByteArray::write_u64(uint64_t p_dest) {

    uint32_t a,b;

    a=p_dest&0xFFFFFFFF;
    b=p_dest>>32;

    if (endian_swap) {

        SWAP( a,b );
    }

    write_u32(a);
    write_u32(b);

}
void _ByteArray::write_8(int8_t p_byte) {

     write_u8(p_byte);
}

void _ByteArray::write_16(int16_t p_dest) {
    write_u16(p_dest);
}
void _ByteArray::write_32(int32_t p_dest) {
    write_u32(p_dest);
}
void _ByteArray::write_64(int64_t p_dest) {
    write_u64(p_dest);
}
void _ByteArray::write_float(float p_dest) {

    MarshallFloat m;
    m.f = p_dest;
    write_u32(m.i);
}

void _ByteArray::write_double(double p_dest) {

    MarshallDouble m;
    m.d = p_dest;
    write_u64(m.l);
}
void _ByteArray::write_string(const String& p_string) {

    if (p_string.length()==0)
        return;

    CharString cs=p_string.utf8();
    _write_buffer((uint8_t*)&cs[0],cs.length());

}

void _ByteArray::write_pascal_string(const String& p_string) {

    CharString cs = p_string.utf8();
    write_u16(cs.length());
    _write_buffer((uint8_t*)&cs[0], cs.length());
}
void _ByteArray::write_line(const String& p_line) {

    write_string(p_line);
    write_u8('\n');
}
void _ByteArray::write_buffer(const DVector<uint8_t> &p_buffer){

    int len = p_buffer.size();
    ERR_FAIL_COND(len<=0);
    DVector<uint8_t>::Read r = p_buffer.read();
    _write_buffer(&r[0],len);

}
void _ByteArray::_write_buffer(const uint8_t *p_src,int p_length) {

    for (int i=0;i<p_length;i++)
        write_u8(p_src[i]);
}

_ByteArray::~_ByteArray()
{

}

