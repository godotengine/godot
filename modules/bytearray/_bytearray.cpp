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

    ObjectTypeDB::bind_method(_MD("get_8"),&_ByteArray::get_8);
    ObjectTypeDB::bind_method(_MD("get_16"),&_ByteArray::get_16);
    ObjectTypeDB::bind_method(_MD("get_32"),&_ByteArray::get_32);
    ObjectTypeDB::bind_method(_MD("get_64"),&_ByteArray::get_64);
    ObjectTypeDB::bind_method(_MD("get_float"),&_ByteArray::get_float);
    ObjectTypeDB::bind_method(_MD("get_double"),&_ByteArray::get_double);
    ObjectTypeDB::bind_method(_MD("get_buffer","len"),&_ByteArray::get_buffer);
    ObjectTypeDB::bind_method(_MD("get_line"),&_ByteArray::get_line);
    ObjectTypeDB::bind_method(_MD("get_as_text"),&_ByteArray::get_as_text);
    ObjectTypeDB::bind_method(_MD("get_csv_line"),&_ByteArray::get_csv_line);
    ObjectTypeDB::bind_method(_MD("get_pascal_string"),&_ByteArray::get_pascal_string);

    ObjectTypeDB::bind_method(_MD("get_endian_swap"),&_ByteArray::get_endian_swap);
    ObjectTypeDB::bind_method(_MD("set_endian_swap","enable"),&_ByteArray::set_endian_swap);

    ObjectTypeDB::bind_method(_MD("store_8","value"),&_ByteArray::store_8);
    ObjectTypeDB::bind_method(_MD("store_16","value"),&_ByteArray::store_16);
    ObjectTypeDB::bind_method(_MD("store_32","value"),&_ByteArray::store_32);
    ObjectTypeDB::bind_method(_MD("store_64","value"),&_ByteArray::store_64);
    ObjectTypeDB::bind_method(_MD("store_float","value"),&_ByteArray::store_float);
    ObjectTypeDB::bind_method(_MD("store_double","value"),&_ByteArray::store_double);
    ObjectTypeDB::bind_method(_MD("store_buffer","buffer"),&_ByteArray::store_buffer);
    ObjectTypeDB::bind_method(_MD("store_line","line"),&_ByteArray::store_line);
    ObjectTypeDB::bind_method(_MD("store_string","string"),&_ByteArray::store_string);
    ObjectTypeDB::bind_method(_MD("store_pascal_string","string"),&_ByteArray::store_pascal_string);



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
uint8_t _ByteArray::get_8() const {
    ERR_FAIL_COND_V(position>=data.size(),0);
    uint8_t v = data.get(position);
    ++position;
    return v;
}

void _ByteArray::store_8(uint8_t p_byte) {

    data.push_back(p_byte);
}

uint16_t _ByteArray::get_16()const {

    uint16_t res;
    uint8_t a,b;

    a=get_8();
    b=get_8();

    if (endian_swap) {

        SWAP( a,b );
    }

    res=b;
    res<<=8;
    res|=a;

    return res;
}
uint32_t _ByteArray::get_32() const{

    uint32_t res;
    uint16_t a,b;

    a=get_16();
    b=get_16();

    if (endian_swap) {

        SWAP( a,b );
    }

    res=b;
    res<<=16;
    res|=a;

    return res;
}
uint64_t _ByteArray::get_64()const {

    uint64_t res;
    uint32_t a,b;

    a=get_32();
    b=get_32();

    if (endian_swap) {

        SWAP( a,b );
    }

    res=b;
    res<<=32;
    res|=a;

    return res;

}

float _ByteArray::get_float() const {

    MarshallFloat m;
    m.i = get_32();
    return m.f;
}

double _ByteArray::get_double() const {

    MarshallDouble m;
    m.l = get_64();
    return m.d;
}

Vector<String> _ByteArray::get_csv_line() const {

    String l;
    int qc=0;
    do {
        l+=get_line();
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
String _ByteArray::get_as_text() const
{
    String text;
    String l = "";
    while(!eof_reached()) {
        l = get_line();
        text+=l+"\n";
    }
    return text;
}
String _ByteArray::get_pascal_string() {

    uint32_t sl = get_32();
    CharString cs;
    cs.resize(sl+1);
    _get_buffer((uint8_t*)cs.ptr(),sl);
    cs[sl]=0;

    String ret;
    ret.parse_utf8(cs.ptr());

    return ret;
}

String _ByteArray::get_line() const {

    CharString line;

    CharType c=get_8();

    while(!eof_reached()) {

        if (c=='\n' || c=='\0') {
            line.push_back(0);
            return String::utf8(line.get_data());
        } else if (c!='\r')
            line.push_back(c);

        c=get_8();
    }
    line.push_back(0);
    return String::utf8(line.get_data());
}

int _ByteArray::_get_buffer(uint8_t *p_dst,int p_length) const{

    int i=0;
    for (i=0; i<p_length && !eof_reached(); i++)
        p_dst[i]=get_8();

    return i;
}

DVector<uint8_t> _ByteArray::get_buffer(int p_length) const{

    DVector<uint8_t> db;
    ERR_FAIL_COND_V(p_length<0,db);
    if (p_length==0)
        return db;
    Error err = db.resize(p_length);
    ERR_FAIL_COND_V(err!=OK,db);
    DVector<uint8_t>::Write w = db.write();
    int len = _get_buffer(&w[0],p_length);
    ERR_FAIL_COND_V( len < 0 , DVector<uint8_t>());
    w = DVector<uint8_t>::Write();

    if (len < p_length)
        db.resize(p_length);

    return db;

}

void _ByteArray::store_16(uint16_t p_dest) {

    uint8_t a,b;

    a=p_dest&0xFF;
    b=p_dest>>8;

    if (endian_swap) {

        SWAP( a,b );
    }

    store_8(a);
    store_8(b);

}
void _ByteArray::store_32(uint32_t p_dest) {


    uint16_t a,b;

    a=p_dest&0xFFFF;
    b=p_dest>>16;

    if (endian_swap) {

        SWAP( a,b );
    }

    store_16(a);
    store_16(b);

}
void _ByteArray::store_64(uint64_t p_dest) {

    uint32_t a,b;

    a=p_dest&0xFFFFFFFF;
    b=p_dest>>32;

    if (endian_swap) {

        SWAP( a,b );
    }

    store_32(a);
    store_32(b);

}

void _ByteArray::store_float(float p_dest) {

    MarshallFloat m;
    m.f = p_dest;
    store_32(m.i);
}

void _ByteArray::store_double(double p_dest) {

    MarshallDouble m;
    m.d = p_dest;
    store_64(m.l);
}
void _ByteArray::store_string(const String& p_string) {

    if (p_string.length()==0)
        return;

    CharString cs=p_string.utf8();
    _store_buffer((uint8_t*)&cs[0],cs.length());

}

void _ByteArray::store_pascal_string(const String& p_string) {

    CharString cs = p_string.utf8();
    store_32(cs.length());
    _store_buffer((uint8_t*)&cs[0], cs.length());
}
void _ByteArray::store_line(const String& p_line) {

    store_string(p_line);
    store_8('\n');
}
void _ByteArray::store_buffer(const DVector<uint8_t> &p_buffer){

    int len = p_buffer.size();
    ERR_FAIL_COND(len<=0);
    DVector<uint8_t>::Read r = p_buffer.read();
    _store_buffer(&r[0],len);

}
void _ByteArray::_store_buffer(const uint8_t *p_src,int p_length) {

    for (int i=0;i<p_length;i++)
        store_8(p_src[i]);
}

_ByteArray::~_ByteArray()
{

}

