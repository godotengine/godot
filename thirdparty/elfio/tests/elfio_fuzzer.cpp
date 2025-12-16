#include <string>
#include <sstream>

#include <elfio/elfio.hpp>
#include <elfio/elfio_dump.hpp>

using namespace ELFIO;

extern "C" int LLVMFuzzerTestOneInput( const std::uint8_t* Data, size_t Size )
{
    std::string        str( (const char*)Data, Size );
    std::istringstream ss( str );
    std::ostringstream oss;

    elfio elf;

    if ( !elf.load( ss ) ) {
        return 0;
    }

    dump::header( oss, elf );
    dump::section_headers( oss, elf );
    dump::segment_headers( oss, elf );
    dump::symbol_tables( oss, elf );
    dump::notes( oss, elf );
    dump::modinfo( oss, elf );
    dump::dynamic_tags( oss, elf );
    dump::section_datas( oss, elf );
    dump::segment_datas( oss, elf );

    return 0;
}
