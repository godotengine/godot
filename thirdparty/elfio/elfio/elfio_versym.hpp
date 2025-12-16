/*
Copyright (C) 2001-present by Serge Lamikhov-Center

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef ELFIO_VERSYM_HPP
#define ELFIO_VERSYM_HPP

namespace ELFIO {

//------------------------------------------------------------------------------
//! \class versym_section_accessor_template
//! \brief Class for accessing version symbol section data
template <class S> class versym_section_accessor_template
{
  public:
    //------------------------------------------------------------------------------
    //! \brief Constructor
    //! \param section Pointer to the section
    explicit versym_section_accessor_template( S* section )
        : versym_section( section )
    {
        if ( section != nullptr ) {
            entries_num = decltype( entries_num )( section->get_size() /
                                                   sizeof( Elf_Half ) );
        }
    }

    //------------------------------------------------------------------------------
    //! \brief Get the number of entries
    //! \return Number of entries
    Elf_Word get_entries_num() const
    {
        if ( versym_section ) {
            return entries_num;
        }
        return 0;
    }

    //------------------------------------------------------------------------------
    //! \brief Get an entry
    //! \param no Index of the entry
    //! \param value Value of the entry
    //! \return True if successful, false otherwise
    bool get_entry( Elf_Word no, Elf_Half& value ) const
    {
        if ( versym_section && ( no < get_entries_num() ) ) {
            value = ( (Elf_Half*)versym_section->get_data() )[no];
            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    //! \brief Modify an entry
    //! \param no Index of the entry
    //! \param value New value of the entry
    //! \return True if successful, false otherwise
    bool modify_entry( Elf_Word no, Elf_Half value )
    {
        if ( versym_section && ( no < get_entries_num() ) ) {
            ( (Elf_Half*)versym_section->get_data() )[no] = value;
            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------------
    //! \brief Add an entry
    //! \param value Value of the entry
    //! \return True if successful, false otherwise
    bool add_entry( Elf_Half value )
    {
        if ( !versym_section ) {
            return false;
        }

        versym_section->append_data( (const char*)&value, sizeof( Elf_Half ) );
        ++entries_num;

        return true;
    }

    //------------------------------------------------------------------------------
  private:
    S*       versym_section = nullptr; //!< Pointer to the section
    Elf_Word entries_num    = 0;       //!< Number of entries
};

using versym_section_accessor = versym_section_accessor_template<section>;
using const_versym_section_accessor =
    versym_section_accessor_template<const section>;

//------------------------------------------------------------------------------
//! \class versym_r_section_accessor_template
//! \brief Class for accessing version requirement section data
template <class S> class versym_r_section_accessor_template
{
  public:
    //------------------------------------------------------------------------------
    //! \brief Constructor
    //! \param elf_file Reference to the ELF file
    //! \param versym_r_section Pointer to the version requirement section
    versym_r_section_accessor_template( const elfio& elf_file,
                                        S*           versym_r_section )
        : elf_file( elf_file ), versym_r_section( versym_r_section ),
          entries_num( 0 )
    {
        // Find .dynamic section
        const section* dynamic_section = elf_file.sections[".dynamic"];

        if ( dynamic_section == nullptr ) {
            return;
        }

        const_dynamic_section_accessor dynamic_section_acc( elf_file,
                                                            dynamic_section );
        Elf_Xword dyn_sec_num = dynamic_section_acc.get_entries_num();
        for ( Elf_Xword i = 0; i < dyn_sec_num; ++i ) {
            Elf_Xword   tag;
            Elf_Xword   value;
            std::string str;

            if ( dynamic_section_acc.get_entry( i, tag, value, str ) &&
                 tag == DT_VERNEEDNUM ) {
                entries_num = (Elf_Word)value;
                break;
            }
        }
    }

    //------------------------------------------------------------------------------
    //! \brief Get the number of entries
    //! \return Number of entries
    Elf_Word get_entries_num() const { return entries_num; }

    //------------------------------------------------------------------------------
    //! \brief Get an entry
    //! \param no Index of the entry
    //! \param version Version of the entry
    //! \param file_name File name of the entry
    //! \param hash Hash of the entry
    //! \param flags Flags of the entry
    //! \param other Other information of the entry
    //! \param dep_name Dependency name of the entry
    //! \return True if successful, false otherwise
    bool get_entry( Elf_Word     no,
                    Elf_Half&    version,
                    std::string& file_name,
                    Elf_Word&    hash,
                    Elf_Half&    flags,
                    Elf_Half&    other,
                    std::string& dep_name ) const
    {
        if ( versym_r_section == nullptr || ( no >= get_entries_num() ) ) {
            return false;
        }

        const_string_section_accessor string_section_acc(
            elf_file.sections[versym_r_section->get_link()] );

        Elfxx_Verneed* verneed = (Elfxx_Verneed*)versym_r_section->get_data();
        Elfxx_Vernaux* veraux =
            (Elfxx_Vernaux*)( (char*)verneed + verneed->vn_aux );
        for ( Elf_Word i = 0; i < no; ++i ) {
            verneed = (Elfxx_Verneed*)( (char*)verneed + verneed->vn_next );
            veraux  = (Elfxx_Vernaux*)( (char*)verneed + verneed->vn_aux );
        }

        version   = verneed->vn_version;
        file_name = string_section_acc.get_string( verneed->vn_file );
        hash      = veraux->vna_hash;
        flags     = veraux->vna_flags;
        other     = veraux->vna_other;
        dep_name  = string_section_acc.get_string( veraux->vna_name );

        return true;
    }

    //------------------------------------------------------------------------------
  private:
    const elfio& elf_file;
    S*           versym_r_section =
        nullptr;              //!< Pointer to the version requirement section
    Elf_Word entries_num = 0; //!< Number of entries
};

using versym_r_section_accessor = versym_r_section_accessor_template<section>;
using const_versym_r_section_accessor =
    versym_r_section_accessor_template<const section>;

//------------------------------------------------------------------------------
//! \class versym_d_section_accessor_template
//! \brief Class for accessing version definition section data
template <class S> class versym_d_section_accessor_template
{
  public:
    //------------------------------------------------------------------------------
    //! \brief Constructor
    //! \param elf_file Reference to the ELF file
    //! \param versym_d_section Pointer to the version definition section
    versym_d_section_accessor_template( const elfio& elf_file,
                                        S*           versym_d_section )
        : elf_file( elf_file ), versym_d_section( versym_d_section ),
          entries_num( 0 )
    {
        // Find .dynamic section
        const section* dynamic_section = elf_file.sections[".dynamic"];

        if ( dynamic_section == nullptr ) {
            return;
        }

        const_dynamic_section_accessor dynamic_section_acc( elf_file,
                                                            dynamic_section );
        Elf_Xword dyn_sec_num = dynamic_section_acc.get_entries_num();
        for ( Elf_Xword i = 0; i < dyn_sec_num; ++i ) {
            Elf_Xword   tag;
            Elf_Xword   value;
            std::string str;

            if ( dynamic_section_acc.get_entry( i, tag, value, str ) &&
                 tag == DT_VERDEFNUM ) {
                entries_num = (Elf_Word)value;
                break;
            }
        }
    }

    //------------------------------------------------------------------------------
    //! \brief Get the number of entries
    //! \return Number of entries
    Elf_Word get_entries_num() const { return entries_num; }

    //------------------------------------------------------------------------------
    //! \brief Get an entry
    //! \param no Index of the entry
    //! \param flags Flags of the entry
    //! \param version_index Version index of the entry
    //! \param hash Hash of the entry
    //! \param dep_name Dependency name of the entry
    //! \return True if successful, false otherwise
    bool get_entry( Elf_Word     no,
                    Elf_Half&    flags,
                    Elf_Half&    version_index,
                    Elf_Word&    hash,
                    std::string& dep_name ) const
    {
        if ( versym_d_section == nullptr || ( no >= get_entries_num() ) ) {
            return false;
        }

        const_string_section_accessor string_section_acc(
            elf_file.sections[versym_d_section->get_link()] );

        Elfxx_Verdef*  verdef = (Elfxx_Verdef*)versym_d_section->get_data();
        Elfxx_Verdaux* verdaux =
            (Elfxx_Verdaux*)( (char*)verdef + verdef->vd_aux );
        for ( Elf_Word i = 0; i < no; ++i ) {
            verdef  = (Elfxx_Verdef*)( (char*)verdef + verdef->vd_next );
            verdaux = (Elfxx_Verdaux*)( (char*)verdef + verdef->vd_aux );
        }

        // verdef->vd_version should always be 1
        // see https://refspecs.linuxfoundation.org/LSB_3.0.0/LSB-PDA/LSB-PDA.junk/symversion.html#VERDEFENTRIES
        // verdef->vd_cnt should always be 1.
        // see https://maskray.me/blog/2020-11-26-all-about-symbol-versioning

        flags         = verdef->vd_flags;
        version_index = verdef->vd_ndx;
        hash          = verdef->vd_hash;
        dep_name      = string_section_acc.get_string( verdaux->vda_name );

        return true;
    }

    //------------------------------------------------------------------------------
  private:
    const elfio& elf_file;
    S*           versym_d_section =
        nullptr;              //!< Pointer to the version definition section
    Elf_Word entries_num = 0; //!< Number of entries
};

using versym_d_section_accessor = versym_d_section_accessor_template<section>;
using const_versym_d_section_accessor =
    versym_d_section_accessor_template<const section>;

} // namespace ELFIO

#endif // ELFIO_VERSYM_HPP
