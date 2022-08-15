///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// RDAT_Macros.inl                                                           //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Defines macros use to define types Dxil Library Runtime Data (RDAT).      //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

// Pay attention to alignment when organizing structures.
// RDAT_STRING and *_REF types are always 4 bytes.
// RDAT_BYTES is 2 * 4 bytes.

// These are the modes set to DEF_RDAT_TYPES and/or DEF_RDAT_ENUMS that drive
// macro expansion for types and enums which define the necessary declarations
// and code.
// While some of associated macros sets are not defined in this file, these
// definitions allow custom paths in the type definition files in certain cases.
#define DEF_RDAT_DEFAULTS 1             // DEF_RDAT_TYPES and DEF_RDAT_ENUMS - define empty macros for anything not already defined
#define DEF_RDAT_TYPES_BASIC_STRUCT 2   // DEF_RDAT_TYPES - define structs with basic types, matching RDAT format
#define DEF_RDAT_TYPES_USE_HELPERS 3    // DEF_RDAT_TYPES - define structs using helpers, matching RDAT format
#define DEF_RDAT_DUMP_DECL 4            // DEF_RDAT_TYPES and DEF_RDAT_ENUMS - write dump declarations
#define DEF_RDAT_DUMP_IMPL 5            // DEF_RDAT_TYPES and DEF_RDAT_ENUMS - write dump implementation
#define DEF_RDAT_TYPES_USE_POINTERS 6   // DEF_RDAT_TYPES - define deserialized version using pointers instead of offsets
#define DEF_RDAT_ENUM_CLASS 7           // DEF_RDAT_ENUMS - declare enums with enum class
#define DEF_RDAT_TRAITS 8               // DEF_RDAT_TYPES - define type traits
#define DEF_RDAT_TYPES_FORWARD_DECL 9   // DEF_RDAT_TYPES - forward declare type struct/class
#define DEF_RDAT_READER_DECL 10         // DEF_RDAT_TYPES and DEF_RDAT_ENUMS - write reader classes
#define DEF_RDAT_READER_IMPL 11         // DEF_RDAT_TYPES and DEF_RDAT_ENUMS - write reader classes
#define DEF_RDAT_STRUCT_VALIDATION 13   // DEF_RDAT_TYPES and DEF_RDAT_ENUMS - define structural validation
// PRERELEASE-TODO: deeper validation for DxilValidation (limiting enum values and other such things)

#define GLUE2(a, b) a##b
#define GLUE(a, b) GLUE2(a, b)

#ifdef DEF_RDAT_TYPES

#if DEF_RDAT_TYPES == DEF_RDAT_TYPES_FORWARD_DECL

  #define RDAT_STRUCT(type) struct type;  class type##_Reader;
  #define RDAT_STRUCT_DERIVED(type, base)               RDAT_STRUCT(type)
  #define RDAT_WRAP_ARRAY(type, count, type_name)       struct type_name { type arr[count]; };

#elif DEF_RDAT_TYPES == DEF_RDAT_TYPES_BASIC_STRUCT

  #define RDAT_STRUCT(type)                   struct type {
  #define RDAT_STRUCT_DERIVED(type, base)     \
    struct type : public base {
  #define RDAT_STRUCT_END()                   };
  #define RDAT_UNION()                        union {
  #define RDAT_UNION_END()                    };
  #define RDAT_RECORD_REF(type, name)         uint32_t name;
  #define RDAT_RECORD_ARRAY_REF(type, name)   uint32_t name;
  #define RDAT_RECORD_VALUE(type, name)       type name;
  #define RDAT_STRING(name)                   uint32_t name;
  #define RDAT_STRING_ARRAY_REF(name)         uint32_t name;
  #define RDAT_VALUE(type, name)              type name;
  #define RDAT_INDEX_ARRAY_REF(name)          uint32_t name;
  #define RDAT_ENUM(sTy, eTy, name)           sTy name;
  #define RDAT_FLAGS(sTy, eTy, name)          sTy name;
  #define RDAT_BYTES(name)                    uint32_t name; uint32_t name##_Size; name;
  #define RDAT_ARRAY_VALUE(type, count, type_name, name) type_name name;

#elif DEF_RDAT_TYPES == DEF_RDAT_TYPES_USE_HELPERS

  #define RDAT_STRUCT(type)                   struct type {
  #define RDAT_STRUCT_DERIVED(type, base)     struct type : public base {
  #define RDAT_STRUCT_END()                   };
  #define RDAT_UNION()                        union {
  #define RDAT_UNION_END()                    };
  #define RDAT_RECORD_REF(type, name)         RecordRef<type> name;
  #define RDAT_RECORD_ARRAY_REF(type, name)   RecordArrayRef<type> name;
  #define RDAT_RECORD_VALUE(type, name)       type name;
  #define RDAT_STRING(name)                   RDATString name;
  #define RDAT_STRING_ARRAY_REF(name)         RDATStringArray name;
  #define RDAT_VALUE(type, name)              type name;
  #define RDAT_INDEX_ARRAY_REF(name)          IndexArrayRef name;
  #define RDAT_ENUM(sTy, eTy, name)           sTy name;
  #define RDAT_FLAGS(sTy, eTy, name)          sTy name;
  #define RDAT_BYTES(name)                    hlsl::RDAT::BytesRef name;
  #define RDAT_ARRAY_VALUE(type, count, type_name, name) type_name name;

#elif DEF_RDAT_TYPES == DEF_RDAT_READER_DECL

  #define RDAT_STRUCT(type)               \
    class type##_Reader : public RecordReader<type##_Reader> { \
    public: \
      typedef type RecordType; \
      type##_Reader(const BaseRecordReader &reader); \
      type##_Reader() : RecordReader<type##_Reader>() {} \
      const RecordType *asRecord() const; \
      const RecordType *operator->() const { return asRecord(); }
  #define RDAT_STRUCT_DERIVED(type, base) \
    class type##_Reader : public base##_Reader { \
    public: \
      typedef type RecordType; \
      type##_Reader(const BaseRecordReader &reader); \
      type##_Reader(); \
      const RecordType *asRecord() const; \
      const RecordType *operator->() const { return asRecord(); }
  #define RDAT_STRUCT_END() };
  #define RDAT_UNION_IF(name, expr)           bool has##name() const;
  #define RDAT_UNION_ELIF(name, expr)         RDAT_UNION_IF(name, expr)
  #define RDAT_RECORD_REF(type, name)         type##_Reader get##name() const;
  #define RDAT_RECORD_ARRAY_REF(type, name)   RecordArrayReader<type##_Reader> get##name() const;
  #define RDAT_RECORD_VALUE(type, name)       type##_Reader get##name() const;
  #define RDAT_STRING(name)                   const char *get##name() const;
  #define RDAT_STRING_ARRAY_REF(name)         StringArrayReader get##name() const;
  #define RDAT_VALUE(type, name)              type get##name() const;
  #define RDAT_INDEX_ARRAY_REF(name)          IndexTableReader::IndexRow get##name() const;
  #define RDAT_ENUM(sTy, eTy, name)           eTy get##name() const;
  #define RDAT_FLAGS(sTy, eTy, name)          sTy get##name() const;
  #define RDAT_BYTES(name)                    const void *get##name() const; \
                                              uint32_t size##name() const;
  #define RDAT_ARRAY_VALUE(type, count, type_name, name) type_name get##name() const;

#elif DEF_RDAT_TYPES == DEF_RDAT_READER_IMPL

  #define RDAT_STRUCT(type)               \
      type##_Reader::type##_Reader(const BaseRecordReader &reader) : RecordReader<type##_Reader>(reader) {} \
      const type *type##_Reader::asRecord() const { return BaseRecordReader::asRecord<type>(); }
  #define RDAT_STRUCT_DERIVED(type, base) \
      type##_Reader::type##_Reader(const BaseRecordReader &reader) : base##_Reader(reader) { \
        if ((m_pContext || m_pRecord) && m_Size < sizeof(type)) \
          InvalidateReader(); \
      } \
      type##_Reader::type##_Reader() : base##_Reader() {} \
      const type *type##_Reader::asRecord() const { return BaseRecordReader::asRecord<type>(); }
  #define RDAT_STRUCT_TABLE(type, table)                RDAT_STRUCT(type)
  #define RDAT_STRUCT_TABLE_DERIVED(type, base, table)  RDAT_STRUCT_DERIVED(type, base)
  #define RDAT_UNION_IF(name, expr)     bool GLUE(RECORD_TYPE,_Reader)::has##name() const  { if (auto *pRecord = asRecord()) return !!(expr); return false; }
  #define RDAT_UNION_ELIF(name, expr)   RDAT_UNION_IF(name, expr)
  #define RDAT_RECORD_REF(type, name)   type##_Reader GLUE(RECORD_TYPE,_Reader)::get##name() const     { return GetField_RecordRef<type##_Reader>      (&(asRecord()->name)); }
  #define RDAT_RECORD_ARRAY_REF(type, name) \
                        RecordArrayReader<type##_Reader> GLUE(RECORD_TYPE,_Reader)::get##name() const  { return GetField_RecordArrayRef<type##_Reader> (&(asRecord()->name)); }
  #define RDAT_RECORD_VALUE(type, name) type##_Reader GLUE(RECORD_TYPE,_Reader)::get##name() const     { return GetField_RecordValue<type##_Reader>    (&(asRecord()->name)); }
  #define RDAT_STRING(name)             const char *GLUE(RECORD_TYPE,_Reader)::get##name() const       { return GetField_String                        (&(asRecord()->name)); }
  #define RDAT_STRING_ARRAY_REF(name)   StringArrayReader GLUE(RECORD_TYPE,_Reader)::get##name() const { return GetField_StringArray                   (&(asRecord()->name)); }
  #define RDAT_VALUE(type, name)        type GLUE(RECORD_TYPE,_Reader)::get##name() const              { return GetField_Value<type, type>             (&(asRecord()->name)); }
  #define RDAT_INDEX_ARRAY_REF(name)    IndexTableReader::IndexRow GLUE(RECORD_TYPE,_Reader)::get##name() const { return GetField_IndexArray           (&(asRecord()->name)); }
  #define RDAT_ENUM(sTy, eTy, name)     eTy GLUE(RECORD_TYPE,_Reader)::get##name() const               { return GetField_Value<eTy, sTy>               (&(asRecord()->name)); }
  #define RDAT_FLAGS(sTy, eTy, name)    sTy GLUE(RECORD_TYPE,_Reader)::get##name() const               { return GetField_Value<sTy, sTy>               (&(asRecord()->name)); }
  #define RDAT_BYTES(name)              const void *GLUE(RECORD_TYPE,_Reader)::get##name() const       { return GetField_Bytes                         (&(asRecord()->name)); } \
                                        uint32_t GLUE(RECORD_TYPE,_Reader)::size##name() const         { return GetField_BytesSize                     (&(asRecord()->name)); }
  #define RDAT_ARRAY_VALUE(type, count, type_name, name) \
                                        type_name GLUE(RECORD_TYPE,_Reader)::get##name() const         { return GetField_Value<type_name, type_name>   (&(asRecord()->name)); }

#elif DEF_RDAT_TYPES == DEF_RDAT_STRUCT_VALIDATION

  #define RDAT_STRUCT(type) \
    template<> bool ValidateRecord<type>(const RDATContext &ctx, const type *pRecord) { \
      type##_Reader reader(BaseRecordReader(&ctx, (void*)pRecord, (uint32_t)RecordTraits<type>::RecordSize()));
  #define RDAT_STRUCT_DERIVED(type, base) RDAT_STRUCT(type)
  #define RDAT_STRUCT_END()                   return true; }
  #define RDAT_UNION_IF(name, expr)           if (reader.has##name()) {
  #define RDAT_UNION_ELIF(name, expr)         } else if (reader.has##name()) {
  #define RDAT_UNION_ENDIF()                  }
  #define RDAT_RECORD_REF(type, name)         if (!ValidateRecordRef<type>(ctx, pRecord->name)) return false;
  #define RDAT_RECORD_ARRAY_REF(type, name)   if (!ValidateRecordArrayRef<type>(ctx, pRecord->name)) return false;
  #define RDAT_RECORD_VALUE(type, name)       if (!ValidateRecord<type>(ctx, &pRecord->name)) return false;
  #define RDAT_STRING(name)                   if (!ValidateStringRef(ctx, pRecord->name)) return false;
  #define RDAT_STRING_ARRAY_REF(name)         if (!ValidateStringArrayRef(ctx, pRecord->name)) return false;
  #define RDAT_INDEX_ARRAY_REF(name)          if (!ValidateIndexArrayRef(ctx, pRecord->name)) return false;

#elif DEF_RDAT_TYPES == DEF_RDAT_DUMP_IMPL

  #define RDAT_STRUCT(type)                                                      \
    template <>                                                                  \
    void RecordDumper<hlsl::RDAT::type>::Dump(                                   \
        const hlsl::RDAT::RDATContext &ctx, DumpContext &d) const {              \
      d.Indent();                                                                \
      const hlsl::RDAT::type *pRecord = this;                                    \
      type##_Reader reader(BaseRecordReader(                                     \
          &ctx, (void *)pRecord, (uint32_t)RecordTraits<type>::RecordSize()));
  #define RDAT_STRUCT_DERIVED(type, base)                                        \
    const char *RecordRefDumper<hlsl::RDAT::base>::TypeNameDerived(              \
        const hlsl::RDAT::RDATContext &ctx) const {                              \
      return TypeName<hlsl::RDAT::type>(ctx);                                    \
    }                                                                            \
    void RecordRefDumper<hlsl::RDAT::base>::DumpDerived(                         \
        const hlsl::RDAT::RDATContext &ctx, DumpContext &d) const {              \
      Dump<hlsl::RDAT::type>(ctx, d);                                            \
    }                                                                            \
    template <>                                                                  \
    void DumpWithBase<hlsl::RDAT::type>(const hlsl::RDAT::RDATContext &ctx,      \
                                        DumpContext &d,                          \
                                        const hlsl::RDAT::type *pRecord) {       \
      DumpWithBase<hlsl::RDAT::base>(ctx, d, pRecord);                           \
      static_cast<const RecordDumper<hlsl::RDAT::type> *>(pRecord)->Dump(ctx,    \
                                                                         d);     \
    }                                                                            \
    RDAT_STRUCT(type)
  #define RDAT_STRUCT_END()                   d.Dedent(); }
  #define RDAT_UNION_IF(name, expr)           if (reader.has##name()) {
  #define RDAT_UNION_ELIF(name, expr)         } else if (reader.has##name()) {
  #define RDAT_UNION_ENDIF()                  }
  #define RDAT_RECORD_REF(type, name)         DumpRecordRef(ctx, d, #type, #name, name);
  #define RDAT_RECORD_ARRAY_REF(type, name)   DumpRecordArrayRef(ctx, d, #type, #name, name);
  #define RDAT_RECORD_VALUE(type, name)       DumpRecordValue(ctx, d, #type, #name, &name);
  #define RDAT_STRING(name)                   d.WriteLn(#name ": ", QuotedStringValue(name.Get(ctx)));
  #define RDAT_STRING_ARRAY_REF(name)         DumpStringArray(ctx, d, #name, name);
  #define RDAT_VALUE(type, name)              d.WriteLn(#name ": ", name);
  #define RDAT_INDEX_ARRAY_REF(name)          DumpIndexArray(ctx, d, #name, name);
  #define RDAT_ENUM(sTy, eTy, name)           d.DumpEnum<eTy>(#name, (eTy)name);
  #define RDAT_FLAGS(sTy, eTy, name)          d.DumpFlags<eTy, sTy>(#name, name);
  #define RDAT_BYTES(name)                    DumpBytesRef(ctx, d, #name, name);
  #define RDAT_ARRAY_VALUE(type, count, type_name, name) DumpValueArray<type>(d, #name, #type, &name, count);

#elif DEF_RDAT_TYPES == DEF_RDAT_TRAITS

  #define RDAT_STRUCT(type) \
    template<> constexpr const char *RecordTraits<type>::TypeName() { return #type; }
  #define RDAT_STRUCT_DERIVED(type, base) RDAT_STRUCT(type)
  #define RDAT_STRUCT_TABLE(type, table) \
    RDAT_STRUCT(type) \
    template<> constexpr RecordTableIndex RecordTraits<type>::TableIndex() { return RecordTableIndex::table; }
  #define RDAT_STRUCT_TABLE_DERIVED(type, base, table) \
    RDAT_STRUCT_DERIVED(type, base) \
    template<> constexpr RecordTableIndex RecordTraits<type>::TableIndex() { return RecordTableIndex::table; }

#endif // DEF_RDAT_TYPES cases

// Define any undefined macros to defaults
#ifndef RDAT_STRUCT
  #define RDAT_STRUCT(type)
#endif
#ifndef RDAT_STRUCT_DERIVED
  #define RDAT_STRUCT_DERIVED(type, base)
#endif
#ifndef RDAT_STRUCT_TABLE
  #define RDAT_STRUCT_TABLE(type, table) RDAT_STRUCT(type)
#endif
#ifndef RDAT_STRUCT_TABLE_DERIVED
  #define RDAT_STRUCT_TABLE_DERIVED(type, base, table) RDAT_STRUCT_DERIVED(type, base)
#endif
#ifndef RDAT_STRUCT_END
  #define RDAT_STRUCT_END()
#endif
#ifndef RDAT_UNION
  #define RDAT_UNION()
#endif
#ifndef RDAT_UNION_IF
  #define RDAT_UNION_IF(name, expr)   // In expr: 'this' is reader; pRecord is record struct
#endif
#ifndef RDAT_UNION_ELIF
  #define RDAT_UNION_ELIF(name, expr) // In expr: 'this' is reader; pRecord is record struct
#endif
#ifndef RDAT_UNION_ENDIF
  #define RDAT_UNION_ENDIF()
#endif
#ifndef RDAT_UNION_END
  #define RDAT_UNION_END()
#endif
#ifndef RDAT_RECORD_REF
  #define RDAT_RECORD_REF(type, name)       // always use base record type in RDAT_RECORD_REF
#endif
#ifndef RDAT_RECORD_ARRAY_REF
  #define RDAT_RECORD_ARRAY_REF(type, name) // always use base record type in RDAT_RECORD_ARRAY_REF
#endif
#ifndef RDAT_RECORD_VALUE
  #define RDAT_RECORD_VALUE(type, name)
#endif
#ifndef RDAT_STRING
  #define RDAT_STRING(name)
#endif
#ifndef RDAT_STRING_ARRAY_REF
  #define RDAT_STRING_ARRAY_REF(name)
#endif
#ifndef RDAT_VALUE
  #define RDAT_VALUE(type, name)
#endif
#ifndef RDAT_INDEX_ARRAY_REF
  #define RDAT_INDEX_ARRAY_REF(name)  // ref to array of uint32_t values
#endif
#ifndef RDAT_ENUM
  #define RDAT_ENUM(sTy, eTy, name)
#endif
#ifndef RDAT_FLAGS
  #define RDAT_FLAGS(sTy, eTy, name)
#endif
#ifndef RDAT_BYTES
  #define RDAT_BYTES(name)
#endif
#ifndef RDAT_WRAP_ARRAY
  #define RDAT_WRAP_ARRAY(type, count, type_name)         // define struct-wrapped array type here
#endif
#ifndef RDAT_ARRAY_VALUE
  #define RDAT_ARRAY_VALUE(type, count, type_name, name)  // define struct-wrapped array member
#endif

#endif // DEF_RDAT_TYPES defined


#if defined(DEF_RDAT_ENUMS) || defined(DEF_DXIL_ENUMS)

#if DEF_RDAT_ENUMS == DEF_RDAT_ENUM_CLASS

  #define RDAT_ENUM_START(eTy, sTy) enum class eTy : sTy {
  // No RDAT_DXIL_ENUM_START, DXIL enums are defined elsewhere
  #define RDAT_ENUM_VALUE(name, value) name = value,
  #define RDAT_ENUM_VALUE_ALIAS(name, value) name = value,
  #define RDAT_ENUM_VALUE_NODEF(name) name,
  #define RDAT_ENUM_END() };

#elif DEF_RDAT_ENUMS == DEF_RDAT_DUMP_DECL

  #define RDAT_ENUM_START(eTy, sTy) const char *ToString(eTy e);
  #define RDAT_DXIL_ENUM_START(eTy, sTy) const char *ToString(eTy e);

#elif DEF_RDAT_ENUMS == DEF_RDAT_DUMP_IMPL

  //#define RDAT_ENUM_START(eTy, sTy) \
  //  const char *ToString(eTy e) { \
  //    switch((sTy)e) {
  //#define RDAT_ENUM_VALUE(name, value) case value: return #name;
  #define RDAT_ENUM_START(eTy, sTy) \
    const char *ToString(eTy e) { \
      typedef eTy thisEnumTy; \
      switch(e) {
  #define RDAT_DXIL_ENUM_START(eTy, sTy) \
    const char *ToString(eTy e) { \
      typedef eTy thisEnumTy; \
      switch(e) {
  #define RDAT_ENUM_VALUE_NODEF(name) case thisEnumTy::name: return #name;
  #define RDAT_ENUM_VALUE(name, value) RDAT_ENUM_VALUE_NODEF(name)
  #define RDAT_ENUM_END() \
      default: return nullptr; \
      } \
    }

#endif // DEF_RDAT_ENUMS cases

// Define any undefined macros to defaults
#ifndef RDAT_ENUM_START
  #define RDAT_ENUM_START(eTy, sTy)
#endif
#ifndef RDAT_DXIL_ENUM_START
  #define RDAT_DXIL_ENUM_START(eTy, sTy)
#endif
#ifndef RDAT_ENUM_VALUE
  #define RDAT_ENUM_VALUE(name, value)        // value only used during declaration
#endif
#ifndef RDAT_ENUM_VALUE_ALIAS
  #define RDAT_ENUM_VALUE_ALIAS(name, value)  // secondary enum names that alias to the same value as another name in the enum
#endif
#ifndef RDAT_ENUM_VALUE_NODEF
  #define RDAT_ENUM_VALUE_NODEF(name)         // enum names that have no explicitly defined value, or are defined elsewhere
#endif
#ifndef RDAT_ENUM_END
  #define RDAT_ENUM_END()
#endif

#endif // DEF_RDAT_ENUMS or DEF_DXIL_ENUMS defined

#include "RDAT_SubobjectTypes.inl"
#include "RDAT_LibraryTypes.inl"

#undef DEF_RDAT_TYPES
#undef DEF_RDAT_ENUMS
#undef DEF_DXIL_ENUMS

#undef RDAT_STRUCT
#undef RDAT_STRUCT_DERIVED
#undef RDAT_STRUCT_TABLE
#undef RDAT_STRUCT_TABLE_DERIVED
#undef RDAT_STRUCT_END
#undef RDAT_UNION
#undef RDAT_UNION_IF
#undef RDAT_UNION_ELIF
#undef RDAT_UNION_ENDIF
#undef RDAT_UNION_END
#undef RDAT_RECORD_REF
#undef RDAT_RECORD_ARRAY_REF
#undef RDAT_RECORD_VALUE
#undef RDAT_STRING
#undef RDAT_STRING_ARRAY_REF
#undef RDAT_VALUE
#undef RDAT_INDEX_ARRAY_REF
#undef RDAT_ENUM
#undef RDAT_FLAGS
#undef RDAT_BYTES
#undef RDAT_WRAP_ARRAY
#undef RDAT_ARRAY_VALUE

#undef RDAT_ENUM_START
#undef RDAT_DXIL_ENUM_START
#undef RDAT_ENUM_VALUE
#undef RDAT_ENUM_VALUE_ALIAS
#undef RDAT_ENUM_VALUE_NODEF
#undef RDAT_ENUM_END
