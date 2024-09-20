#pragma once

#include "Runtime/Serialize/SerializeTraitsBase.h"

#include "Runtime/Math/Simd/vec-types.h"
#include "Runtime/Math/Simd/vec-trs.h"

template<>
class SerializeTraits<math::float3> : public SerializeTraitsBase<math::float3>
{
public:
    inline static const char* GetTypeString(value_type*)   { return "float3"; }
    inline static bool MightContainPPtr()                  { return false; }
    inline static bool AllowTransferOptimization()         { return false; }

    typedef math::float3    value_type;

    template<class TransferFunction> inline
    static void Transfer(value_type& data, TransferFunction& transfer)
    {
        float* buf = reinterpret_cast<float*>(&data);
        transfer.AddMetaFlag(kTransferUsingFlowMappingStyle);
        transfer.Transfer(buf[0], "x");
        transfer.Transfer(buf[1], "y");
        transfer.Transfer(buf[2], "z");

        // Blob writer must match layout in memory.
        if (transfer.IsBlobWrite() && sizeof(math::float3) == sizeof(math::float4))
            transfer.Transfer(buf[3], "w");
    }
};

template<>
class SerializeTraits<math::float4> : public SerializeTraitsBase<math::float4>
{
public:
    inline static const char* GetTypeString(value_type*)   { return "float4"; }
    inline static bool MightContainPPtr()                  { return false; }
    inline static bool AllowTransferOptimization()         { return true; }

    typedef math::float4    value_type;

    template<class TransferFunction> inline
    static void Transfer(value_type& data, TransferFunction& transfer)
    {
        float* buf = reinterpret_cast<float*>(&data);
        transfer.AddMetaFlag(kTransferUsingFlowMappingStyle);
        transfer.Transfer(buf[0], "x");
        transfer.Transfer(buf[1], "y");
        transfer.Transfer(buf[2], "z");
        transfer.Transfer(buf[3], "w");
    }
};

template<>
class SerializeTraits<math::trsX> : public SerializeTraitsBase<math::trsX>
{
public:
    inline static const char* GetTypeString(value_type*)   { return "xform"; }
    inline static bool MightContainPPtr()                  { return false; }
    inline static bool AllowTransferOptimization()         { return false; }

    typedef math::trsX  value_type;

    template<class TransferFunction> inline
    static void Transfer(value_type& data, TransferFunction& transfer)
    {
        transfer.Transfer(data.t, "t");
        transfer.Transfer(data.q, "q");
        transfer.Transfer(data.s, "s");
    }
};

template<>
class SerializeTraits<math::float2_storage> : public SerializeTraitsBase<math::float2_storage>
{
public:
    inline static const char* GetTypeString(value_type*)   { return CommonString(Vector2f); }
    inline static bool MightContainPPtr()                  { return false; }
    inline static bool AllowTransferOptimization()         { return true; }

    typedef math::float2_storage value_type;

    template<class TransferFunction> inline
    static void Transfer(value_type& data, TransferFunction& transfer)
    {
        float* buf = reinterpret_cast<float*>(&data);
        transfer.AddMetaFlag(kTransferUsingFlowMappingStyle);
        transfer.Transfer(buf[0], "x");
        transfer.Transfer(buf[1], "y");
    }
};

template<>
class SerializeTraits<math::float3_storage> : public SerializeTraitsBase<math::float3_storage>
{
public:
    inline static const char* GetTypeString(value_type*)   { return CommonString(Vector3f); }
    inline static bool MightContainPPtr()                  { return false; }
    inline static bool AllowTransferOptimization()         { return true; }

    typedef math::float3_storage value_type;

    template<class TransferFunction> inline
    static void Transfer(value_type& data, TransferFunction& transfer)
    {
        float* buf = reinterpret_cast<float*>(&data);
        transfer.AddMetaFlag(kTransferUsingFlowMappingStyle);
        transfer.Transfer(buf[0], "x");
        transfer.Transfer(buf[1], "y");
        transfer.Transfer(buf[2], "z");
    }
};

template<>
class SerializeTraits<math::float4_storage> : public SerializeTraitsBase<math::float4_storage>
{
public:
    inline static const char* GetTypeString(value_type*)   { return CommonString(Vector4f); }
    inline static bool MightContainPPtr()                  { return false; }
    inline static bool AllowTransferOptimization()         { return true; }

    typedef math::float4_storage value_type;

    template<class TransferFunction> inline
    static void Transfer(value_type& data, TransferFunction& transfer)
    {
        float* buf = reinterpret_cast<float*>(&data);
        transfer.AddMetaFlag(kTransferUsingFlowMappingStyle);
        transfer.Transfer(buf[0], "x");
        transfer.Transfer(buf[1], "y");
        transfer.Transfer(buf[2], "z");
        transfer.Transfer(buf[3], "w");
    }
};

template<>
class SerializeTraits<math::float4_storage_aligned> : public SerializeTraitsBase<math::float4_storage_aligned>
{
public:
    inline static const char* GetTypeString(value_type*)   { return CommonString(Vector4f); }
    inline static bool MightContainPPtr()                  { return false; }
    inline static bool AllowTransferOptimization()         { return true; }

    typedef math::float4_storage_aligned value_type;

    template<class TransferFunction> inline
    static void Transfer(value_type& data, TransferFunction& transfer)
    {
        float* buf = reinterpret_cast<float*>(&data);
        transfer.AddMetaFlag(kTransferUsingFlowMappingStyle);
        transfer.Transfer(buf[0], "x");
        transfer.Transfer(buf[1], "y");
        transfer.Transfer(buf[2], "z");
        transfer.Transfer(buf[3], "w");
    }
};

template<>
class SerializeTraits<math::int2_storage> : public SerializeTraitsBase<math::int2_storage>
{
public:
    inline static const char* GetTypeString(value_type*) { return "int2_storage"; }
    inline static bool MightContainPPtr() { return false; }
    inline static bool AllowTransferOptimization() { return true; }

    typedef math::int2_storage value_type;

    template<class TransferFunction> inline
    static void Transfer(value_type& data, TransferFunction& transfer)
    {
        int* buf = reinterpret_cast<int*>(&data);
        transfer.AddMetaFlag(kTransferUsingFlowMappingStyle);
        transfer.Transfer(buf[0], "x");
        transfer.Transfer(buf[1], "y");
    }
};

template<>
class SerializeTraits<math::int3_storage> : public SerializeTraitsBase<math::int3_storage>
{
public:
    inline static const char* GetTypeString(value_type*) { return "int3_storage"; }
    inline static bool MightContainPPtr() { return false; }
    inline static bool AllowTransferOptimization() { return true; }

    typedef math::int3_storage value_type;

    template<class TransferFunction> inline
    static void Transfer(value_type& data, TransferFunction& transfer)
    {
        int* buf = reinterpret_cast<int*>(&data);
        transfer.AddMetaFlag(kTransferUsingFlowMappingStyle);
        transfer.Transfer(buf[0], "x");
        transfer.Transfer(buf[1], "y");
        transfer.Transfer(buf[2], "z");
    }
};

template<>
class SerializeTraits<math::int4_storage> : public SerializeTraitsBase<math::int4_storage>
{
public:
    inline static const char* GetTypeString(value_type*) { return "int4_storage"; }
    inline static bool MightContainPPtr() { return false; }
    inline static bool AllowTransferOptimization() { return true; }

    typedef math::int4_storage value_type;

    template<class TransferFunction> inline
    static void Transfer(value_type& data, TransferFunction& transfer)
    {
        int* buf = reinterpret_cast<int*>(&data);
        transfer.AddMetaFlag(kTransferUsingFlowMappingStyle);
        transfer.Transfer(buf[0], "x");
        transfer.Transfer(buf[1], "y");
        transfer.Transfer(buf[2], "z");
        transfer.Transfer(buf[3], "w");
    }
};
