using System;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;

namespace Godot.SourceGenerators
{
    internal static class MarshalUtils
    {
        public class TypeCache
        {
            public INamedTypeSymbol GodotObjectType { get; }

            public TypeCache(Compilation compilation)
            {
                INamedTypeSymbol GetTypeByMetadataNameOrThrow(string fullyQualifiedMetadataName)
                {
                    return compilation.GetTypeByMetadataName(fullyQualifiedMetadataName) ??
                           throw new InvalidOperationException($"Type not found: '{fullyQualifiedMetadataName}'.");
                }

                GodotObjectType = GetTypeByMetadataNameOrThrow("Godot.Object");
            }
        }

        public static VariantType? ConvertMarshalTypeToVariantType(MarshalType marshalType)
            => marshalType switch
            {
                MarshalType.Boolean => VariantType.Bool,
                MarshalType.Char => VariantType.Int,
                MarshalType.SByte => VariantType.Int,
                MarshalType.Int16 => VariantType.Int,
                MarshalType.Int32 => VariantType.Int,
                MarshalType.Int64 => VariantType.Int,
                MarshalType.Byte => VariantType.Int,
                MarshalType.UInt16 => VariantType.Int,
                MarshalType.UInt32 => VariantType.Int,
                MarshalType.UInt64 => VariantType.Int,
                MarshalType.Single => VariantType.Float,
                MarshalType.Double => VariantType.Float,
                MarshalType.String => VariantType.String,
                MarshalType.Vector2 => VariantType.Vector2,
                MarshalType.Vector2i => VariantType.Vector2i,
                MarshalType.Rect2 => VariantType.Rect2,
                MarshalType.Rect2i => VariantType.Rect2i,
                MarshalType.Transform2D => VariantType.Transform2d,
                MarshalType.Vector3 => VariantType.Vector3,
                MarshalType.Vector3i => VariantType.Vector3i,
                MarshalType.Basis => VariantType.Basis,
                MarshalType.Quaternion => VariantType.Quaternion,
                MarshalType.Transform3D => VariantType.Transform3d,
                MarshalType.Vector4 => VariantType.Vector4,
                MarshalType.Vector4i => VariantType.Vector4i,
                MarshalType.Projection => VariantType.Projection,
                MarshalType.AABB => VariantType.Aabb,
                MarshalType.Color => VariantType.Color,
                MarshalType.Plane => VariantType.Plane,
                MarshalType.Callable => VariantType.Callable,
                MarshalType.SignalInfo => VariantType.Signal,
                MarshalType.Enum => VariantType.Int,
                MarshalType.ByteArray => VariantType.PackedByteArray,
                MarshalType.Int32Array => VariantType.PackedInt32Array,
                MarshalType.Int64Array => VariantType.PackedInt64Array,
                MarshalType.Float32Array => VariantType.PackedFloat32Array,
                MarshalType.Float64Array => VariantType.PackedFloat64Array,
                MarshalType.StringArray => VariantType.PackedStringArray,
                MarshalType.Vector2Array => VariantType.PackedVector2Array,
                MarshalType.Vector3Array => VariantType.PackedVector3Array,
                MarshalType.ColorArray => VariantType.PackedColorArray,
                MarshalType.GodotObjectOrDerivedArray => VariantType.Array,
                MarshalType.SystemArrayOfStringName => VariantType.Array,
                MarshalType.SystemArrayOfNodePath => VariantType.Array,
                MarshalType.SystemArrayOfRID => VariantType.Array,
                MarshalType.Variant => VariantType.Nil,
                MarshalType.GodotObjectOrDerived => VariantType.Object,
                MarshalType.StringName => VariantType.StringName,
                MarshalType.NodePath => VariantType.NodePath,
                MarshalType.RID => VariantType.Rid,
                MarshalType.GodotDictionary => VariantType.Dictionary,
                MarshalType.GodotArray => VariantType.Array,
                MarshalType.GodotGenericDictionary => VariantType.Dictionary,
                MarshalType.GodotGenericArray => VariantType.Array,
                _ => null
            };

        public static MarshalType? ConvertManagedTypeToMarshalType(ITypeSymbol type, TypeCache typeCache)
        {
            var specialType = type.SpecialType;

            switch (specialType)
            {
                case SpecialType.System_Boolean:
                    return MarshalType.Boolean;
                case SpecialType.System_Char:
                    return MarshalType.Char;
                case SpecialType.System_SByte:
                    return MarshalType.SByte;
                case SpecialType.System_Int16:
                    return MarshalType.Int16;
                case SpecialType.System_Int32:
                    return MarshalType.Int32;
                case SpecialType.System_Int64:
                    return MarshalType.Int64;
                case SpecialType.System_Byte:
                    return MarshalType.Byte;
                case SpecialType.System_UInt16:
                    return MarshalType.UInt16;
                case SpecialType.System_UInt32:
                    return MarshalType.UInt32;
                case SpecialType.System_UInt64:
                    return MarshalType.UInt64;
                case SpecialType.System_Single:
                    return MarshalType.Single;
                case SpecialType.System_Double:
                    return MarshalType.Double;
                case SpecialType.System_String:
                    return MarshalType.String;
                default:
                {
                    var typeKind = type.TypeKind;

                    if (typeKind == TypeKind.Enum)
                        return MarshalType.Enum;

                    if (typeKind == TypeKind.Struct)
                    {
                        if (type.ContainingAssembly?.Name == "GodotSharp" &&
                            type.ContainingNamespace?.Name == "Godot")
                        {
                            return type switch
                            {
                                { Name: "Vector2" } => MarshalType.Vector2,
                                { Name: "Vector2i" } => MarshalType.Vector2i,
                                { Name: "Rect2" } => MarshalType.Rect2,
                                { Name: "Rect2i" } => MarshalType.Rect2i,
                                { Name: "Transform2D" } => MarshalType.Transform2D,
                                { Name: "Vector3" } => MarshalType.Vector3,
                                { Name: "Vector3i" } => MarshalType.Vector3i,
                                { Name: "Basis" } => MarshalType.Basis,
                                { Name: "Quaternion" } => MarshalType.Quaternion,
                                { Name: "Transform3D" } => MarshalType.Transform3D,
                                { Name: "Vector4" } => MarshalType.Vector4,
                                { Name: "Vector4i" } => MarshalType.Vector4i,
                                { Name: "Projection" } => MarshalType.Projection,
                                { Name: "AABB" } => MarshalType.AABB,
                                { Name: "Color" } => MarshalType.Color,
                                { Name: "Plane" } => MarshalType.Plane,
                                { Name: "RID" } => MarshalType.RID,
                                { Name: "Callable" } => MarshalType.Callable,
                                { Name: "SignalInfo" } => MarshalType.SignalInfo,
                                { Name: "Variant" } => MarshalType.Variant,
                                _ => null
                            };
                        }
                    }
                    else if (typeKind == TypeKind.Array)
                    {
                        var arrayType = (IArrayTypeSymbol)type;

                        if (arrayType.Rank != 1)
                            return null;

                        var elementType = arrayType.ElementType;

                        switch (elementType.SpecialType)
                        {
                            case SpecialType.System_Byte:
                                return MarshalType.ByteArray;
                            case SpecialType.System_Int32:
                                return MarshalType.Int32Array;
                            case SpecialType.System_Int64:
                                return MarshalType.Int64Array;
                            case SpecialType.System_Single:
                                return MarshalType.Float32Array;
                            case SpecialType.System_Double:
                                return MarshalType.Float64Array;
                            case SpecialType.System_String:
                                return MarshalType.StringArray;
                        }

                        if (elementType.SimpleDerivesFrom(typeCache.GodotObjectType))
                            return MarshalType.GodotObjectOrDerivedArray;

                        if (elementType.ContainingAssembly?.Name == "GodotSharp" &&
                            elementType.ContainingNamespace?.Name == "Godot")
                        {
                            switch (elementType)
                            {
                                case { Name: "Vector2" }:
                                    return MarshalType.Vector2Array;
                                case { Name: "Vector3" }:
                                    return MarshalType.Vector3Array;
                                case { Name: "Color" }:
                                    return MarshalType.ColorArray;
                                case { Name: "StringName" }:
                                    return MarshalType.SystemArrayOfStringName;
                                case { Name: "NodePath" }:
                                    return MarshalType.SystemArrayOfNodePath;
                                case { Name: "RID" }:
                                    return MarshalType.SystemArrayOfRID;
                            }
                        }

                        return null;
                    }
                    else
                    {
                        if (type.SimpleDerivesFrom(typeCache.GodotObjectType))
                            return MarshalType.GodotObjectOrDerived;

                        if (type.ContainingAssembly?.Name == "GodotSharp")
                        {
                            switch (type.ContainingNamespace?.Name)
                            {
                                case "Godot":
                                    return type switch
                                    {
                                        { Name: "StringName" } => MarshalType.StringName,
                                        { Name: "NodePath" } => MarshalType.NodePath,
                                        _ => null
                                    };
                                case "Collections"
                                    when type.ContainingNamespace?.FullQualifiedNameOmitGlobal() == "Godot.Collections":
                                    return type switch
                                    {
                                        { Name: "Dictionary" } =>
                                            type is INamedTypeSymbol { IsGenericType: false } ?
                                                MarshalType.GodotDictionary :
                                                MarshalType.GodotGenericDictionary,
                                        { Name: "Array" } =>
                                            type is INamedTypeSymbol { IsGenericType: false } ?
                                                MarshalType.GodotArray :
                                                MarshalType.GodotGenericArray,
                                        _ => null
                                    };
                            }
                        }
                    }

                    break;
                }
            }

            return null;
        }

        private static bool SimpleDerivesFrom(this ITypeSymbol? type, ITypeSymbol candidateBaseType)
        {
            while (type != null)
            {
                if (SymbolEqualityComparer.Default.Equals(type, candidateBaseType))
                    return true;

                type = type.BaseType;
            }

            return false;
        }

        public static ITypeSymbol? GetArrayElementType(ITypeSymbol typeSymbol)
        {
            if (typeSymbol.TypeKind == TypeKind.Array)
            {
                var arrayType = (IArrayTypeSymbol)typeSymbol;
                return arrayType.ElementType;
            }

            if (typeSymbol is INamedTypeSymbol { IsGenericType: true } genericType)
                return genericType.TypeArguments.FirstOrDefault();

            return null;
        }

        private static StringBuilder Append(this StringBuilder source, string a, string b)
            => source.Append(a).Append(b);

        private static StringBuilder Append(this StringBuilder source, string a, string b, string c)
            => source.Append(a).Append(b).Append(c);

        private static StringBuilder Append(this StringBuilder source, string a, string b,
            string c, string d)
            => source.Append(a).Append(b).Append(c).Append(d);

        private static StringBuilder Append(this StringBuilder source, string a, string b,
            string c, string d, string e)
            => source.Append(a).Append(b).Append(c).Append(d).Append(e);

        private static StringBuilder Append(this StringBuilder source, string a, string b,
            string c, string d, string e, string f)
            => source.Append(a).Append(b).Append(c).Append(d).Append(e).Append(f);

        private static StringBuilder Append(this StringBuilder source, string a, string b,
            string c, string d, string e, string f, string g)
            => source.Append(a).Append(b).Append(c).Append(d).Append(e).Append(f).Append(g);

        private static StringBuilder Append(this StringBuilder source, string a, string b,
            string c, string d, string e, string f, string g, string h)
            => source.Append(a).Append(b).Append(c).Append(d).Append(e).Append(f).Append(g).Append(h);

        private const string VariantUtils = "global::Godot.NativeInterop.VariantUtils";

        public static StringBuilder AppendNativeVariantToManagedExpr(this StringBuilder source,
            string inputExpr, ITypeSymbol typeSymbol, MarshalType marshalType)
        {
            return marshalType switch
            {
                MarshalType.Boolean =>
                    source.Append(VariantUtils, ".ConvertToBool(", inputExpr, ")"),
                MarshalType.Char =>
                    source.Append("(char)", VariantUtils, ".ConvertToUInt16(", inputExpr, ")"),
                MarshalType.SByte =>
                    source.Append(VariantUtils, ".ConvertToInt8(", inputExpr, ")"),
                MarshalType.Int16 =>
                    source.Append(VariantUtils, ".ConvertToInt16(", inputExpr, ")"),
                MarshalType.Int32 =>
                    source.Append(VariantUtils, ".ConvertToInt32(", inputExpr, ")"),
                MarshalType.Int64 =>
                    source.Append(VariantUtils, ".ConvertToInt64(", inputExpr, ")"),
                MarshalType.Byte =>
                    source.Append(VariantUtils, ".ConvertToUInt8(", inputExpr, ")"),
                MarshalType.UInt16 =>
                    source.Append(VariantUtils, ".ConvertToUInt16(", inputExpr, ")"),
                MarshalType.UInt32 =>
                    source.Append(VariantUtils, ".ConvertToUInt32(", inputExpr, ")"),
                MarshalType.UInt64 =>
                    source.Append(VariantUtils, ".ConvertToUInt64(", inputExpr, ")"),
                MarshalType.Single =>
                    source.Append(VariantUtils, ".ConvertToFloat32(", inputExpr, ")"),
                MarshalType.Double =>
                    source.Append(VariantUtils, ".ConvertToFloat64(", inputExpr, ")"),
                MarshalType.String =>
                    source.Append(VariantUtils, ".ConvertToStringObject(", inputExpr, ")"),
                MarshalType.Vector2 =>
                    source.Append(VariantUtils, ".ConvertToVector2(", inputExpr, ")"),
                MarshalType.Vector2i =>
                    source.Append(VariantUtils, ".ConvertToVector2i(", inputExpr, ")"),
                MarshalType.Rect2 =>
                    source.Append(VariantUtils, ".ConvertToRect2(", inputExpr, ")"),
                MarshalType.Rect2i =>
                    source.Append(VariantUtils, ".ConvertToRect2i(", inputExpr, ")"),
                MarshalType.Transform2D =>
                    source.Append(VariantUtils, ".ConvertToTransform2D(", inputExpr, ")"),
                MarshalType.Vector3 =>
                    source.Append(VariantUtils, ".ConvertToVector3(", inputExpr, ")"),
                MarshalType.Vector3i =>
                    source.Append(VariantUtils, ".ConvertToVector3i(", inputExpr, ")"),
                MarshalType.Basis =>
                    source.Append(VariantUtils, ".ConvertToBasis(", inputExpr, ")"),
                MarshalType.Quaternion =>
                    source.Append(VariantUtils, ".ConvertToQuaternion(", inputExpr, ")"),
                MarshalType.Transform3D =>
                    source.Append(VariantUtils, ".ConvertToTransform3D(", inputExpr, ")"),
                MarshalType.Vector4 =>
                    source.Append(VariantUtils, ".ConvertToVector4(", inputExpr, ")"),
                MarshalType.Vector4i =>
                    source.Append(VariantUtils, ".ConvertToVector4i(", inputExpr, ")"),
                MarshalType.Projection =>
                    source.Append(VariantUtils, ".ConvertToProjection(", inputExpr, ")"),
                MarshalType.AABB =>
                    source.Append(VariantUtils, ".ConvertToAABB(", inputExpr, ")"),
                MarshalType.Color =>
                    source.Append(VariantUtils, ".ConvertToColor(", inputExpr, ")"),
                MarshalType.Plane =>
                    source.Append(VariantUtils, ".ConvertToPlane(", inputExpr, ")"),
                MarshalType.Callable =>
                    source.Append(VariantUtils, ".ConvertToCallableManaged(", inputExpr, ")"),
                MarshalType.SignalInfo =>
                    source.Append(VariantUtils, ".ConvertToSignalInfo(", inputExpr, ")"),
                MarshalType.Enum =>
                    source.Append("(", typeSymbol.FullQualifiedNameIncludeGlobal(),
                        ")", VariantUtils, ".ConvertToInt32(", inputExpr, ")"),
                MarshalType.ByteArray =>
                    source.Append(VariantUtils, ".ConvertAsPackedByteArrayToSystemArray(", inputExpr, ")"),
                MarshalType.Int32Array =>
                    source.Append(VariantUtils, ".ConvertAsPackedInt32ArrayToSystemArray(", inputExpr, ")"),
                MarshalType.Int64Array =>
                    source.Append(VariantUtils, ".ConvertAsPackedInt64ArrayToSystemArray(", inputExpr, ")"),
                MarshalType.Float32Array =>
                    source.Append(VariantUtils, ".ConvertAsPackedFloat32ArrayToSystemArray(", inputExpr, ")"),
                MarshalType.Float64Array =>
                    source.Append(VariantUtils, ".ConvertAsPackedFloat64ArrayToSystemArray(", inputExpr, ")"),
                MarshalType.StringArray =>
                    source.Append(VariantUtils, ".ConvertAsPackedStringArrayToSystemArray(", inputExpr, ")"),
                MarshalType.Vector2Array =>
                    source.Append(VariantUtils, ".ConvertAsPackedVector2ArrayToSystemArray(", inputExpr, ")"),
                MarshalType.Vector3Array =>
                    source.Append(VariantUtils, ".ConvertAsPackedVector3ArrayToSystemArray(", inputExpr, ")"),
                MarshalType.ColorArray =>
                    source.Append(VariantUtils, ".ConvertAsPackedColorArrayToSystemArray(", inputExpr, ")"),
                MarshalType.GodotObjectOrDerivedArray =>
                    source.Append(VariantUtils, ".ConvertToSystemArrayOfGodotObject<",
                        ((IArrayTypeSymbol)typeSymbol).ElementType.FullQualifiedNameIncludeGlobal(), ">(", inputExpr, ")"),
                MarshalType.SystemArrayOfStringName =>
                    source.Append(VariantUtils, ".ConvertToSystemArrayOfStringName(", inputExpr, ")"),
                MarshalType.SystemArrayOfNodePath =>
                    source.Append(VariantUtils, ".ConvertToSystemArrayOfNodePath(", inputExpr, ")"),
                MarshalType.SystemArrayOfRID =>
                    source.Append(VariantUtils, ".ConvertToSystemArrayOfRID(", inputExpr, ")"),
                MarshalType.Variant =>
                    source.Append("global::Godot.Variant.CreateCopyingBorrowed(", inputExpr, ")"),
                MarshalType.GodotObjectOrDerived =>
                    source.Append("(", typeSymbol.FullQualifiedNameIncludeGlobal(),
                        ")", VariantUtils, ".ConvertToGodotObject(", inputExpr, ")"),
                MarshalType.StringName =>
                    source.Append(VariantUtils, ".ConvertToStringNameObject(", inputExpr, ")"),
                MarshalType.NodePath =>
                    source.Append(VariantUtils, ".ConvertToNodePathObject(", inputExpr, ")"),
                MarshalType.RID =>
                    source.Append(VariantUtils, ".ConvertToRID(", inputExpr, ")"),
                MarshalType.GodotDictionary =>
                    source.Append(VariantUtils, ".ConvertToDictionaryObject(", inputExpr, ")"),
                MarshalType.GodotArray =>
                    source.Append(VariantUtils, ".ConvertToArrayObject(", inputExpr, ")"),
                MarshalType.GodotGenericDictionary =>
                    source.Append(VariantUtils, ".ConvertToDictionaryObject<",
                        ((INamedTypeSymbol)typeSymbol).TypeArguments[0].FullQualifiedNameIncludeGlobal(), ", ",
                        ((INamedTypeSymbol)typeSymbol).TypeArguments[1].FullQualifiedNameIncludeGlobal(), ">(", inputExpr, ")"),
                MarshalType.GodotGenericArray =>
                    source.Append(VariantUtils, ".ConvertToArrayObject<",
                        ((INamedTypeSymbol)typeSymbol).TypeArguments[0].FullQualifiedNameIncludeGlobal(), ">(", inputExpr, ")"),
                _ => throw new ArgumentOutOfRangeException(nameof(marshalType), marshalType,
                    "Received unexpected marshal type")
            };
        }

        public static StringBuilder AppendManagedToNativeVariantExpr(
            this StringBuilder source, string inputExpr, MarshalType marshalType)
        {
            return marshalType switch
            {
                MarshalType.Boolean =>
                    source.Append(VariantUtils, ".CreateFromBool(", inputExpr, ")"),
                MarshalType.Char =>
                    source.Append(VariantUtils, ".CreateFromInt((ushort)", inputExpr, ")"),
                MarshalType.SByte =>
                    source.Append(VariantUtils, ".CreateFromInt(", inputExpr, ")"),
                MarshalType.Int16 =>
                    source.Append(VariantUtils, ".CreateFromInt(", inputExpr, ")"),
                MarshalType.Int32 =>
                    source.Append(VariantUtils, ".CreateFromInt(", inputExpr, ")"),
                MarshalType.Int64 =>
                    source.Append(VariantUtils, ".CreateFromInt(", inputExpr, ")"),
                MarshalType.Byte =>
                    source.Append(VariantUtils, ".CreateFromInt(", inputExpr, ")"),
                MarshalType.UInt16 =>
                    source.Append(VariantUtils, ".CreateFromInt(", inputExpr, ")"),
                MarshalType.UInt32 =>
                    source.Append(VariantUtils, ".CreateFromInt(", inputExpr, ")"),
                MarshalType.UInt64 =>
                    source.Append(VariantUtils, ".CreateFromInt(", inputExpr, ")"),
                MarshalType.Single =>
                    source.Append(VariantUtils, ".CreateFromFloat(", inputExpr, ")"),
                MarshalType.Double =>
                    source.Append(VariantUtils, ".CreateFromFloat(", inputExpr, ")"),
                MarshalType.String =>
                    source.Append(VariantUtils, ".CreateFromString(", inputExpr, ")"),
                MarshalType.Vector2 =>
                    source.Append(VariantUtils, ".CreateFromVector2(", inputExpr, ")"),
                MarshalType.Vector2i =>
                    source.Append(VariantUtils, ".CreateFromVector2i(", inputExpr, ")"),
                MarshalType.Rect2 =>
                    source.Append(VariantUtils, ".CreateFromRect2(", inputExpr, ")"),
                MarshalType.Rect2i =>
                    source.Append(VariantUtils, ".CreateFromRect2i(", inputExpr, ")"),
                MarshalType.Transform2D =>
                    source.Append(VariantUtils, ".CreateFromTransform2D(", inputExpr, ")"),
                MarshalType.Vector3 =>
                    source.Append(VariantUtils, ".CreateFromVector3(", inputExpr, ")"),
                MarshalType.Vector3i =>
                    source.Append(VariantUtils, ".CreateFromVector3i(", inputExpr, ")"),
                MarshalType.Basis =>
                    source.Append(VariantUtils, ".CreateFromBasis(", inputExpr, ")"),
                MarshalType.Quaternion =>
                    source.Append(VariantUtils, ".CreateFromQuaternion(", inputExpr, ")"),
                MarshalType.Transform3D =>
                    source.Append(VariantUtils, ".CreateFromTransform3D(", inputExpr, ")"),
                MarshalType.Vector4 =>
                    source.Append(VariantUtils, ".CreateFromVector4(", inputExpr, ")"),
                MarshalType.Vector4i =>
                    source.Append(VariantUtils, ".CreateFromVector4i(", inputExpr, ")"),
                MarshalType.Projection =>
                    source.Append(VariantUtils, ".CreateFromProjection(", inputExpr, ")"),
                MarshalType.AABB =>
                    source.Append(VariantUtils, ".CreateFromAABB(", inputExpr, ")"),
                MarshalType.Color =>
                    source.Append(VariantUtils, ".CreateFromColor(", inputExpr, ")"),
                MarshalType.Plane =>
                    source.Append(VariantUtils, ".CreateFromPlane(", inputExpr, ")"),
                MarshalType.Callable =>
                    source.Append(VariantUtils, ".CreateFromCallable(", inputExpr, ")"),
                MarshalType.SignalInfo =>
                    source.Append(VariantUtils, ".CreateFromSignalInfo(", inputExpr, ")"),
                MarshalType.Enum =>
                    source.Append(VariantUtils, ".CreateFromInt((int)", inputExpr, ")"),
                MarshalType.ByteArray =>
                    source.Append(VariantUtils, ".CreateFromPackedByteArray(", inputExpr, ")"),
                MarshalType.Int32Array =>
                    source.Append(VariantUtils, ".CreateFromPackedInt32Array(", inputExpr, ")"),
                MarshalType.Int64Array =>
                    source.Append(VariantUtils, ".CreateFromPackedInt64Array(", inputExpr, ")"),
                MarshalType.Float32Array =>
                    source.Append(VariantUtils, ".CreateFromPackedFloat32Array(", inputExpr, ")"),
                MarshalType.Float64Array =>
                    source.Append(VariantUtils, ".CreateFromPackedFloat64Array(", inputExpr, ")"),
                MarshalType.StringArray =>
                    source.Append(VariantUtils, ".CreateFromPackedStringArray(", inputExpr, ")"),
                MarshalType.Vector2Array =>
                    source.Append(VariantUtils, ".CreateFromPackedVector2Array(", inputExpr, ")"),
                MarshalType.Vector3Array =>
                    source.Append(VariantUtils, ".CreateFromPackedVector3Array(", inputExpr, ")"),
                MarshalType.ColorArray =>
                    source.Append(VariantUtils, ".CreateFromPackedColorArray(", inputExpr, ")"),
                MarshalType.GodotObjectOrDerivedArray =>
                    source.Append(VariantUtils, ".CreateFromSystemArrayOfGodotObject(", inputExpr, ")"),
                MarshalType.SystemArrayOfStringName =>
                    source.Append(VariantUtils, ".CreateFromSystemArrayOfStringName(", inputExpr, ")"),
                MarshalType.SystemArrayOfNodePath =>
                    source.Append(VariantUtils, ".CreateFromSystemArrayOfNodePath(", inputExpr, ")"),
                MarshalType.SystemArrayOfRID =>
                    source.Append(VariantUtils, ".CreateFromSystemArrayOfRID(", inputExpr, ")"),
                MarshalType.Variant =>
                    source.Append(inputExpr, ".CopyNativeVariant()"),
                MarshalType.GodotObjectOrDerived =>
                    source.Append(VariantUtils, ".CreateFromGodotObject(", inputExpr, ")"),
                MarshalType.StringName =>
                    source.Append(VariantUtils, ".CreateFromStringName(", inputExpr, ")"),
                MarshalType.NodePath =>
                    source.Append(VariantUtils, ".CreateFromNodePath(", inputExpr, ")"),
                MarshalType.RID =>
                    source.Append(VariantUtils, ".CreateFromRID(", inputExpr, ")"),
                MarshalType.GodotDictionary =>
                    source.Append(VariantUtils, ".CreateFromDictionary(", inputExpr, ")"),
                MarshalType.GodotArray =>
                    source.Append(VariantUtils, ".CreateFromArray(", inputExpr, ")"),
                MarshalType.GodotGenericDictionary =>
                    source.Append(VariantUtils, ".CreateFromDictionary(", inputExpr, ")"),
                MarshalType.GodotGenericArray =>
                    source.Append(VariantUtils, ".CreateFromArray(", inputExpr, ")"),
                _ => throw new ArgumentOutOfRangeException(nameof(marshalType), marshalType,
                    "Received unexpected marshal type")
            };
        }

        public static StringBuilder AppendVariantToManagedExpr(this StringBuilder source,
            string inputExpr, ITypeSymbol typeSymbol, MarshalType marshalType)
        {
            return marshalType switch
            {
                MarshalType.Boolean => source.Append(inputExpr, ".AsBool()"),
                MarshalType.Char => source.Append(inputExpr, ".AsChar()"),
                MarshalType.SByte => source.Append(inputExpr, ".AsSByte()"),
                MarshalType.Int16 => source.Append(inputExpr, ".AsInt16()"),
                MarshalType.Int32 => source.Append(inputExpr, ".AsInt32()"),
                MarshalType.Int64 => source.Append(inputExpr, ".AsInt64()"),
                MarshalType.Byte => source.Append(inputExpr, ".AsByte()"),
                MarshalType.UInt16 => source.Append(inputExpr, ".AsUInt16()"),
                MarshalType.UInt32 => source.Append(inputExpr, ".AsUInt32()"),
                MarshalType.UInt64 => source.Append(inputExpr, ".AsUInt64()"),
                MarshalType.Single => source.Append(inputExpr, ".AsSingle()"),
                MarshalType.Double => source.Append(inputExpr, ".AsDouble()"),
                MarshalType.String => source.Append(inputExpr, ".AsString()"),
                MarshalType.Vector2 => source.Append(inputExpr, ".AsVector2()"),
                MarshalType.Vector2i => source.Append(inputExpr, ".AsVector2i()"),
                MarshalType.Rect2 => source.Append(inputExpr, ".AsRect2()"),
                MarshalType.Rect2i => source.Append(inputExpr, ".AsRect2i()"),
                MarshalType.Transform2D => source.Append(inputExpr, ".AsTransform2D()"),
                MarshalType.Vector3 => source.Append(inputExpr, ".AsVector3()"),
                MarshalType.Vector3i => source.Append(inputExpr, ".AsVector3i()"),
                MarshalType.Basis => source.Append(inputExpr, ".AsBasis()"),
                MarshalType.Quaternion => source.Append(inputExpr, ".AsQuaternion()"),
                MarshalType.Transform3D => source.Append(inputExpr, ".AsTransform3D()"),
                MarshalType.Vector4 => source.Append(inputExpr, ".AsVector4()"),
                MarshalType.Vector4i => source.Append(inputExpr, ".AsVector4i()"),
                MarshalType.Projection => source.Append(inputExpr, ".AsProjection()"),
                MarshalType.AABB => source.Append(inputExpr, ".AsAABB()"),
                MarshalType.Color => source.Append(inputExpr, ".AsColor()"),
                MarshalType.Plane => source.Append(inputExpr, ".AsPlane()"),
                MarshalType.Callable => source.Append(inputExpr, ".AsCallable()"),
                MarshalType.SignalInfo => source.Append(inputExpr, ".AsSignalInfo()"),
                MarshalType.Enum =>
                    source.Append("(", typeSymbol.FullQualifiedNameIncludeGlobal(), ")", inputExpr, ".AsInt64()"),
                MarshalType.ByteArray => source.Append(inputExpr, ".AsByteArray()"),
                MarshalType.Int32Array => source.Append(inputExpr, ".AsInt32Array()"),
                MarshalType.Int64Array => source.Append(inputExpr, ".AsInt64Array()"),
                MarshalType.Float32Array => source.Append(inputExpr, ".AsFloat32Array()"),
                MarshalType.Float64Array => source.Append(inputExpr, ".AsFloat64Array()"),
                MarshalType.StringArray => source.Append(inputExpr, ".AsStringArray()"),
                MarshalType.Vector2Array => source.Append(inputExpr, ".AsVector2Array()"),
                MarshalType.Vector3Array => source.Append(inputExpr, ".AsVector3Array()"),
                MarshalType.ColorArray => source.Append(inputExpr, ".AsColorArray()"),
                MarshalType.GodotObjectOrDerivedArray => source.Append(inputExpr, ".AsGodotObjectArray<",
                    ((IArrayTypeSymbol)typeSymbol).ElementType.FullQualifiedNameIncludeGlobal(), ">()"),
                MarshalType.SystemArrayOfStringName => source.Append(inputExpr, ".AsSystemArrayOfStringName()"),
                MarshalType.SystemArrayOfNodePath => source.Append(inputExpr, ".AsSystemArrayOfNodePath()"),
                MarshalType.SystemArrayOfRID => source.Append(inputExpr, ".AsSystemArrayOfRID()"),
                MarshalType.Variant => source.Append(inputExpr),
                MarshalType.GodotObjectOrDerived => source.Append("(",
                    typeSymbol.FullQualifiedNameIncludeGlobal(), ")", inputExpr, ".AsGodotObject()"),
                MarshalType.StringName => source.Append(inputExpr, ".AsStringName()"),
                MarshalType.NodePath => source.Append(inputExpr, ".AsNodePath()"),
                MarshalType.RID => source.Append(inputExpr, ".AsRID()"),
                MarshalType.GodotDictionary => source.Append(inputExpr, ".AsGodotDictionary()"),
                MarshalType.GodotArray => source.Append(inputExpr, ".AsGodotArray()"),
                MarshalType.GodotGenericDictionary => source.Append(inputExpr, ".AsGodotDictionary<",
                    ((INamedTypeSymbol)typeSymbol).TypeArguments[0].FullQualifiedNameIncludeGlobal(), ", ",
                    ((INamedTypeSymbol)typeSymbol).TypeArguments[1].FullQualifiedNameIncludeGlobal(), ">()"),
                MarshalType.GodotGenericArray => source.Append(inputExpr, ".AsGodotArray<",
                    ((INamedTypeSymbol)typeSymbol).TypeArguments[0].FullQualifiedNameIncludeGlobal(), ">()"),
                _ => throw new ArgumentOutOfRangeException(nameof(marshalType), marshalType,
                    "Received unexpected marshal type")
            };
        }

        public static StringBuilder AppendManagedToVariantExpr(this StringBuilder source,
            string inputExpr, MarshalType marshalType)
        {
            switch (marshalType)
            {
                case MarshalType.Boolean:
                case MarshalType.Char:
                case MarshalType.SByte:
                case MarshalType.Int16:
                case MarshalType.Int32:
                case MarshalType.Int64:
                case MarshalType.Byte:
                case MarshalType.UInt16:
                case MarshalType.UInt32:
                case MarshalType.UInt64:
                case MarshalType.Single:
                case MarshalType.Double:
                case MarshalType.String:
                case MarshalType.Vector2:
                case MarshalType.Vector2i:
                case MarshalType.Rect2:
                case MarshalType.Rect2i:
                case MarshalType.Transform2D:
                case MarshalType.Vector3:
                case MarshalType.Vector3i:
                case MarshalType.Basis:
                case MarshalType.Quaternion:
                case MarshalType.Transform3D:
                case MarshalType.Vector4:
                case MarshalType.Vector4i:
                case MarshalType.Projection:
                case MarshalType.AABB:
                case MarshalType.Color:
                case MarshalType.Plane:
                case MarshalType.Callable:
                case MarshalType.SignalInfo:
                case MarshalType.ByteArray:
                case MarshalType.Int32Array:
                case MarshalType.Int64Array:
                case MarshalType.Float32Array:
                case MarshalType.Float64Array:
                case MarshalType.StringArray:
                case MarshalType.Vector2Array:
                case MarshalType.Vector3Array:
                case MarshalType.ColorArray:
                case MarshalType.GodotObjectOrDerivedArray:
                case MarshalType.SystemArrayOfStringName:
                case MarshalType.SystemArrayOfNodePath:
                case MarshalType.SystemArrayOfRID:
                case MarshalType.GodotObjectOrDerived:
                case MarshalType.StringName:
                case MarshalType.NodePath:
                case MarshalType.RID:
                case MarshalType.GodotDictionary:
                case MarshalType.GodotArray:
                case MarshalType.GodotGenericDictionary:
                case MarshalType.GodotGenericArray:
                    return source.Append("Variant.CreateFrom(", inputExpr, ")");
                case MarshalType.Enum:
                    return source.Append("Variant.CreateFrom((long)", inputExpr, ")");
                case MarshalType.Variant:
                    return source.Append(inputExpr);
                default:
                    throw new ArgumentOutOfRangeException(nameof(marshalType), marshalType,
                        "Received unexpected marshal type");
            }
        }
    }
}
