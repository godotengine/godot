using System;
using System.Linq;
using Microsoft.CodeAnalysis;

namespace Godot.SourceGenerators
{
    internal static class MarshalUtils
    {
        public class TypeCache
        {
            public INamedTypeSymbol GodotObjectType { get; }
            public INamedTypeSymbol GodotGenericDictionary { get; }
            public INamedTypeSymbol GodotGenericArray { get; }
            public INamedTypeSymbol IDictionary { get; }
            public INamedTypeSymbol ICollection { get; }
            public INamedTypeSymbol GenericIDictionary { get; }
            public INamedTypeSymbol SystemGenericDictionary { get; }
            public INamedTypeSymbol SystemGenericList { get; }

            public TypeCache(GeneratorExecutionContext context)
            {
                INamedTypeSymbol GetTypeByMetadataNameOrThrow(string fullyQualifiedMetadataName)
                {
                    return context.Compilation.GetTypeByMetadataName(fullyQualifiedMetadataName) ??
                           throw new InvalidOperationException("Type not found: " + fullyQualifiedMetadataName);
                }

                GodotObjectType = GetTypeByMetadataNameOrThrow("Godot.Object");
                GodotGenericDictionary = GetTypeByMetadataNameOrThrow("Godot.Collections.Dictionary`2");
                GodotGenericArray = GetTypeByMetadataNameOrThrow("Godot.Collections.Array`1");
                IDictionary = GetTypeByMetadataNameOrThrow("System.Collections.IDictionary");
                ICollection = GetTypeByMetadataNameOrThrow("System.Collections.ICollection");
                GenericIDictionary = GetTypeByMetadataNameOrThrow("System.Collections.Generic.IDictionary`2");
                SystemGenericDictionary = GetTypeByMetadataNameOrThrow("System.Collections.Generic.Dictionary`2");
                SystemGenericList = GetTypeByMetadataNameOrThrow("System.Collections.Generic.List`1");
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
                MarshalType.SingleArray => VariantType.PackedFloat32Array,
                MarshalType.DoubleArray => VariantType.PackedFloat64Array,
                MarshalType.StringArray => VariantType.PackedStringArray,
                MarshalType.Vector2Array => VariantType.PackedVector2Array,
                MarshalType.Vector3Array => VariantType.PackedVector3Array,
                MarshalType.ColorArray => VariantType.PackedColorArray,
                MarshalType.GodotObjectOrDerivedArray => VariantType.Array,
                MarshalType.SystemObjectArray => VariantType.Array,
                MarshalType.GodotGenericDictionary => VariantType.Dictionary,
                MarshalType.GodotGenericArray => VariantType.Array,
                MarshalType.SystemGenericDictionary => VariantType.Dictionary,
                MarshalType.SystemGenericList => VariantType.Array,
                MarshalType.GenericIDictionary => VariantType.Dictionary,
                MarshalType.GenericICollection => VariantType.Array,
                MarshalType.GenericIEnumerable => VariantType.Array,
                MarshalType.SystemObject => VariantType.Nil,
                MarshalType.GodotObjectOrDerived => VariantType.Object,
                MarshalType.StringName => VariantType.StringName,
                MarshalType.NodePath => VariantType.NodePath,
                MarshalType.RID => VariantType.Rid,
                MarshalType.GodotDictionary => VariantType.Dictionary,
                MarshalType.GodotArray => VariantType.Array,
                MarshalType.IDictionary => VariantType.Dictionary,
                MarshalType.ICollection => VariantType.Array,
                MarshalType.IEnumerable => VariantType.Array,
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
                case SpecialType.System_Object:
                    return MarshalType.SystemObject;
                default:
                {
                    var typeKind = type.TypeKind;

                    if (typeKind == TypeKind.Enum)
                        return MarshalType.Enum;

                    if (typeKind == TypeKind.Struct)
                    {
                        if (type.ContainingAssembly.Name == "GodotSharp" &&
                            type.ContainingNamespace.Name == "Godot")
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
                                _ => null
                            };
                        }
                    }
                    else if (typeKind == TypeKind.Array)
                    {
                        var arrayType = (IArrayTypeSymbol)type;
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
                                return MarshalType.SingleArray;
                            case SpecialType.System_Double:
                                return MarshalType.DoubleArray;
                            case SpecialType.System_String:
                                return MarshalType.StringArray;
                            case SpecialType.System_Object:
                                return MarshalType.SystemObjectArray;
                        }

                        if (elementType.SimpleDerivesFrom(typeCache.GodotObjectType))
                            return MarshalType.GodotObjectOrDerivedArray;

                        if (elementType.ContainingAssembly.Name == "GodotSharp" &&
                            elementType.ContainingNamespace.Name == "Godot")
                        {
                            switch (elementType)
                            {
                                case { Name: "Vector2" }:
                                    return MarshalType.Vector2Array;
                                case { Name: "Vector3" }:
                                    return MarshalType.Vector3Array;
                                case { Name: "Color" }:
                                return MarshalType.ColorArray;
                            }
                        }

                        if (ConvertManagedTypeToMarshalType(elementType, typeCache) != null)
                            return MarshalType.GodotArray;

                        return null;
                    }
                    else if (type is INamedTypeSymbol { IsGenericType: true } genericType)
                    {
                        var genericTypeDef = genericType.ConstructedFrom;

                        if (SymbolEqualityComparer.Default.Equals(genericTypeDef, typeCache.GodotGenericDictionary))
                            return MarshalType.GodotGenericDictionary;

                        if (SymbolEqualityComparer.Default.Equals(genericTypeDef, typeCache.GodotGenericArray))
                            return MarshalType.GodotGenericArray;

                        if (SymbolEqualityComparer.Default.Equals(genericTypeDef, typeCache.SystemGenericDictionary))
                            return MarshalType.SystemGenericDictionary;

                        if (SymbolEqualityComparer.Default.Equals(genericTypeDef, typeCache.SystemGenericList))
                            return MarshalType.SystemGenericList;

                        if (SymbolEqualityComparer.Default.Equals(genericTypeDef, typeCache.GenericIDictionary))
                            return MarshalType.GenericIDictionary;

                        return genericTypeDef.SpecialType switch
                        {
                            SpecialType.System_Collections_Generic_ICollection_T => MarshalType.GenericICollection,
                            SpecialType.System_Collections_Generic_IEnumerable_T => MarshalType.GenericIEnumerable,
                            _ => null
                        };
                    }
                    else
                    {
                        if (type.SimpleDerivesFrom(typeCache.GodotObjectType))
                            return MarshalType.GodotObjectOrDerived;

                        if (SymbolEqualityComparer.Default.Equals(type, typeCache.IDictionary))
                            return MarshalType.IDictionary;

                        if (SymbolEqualityComparer.Default.Equals(type, typeCache.ICollection))
                            return MarshalType.ICollection;

                        if (specialType == SpecialType.System_Collections_IEnumerable)
                            return MarshalType.IEnumerable;

                        if (type.ContainingAssembly.Name == "GodotSharp")
                        {
                            switch (type.ContainingNamespace.Name)
                            {
                                case "Godot":
                                    return type switch
                                    {
                                        { Name: "StringName" } => MarshalType.StringName,
                                        { Name: "NodePath" } => MarshalType.NodePath,
                                        _ => null
                                    };
                                case "Collections"
                                    when !(type is INamedTypeSymbol { IsGenericType: true }) &&
                                         type.ContainingNamespace.FullQualifiedName() ==
                                         "Godot.Collections":
                                    return type switch
                                    {
                                        { Name: "Dictionary" } => MarshalType.GodotDictionary,
                                        { Name: "Array" } => MarshalType.GodotArray,
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
    }
}
