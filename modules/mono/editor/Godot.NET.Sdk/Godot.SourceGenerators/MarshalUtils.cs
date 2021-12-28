using System;
using Microsoft.CodeAnalysis;

namespace Godot.SourceGenerators
{
    public static class MarshalUtils
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

        public static MarshalType? ConvertManagedTypeToVariantType(ITypeSymbol type, TypeCache typeCache)
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
                case SpecialType.System_ValueType:
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
                            { Name: "AABB" } => MarshalType.AABB,
                            { Name: "Color" } => MarshalType.Color,
                            { Name: "Plane" } => MarshalType.Plane,
                            { Name: "RID" } => MarshalType.RID,
                            { Name: "Callable" } => MarshalType.Callable,
                            { Name: "SignalInfo" } => MarshalType.SignalInfo,
                            { TypeKind: TypeKind.Enum } => MarshalType.Enum,
                            _ => null
                        };
                    }

                    return null;
                }
                default:
                {
                    if (type.TypeKind == TypeKind.Array)
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

                        if (type.ContainingAssembly.Name == "GodotSharp" &&
                            type.ContainingNamespace.Name == "Godot")
                        {
                            return elementType switch
                            {
                                { Name: "Vector2" } => MarshalType.Vector2Array,
                                { Name: "Vector3" } => MarshalType.Vector3Array,
                                { Name: "Color" } => MarshalType.ColorArray,
                                _ => null
                            };
                        }
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
                                case "Godot.Collections" when !(type is INamedTypeSymbol { IsGenericType: true }):
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
    }
}
