using System.Collections.Immutable;
using Microsoft.CodeAnalysis;

namespace Godot.SourceGenerators
{
    public readonly struct GodotMethodData
    {
        public GodotMethodData(IMethodSymbol method, ImmutableArray<MarshalType> paramTypes,
            ImmutableArray<ITypeSymbol> paramTypeSymbols, (MarshalType MarshalType, ITypeSymbol TypeSymbol)? retType)
        {
            Method = method;
            ParamTypes = paramTypes;
            ParamTypeSymbols = paramTypeSymbols;
            RetType = retType;
        }

        public IMethodSymbol Method { get; }
        public ImmutableArray<MarshalType> ParamTypes { get; }
        public ImmutableArray<ITypeSymbol> ParamTypeSymbols { get; }
        public (MarshalType MarshalType, ITypeSymbol TypeSymbol)? RetType { get; }
    }

    public readonly struct GodotSignalDelegateData
    {
        public GodotSignalDelegateData(string name, INamedTypeSymbol delegateSymbol, GodotMethodData invokeMethodData)
        {
            Name = name;
            DelegateSymbol = delegateSymbol;
            InvokeMethodData = invokeMethodData;
        }

        public string Name { get; }
        public INamedTypeSymbol DelegateSymbol { get; }
        public GodotMethodData InvokeMethodData { get; }
    }

    public enum PropertyType
    {
        Unallowed,
        Field,
        Property,
        Embed,
    }

    public class GodotPropertyData
    {
        public GodotPropertyData(
            ISymbol symbol,
            ITypeSymbol typeSymbol,
            MarshalType marshalType,
            PropertyType propertyType,
            GodotPropertyData? containingProperty,
            bool isReadOnly,
            bool isWriteOnly)
        {
            Symbol = symbol;
            PropertyTypeSymbol = typeSymbol;
            MarshalType = marshalType;
            PropertyType = propertyType;
            ContainingProperty = containingProperty;
            IsReadOnly = isReadOnly;
            IsWriteOnly = isWriteOnly;
        }

        public GodotPropertyData(IPropertySymbol propertySymbol, MarshalType type, PropertyType propertyType, GodotPropertyData? containingProperty)
            : this(propertySymbol, propertySymbol.Type, type, propertyType, containingProperty,
                propertySymbol.IsReadOnly || propertySymbol.SetMethodOrBaseSetMethod() is { IsInitOnly: true }, propertySymbol.IsWriteOnly)
        {
        }

        public GodotPropertyData(IFieldSymbol fieldSymbol, MarshalType type, PropertyType propertyType, GodotPropertyData? containingProperty)
            : this(fieldSymbol, fieldSymbol.Type, type, propertyType, containingProperty, fieldSymbol.IsReadOnly, false)
        {
        }

        public GodotPropertyData? ContainingProperty { get; }
        public ISymbol Symbol { get; }
        public ITypeSymbol PropertyTypeSymbol { get; }
        public MarshalType MarshalType { get; }
        public PropertyType PropertyType { get; }
        [System.Diagnostics.CodeAnalysis.MemberNotNullWhen(true, nameof(ContainingProperty))]
        public bool InNullable => ContainingProperty is not null && (ContainingProperty.InNullable || ContainingProperty.PropertyTypeSymbol.IsReferenceType);
        public string MemberName => string.Concat(ContainingProperty is not null ? $"{ContainingProperty.MemberName}.@" : "", Symbol.Name);
        public string MemberNameNullable => string.Concat(
            ContainingProperty is not null
            ? string.Concat(ContainingProperty.MemberNameNullable, ContainingProperty.PropertyTypeSymbol.IsReferenceType ? $"?" : "", ".@")
            : "", Symbol.Name);
        public string PropertyName => string.Concat(ContainingProperty is not null ? $"{ContainingProperty.PropertyName}_" : "", Symbol.Name);
        public string PropertyNameHint => string.Concat(ContainingProperty is not null ? $"{ContainingProperty.PropertyGroupName}/" : "", Symbol.Name);
        public string PropertyGroupName => string.Concat(ContainingProperty is not null ? $"{ContainingProperty.PropertyGroupName}/" : "", Capitalize(Symbol.Name));
        public bool IsReadOnly { get; }
        public bool IsWriteOnly { get; }

        private static string Capitalize(string input)
        {
            return input[0].ToString().ToUpper() + input.Substring(1);
        }
    }
}
