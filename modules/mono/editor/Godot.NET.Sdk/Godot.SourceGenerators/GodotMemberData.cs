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

    public readonly struct GodotPropertyData
    {
        public GodotPropertyData(IPropertySymbol propertySymbol, MarshalType type)
        {
            PropertySymbol = propertySymbol;
            Type = type;
        }

        public IPropertySymbol PropertySymbol { get; }
        public MarshalType Type { get; }
    }

    public readonly struct GodotFieldData
    {
        public GodotFieldData(IFieldSymbol fieldSymbol, MarshalType type)
        {
            FieldSymbol = fieldSymbol;
            Type = type;
        }

        public IFieldSymbol FieldSymbol { get; }
        public MarshalType Type { get; }
    }

    public struct GodotPropertyOrFieldData
    {
        public GodotPropertyOrFieldData(ISymbol symbol, MarshalType type)
        {
            Symbol = symbol;
            Type = type;
        }

        public GodotPropertyOrFieldData(GodotPropertyData propertyData)
            : this(propertyData.PropertySymbol, propertyData.Type)
        {
        }

        public GodotPropertyOrFieldData(GodotFieldData fieldData)
            : this(fieldData.FieldSymbol, fieldData.Type)
        {
        }

        public ISymbol Symbol { get; }
        public MarshalType Type { get; }
    }
}
