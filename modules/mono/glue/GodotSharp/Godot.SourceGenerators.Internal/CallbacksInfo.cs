using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;

namespace Godot.SourceGenerators.Internal;

internal readonly struct CallbacksData
{
    public CallbacksData(INamedTypeSymbol nativeTypeSymbol, INamedTypeSymbol funcStructSymbol)
    {
        NativeTypeSymbol = nativeTypeSymbol;
        FuncStructSymbol = funcStructSymbol;
        Methods = NativeTypeSymbol.GetMembers()
            .Where(symbol => symbol is IMethodSymbol { IsPartialDefinition: true })
            .Cast<IMethodSymbol>()
            .ToImmutableArray();
    }

    public INamedTypeSymbol NativeTypeSymbol { get; }

    public INamedTypeSymbol FuncStructSymbol { get; }

    public ImmutableArray<IMethodSymbol> Methods { get; }
}
