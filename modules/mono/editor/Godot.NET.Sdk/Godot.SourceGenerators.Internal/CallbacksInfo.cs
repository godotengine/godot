using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;

namespace Godot.SourceGenerators
{
    public struct CallbacksData
    {
        public CallbacksData(INamedTypeSymbol symbol, INamedTypeSymbol structSymbol)
        {
            NativeTypeSymbol = symbol;
            FuncStructSymbol = structSymbol;
            Methods = NativeTypeSymbol.GetMembers()
                .Where(symbol => symbol is IMethodSymbol mds && mds.IsPartialDefinition)
                .Cast<IMethodSymbol>().ToImmutableArray();
        }

        public INamedTypeSymbol NativeTypeSymbol { get; }

        public INamedTypeSymbol FuncStructSymbol { get; }

        public ImmutableArray<IMethodSymbol> Methods { get; }
    }
}
