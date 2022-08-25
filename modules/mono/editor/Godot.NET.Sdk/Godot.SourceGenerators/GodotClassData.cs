using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Godot.SourceGenerators
{
    public readonly struct GodotClassData
    {
        public GodotClassData(ClassDeclarationSyntax cds, INamedTypeSymbol symbol)
        {
            DeclarationSyntax = cds;
            Symbol = symbol;
        }

        public ClassDeclarationSyntax DeclarationSyntax { get; }
        public INamedTypeSymbol Symbol { get; }
    }
}
