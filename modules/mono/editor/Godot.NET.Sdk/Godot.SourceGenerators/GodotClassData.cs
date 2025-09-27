using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Godot.SourceGenerators
{
    public record GodotClassData(ClassDeclarationSyntax DeclarationSyntax, INamedTypeSymbol Symbol)
    {
        public ClassDeclarationSyntax DeclarationSyntax { get; } = DeclarationSyntax;
        public INamedTypeSymbol Symbol { get; } = Symbol;
    }
}
