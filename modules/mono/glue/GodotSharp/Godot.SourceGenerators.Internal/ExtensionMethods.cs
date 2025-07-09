using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Godot.SourceGenerators.Internal;

internal static class ExtensionMethods
{
    public static AttributeData? GetGenerateUnmanagedCallbacksAttribute(this INamedTypeSymbol symbol)
        => symbol.GetAttributes()
            .FirstOrDefault(a => a.AttributeClass?.IsGenerateUnmanagedCallbacksAttribute() ?? false);

    private static bool HasGenerateUnmanagedCallbacksAttribute(
        this ClassDeclarationSyntax cds, Compilation compilation,
        out INamedTypeSymbol? symbol
    )
    {
        var sm = compilation.GetSemanticModel(cds.SyntaxTree);

        var classTypeSymbol = sm.GetDeclaredSymbol(cds);
        if (classTypeSymbol == null)
        {
            symbol = null;
            return false;
        }

        if (!classTypeSymbol.GetAttributes()
                .Any(a => a.AttributeClass?.IsGenerateUnmanagedCallbacksAttribute() ?? false))
        {
            symbol = null;
            return false;
        }

        symbol = classTypeSymbol;
        return true;
    }

    private static bool IsGenerateUnmanagedCallbacksAttribute(this INamedTypeSymbol symbol)
        => symbol.FullQualifiedNameOmitGlobal() == GeneratorClasses.GenerateUnmanagedCallbacksAttr;

    public static IEnumerable<(ClassDeclarationSyntax cds, INamedTypeSymbol symbol)> SelectUnmanagedCallbacksClasses(
        this IEnumerable<ClassDeclarationSyntax> source,
        Compilation compilation
    )
    {
        foreach (var cds in source)
        {
            if (cds.HasGenerateUnmanagedCallbacksAttribute(compilation, out var symbol))
                yield return (cds, symbol!);
        }
    }

    public static bool IsNested(this TypeDeclarationSyntax cds)
        => cds.Parent is TypeDeclarationSyntax;

    public static bool IsPartial(this TypeDeclarationSyntax cds)
        => cds.Modifiers.Any(SyntaxKind.PartialKeyword);

    public static bool AreAllOuterTypesPartial(
        this TypeDeclarationSyntax cds,
        out TypeDeclarationSyntax? typeMissingPartial
    )
    {
        SyntaxNode? outerSyntaxNode = cds.Parent;

        while (outerSyntaxNode is TypeDeclarationSyntax outerTypeDeclSyntax)
        {
            if (!outerTypeDeclSyntax.IsPartial())
            {
                typeMissingPartial = outerTypeDeclSyntax;
                return false;
            }

            outerSyntaxNode = outerSyntaxNode.Parent;
        }

        typeMissingPartial = null;
        return true;
    }

    public static string GetDeclarationKeyword(this INamedTypeSymbol namedTypeSymbol)
    {
        string? keyword = namedTypeSymbol.DeclaringSyntaxReferences
            .OfType<TypeDeclarationSyntax>().FirstOrDefault()?
            .Keyword.Text;

        return keyword ?? namedTypeSymbol.TypeKind switch
        {
            TypeKind.Interface => "interface",
            TypeKind.Struct => "struct",
            _ => "class"
        };
    }

    public static string NameWithTypeParameters(this INamedTypeSymbol symbol)
    {
        return symbol.IsGenericType && symbol.TypeParameters.Length > 0 ?
            string.Concat(symbol.Name, "<", string.Join(", ", symbol.TypeParameters), ">") :
            symbol.Name;
    }

    private static SymbolDisplayFormat FullyQualifiedFormatOmitGlobal { get; } =
        SymbolDisplayFormat.FullyQualifiedFormat
            .WithGlobalNamespaceStyle(SymbolDisplayGlobalNamespaceStyle.Omitted);

    private static SymbolDisplayFormat FullyQualifiedFormatIncludeGlobal { get; } =
        SymbolDisplayFormat.FullyQualifiedFormat
            .WithGlobalNamespaceStyle(SymbolDisplayGlobalNamespaceStyle.Included);

    public static string FullQualifiedNameOmitGlobal(this ITypeSymbol symbol)
        => symbol.ToDisplayString(NullableFlowState.NotNull, FullyQualifiedFormatOmitGlobal);

    public static string FullQualifiedNameOmitGlobal(this INamespaceSymbol namespaceSymbol)
        => namespaceSymbol.ToDisplayString(FullyQualifiedFormatOmitGlobal);

    public static string FullQualifiedNameIncludeGlobal(this ITypeSymbol symbol)
        => symbol.ToDisplayString(NullableFlowState.NotNull, FullyQualifiedFormatIncludeGlobal);

    public static string FullQualifiedNameIncludeGlobal(this INamespaceSymbol namespaceSymbol)
        => namespaceSymbol.ToDisplayString(FullyQualifiedFormatIncludeGlobal);

    public static string SanitizeQualifiedNameForUniqueHint(this string qualifiedName)
        => qualifiedName
            // AddSource() doesn't support angle brackets
            .Replace("<", "(Of ")
            .Replace(">", ")");
}
