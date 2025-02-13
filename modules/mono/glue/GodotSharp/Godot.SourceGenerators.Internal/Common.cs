using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Godot.SourceGenerators.Internal;

internal static class Common
{
    public static void ReportNonPartialUnmanagedCallbacksClass(
        GeneratorExecutionContext context,
        ClassDeclarationSyntax cds, INamedTypeSymbol symbol
    )
    {
        string message =
            "Missing partial modifier on declaration of type '" +
            $"{symbol.FullQualifiedNameOmitGlobal()}' which has attribute '{GeneratorClasses.GenerateUnmanagedCallbacksAttr}'";

        string description = $"{message}. Classes with attribute '{GeneratorClasses.GenerateUnmanagedCallbacksAttr}' " +
                             "must be declared with the partial modifier.";

        context.ReportDiagnostic(Diagnostic.Create(
            new DiagnosticDescriptor(id: "GODOT-INTERNAL-G0001",
                title: message,
                messageFormat: message,
                category: "Usage",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true,
                description),
            cds.GetLocation(),
            cds.SyntaxTree.FilePath));
    }

    public static void ReportNonPartialUnmanagedCallbacksOuterClass(
        GeneratorExecutionContext context,
        TypeDeclarationSyntax outerTypeDeclSyntax
    )
    {
        var outerSymbol = context.Compilation
            .GetSemanticModel(outerTypeDeclSyntax.SyntaxTree)
            .GetDeclaredSymbol(outerTypeDeclSyntax);

        string fullQualifiedName = outerSymbol is INamedTypeSymbol namedTypeSymbol ?
            namedTypeSymbol.FullQualifiedNameOmitGlobal() :
            "type not found";

        string message =
            $"Missing partial modifier on declaration of type '{fullQualifiedName}', " +
            $"which contains one or more subclasses with attribute " +
            $"'{GeneratorClasses.GenerateUnmanagedCallbacksAttr}'";

        string description = $"{message}. Classes with attribute " +
                             $"'{GeneratorClasses.GenerateUnmanagedCallbacksAttr}' and their " +
                             "containing types must be declared with the partial modifier.";

        context.ReportDiagnostic(Diagnostic.Create(
            new DiagnosticDescriptor(id: "GODOT-INTERNAL-G0002",
                title: message,
                messageFormat: message,
                category: "Usage",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true,
                description),
            outerTypeDeclSyntax.GetLocation(),
            outerTypeDeclSyntax.SyntaxTree.FilePath));
    }
}
