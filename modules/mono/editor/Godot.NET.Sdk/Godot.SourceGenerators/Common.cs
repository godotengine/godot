using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Godot.SourceGenerators
{
    public static class Common
    {
        public static void ReportNonPartialGodotScriptClass(
            GeneratorExecutionContext context,
            ClassDeclarationSyntax cds, INamedTypeSymbol symbol
        )
        {
            string message =
                "Missing partial modifier on declaration of type '" +
                $"{symbol.FullQualifiedName()}' which is a subclass of '{GodotClasses.Object}'";

            string description = $"{message}. Subclasses of '{GodotClasses.Object}' must be " +
                                 "declared with the partial modifier or annotated with the " +
                                 $"attribute '{GodotClasses.DisableGodotGeneratorsAttr}'.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GODOT-G0001",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description),
                cds.GetLocation(),
                cds.SyntaxTree.FilePath));
        }
    }
}
