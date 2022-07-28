using System.Linq;
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

            string description = $"{message}. Subclasses of '{GodotClasses.Object}' " +
                                 "must be declared with the partial modifier.";

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

        public static void ReportNonPartialGodotScriptOuterClass(
            GeneratorExecutionContext context,
            TypeDeclarationSyntax outerTypeDeclSyntax
        )
        {
            var outerSymbol = context.Compilation
                .GetSemanticModel(outerTypeDeclSyntax.SyntaxTree)
                .GetDeclaredSymbol(outerTypeDeclSyntax);

            string fullQualifiedName = outerSymbol is INamedTypeSymbol namedTypeSymbol ?
                namedTypeSymbol.FullQualifiedName() :
                "type not found";

            string message =
                $"Missing partial modifier on declaration of type '{fullQualifiedName}', " +
                $"which contains one or more subclasses of '{GodotClasses.Object}'";

            string description = $"{message}. Subclasses of '{GodotClasses.Object}' and their " +
                                 "containing types must be declared with the partial modifier.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GODOT-G0002",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description),
                outerTypeDeclSyntax.GetLocation(),
                outerTypeDeclSyntax.SyntaxTree.FilePath));
        }

        public static void ReportExportedMemberIsStatic(
            GeneratorExecutionContext context,
            ISymbol exportedMemberSymbol
        )
        {
            var locations = exportedMemberSymbol.Locations;
            var location = locations.FirstOrDefault(l => l.SourceTree != null) ?? locations.FirstOrDefault();
            bool isField = exportedMemberSymbol is IFieldSymbol;

            string message = $"Attempted to export static {(isField ? "field" : "property")}: " +
                             $"'{exportedMemberSymbol.ToDisplayString()}'";

            string description = $"{message}. Only instance fields and properties can be exported." +
                                 " Remove the 'static' modifier or the '[Export]' attribute.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GODOT-G0101",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description),
                location,
                location?.SourceTree?.FilePath));
        }

        public static void ReportExportedMemberTypeNotSupported(
            GeneratorExecutionContext context,
            ISymbol exportedMemberSymbol
        )
        {
            var locations = exportedMemberSymbol.Locations;
            var location = locations.FirstOrDefault(l => l.SourceTree != null) ?? locations.FirstOrDefault();
            bool isField = exportedMemberSymbol is IFieldSymbol;

            string message = $"The type of the exported {(isField ? "field" : "property")} " +
                             $"is not supported: '{exportedMemberSymbol.ToDisplayString()}'";

            string description = $"{message}. Use a supported type or remove the '[Export]' attribute.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GODOT-G0102",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description),
                location,
                location?.SourceTree?.FilePath));
        }

        public static void ReportExportedMemberIsReadOnly(
            GeneratorExecutionContext context,
            ISymbol exportedMemberSymbol
        )
        {
            var locations = exportedMemberSymbol.Locations;
            var location = locations.FirstOrDefault(l => l.SourceTree != null) ?? locations.FirstOrDefault();
            bool isField = exportedMemberSymbol is IFieldSymbol;

            string message = $"The exported {(isField ? "field" : "property")} " +
                             $"is read-only: '{exportedMemberSymbol.ToDisplayString()}'";

            string description = isField ?
                $"{message}. Exported fields cannot be read-only." :
                $"{message}. Exported properties must be writable.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GODOT-G0103",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description),
                location,
                location?.SourceTree?.FilePath));
        }

        public static void ReportExportedMemberIsWriteOnly(
            GeneratorExecutionContext context,
            ISymbol exportedMemberSymbol
        )
        {
            var locations = exportedMemberSymbol.Locations;
            var location = locations.FirstOrDefault(l => l.SourceTree != null) ?? locations.FirstOrDefault();

            string message = $"The exported property is write-only: '{exportedMemberSymbol.ToDisplayString()}'";

            string description = $"{message}. Exported properties must be readable.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GODOT-G0104",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description),
                location,
                location?.SourceTree?.FilePath));
        }

        public static void ReportSignalDelegateMissingSuffix(
            GeneratorExecutionContext context,
            INamedTypeSymbol delegateSymbol)
        {
            var locations = delegateSymbol.Locations;
            var location = locations.FirstOrDefault(l => l.SourceTree != null) ?? locations.FirstOrDefault();

            string message = "The name of the delegate must end with 'EventHandler': " +
                             delegateSymbol.ToDisplayString() +
                             $". Did you mean '{delegateSymbol.Name}EventHandler'?";

            string description = $"{message}. Rename the delegate accordingly or remove the '[Signal]' attribute.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GODOT-G0201",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description),
                location,
                location?.SourceTree?.FilePath));
        }

        public static void ReportSignalDelegateSignatureNotSupported(
            GeneratorExecutionContext context,
            INamedTypeSymbol delegateSymbol)
        {
            var locations = delegateSymbol.Locations;
            var location = locations.FirstOrDefault(l => l.SourceTree != null) ?? locations.FirstOrDefault();

            string message = "The delegate signature of the signal " +
                             $"is not supported: '{delegateSymbol.ToDisplayString()}'";

            string description = $"{message}. Use supported types only or remove the '[Signal]' attribute.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GODOT-G0202",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description),
                location,
                location?.SourceTree?.FilePath));
        }
    }
}
