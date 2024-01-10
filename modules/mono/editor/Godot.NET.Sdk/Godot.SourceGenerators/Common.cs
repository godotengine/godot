using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Godot.SourceGenerators
{
    public static partial class Common
    {
        private static readonly string _helpLinkFormat = $"{VersionDocsUrl}/tutorials/scripting/c_sharp/diagnostics/{{0}}.html";

        public static void ReportNonPartialGodotScriptClass(
            GeneratorExecutionContext context,
            ClassDeclarationSyntax cds, INamedTypeSymbol symbol
        )
        {
            string message =
                "Missing partial modifier on declaration of type '" +
                $"{symbol.FullQualifiedNameOmitGlobal()}' that derives from '{GodotClasses.GodotObject}'";

            string description = $"{message}. Classes that derive from '{GodotClasses.GodotObject}' " +
                                 "must be declared with the partial modifier.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GD0001",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0001")),
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
                namedTypeSymbol.FullQualifiedNameOmitGlobal() :
                "type not found";

            string message =
                $"Missing partial modifier on declaration of type '{fullQualifiedName}', " +
                $"which contains nested classes that derive from '{GodotClasses.GodotObject}'";

            string description = $"{message}. Classes that derive from '{GodotClasses.GodotObject}' and their " +
                                 "containing types must be declared with the partial modifier.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GD0002",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0002")),
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
                new DiagnosticDescriptor(id: "GD0101",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0101")),
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
                new DiagnosticDescriptor(id: "GD0102",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0102")),
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
                new DiagnosticDescriptor(id: "GD0103",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0103")),
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
                new DiagnosticDescriptor(id: "GD0104",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0104")),
                location,
                location?.SourceTree?.FilePath));
        }

        public static void ReportExportedMemberIsIndexer(
            GeneratorExecutionContext context,
            ISymbol exportedMemberSymbol
        )
        {
            var locations = exportedMemberSymbol.Locations;
            var location = locations.FirstOrDefault(l => l.SourceTree != null) ?? locations.FirstOrDefault();

            string message = $"Attempted to export indexer property: " +
                             $"'{exportedMemberSymbol.ToDisplayString()}'";

            string description = $"{message}. Indexer properties can't be exported." +
                                 " Remove the '[Export]' attribute.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GD0105",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0105")),
                location,
                location?.SourceTree?.FilePath));
        }

        public static void ReportExportedMemberIsExplicitInterfaceImplementation(
            GeneratorExecutionContext context,
            ISymbol exportedMemberSymbol
        )
        {
            var locations = exportedMemberSymbol.Locations;
            var location = locations.FirstOrDefault(l => l.SourceTree != null) ?? locations.FirstOrDefault();

            string message = $"Attempted to export explicit interface property implementation: " +
                             $"'{exportedMemberSymbol.ToDisplayString()}'";

            string description = $"{message}. Explicit interface implementations can't be exported." +
                                 " Remove the '[Export]' attribute.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GD0106",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0106")),
                location,
                location?.SourceTree?.FilePath));
        }

        public static void ReportOnlyNodesShouldExportNodes(
            GeneratorExecutionContext context,
            ISymbol exportedMemberSymbol
        )
        {
            var locations = exportedMemberSymbol.Locations;
            var location = locations.FirstOrDefault(l => l.SourceTree != null) ?? locations.FirstOrDefault();
            bool isField = exportedMemberSymbol is IFieldSymbol;

            string message = $"Types not derived from Node should not export Node {(isField ? "fields" : "properties")}";

            string description = $"{message}. Node export is only supported in Node-derived classes.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GD0107",
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
                new DiagnosticDescriptor(id: "GD0201",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0201")),
                location,
                location?.SourceTree?.FilePath));
        }

        public static void ReportSignalParameterTypeNotSupported(
            GeneratorExecutionContext context,
            IParameterSymbol parameterSymbol)
        {
            var locations = parameterSymbol.Locations;
            var location = locations.FirstOrDefault(l => l.SourceTree != null) ?? locations.FirstOrDefault();

            string message = "The parameter of the delegate signature of the signal " +
                             $"is not supported: '{parameterSymbol.ToDisplayString()}'";

            string description = $"{message}. Use supported types only or remove the '[Signal]' attribute.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GD0202",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0202")),
                location,
                location?.SourceTree?.FilePath));
        }

        public static void ReportSignalDelegateSignatureMustReturnVoid(
            GeneratorExecutionContext context,
            INamedTypeSymbol delegateSymbol)
        {
            var locations = delegateSymbol.Locations;
            var location = locations.FirstOrDefault(l => l.SourceTree != null) ?? locations.FirstOrDefault();

            string message = "The delegate signature of the signal " +
                             $"must return void: '{delegateSymbol.ToDisplayString()}'";

            string description = $"{message}. Return void or remove the '[Signal]' attribute.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GD0203",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0203")),
                location,
                location?.SourceTree?.FilePath));
        }

        public static readonly DiagnosticDescriptor GenericTypeArgumentMustBeVariantRule =
            new DiagnosticDescriptor(id: "GD0301",
                title: "The generic type argument must be a Variant compatible type",
                messageFormat: "The generic type argument must be a Variant compatible type: {0}",
                category: "Usage",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true,
                "The generic type argument must be a Variant compatible type. Use a Variant compatible type as the generic type argument.",
                helpLinkUri: string.Format(_helpLinkFormat, "GD0301"));

        public static void ReportGenericTypeArgumentMustBeVariant(
            SyntaxNodeAnalysisContext context,
            SyntaxNode typeArgumentSyntax,
            ISymbol typeArgumentSymbol)
        {
            string message = "The generic type argument " +
                            $"must be a Variant compatible type: '{typeArgumentSymbol.ToDisplayString()}'";

            string description = $"{message}. Use a Variant compatible type as the generic type argument.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GD0301",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0301")),
                typeArgumentSyntax.GetLocation(),
                typeArgumentSyntax.SyntaxTree.FilePath));
        }

        public static readonly DiagnosticDescriptor GenericTypeParameterMustBeVariantAnnotatedRule =
            new DiagnosticDescriptor(id: "GD0302",
                title: "The generic type parameter must be annotated with the MustBeVariant attribute",
                messageFormat: "The generic type argument must be a Variant type: {0}",
                category: "Usage",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true,
                "The generic type argument must be a Variant type. Use a Variant type as the generic type argument.",
                helpLinkUri: string.Format(_helpLinkFormat, "GD0302"));

        public static void ReportGenericTypeParameterMustBeVariantAnnotated(
            SyntaxNodeAnalysisContext context,
            SyntaxNode typeArgumentSyntax,
            ISymbol typeArgumentSymbol)
        {
            string message = "The generic type parameter must be annotated with the MustBeVariant attribute";

            string description = $"{message}. Add the MustBeVariant attribute to the generic type parameter.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GD0302",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0302")),
                typeArgumentSyntax.GetLocation(),
                typeArgumentSyntax.SyntaxTree.FilePath));
        }

        public static readonly DiagnosticDescriptor TypeArgumentParentSymbolUnhandledRule =
            new DiagnosticDescriptor(id: "GD0303",
                title: "The generic type parameter must be annotated with the MustBeVariant attribute",
                messageFormat: "The generic type argument must be a Variant type: {0}",
                category: "Usage",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true,
                "The generic type argument must be a Variant type. Use a Variant type as the generic type argument.",
                helpLinkUri: string.Format(_helpLinkFormat, "GD0303"));

        public static void ReportTypeArgumentParentSymbolUnhandled(
            SyntaxNodeAnalysisContext context,
            SyntaxNode typeArgumentSyntax,
            ISymbol parentSymbol)
        {
            string message = $"Symbol '{parentSymbol.ToDisplayString()}' parent of a type argument " +
                             "that must be Variant compatible was not handled.";

            string description = $"{message}. Handle type arguments that are children of the unhandled symbol type.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GD0303",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0303")),
                typeArgumentSyntax.GetLocation(),
                typeArgumentSyntax.SyntaxTree.FilePath));
        }

        public static readonly DiagnosticDescriptor GlobalClassMustDeriveFromGodotObjectRule =
            new DiagnosticDescriptor(id: "GD0401",
                title: "The class must derive from GodotObject or a derived class",
                messageFormat: "The class '{0}' must derive from GodotObject or a derived class",
                category: "Usage",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true,
                "The class must derive from GodotObject or a derived class. Change the base class or remove the '[GlobalClass]' attribute.",
                helpLinkUri: string.Format(_helpLinkFormat, "GD0401"));

        public static void ReportGlobalClassMustDeriveFromGodotObject(
            SyntaxNodeAnalysisContext context,
            SyntaxNode classSyntax,
            ISymbol typeSymbol)
        {
            string message = $"The class '{typeSymbol.ToDisplayString()}' must derive from GodotObject or a derived class";

            string description = $"{message}. Change the base class or remove the '[GlobalClass]' attribute.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GD0401",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0401")),
                classSyntax.GetLocation(),
                classSyntax.SyntaxTree.FilePath));
        }

        public static readonly DiagnosticDescriptor GlobalClassMustNotBeGenericRule =
            new DiagnosticDescriptor(id: "GD0402",
                title: "The class must not contain generic arguments",
                messageFormat: "The class '{0}' must not contain generic arguments",
                category: "Usage",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true,
                "The class must be a non-generic type. Remove the generic arguments or the '[GlobalClass]' attribute.",
                helpLinkUri: string.Format(_helpLinkFormat, "GD0401"));

        public static void ReportGlobalClassMustNotBeGeneric(
            SyntaxNodeAnalysisContext context,
            SyntaxNode classSyntax,
            ISymbol typeSymbol)
        {
            string message = $"The class '{typeSymbol.ToDisplayString()}' must not contain generic arguments";

            string description = $"{message}. Remove the generic arguments or the '[GlobalClass]' attribute.";

            context.ReportDiagnostic(Diagnostic.Create(
                new DiagnosticDescriptor(id: "GD0402",
                    title: message,
                    messageFormat: message,
                    category: "Usage",
                    DiagnosticSeverity.Error,
                    isEnabledByDefault: true,
                    description,
                    helpLinkUri: string.Format(_helpLinkFormat, "GD0402")),
                classSyntax.GetLocation(),
                classSyntax.SyntaxTree.FilePath));
        }
    }
}
