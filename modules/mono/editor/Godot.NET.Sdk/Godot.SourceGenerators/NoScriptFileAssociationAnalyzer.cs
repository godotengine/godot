using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using System.Linq;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Godot.SourceGenerators;

[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class NoScriptFileAssociationAnalyzer : DiagnosticAnalyzer
{
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
        ImmutableArray.Create(
            Common.NoScriptFileAssociationRuleNotGodotObject,
            Common.NoScriptFileAssociationRuleConflictingAttributes
        );

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSymbolAction(AnalyzeNamedType, SymbolKind.NamedType);
    }

    private static void AnalyzeNamedType(SymbolAnalysisContext context)
    {
        var namedType = (INamedTypeSymbol)context.Symbol;
        var attributes = namedType.GetAttributes();

        if (!attributes.Any(a => a.AttributeClass is
            {
                Name: "NoScriptFileAssociationAttribute",
                ContainingNamespace: { Name: "Godot", ContainingNamespace.IsGlobalNamespace: true }
            }))
        {
            return;
        }

        if (!namedType.IsOrInheritsFrom("GodotSharp", GodotClasses.GodotObject))
        {
            context.ReportDiagnostic(Diagnostic.Create(
                Common.NoScriptFileAssociationRuleNotGodotObject,
                namedType.Locations[0],
                namedType.Name));
            return;
        }

        if (attributes.FirstOrDefault(a => a.AttributeClass is
            {
                Name: "ScriptPathAttribute",
                ContainingNamespace: { Name: "Godot", ContainingNamespace.IsGlobalNamespace: true }
            }) is { } scriptPathAttribute)
        {
            context.ReportDiagnostic(Diagnostic.Create(
                Common.NoScriptFileAssociationRuleConflictingAttributes,
                scriptPathAttribute.ApplicationSyntaxReference!.GetSyntax().GetLocation(),
                namedType.Name));
        }
    }
}
