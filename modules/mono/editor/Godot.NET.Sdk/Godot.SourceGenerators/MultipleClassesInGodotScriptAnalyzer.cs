using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Godot.SourceGenerators;

[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class MultipleClassesInGodotScriptAnalyzer : DiagnosticAnalyzer
{
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
        => ImmutableArray.Create(Common.MultipleClassesInGodotScriptRule);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSemanticModelAction(AnalyzeSemanticModel);
    }

    private static void AnalyzeSemanticModel(SemanticModelAnalysisContext context)
    {
        var semanticModel = context.SemanticModel;
        var syntaxTree = semanticModel.SyntaxTree;

        string fileName = System.IO.Path.GetFileNameWithoutExtension(syntaxTree.FilePath);

        if (string.IsNullOrEmpty(fileName))
            return;

        var matchingTypes = syntaxTree.GetRoot(context.CancellationToken)
            .DescendantNodes()
            .OfType<ClassDeclarationSyntax>()
            .Where(t => !t.IsNested() && t.Identifier.Text == fileName);

        var candidates = matchingTypes
            // Ignore nested classes.
            .Where(cds => !cds.IsNested())
            .SelectGodotScriptClasses(semanticModel)
            // Skip non-partial classes.
            .Where(x => x.cds.IsPartial())
            .GroupBy(x => x.symbol, SymbolEqualityComparer.Default)
            .Select(group => group.First());

        // Use the enumerator directly to avoid ToList() overhead.
        using var enumerator = candidates.GetEnumerator();

        // Try to get the first two items
        if (!enumerator.MoveNext())
            return; // Zero matches - nothing to report

        var first = enumerator.Current;

        if (!enumerator.MoveNext())
            return; // Exactly one match - nothing to report

        var second = enumerator.Current;

        // If we reach here, we have at least two. Report the first and second.
        Report(context, first.cds, fileName);
        Report(context, second.cds, fileName);

        // Continue reporting any remaining matches in the sequence.
        while (enumerator.MoveNext())
            Report(context, enumerator.Current.cds, fileName);

        return;

        static void Report(SemanticModelAnalysisContext context, TypeDeclarationSyntax type, string fileName)
        {
            context.ReportDiagnostic(Diagnostic.Create(
                Common.MultipleClassesInGodotScriptRule,
                type.Identifier.GetLocation(),
                fileName
            ));
        }
    }
}
