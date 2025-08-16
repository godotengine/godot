using System.Collections.Immutable;
using System.Linq;
using System.Threading;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Godot.SourceGenerators
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public class EventHandlerSuffixSuppressor : DiagnosticSuppressor
    {
        private static readonly SuppressionDescriptor _descriptor = new(
            id: "GDSP0001",
            suppressedDiagnosticId: "CA1711",
            justification: "Signal delegates are used in events so the naming follows the guidelines.");

        public override ImmutableArray<SuppressionDescriptor> SupportedSuppressions =>
            ImmutableArray.Create(_descriptor);

        public override void ReportSuppressions(SuppressionAnalysisContext context)
        {
            foreach (var diagnostic in context.ReportedDiagnostics)
            {
                AnalyzeDiagnostic(context, diagnostic, context.CancellationToken);
            }
        }

        private static void AnalyzeDiagnostic(SuppressionAnalysisContext context, Diagnostic diagnostic, CancellationToken cancellationToken = default)
        {
            var location = diagnostic.Location;
            var root = location.SourceTree?.GetRoot(cancellationToken);
            var dds = root?
                .FindNode(location.SourceSpan)
                .DescendantNodesAndSelf()
                .OfType<DelegateDeclarationSyntax>()
                .FirstOrDefault();

            if (dds == null)
                return;

            var semanticModel = context.GetSemanticModel(dds.SyntaxTree);
            var delegateSymbol = semanticModel.GetDeclaredSymbol(dds, cancellationToken);
            if (delegateSymbol == null)
                return;

            if (delegateSymbol.GetAttributes().Any(a => a.AttributeClass?.IsGodotSignalAttribute() ?? false))
            {
                context.ReportSuppression(Suppression.Create(_descriptor, diagnostic));
            }
        }
    }
}
