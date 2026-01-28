using System.Collections.Immutable;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CodeActions;
using Microsoft.CodeAnalysis.CodeFixes;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Godot.SourceGenerators
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class ClassPartialModifierAnalyzer : DiagnosticAnalyzer
    {
        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(Common.ClassPartialModifierRule, Common.OuterClassPartialModifierRule);

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(AnalyzeNode, SyntaxKind.ClassDeclaration);
        }

        private void AnalyzeNode(SyntaxNodeAnalysisContext context)
        {
            if (context.Node is not ClassDeclarationSyntax classDeclaration)
                return;

            if (context.ContainingSymbol is not INamedTypeSymbol typeSymbol)
                return;

            if (!typeSymbol.InheritsFrom("GodotSharp", GodotClasses.GodotObject))
                return;

            if (!classDeclaration.IsPartial())
                context.ReportDiagnostic(Diagnostic.Create(
                    Common.ClassPartialModifierRule,
                    classDeclaration.Identifier.GetLocation(),
                    typeSymbol.ToDisplayString()));

            var outerClassDeclaration = context.Node.Parent as ClassDeclarationSyntax;
            while (outerClassDeclaration is not null)
            {
                var outerClassTypeSymbol = context.SemanticModel.GetDeclaredSymbol(outerClassDeclaration);
                if (outerClassTypeSymbol == null)
                    return;

                if (!outerClassDeclaration.IsPartial())
                    context.ReportDiagnostic(Diagnostic.Create(
                        Common.OuterClassPartialModifierRule,
                        outerClassDeclaration.Identifier.GetLocation(),
                        outerClassTypeSymbol.ToDisplayString()));

                outerClassDeclaration = outerClassDeclaration.Parent as ClassDeclarationSyntax;
            }
        }
    }

    [ExportCodeFixProvider(LanguageNames.CSharp)]
    public sealed class ClassPartialModifierCodeFixProvider : CodeFixProvider
    {
        public override ImmutableArray<string> FixableDiagnosticIds =>
            ImmutableArray.Create(Common.ClassPartialModifierRule.Id);

        public override FixAllProvider GetFixAllProvider() => WellKnownFixAllProviders.BatchFixer;

        public override async Task RegisterCodeFixesAsync(CodeFixContext context)
        {
            // Get the syntax root of the document.
            var root = await context.Document.GetSyntaxRootAsync(context.CancellationToken).ConfigureAwait(false);

            // Get the diagnostic to fix.
            var diagnostic = context.Diagnostics.First();

            // Get the location of code issue.
            var diagnosticSpan = diagnostic.Location.SourceSpan;

            // Use that location to find the containing class declaration.
            var classDeclaration = root?.FindToken(diagnosticSpan.Start)
                .Parent?
                .AncestorsAndSelf()
                .OfType<ClassDeclarationSyntax>()
                .First();

            if (classDeclaration == null)
                return;

            context.RegisterCodeFix(
                CodeAction.Create(
                    "Add partial modifier",
                    cancellationToken => AddPartialModifierAsync(context.Document, classDeclaration, cancellationToken),
                    classDeclaration.ToFullString()),
                context.Diagnostics);
        }

        private static async Task<Document> AddPartialModifierAsync(Document document,
            ClassDeclarationSyntax classDeclaration, CancellationToken cancellationToken)
        {
            // Create a new partial modifier.
            var partialModifier = SyntaxFactory.Token(SyntaxKind.PartialKeyword);
            var modifiedClassDeclaration = classDeclaration.AddModifiers(partialModifier);
            var root = await document.GetSyntaxRootAsync(cancellationToken).ConfigureAwait(false);
            // Replace the old class declaration with the modified one in the syntax root.
            var newRoot = root!.ReplaceNode(classDeclaration, modifiedClassDeclaration);
            var newDocument = document.WithSyntaxRoot(newRoot);
            return newDocument;
        }
    }
}
