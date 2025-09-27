using System.Collections.Generic;
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
using Microsoft.CodeAnalysis.Formatting;
using Microsoft.CodeAnalysis.Operations;
using Microsoft.CodeAnalysis.Simplification;

using static Microsoft.CodeAnalysis.CSharp.SyntaxFactory;

namespace Godot.SourceGenerators;

[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class StringNameAnalyzer : DiagnosticAnalyzer
{
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } =
        ImmutableArray.Create(Common.ImplicitStringNameShouldNotBeUsedRule);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterOperationAction(AnalyzeOperation, OperationKind.Conversion);
    }

    private void AnalyzeOperation(OperationAnalysisContext context)
    {
        // this is a conversion operation
        if (context.Operation is not IConversionOperation { } conversionOperation)
            return;

        // filter by implicit conversions only
        if (!conversionOperation.IsImplicit)
            return;

        // filter by only the StringName(string) implicit operator
        if (conversionOperation.Conversion.MethodSymbol?.ReturnType?.InheritsFrom("GodotSharp", GodotClasses.StringName) is not true
            || conversionOperation.Conversion.MethodSymbol.Parameters.Length != 1
            || conversionOperation.Conversion.MethodSymbol.Parameters[0].Type?.SpecialType is not SpecialType.System_String)
        {
            return;
        }

        context.ReportDiagnostic(Diagnostic.Create(
            Common.ImplicitStringNameShouldNotBeUsedRule,
            conversionOperation.Syntax.GetLocation(),
            conversionOperation.Conversion.MethodSymbol.ToDisplayString()
        ));
    }
}

[ExportCodeFixProvider(LanguageNames.CSharp)]
public sealed class StringNameCodeFixProvider : CodeFixProvider
{
    public override ImmutableArray<string> FixableDiagnosticIds { get; } =
        ImmutableArray.Create(Common.ImplicitStringNameShouldNotBeUsedRule.Id);

    public override FixAllProvider? GetFixAllProvider() => WellKnownFixAllProviders.BatchFixer;

    public override async Task RegisterCodeFixesAsync(CodeFixContext context)
    {
        if (await context.Document.GetSyntaxRootAsync(context.CancellationToken).ConfigureAwait(false) is not { } root)
            return;

        var diagnostic = context.Diagnostics.First();
        var diagnosticSpan = diagnostic.Location.SourceSpan;

        // find the full syntax node highlighted to get the string name source
        if (root.FindNode(diagnosticSpan, findInsideTrivia: true) is not ArgumentSyntax { } stringNameValueArgumentSyntax
            || stringNameValueArgumentSyntax.Expression is not { } stringNameValueSyntax)
        {
            return;
        }

        // don't register a code fix for non-constant values
        if (await context.Document.GetSemanticModelAsync(context.CancellationToken).ConfigureAwait(false) is not { } semanticModel)
            return;
        if (semanticModel.GetConstantValue(stringNameValueSyntax, context.CancellationToken) is not { HasValue: true, Value: string { } })
            return;

        context.RegisterCodeFix(
            CodeAction.Create(
                "Cache the StringName instance between calls",
                ct => CacheStringNameAsync(context.Document, stringNameValueSyntax, ct),
                nameof(StringNameCodeFixProvider)),
            context.Diagnostics);
    }

    private static async Task<Document> CacheStringNameAsync(Document document,
        ExpressionSyntax stringNameValueSyntaxNode, CancellationToken ct)
    {
        // get the syntax root and semantic model
        if (await document.GetSyntaxRootAsync(ct).ConfigureAwait(false) is not { } root)
            return document;
        if (await document.GetSemanticModelAsync(ct).ConfigureAwait(false) is not { } semanticModel)
            return document;

        // compute the constant StringName value
        if (semanticModel.GetConstantValue(stringNameValueSyntaxNode, ct) is not { HasValue: true, Value: string { } stringNameConstantValue })
            return document;
        var stringNameFieldName = stringNameConstantValue.UnderscoreToCamelCaseIdentifierName()! + "StringName";

        // find the innermost class/struct/record declaration containing our StringName value, if any
        if (stringNameValueSyntaxNode.Ancestors().OfType<TypeDeclarationSyntax>().FirstOrDefault() is not { } typeDeclaration)
            return document;

        // build a field declaration for the static readonly StringName instance, if it doesn't already exist
        var fieldDeclarationAlreadyExists = typeDeclaration.Members.Any(m =>
            m is FieldDeclarationSyntax { Declaration: VariableDeclarationSyntax { Variables: { } variables } variableDeclarationSyntax }
            && variables.Any(variable => variable.Identifier.Text == stringNameFieldName)
            && semanticModel.GetTypeInfo(variableDeclarationSyntax.Type, ct).Type?.FullQualifiedNameOmitGlobal() == GodotClasses.StringName);
        var fieldDeclaration = fieldDeclarationAlreadyExists ? null : FieldDeclaration(VariableDeclaration(ParseTypeName(GodotClasses.StringName))
            .AddVariables(VariableDeclarator(stringNameFieldName)
                .WithInitializer(EqualsValueClause(ImplicitObjectCreationExpression()
                    .AddArgumentListArguments(Argument(LiteralExpression(SyntaxKind.StringLiteralExpression, Literal(stringNameConstantValue))))))))
            .AddModifiers(Token(SyntaxKind.PrivateKeyword), Token(SyntaxKind.StaticKeyword), Token(SyntaxKind.ReadOnlyKeyword));

        // replace every instance of the StringName string with our new variable
        Dictionary<SyntaxNode, SyntaxNode> replacements = new();
        foreach (var argumentSyntaxCandidate in typeDeclaration.DescendantNodes().OfType<ArgumentSyntax>())
            if (argumentSyntaxCandidate.Expression is not null
                && semanticModel.GetConstantValue(argumentSyntaxCandidate.Expression, ct) is { HasValue: true, Value: string { } argumentSyntaxCandidateValue }
                && argumentSyntaxCandidateValue == stringNameConstantValue
                && semanticModel.GetTypeInfo(argumentSyntaxCandidate.Expression) is { } argumentSyntaxCandidateTypeInfo
                && argumentSyntaxCandidateTypeInfo.Type?.SpecialType == SpecialType.System_String                                   // from string
                && argumentSyntaxCandidateTypeInfo.ConvertedType?.FullQualifiedNameOmitGlobal() == GodotClasses.StringName)         // to StringName
            {
                replacements[argumentSyntaxCandidate.Expression] = IdentifierName(stringNameFieldName);
            }
        root = root.ReplaceNodes(replacements.Keys, (originalNode, rewrittenNode) => replacements.TryGetValue(originalNode, out var newNode) ? newNode : rewrittenNode);

        // find the equivalent of typeDeclaration in the new root
        if (root.DescendantNodesAndSelf().OfType<TypeDeclarationSyntax>()
            .FirstOrDefault(td => td.Identifier.Text == typeDeclaration.Identifier.Text) is not { } newRootTypeDeclaration)
        {
            return document;
        }

        if (!fieldDeclarationAlreadyExists)
        {
            // build a new root with the field declaration inserted (before any methods)
            var insertIndex = newRootTypeDeclaration.Members.IndexOf(static mds => mds.Kind() is SyntaxKind.ConstructorDeclaration or SyntaxKind.DestructorDeclaration or SyntaxKind.IndexerDeclaration
                or SyntaxKind.MethodDeclaration or SyntaxKind.RecordDeclaration or SyntaxKind.RecordStructDeclaration);
            var newTypeDeclaration = newRootTypeDeclaration.WithMembers(newRootTypeDeclaration.Members.Insert(insertIndex, fieldDeclaration!));
            root = root.ReplaceNode(newRootTypeDeclaration, newTypeDeclaration);
        }

        return document.WithSyntaxRoot(root.WithAdditionalAnnotations(Formatter.Annotation, Simplifier.Annotation));
    }
}
