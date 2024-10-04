using System.Collections.Immutable;
using System.Threading;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators;

/// <summary>
/// Allows consumers of Godot.SourceGenerators.Implementation to pass revised compilations.
/// This is not possible with <see cref="Microsoft.CodeAnalysis.GeneratorExecutionContext"/> as its constructor is internal.
/// </summary>
public interface IGeneratorExecutionContext
{
    public Compilation Compilation { get; }

    public ParseOptions ParseOptions { get; }

    public ImmutableArray<AdditionalText> AdditionalFiles { get; }

    public AnalyzerConfigOptionsProvider AnalyzerConfigOptions { get; }

    public ISyntaxReceiver? SyntaxReceiver { get; }

    public ISyntaxContextReceiver? SyntaxContextReceiver { get; }

    public CancellationToken CancellationToken { get; }

    public void AddSource(string hintName, string source);

    public void AddSource(string hintName, SourceText sourceText);

    public void ReportDiagnostic(Diagnostic diagnostic);
}
