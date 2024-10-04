using System.Collections.Immutable;
using System.Threading;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators;

public class GeneratorExecutionContext : IGeneratorExecutionContext
{
    private readonly Microsoft.CodeAnalysis.GeneratorExecutionContext _generatorExecutionContext;

    public GeneratorExecutionContext(Microsoft.CodeAnalysis.GeneratorExecutionContext generatorExecutionContext)
    {
        _generatorExecutionContext = generatorExecutionContext;
    }

    public Compilation Compilation => _generatorExecutionContext.Compilation;

    public ParseOptions ParseOptions => _generatorExecutionContext.ParseOptions;

    public ImmutableArray<AdditionalText> AdditionalFiles => _generatorExecutionContext.AdditionalFiles;

    public AnalyzerConfigOptionsProvider AnalyzerConfigOptions => _generatorExecutionContext.AnalyzerConfigOptions;

    public ISyntaxReceiver? SyntaxReceiver => _generatorExecutionContext.SyntaxReceiver;

    public ISyntaxContextReceiver? SyntaxContextReceiver => _generatorExecutionContext.SyntaxContextReceiver;

    public CancellationToken CancellationToken => _generatorExecutionContext.CancellationToken;

    public void AddSource(string hintName, string source)
    {
        _generatorExecutionContext.AddSource(hintName, source);
    }

    public void AddSource(string hintName, SourceText sourceText)
    {
        _generatorExecutionContext.AddSource(hintName, sourceText);
    }

    public void ReportDiagnostic(Diagnostic diagnostic)
    {
        _generatorExecutionContext.ReportDiagnostic(diagnostic);
    }
}
