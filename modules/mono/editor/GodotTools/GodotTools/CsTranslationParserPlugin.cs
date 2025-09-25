using System.Collections.Generic;
using System.IO;
using System.Linq;
using Godot;
using Godot.Collections;
using GodotTools.Internals;
using Microsoft.Build.Evaluation;
using Microsoft.Build.Execution;
using Microsoft.Build.Locator;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace GodotTools;

public partial class CsTranslationParserPlugin : EditorTranslationParserPlugin
{

    private class CommentData
    {
        public string Comment = "";
        public int StartLine;
        public int EndLine;
        public bool Newline = true;
    }

    private List<MetadataReference>? _projectReferences;
    private Array<string[]> _ret = new Array<string[]>();
    private List<SyntaxTree> _syntaxTreeCaches = new List<SyntaxTree>();

    private const string TranslationCommentPrefix = "TRANSLATORS:";
    private const string NoTranslateComment = "NO_TRANSLATE";
    private const string TranslationStaticClass = "Godot.TranslationServer";
    private const string TranslationMethod = "Translate";
    private const string TranslationPluralMethod = "TranslatePlural";
    private const string TranslationClass = "Godot.GodotObject";
    private const string TranslationMethodTr = "Tr";
    private const string TranslationMethodTrN = "TrN";
    private static readonly string[] _configurations = ["Debug", "Release"];
    private static readonly string[] _targetPlatforms = ["windows", "linuxbsd", "macos", "android", "ios", "web"];

    public override string[] _GetRecognizedExtensions()
    {
        return ["cs"];
    }

    public override Array<string[]> _ParseFile(string path)
    {
        _ret = [];

        if (_projectReferences == null)
        {
            _projectReferences = new List<MetadataReference>();
            foreach (string configuration in _configurations)
            {
                foreach (string targetPlatform in _targetPlatforms)
                {
                    GetProjectReferences(GodotSharpDirs.ProjectCsProjPath, configuration, targetPlatform).ForEach(reference =>
                    {
                        if (!_projectReferences.Contains(reference))
                        {
                            _projectReferences.Add(reference);
                        }
                    });
                }
            }
            System.AppDomain.CurrentDomain.GetAssemblies()
                .Where(a => !a.IsDynamic)
                .Where(a => a.Location != "")
                .Select(a => MetadataReference.CreateFromFile(a.Location))
                .Cast<MetadataReference>()
                .ToList()
                .ForEach(reference =>
                {
                    if (!_projectReferences.Contains(reference))
                    {
                        _projectReferences.Add(reference);
                    }
                });
        }

        var res = ResourceLoader.Load<CSharpScript>(path, "Script");
        var text = res.SourceCode;

        foreach (string configuration in _configurations)
        {
            foreach (string targetPlatform in _targetPlatforms)
            {
                var symbols = GetProjectDefineConstants(GodotSharpDirs.ProjectCsProjPath, configuration, targetPlatform);
                ParseCode(text, symbols, _projectReferences);
            }
        }
        _syntaxTreeCaches.Clear();
        return _ret;
    }

    private void ParseCode(string code, string[] symbols, List<MetadataReference> references)
    {
        var options = new CSharpParseOptions(LanguageVersion.Default, DocumentationMode.Parse, SourceCodeKind.Script, symbols);
        var tree = CSharpSyntaxTree.ParseText(code, options);
        if (SyntaxTreeContains(tree) || tree == null)
        {
            return;
        }
        _syntaxTreeCaches.Add(tree);
        var compilation = CSharpCompilation.Create("TranslationParser", options: new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary))
            .AddReferences(references)
            .AddSyntaxTrees(tree);

        var semanticModel = compilation.GetSemanticModel(tree);
        var comments = tree.GetRoot().DescendantNodes()
            .SelectMany(
                node => node.GetTrailingTrivia()
                    .Where(trivia => trivia.IsKind(SyntaxKind.SingleLineCommentTrivia))
                    .Concat(node.GetLeadingTrivia().Where(trivia => trivia.IsKind(SyntaxKind.SingleLineCommentTrivia)
                                                                    || trivia.IsKind(SyntaxKind.MultiLineCommentTrivia))))
            .Select(trivia => new CommentData
            {
                Comment = trivia.ToFullString(),
                StartLine = GetStartLine(trivia.GetLocation()),
                EndLine = GetEndLine(trivia.GetLocation()),
                Newline = tree.GetRoot().DescendantNodes()
                    .FirstOrDefault(node => GetStartLine(node.GetLocation()) == GetStartLine(trivia.GetLocation())) == null
            })
            .ToArray();

        foreach (var syntaxNode in tree.GetRoot().DescendantNodes().Where(node => node is InvocationExpressionSyntax))
        {
            var invocation = (InvocationExpressionSyntax)syntaxNode;
            var commentText = "";
            var skip = false;
            // Parse inline comment
            var line = GetStartLine(syntaxNode.GetLocation());

            var commentData = comments.FirstOrDefault(comment => comment.StartLine == line);
            if (commentData != null)
            {
                commentText = commentData.Comment.TrimStart('/').Trim();
                if (commentText.StartsWith(TranslationCommentPrefix))
                {
                    commentText = commentText.TrimPrefix(TranslationCommentPrefix).Trim();
                }
                else if (commentText == NoTranslateComment || commentText.StartsWith(NoTranslateComment + ":"))
                {
                    skip = true;
                }
            }
            else
            {
                // Parse multiline comment
                for (var index = line - 1; index >= 0; index--)
                {
                    var multilineCommentData =
                        comments.FirstOrDefault(comment => comment.EndLine == index && comment.Newline);
                    if (multilineCommentData == null)
                    {
                        commentText = "";
                        break;
                    }
                    // multiline comment
                    if (multilineCommentData.StartLine != multilineCommentData.EndLine)
                    {
                        var multilineComments = multilineCommentData.Comment.TrimSuffix("*/").Trim().Split("\n")
                            .Select(lineStr => lineStr.TrimPrefix("/*").Trim().TrimPrefix(TranslationCommentPrefix));
                        commentText = string.Join("\n", multilineComments);
                        if (commentText == NoTranslateComment || commentText.StartsWith(NoTranslateComment + ":"))
                        {
                            commentText = "";
                            skip = true;
                        }
                        break;
                    }
                    // multiline single line comment
                    var currentComment = multilineCommentData.Comment.TrimStart('/').Trim();
                    if (currentComment == "")
                    {
                        continue;
                    }
                    if (commentText == "")
                    {
                        commentText = currentComment;
                    }
                    else
                    {
                        commentText = currentComment + "\n" + commentText;
                    }
                    if (currentComment.StartsWith(TranslationCommentPrefix))
                    {
                        commentText = commentText.TrimPrefix(TranslationCommentPrefix).Trim();
                        break;
                    }
                    if (currentComment == NoTranslateComment || currentComment.StartsWith(NoTranslateComment + ":"))
                    {
                        commentText = "";
                        skip = true;
                        break;
                    }
                }
            }

            SymbolInfo? symbolInfo = null;
            if (invocation.Expression is IdentifierNameSyntax identifierNameSyntax)
            {
                symbolInfo = semanticModel.GetSymbolInfo(identifierNameSyntax);
            }
            if (invocation.Expression is MemberAccessExpressionSyntax { Name: IdentifierNameSyntax nameSyntax })
            {
                symbolInfo = semanticModel.GetSymbolInfo(nameSyntax);
            }

            var methodSymbol = symbolInfo?.Symbol as IMethodSymbol;
            if (methodSymbol == null)
            {
                continue;
            }
            if (methodSymbol.Name == TranslationMethod &&
                methodSymbol.ContainingType.ToDisplayString() == TranslationStaticClass)
            {
                if (skip)
                {
                    continue;
                }
                AddMsg(invocation.ArgumentList.Arguments, semanticModel, commentText);
            }

            if (methodSymbol.Name == TranslationPluralMethod &&
                methodSymbol.ContainingType.ToDisplayString() == TranslationStaticClass)
            {
                if (skip)
                {
                    continue;
                }
                AddPluralMsg(invocation.ArgumentList.Arguments, semanticModel, commentText);
            }

            if (methodSymbol.Name is TranslationMethodTr or TranslationMethodTrN
                && methodSymbol.MethodKind == MethodKind.Ordinary)
            {
                var receiverType = methodSymbol.ReceiverType ?? methodSymbol.ContainingType;

                if (receiverType != null && InheritsFromGodotObject(receiverType))
                {
                    if (skip)
                    {
                        continue;
                    }
                    if (methodSymbol.Name == TranslationMethodTr)
                    {
                        AddMsg(invocation.ArgumentList.Arguments, semanticModel, commentText);
                    }
                    else
                    {
                        AddPluralMsg(invocation.ArgumentList.Arguments, semanticModel, commentText);
                    }
                }
            }
        }
    }

    private bool SyntaxTreeContains(SyntaxTree otherTree)
    {
        return _syntaxTreeCaches.Any(syntaxTree => syntaxTree.GetRoot().IsEquivalentTo(otherTree.GetRoot()));
    }
    private int GetStartLine(Location location)
    {
        return location.GetLineSpan().StartLinePosition.Line;
    }

    private int GetEndLine(Location location)
    {
        return location.GetLineSpan().EndLinePosition.Line;
    }

    private bool InheritsFromGodotObject(ITypeSymbol typeSymbol)
    {
        while (typeSymbol != null)
        {
            if (typeSymbol.ToDisplayString() == TranslationClass)
            {
                return true;
            }
#pragma warning disable CS8600
            typeSymbol = typeSymbol.BaseType;
#pragma warning restore CS8600
        }
        return false;
    }

    private void AddMsg(SeparatedSyntaxList<ArgumentSyntax> arguments, SemanticModel semanticModel, string comment)
    {
        switch (arguments.Count)
        {
            case 1:
            {
                var argExpr = arguments[0].Expression;
                var constantValue = semanticModel.GetConstantValue(argExpr);

                if (constantValue is { HasValue: true, Value: string message })
                {
                    _ret.Add([message, "", "", comment]);
                }

                break;
            }
            case 2:
            {
                var msgExpr = arguments[0].Expression;
                var ctxExpr = arguments[1].Expression;

                var msgValue = semanticModel.GetConstantValue(msgExpr);
                var ctxValue = semanticModel.GetConstantValue(ctxExpr);

                if (msgValue is { HasValue: true, Value: string message } &&
                    ctxValue is { HasValue: true, Value: string context })
                {
                    _ret.Add([message, context, "", comment]);
                }

                break;
            }
        }
    }

    private void AddPluralMsg(SeparatedSyntaxList<ArgumentSyntax> arguments, SemanticModel semanticModel, string comment)
    {
        var singularExpr = arguments[0].Expression;
        var pluralExpr = arguments[1].Expression;

        var singularValue = semanticModel.GetConstantValue(singularExpr);
        var pluralValue = semanticModel.GetConstantValue(pluralExpr);

        if (!singularValue.HasValue || singularValue.Value is not string singular ||
            !pluralValue.HasValue || pluralValue.Value is not string plural)
        {
            return;
        }

        var context = "";
        if (arguments.Count == 4)
        {
            var ctxExpr = arguments[3].Expression;
            var ctxValue = semanticModel.GetConstantValue(ctxExpr);
            if (ctxValue is { HasValue: true, Value: string ctx })
            {
                context = ctx;
            }
        }
        _ret.Add([singular, context, plural, comment]);
    }

    private List<MetadataReference> GetProjectReferences(string projectPath, string configuration = "Debug", string? targetPlatform = null)
    {
        if (!MSBuildLocator.IsRegistered)
        {
            MSBuildLocator.RegisterDefaults();
        }

        var referencePaths = GetProjectReferencePaths(projectPath, configuration, targetPlatform ?? OS.GetName());

        var metadataReferences = new List<MetadataReference>();
        foreach (var dllPath in referencePaths)
        {
            if (File.Exists(dllPath))
            {
                var metadataReference = MetadataReference.CreateFromFile(dllPath);
                metadataReferences.Add(metadataReference);
            }
        }

        return metadataReferences;
    }

    private List<string> GetProjectReferencePaths(string projectPath, string configuration, string targetPlatform)
    {
        var referencePaths = new List<string>();

        var projectCollection = new ProjectCollection();
        var project = projectCollection.LoadProject(projectPath);

        project.SetProperty("Configuration", configuration);
        project.SetProperty("Platform", "Any CPU");
        project.SetProperty("GodotTargetPlatform", targetPlatform);

        var buildParameters = new BuildParameters(projectCollection);
        var buildRequest = new BuildRequestData(project.FullPath, project.GlobalProperties, null, ["GetTargetPath"], null);
        var buildResult = BuildManager.DefaultBuildManager.Build(buildParameters, buildRequest);

        if (buildResult.OverallResult == BuildResultCode.Success)
        {
            referencePaths.AddRange(buildResult.ResultsByTarget["GetTargetPath"].Items.Select(item => item.ItemSpec));
        }

        projectCollection.UnloadAllProjects();
        projectCollection.Dispose();

        return referencePaths;
    }

    private string[] GetProjectDefineConstants(string projectPath, string configuration = "Debug", string? targetPlatform = null)
    {
        if (!MSBuildLocator.IsRegistered)
        {
            MSBuildLocator.RegisterDefaults();
        }
        string[] defineConstants = [];

        var projectCollection = new ProjectCollection();
        var project = projectCollection.LoadProject(projectPath);

        project.SetProperty("Configuration", configuration);
        project.SetProperty("Platform", "Any CPU");
        project.SetProperty("GodotTargetPlatform", targetPlatform ?? OS.GetName());

        var target = project.Xml.AddTarget("GetDefineConstants");
        var propertyGroup = target.AddPropertyGroup();
        propertyGroup.AddProperty("DefineConstantsValue", "$(DefineConstants)");
        var itemGroup = target.AddItemGroup();
        itemGroup.AddItem("DefineConstantsItem", "$(DefineConstantsValue)");
        var task = target.AddTask("WriteLinesToFile");
        var tempFilePath = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        task.SetParameter("File", tempFilePath);
        task.SetParameter("Lines", "@(DefineConstantsItem)");
        task.SetParameter("Overwrite", "true");

        var buildParameters = new BuildParameters(projectCollection);
        var buildRequest = new BuildRequestData(project.FullPath, project.GlobalProperties, null, ["GetDefineConstants"], null);
        var buildResult = BuildManager.DefaultBuildManager.Build(buildParameters, buildRequest);

        if (buildResult.OverallResult == BuildResultCode.Success)
        {
            var defineConstantsOutput = File.ReadAllText(tempFilePath);
            if (string.IsNullOrEmpty(defineConstantsOutput))
            {
                defineConstants = defineConstantsOutput.Split('\n')
                    .Select(symbol => symbol.Trim('\r').Trim('\n'))
                    .Where(defineConstant => defineConstant != "").ToArray();
            }
            File.Delete(tempFilePath);
        }

        projectCollection.UnloadAllProjects();
        projectCollection.Dispose();

        return defineConstants;
    }
}
