using System;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators;

[Generator]
public class RegisterScriptTypesGenerator : IIncrementalGenerator
{
    // ReSharper disable once StructCanBeMadeReadOnly
    private record struct Model(
        string Namespace,
        string FullQualifiedNameOmitGlobal,
        string DisplayStringMinimallyQualifiedFormat,
        GenericInfoModel? GenericInfoModel,
        ContainingTypeModel[]? ContainingTypeModels,
        string FilePath
    )
    {
        public readonly bool Equals(Model other)
            => FullQualifiedNameOmitGlobal == other.FullQualifiedNameOmitGlobal;

        public readonly override int GetHashCode()
            => FullQualifiedNameOmitGlobal.GetHashCode();
    }

    private record struct GenericInfoModel(
        string FullQualifiedNameUnboundGeneric
    );

    private record struct OptionsModel(
        string? GodotProjectDir,
        bool GodotRegisterScriptPaths,
        bool IsGodotToolsProject
    );

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var optionsProvider = context.AnalyzerConfigOptionsProvider
            .Select((options, _) =>
            {
                var isGodotToolsProject =
                    options.GlobalOptions.TryGetValue("build_property.IsGodotToolsProject",
                        out string? isGodotToolsProjectStr)
                    && isGodotToolsProjectStr.Equals("true", StringComparison.OrdinalIgnoreCase);

                var godotRegisterScriptPaths =
                    options.GlobalOptions.TryGetValue("build_property.GodotRegisterScriptPaths",
                        out string? godotRegisterScriptPathsStr)
                    && (
                        godotRegisterScriptPathsStr.Length == 0 // If empty, assuming true is less risky.
                        || godotRegisterScriptPathsStr.Equals("true", StringComparison.OrdinalIgnoreCase)
                    );

                if (!options.GlobalOptions.TryGetValue("build_property.GodotProjectDirBase64",
                        out string? godotProjectDir) || godotProjectDir.Length == 0)
                {
                    if (!options.GlobalOptions.TryGetValue("build_property.GodotProjectDir", out godotProjectDir) ||
                        godotProjectDir.Length == 0)
                    {
                        godotProjectDir = null;
                    }
                }
                else
                {
                    // Workaround for https://github.com/dotnet/roslyn/issues/51692
                    godotProjectDir = Encoding.UTF8.GetString(Convert.FromBase64String(godotProjectDir));
                }

                return new OptionsModel(
                    GodotProjectDir: godotProjectDir,
                    GodotRegisterScriptPaths: godotRegisterScriptPaths,
                    IsGodotToolsProject: isGodotToolsProject);
            });

        var modelsProvider = context.SyntaxProvider
            .CreateSyntaxProvider(
                predicate: (s, _) => s is ClassDeclarationSyntax { BaseList: not null },
                transform: (ctx, _) =>
                {
                    var symbol = Common.GetIfIsGodotScriptClass(ctx);

                    if (symbol == null)
                        return (Model?)null;

                    string filePath = ctx.Node.SyntaxTree.FilePath;

                    // The class to associate with the script path.
                    bool shouldRegisterScriptPath =
                        // Ignore nested classes.
                        !symbol.IsNested()
                        && symbol.GetAttributes().Any(a => a.AttributeClass?.IsGodotToolAttribute() ?? false)
                        // Ignore classes whose name is not the same as the file name.
                        && Path.GetFileNameWithoutExtension(filePath) == symbol.Name;

                    if (!shouldRegisterScriptPath)
                        return null;

                    return new Model(
                        Namespace: symbol.ContainingNamespace is { IsGlobalNamespace: false } namespaceSymbol
                            ? namespaceSymbol.FullQualifiedNameOmitGlobal()
                            : string.Empty,
                        FullQualifiedNameOmitGlobal: symbol.FullQualifiedNameOmitGlobal(),
                        DisplayStringMinimallyQualifiedFormat: symbol.ToDisplayString(SymbolDisplayFormat
                            .MinimallyQualifiedFormat),
                        GenericInfoModel: symbol.IsGenericType
                            ? new GenericInfoModel(
                                FullQualifiedNameUnboundGeneric: symbol
                                    .FullQualifiedNameUnboundGeneric(includeGlobal: false)
                            )
                            : null,
                        ContainingTypeModels: ContainingTypeModel.GetContainingTypesFor(symbol),
                        FilePath: ctx.Node.SyntaxTree.FilePath
                    );
                })
            .Where(model => model.HasValue)
            .Select((model, _) => model!.Value)
            // Avoid duplicates (because of partial).
            .Collect().SelectMany((items, _) => items.Distinct());

        var allModelsProvider = modelsProvider.Collect();
        var allModelsWithOptionsProvider = allModelsProvider.Combine(optionsProvider);

        // ScriptTypeMetaProviderAttribute
        // interface IScriptTypeMetaProvider { static abstract ScriptTypeMeta GetGodotClassScriptMeta(); }
        // EnableLegacyScriptTypeMetaResolver

        context.RegisterSourceOutput(allModelsWithOptionsProvider, (spc, allModelsWithOptions) =>
        {
            var allModels = allModelsWithOptions.Left;
            var optionsModel = allModelsWithOptions.Right;

            string uniqueHint = optionsModel.IsGodotToolsProject
                ? "GodotPlugins.Tools.Main.RegisterScriptTypes.g.cs"
                : "GodotPlugins.Game.Main.RegisterScriptTypes.g.cs";

            var sb = new StringBuilder();
            sb.Append(optionsModel.IsGodotToolsProject
                ? "namespace GodotPlugins.Tools;\n"
                : "namespace GodotPlugins.Game;\n");
            sb.Append("internal static partial class Main\n");
            sb.Append("{\n");

            string maybePartial = optionsModel.IsGodotToolsProject ? "" : " partial";

            sb.Append("    public static").Append(maybePartial).Append(" void RegisterScriptTypes()\n");
            sb.Append("    {\n");

            if (optionsModel.GodotRegisterScriptPaths)
            {
                if (optionsModel.GodotProjectDir == null)
                    throw new InvalidOperationException(
                        "Build property 'GodotProjectDir' is null or empty. " +
                        "Required by source generator when the GodotRegisterScriptPaths build property is not disabled.");

                foreach (var model in allModels)
                {
                    string scriptPath = Common.PathRelativeToDir(model.FilePath, optionsModel.GodotProjectDir);

                    if (model.GenericInfoModel is { } genericInfoModel)
                    {
                        sb.Append("        global::Godot.Bridge.ScriptManagerBridge")
                            .Append(".RegisterScriptPathForGenericTypeDefinition(typeof(global::")
                            .Append(genericInfoModel.FullQualifiedNameUnboundGeneric)
                            .Append("), \"res://").Append(scriptPath).Append("\");\n");
                    }
                    else
                    {
                        sb.Append("        global::Godot.Bridge.ScriptManagerBridge")
                            .Append(".RegisterScriptPathForType(typeof(global::")
                            .Append(model.FullQualifiedNameOmitGlobal)
                            .Append("), \"res://").Append(scriptPath).Append("\");\n");
                    }
                }
            }

            sb.Append("    }\n");
            sb.Append("}\n");

            spc.AddSource(uniqueHint, SourceText.From(sb.ToString(), Encoding.UTF8));
        });

        var modelsWithOptionsProvider = modelsProvider.Combine(optionsProvider);

        // ScriptPathAttribute
        context.RegisterSourceOutput(modelsWithOptionsProvider, (spc, modelsWithOptions) =>
        {
            var model = modelsWithOptions.Left;
            var optionsModel = modelsWithOptions.Right;

            if (!optionsModel.GodotRegisterScriptPaths)
                return;

            if (optionsModel.GodotProjectDir == null)
                throw new InvalidOperationException(
                    "Build property 'GodotProjectDir' is null or empty. " +
                    "Required by source generator when the GodotRegisterScriptPaths build property is not disabled.");

            bool hasNamespace = model.Namespace.Length != 0;

            string uniqueHint = model.FullQualifiedNameOmitGlobal.SanitizeQualifiedNameForUniqueHint()
                                + "_GetGodotClassScriptMeta.g.cs";

            var sb = new StringBuilder();

            if (hasNamespace)
            {
                sb.Append("namespace ");
                sb.Append(model.Namespace);
                sb.Append(";\n\n");
            }

            if (model.ContainingTypeModels != null)
            {
                foreach (var containingType in model.ContainingTypeModels)
                {
                    sb.Append("partial ");
                    sb.Append(containingType.DeclarationKeyword);
                    sb.Append(" ");
                    sb.Append(containingType.DisplayStringMinimallyQualifiedFormat);
                    sb.Append("\n{\n");
                }
            }

            string scriptPath = Common.PathRelativeToDir(model.FilePath, optionsModel.GodotProjectDir);

            sb.Append(@"[global::Godot.ScriptPathAttribute(""res://");
            sb.Append(scriptPath);
            sb.Append(@""")]");
            sb.Append("partial class ");
            sb.Append(model.DisplayStringMinimallyQualifiedFormat);
            sb.Append("\n{\n");
            sb.Append("}\n"); // partial class

            if (model.ContainingTypeModels != null)
            {
                foreach (var _ in model.ContainingTypeModels)
                    sb.Append("}\n"); // outer class
            }

            spc.AddSource(uniqueHint, SourceText.From(sb.ToString(), Encoding.UTF8));
        });
    }
}
