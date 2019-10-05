using System.Collections.Generic;
using System.IO;

namespace GodotTools.ProjectEditor
{
    public static class ApiSolutionGenerator
    {
        public static void GenerateApiSolution(string solutionDir,
            string coreProjDir, IEnumerable<string> coreCompileItems,
            string editorProjDir, IEnumerable<string> editorCompileItems)
        {
            var solution = new DotNetSolution(ApiAssemblyNames.SolutionName);

            solution.DirectoryPath = solutionDir;

            // GodotSharp project

            const string coreApiAssemblyName = ApiAssemblyNames.Core;

            string coreGuid = ProjectGenerator.GenCoreApiProject(coreProjDir, coreCompileItems);

            var coreProjInfo = new DotNetSolution.ProjectInfo
            {
                Guid = coreGuid,
                PathRelativeToSolution = Path.Combine(coreApiAssemblyName, $"{coreApiAssemblyName}.csproj")
            };
            coreProjInfo.Configs.Add("Debug");
            coreProjInfo.Configs.Add("Release");

            solution.AddNewProject(coreApiAssemblyName, coreProjInfo);

            // GodotSharpEditor project

            const string editorApiAssemblyName = ApiAssemblyNames.Editor;

            string editorGuid = ProjectGenerator.GenEditorApiProject(editorProjDir,
                $"../{coreApiAssemblyName}/{coreApiAssemblyName}.csproj", editorCompileItems);

            var editorProjInfo = new DotNetSolution.ProjectInfo();
            editorProjInfo.Guid = editorGuid;
            editorProjInfo.PathRelativeToSolution = Path.Combine(editorApiAssemblyName, $"{editorApiAssemblyName}.csproj");
            editorProjInfo.Configs.Add("Debug");
            editorProjInfo.Configs.Add("Release");

            solution.AddNewProject(editorApiAssemblyName, editorProjInfo);

            // Save solution

            solution.Save();
        }
    }
}
