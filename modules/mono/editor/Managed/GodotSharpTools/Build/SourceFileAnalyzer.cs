using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace GodotSharpTools.Build
{
    
    /// <summary>
    /// Helper class for Godot to extract class definitions from
    /// C# source files, if they extend from Godot.Object.
    /// </summary>
    public static class SourceFileAnalyzer
    {

        public static string FindTopLevelClassInFile(string originalPath)
        {
            var content = File.ReadAllText(originalPath);
            return FindTopLevelClass(content, originalPath);
        }

        /// <summary>
        /// Attempts to find the fully qualified name of a top level class in the given source code,
        /// if that class is 1) not generic and 2) not a nested class 3) inherits from some other class,
        /// and 4) has the same name as the file it is contained in.
        /// </summary>
        /// <param name="sourceCode"></param>
        /// <param name="originalPath"></param>
        /// <returns></returns>
        public static string FindTopLevelClass(string sourceCode, string originalPath)
        {
            var basename = Path.GetFileNameWithoutExtension(originalPath);

            var preprocessorSymbols = new []{"GODOT"};

            var options = new CSharpParseOptions(LanguageVersion.Latest, DocumentationMode.None, SourceCodeKind.Regular, preprocessorSymbols);

            var tree = CSharpSyntaxTree.ParseText(sourceCode, options, originalPath);
            
            var rootNode = tree.GetCompilationUnitRoot();
            
            // Visit all class declarations found in the file
            var classVisitor = new ClassDeclarationVisitor(basename);
            classVisitor.Visit(rootNode);

            var classes = classVisitor.TopLevelClasses;

            if (classes.Count == 0)
            {
                return null;
            }
            else if (classes.Count == 1)
            {
                return classes[0];
            }
            else
            {
                throw new ArgumentException(
                    $"Source file '{originalPath}' contains multiple top level classes: {string.Join(", ", classes)}");
            }

        }
    }

    internal class ClassDeclarationVisitor : CSharpSyntaxWalker
    {

        private readonly string fileBasename;

        private readonly List<string> prefixStack = new List<string>();

        public List<string> TopLevelClasses { get; } = new List<string>();

        public ClassDeclarationVisitor(string fileBasename)
        {
            this.fileBasename = fileBasename;
        }

        public override void VisitNamespaceDeclaration(NamespaceDeclarationSyntax node)
        {
            
            prefixStack.Add(node.Name + ".");
            DefaultVisit(node);
            prefixStack.RemoveAt(prefixStack.Count - 1);

        }

        public override void VisitClassDeclaration(ClassDeclarationSyntax node)
        {

            var className = node.Identifier.ValueText;
            // Only consider classes that have the same base name as the file
            if (className != fileBasename)
            {
                return;
            }

            // Skip generic classes
            if (node.TypeParameterList != null)
            {
                return;
            }
            // Skip classes without base classes
            if (node.BaseList == null)
            {
                return;
            }

            var fullClassName = string.Join("", prefixStack) + className;
            TopLevelClasses.Add(fullClassName);
        }

    }

}
