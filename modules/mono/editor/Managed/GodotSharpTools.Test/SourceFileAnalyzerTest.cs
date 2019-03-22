using System;
using GodotSharpTools.Build;
using NUnit.Framework;

namespace GodotSharpToolsTest
{
    public class SourceFileAnalyzerTest
    {

        [Test]
        public void TestPlainClasses()
        {

            var code = @"class MyClass : System.Object {}";
            Assert.AreEqual("MyClass", SourceFileAnalyzer.FindTopLevelClass(code, "IgnoreDir/MyClass.cs"));
            
            code = @"namespace ABC { class MyClass : System.Object {} }";
            Assert.AreEqual("ABC.MyClass", SourceFileAnalyzer.FindTopLevelClass(code, "MyClass.cs"));

            code = @"namespace ABC.QWD { class MyClass : System.Object {} }";
            Assert.AreEqual("ABC.QWD.MyClass", SourceFileAnalyzer.FindTopLevelClass(code, "MyClass.cs"));

            code = @"namespace ABC.QWD { namespace XXX { class MyClass : System.Object {} } }";
            Assert.AreEqual("ABC.QWD.XXX.MyClass", SourceFileAnalyzer.FindTopLevelClass(code, "MyClass.cs"));

        }

        [Test]
        public void TestSkipGenericClasses()
        {
            var code = @"class MyClass<T> : System.Object {}";
            Assert.IsNull(SourceFileAnalyzer.FindTopLevelClass(code, "MyClass.cs"));
        }

        [Test]
        public void TestIncludeOnlyClassesMatchingFilename()
        {
            var code = @"class OtherClass : System.Object {} class MyClass : System.Object {}";
            Assert.AreEqual("MyClass", SourceFileAnalyzer.FindTopLevelClass(code, "MyClass.cs"));
        }

        [Test]
        public void TestFailOnAmbiguity()
        {
            var e = Assert.Throws<ArgumentException>(() =>
            {
                var code = @"class MyClass : System.Object {} namespace X { class MyClass : System.Object {} }";
                SourceFileAnalyzer.FindTopLevelClass(code, "MyClass.cs");
            });

            Assert.AreEqual(e.Message, "Source file 'MyClass.cs' contains multiple top level classes: MyClass, X.MyClass");

        }
        
        [Test]
        public void TestWithPreprocessorDefines()
        {
            var code = @"#if !GODOT
class MyClass {
#else
class MyClass : System.Object {
#endif
}
";
            Assert.AreEqual("MyClass", SourceFileAnalyzer.FindTopLevelClass(code, "MyClass.cs"));
        }

        [Test]
        public void TestSyntaxErrors()
        {
            var code = "class MyClass : System.Object { { xxx\n some junk";
            Assert.AreEqual("MyClass", SourceFileAnalyzer.FindTopLevelClass(code, "MyClass.cs"));
        }
        
    }
}
