The directory `Files` contains C# files from the core assembly project that are not part of the generated API. Any file with the `.cs` extension in this directory will be added to the core assembly project.

A dummy solution and project is provided to get tooling help while editing these files, like code completion and name refactoring.

The directory `IgnoredFiles` contains C# files that are needed to build the dummy project but must not be added to the core assembly project. They contain placeholders for the declarations that are part of the generated API.
