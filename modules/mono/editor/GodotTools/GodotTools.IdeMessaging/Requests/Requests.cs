// ReSharper disable ClassNeverInstantiated.Global
// ReSharper disable UnusedMember.Global
// ReSharper disable UnusedAutoPropertyAccessor.Global

using Newtonsoft.Json;

namespace GodotTools.IdeMessaging.Requests
{
    public abstract class Request
    {
        [JsonIgnore] public string Id { get; }

        protected Request(string id)
        {
            Id = id;
        }
    }

    public abstract class Response
    {
        [JsonIgnore] public MessageStatus Status { get; set; } = MessageStatus.Ok;
    }

    public sealed class CodeCompletionRequest : Request
    {
        public enum CompletionKind
        {
            InputActions = 0,
            NodePaths,
            ResourcePaths,
            ScenePaths,
            ShaderParams,
            Signals,
            ThemeColors,
            ThemeConstants,
            ThemeFonts,
            ThemeStyles
        }

        public CompletionKind Kind { get; set; }
        public string ScriptFile { get; set; }

        public new const string Id = "CodeCompletion";

        public CodeCompletionRequest() : base(Id)
        {
        }
    }

    public sealed class CodeCompletionResponse : Response
    {
        public CodeCompletionRequest.CompletionKind Kind;
        public string ScriptFile { get; set; }
        public string[] Suggestions { get; set; }
    }

    public sealed class PlayRequest : Request
    {
        public new const string Id = "Play";

        public PlayRequest() : base(Id)
        {
        }
    }

    public sealed class PlayResponse : Response
    {
    }

    public sealed class StopPlayRequest : Request
    {
        public new const string Id = "StopPlay";

        public StopPlayRequest() : base(Id)
        {
        }
    }

    public sealed class StopPlayResponse : Response
    {
    }

    public sealed class DebugPlayRequest : Request
    {
        public string DebuggerHost { get; set; }
        public int DebuggerPort { get; set; }
        public bool? BuildBeforePlaying { get; set; }

        public new const string Id = "DebugPlay";

        public DebugPlayRequest() : base(Id)
        {
        }
    }

    public sealed class DebugPlayResponse : Response
    {
    }

    public sealed class OpenFileRequest : Request
    {
        public string File { get; set; }
        public int? Line { get; set; }
        public int? Column { get; set; }

        public new const string Id = "OpenFile";

        public OpenFileRequest() : base(Id)
        {
        }
    }

    public sealed class OpenFileResponse : Response
    {
    }

    public sealed class ReloadScriptsRequest : Request
    {
        public new const string Id = "ReloadScripts";

        public ReloadScriptsRequest() : base(Id)
        {
        }
    }

    public sealed class ReloadScriptsResponse : Response
    {
    }
}
