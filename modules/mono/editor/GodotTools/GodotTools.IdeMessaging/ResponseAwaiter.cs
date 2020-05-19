using GodotTools.IdeMessaging.Requests;
using GodotTools.IdeMessaging.Utils;
using Newtonsoft.Json;

namespace GodotTools.IdeMessaging
{
    public abstract class ResponseAwaiter : NotifyAwaiter<Response>
    {
        public abstract void SetResult(MessageContent content);
    }

    public class ResponseAwaiter<T> : ResponseAwaiter
        where T : Response, new()
    {
        public override void SetResult(MessageContent content)
        {
            if (content.Status == MessageStatus.Ok)
                SetResult(JsonConvert.DeserializeObject<T>(content.Body));
            else
                SetResult(new T {Status = content.Status});
        }
    }
}
