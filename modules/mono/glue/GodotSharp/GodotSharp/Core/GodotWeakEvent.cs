using System;
using System.Collections.Generic;
using System.Reflection;

#nullable enable

namespace Godot
{
    internal readonly record struct GodotWeakEventSubscription(WeakReference? Target, MethodInfo Handler);

    /// <summary>
    /// Represents a weak event that can be subscribed to. If the target of the
    /// subscription handler is garbage collected, the event will eventually remove
    /// the subscription and avoids raising the event for collected targets.
    /// </summary>
    public class GodotWeakEvent<TEventHandler> where TEventHandler : Delegate
    {
        internal List<GodotWeakEventSubscription> Subscriptions { get; } = new();

        /// <summary>
        /// Subscribe to the event adding a <paramref name="handler"/> as callback
        /// that will be invoked when the event is raised.
        /// </summary>
        /// <param name="handler">The callback to invoke when the event is raised.</param>
        public void AddEventHandler(TEventHandler handler)
        {
            if (handler == null)
                throw new ArgumentNullException(nameof(handler));

            AddEventHandlerCore(handler.Target, handler.GetMethodInfo());
        }

        private void AddEventHandlerCore(object? handlerTarget, MethodInfo methodInfo)
        {
            WeakReference? target = handlerTarget != null ? new WeakReference(handlerTarget) : null;
            Subscriptions.Add(new GodotWeakEventSubscription(target, methodInfo));
        }

        /// <summary>
        /// Unsubscribes from the event by removing a <paramref name="handler"/>
        /// that has been previously added as callback to be invoked when the event
        /// is raised.
        /// If the <paramref name="handler"/> is not found in the subscriptions,
        /// it does nothing.
        /// </summary>
        /// <param name="handler">The callback to remove from the subscriptions.</param>
        public void RemoveEventHandler(TEventHandler handler)
        {
            if (handler == null)
                throw new ArgumentNullException(nameof(handler));

            RemoveEventHandlerCore(handler.Target, handler.GetMethodInfo());
        }

        private void RemoveEventHandlerCore(object? handlerTarget, MethodInfo methodInfo)
        {
            for (int i = Subscriptions.Count - 1; i >= 0; i--)
            {
                var subscription = Subscriptions[i];

                if (!ObjectIsAlive(subscription.Target))
                {
                    // Found a subscription with a disposed target.
                    Subscriptions.RemoveAt(i);
                    continue;
                }

                if (subscription.Target?.Target == handlerTarget && subscription.Handler.Name == methodInfo.Name)
                {
                    // Found the subscription.
                    Subscriptions.RemoveAt(i);
                    break;
                }
            }
        }

        /// <summary>
        /// Raises the event, invoking the handlers of all current subscriptions.
        /// Subscriptions with a collected target are ignored and removed.
        /// </summary>
        /// <param name="args">Parameters to pass to the invoked methods.</param>
        public void RaiseEvent(params object?[]? args)
        {
            for (int i = Subscriptions.Count - 1; i >= 0; i--)
            {
                var subscription = Subscriptions[i];

                // If the target is null, the handler is a static method.
                if (subscription.Target == null)
                {
                    RaiseEventCore(null, subscription.Handler, args);
                    continue;
                }

                object? target = subscription.Target.Target;
                if (!ObjectIsAlive(subscription.Target))
                {
                    // Found a subscription with a disposed target.
                    Subscriptions.RemoveAt(i);
                }
                else
                {
                    RaiseEventCore(target, subscription.Handler, args);
                }
            }

            static void RaiseEventCore(object? target, MethodInfo methodInfo, object?[]? args)
            {
                methodInfo.Invoke(target, args);
            }
        }

        private static bool ObjectIsAlive(WeakReference? obj)
        {
            // If the reference is null, there is no target because the handler is a static method.
            if (obj == null)
            {
                return true;
            }

            if (obj is { IsAlive: false })
            {
                return false;
            }

            // If the target is a GodotObject, the instance may not be valid even if the C# object is alive.
            if (obj.Target is GodotObject godotObject && !GodotObject.IsInstanceValid(godotObject))
            {
                return false;
            }

            return true;
        }
    }
}
