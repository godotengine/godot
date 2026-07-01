using System;
using System.Collections.Generic;
using System.Runtime.InteropServices.JavaScript;
using static Godot.GodotBridge;

namespace Godot
{
	public static partial class SignalManager
	{
		private static ulong? _singletonId;
		private static ulong SingletonId
		{
			get
			{
				if (_singletonId == null)
					_singletonId = Engine.get_object_singleton("CrossRuntimeEventSignal");
				return _singletonId.Value;
			}
		}

		private static readonly Dictionary<(ulong, string), List<Delegate>> _subs = new();

		[JSExport]
		internal static void Receive(double objectId, string signalName, double[] args)
		{
			var key = ((ulong)objectId, signalName);
			if (!_subs.TryGetValue(key, out var list))
			{
				return;
			}
			foreach (var d in list)
			{
				var paramTypes = d.Method.GetParameters();
				var invoke = new object[paramTypes.Length];
				for (int i = 0; i < paramTypes.Length && i < args.Length; i++)
				{
					var t = paramTypes[i].ParameterType;
					var id = (ulong)args[i];
					invoke[i] = Activator.CreateInstance(t, id);
				}
				d.DynamicInvoke(invoke);
			}
		}

		public static void Subscribe(ulong objectId, string signalName, Delegate handler)
		{
			var key = (objectId, signalName);
			if (!_subs.TryGetValue(key, out var list))
			{
				list = new List<Delegate>();
				_subs[key] = list;
			}
			list.Add(handler);

			var sid = SingletonId;
			if (sid == 0) throw new InvalidOperationException("CrossRuntimeEventSignal singleton not found");

			new GodotObject(SingletonId).Call("connect_signal", (double)(ulong)objectId, signalName);
		}

		public static void Unsubscribe(ulong objectId, string signalName, Delegate handler)
		{
			var key = (objectId, signalName);
			if (_subs.TryGetValue(key, out var list))
			{
				list.Remove(handler);
				if (list.Count == 0)
					_subs.Remove(key);
			}
			new GodotObject(SingletonId).Call("disconnect_signal", (double)(ulong)objectId, signalName);
		}
	}
}
