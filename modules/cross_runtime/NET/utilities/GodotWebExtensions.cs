using System;
#nullable enable

namespace Godot
{
	// GodotExtensions.cs — typed extension helpers over the generated bridge wrappers.
	// All methods recover a typed C# wrapper from a raw GodotObject by constructing
	// T(ulong id) via Activator, matching the single-ulong-id constructor emitted by gen_cs.py.
	public static class GodotExtension
	{
		// Returns the SceneTree root as a plain Node wrapper.
		public static Node GetRootNode(this SceneTree tree) => new Node(tree.Root.Id);

		// Path-based node lookup with typed return. Throws if the node is absent
		// rather than returning null — callers are expected to know the scene structure.
		public static T GetNode<T>(this Node parent, string path) where T : GodotObject
		{
			GodotObject obj = parent.GetNode(path);
			if (obj == null || obj.Id == 0)
				throw new InvalidOperationException($"Node not found: {path}");
			return (T)Activator.CreateInstance(typeof(T), obj.Id)!;
		}

		// Index-based child access with typed return.
		public static T GetChild<T>(this Node parent, long idx, bool includeInternal = false) where T : GodotObject
		{
			GodotObject obj = parent.GetChild(idx, includeInternal);
			if (obj == null || obj.Id == 0)
				throw new InvalidOperationException($"Child at index {idx} not found.");
			return (T)Activator.CreateInstance(typeof(T), obj.Id)!;
		}

		// Renamed from Instantiate to avoid name colliding with ClassDB.instantiate in call sites.
		public static T InstantiateAs<T>(this PackedScene scene, PackedScene.GENEDITSTATE edit_state = 0) where T : GodotObject
		{
			GodotObject obj = scene.Instantiate(edit_state);
			if (obj == null || obj.Id == 0)
				throw new InvalidOperationException("Instantiate returned null ObjectID.");
			return (T)Activator.CreateInstance(typeof(T), obj.Id)!;
		}

		// FindChild wrapper with typed return. Uses recursive search, excludes internal nodes.
		public static T Create<T>(this Node root, string path) where T : GodotObject
		{
			GodotObject obj = root.FindChild(path, true, false);
			if (obj == null || obj.Id == 0)
				throw new InvalidOperationException($"Node not found: {path}");
			return (T)Activator.CreateInstance(typeof(T), obj.Id)!;
		}

		// Stateless overload — bootstraps the SceneTree from the engine singleton
		// and delegates to the instance overload above.
		public static T Create<T>(string path) where T : GodotObject
		{
			ulong treeId = Engine.get_SceneTree_singleton();
			if (treeId == 0)
				throw new InvalidOperationException("Engine returned null SceneTree ID.");
			Node root = new Node(new SceneTree(treeId).Root.Id);
			return root.Create<T>(path);
		}
	}
}
