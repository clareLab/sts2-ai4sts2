using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Assets;

namespace ai4sts2;

[HarmonyPatch(typeof(PreloadManager), "LoadAssets", [typeof(IEnumerable<string>), typeof(string)])]
internal static class WorkerAssets
{
    private static bool Prefix(IEnumerable<string> assetPaths, ref AssetLoadingSession __result)
    {
        foreach (string path in assetPaths)
        {
            var resource = ResourceLoader.Load<Resource>(path);
            if (resource == null) throw new InvalidOperationException($"Asset could not be loaded: {path}");
            PreloadManager.Cache.SetAsset(path, resource);
        }
        __result = AssetLoadingSession.Empty();
        return false;
    }
}
