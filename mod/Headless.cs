using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Saves;

namespace ai4sts2;

internal static class Headless
{
    internal static void Install()
    {
        var harmony = new Harmony("clarelab.ai4sts2.kernel");
        foreach (var type in new[] { typeof(Assets), typeof(RunSave), typeof(ProgressSave), typeof(AudioPlay), typeof(AudioStop), typeof(CombatPresentation) })
            harmony.CreateClassProcessor(type).Patch();
    }

    [HarmonyPatch(typeof(PreloadManager), "LoadAssets", [typeof(IEnumerable<string>), typeof(string)])]
    private static class Assets
    {
        private static bool Prefix(IEnumerable<string> assetPaths, ref AssetLoadingSession __result)
        {
            foreach (string path in assetPaths)
                PreloadManager.Cache.SetAsset(path, ResourceLoader.Load<Resource>(path)
                    ?? throw new InvalidOperationException($"Asset could not be loaded: {path}"));
            __result = AssetLoadingSession.Empty();
            return false;
        }
    }

    [HarmonyPatch(typeof(SaveManager), nameof(SaveManager.SaveRun))]
    private static class RunSave
    {
        private static bool Prefix(ref Task __result)
        {
            __result = Task.CompletedTask;
            return false;
        }
    }

    [HarmonyPatch(typeof(SaveManager), nameof(SaveManager.SaveProgressFile))]
    private static class ProgressSave
    {
        private static bool Prefix() => false;
    }

    [HarmonyPatch(typeof(NDebugAudioManager), nameof(NDebugAudioManager.Play))]
    private static class AudioPlay
    {
        private static bool Prefix(ref int __result)
        {
            __result = -1;
            return false;
        }
    }

    [HarmonyPatch(typeof(NDebugAudioManager), nameof(NDebugAudioManager.Stop))]
    private static class AudioStop
    {
        private static bool Prefix([HarmonyArgument(0)] int id) => id != -1;
    }

    [HarmonyPatch(typeof(CombatStateTracker), "NotifyCombatStateChanged")]
    private static class CombatPresentation
    {
        private static bool Prefix() => false;
    }
}
