using System.Diagnostics;
using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.Runs;

namespace ai4sts2;

internal static class Probe
{
    internal static readonly Dictionary<string, (int Count, double Milliseconds)> Timings = new();

    internal static void Install()
    {
        var harmony = new Harmony("clarelab.ai4sts2.probe");
        var prefix = new HarmonyMethod(typeof(Probe), nameof(Prefix));
        var postfix = new HarmonyMethod(typeof(Probe), nameof(Postfix));
        var plain = new HarmonyMethod(typeof(Probe), nameof(PostfixPlain));
        foreach (string name in new[] { "CleanUp", "SetUpNewSingleplayer", "EnterAct", "GenerateMap", "SetActInternal", "ExitCurrentRooms", "EnterMapPointInternal", "GenerateRooms", "FinalizeStartingRelics", "EnterNextAct", "EnterRoomInternal" })
            foreach (var method in typeof(RunManager).GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic).Where(m => m.Name == name))
                harmony.Patch(method, prefix, typeof(Task).IsAssignableFrom(method.ReturnType) ? postfix : plain);
        foreach (string spec in (System.Environment.GetEnvironmentVariable("AI4STS2_PROBE_METHODS") ?? "").Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            var parts = spec.Split(':');
            var type = AccessTools.TypeByName(parts[0]) ?? throw new InvalidOperationException($"Unknown probe type {parts[0]}");
            IEnumerable<MethodBase> methods = parts[1] == ".ctor" ? type.GetConstructors(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                : type.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly).Where(m => m.Name == parts[1]);
            foreach (var method in methods)
                harmony.Patch(method, prefix, method is MethodInfo info && typeof(Task).IsAssignableFrom(info.ReturnType) ? postfix : plain);
        }
    }

    private static void Prefix(out long __state) => __state = Stopwatch.GetTimestamp();

    private static void Postfix(MethodBase __originalMethod, long __state, Task __result)
    {
        string name = __originalMethod.Name;
        if (!__result.IsCompleted) __result.ContinueWith(_ => Record(name, __state), TaskContinuationOptions.ExecuteSynchronously);
        else Record(name, __state);
    }

    private static void PostfixPlain(MethodBase __originalMethod, long __state) => Record(__originalMethod.Name, __state);

    private static void Record(string name, long started)
    {
        var (count, total) = Timings.GetValueOrDefault(name);
        Timings[name] = (count + 1, total + Stopwatch.GetElapsedTime(started).TotalMilliseconds);
    }
}
