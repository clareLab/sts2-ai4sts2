using System.Text.Json;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Models;

namespace ai4sts2;

internal static class Kernel
{
    internal static async Task<object> Handle(string op, JsonElement request) => op switch
    {
        "hello" => new
        {
            game = ReleaseInfoManager.Instance.ReleaseInfo?.Version,
            commit = ReleaseInfoManager.Instance.ReleaseInfo?.Commit,
            characters = ModelDb.AllCharacters.Select(c => c.Id.Entry).ToArray()
        },
        "begin" => Describe(await Game.Begin(request.GetProperty("character").GetString()!, request.GetProperty("seed").GetString()!,
            request.GetProperty("ascension").GetInt32(), request.TryGetProperty("invincible", out var invincible) && invincible.GetBoolean())),
        "step" => Describe(await Game.Step(request.GetProperty("index").GetInt32())),
        "waits" => new { frames = Game.Waits, timings = Probe.Timings.ToDictionary(t => t.Key, t => new { count = t.Value.Count, ms = t.Value.Milliseconds }), gc = new[] { GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2) }, heap = GC.GetTotalMemory(false) },
        "probe_nre" => ProbeNre(),
        _ => throw new ArgumentException($"Unknown operation: {op}")
    };

    private static object ProbeNre()
    {
        try
        {
            object? value = request_null;
            return value!.GetHashCode();
        }
        catch (NullReferenceException)
        {
            return "caught";
        }
    }

    private static readonly object? request_null = null;

    private static object Describe(Decision decision) => new
    {
        kind = decision.Kind,
        victory = decision.Victory,
        state = Game.Summary(),
        options = decision.Options.Select(o => new { kind = o.Kind, data = o.Payload }).ToArray()
    };
}
