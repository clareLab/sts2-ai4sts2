using System.Text.Json;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Models;

namespace ai4sts2;

internal sealed record Reply(object Json, byte[] Binary);

internal static class Kernel
{
    internal static async Task<object> Handle(string op, JsonElement request) => op switch
    {
        "hello" => new
        {
            game = ReleaseInfoManager.Instance.ReleaseInfo?.Version,
            commit = ReleaseInfoManager.Instance.ReleaseInfo?.Commit,
            characters = ModelDb.AllCharacters.Select(c => c.Id.Entry).ToArray(),
            schema = Encoder.Describe()
        },
        "begin" => Describe(await Game.Begin(request.GetProperty("character").GetString()!, request.GetProperty("seed").GetString()!,
            request.GetProperty("ascension").GetInt32(), request.TryGetProperty("invincible", out var invincible) && invincible.GetBoolean())),
        "step" => Describe(await Game.Step(request.GetProperty("index").GetInt32())),
        "replay" => Describe(await Game.Replay(request.GetProperty("actions").EnumerateArray().Select(a => a.GetInt32()).ToArray())),
        "determinize" => Describe(Game.Determinize(request.GetProperty("seed").GetUInt64())),
        "waits" => new { frames = Game.Waits, timings = Probe.Timings.ToDictionary(t => t.Key, t => new { count = t.Value.Count, ms = t.Value.Milliseconds }), gc = new[] { GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2) }, heap = GC.GetTotalMemory(false) },
        _ => throw new ArgumentException($"Unknown operation: {op}")
    };

    private static Reply Describe(Decision decision)
    {
        var encoder = new Encoder(Game.Run!, decision.Kind, Game.PendingSelection);
        decision.Prelude?.Invoke(encoder);
        foreach (var option in decision.Options) option.Encode(encoder);
        return new Reply(new
        {
            kind = decision.Kind,
            victory = decision.Victory,
            state = Game.Summary(),
            options = decision.Options.Select(o => o.Kind).ToArray(),
            layout = encoder.Layout(),
            vocab = Vocab.Drain()
        }, encoder.Serialize());
    }
}
