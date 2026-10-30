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
        "snapshot" => new { id = Game.Save() },
        "rooms" => new { acts = Game.Rooms(), current = Game.Run!.CurrentActIndex },
        "audit" => await Game.Audit(request.GetProperty("seed").GetUInt64()),
        "restore" => Describe(await Game.Load(request.GetProperty("id").GetInt32())),
        "release" => Release(request.GetProperty("id").GetInt32()),
        _ => throw new ArgumentException($"Unknown operation: {op}")
    };

    private static object Release(int id)
    {
        Game.Drop(id);
        return new { };
    }

    private static Reply Describe(Decision decision)
    {
        var encoder = new Encoder(Game.Run!, decision.Kind, Game.PendingSelection);
        decision.Prelude?.Invoke(encoder);
        foreach (var option in decision.Options) option.Encode(encoder);
        return new Reply(new
        {
            kind = decision.Kind,
            victory = decision.Victory,
            snapshot = Game.Branchable,
            state = Game.Summary(),
            options = decision.Options.Select(o => o.Kind).ToArray(),
            layout = encoder.Layout(),
            vocab = Vocab.Drain()
        }, encoder.Serialize());
    }
}
