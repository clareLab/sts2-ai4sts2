using System.Text.Json;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Models;

namespace ai4sts2;

internal static class Kernel
{
    internal static async Task<object> Handle(string op, JsonElement request)
    {
        switch (op)
        {
            case "hello":
                return new
                {
                    game = ReleaseInfoManager.Instance.ReleaseInfo?.Version,
                    commit = ReleaseInfoManager.Instance.ReleaseInfo?.Commit,
                    characters = ModelDb.AllCharacters.Select(c => c.Id.Entry).ToArray()
                };
            case "begin":
                await Session.Begin(request.GetProperty("character").GetString()!, request.GetProperty("seed").GetString()!, request.GetProperty("ascension").GetInt32());
                return new { room = Session.Run!.CurrentRoom?.RoomType.ToString(), floor = Session.Run.TotalFloor };
            case "combat":
                await Session.EnterCombat();
                return Session.Combat();
            case "act":
                await Session.Act(request.GetProperty("index").GetInt32(), request.GetProperty("choice").GetInt32());
                return Session.Combat();
            case "frames":
                return Session.Frames;
            default:
                throw new ArgumentException($"Unknown operation: {op}");
        }
    }
}
