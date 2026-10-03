using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace ai4sts2;

internal static class RunFixture
{
    internal static bool Enabled => System.Environment.GetEnvironmentVariable("AI4STS2_DIAGNOSTIC") == "1";

    internal static async Task Apply(RunState run)
    {
        if (!Enabled) return;
        var player = run.Players.Single();
        await CreatureCmd.GainMaxHp(player.Creature, 10000);
        await CreatureCmd.Heal(player.Creature, player.Creature.MaxHp);
        await PlayerCmd.GainGold(10000, player);
        foreach (var damage in player.Deck.Cards.SelectMany(c => c.DynamicVars.Values).OfType<DamageVar>())
            damage.BaseValue = 1000;
        string? eventId = System.Environment.GetEnvironmentVariable("AI4STS2_DIAGNOSTIC_EVENT");
        if (!string.IsNullOrEmpty(eventId))
            await RunManager.Instance.EnterRoomDebug(RoomType.Event, model: ModelDb.AllEvents.Single(e => e.Id.Entry == eventId));
    }
}
