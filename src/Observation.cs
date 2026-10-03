using System.Text.Json;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Runs;

namespace ai4sts2;

internal static class Observation
{
    internal static object Card(CardModel card, Creature? target = null)
    {
        var variables = card.DynamicVars.Clone(card);
        card.UpdateDynamicVarPreview(CardPreviewMode.Normal, target, variables);
        return new
        {
            model = card.Id.Entry,
            type = card.Type.ToString(),
            cost = card.EnergyCost.GetResolved(),
            stars = card.GetStarCostWithModifiers(),
            upgrades = card.CurrentUpgradeLevel,
            enchantment = card.Enchantment?.Id.Entry,
            keywords = card.Keywords.Select(k => k.ToString()).Order().ToArray(),
            variables = variables.Values.OrderBy(v => v.Name).Select(v => new { model = v.Name, amount = v.PreviewValue }).ToArray()
        };
    }

    internal static object Creature(Creature creature) => new
    {
        model = creature.ModelId.Entry,
        side = creature.Side.ToString(),
        hp = creature.CurrentHp,
        max_hp = creature.MaxHp,
        block = creature.Block,
        powers = creature.Powers.Select(p => new { model = p.Id.Entry, amount = p.Amount }).ToArray(),
        intents = creature.Monster?.NextMove?.Intents.Select(i => new
        {
            type = i.IntentType.ToString(),
            damage = i is AttackIntent attack ? (decimal?)attack.GetSingleDamage(creature.CombatState!.Allies, creature) : null,
            repeats = i is AttackIntent repeated ? (int?)repeated.Repeats : null
        }).ToArray()
    };

    internal static object Capture(RunState run)
    {
        var player = run.Players.Single();
        var combat = player.PlayerCombatState;
        object[] Pile(PileType type, bool unordered = false)
        {
            if (combat == null) return [];
            var cards = type.GetPile(player).Cards.Select(c => Card(c));
            return (unordered ? cards.OrderBy(c => JsonSerializer.Serialize(c)) : cards).ToArray();
        }
        return new
        {
            character = player.Character.Id.Entry,
            ascension = run.AscensionLevel,
            floor = run.TotalFloor,
            act = run.CurrentActIndex,
            screen = Decisions.ScreenName,
            player = Creature(player.Creature),
            gold = player.Gold,
            deck = player.Deck.Cards.OrderBy(c => c.Id.Entry).ThenBy(c => c.CurrentUpgradeLevel).Select(c => Card(c)).ToArray(),
            relics = player.Relics.Select(r => r.Id.Entry).Order().ToArray(),
            potions = player.Potions.Select(p => p.Id.Entry).ToArray(),
            energy = combat?.Energy ?? 0,
            stars = combat?.Stars ?? 0,
            orbs = combat?.OrbQueue.Orbs.Select(o => new { model = o.Id.Entry, passive = o.PassiveVal, evoke = o.EvokeVal }).ToArray(),
            turn = combat?.TurnNumber ?? 0,
            hand = Pile(PileType.Hand),
            draw = Pile(PileType.Draw, true),
            discard = Pile(PileType.Discard, true),
            exhaust = Pile(PileType.Exhaust, true),
            creatures = player.Creature.CombatState?.Creatures.Select(Creature).ToArray() ?? []
        };
    }
}
