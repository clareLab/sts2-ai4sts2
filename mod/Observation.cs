using System.Runtime.CompilerServices;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace ai4sts2;

internal static class Vocab
{
    private static readonly Dictionary<string, int> Ids = new(StringComparer.Ordinal);
    private static readonly Dictionary<ModelId, int> Models = [];
    private static readonly Dictionary<Enum, int> Enums = [];
    private static readonly Dictionary<(string, string), int> Prefixed = [];
    private static readonly List<string> Fresh = [];

    internal static int Id(string? value)
    {
        if (value == null) return 0;
        if (!Ids.TryGetValue(value, out int id))
        {
            id = Ids.Count + 1;
            Ids[value] = id;
            Fresh.Add(value);
        }
        return id;
    }

    internal static int Id(ModelId? id)
    {
        if (id == null) return 0;
        if (!Models.TryGetValue(id, out int value)) Models[id] = value = Id(id.ToString());
        return value;
    }

    internal static int Id(Enum value)
    {
        if (!Enums.TryGetValue(value, out int id)) Enums[value] = id = Id(value.GetType().Name + "." + value);
        return id;
    }

    internal static int Id(string prefix, string value)
    {
        if (!Prefixed.TryGetValue((prefix, value), out int id)) Prefixed[(prefix, value)] = id = Id(prefix + value);
        return id;
    }

    internal static string[] Drain()
    {
        var fresh = Fresh.ToArray();
        Fresh.Clear();
        return fresh;
    }
}

internal sealed class Table(string name, params string[] columns)
{
    private readonly List<float> _values = [];
    internal string Name { get; } = name;
    internal string[] Columns { get; } = columns;
    internal int Rows => _values.Count / Columns.Length;

    internal int Add(params float[] row)
    {
        if (row.Length != Columns.Length) throw new InvalidOperationException($"Row width mismatch in {Name}.");
        _values.AddRange(row);
        return Rows - 1;
    }

    internal void Set(int row, string column, float value) => _values[row * Columns.Length + Array.IndexOf(Columns, column)] = value;

    internal void Write(BinaryWriter writer)
    {
        foreach (float value in _values) writer.Write(value);
    }
}

internal sealed class Encoder
{
    internal const int VarSlots = 6;
    private static readonly string[] VarColumns = Enumerable.Range(0, VarSlots).SelectMany(i => new[] { $"var{i}", $"var{i}_base", $"var{i}_preview" }).ToArray();

    internal static readonly Table[] Schema =
    [
        new("global", "character", "ascension", "act", "act_floor", "floor", "gold", "hp", "max_hp", "block", "energy", "max_energy", "stars", "turn",
            "potion_slots", "orb_slots", "room", "decision", "in_combat", "draw", "discard", "exhaust", "deck", "boss", "second_boss", "act_model",
            "event", "selection_min", "selection_max", "selection_chosen"),
        new("card", ["zone", "model", "upgrades", "cost", "base_cost", "x_cost", "star_cost", "type", "rarity", "target", "keywords", "enchantment",
            "enchantment_amount", "affliction", "affliction_amount", "playable", "selected", "offer", .. VarColumns]),
        new("relic", "zone", "model", "counter", "status", "used_up", "offer"),
        new("potion", "zone", "model", "slot", "usage", "offer"),
        new("creature", "side", "model", "hp", "max_hp", "hp_hidden", "block", "alive", "hittable", "pet", "intent", "intent_damage", "intent_hits",
            "intent2", "intent2_damage", "intent2_hits", "intents"),
        new("power", "owner", "model", "amount", "type"),
        new("orb", "model", "passive", "evoke", "position"),
        new("node", "row", "column", "type", "visited", "current", "next"),
        new("edge", "parent", "child"),
        new("cell", "x", "y", "hidden", "item"),
        new("option", "kind", "card", "creature", "relic", "potion", "node", "cell", "key", "value")
    ];

    private readonly Dictionary<string, Table> _tables = Schema.ToDictionary(t => t.Name, t => new Table(t.Name, t.Columns));
    private readonly Dictionary<object, int> _rows = new(ReferenceEqualityComparer.Instance);
    private readonly HashSet<CardModel> _selected = new(ReferenceEqualityComparer.Instance);

    private Table T(string name) => _tables[name];

    internal static object Describe() => Schema.ToDictionary(t => t.Name, t => t.Columns);

    internal Encoder(RunState run, string decision, Selection? selection)
    {
        var player = run.Players[0];
        var combat = player.PlayerCombatState;
        bool inCombat = CombatManager.Instance.IsInProgress && combat != null;
        if (selection != null) foreach (var card in selection.Chosen) _selected.Add(card);
        var map = run.Map;
        var eventModel = run.CurrentRoom is EventRoom room ? room.CanonicalEvent : null;
        T("global").Add(Vocab.Id(player.Character.Id), run.AscensionLevel, run.CurrentActIndex, run.ActFloor, run.TotalFloor, player.Gold,
            player.Creature.CurrentHp, player.Creature.MaxHp, player.Creature.Block, combat?.Energy ?? 0, combat?.MaxEnergy ?? player.MaxEnergy,
            combat?.Stars ?? 0, combat?.TurnNumber ?? 0, player.PotionSlots.Count, combat?.OrbQueue.Capacity ?? 0,
            Vocab.Id(run.CurrentRoom?.RoomType.ToString()), Vocab.Id("decision.", decision), inCombat ? 1 : 0,
            inCombat ? combat!.DrawPile.Cards.Count : 0, inCombat ? combat!.DiscardPile.Cards.Count : 0, inCombat ? combat!.ExhaustPile.Cards.Count : 0,
            player.Deck.Cards.Count, Vocab.Id(run.Act.BossEncounter?.Id), Vocab.Id(run.Act.SecondBossEncounter?.Id), Vocab.Id(run.Act.Id),
            Vocab.Id(eventModel?.Id), selection?.Minimum ?? 0, selection?.Maximum ?? 0, selection?.Chosen.Count ?? 0);
        if (inCombat)
        {
            foreach (var card in combat!.Hand.Cards) Card(card, "hand");
            foreach (var card in Canonical(combat.DrawPile.Cards)) Card(card, "draw");
            foreach (var card in combat.DiscardPile.Cards) Card(card, "discard");
            foreach (var card in combat.ExhaustPile.Cards) Card(card, "exhaust");
            foreach (var card in combat.PlayPile.Cards) Card(card, "play");
            foreach (var creature in player.Creature.CombatState!.Creatures) Creature(creature);
            int position = 0;
            foreach (var orb in combat.OrbQueue.Orbs) T("orb").Add(Vocab.Id(orb.Id), (float)orb.PassiveVal, (float)orb.EvokeVal, position++);
        }
        else
        {
            foreach (var card in Canonical(player.Deck.Cards)) Card(card, "deck");
            Creature(player.Creature);
        }
        foreach (var relic in player.Relics) Relic(relic, "owned");
        for (int slot = 0; slot < player.PotionSlots.Count; slot++)
            if (player.PotionSlots[slot] is { } potion) Potion(potion, "owned", slot);
        if (inCombat) return;
        var visited = run.VisitedMapCoords.ToHashSet();
        var points = map.GetAllMapPoints().Append(map.StartingMapPoint).Append(map.BossMapPoint);
        if (map.SecondBossMapPoint != null) points = points.Append(map.SecondBossMapPoint);
        foreach (var point in points.Distinct().OrderBy(p => p.coord.row).ThenBy(p => p.coord.col))
            _rows[point] = T("node").Add(point.coord.row, point.coord.col, Vocab.Id(point.PointType), visited.Contains(point.coord) ? 1 : 0,
                run.CurrentMapCoord == point.coord ? 1 : 0, 0);
        foreach (var point in points.Distinct())
            foreach (var child in point.Children)
                if (_rows.TryGetValue(child, out int target)) T("edge").Add(_rows[point], target);
    }

    private static IEnumerable<CardModel> Canonical(IEnumerable<CardModel> cards) =>
        cards.OrderBy(c => c.Id).ThenBy(c => c.CurrentUpgradeLevel).ThenBy(c => c.Enchantment?.Id).ThenBy(c => c.Affliction?.Id);

    internal int Card(CardModel card, string zone, int offer = -1)
    {
        if (_rows.TryGetValue(card, out int existing))
        {
            if (offer >= 0) T("card").Set(existing, "offer", offer);
            return existing;
        }
        var row = new float[T("card").Columns.Length];
        int keywords = 0;
        foreach (var keyword in card.Keywords) keywords |= 1 << (int)keyword;
        var values = new float[]
        {
            Vocab.Id("zone.", zone), Vocab.Id(card.Id), card.CurrentUpgradeLevel, card.EnergyCost.GetWithModifiers(CostModifiers.All),
            card.EnergyCost.GetWithModifiers(CostModifiers.None), card.EnergyCost.CostsX ? 1 : 0, card.GetStarCostWithModifiers(),
            Vocab.Id(card.Type), Vocab.Id(card.Rarity), Vocab.Id(card.TargetType), keywords, Vocab.Id(card.Enchantment?.Id),
            card.Enchantment?.Amount ?? 0, Vocab.Id(card.Affliction?.Id), card.Affliction?.Amount ?? 0,
            0, _selected.Contains(card) ? 1 : 0, offer
        };
        values.CopyTo(row, 0);
        var variables = card.DynamicVars;
        if (card.Pile?.Type is PileType.Hand or PileType.Play)
        {
            variables = variables.Clone(card);
            card.UpdateDynamicVarPreview(CardPreviewMode.Normal, null, variables);
        }
        int slot = 0;
        foreach (var variable in variables.Values.OrderBy(v => v.Name, StringComparer.Ordinal))
        {
            if (slot == VarSlots) break;
            row[values.Length + slot * 3] = Vocab.Id("var.", variable.Name);
            row[values.Length + slot * 3 + 1] = (float)variable.BaseValue;
            row[values.Length + slot * 3 + 2] = (float)variable.PreviewValue;
            slot++;
        }
        return _rows[card] = T("card").Add(row);
    }

    internal int Relic(RelicModel relic, string zone, int offer = -1)
    {
        if (_rows.TryGetValue(relic, out int existing)) return existing;
        return _rows[relic] = T("relic").Add(Vocab.Id("zone.", zone), Vocab.Id(relic.Id), relic.ShowCounter ? relic.DisplayAmount : 0,
            Vocab.Id(relic.Status), relic.IsUsedUp ? 1 : 0, offer);
    }

    internal int Potion(PotionModel potion, string zone, int slot = -1, int offer = -1)
    {
        if (_rows.TryGetValue(potion, out int existing)) return existing;
        return _rows[potion] = T("potion").Add(Vocab.Id("zone.", zone), Vocab.Id(potion.Id), slot, Vocab.Id(potion.Usage), offer);
    }

    internal int Creature(Creature creature)
    {
        if (_rows.TryGetValue(creature, out int existing)) return existing;
        var intents = creature.Monster?.NextMove?.Intents.ToArray() ?? [];
        (float type, float damage, float hits) Intent(int index)
        {
            if (index >= intents.Length) return (0, 0, 0);
            var intent = intents[index];
            return intent is AttackIntent attack
                ? (Vocab.Id(intent.IntentType), (float)attack.GetSingleDamage(creature.CombatState!.Allies, creature), attack.Repeats)
                : (Vocab.Id(intent.IntentType), 0, 0);
        }
        var (first, firstDamage, firstHits) = Intent(0);
        var (second, secondDamage, secondHits) = Intent(1);
        var model = creature.Monster?.Id ?? creature.Player?.Character.Id;
        bool hidden = creature.HpDisplay == HpDisplay.InfiniteWithoutNumbers || creature.Monster?.IsHealthBarVisible == false;
        int row = T("creature").Add(Vocab.Id(creature.Side), Vocab.Id(model), hidden ? 0 : creature.CurrentHp, hidden ? 0 : creature.MaxHp,
            hidden ? 1 : 0, creature.Block,
            creature.IsAlive ? 1 : 0, creature.IsHittable ? 1 : 0, creature.IsPet ? 1 : 0, first, firstDamage, firstHits, second, secondDamage,
            secondHits, intents.Length);
        _rows[creature] = row;
        foreach (var power in creature.Powers) T("power").Add(row, Vocab.Id(power.Id), power.Amount, Vocab.Id(power.Type));
        return row;
    }

    internal int Node(MapPoint point)
    {
        int row = _rows[point];
        T("node").Set(row, "next", 1);
        return row;
    }

    internal int Cell(int x, int y, bool hidden, string? item) => T("cell").Add(x, y, hidden ? 1 : 0, Vocab.Id(item));

    internal int NextOption => T("option").Rows;

    internal int Option(string kind, int card = -1, int creature = -1, int relic = -1, int potion = -1, int node = -1, int cell = -1, string? key = null,
        double value = 0)
    {
        if (kind == "play") T("card").Set(card, "playable", 1);
        return T("option").Add(Vocab.Id("option.", kind), card, creature, relic, potion, node, cell, Vocab.Id(key), (float)value);
    }

    internal object Layout() => _tables.Values.ToDictionary(t => t.Name, t => t.Rows);

    internal byte[] Serialize()
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream))
            foreach (var table in Schema) _tables[table.Name].Write(writer);
        return stream.ToArray();
    }
}
