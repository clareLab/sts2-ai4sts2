using System.Reflection;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Runs;

namespace ai4sts2;

internal sealed record Decision(string Key, object Visible, Func<Task> Execute)
{
    internal Decision(string key, object visible, Action execute) : this(key, visible, () =>
    {
        execute();
        return Task.CompletedTask;
    })
    { }
}

internal static class Decisions
{
    private static readonly FieldInfo HighlightedCards = typeof(NCardGrid).GetField("_highlightedCards", BindingFlags.Instance | BindingFlags.NonPublic)
        ?? throw new MissingFieldException(typeof(NCardGrid).FullName, "_highlightedCards");
    internal static string ScreenName => ActiveScreenContext.Instance.GetCurrentScreen()?.GetType().Name ?? "none";

    internal static Decision[] Capture(RunState run)
    {
        if (NGame.Instance!.Transition.InTransition || NMapScreen.Instance?.IsTraveling == true) return [];
        var screen = ActiveScreenContext.Instance.GetCurrentScreen() as Node;
        if (screen == null) return [];
        if (!RoomDecisions.Prepare(screen)) return [];
        if (screen is NInspectCardScreen && !screen.IsProcessingInput()) return [];
        var selection = CardSelection.Capture(screen, run);
        if (selection != null) return selection;
        var specialised = ScreenDecisions.Capture(screen);
        if (specialised != null) return specialised;
        var player = run.Players.Single();
        if (screen is NCombatRoom && CombatManager.Instance.IsInProgress)
        {
            if (NPlayerHand.Instance?.IsInCardSelection == true) return Controls(NPlayerHand.Instance, player).ToArray();
            if (player.PlayerCombatState?.Phase != PlayerTurnPhase.Play || RunManager.Instance.ActionExecutor.IsRunning || CombatManager.Instance.PlayerActionsDisabled) return [];
            return Combat(player).ToArray();
        }
        if (RunManager.Instance.ActionExecutor.IsRunning) return [];
        if (screen is NMerchantInventory shop)
            return RoomDecisions.Shop(shop, player).Concat(Potions(player, false)).ToArray();
        if (screen is NMapScreen map)
            return Descendants(map).OfType<NMapPoint>().Where(p => p.IsEnabled && p.IsVisibleInTree())
                .Select(p => new Decision(p.GetInstanceId().ToString(), new { kind = "map", row = p.Point.coord.row, column = p.Point.coord.col, room = p.Point.PointType.ToString() }, p.ForceClick)).Concat(Potions(player, false)).ToArray();
        return Controls(screen, player).Concat(Potions(player, false)).ToArray();
    }

    private static IEnumerable<Decision> Combat(Player player)
    {
        var state = player.Creature.CombatState!;
        foreach (var card in PileType.Hand.GetPile(player).Cards)
        {
            if (!card.CanPlay(out _, out _)) continue;
            IEnumerable<Creature?> targets = card.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly
                ? state.Creatures.Where(card.IsValidTarget).Cast<Creature?>() : [null];
            foreach (var target in targets)
            {
                if (!card.IsValidTarget(target)) continue;
                var captured = target;
                yield return new Decision($"card:{card.GetHashCode()}:{target?.CombatId}",
                    new { kind = "play", card = Observation.Card(card, target), target = target == null ? null : Observation.Creature(target) },
                    () => RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new PlayCardAction(card, captured)));
            }
        }
        foreach (var action in Potions(player, true)) yield return action;
        yield return new Decision($"end:{player.PlayerCombatState!.TurnNumber}", new { kind = "end_turn" },
            () => RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(player, player.PlayerCombatState.TurnNumber)));
    }

    private static IEnumerable<Decision> Potions(Player player, bool inCombat)
    {
        foreach (var potion in player.Potions)
        {
            if (potion.IsQueued || !player.CanUseOrRemovePotions) continue;
            uint slot = (uint)player.PotionSlots.ToList().IndexOf(potion);
            yield return new Decision($"discard:{potion.GetHashCode()}", new { kind = "discard_potion", potion = Observation.Potion(potion) },
                () => RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new DiscardPotionGameAction(player, slot, inCombat)));
            if (!(potion.Usage == PotionUsage.AnyTime || inCombat && potion.Usage == PotionUsage.CombatOnly) || !potion.PassesCustomUsabilityCheck) continue;
            IEnumerable<Creature> creatures = inCombat ? player.Creature.CombatState!.Creatures : [player.Creature];
            IEnumerable<Creature?> targets = potion.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly or TargetType.AnyPlayer or TargetType.Self
                ? creatures.Where(potion.IsValidTarget).Cast<Creature?>() : [null];
            foreach (var target in targets)
            {
                if (!potion.IsValidTarget(target)) continue;
                var captured = target;
                yield return new Decision($"potion:{potion.GetHashCode()}:{target?.CombatId}",
                    new { kind = "use_potion", potion = Observation.Potion(potion), target = target == null ? null : Observation.Creature(target) },
                    () => potion.EnqueueManualUse(captured));
            }
        }
    }

    private static IEnumerable<Decision> Controls(Node screen, Player player)
    {
        var selected = Descendants(screen).OfType<NCardGrid>().SelectMany(g => (IEnumerable<CardModel>)HighlightedCards.GetValue(g)!).ToHashSet();
        var allHolders = Descendants(screen).OfType<NCardHolder>().Where(h => h.CardModel != null).ToArray();
        if (screen is NPlayerHand)
        {
            var represented = allHolders.Select(h => h.CardModel!).ToHashSet();
            if (PileType.Hand.GetPile(player).Cards.Any(c => !represented.Contains(c))) yield break;
            if (allHolders.Any(h => h.IsVisibleInTree() && !h.Hitbox.IsEnabled)) yield break;
        }
        var blockedGrids = Descendants(screen).OfType<NCardGrid>().Where(g => g.FocusBehaviorRecursive == Control.FocusBehaviorRecursiveEnum.Disabled).ToArray();
        var holders = allHolders.Where(h => h is not NPreviewCardHolder && h.IsVisibleInTree() && h.Hitbox.IsEnabled && !blockedGrids.Any(g => g.IsAncestorOf(h))).ToArray();
        foreach (var holder in holders)
            yield return new Decision(holder.GetInstanceId().ToString(), new { kind = "select_card", card = Observation.Card(holder.CardModel!), selected = selected.Contains(holder.CardModel!) || holder is NSelectedHandCardHolder },
                () => holder.EmitSignal(NCardHolder.SignalName.Pressed, holder));
        var buttons = Descendants(screen).OfType<NButton>().Where(b => b.IsEnabled && b.IsVisibleInTree()).ToArray();
        bool hasEventOptions = screen is NEventRoom && buttons.OfType<NEventOptionButton>().Any(b => !b.Option.IsLocked);
        foreach (var button in buttons)
        {
            if (!button.IsEnabled || !button.IsVisibleInTree() || holders.Any(h => h.IsAncestorOf(button))) continue;
            if (button is NPeekButton or NCardHolderHitbox) continue;
            if (button is NBackButton && System.Environment.GetEnvironmentVariable("AI4STS2_RAW_SELECTION") != "1") continue;
            if (!RoomDecisions.Available(button, player)) continue;
            if (blockedGrids.Any(g => g.IsAncestorOf(button))) continue;
            if (hasEventOptions && button is NAncientDialogueHitbox) continue;
            if (screen is NEventRoom && button is not (NEventOptionButton or NAncientDialogueHitbox or NProceedButton)) continue;
            if (button is NEventOptionButton { Option.IsLocked: true }) continue;
            yield return new Decision(button.GetInstanceId().ToString(), RoomDecisions.Describe(button), () => RoomDecisions.Select(button));
        }
    }

    internal static IEnumerable<Node> Descendants(Node node)
    {
        foreach (var child in node.GetChildren())
        {
            yield return child;
            foreach (var descendant in Descendants(child)) yield return descendant;
        }
    }
}
