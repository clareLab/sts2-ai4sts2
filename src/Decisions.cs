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
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Runs;

namespace ai4sts2;

internal sealed record Decision(string Key, object Visible, Action Execute);

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
        var player = run.Players.Single();
        if (screen is NCombatRoom && CombatManager.Instance.IsInProgress)
        {
            if (NPlayerHand.Instance?.IsInCardSelection == true) return Controls(NPlayerHand.Instance).ToArray();
            if (player.PlayerCombatState?.Phase != PlayerTurnPhase.Play || RunManager.Instance.ActionExecutor.IsRunning || CombatManager.Instance.PlayerActionsDisabled) return [];
            return Combat(player).ToArray();
        }
        if (screen is NMapScreen map)
            return Descendants(map).OfType<NMapPoint>().Where(p => p.IsEnabled && p.IsVisibleInTree())
                .Select(p => new Decision(p.GetInstanceId().ToString(), new { kind = "map", row = p.Point.coord.row, column = p.Point.coord.col, room = p.Point.PointType.ToString() }, p.ForceClick)).ToArray();
        return Controls(screen).ToArray();
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
                    new { kind = "play", card = Observation.Card(card), target = target == null ? null : Observation.Creature(target) },
                    () => RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new PlayCardAction(card, captured)));
            }
        }
        foreach (var potion in player.Potions)
        {
            if (potion.IsQueued || !player.CanUseOrRemovePotions) continue;
            yield return new Decision($"discard:{potion.GetHashCode()}", new { kind = "discard_potion", model = potion.Id.Entry }, potion.Discard);
            if (potion.Usage is not (PotionUsage.CombatOnly or PotionUsage.AnyTime) || !potion.PassesCustomUsabilityCheck) continue;
            IEnumerable<Creature?> targets = potion.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly or TargetType.AnyPlayer or TargetType.Self
                ? state.Creatures.Where(potion.IsValidTarget).Cast<Creature?>() : [null];
            foreach (var target in targets)
            {
                if (!potion.IsValidTarget(target)) continue;
                var captured = target;
                yield return new Decision($"potion:{potion.GetHashCode()}:{target?.CombatId}",
                    new { kind = "use_potion", model = potion.Id.Entry, target = target == null ? null : Observation.Creature(target) },
                    () => potion.EnqueueManualUse(captured));
            }
        }
        yield return new Decision($"end:{player.PlayerCombatState!.TurnNumber}", new { kind = "end_turn" },
            () => RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(player, player.PlayerCombatState.TurnNumber)));
    }

    private static IEnumerable<Decision> Controls(Node screen)
    {
        var selected = Descendants(screen).OfType<NCardGrid>().SelectMany(g => (IEnumerable<CardModel>)HighlightedCards.GetValue(g)!).ToHashSet();
        var holders = Descendants(screen).OfType<NCardHolder>().Where(h => h.IsVisibleInTree() && h.Hitbox.IsEnabled && h.CardModel != null).ToArray();
        foreach (var holder in holders)
            yield return new Decision(holder.GetInstanceId().ToString(), new { kind = "select_card", card = Observation.Card(holder.CardModel!), selected = selected.Contains(holder.CardModel!) || holder is NSelectedHandCardHolder },
                () => holder.EmitSignal(NCardHolder.SignalName.Pressed, holder));
        foreach (var button in Descendants(screen).OfType<NButton>())
        {
            if (!button.IsEnabled || !button.IsVisibleInTree() || holders.Any(h => h.IsAncestorOf(button))) continue;
            if (button is NPeekButton) continue;
            if (screen is NEventRoom && button is not (NEventOptionButton or NAncientDialogueHitbox or NProceedButton)) continue;
            if (button is NEventOptionButton { Option.IsLocked: true }) continue;
            var labels = Descendants(button).OfType<RichTextLabel>().Where(l => l.IsVisibleInTree()).Select(l => l.GetParsedText());
            string label = string.Join(' ', labels);
            if (button is NEventOptionButton option) label = option.Option.Title.GetFormattedText();
            yield return new Decision(button.GetInstanceId().ToString(), new { kind = "select", control = button.GetType().Name, label }, button.ForceClick);
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
