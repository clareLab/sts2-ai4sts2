using System.Diagnostics;
using Godot;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Rewards;

namespace ai4sts2;

internal static class RoomDecisions
{
    private static readonly HashSet<Reward> Skipped = new();
    private static readonly HashSet<ulong> Watched = new();
    private static Task? _purchase;
    private static NMerchantInventory? _purchasing;
    internal static bool CommittedSelection { get; set; }

    internal static void Reset()
    {
        Skipped.Clear();
        Watched.Clear();
        _purchase = null;
        _purchasing = null;
        CommittedSelection = false;
    }

    internal static bool Prepare(Node screen)
    {
        if (_purchase is { IsCompleted: true })
        {
            _purchase.GetAwaiter().GetResult();
            _purchase = null;
            _purchasing = null;
        }
        if (_purchase != null && screen == _purchasing) return false;
        var merchant = Decisions.Descendants(screen).OfType<NMerchantButton>().FirstOrDefault(b => b.IsEnabled && b.IsVisibleInTree());
        if (merchant != null)
        {
            merchant.ForceClick();
            return false;
        }
        return true;
    }

    internal static bool CanTakePotion(Player player, MegaCrit.Sts2.Core.Models.PotionModel potion) =>
        player.HasOpenPotionSlots && Hook.ShouldProcurePotion(player.RunState, player.Creature.CombatState, potion, player);

    internal static void Select(NButton button)
    {
        CommittedSelection = button is NRestSiteButton;
        button.ForceClick();
    }

    internal static IEnumerable<Decision> Shop(NMerchantInventory shop, Player player)
    {
        foreach (var slot in shop.GetAllSlots().Where(s => s.IsVisibleInTree() && s.Entry is { IsStocked: true, EnoughGold: true }))
        {
            var entry = slot.Entry;
            if (entry is MerchantCardEntry { CreationResult: { } result } && !Hook.ShouldAddToDeck(player.RunState, result.Card, out _)) continue;
            if (entry is MerchantPotionEntry { Model: { } potion } && !CanTakePotion(player, potion)) continue;
            if (entry is MerchantCardRemovalEntry && (!Hook.ShouldAllowMerchantCardRemoval(player.RunState, player) || !player.Deck.Cards.Any(c => c.IsRemovable))) continue;
            yield return new Decision($"buy:{slot.GetInstanceId()}", Offer(entry), () =>
            {
                if (_purchase != null) throw new InvalidOperationException("A purchase is already pending.");
                _purchasing = shop;
                _purchase = Purchase(entry, shop);
            });
        }
        var back = Decisions.Descendants(shop).OfType<NBackButton>().SingleOrDefault(b => b.IsEnabled && b.IsVisibleInTree());
        if (back != null)
            yield return new Decision("leave_shop", new { kind = "leave_shop" }, () => LeaveShop(shop, back));
    }

    private static async Task Purchase(MerchantEntry entry, NMerchantInventory shop)
    {
        bool success = entry is MerchantCardRemovalEntry removal
            ? await removal.OnTryPurchaseWrapper(shop.Inventory, cancelable: false)
            : await entry.OnTryPurchaseWrapper(shop.Inventory);
        if (!success) throw new InvalidOperationException("A legal purchase was rejected by the game.");
    }

    private static async Task LeaveShop(Node shop, NBackButton back)
    {
        back.ForceClick();
        var timer = Stopwatch.StartNew();
        while (true)
        {
            if (ActiveScreenContext.Instance.GetCurrentScreen() is Node room && room != shop)
            {
                var proceed = Decisions.Descendants(room).OfType<NProceedButton>().SingleOrDefault(b => b.IsEnabled && b.IsVisibleInTree());
                if (proceed != null)
                {
                    proceed.ForceClick();
                    return;
                }
            }
            if (timer.Elapsed.TotalSeconds > 15) throw new TimeoutException("Leaving the shop did not complete.");
            await Worker.Frame();
        }
    }

    internal static object Offer(MerchantEntry entry) => new
    {
        kind = "buy",
        type = entry.GetType().Name,
        cost = entry.Cost,
        card = entry is MerchantCardEntry { CreationResult: { } card } ? Observation.Card(card.Card) : null,
        relic = entry is MerchantRelicEntry { Model: { } relic } ? Observation.Relic(relic) : null,
        potion = entry is MerchantPotionEntry { Model: { } potion } ? Observation.Potion(potion) : null
    };

    internal static bool Available(NButton button, Player player)
    {
        if (button is NRestSiteButton rest && !rest.Option.IsEnabled) return false;
        if (button is not NRewardButton { Reward: { } reward } rewardButton) return true;
        if (Watched.Add(button.GetInstanceId()))
            rewardButton.RewardSkipped += b =>
            {
                if (b.Reward is CardReward cardReward) Skipped.Add(cardReward);
            };
        if (Skipped.Contains(reward)) return false;
        return reward is not PotionReward { Potion: { } potion } || CanTakePotion(player, potion);
    }

    internal static string Text(Node node) => string.Join(' ', Decisions.Descendants(node).OfType<Control>().Where(c => c.IsVisibleInTree()).Select(c => c switch
        {
            RichTextLabel rich => rich.GetParsedText(),
            Label plain => plain.Text,
            _ => ""
        }).Where(s => s.Length > 0));

    internal static object Describe(NButton button)
    {
        string label = Text(button);
        return button switch
        {
            NEventOptionButton eventButton => new { kind = "event", control = nameof(NEventOptionButton), label = eventButton.Option.Title.GetFormattedText(), description = eventButton.Option.Description.GetFormattedText(), relic = eventButton.Option.Relic == null ? null : Observation.Relic(eventButton.Option.Relic) },
            NRestSiteButton rest => new { kind = "rest", model = rest.Option.OptionId, label = rest.Option.Title.GetFormattedText(), description = rest.Option.Description.GetFormattedText() },
            NRewardButton { Reward: { } reward } => new { kind = "reward", type = reward.GetType().Name, label = reward.Description.GetFormattedText(), amount = reward is GoldReward gold ? (int?)gold.Amount : null, relic = reward is RelicReward { Relic: { } relic } ? Observation.Relic(relic) : null, potion = reward is PotionReward { Potion: { } potion } ? Observation.Potion(potion) : null },
            NTreasureRoomRelicHolder treasure => new { kind = "take_relic", relic = Observation.Relic(treasure.Relic.Model) },
            NRelicBasicHolder relic => new { kind = "take_relic", relic = Observation.Relic(relic.Relic.Model) },
            NProceedButton proceed => new { kind = "proceed", skippable = proceed.IsSkip },
            NChoiceSelectionSkipButton => new { kind = "skip_selection" },
            NCardRewardAlternativeButton => new { kind = "reward_alternative", label },
            _ => new { kind = "select", control = button.GetType().Name, label }
        };
    }
}
