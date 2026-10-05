using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.TreasureRelicPicking;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Unlocks;
using HarmonyLib;

namespace ai4sts2;

internal sealed record Option(string Kind, Func<Encoder, int> Encode, Func<Task> Execute);

internal sealed record Decision(string Kind, Option[] Options, bool? Victory = null, Action<Encoder>? Prelude = null);

internal static class Game
{
    private static readonly Selector Selector = new();
    private static readonly List<RewardsSet> Frames = [];
    private static readonly Func<RewardsSet, Task> Register = set =>
    {
        set.ThrowInTestIfRewardsNotTaken = false;
        Frames.Add(set);
        return Task.CompletedTask;
    };
    private static readonly List<Task> Tasks = [];
    private static readonly HashSet<(object Item, string Flag)> Marks = new(new MarkComparer());
    private static readonly Func<EventOption, bool> DisableOnChosen =
        AccessTools.MethodDelegate<Func<EventOption, bool>>(AccessTools.PropertyGetter(typeof(EventOption), "DisableOnChosen"));
    private static readonly AccessTools.FieldRef<ActionQueueSet, System.Collections.IList> ActionQueues =
        AccessTools.FieldRefAccess<ActionQueueSet, System.Collections.IList>("_actionQueues");
    private static readonly FieldInfo QueuedActions = AccessTools.Field(AccessTools.Inner(typeof(ActionQueueSet), "ActionQueue"), "actions");
    private static IDisposable? _selectorScope;
    private static Exception? _fault;
    private static int? _actChange;
    private static RunManager Manager => RunManager.Instance;
    private static Player Me => Run!.Players[0];
    internal static RunState? Run { get; private set; }
    internal static Decision? Current { get; private set; }
    internal static Selection? PendingSelection => Selector.Pending;
    internal static Dictionary<string, int> Waits { get; } = new();

    static Game() => TaskHelper.UnobservedFault += error => _fault ??= error;

    internal static async Task<Decision> Begin(string character, string seed, int ascension, bool invincible = false)
    {
        End();
        TestRngInjector.Cleanup();
        MegaCrit.Sts2.Core.Entities.RestSite.RestSiteOption.generateForTests = null;
        TestMode.IsOn = true;
        NonInteractiveMode.AutoSlayerCheck = static () => true;
        if (NGame.Instance!.RootSceneContainer.CurrentScene == null)
        {
            NGame.Instance.RootSceneContainer.SetCurrentScene(new Control());
            await Host.Frame();
        }
        NGame.Instance.SetScreenShakeTarget(NGame.Instance.RootSceneContainer.CurrentScene!);
        seed = SeedHelper.CanonicalizeSeed(seed);
        var unlocks = UnlockState.all;
        var acts = ActModel.GetRandomList(new Rng(StringHelper.GetDeterministicHashCode(seed), "act_selection"), unlocks, false);
        var character_ = ModelDb.AllCharacters.Single(c => c.Id.Entry == character);
        var run = RunState.CreateForNewRun([Player.CreateForNewRun(character_, unlocks, 1uL)],
            acts.Select(a => a.ToMutable()).ToList(), [], GameMode.Standard, ascension, seed);
        Manager.SetUpNewSingleplayer(run, false);
        Manager.CombatReplayWriter.IsEnabled = false;
        Manager.ChecksumTracker.IsEnabled = false;
        Manager.ActionExecutor.AfterActionExecuted += AfterAction;
        Manager.TreasureRoomRelicSynchronizer.RelicsAwarded += Awarded;
        Manager.RestSiteSynchronizer.AfterPlayerOptionChosen += Rested;
        _selectorScope = CardSelectCmd.PushSelector(Selector, localOnly: true);
        RewardsSet.testSelector = Register;
        Run = run;
        await Manager.FinalizeStartingRelics();
        if (invincible)
        {
            await CreatureCmd.GainMaxHp(Me.Creature, 100000);
            await CreatureCmd.Heal(Me.Creature, Me.Creature.MaxHp);
        }
        Manager.Launch();
        Track(Manager.EnterAct(0, doTransition: false));
        return await Settle();
    }

    internal static void End()
    {
        _selectorScope?.Dispose();
        _selectorScope = null;
        if (RewardsSet.testSelector == Register) RewardsSet.testSelector = null;
        if (Manager.IsInProgress)
        {
            Manager.ActionExecutor.AfterActionExecuted -= AfterAction;
            Manager.TreasureRoomRelicSynchronizer.RelicsAwarded -= Awarded;
            Manager.RestSiteSynchronizer.AfterPlayerOptionChosen -= Rested;
            ReleaseCardSubscriptions(NetCombatCardDb.Instance, Run?.CurrentRoom as CombatRoom);
            try { Manager.CleanUp(); }
            finally { NetCombatCardDb.Instance.ClearCardsForTesting(); }
        }
        Selector.Reset();
        Patches.PendingBundle = null;
        Patches.PendingSphere = null;
        Tasks.Clear();
        Frames.Clear();
        Marks.Clear();
        _fault = null;
        _actChange = null;
        Run = null;
        Current = null;
    }

    [UnsafeAccessor(UnsafeAccessorKind.Method, Name = "OnCombatEnded")]
    private static extern void ReleaseCardSubscriptions(NetCombatCardDb database, CombatRoom? room);

    [UnsafeAccessor(UnsafeAccessorKind.Field, Name = "_pendingOptionTasks")]
    private static extern ref List<Task> PendingOptionTasks(EventSynchronizer synchronizer);

    private static void AfterAction(GameAction action) => _fault ??= action.Exception;

    private static void Awarded(List<RelicPickingResult> results)
    {
        foreach (var result in results)
            if (result.type != RelicPickingResultType.Skipped && result.player != null)
                Track(RelicCmd.Obtain(result.relic.ToMutable(), result.player));
    }

    private static void Rested(MegaCrit.Sts2.Core.Entities.RestSite.RestSiteOption option, bool success, ulong player)
    {
        if (success && Run?.CurrentRoom is RestSiteRoom room) Marks.Add((room, "rested"));
    }

    private static bool _progress;

    private static void Track(Task task)
    {
        _progress = true;
        if (task.IsFaulted) throw new InvalidOperationException("Official game logic failed.", task.Exception);
        if (!task.IsCompleted) Tasks.Add(task);
    }

    private sealed class MarkComparer : IEqualityComparer<(object Item, string Flag)>
    {
        public bool Equals((object Item, string Flag) x, (object Item, string Flag) y) => ReferenceEquals(x.Item, y.Item) && x.Flag == y.Flag;

        public int GetHashCode((object Item, string Flag) mark) => HashCode.Combine(RuntimeHelpers.GetHashCode(mark.Item), mark.Flag);
    }

    private static bool Mark(object item, string flag) => Marks.Add((item, flag));

    private static bool Marked(object item, string flag) => Marks.Contains((item, flag));

    internal static async Task<Decision> Step(int index)
    {
        var decision = Current ?? throw new InvalidOperationException("No decision is pending.");
        if (index < 0 || index >= decision.Options.Length) throw new ArgumentOutOfRangeException(nameof(index));
        Current = null;
        Track(decision.Options[index].Execute());
        return await Settle();
    }

    private static async Task<Decision> Settle()
    {
        var timer = Stopwatch.StartNew();
        while (true)
        {
            Dispatcher.SynchronizationContext.ExecutePendingContinuations();
            if (_fault != null) throw new InvalidOperationException("Official game logic failed.", _fault);
            var failed = Tasks.FirstOrDefault(t => t.IsFaulted);
            if (failed != null) throw new InvalidOperationException("Official game logic failed.", failed.Exception);
            _progress = false;
            if ((Current = Capture()) != null) return Current;
            if (_progress) continue;
            if (timer.Elapsed.TotalSeconds > 30) throw new TimeoutException($"No decision in {Run?.CurrentRoom?.GetType().Name} at floor {Run?.TotalFloor}.");
            string reason = $"{Run?.CurrentRoom?.GetType().Name}|loading={Manager.NetService.IsGameLoading}|exec={Manager.ActionExecutor.IsRunning}|queues={QueuesEmpty}|tasks={Tasks.Count(t => !t.IsCompleted)}|frames={Frames.Count}|act={_actChange}|combat={CombatManager.Instance.IsInProgress}";
            Waits[reason] = Waits.GetValueOrDefault(reason) + 1;
            await Host.Frame();
        }
    }

    private static bool Quiet => !Manager.NetService.IsGameLoading && !Manager.ActionExecutor.IsRunning && QueuesEmpty;

    private static bool QueuesEmpty
    {
        get
        {
            foreach (var queue in ActionQueues(Manager.ActionQueueSet))
                if (((System.Collections.ICollection)QueuedActions.GetValue(queue)!).Count > 0) return false;
            return true;
        }
    }

    private static Decision? Capture()
    {
        var run = Run!;
        if (run.CurrentRoom?.IsVictoryRoom == true && !Manager.NetService.IsGameLoading) return new Decision("end", [], true);
        if (run.IsGameOver && !CombatManager.Instance.IsInProgress) return new Decision("end", [], false);
        if (Patches.PendingBundle is { } bundle) return Bundle(bundle);
        if (Patches.PendingSphere is { } sphere)
        {
            if (!sphere.IsFinished) return Sphere(sphere);
            Patches.PendingSphere = null;
        }
        if (Selector.Pending is { } selection) return Select(selection);
        if (!Quiet) return null;
        var synchronizer = Manager.RewardsSetSynchronizer;
        Frames.RemoveAll(f => f.Room is CombatRoom ? Marked(f, "proceeded") : synchronizer.IsRewardsSetCompleted(f));
        if (Frames.Count > 0) return Rewards(Frames[^1]);
        Tasks.RemoveAll(t => t.IsCompleted);
        if (Tasks.Count > 0) return null;
        if (_actChange is { } act)
        {
            if (run.CurrentActIndex == act) return null;
            _actChange = null;
        }
        if (CombatManager.Instance.IsInProgress) return Stable(run) ? Combat(run) : null;
        switch (run.CurrentRoom)
        {
            case CombatRoom combat:
                if (!combat.IsPreFinished) return null;
                if (Mark(combat, "offered"))
                {
                    if (combat.Encounter.ShouldGiveRewards) Track(combat.OfferRoomEndRewards());
                    else Track(Manager.ProceedFromTerminalRewardsScreen());
                    return null;
                }
                return Map(run);
            case EventRoom eventRoom:
                if (PendingOptionTasks(Manager.EventSynchronizer).Any(t => !t.IsCompleted)) return null;
                var ev = Manager.EventSynchronizer.GetLocalEvent();
                if (ev.Id != eventRoom.CanonicalEvent.Id) return null;
                if (ev.IsFinished) return Map(run);
                var choices = ev.CurrentOptions.Select((o, i) => (Option: o, Index: i))
                    .Where(x => !x.Option.IsLocked && !x.Option.IsProceed && !(DisableOnChosen(x.Option) && x.Option.WasChosen))
                    .Select(x => new Option("event", e => e.Option("event", relic: x.Option.Relic == null ? -1 : e.Relic(x.Option.Relic, "offer", e.NextOption),
                        key: "event." + x.Option.TextKey), () =>
                    {
                        Manager.EventSynchronizer.ChooseLocalOption(x.Index);
                        return Task.CompletedTask;
                    })).ToArray();
                return choices.Length == 0 ? null : new Decision("event", [.. choices, .. Potions(false)]);
            case RestSiteRoom rest:
                var restOptions = Manager.RestSiteSynchronizer.GetLocalOptions();
                if (restOptions.Count == 0 || Marked(rest, "left")) return Map(run);
                var restChoices = restOptions.Select((o, i) => (Option: o, Index: i)).Where(x => x.Option.IsEnabled)
                    .Select(x => new Option("rest", e => e.Option("rest", key: "rest." + x.Option.OptionId), () => Manager.RestSiteSynchronizer.ChooseLocalOption(x.Index))).ToList();
                if (Marked(rest, "rested")) restChoices.Add(new Option("leave", e => e.Option("leave"), () => Leave(rest)));
                return new Decision("rest", [.. restChoices, .. Potions(false)]);
            case MerchantRoom shop:
                return Marked(shop, "left") ? Map(run) : Shop(run, shop);
            case TreasureRoom chest:
                if (Mark(chest, "opened"))
                {
                    Track(Open(chest));
                    return null;
                }
                if (Manager.TreasureRoomRelicSynchronizer.CurrentRelics is { Count: > 0 } relics && !Marked(chest, "skipped"))
                    return new Decision("treasure",
                    [
                        new Option("take_relic", e => e.Option("take_relic", relic: e.Relic(relics[0], "offer", e.NextOption)), () =>
                        {
                            Manager.TreasureRoomRelicSynchronizer.PickRelicLocally(0);
                            return Task.CompletedTask;
                        }),
                        new Option("skip_relic", e => e.Option("skip_relic"), () =>
                        {
                            Mark(chest, "skipped");
                            Manager.TreasureRoomRelicSynchronizer.SkipRelicLocally();
                            return Task.CompletedTask;
                        }),
                        .. Potions(false)
                    ]);
                return Map(run);
            case MapRoom:
                return Map(run);
            default:
                return null;
        }
    }

    private static Task Leave(AbstractRoom room)
    {
        Mark(room, "left");
        return Task.CompletedTask;
    }

    private static async Task Open(TreasureRoom chest)
    {
        await chest.DoNormalRewards();
        await chest.DoExtraRewardsIfNeeded();
    }

    private static bool Stable(RunState run)
    {
        var combat = CombatManager.Instance;
        var player = run.Players[0];
        return !combat.IsStarting && !combat.IsOverOrEnding && !combat.EndingPlayerTurnPhaseOne && !combat.EndingPlayerTurnPhaseTwo
            && !combat.IsEnemyTurnStarted && !combat.PlayerActionsDisabled && player.Creature.CombatState?.CurrentSide == CombatSide.Player
            && !combat.IsPlayerReadyToEndTurn(player) && combat.IsPartOfPlayerTurn(player) && player.PlayerCombatState?.Phase == PlayerTurnPhase.Play
            && Manager.ActionQueueSynchronizer.CombatState == ActionSynchronizerCombatState.PlayPhase && player.Creature.IsAlive;
    }

    private static bool Targetable(Creature creature) => creature.IsHittable && Hook.ShouldAllowTargeting(creature.CombatState!, creature, out _);

    private static Decision Combat(RunState run)
    {
        var player = Me;
        var creatures = player.Creature.CombatState!.Creatures;
        int turn = player.PlayerCombatState!.TurnNumber;
        var options = new List<Option>
        {
            new("end_turn", e => e.Option("end_turn"), () =>
            {
                Manager.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(player, turn));
                return Task.CompletedTask;
            })
        };
        foreach (var card in player.PlayerCombatState.Hand.Cards)
        {
            if (!card.CanPlay()) continue;
            IEnumerable<Creature?> targets = card.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly or TargetType.AnyPlayer
                ? creatures.Where(c => card.IsValidTarget(c) && Targetable(c)).Cast<Creature?>() : card.IsValidTarget(null) ? [null] : [];
            foreach (var target in targets)
                options.Add(new Option("play", e => e.Option("play", card: e.Card(card, "hand"), creature: target == null ? -1 : e.Creature(target)), () =>
                {
                    Manager.ActionQueueSynchronizer.RequestEnqueue(new PlayCardAction(card, target));
                    return Task.CompletedTask;
                }));
        }
        options.AddRange(Potions(true));
        return new Decision("combat", options.ToArray());
    }

    private static IEnumerable<Option> Potions(bool inCombat)
    {
        var player = Me;
        if (!player.CanUseOrRemovePotions) yield break;
        if (!inCombat && (Manager.ActionQueueSynchronizer.CombatState != ActionSynchronizerCombatState.NotInCombat || Manager.ActionExecutor.IsRunning)) yield break;
        for (int slot = 0; slot < player.PotionSlots.Count; slot++)
        {
            if (player.PotionSlots[slot] is not { IsQueued: false } potion) continue;
            uint index = (uint)slot;
            int potionSlot = slot;
            yield return new Option("discard_potion", e => e.Option("discard_potion", potion: e.Potion(potion, "owned", potionSlot)), () =>
            {
                Manager.ActionQueueSynchronizer.RequestEnqueue(new DiscardPotionGameAction(player, index, inCombat));
                return Task.CompletedTask;
            });
            bool usable = potion.Usage == PotionUsage.AnyTime || inCombat && potion.Usage == PotionUsage.CombatOnly;
            if (!usable || !potion.PassesCustomUsabilityCheck) continue;
            IEnumerable<Creature> creatures = inCombat ? player.Creature.CombatState!.Creatures : [player.Creature];
            IEnumerable<Creature?> targets = potion.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly or TargetType.AnyPlayer or TargetType.Self
                ? creatures.Where(c => potion.IsValidTarget(c) && (c.Side == CombatSide.Player || Targetable(c))).Cast<Creature?>() : potion.IsValidTarget(null) ? [null] : [];
            foreach (var target in targets)
                yield return new Option("use_potion", e => e.Option("use_potion", potion: e.Potion(potion, "owned", potionSlot),
                    creature: target == null ? -1 : e.Creature(target)), () =>
                {
                    potion.EnqueueManualUse(target);
                    return Task.CompletedTask;
                });
        }
    }

    private static Decision Map(RunState run)
    {
        var map = run.Map;
        var visited = run.VisitedMapCoords;
        IEnumerable<MapPoint> points;
        if (visited.Count == 0) points = [map.StartingMapPoint];
        else
        {
            var last = visited[^1];
            if (map.SecondBossMapPoint != null && last == map.BossMapPoint.coord) points = [map.SecondBossMapPoint];
            else if (last.row == map.GetRowCount() - 1) points = [map.BossMapPoint];
            else points = MapTravel.GetTravelablePointsFrom(run, map.GetPoint(last) ?? (last == map.StartingMapPoint.coord ? map.StartingMapPoint
                : throw new InvalidOperationException($"Unknown map coordinate {last}.")));
        }
        var options = points.Select(p => new Option("map", e => e.Option("map", node: e.Node(p)), () =>
        {
            Manager.ActionQueueSynchronizer.RequestEnqueue(new VoteForMapCoordAction(Me, run.MapLocation,
                new MapVote { coord = p.coord, mapGenerationCount = Manager.MapSelectionSynchronizer.MapGenerationCount }));
            return Task.CompletedTask;
        })).ToArray();
        if (options.Length == 0) throw new InvalidOperationException("The map has no legal destination.");
        return new Decision("map", [.. options, .. Potions(false)]);
    }

    private static Decision Rewards(RewardsSet frame)
    {
        var synchronizer = Manager.RewardsSetSynchronizer;
        var options = new List<Option>();
        foreach (var reward in frame.Rewards.Where(r => !r.SuccessfullySelected))
        {
            if (reward is CardReward cards)
            {
                int index = 0;
                foreach (var card in cards.Cards)
                {
                    int choice = index++;
                    options.Add(new Option("reward_card", e => e.Option("reward_card", card: e.Card(card, "offer", e.NextOption)), () => Pick(cards, choice, null)));
                }
                foreach (var alternative in CardRewardAlternative.Generate(cards).Where(a => a.OptionId != "Skip"))
                    options.Add(new Option("reward_alternative", e => e.Option("reward_alternative", key: "alternative." + alternative.OptionId),
                        () => Pick(cards, 0, alternative.OptionId)));
            }
            else if (reward is not PotionReward || Me.HasOpenPotionSlots)
                options.Add(new Option("reward", e => e.Option("reward", key: "reward." + reward.GetType().Name,
                    relic: reward is RelicReward { Relic: { } relic } ? e.Relic(relic, "offer", e.NextOption) : -1,
                    potion: reward is PotionReward { Potion: { } potion } ? e.Potion(potion, "offer", -1, e.NextOption) : -1,
                    value: reward is GoldReward gold ? gold.Amount : 0), () => synchronizer.SelectLocalReward(reward)));
        }
        if (frame.Room is CombatRoom)
            options.Add(new Option("proceed", e => e.Option("proceed"), () => Proceed(frame)));
        else if (!frame.DisallowSkipping)
            options.Add(new Option("skip_rewards", e => e.Option("skip_rewards"), () =>
            {
                synchronizer.SkipLocalRewardsSet();
                return Task.CompletedTask;
            }));
        return new Decision("rewards", [.. options, .. Potions(false)]);
    }

    private static Task Pick(CardReward reward, int card, string? alternative)
    {
        using (CardSelectCmd.PushSelector(new RewardPick(card, alternative)))
            return Manager.RewardsSetSynchronizer.SelectLocalReward(reward);
    }

    private static Task Proceed(RewardsSet frame)
    {
        var run = Run!;
        Mark(frame, "proceeded");
        var map = run.Map;
        if (run.CurrentRoom?.RoomType == RoomType.Boss && !(map.SecondBossMapPoint != null && run.CurrentMapCoord == map.BossMapPoint.coord))
        {
            _actChange = run.CurrentActIndex;
            Manager.ActChangeSynchronizer.SetLocalPlayerReady();
            return Task.CompletedTask;
        }
        return Manager.ProceedFromTerminalRewardsScreen();
    }

    private static Decision Shop(RunState run, MerchantRoom shop)
    {
        var inventory = shop.GetLocalInventory();
        var player = Me;
        var options = new List<Option>();
        foreach (var entry in inventory.CardEntries)
            if (entry is { IsStocked: true, EnoughGold: true, CreationResult: { } result } && Hook.ShouldAddToDeck(run, result.Card, out _))
                options.Add(new Option("buy_card", e => e.Option("buy_card", card: e.Card(result.Card, "offer", e.NextOption), value: entry.Cost),
                    () => entry.OnTryPurchaseWrapper(inventory)));
        foreach (var entry in inventory.RelicEntries)
            if (entry is { IsStocked: true, EnoughGold: true, Model: { } relic })
                options.Add(new Option("buy_relic", e => e.Option("buy_relic", relic: e.Relic(relic, "offer", e.NextOption), value: entry.Cost),
                    () => entry.OnTryPurchaseWrapper(inventory)));
        foreach (var entry in inventory.PotionEntries)
            if (entry is { IsStocked: true, EnoughGold: true, Model: { } potion } && player.HasOpenPotionSlots && Hook.ShouldProcurePotion(run, null, potion, player))
                options.Add(new Option("buy_potion", e => e.Option("buy_potion", potion: e.Potion(potion, "offer", -1, e.NextOption), value: entry.Cost),
                    () => entry.OnTryPurchaseWrapper(inventory)));
        if (inventory.CardRemovalEntry is { Used: false, EnoughGold: true } removal && player.Deck.Cards.Any(c => c.IsRemovable))
            options.Add(new Option("buy_removal", e => e.Option("buy_removal", value: removal.Cost), () => removal.OnTryPurchaseWrapper(inventory)));
        options.Add(new Option("leave", e => e.Option("leave"), () => Leave(shop)));
        return new Decision("shop", [.. options, .. Potions(false)]);
    }

    private static Decision Select(Selection selection)
    {
        var options = selection.Remaining.Select(card => new Option("select_card", e => e.Option("select_card", card: e.Card(card, "choice", e.NextOption),
            key: "select." + selection.Method), () =>
        {
            selection.Chosen.Add(card);
            if (selection.Chosen.Count == selection.Maximum) Selector.Finish(false);
            return Task.CompletedTask;
        })).ToList();
        if (selection.CanConfirm && selection.Chosen.Count < selection.Maximum)
            options.Add(new Option("confirm_selection", e => e.Option("confirm_selection", key: "select." + selection.Method), () =>
            {
                Selector.Finish(false);
                return Task.CompletedTask;
            }));
        if (selection.CanSkip)
            options.Add(new Option("skip_selection", e => e.Option("skip_selection", key: "select." + selection.Method), () =>
            {
                Selector.Finish(true);
                return Task.CompletedTask;
            }));
        return new Decision("select", options.ToArray());
    }

    private static Decision Bundle(Bundle bundle) => new("bundle", bundle.Options.Select(cards => new Option("bundle", e =>
        {
            int index = e.NextOption;
            foreach (var card in cards) e.Card(card, "offer", index);
            return e.Option("bundle", value: cards.Count);
        }, () =>
        {
            Patches.PendingBundle = null;
            bundle.Source.SetResult(cards);
            return Task.CompletedTask;
        })).ToArray());

    private static Decision Sphere(CrystalSphereMinigame sphere)
    {
        var options = new List<Option>();
        int width = sphere.cells.GetLength(0), height = sphere.cells.GetLength(1);
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
            {
                var cell = sphere.cells[x, y];
                bool near = false;
                for (int dx = -1; dx <= 1 && !near; dx++)
                    for (int dy = -1; dy <= 1 && !near; dy++)
                        near = x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height && sphere.cells[x + dx, y + dy].IsHidden;
                if (cell.IsHidden) options.Add(SphereOption(sphere, CrystalSphereMinigame.CrystalSphereToolType.Small, cell, x, y));
                if (near) options.Add(SphereOption(sphere, CrystalSphereMinigame.CrystalSphereToolType.Big, cell, x, y));
            }
        return new Decision("sphere", options.ToArray(), Prelude: e =>
        {
            var revealed = sphere.cells.Cast<CrystalSphereCell>().Where(c => c.Item != null).GroupBy(c => c.Item).Where(g => g.All(c => !c.IsHidden))
                .Select(g => g.Key).ToHashSet();
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                {
                    var cell = sphere.cells[x, y];
                    e.Cell(x, y, cell.IsHidden, cell.Item != null && revealed.Contains(cell.Item) ? cell.Item.GetType().Name : null);
                }
        });
    }

    private static Option SphereOption(CrystalSphereMinigame sphere, CrystalSphereMinigame.CrystalSphereToolType tool, CrystalSphereCell cell, int x, int y) =>
        new("sphere", e => e.Option("sphere", cell: x * sphere.cells.GetLength(1) + y, key: "tool." + tool), () =>
        {
            sphere.SetTool(tool);
            return sphere.CellClicked(cell);
        });

    internal static object Summary()
    {
        var run = Run!;
        var player = Me;
        return new
        {
            act = run.CurrentActIndex,
            floor = run.TotalFloor,
            room = run.CurrentRoom?.GetType().Name,
            hp = player.Creature.CurrentHp,
            max_hp = player.Creature.MaxHp,
            gold = player.Gold,
            deck = player.Deck.Cards.Count,
            relics = player.Relics.Count
        };
    }
}
