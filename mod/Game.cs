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
using MegaCrit.Sts2.Core.Entities.Rngs;
using MegaCrit.Sts2.Core.Entities.TreasureRelicPicking;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Unlocks;
using HarmonyLib;

namespace ai4sts2;

internal sealed record Option(string Kind, Func<Encoder, int> Encode, Func<Task> Execute);

internal sealed record Decision(string Kind, Option[] Options, bool? Victory = null, Action<Encoder>? Prelude = null);

internal static class Game
{
    private static readonly Selector Selector = new();
    private static readonly Ledger Book = new();
    private static List<RewardsSet> Frames => Book.Frames;
    private static readonly Func<RewardsSet, Task> Register = set =>
    {
        set.ThrowInTestIfRewardsNotTaken = false;
        Frames.Add(set);
        return Task.CompletedTask;
    };
    private static List<Task> Tasks => Book.Tasks;
    private static HashSet<(object Item, string Flag)> Marks => Book.Marks;
    private static readonly Func<EventOption, bool> DisableOnChosen =
        AccessTools.MethodDelegate<Func<EventOption, bool>>(AccessTools.PropertyGetter(typeof(EventOption), "DisableOnChosen"));
    private static readonly AccessTools.FieldRef<ActionQueueSet, System.Collections.IList> ActionQueues =
        AccessTools.FieldRefAccess<ActionQueueSet, System.Collections.IList>("_actionQueues");
    private static readonly FieldInfo QueuedActions = AccessTools.Field(AccessTools.Inner(typeof(ActionQueueSet), "ActionQueue"), "actions");
    private static Dictionary<Creature, int> Foes => Book.Foes;
    private static IDisposable? _selectorScope;
    private static Exception? _fault;
    private static RunManager Manager => RunManager.Instance;
    private static Player Me => Run!.Players[0];
    internal static RunState? Run { get; private set; }
    internal static Decision? Current { get; private set; }
    internal static Selection? PendingSelection => Selector.Pending;

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
        Foes.Clear();
        Book.Pulls.Clear();
        Book.Combat = null;
        Saves.Clear();
        _streams = default;
        _restored = null;
        _fault = null;
        Book.ActChange = null;
        Run = null;
        Current = null;
    }

    [UnsafeAccessor(UnsafeAccessorKind.Method, Name = "OnCombatEnded")]
    private static extern void ReleaseCardSubscriptions(NetCombatCardDb database, CombatRoom? room);

    [UnsafeAccessor(UnsafeAccessorKind.Field, Name = "_pendingOptionTasks")]
    private static extern ref List<Task> PendingOptionTasks(EventSynchronizer synchronizer);

    private static void AfterAction(GameAction action)
    {
        _fault ??= action.Exception;
        if (CombatManager.Instance.DebugOnlyGetState() is not { } state) return;
        if (state != Book.Combat)
        {
            Book.Combat = state;
            Foes.Clear();
        }
        foreach (var enemy in state.Enemies)
            if (enemy.HpDisplay != HpDisplay.InfiniteWithoutNumbers) Foes[enemy] = Math.Max(Foes.GetValueOrDefault(enemy), enemy.MaxHp);
    }

    private static int Won()
    {
        var run = Run!;
        if (run.CurrentActIndex >= run.MapPointHistory.Count) return 0;
        int count = run.MapPointHistory[run.CurrentActIndex].Sum(e => e.Rooms.Count(r => r.RoomType is RoomType.Monster or RoomType.Elite or RoomType.Boss));
        if (run.CurrentRoom is CombatRoom && (CombatManager.Instance.IsInProgress || run.IsGameOver)) count--;
        return Math.Max(0, count);
    }

    private static float? Foe()
    {
        if (Run!.CurrentRoom is not CombatRoom || Book.Combat == null || Foes.Count == 0) return null;
        return (float)Book.Combat.Enemies.Where(e => e.IsAlive && Foes.ContainsKey(e) && e.HpDisplay != HpDisplay.InfiniteWithoutNumbers)
            .Sum(e => e.CurrentHp) / Math.Max(1, Foes.Values.Sum());
    }

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

    internal sealed class MarkComparer : IEqualityComparer<(object Item, string Flag)>
    {
        public bool Equals((object Item, string Flag) x, (object Item, string Flag) y) => ReferenceEquals(x.Item, y.Item) && x.Flag == y.Flag;

        public int GetHashCode((object Item, string Flag) mark) => HashCode.Combine(RuntimeHelpers.GetHashCode(mark.Item), mark.Flag);
    }

    private static bool Mark(object item, string flag) => Marks.Add((item, flag));

    private static bool Marked(object item, string flag) => Marks.Contains((item, flag));

    internal static void Pulling(ActModel act, RunState run)
    {
        var rooms = ActRooms(act);
        var fits = rooms.events.Where(e => !e.IsAllowed(run) || run.VisitedEventIds.Contains(e.Id)).Select(e => e.Id).ToHashSet();
        Book.Pulls.Add((rooms, fits, null, false, false));
    }

    internal static void Pulled(ActModel act, EventModel result)
    {
        var rooms = ActRooms(act);
        int last = Book.Pulls.FindLastIndex(p => ReferenceEquals(p.Rooms, rooms));
        var pull = Book.Pulls[last];
        Book.Pulls[last] = pull with { Seen = result.Id, Replaced = result.Id != rooms.NextEvent.Id };
    }

    internal static void Consumed(RoomSet rooms)
    {
        int last = Book.Pulls.FindLastIndex(p => ReferenceEquals(p.Rooms, rooms));
        if (last >= 0 && Book.Pulls[last] is { Fits: not null, Consumed: false } pull) Book.Pulls[last] = pull with { Consumed = true };
        else Book.Pulls.Add((rooms, null, null, false, true));
    }

    internal static async Task<Decision> Step(int index)
    {
        var decision = Current ?? throw new InvalidOperationException("No decision is pending.");
        if (index < 0 || index >= decision.Options.Length) throw new ArgumentOutOfRangeException(nameof(index));
        Current = null;
        _restored = null;
        Track(decision.Options[index].Execute());
        return await Settle();
    }

    private sealed record Saved(Snapshot Graph, object? Turn, long WinTime);

    private static readonly Dictionary<int, Saved> Saves = [];
    private static int _saves;
    private static readonly FieldInfo TurnField = AccessTools.Field(typeof(CombatManager), "_turnState");
    private static readonly FieldInfo LoopField = AccessTools.Field(typeof(CombatManager), "_turnLoopTask");
    private static readonly Type TurnType = TurnField.FieldType;
    private static readonly FieldInfo SourceField = AccessTools.Field(TurnType, "_cts");
    private static readonly MethodInfo CancelMethod = AccessTools.Method(TurnType, "Cancel");
    private static readonly PropertyInfo StateProperty = AccessTools.Property(TurnType, "State");
    private static readonly PropertyInfo[] Signals = [AccessTools.Property(TurnType, "EndTurnSignalSource"), AccessTools.Property(TurnType, "BeginEnemyTurnSignalSource")];
    private static readonly MethodInfo SwitchMethod = AccessTools.Method(typeof(CombatManager), "AwaitTurnEndAndSwitchSides");
    private static readonly MethodInfo StartMethod = AccessTools.Method(typeof(CombatManager), "StartTurn");

    internal static bool Branchable => Current is { Kind: not "end" } && Selector.Pending == null && Patches.PendingBundle == null
        && Patches.PendingSphere == null && Tasks.All(t => t.IsCompleted) && PendingOptionTasks(Manager.EventSynchronizer).All(t => t.IsCompleted);

    internal static int Save()
    {
        if (!Branchable) throw new InvalidOperationException("Snapshots require a decision with no pending prompt or task.");
        var turn = CombatManager.Instance.IsInProgress ? TurnField.GetValue(CombatManager.Instance) : null;
        var graph = new Snapshot(Run!, CombatManager.Instance, Book, Manager, Manager.ActionQueueSet, Manager.ActionQueueSynchronizer,
            Manager.PlayerChoiceSynchronizer, Manager.ActionExecutor, NetCombatCardDb.Instance, SaveManager.Instance.Progress);
        Saves[++_saves] = new Saved(graph, turn, Manager.WinTime);
        _restored = Saves[_saves];
        return _saves;
    }

    internal static void Drop(int id) => Saves.Remove(id);

    internal static async Task<Decision> Load(int id)
    {
        var saved = Saves[id];
        var manager = CombatManager.Instance;
        var previous = (Task?)LoopField.GetValue(manager);
        if (previous is { IsCompleted: false })
        {
            CancelMethod.Invoke(TurnField.GetValue(manager), null);
            var timer = Stopwatch.StartNew();
            while (!previous.IsCompleted)
            {
                Dispatcher.SynchronizationContext.ExecutePendingContinuations();
                if (previous.IsCompleted) break;
                if (timer.Elapsed.TotalSeconds > 30) throw new TimeoutException("The previous turn loop did not stop.");
                await Host.Frame();
            }
            if (previous.IsFaulted) throw new InvalidOperationException("The previous turn loop failed.", previous.Exception);
        }
        Selector.Reset();
        Patches.PendingBundle = null;
        Patches.PendingSphere = null;
        saved.Graph.Restore();
        Manager.WinTime = saved.WinTime;
        _restored = saved;
        if (saved.Turn is { } turn)
        {
            var source = (CancellationTokenSource?)SourceField.GetValue(turn);
            SourceField.SetValue(turn, new CancellationTokenSource());
            source?.Dispose();
            foreach (var signal in Signals) signal.SetValue(turn, Activator.CreateInstance(signal.PropertyType));
            LoopField.SetValue(manager, TaskHelper.RunSafely(Resume(turn)));
        }
        Current = null;
        _fault = null;
        return await Settle();
    }

    private static async Task Resume(object turn)
    {
        var manager = CombatManager.Instance;
        object? action = await Switch(turn);
        while (manager.IsInProgress && ReferenceEquals(turn, TurnField.GetValue(manager)))
        {
            var state = (CombatState)StateProperty.GetValue(turn)!;
            bool player = state.CurrentSide == CombatSide.Player;
            await (Task)StartMethod.Invoke(manager, [turn, player ? null : action])!;
            action = player ? await Switch(turn) : null;
        }
    }

    private static async Task<object?> Switch(object turn)
    {
        var task = (Task)SwitchMethod.Invoke(CombatManager.Instance, [turn])!;
        await task;
        return task.GetType().GetProperty("Result")!.GetValue(task);
    }

    internal static async Task<Decision> Replay(IEnumerable<int> actions)
    {
        var decision = Current ?? throw new InvalidOperationException("No decision is pending.");
        foreach (int index in actions) decision = await Step(index);
        return decision;
    }

    private static readonly AccessTools.FieldRef<RunRngSet, ulong> RunSeed = AccessTools.FieldRefAccess<RunRngSet, ulong>("<Seed>k__BackingField");
    private static (object? Anchor, uint Newest, Rng[] Streams) _streams;
    private static object? _restored;
    private static readonly AccessTools.FieldRef<ActModel, RoomSet> ActRooms = AccessTools.FieldRefAccess<ActModel, RoomSet>("_rooms");
    private static readonly FieldInfo DequesField = AccessTools.Field(typeof(RelicGrabBag), "_deques");
    private static readonly FieldInfo FallbackField = AccessTools.Field(typeof(RelicGrabBag), "_mpFallbackDequeue");
    private static readonly MethodInfo DrawEncounter = AccessTools.Method(typeof(ActModel), "AddWithoutRepeatingTags");
    private static readonly PropertyInfo WeakEncounters = AccessTools.Property(typeof(ActModel), "NumberOfWeakEncounters");
    private static readonly AccessTools.FieldRef<PlayerRngSet, ulong> PlayerSeed = AccessTools.FieldRefAccess<PlayerRngSet, ulong>("<Seed>k__BackingField");

    internal static Decision Determinize(ulong seed)
    {
        var decision = Current ?? throw new InvalidOperationException("No decision is pending.");
        var run = Run!;
        var source = new Rng(seed);
        uint newest = Book.Combat?.Creatures.Select(c => c.CombatId ?? 0).DefaultIfEmpty().Max() ?? 0;
        object anchor = CombatManager.Instance.IsInProgress ? (object?)Book.Combat ?? run : _restored ?? new object();
        if (!ReferenceEquals(_streams.Anchor, anchor) || _streams.Newest != newest || _streams.Streams == null)
            _streams = (anchor, newest, Snapshot.Reachable(typeof(Rng), run, CombatManager.Instance, Manager).Cast<Rng>().ToArray());
        foreach (var rng in _streams.Streams) Reseed(rng, source);
        RunSeed(run.Rng) = source.NextUnsignedLong();
        foreach (var player in run.Players)
        {
            PlayerSeed(player.PlayerRng) = source.NextUnsignedLong();
            if (player.PlayerCombatState?.DrawPile is not { } pile) continue;
            var cards = PileCards(pile);
            for (int i = cards.Count - 1; i > 0; i--)
            {
                int j = source.NextInt(i + 1);
                (cards[i], cards[j]) = (cards[j], cards[i]);
            }
        }
        Resample(run, source);
        return decision;
    }

    private static void Resample(RunState run, Rng source)
    {
        var act = run.Act;
        var rooms = ActRooms(act);
        int weak = (int)WeakEncounters.GetValue(act)!;
        Redraw(rooms.normalEncounters, rooms.normalEncountersVisited, i => i < weak ? act.AllWeakEncounters : act.AllRegularEncounters, i => i == 0 || i == weak, source);
        Redraw(rooms.eliteEncounters, rooms.eliteEncountersVisited, _ => act.AllEliteEncounters, i => i == 0, source);
        RedrawEvents(rooms, source);
        foreach (var bag in run.Players.Select(p => p.RelicGrabBag).Append(run.SharedRelicGrabBag))
        {
            foreach (System.Collections.IList deque in ((System.Collections.IDictionary)DequesField.GetValue(bag)!).Values) Permute(deque, source);
            Permute((System.Collections.IList)FallbackField.GetValue(bag)!, source);
        }
        var subsets = Subsets(run, source);
        for (int index = run.CurrentActIndex + 1; index < run.Acts.Count; index++)
        {
            var future = run.Acts[index];
            bool doubled = ActRooms(future).SecondBoss != null;
            future.SetSharedAncientSubset(subsets[index - 1]);
            ActRooms(future) = new RoomSet();
            var rng = new Rng(source.NextUnsignedLong());
            future.GenerateRooms(rng, run.UnlockState);
            if (doubled) future.SetSecondBossEncounter(rng.NextItem(future.AllBossEncounters.Where(e => e.Id != future.BossEncounter.Id))!);
        }
    }

    private static List<List<AncientEventModel>> Subsets(RunState run, Rng source)
    {
        var shared = run.UnlockState.SharedAncients.ToList();
        for (int attempt = 0; attempt < 1000; attempt++)
        {
            var rest = shared.ToList();
            Permute(rest, source);
            var subsets = new List<List<AncientEventModel>>();
            for (int index = 1; index < run.Acts.Count; index++)
            {
                int count = source.NextInt(rest.Count + 1);
                subsets.Add(rest.Take(count).ToList());
                rest = rest.Skip(count).ToList();
            }
            double accept = 1;
            for (int index = 1; index <= run.CurrentActIndex && index < run.Acts.Count; index++)
            {
                var rooms = ActRooms(run.Acts[index]);
                if (!rooms.HasAncient) continue;
                var own = run.Acts[index].GetUnlockedAncients(run.UnlockState).ToList();
                var options = own.Concat(subsets[index - 1]).ToList();
                accept *= options.Any(a => a.Id == rooms.Ancient.Id) ? (double)Math.Max(own.Count, 1) / options.Count : 0;
            }
            if (source.NextDouble() < accept) return subsets;
        }
        throw new InvalidOperationException("No shared-ancient split explains the ancients seen.");
    }

    private static void RedrawEvents(RoomSet rooms, Rng source)
    {
        var pulls = new List<(RoomSet Rooms, HashSet<ModelId>? Fits, ModelId? Seen, bool Replaced, bool Consumed)>();
        foreach (var pull in Book.Pulls.Where(p => ReferenceEquals(p.Rooms, rooms)))
        {
            if (pulls.Count > 0 && pulls[^1] is { Fits: not null, Consumed: false, Replaced: false } held && pull is { Fits: not null, Replaced: false } && pull.Seen == held.Seen)
                pulls[^1] = held with { Consumed = pull.Consumed };
            else pulls.Add(pull);
        }
        var remaining = rooms.events.ToList();
        var drawn = new List<EventModel>();
        int pointer = 0;
        for (int t = 0; t < pulls.Count; t++)
        {
            var pull = pulls[t];
            var later = pulls.Skip(t + 1).Where(p => p is { Seen: not null, Replaced: false }).Select(p => p.Seen!).ToHashSet();
            var free = remaining.Where(e => !later.Contains(e.Id)).ToList();
            if (pull.Fits == null)
            {
                if (free.Count == 0) throw new InvalidOperationException("No event left for a silently consumed slot.");
                drawn.Add(Take(remaining, free[source.NextInt(free.Count)]));
                pointer = drawn.Count;
                continue;
            }
            var fitting = free.Where(e => pull.Fits.Contains(e.Id)).ToList();
            int ahead = remaining.Count(e => later.Contains(e.Id));
            int skipped, attempts = 0;
            do
            {
                if (++attempts > 1000) throw new InvalidOperationException("No skip count is consistent with the events seen later.");
                skipped = 0;
                while (skipped < fitting.Count && source.NextDouble() * (free.Count - skipped) < fitting.Count - skipped) skipped++;
            } while (source.NextDouble() >= Clear(remaining.Count, ahead, skipped));
            Permute(fitting, source);
            foreach (var skip in fitting.Take(skipped)) drawn.Add(Take(remaining, skip));
            var open = free.Where(e => !pull.Fits.Contains(e.Id)).ToList();
            var stop = pull.Replaced ? open.Count > 0 ? open[source.NextInt(open.Count)] : null : open.FirstOrDefault(e => e.Id == pull.Seen);
            if (stop == null) throw new InvalidOperationException("No event matches a pull the player saw.");
            drawn.Add(Take(remaining, stop));
            pointer = pull.Consumed ? drawn.Count : drawn.Count - 1;
        }
        Permute(remaining, source);
        drawn.AddRange(remaining);
        rooms.events.Clear();
        rooms.events.AddRange(drawn);
        rooms.eventsVisited = pointer;
    }

    private static double Clear(int count, int ahead, int skipped)
    {
        double chance = 1;
        for (int i = 0; i < ahead; i++) chance *= Math.Max(0, count - skipped - 1 - i) / (double)(count - 1 - i);
        return chance;
    }

    private static T Take<T>(List<T> list, T item)
    {
        list.Remove(item);
        return item;
    }

    private static void Redraw(List<EncounterModel> list, int visited, Func<int, IEnumerable<EncounterModel>> pool, Func<int, bool> starts, Rng source)
    {
        if (visited >= list.Count) return;
        var rng = new Rng(source.NextUnsignedLong());
        var drawn = list.Take(visited).ToList();
        var bag = new GrabBag<EncounterModel>();
        for (int i = 0; i < list.Count; i++)
        {
            if (starts(i)) bag = new GrabBag<EncounterModel>();
            if (!bag.Any())
                foreach (var encounter in pool(i)) bag.Add(encounter, 1.0);
            if (i < visited) bag.GrabAndRemove(rng, e => e == list[i]);
            else DrawEncounter.Invoke(null, [drawn, bag, rng]);
        }
        for (int i = visited; i < list.Count && i < drawn.Count; i++) list[i] = drawn[i];
    }

    private static void Permute(System.Collections.IList list, Rng source) => Permute(list, Enumerable.Range(0, list.Count).ToList(), source);

    private static void Permute(System.Collections.IList list, List<int> slots, Rng source)
    {
        for (int k = slots.Count - 1; k > 0; k--)
        {
            int j = source.NextInt(k + 1);
            (list[slots[k]], list[slots[j]]) = (list[slots[j]], list[slots[k]]);
        }
    }

    internal static object Rooms() => Run!.Acts.Select(act =>
    {
        var rooms = ActRooms(act);
        return new
        {
            normal = rooms.normalEncounters.Select(e => e.Id.Entry).ToArray(),
            normal_visited = rooms.normalEncountersVisited,
            elite = rooms.eliteEncounters.Select(e => e.Id.Entry).ToArray(),
            elite_visited = rooms.eliteEncountersVisited,
            events = rooms.events.Select(e => e.Id.Entry).ToArray(),
            events_visited = rooms.eventsVisited,
            pulls = Book.Pulls.Where(p => ReferenceEquals(p.Rooms, rooms)).Select(p => new { fits = p.Fits?.Select(id => id.Entry).ToArray(), seen = p.Seen?.Entry, replaced = p.Replaced, consumed = p.Consumed }).ToArray(),
            boss = rooms.Boss.Id.Entry,
            second = rooms.SecondBoss?.Id.Entry,
            ancient = rooms.HasAncient ? rooms.Ancient.Id.Entry : null
        };
    }).ToArray();

    internal static readonly bool Audited = System.Environment.GetEnvironmentVariable("AI4STS2_AUDIT") == "1";
    private static bool _auditing;
    private static readonly HashSet<Rng> Drawn = new(ReferenceEqualityComparer.Instance);
    private static readonly HashSet<Rng> Created = new(ReferenceEqualityComparer.Instance);
    private static readonly Dictionary<Rng, string> Traces = new(ReferenceEqualityComparer.Instance);

    internal static void Instrument(Harmony harmony)
    {
        var draw = new HarmonyMethod(typeof(Game), nameof(OnDraw));
        foreach (var method in typeof(Rng).GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.DeclaredOnly))
            if (method.Name.StartsWith("Next") && !method.IsGenericMethodDefinition) harmony.Patch(method, prefix: draw);
        var made = new HarmonyMethod(typeof(Game), nameof(OnCreate));
        foreach (var constructor in typeof(Rng).GetConstructors(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
            harmony.Patch(constructor, postfix: made);
    }

    private static void OnDraw(Rng __instance)
    {
        if (_auditing && Drawn.Add(__instance)) Traces[__instance] = System.Environment.StackTrace;
    }

    private static void OnCreate(Rng __instance)
    {
        if (_auditing) Created.Add(__instance);
    }

    internal static async Task<object> Audit(ulong seed)
    {
        if (!Audited) throw new InvalidOperationException("Start the kernel with AI4STS2_AUDIT=1.");
        int root = Save();
        var random = new System.Random((int)seed);
        Drawn.Clear();
        Created.Clear();
        Traces.Clear();
        _auditing = true;
        int steps = 0;
        try
        {
            Determinize(seed);
            var reseeded = new HashSet<Rng>(_streams.Streams, ReferenceEqualityComparer.Instance);
            Created.Clear();
            Drawn.Clear();
            Traces.Clear();
            var decision = Current!;
            do decision = await Step(random.Next(decision.Options.Length));
            while (decision.Kind != "end" && (CombatManager.Instance.IsInProgress || !Branchable) && ++steps < 300);
            _auditing = false;
            var leaks = Drawn.Where(r => !reseeded.Contains(r) && !Created.Contains(r) && !ReferenceEquals(r, Rng.Chaotic))
                .Select(r => string.Join(" < ", Traces[r].Split('\n').Select(l => l.Trim()).Where(l => l.StartsWith("at MegaCrit") && !l.StartsWith("at MegaCrit.Sts2.Core.Random"))
                    .Take(5))).ToArray();
            return new { reseeded = reseeded.Count, drawn = Drawn.Count, created = Created.Count, steps, chaotic = Drawn.Contains(Rng.Chaotic), leaks };
        }
        finally
        {
            _auditing = false;
            await Load(root);
            Drop(root);
        }
    }

    private static void Reseed(Rng rng, Rng source) => rng.LoadFromSerializable(new Rng(source.NextUnsignedLong()).ToSerializable());

    [UnsafeAccessor(UnsafeAccessorKind.Field, Name = "_cards")]
    private static extern ref List<CardModel> PileCards(CardPile pile);

    private static async Task<Decision> Settle()
    {
        var timer = Stopwatch.StartNew();
        var total = Stopwatch.StartNew();
        while (true)
        {
            if (total.Elapsed.TotalSeconds > 300) throw new TimeoutException($"No decision within 300 s in {Run?.CurrentRoom?.GetType().Name} at floor {Run?.TotalFloor}.");
            Dispatcher.SynchronizationContext.ExecutePendingContinuations();
            if (_fault != null) throw new InvalidOperationException("Official game logic failed.", _fault);
            var failed = Tasks.FirstOrDefault(t => t.IsFaulted);
            if (failed != null) throw new InvalidOperationException("Official game logic failed.", failed.Exception);
            _progress = false;
            if ((Current = Capture()) != null)
            {
                if (Current.Options.Length != 1) return Current;
                var forced = Current.Options[0];
                Current = null;
                Track(forced.Execute());
                timer.Restart();
                continue;
            }
            if (_progress) continue;
            if (timer.Elapsed.TotalSeconds > 30) throw new TimeoutException($"No decision in {Run?.CurrentRoom?.GetType().Name} at floor {Run?.TotalFloor}.");
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
        if (Book.ActChange is { } act)
        {
            if (run.CurrentActIndex == act) return null;
            Book.ActChange = null;
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
                if (Marked(eventRoom, "left")) return Map(run);
                if (ev is FakeMerchant fake) return new Decision("shop",
                    [.. fake.StartedFight ? [] : Relics(fake.Inventory), new Option("leave", e => e.Option("leave"), () => Leave(eventRoom)), .. Potions(false)]);
                if (ev.IsFinished) return Map(run);
                var choices = ev.CurrentOptions.Select((o, i) => (Option: o, Index: i))
                    .Where(x => !x.Option.IsLocked && !x.Option.IsProceed && !(DisableOnChosen(x.Option) && x.Option.WasChosen))
                    .Select(x => new Option("event", e => e.Option("event", relic: x.Option.Relic == null ? -1 : e.Relic(x.Option.Relic, "offer", e.NextOption),
                        key: "event." + x.Option.TextKey, tips: x.Option.HoverTips), () =>
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
                if (Marked(rest, "rested") || restChoices.Count == 0) restChoices.Add(new Option("leave", e => e.Option("leave"), () => Leave(rest)));
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

    private static Decision? Rewards(RewardsSet frame)
    {
        var synchronizer = Manager.RewardsSetSynchronizer;
        foreach (var reward in frame.Rewards.OfType<CardReward>())
            if (!reward.SuccessfullySelected && Mark(reward, "peeked"))
            {
                Track(Pick(reward, -1, null));
                return null;
            }
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
                    card: reward is SpecialCardReward special && SpecialCard(special) is { } card ? e.Card(card, "offer", e.NextOption) : -1,
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

    private static readonly AccessTools.FieldRef<SpecialCardReward, CardModel?> SpecialCard = AccessTools.FieldRefAccess<SpecialCardReward, CardModel?>("_card");

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
            Book.ActChange = run.CurrentActIndex;
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
        options.AddRange(Relics(inventory));
        foreach (var entry in inventory.PotionEntries)
            if (entry is { IsStocked: true, EnoughGold: true, Model: { } potion } && player.HasOpenPotionSlots && Hook.ShouldProcurePotion(run, null, potion, player))
                options.Add(new Option("buy_potion", e => e.Option("buy_potion", potion: e.Potion(potion, "offer", -1, e.NextOption), value: entry.Cost),
                    () => entry.OnTryPurchaseWrapper(inventory)));
        if (inventory.CardRemovalEntry is { Used: false, EnoughGold: true } removal && player.Deck.Cards.Any(c => c.IsRemovable))
            options.Add(new Option("buy_removal", e => e.Option("buy_removal", value: removal.Cost), () => removal.OnTryPurchaseWrapper(inventory)));
        options.Add(new Option("leave", e => e.Option("leave"), () => Leave(shop)));
        return new Decision("shop", [.. options, .. Potions(false)]);
    }

    private static IEnumerable<Option> Relics(MerchantInventory inventory) =>
        inventory.RelicEntries.Where(entry => entry is { IsStocked: true, EnoughGold: true, Model: not null }).Select(entry =>
            new Option("buy_relic", e => e.Option("buy_relic", relic: e.Relic(entry.Model!, "offer", e.NextOption), value: entry.Cost),
                () => entry.OnTryPurchaseWrapper(inventory)));

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
                if (sphere.cells[x, y].IsHidden)
                {
                    options.Add(SphereOption(sphere, CrystalSphereMinigame.CrystalSphereToolType.Small, sphere.cells[x, y], x, y));
                    options.Add(SphereOption(sphere, CrystalSphereMinigame.CrystalSphereToolType.Big, sphere.cells[x, y], x, y));
                }
        return new Decision("sphere", options.ToArray(), Prelude: e =>
        {
            var revealed = sphere.cells.Cast<CrystalSphereCell>().Where(c => c.Item != null).GroupBy(c => c.Item).Where(g => g.All(c => !c.IsHidden))
                .Select(g => g.Key).ToHashSet();
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                {
                    var cell = sphere.cells[x, y];
                    var item = cell.IsHidden ? null : cell.Item;
                    e.Cell(x, y, cell.IsHidden, item != null && revealed.Contains(item), item?.GetType().Name);
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
            room = run.CurrentRoom?.RoomType.ToString(),
            combat = CombatManager.Instance.IsInProgress,
            turn = player.PlayerCombatState?.TurnNumber ?? 0,
            hp = player.Creature.CurrentHp,
            max_hp = player.Creature.MaxHp,
            gold = player.Gold,
            deck = player.Deck.Cards.Count,
            relics = player.Relics.Count,
            won = Won(),
            foe = Foe()
        };
    }
}

internal sealed class Ledger
{
    internal readonly List<RewardsSet> Frames = [];
    internal readonly List<Task> Tasks = [];
    internal readonly HashSet<(object Item, string Flag)> Marks = new(new Game.MarkComparer());
    internal readonly Dictionary<Creature, int> Foes = new(ReferenceEqualityComparer.Instance);
    internal readonly List<(RoomSet Rooms, HashSet<ModelId>? Fits, ModelId? Seen, bool Replaced, bool Consumed)> Pulls = [];
    internal CombatState? Combat;
    internal int? ActChange;
}
