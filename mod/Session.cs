using System.Diagnostics;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Timeline;

namespace ai4sts2;

internal static class Session
{
    private static readonly Choices Selector = new();
    private static IDisposable? _selector;
    private static ActionExecutor? _executor;
    private static Exception? _failure;
    private static bool _ended;
    private static bool _won;
    private static RunManager Manager => RunManager.Instance;
    internal static RunState? Run { get; private set; }

    internal static async Task Begin(string character, string seed, int ascension)
    {
        End();
        TestMode.IsOn = true;
        NonInteractiveMode.AutoSlayerCheck = static () => true;
        if (NGame.Instance!.RootSceneContainer.CurrentScene == null)
        {
            NGame.Instance.RootSceneContainer.SetCurrentScene(new Control());
            await Host.Frame();
        }
        NGame.Instance.SetScreenShakeTarget(NGame.Instance.RootSceneContainer.CurrentScene!);
        foreach (string epoch in EpochModel.AllEpochIds) SaveManager.Instance.Progress.ObtainEpochOverride(epoch, EpochState.Revealed);
        var model = ModelDb.AllCharacters.Single(c => c.Id.Entry == character);
        var unlocks = SaveManager.Instance.GenerateUnlockStateFromProgress();
        var run = RunState.CreateForNewRun([Player.CreateForNewRun(model, unlocks, 1uL)],
            ActModel.GetDefaultList().Select(a => a.ToMutable()).ToList(), [], GameMode.Standard, ascension, seed);
        Manager.SetUpNewSingleplayer(run, false);
        Manager.CombatReplayWriter.IsEnabled = false;
        Manager.ChecksumTracker.IsEnabled = false;
        _selector = CardSelectCmd.PushSelector(Selector, localOnly: true);
        (_executor = Manager.ActionExecutor).AfterActionExecuted += AfterAction;
        CombatManager.Instance.CombatWon += Won;
        CombatManager.Instance.CombatEnded += Ended;
        Run = run;
        await Manager.FinalizeStartingRelics();
        Manager.Launch();
        await Manager.EnterAct(0, doTransition: false);
    }

    internal static void End()
    {
        _selector?.Dispose();
        _selector = null;
        if (_executor != null) _executor.AfterActionExecuted -= AfterAction;
        _executor = null;
        CombatManager.Instance.CombatWon -= Won;
        CombatManager.Instance.CombatEnded -= Ended;
        if (Manager.IsInProgress) Manager.CleanUp();
        Run = null;
        _failure = null;
        _ended = _won = false;
    }

    private static void AfterAction(GameAction action) => _failure ??= action.Exception;
    private static void Won(CombatRoom room) => _won = true;
    private static void Ended(CombatRoom room) => _ended = true;

    internal static async Task EnterCombat()
    {
        _ended = _won = false;
        await Manager.EnterRoomDebug(RoomType.Monster, showTransition: false);
        await Until(() => Selector.Options != null || CombatDone || Stable(Run!), "combat start");
    }

    internal static bool CombatDone => _ended && !Manager.ActionExecutor.IsRunning && Manager.ActionQueueSet.IsEmpty;

    internal static bool Stable(RunState run) => Selector.Options == null && CombatManager.Instance.IsInProgress
        && Manager.ActionExecutor is { IsRunning: false } && Manager.ActionQueueSet is { IsEmpty: true }
        && !CombatManager.Instance.IsStarting && !CombatManager.Instance.EndingPlayerTurnPhaseOne
        && !CombatManager.Instance.EndingPlayerTurnPhaseTwo && !CombatManager.Instance.PlayerActionsDisabled
        && run.Players[0].PlayerCombatState?.Phase == PlayerTurnPhase.Play;

    internal static object Combat()
    {
        var run = Run!;
        var player = run.Players[0];
        if (Selector.Options is { } options)
            return new { selection = options.Select(c => c.Id.Entry).ToArray(), minimum = Selector.Minimum, maximum = Selector.Maximum };
        if (CombatDone) return new { done = true, won = _won, hp = player.Creature.CurrentHp };
        if (!Stable(run)) throw new InvalidOperationException("Combat is not at a decision boundary.");
        return new { hp = player.Creature.CurrentHp, actions = Actions(run).Count };
    }

    private static List<(CardModel? Card, PotionModel? Potion, Creature? Target)> Actions(RunState run)
    {
        var player = run.Players[0];
        var creatures = player.Creature.CombatState!.Creatures;
        var actions = new List<(CardModel?, PotionModel?, Creature?)> { (null, null, null) };
        foreach (var card in player.PlayerCombatState!.Hand.Cards)
        {
            if (!card.CanPlay()) continue;
            if (card.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly or TargetType.AnyPlayer)
                actions.AddRange(creatures.Where(card.IsValidTarget).Select(t => ((CardModel?)card, (PotionModel?)null, (Creature?)t)));
            else if (card.IsValidTarget(null)) actions.Add((card, null, null));
        }
        foreach (var potion in player.Potions)
        {
            if (potion.IsQueued || !player.CanUseOrRemovePotions || potion.Usage is not (PotionUsage.AnyTime or PotionUsage.CombatOnly) || !potion.PassesCustomUsabilityCheck) continue;
            if (potion.TargetType is TargetType.AnyEnemy or TargetType.AnyAlly or TargetType.AnyPlayer or TargetType.Self)
                actions.AddRange(creatures.Where(potion.IsValidTarget).Select(t => ((CardModel?)null, (PotionModel?)potion, (Creature?)t)));
            else if (potion.IsValidTarget(null)) actions.Add((null, potion, null));
        }
        return actions;
    }

    internal static async Task Act(int index, int choice)
    {
        var run = Run!;
        var player = run.Players[0];
        if (Selector.Options is { } options)
        {
            var count = Math.Max(Selector.Minimum, Math.Min(1, Selector.Maximum));
            Selector.Complete(options.Skip(choice % Math.Max(1, options.Length - count + 1)).Take(count).ToArray());
            await Until(() => Selector.Options != null || CombatDone || Stable(run), "selection");
            return;
        }
        var actions = Actions(run);
        var (card, potion, target) = actions[index % actions.Count];
        if (card != null) Manager.ActionQueueSynchronizer.RequestEnqueue(new PlayCardAction(card, target));
        else if (potion != null) potion.EnqueueManualUse(target);
        else
        {
            int turn = player.PlayerCombatState!.TurnNumber;
            Manager.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(player, turn));
            await Until(() => Selector.Options != null || CombatDone || (Stable(run) && player.PlayerCombatState!.TurnNumber > turn), "end turn");
            return;
        }
        await Until(() => Selector.Options != null || CombatDone || Stable(run), "action");
    }

    internal static async Task Until(Func<bool> ready, string stage)
    {
        var timer = Stopwatch.StartNew();
        while (!ready())
        {
            if (_failure != null) throw new InvalidOperationException("Official action execution failed.", _failure);
            if (timer.Elapsed.TotalSeconds > 30) throw new TimeoutException(stage);
            Dispatcher.SynchronizationContext.ExecutePendingContinuations();
            if (ready()) break;
            Frames++;
            await Host.Frame();
        }
        if (_failure != null) throw new InvalidOperationException("Official action execution failed.", _failure);
    }

    internal static long Frames;
}
