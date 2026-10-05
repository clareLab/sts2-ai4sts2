using System.Reflection;
using System.Reflection.Emit;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Managers;
using MegaCrit.Sts2.Core.TestSupport;

namespace ai4sts2;

internal sealed class Bundle
{
    internal required IReadOnlyList<IReadOnlyList<CardModel>> Options { get; init; }
    internal TaskCompletionSource<IEnumerable<CardModel>> Source { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
}

internal static class Patches
{
    internal static Bundle? PendingBundle;
    internal static CrystalSphereMinigame? PendingSphere;

    internal static void Install()
    {
        var harmony = new Harmony("clarelab.ai4sts2.rules");
        foreach (var type in new[]
        {
            typeof(SelectionContractPatch), typeof(BundleChoice), typeof(SphereScreen), typeof(MerchantRemoval), typeof(RealRules),
            typeof(Persistence), typeof(CancelWithoutFrame), typeof(NoRealDelay), typeof(TrialHeadless),
            typeof(FoulPotionTarget), typeof(FoulPotionThrow)
        })
            harmony.CreateClassProcessor(type).Patch();
    }

    private static bool NotMe(Player? player) => false;

    private static Task NoDelay(int milliseconds) => Task.CompletedTask;

    private static readonly AccessTools.FieldRef<GameAction, TaskCompletionSource> Completion = AccessTools.FieldRefAccess<GameAction, TaskCompletionSource>("_completionSource");

    private static IEnumerable<CodeInstruction> Replace(IEnumerable<CodeInstruction> instructions, MethodInfo original, MethodInfo replacement)
    {
        int count = 0;
        foreach (var instruction in instructions)
        {
            if (instruction.Calls(original))
            {
                count++;
                yield return new CodeInstruction(OpCodes.Call, replacement).WithLabels(instruction.labels).WithBlocks(instruction.blocks);
            }
            else yield return instruction;
        }
        if (count == 0) throw new InvalidOperationException($"Transpiler found no call to {original.Name}.");
    }

    [HarmonyPatch]
    private static class Persistence
    {
        private static IEnumerable<MethodBase> TargetMethods() =>
        [
            AccessTools.Method(typeof(ProgressSaveManager), nameof(ProgressSaveManager.SaveProgress)),
            AccessTools.Method(typeof(SaveManager), nameof(SaveManager.UpdateProgressAfterCombatWon)),
            AccessTools.Method(typeof(SaveManager), nameof(SaveManager.SaveRunHistory)),
            AccessTools.Method(typeof(SaveManager), nameof(SaveManager.DeleteCurrentRun))
        ];

        private static bool Prefix() => false;
    }

    [HarmonyPatch(typeof(GameAction), nameof(GameAction.Cancel))]
    private static class CancelWithoutFrame
    {
        private static void Postfix(GameAction __instance)
        {
            var completion = Completion(__instance);
            Dispatcher.SynchronizationContext.Post(_ => completion.TrySetCanceled(), null);
        }
    }

    [HarmonyPatch]
    private static class NoRealDelay
    {
        private static IEnumerable<MethodBase> TargetMethods() =>
        [
            AccessTools.AsyncMoveNext(AccessTools.Method(typeof(Amalgamator), "CombineStrikes")),
            AccessTools.AsyncMoveNext(AccessTools.Method(typeof(Amalgamator), "CombineDefends"))
        ];

        private static IEnumerable<CodeInstruction> Transpiler(IEnumerable<CodeInstruction> instructions) =>
            Replace(instructions, AccessTools.Method(typeof(Task), nameof(Task.Delay), [typeof(int)]), AccessTools.Method(typeof(Patches), nameof(NoDelay)));
    }

    [HarmonyPatch]
    private static class TrialHeadless
    {
        private static IEnumerable<MethodBase> TargetMethods() =>
        [
            AccessTools.Method(typeof(Trial), "Accept"),
            AccessTools.Method(typeof(Trial), "AddVfxAnchoredToPortrait")
        ];

        private static IEnumerable<CodeInstruction> Transpiler(IEnumerable<CodeInstruction> instructions) =>
            Replace(instructions, AccessTools.Method(typeof(LocalContext), nameof(LocalContext.IsMe), [typeof(Player)]), AccessTools.Method(typeof(Patches), nameof(NotMe)));
    }

    private static bool AtMerchant(AbstractRoom? room) => room is MerchantRoom || room is EventRoom { CanonicalEvent: FakeMerchant };

    [HarmonyPatch(typeof(FoulPotion), nameof(FoulPotion.PassesCustomUsabilityCheck), MethodType.Getter)]
    private static class FoulPotionTarget
    {
        private static bool Prefix(FoulPotion __instance, ref bool __result)
        {
            if (CombatManager.Instance.IsInProgress) return true;
            __result = AtMerchant(__instance.Owner.RunState.CurrentRoom);
            return false;
        }
    }

    [HarmonyPatch(typeof(FoulPotion), "OnUse")]
    private static class FoulPotionThrow
    {
        private static bool Prefix(FoulPotion __instance, ref Task __result)
        {
            if (CombatManager.Instance.IsInProgress || __instance.Owner.RunState.CurrentRoom is not EventRoom { CanonicalEvent: FakeMerchant }) return true;
            __result = Task.WhenAll(__instance.Owner.RunState.Players.Select(p =>
                ((FakeMerchant)RunManager.Instance.EventSynchronizer.GetEventForPlayer(p)).FoulPotionThrown(__instance)));
            return false;
        }
    }

    [HarmonyPatch(typeof(CardSelectCmd), nameof(CardSelectCmd.FromChooseABundleScreen))]
    private static class BundleChoice
    {
        private static bool Prefix(Player player, IReadOnlyList<IReadOnlyList<CardModel>> bundles, ref Task<IEnumerable<CardModel>> __result)
        {
            if (PendingBundle != null) throw new NotSupportedException("Concurrent bundle choices are unsupported.");
            RunManager.Instance.PlayerChoiceSynchronizer.ReserveChoiceId(player);
            PendingBundle = new Bundle { Options = bundles };
            __result = PendingBundle.Source.Task;
            return false;
        }
    }

    [HarmonyPatch(typeof(NCrystalSphereScreen), nameof(NCrystalSphereScreen.ShowScreen))]
    private static class SphereScreen
    {
        private static bool Prefix(CrystalSphereMinigame grid, ref NCrystalSphereScreen? __result)
        {
            PendingSphere = grid;
            __result = null;
            return false;
        }
    }

    [HarmonyPatch(typeof(MerchantInventory), nameof(MerchantInventory.CreateForNormalMerchant))]
    private static class MerchantRemoval
    {
        private static void Postfix(Player player, MerchantInventory __result)
        {
            if (__result.CardRemovalEntry is not { } removal) return;
            if (!Hook.ShouldAllowMerchantCardRemoval(player.RunState, player)) removal.SetUsed();
            removal.PurchaseCompleted += (_, entry) => ((MerchantCardRemovalEntry)entry).SetUsed();
        }
    }

    [HarmonyPatch]
    private static class RealRules
    {
        private static IEnumerable<System.Reflection.MethodBase> TargetMethods() =>
        [
            AccessTools.Method(typeof(MegaCrit.Sts2.Core.Models.Relics.CallingBell), "GenerateRewards"),
            AccessTools.Method(typeof(MegaCrit.Sts2.Core.Models.Relics.Cauldron), "GenerateRewards"),
            AccessTools.Method(typeof(MerchantPotionEntry), nameof(MerchantPotionEntry.CalcCost))
        ];

        private static void Prefix(out bool __state)
        {
            __state = TestMode.IsOn;
            TestMode.IsOn = false;
        }

        private static void Finalizer(bool __state) => TestMode.IsOn = __state;
    }
}
