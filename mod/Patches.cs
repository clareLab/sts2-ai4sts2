using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Managers;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.ValueProps;

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
            typeof(FoulPotionTarget), typeof(FoulPotionThrow), typeof(CosmeticOnly), typeof(MapSegments), typeof(BoulderHits)
        })
            harmony.CreateClassProcessor(type).Patch();
        if (Game.Audited) Game.Instrument(harmony);
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

    [HarmonyPatch(typeof(MegaCrit.Sts2.Core.Models.Monsters.SoulNexus), "AfterDeath")]
    private static class CosmeticOnly
    {
        private static Exception? Finalizer(Exception? __exception) => __exception is NullReferenceException ? null : __exception;
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
            AccessTools.Method(typeof(MerchantPotionEntry), nameof(MerchantPotionEntry.CalcCost)),
            AccessTools.Method(typeof(CardSelectCmd), nameof(CardSelectCmd.Reset))
        ];

        private static void Prefix(out bool __state)
        {
            __state = TestMode.IsOn;
            TestMode.IsOn = false;
        }

        private static void Finalizer(bool __state) => TestMode.IsOn = __state;
    }

    [HarmonyPatch(typeof(RollingBoulderPower), nameof(RollingBoulderPower.AfterPlayerTurnStart))]
    private static class BoulderHits
    {
        private static readonly Action<PowerModel> Flash = AccessTools.MethodDelegate<Action<PowerModel>>(AccessTools.Method(typeof(PowerModel), "Flash"));

        private static bool Prefix(RollingBoulderPower __instance, PlayerChoiceContext choiceContext, Player player, ref Task __result)
        {
            __result = Roll(__instance, choiceContext, player);
            return false;
        }

        private static async Task Roll(RollingBoulderPower power, PlayerChoiceContext context, Player player)
        {
            if (player != power.Owner.Player) return;
            Flash(power);
            foreach (var creature in power.CombatState.HittableEnemies)
                await CreatureCmd.Damage(context, [creature], power.Amount, ValueProp.Unpowered, power.Owner);
            power.SetAmount(power.Amount + power.DynamicVars.Damage.IntValue);
        }
    }

    [HarmonyPatch(typeof(MapPathPruning), nameof(MapPathPruning.FindMatchingSegments))]
    private static class MapSegments
    {
        private static readonly Func<IReadOnlyList<MapPoint>, string> Key =
            AccessTools.MethodDelegate<Func<IReadOnlyList<MapPoint>, string>>(AccessTools.Method(typeof(MapPathPruning), "GenerateSegmentKey"));
        private static readonly Func<IEnumerable<MapPoint[]>, IReadOnlyList<MapPoint>, bool> Overlaps =
            AccessTools.MethodDelegate<Func<IEnumerable<MapPoint[]>, IReadOnlyList<MapPoint>, bool>>(AccessTools.Method(typeof(MapPathPruning), "AnyOverlappingSegments"));
        private static readonly bool Verify = System.Environment.GetEnvironmentVariable("AI4STS2_VERIFY_MAPS") == "1";
        private static bool _original;

        private static bool Prefix(MapPoint startingMapPoint, ref List<List<MapPoint[]>> __result)
        {
            if (_original) return true;
            __result = Segments(startingMapPoint);
            if (!Verify) return false;
            _original = true;
            try
            {
                var expected = MapPathPruning.FindMatchingSegments(startingMapPoint);
                if (expected.Count != __result.Count || !expected.Zip(__result).All(p => p.First.Count == p.Second.Count
                    && p.First.Zip(p.Second).All(s => s.First.SequenceEqual(s.Second))))
                    throw new InvalidOperationException("Map segment search diverged from the official implementation.");
            }
            finally { _original = false; }
            return false;
        }

        private static List<List<MapPoint[]>> Segments(MapPoint start)
        {
            var segments = new SortedDictionary<string, List<MapPoint[]>>(StringComparer.Ordinal);
            var seen = new Dictionary<ulong, List<MapPoint[]>>();
            var path = new List<MapPoint>();
            var hashes = new List<ulong>();

            void Visit(MapPoint point)
            {
                path.Add(point);
                if (point.PointType == MapPointType.Boss) Emit();
                else foreach (var child in point.Children) Visit(child);
                path.RemoveAt(path.Count - 1);
            }

            void Emit()
            {
                int count = path.Count;
                for (int i = 0; i < count - 1; i++)
                {
                    var first = path[i];
                    if (first.Children.Count <= 1 && first.coord.row != 0) continue;
                    ulong hash = (ulong)RuntimeHelpers.GetHashCode(first);
                    for (int k = i + 1; k < count; k++)
                    {
                        hash = hash * 0x100000001b3UL ^ (ulong)RuntimeHelpers.GetHashCode(path[k]);
                        if (k - i < 2 || path[k].parents.Count < 2) continue;
                        var span = CollectionsMarshal.AsSpan(path).Slice(i, k - i + 1);
                        if (seen.TryGetValue(hash, out var bucket) && Contains(bucket, span)) continue;
                        var segment = span.ToArray();
                        if (bucket == null) seen[hash] = bucket = [];
                        bucket.Add(segment);
                        string key = Key(segment);
                        if (!segments.TryGetValue(key, out var list)) segments[key] = [segment];
                        else if (!Overlaps(list, segment)) list.Add(segment);
                    }
                }
            }

            Visit(start);
            return segments.Values.Where(list => list.Count > 1).ToList();
        }

        private static bool Contains(List<MapPoint[]> bucket, ReadOnlySpan<MapPoint> span)
        {
            foreach (var segment in bucket)
                if (span.SequenceEqual(segment)) return true;
            return false;
        }
    }
}
