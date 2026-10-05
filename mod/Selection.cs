using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.TestSupport;

namespace ai4sts2;

internal sealed record SelectionContract(string Method, bool CanSkip, bool Cancelable)
{
    internal static readonly AsyncLocal<SelectionContract?> Current = new();
}

[HarmonyPatch]
internal static class SelectionContractPatch
{
    private static IEnumerable<MethodBase> TargetMethods() => typeof(CardSelectCmd).GetMethods(BindingFlags.Public | BindingFlags.Static)
        .Where(m => m.Name.StartsWith("From", StringComparison.Ordinal) && m.Name != nameof(CardSelectCmd.FromChooseABundleScreen));

    private static void Prefix(MethodBase __originalMethod, object[] __args, out SelectionContract? __state)
    {
        __state = SelectionContract.Current.Value;
        var prefs = __args.OfType<CardSelectorPrefs>().FirstOrDefault();
        bool choose = __originalMethod.Name == nameof(CardSelectCmd.FromChooseACardScreen);
        SelectionContract.Current.Value = new SelectionContract(__originalMethod.Name, choose && (bool)__args[3], !choose && prefs.Cancelable);
    }

    private static void Postfix(SelectionContract? __state) => SelectionContract.Current.Value = __state;
}

internal sealed class Selection
{
    internal required CardModel[] Options { get; init; }
    internal required int Minimum { get; init; }
    internal required int Maximum { get; init; }
    internal required bool Cancelable { get; init; }
    internal required string Method { get; init; }
    internal required TaskCompletionSource<IEnumerable<CardModel>> Source { get; init; }
    internal List<CardModel> Chosen { get; } = [];
    internal bool CanConfirm => Chosen.Count >= Minimum && Chosen.Count > 0;
    internal bool CanSkip => Chosen.Count == 0 && (Minimum == 0 || Cancelable);
    internal IEnumerable<CardModel> Remaining => Chosen.Count >= Maximum ? [] : Options.Where(c => !Chosen.Contains(c));
}

internal sealed class Selector : ICardSelector
{
    internal Selection? Pending { get; private set; }

    public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        if (Pending != null) throw new NotSupportedException("Concurrent card selections are unsupported.");
        var contract = SelectionContract.Current.Value ?? throw new InvalidOperationException("Card selection without a known command.");
        var list = options.ToArray();
        if (contract.Method == nameof(CardSelectCmd.FromChooseACardScreen) && !contract.CanSkip) minSelect = Math.Max(1, minSelect);
        Pending = new Selection
        {
            Options = list,
            Minimum = Math.Min(minSelect, list.Length),
            Maximum = Math.Min(maxSelect, list.Length),
            Cancelable = contract.Cancelable || contract.CanSkip,
            Method = contract.Method,
            Source = new TaskCompletionSource<IEnumerable<CardModel>>(TaskCreationOptions.RunContinuationsAsynchronously)
        };
        return Pending.Source.Task;
    }

    internal void Finish(bool skip)
    {
        var pending = Pending ?? throw new InvalidOperationException("No card selection is pending.");
        if (skip ? !pending.CanSkip : !pending.CanConfirm) throw new InvalidOperationException("Illegal card selection.");
        Pending = null;
        pending.Source.SetResult(skip ? [] : pending.Chosen.ToArray());
    }

    internal void Reset() => Pending = null;

    public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
        => throw new NotSupportedException("Card rewards use a dedicated selector.");
}

internal sealed class RewardPick(int card, string? alternative) : ICardSelector
{
    private bool _used;

    public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
    {
        if (_used) return default;
        _used = true;
        if (alternative != null) return new CardRewardSelection { alternative = alternatives.Single(a => a.OptionId == alternative) };
        return card < 0 ? default : new CardRewardSelection { card = options[card].Card };
    }

    public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
        => throw new NotSupportedException("Card reward picks cannot answer card selections.");
}
