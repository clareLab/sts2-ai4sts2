using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.TestSupport;

namespace ai4sts2;

internal sealed class Choices : ICardSelector
{
    private TaskCompletionSource<IEnumerable<CardModel>>? _completion;
    internal CardModel[]? Options { get; private set; }
    internal int Minimum { get; private set; }
    internal int Maximum { get; private set; }

    public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        if (_completion != null) throw new NotSupportedException("Concurrent card selections are unsupported.");
        Options = options.ToArray();
        Minimum = minSelect;
        Maximum = Math.Min(maxSelect, Options.Length);
        _completion = new TaskCompletionSource<IEnumerable<CardModel>>();
        return _completion.Task;
    }

    internal void Complete(CardModel[] cards)
    {
        var completion = _completion ?? throw new InvalidOperationException("No card selection is pending.");
        if (cards.Length < Minimum || cards.Length > Maximum || cards.Distinct().Count() != cards.Length || cards.Any(c => !Options!.Contains(c)))
            throw new InvalidOperationException("Illegal card selection.");
        _completion = null;
        Options = null;
        completion.SetResult(cards);
    }

    public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
        => throw new NotSupportedException("Card rewards are not wired yet.");
}
