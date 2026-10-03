namespace ai4sts2;

internal sealed class SelectionDraft<T> where T : notnull
{
    private readonly List<T> _selected = [];
    internal IReadOnlyList<T> Options { get; }
    internal IReadOnlyList<T> Selected => _selected;
    internal int Minimum { get; }
    internal int Maximum { get; }
    internal bool Cancelable { get; }
    internal bool Completed { get; private set; }
    internal bool CanFinish => !Completed && _selected.Count > 0 && _selected.Count >= Minimum;
    internal bool CanSkip => !Completed && _selected.Count == 0 && (Minimum == 0 || Cancelable);
    internal IEnumerable<T> Remaining => Completed || _selected.Count >= Maximum ? [] : Options.Except(_selected);

    internal SelectionDraft(IEnumerable<T> options, int minimum, int maximum, bool cancelable)
    {
        Options = options.ToArray();
        Minimum = minimum;
        Maximum = Math.Min(maximum, Options.Count);
        Cancelable = cancelable;
        if (minimum < 0 || minimum > Maximum || Options.Distinct().Count() != Options.Count)
            throw new ArgumentException("Invalid card selection bounds or duplicate options.");
    }

    internal bool Add(T option)
    {
        if (Completed || _selected.Count >= Maximum || !Options.Contains(option) || _selected.Contains(option))
            throw new InvalidOperationException("Illegal card selection.");
        _selected.Add(option);
        return _selected.Count == Maximum;
    }

    internal T[] Finish(bool skip = false)
    {
        if (skip ? !CanSkip : !CanFinish) throw new InvalidOperationException("Incomplete card selection.");
        Completed = true;
        return skip ? [] : _selected.ToArray();
    }
}
