using ai4sts2;

static void Require(bool condition)
{
    if (!condition) throw new InvalidOperationException("Selection contract failed.");
}

static void Reject(Action action)
{
    try { action(); }
    catch (InvalidOperationException) { return; }
    throw new InvalidOperationException("An illegal selection was accepted.");
}

static HashSet<string> Outcomes(int count, int minimum, int maximum, bool cancelable)
{
    var outcomes = new HashSet<string>();
    void Visit(int[] prefix)
    {
        var draft = new SelectionDraft<int>(Enumerable.Range(0, count), minimum, maximum, cancelable);
        foreach (int item in prefix) draft.Add(item);
        if (draft.CanFinish)
        {
            outcomes.Add(string.Join(',', draft.Finish()));
            Reject(() => draft.Finish());
            Reject(() => draft.Add(0));
            Require(!draft.Remaining.Any());
            draft = new SelectionDraft<int>(Enumerable.Range(0, count), minimum, maximum, cancelable);
            foreach (int item in prefix) draft.Add(item);
        }
        else Reject(() => draft.Finish());
        if (draft.CanSkip)
        {
            var skipped = new SelectionDraft<int>(Enumerable.Range(0, count), minimum, maximum, cancelable);
            Require(skipped.Finish(true).Length == 0 && skipped.Completed);
            Reject(() => skipped.Add(0));
            outcomes.Add("");
        }
        else Reject(() => draft.Finish(true));
        if (prefix.Length > 0)
        {
            Require(!draft.CanSkip);
            Reject(() => draft.Add(prefix[0]));
        }
        Reject(() => draft.Add(count));
        foreach (int next in draft.Remaining.ToArray()) Visit([.. prefix, next]);
    }
    Visit([]);
    return outcomes;
}

int cases = 0;
for (int count = 0; count <= 5; count++)
    for (int minimum = 0; minimum <= count; minimum++)
        for (int maximum = minimum; maximum <= count; maximum++)
            foreach (bool cancelable in new[] { false, true })
            {
                var outcomes = Outcomes(count, minimum, maximum, cancelable);
                var expected = new HashSet<string>();
                void Enumerate(int[] prefix)
                {
                    if (prefix.Length >= minimum && prefix.Length <= maximum) expected.Add(string.Join(',', prefix));
                    if (prefix.Length == count) return;
                    foreach (int value in Enumerable.Range(0, count).Except(prefix)) Enumerate([.. prefix, value]);
                }
                Enumerate([]);
                if (cancelable) expected.Add("");
                Require(outcomes.SetEquals(expected));
                cases++;
            }
var bounded = new SelectionDraft<int>([1, 2], 1, int.MaxValue, false);
Require(bounded.Maximum == 2 && !bounded.Add(2) && bounded.Add(1));
Require(bounded.Finish().SequenceEqual(new[] { 2, 1 }));
Console.WriteLine($"PASS {cases} exhaustive selection cases");
