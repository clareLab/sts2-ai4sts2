using System.Diagnostics;
using System.Reflection;
using Godot;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Factories;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Runs;

namespace ai4sts2;

internal static class CardSelection
{
    private sealed record Session(Node Owner, CardSelectorPrefs Prefs, TaskCompletionSource<IEnumerable<CardModel>> Source, SelectionDraft<CardModel> Draft, IReadOnlyDictionary<CardModel, object> Choices);
    private static readonly Dictionary<(Type, string), FieldInfo> Fields = new();
    private static readonly MethodInfo SelectGridCard = typeof(NCardGridSelectionScreen).GetMethod("OnCardClicked", BindingFlags.Instance | BindingFlags.NonPublic)
        ?? throw new MissingMethodException(typeof(NCardGridSelectionScreen).FullName, "OnCardClicked");
    private static readonly bool Raw = System.Environment.GetEnvironmentVariable("AI4STS2_RAW_SELECTION") == "1";
    private static Session? _session;

    internal static object? Visible => _session is { Source.Task.IsCompleted: false } session ? new
    {
        prompt = session.Prefs.Prompt.GetFormattedText(),
        minimum = session.Draft.Minimum,
        maximum = session.Draft.Maximum,
        skippable = session.Draft.Cancelable || session.Draft.Minimum == 0,
        selected = (Raw ? Read<IEnumerable<CardModel>>(session.Owner, "_selectedCards") : session.Draft.Selected).Select(c => Observation.Card(c)).ToArray()
    } : null;

    internal static void Reset() => _session = null;

    internal static Decision[]? Capture(Node screen, RunState run)
    {
        Node? owner = NPlayerHand.Instance?.IsInCardSelection == true ? NPlayerHand.Instance : screen is NCardGridSelectionScreen ? screen : null;
        if (owner == null)
        {
            Reset();
            return null;
        }
        var source = Read<TaskCompletionSource<IEnumerable<CardModel>>>(owner, owner is NPlayerHand ? "_selectionCompletionSource" : "_completionSource");
        if (source.Task.IsCompleted) return [];
        if (_session?.Source != source)
        {
            var prefs = Read<CardSelectorPrefs>(owner, "_prefs");
            IEnumerable<CardModel> options;
            if (owner is NPlayerHand)
            {
                var filter = Read<Func<CardModel, bool>?>(owner, "_currentSelectionFilter");
                options = PileType.Hand.GetPile(run.Players.Single()).Cards.Where(c => filter?.Invoke(c) ?? true);
            }
            else options = Read<IReadOnlyList<CardModel>>(Read<NCardGrid>(owner, "_grid"), "_cards");
            var draft = new SelectionDraft<CardModel>(options, prefs.MinSelect, prefs.MaxSelect, prefs.Cancelable);
            var choices = draft.Options.ToDictionary(c => c, c => (object)new
            {
                kind = "choose_card",
                card = Observation.Card(c),
                upgrade = Observation.Upgrade(c),
                preview = Preview(owner, c)
            });
            _session = new Session(owner, prefs, source, draft, choices);
        }
        var session = _session;
        if (Raw) return null;
        var decisions = new List<Decision>();
        foreach (var card in session.Draft.Remaining)
            decisions.Add(new Decision($"choose:{card.GetHashCode()}", session.Choices[card], async () =>
            {
                if (session.Draft.Add(card)) await Complete(session);
            }));
        if (session.Draft.CanFinish)
            decisions.Add(new Decision("finish_selection", new { kind = "finish_selection" }, () => Complete(session)));
        if (session.Draft.CanSkip)
            decisions.Add(new Decision("skip_selection", new { kind = "skip_selection" }, () => Complete(session, true)));
        return decisions.ToArray();
    }

    private static object? Preview(Node owner, CardModel card)
    {
        if (owner is NDeckTransformSelectScreen)
        {
            var transformation = Read<Func<CardModel, CardTransformation>>(owner, "_cardToTransformation")(card);
            var options = transformation.Replacement == null
                ? (transformation.ReplacementOptions ?? CardFactory.GetDefaultTransformationOptions(card, transformation.IsInCombat)).Select(c => c.Id.Entry).Order().ToArray()
                : [];
            return new { kind = "transform", random = transformation.Replacement == null, card = transformation.Replacement == null ? null : Observation.Card(transformation.Replacement), options };
        }
        if (owner is NDeckEnchantSelectScreen)
        {
            var enchantment = Read<EnchantmentModel>(owner, "_enchantment").ToMutable();
            var preview = (CardModel)card.MutableClone();
            int amount = Read<int>(owner, "_enchantmentAmount");
            preview.EnchantInternal(enchantment, amount);
            preview.IsEnchantmentPreview = true;
            enchantment.ModifyCard();
            return new { kind = "enchant", model = enchantment.Id.Entry, amount, card = Observation.Card(preview) };
        }
        return null;
    }

    private static async Task Complete(Session session, bool skip = false)
    {
        var selected = session.Draft.Finish(skip);
        var timer = Stopwatch.StartNew();
        foreach (var card in selected)
        {
            if (session.Source.Task.IsCompleted) throw new InvalidOperationException("Native selection completed before all chosen cards were submitted.");
            if (session.Owner is NCardGridSelectionScreen)
                SelectGridCard.Invoke(session.Owner, [card]);
            else
            {
                NCardHolder? holder;
                while ((holder = Decisions.Descendants(session.Owner).OfType<NHandCardHolder>().FirstOrDefault(h => h.CardModel == card && h.IsVisibleInTree() && h.Hitbox.IsEnabled)) == null)
                    await Wait(timer);
                holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);
            }
        }
        var clicked = new HashSet<ulong>();
        while (!session.Source.Task.IsCompleted)
        {
            var button = Decisions.Descendants(session.Owner).OfType<NButton>().FirstOrDefault(b =>
                (skip && session.Prefs.Cancelable ? b is NBackButton : b is NConfirmButton) && b.IsVisibleInTree() && b.IsEnabled && !clicked.Contains(b.GetInstanceId()));
            if (button != null)
            {
                clicked.Add(button.GetInstanceId());
                button.ForceClick();
            }
            else await Wait(timer);
        }
        var actual = await session.Source.Task;
        if (!actual.SequenceEqual(selected)) throw new InvalidOperationException("Native selection returned different cards or order.");
        Reset();
    }

    private static async Task Wait(Stopwatch timer)
    {
        if (timer.Elapsed.TotalSeconds > 15) throw new TimeoutException("Native card selection did not complete.");
        await Worker.Frame();
    }

    private static T Read<T>(object owner, string name)
    {
        var key = (owner.GetType(), name);
        if (!Fields.TryGetValue(key, out var field))
        {
            for (Type? type = key.Item1; type != null && field == null; type = type.BaseType)
                field = type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
            Fields[key] = field ?? throw new MissingFieldException(key.Item1.FullName, name);
        }
        return (T)field.GetValue(owner)!;
    }
}
