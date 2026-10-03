using System.Diagnostics;
using Godot;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

namespace ai4sts2;

internal static class ScreenDecisions
{
    private static NCrystalSphereScreen? _sphere;
    private static NDivinationButton? _tool;

    internal static void Reset()
    {
        _sphere = null;
        _tool = null;
    }

    internal static object? Visible
    {
        get
        {
            if (_sphere == null || !GodotObject.IsInstanceValid(_sphere)) return null;
            var cells = Decisions.Descendants(_sphere).OfType<NCrystalSphereCell>().Select(c => c.Entity).ToArray();
            var revealed = cells.Where(c => c.Item != null).GroupBy(c => c.Item).Where(g => g.All(c => !c.IsHidden)).Select(g => g.Key).ToHashSet();
            return new
            {
                tool = _tool == null ? null : RoomDecisions.Text(_tool),
                description = RoomDecisions.Text(_sphere),
                cells = cells.OrderBy(c => c.Y).ThenBy(c => c.X).Select(c => new { row = c.Y, column = c.X, hidden = c.IsHidden, model = c.Item != null && revealed.Contains(c.Item) ? c.Item.GetType().Name : null }).ToArray()
            };
        }
    }

    internal static Decision[]? Capture(Node screen)
    {
        if (screen is NCardsViewScreen)
            return Decisions.Descendants(screen).OfType<NButton>().Where(b => b is NConfirmButton or NBackButton && b.IsEnabled && b.IsVisibleInTree())
                .Select(b => new Decision("proceed", new { kind = "proceed" }, b.ForceClick)).ToArray();
        if (screen is NChooseABundleSelectionScreen bundles)
            return Decisions.Descendants(bundles).OfType<NCardBundle>().Where(b => b.IsVisibleInTree()).Select(b =>
                new Decision($"bundle:{b.GetInstanceId()}", new { kind = "choose_bundle", cards = b.Bundle.Select(c => Observation.Card(c)).ToArray() }, () => ChooseBundle(bundles, b))).ToArray();
        if (screen is not NCrystalSphereScreen sphere) return null;
        if (_sphere != sphere)
        {
            _sphere = sphere;
            _tool = null;
        }
        var proceed = Decisions.Descendants(sphere).OfType<NProceedButton>().FirstOrDefault(b => b.IsEnabled && b.IsVisibleInTree());
        if (proceed != null)
            return [new Decision("proceed", new { kind = "proceed" }, proceed.ForceClick)];
        if (_tool == null)
            return Decisions.Descendants(sphere).OfType<NDivinationButton>().Where(b => b.IsEnabled && b.IsVisibleInTree()).Select(b =>
                new Decision($"tool:{b.GetInstanceId()}", new { kind = "choose_tool", label = RoomDecisions.Text(b) }, () =>
                {
                    b.ForceClick();
                    _tool = b;
                })).ToArray();
        return Decisions.Descendants(sphere).OfType<NCrystalSphereCell>().Where(c => c.IsVisibleInTree() && c.Entity.IsHidden).Select(c =>
            new Decision($"reveal:{c.Entity.X}:{c.Entity.Y}", new { kind = "reveal_cell", row = c.Entity.Y, column = c.Entity.X }, () =>
            {
                c.ForceClick();
                _tool = null;
            })).ToArray();
    }

    private static async Task ChooseBundle(Node screen, NCardBundle bundle)
    {
        bundle.Hitbox.ForceClick();
        var timer = Stopwatch.StartNew();
        while (true)
        {
            var confirm = Decisions.Descendants(screen).OfType<NConfirmButton>().SingleOrDefault(b => b.IsEnabled && b.IsVisibleInTree());
            if (confirm != null)
            {
                confirm.ForceClick();
                return;
            }
            if (timer.Elapsed.TotalSeconds > 15) throw new TimeoutException("Bundle confirmation did not appear.");
            await Worker.Frame();
        }
    }
}
