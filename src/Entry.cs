using Godot;
using MegaCrit.Sts2.Core.Modding;

namespace ai4sts2;

[ModInitializer(nameof(Initialize))]
public static class Entry
{
    public static void Initialize()
    {
        GD.Print($"[ai4sts2] Loaded {typeof(Entry).Assembly.GetName().Version?.ToString(3)}");
    }
}
