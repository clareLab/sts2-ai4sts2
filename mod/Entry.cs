using Godot;
using MegaCrit.Sts2.Core.Modding;

namespace ai4sts2;

[ModInitializer(nameof(Initialize))]
public static class Entry
{
    public static void Initialize()
    {
        if (OS.GetCmdlineArgs().Contains("--ai4sts2-kernel")) Host.Start();
    }
}
