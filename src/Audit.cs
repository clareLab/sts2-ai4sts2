using System.Security.Cryptography;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Runs;

namespace ai4sts2;

internal static class Audit
{
    internal static string Capture(RunState run)
    {
        var writer = new PacketWriter();
        writer.Write(NetFullCombatState.FromRun(run, null));
        foreach (var player in run.Players) writer.Write(player.PlayerRng.ToSerializable());
        writer.ZeroByteRemainder();
        return Convert.ToHexString(SHA256.HashData(writer.Buffer.AsSpan(0, writer.BytePosition)));
    }
}
