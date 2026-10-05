using System.Buffers.Binary;
using System.Net.Sockets;
using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using GameLogger = MegaCrit.Sts2.Core.Logging.Logger;
using LogLevel = MegaCrit.Sts2.Core.Logging.LogLevel;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace ai4sts2;

internal static class Host
{
    private static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
    private static SceneTree Tree => (SceneTree)Engine.GetMainLoop();
    private static Socket _socket = null!;
    private static Task<object>? _pending;
    private static bool _failed;
    private static bool _closing;

    internal static void Start()
    {
        if (DisplayServer.GetName() != "headless" || !File.Exists(ProjectSettings.GlobalizePath("user://.ai4sts2-kernel")))
            throw new InvalidOperationException("The kernel requires an isolated headless profile.");
        System.Runtime.CompilerServices.RuntimeHelpers.RunClassConstructor(typeof(MegaCrit.Sts2.Core.AutoSlay.AutoSlayer).TypeHandle);
        Headless.Install();
        Patches.Install();
        if (System.Environment.GetEnvironmentVariable("AI4STS2_PROBE") == "1") Probe.Install();
        GameLogger.GlobalLogLevel = LogLevel.Warn;
        foreach (var type in GameLogger.logLevelTypeMap.Keys.ToArray()) GameLogger.logLevelTypeMap[type] = LogLevel.Warn;
        NGame.Instance!.StartOnMainMenu = false;
        _socket = new Socket(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
        _socket.Connect(new UnixDomainSocketEndPoint(System.Environment.GetEnvironmentVariable("AI4STS2_SOCKET")
            ?? throw new InvalidOperationException("Missing kernel socket.")));
        _ = Boot();
    }

    internal static async Task Frame() => await Tree.ToSignal(Tree, SceneTree.SignalName.ProcessFrame);

    private static async Task Boot()
    {
        while (NGame.Instance == null || !SaveManager.Instance.IsProfileInitialized) await Frame();
        await NGame.Instance.GameStartupComplete;
        while (NGame.Instance.Transition.InTransition) await Frame();
        await NAssetLoader.Instance.LoadInTheBackground(PreloadManager.Cache.CreateSession("ai4sts2-kernel", []));
        NGame.Instance.ProcessMode = Node.ProcessModeEnum.Disabled;
        SaveManager.Instance.SetFtuesEnabled(false);
        SaveManager.Instance.PrefsSave.FastMode = FastModeType.Instant;
        Engine.MaxFps = 0;
        Tree.ProcessFrame += Serve;
    }

    private static void Serve()
    {
        while (true)
        {
            if (_pending == null)
            {
                if (_closing)
                {
                    Tree.ProcessFrame -= Serve;
                    Tree.Quit();
                    return;
                }
                _pending = Dispatch(Receive());
            }
            Dispatcher.SynchronizationContext.ExecutePendingContinuations();
            if (!_pending.IsCompleted) return;
            object reply;
            try
            {
                reply = _pending.GetAwaiter().GetResult();
            }
            catch (Exception error)
            {
                _failed = true;
                reply = new { ok = false, error = error.ToString() };
            }
            _pending = null;
            Send(JsonSerializer.SerializeToUtf8Bytes(reply, Json));
        }
    }

    private static async Task<object> Dispatch(JsonDocument request)
    {
        using (request)
        {
            var root = request.RootElement;
            string op = root.GetProperty("op").GetString()!;
            if (op == "close")
            {
                _closing = true;
                return new { ok = true };
            }
            if (_failed) throw new InvalidOperationException("The kernel must be restarted after a failure.");
            return new { ok = true, result = await Kernel.Handle(op, root) };
        }
    }

    private static JsonDocument Receive()
    {
        Span<byte> header = stackalloc byte[4];
        ReadExactly(header);
        int length = BinaryPrimitives.ReadInt32LittleEndian(header);
        if (length <= 0) throw new InvalidDataException("Invalid kernel request size.");
        var body = new byte[length];
        ReadExactly(body);
        return JsonDocument.Parse(body);
    }

    private static void ReadExactly(Span<byte> buffer)
    {
        while (buffer.Length > 0)
        {
            int read = _socket.Receive(buffer);
            if (read == 0)
            {
                Tree.Quit();
                throw new EndOfStreamException("Kernel socket closed.");
            }
            buffer = buffer[read..];
        }
    }

    private static void Send(byte[] body)
    {
        var header = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(header, body.Length);
        _socket.Send(header);
        _socket.Send(body);
    }
}
