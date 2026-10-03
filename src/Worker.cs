using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Timeline;

namespace ai4sts2;

internal static class Worker
{
    private static readonly ConcurrentQueue<string> Requests = new();
    private static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
    private static readonly ExecutionOptions Execution = ExecutionOptions.Load(Json);
    private static readonly bool AuditEnabled = System.Environment.GetEnvironmentVariable("AI4STS2_AUDIT") == "1";
    private static Dictionary<string, double> _timings = new();
    private static Task? _pending;
    private static RunState? _run;
    private static long _revision;
    private static bool _poisoned;
    private static bool _combatEnded;
    private static bool _combatWon;
    private static string _scope = "run";
    private static Decision[] _decisions = [];
    private static SceneTree Tree => (SceneTree)Engine.GetMainLoop();

    internal static void Initialize()
    {
        if (!File.Exists(ProjectSettings.GlobalizePath("user://.ai4sts2-worker")))
            throw new InvalidOperationException("An isolated AI4STS2 worker directory is required.");
        if (TestMode.IsOn) throw new InvalidOperationException("The worker requires normal game rules.");
        CombatManager.Instance.CombatWon += _ => _combatWon = true;
        CombatManager.Instance.CombatEnded += _ => _combatEnded = true;
        Tree.ProcessFrame += Tick;
        _ = Task.Run(async () =>
        {
            while (await Console.In.ReadLineAsync() is { } line) Requests.Enqueue(line);
            Requests.Enqueue("{\"id\":\"eof\",\"method\":\"close\"}");
        });
    }

    private static void Tick()
    {
        if (_pending is { IsCompleted: false } || !Requests.TryDequeue(out var line)) return;
        _pending = Handle(line);
    }

    private static async Task Handle(string line)
    {
        var timer = Stopwatch.StartNew();
        _timings = new();
        string? id = null;
        try
        {
            using var document = JsonDocument.Parse(line);
            var request = document.RootElement;
            id = request.GetProperty("id").GetString();
            string method = request.GetProperty("method").GetString()!;
            if (method == "close")
            {
                Reply(new { id, ok = true, result = new { closed = true } });
                Tree.Quit();
                return;
            }
            if (_poisoned) throw new InvalidOperationException("The worker must be restarted after a failed operation.");
            await Ready();
            object result;
            if (method == "hello")
                result = new { protocol = 2, engine = "official", test_mode = TestMode.IsOn, execution = Execution, characters = ModelDb.AllCharacters.Select(c => c.Id.Entry).ToArray() };
            else if (method == "reset") result = await Reset(request.GetProperty("params"));
            else if (method == "observe") result = await Observe();
            else if (method == "step") result = await Step(request.GetProperty("params"));
            else throw new ArgumentException($"Unknown method: {method}");
            _timings["engine_ms"] = timer.Elapsed.TotalMilliseconds;
            Reply(new { id, ok = true, result, timing = _timings });
        }
        catch (Exception error)
        {
            _poisoned = true;
            Reply(new { id, ok = false, error = error.ToString() });
        }
    }

    private static void Reply(object response)
    {
        Console.WriteLine("AI4STS2 " + JsonSerializer.Serialize(response, Json));
        Console.Out.Flush();
    }

    private static async Task Ready()
    {
        var timer = Stopwatch.StartNew();
        while (NGame.Instance == null || !SaveManager.Instance.IsProfileInitialized || !NGame.Instance.GameStartupComplete.IsCompleted)
        {
            if (timer.Elapsed.TotalSeconds > 60) throw new TimeoutException("Official game startup timed out.");
            await Frame();
        }
        await NGame.Instance.GameStartupComplete;
        Execution.Apply();
    }

    private static async Task<object> Reset(JsonElement parameters)
    {
        string character = parameters.GetProperty("character").GetString()!;
        string seed = parameters.GetProperty("seed").GetString()!;
        _scope = parameters.GetProperty("scope").GetString()!;
        if (_scope is not ("run" or "first_combat")) throw new ArgumentException("Unknown episode scope.");
        if (_run != null)
        {
            await NGame.Instance!.ReturnToMainMenu();
            _run = null;
        }
        SaveManager.Instance.SetFtuesEnabled(false);
        SaveManager.Instance.PrefsSave.FastMode = FastModeType.Instant;
        foreach (string epoch in EpochModel.AllEpochIds) SaveManager.Instance.Progress.ObtainEpochOverride(epoch, EpochState.Revealed);
        foreach (var encounter in ModelDb.AllEncounters) SaveManager.Instance.Progress.GetOrCreateEncounterStats(encounter.Id);
        var model = ModelDb.AllCharacters.Single(c => c.Id.Entry == character);
        SaveManager.Instance.Progress.GetOrCreateCharacterStats(model.Id).TotalLosses = 100;
        _combatEnded = false;
        _combatWon = false;
        _decisions = [];
        _revision++;
        _run = await NGame.Instance!.StartNewSingleplayerRun(model, false, ActModel.GetDefaultList(), [], seed, GameMode.Standard, 10);
        return await Observe();
    }

    private static async Task<object> Step(JsonElement parameters)
    {
        if (parameters.GetProperty("revision").GetInt64() != _revision) throw new ArgumentException("Stale decision.");
        int index = parameters.GetProperty("action").GetInt32();
        if (index < 0 || index >= _decisions.Length) throw new ArgumentOutOfRangeException(nameof(parameters));
        var decision = _decisions[index];
        _decisions = [];
        decision.Execute();
        for (int i = 0; i < Execution.StepFrames; i++) await Frame();
        return await Observe();
    }

    private static async Task<object> Observe()
    {
        if (_run == null) throw new InvalidOperationException("Reset is required.");
        var timer = Stopwatch.StartNew();
        string previous = "";
        int stableFrames = 0;
        while (true)
        {
            bool inCombat = CombatManager.Instance.IsInProgress;
            bool dead = _run.IsGameOver && !inCombat && !RunManager.Instance.ActionExecutor.IsRunning;
            bool victory = RunManager.Instance.WinTime > 0;
            bool combatComplete = _scope == "first_combat" && _combatEnded;
            if (dead || victory || combatComplete)
            {
                _revision++;
                _decisions = [];
                return Snapshot(true, !dead && (victory || (combatComplete && _combatWon)));
            }
            var available = Decisions.Capture(_run);
            string signature = string.Join('|', available.Select(a => a.Key));
            stableFrames = available.Length > 0 && signature == previous ? stableFrames + 1 : 0;
            previous = signature;
            if (stableFrames >= Execution.SettleFrames)
            {
                _revision++;
                _decisions = available;
                return Snapshot(false, false);
            }
            if (timer.Elapsed.TotalSeconds > 30) throw new TimeoutException($"No stable decision: {Decisions.ScreenName}, floor {_run.TotalFloor}.");
            await Frame();
        }
    }

    private static object Snapshot(bool terminated, bool victory)
    {
        var timer = Stopwatch.StartNew();
        var observation = Observation.Capture(_run!);
        var actions = _decisions.Select(a => a.Visible).ToArray();
        _timings["observation_ms"] = timer.Elapsed.TotalMilliseconds;
        timer.Restart();
        string? audit = AuditEnabled ? Audit.Capture(_run!) : null;
        _timings["audit_ms"] = timer.Elapsed.TotalMilliseconds;
        return new { revision = _revision, observation, actions, terminated, victory, scope = _scope, audit };
    }

    private static async Task Frame()
    {
        var timer = Stopwatch.StartNew();
        await Tree.ToSignal(Tree, SceneTree.SignalName.ProcessFrame);
        _timings["frame_wait_ms"] = _timings.GetValueOrDefault("frame_wait_ms") + timer.Elapsed.TotalMilliseconds;
    }
}
