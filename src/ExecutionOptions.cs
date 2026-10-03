using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Saves;

namespace ai4sts2;

internal sealed record ExecutionOptions(int Fps = 60, int SettleFrames = 3, int StepFrames = 2, bool NonInteractive = false)
{
    internal static ExecutionOptions Load(JsonSerializerOptions json)
    {
        var options = JsonSerializer.Deserialize<ExecutionOptions>(System.Environment.GetEnvironmentVariable("AI4STS2_EXECUTION") ?? "{}", json)!;
        if (options.Fps < 0 || options.SettleFrames < 1 || options.StepFrames < 1)
            throw new ArgumentException("Invalid execution options.");
        return options;
    }

    internal void Apply()
    {
        SaveManager.Instance.SettingsSave.FpsLimit = Fps;
        Engine.MaxFps = Fps;
        NonInteractiveMode.AutoSlayerCheck = () => NonInteractive;
    }
}
