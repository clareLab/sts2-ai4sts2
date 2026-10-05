using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using GodotObject = Godot.GodotObject;
using Label = System.Reflection.Emit.Label;
using MegaCrit.Sts2.Core.Models;

namespace ai4sts2;

internal sealed class Snapshot
{
    private const BindingFlags Instance = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
    private const BindingFlags Static = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
    private enum Kind : byte { Owned, Shared, Singleton }
    private static readonly Dictionary<Type, FieldInfo[]> Layouts = [];
    private static readonly Dictionary<Type, Kind> Kinds = [];
    private static readonly Dictionary<Type, FieldInfo[]> SelfFields = [];
    private static readonly Dictionary<Type, Action<Snapshot, object>> Encoders = [];
    private static readonly Dictionary<Type, Action<object, ulong[], int, object?[]>> Decoders = [];
    private static readonly MethodInfo WordMethod = typeof(Snapshot).GetMethod(nameof(Word), BindingFlags.Instance | BindingFlags.NonPublic)!;
    private static readonly MethodInfo ReferenceMethod = typeof(Snapshot).GetMethod(nameof(WriteReference), BindingFlags.Instance | BindingFlags.NonPublic)!;

    private readonly List<ulong> _words = [];
    private readonly List<object?> _references = [null];
    private readonly Dictionary<object, int> _ids = new(ReferenceEqualityComparer.Instance);
    private readonly HashSet<object> _roots = new(ReferenceEqualityComparer.Instance);
    private readonly List<object> _pending = [];
    private readonly List<int> _offsets = [];
    private readonly ulong[] _buffer;
    private readonly object?[] _table;
    private readonly object[] _targets;
    private readonly Action<object, ulong[], int, object?[]>?[] _decoders;

    internal int Objects => _targets.Length;
    internal int Words => _buffer.Length;

    internal Snapshot(params object[] roots)
    {
        foreach (object root in roots) _roots.Add(root);
        foreach (object root in roots) Reference(root);
        for (int index = 0; index < _pending.Count; index++)
        {
            object target = _pending[index];
            _offsets.Add(_words.Count);
            Type type = target.GetType();
            if (target is Array array && !type.IsSZArray)
            {
                Type element = type.GetElementType()!;
                foreach (object? item in array) Write(element, item);
                continue;
            }
            if (!Encoders.TryGetValue(type, out var encoder)) Encoders[type] = encoder = CompileEncoder(type);
            encoder(this, target);
        }
        _buffer = _words.ToArray();
        _table = _references.ToArray();
        _targets = _pending.ToArray();
        _decoders = new Action<object, ulong[], int, object?[]>?[_targets.Length];
        for (int index = 0; index < _targets.Length; index++)
        {
            Type type = _targets[index].GetType();
            if (type.IsArray && !type.IsSZArray) continue;
            if (!Decoders.TryGetValue(type, out var decoder)) Decoders[type] = decoder = CompileDecoder(type);
            _decoders[index] = decoder;
        }
        _words.Clear();
        _references.Clear();
        _ids.Clear();
        _roots.Clear();
        _pending.Clear();
    }

    internal void Restore()
    {
        for (int index = 0; index < _targets.Length; index++)
        {
            int offset = _offsets[index];
            if (_decoders[index] is { } decoder)
            {
                decoder(_targets[index], _buffer, offset, _table);
                continue;
            }
            var array = (Array)_targets[index];
            Type element = array.GetType().GetElementType()!;
            int[] indices = Enumerable.Range(0, array.Rank).Select(array.GetLowerBound).ToArray();
            for (int item = 0; item < array.Length; item++)
            {
                array.SetValue(Read(element, ref offset), indices);
                for (int dimension = array.Rank - 1; dimension >= 0; dimension--)
                    if (++indices[dimension] <= array.GetUpperBound(dimension)) break;
                    else indices[dimension] = array.GetLowerBound(dimension);
            }
        }
    }

    private void Word(ulong value) => _words.Add(value);

    private void WriteReference(object? value) => _words.Add((ulong)Reference(value));

    private int Reference(object? value)
    {
        if (value == null) return 0;
        if (_ids.TryGetValue(value, out int id)) return id;
        id = _references.Count;
        _references.Add(value);
        _ids.Add(value, id);
        if (value is Delegate callback)
        {
            foreach (Delegate part in callback.GetInvocationList())
                if (part.Target != null) Reference(part.Target);
        }
        else if (Owned(value)) _pending.Add(value);
        return id;
    }

    private bool Owned(object value)
    {
        if (value is AbstractModel { IsMutable: false }) return false;
        if (_roots.Contains(value)) return true;
        Type type = value.GetType();
        return Classify(type) switch
        {
            Kind.Owned => true,
            Kind.Singleton => !SelfFields[type].Any(field => ReferenceEquals(field.GetValue(null), value)),
            _ => false
        };
    }

    private static Kind Classify(Type type)
    {
        if (Kinds.TryGetValue(type, out Kind kind)) return kind;
        kind = Shared(type) ? Kind.Shared
            : type.IsArray ? Kind.Owned
            : Layout(type).Length == 0 ? Kind.Shared
            : Singletons(type).Length != 0 ? Kind.Singleton
            : Kind.Owned;
        return Kinds[type] = kind;
    }

    private static bool Shared(Type type) => type == typeof(string) || typeof(MemberInfo).IsAssignableFrom(type)
        || typeof(Delegate).IsAssignableFrom(type) || typeof(GodotObject).IsAssignableFrom(type) || Runtime(type)
        || type.Assembly == typeof(Snapshot).Assembly && type != typeof(Ledger);

    private static bool Runtime(Type type)
    {
        for (Type? current = type; current != null; current = current.BaseType)
            if (current == typeof(Task) || current == typeof(CancellationTokenSource) || current == typeof(Lock)
                || current == typeof(WaitHandle) || current == typeof(Thread) || current == typeof(TaskCompletionSource)
                || current == typeof(SynchronizationContext)
                || current.IsGenericType && current.GetGenericTypeDefinition() == typeof(TaskCompletionSource<>))
                return true;
        return false;
    }

    private static FieldInfo[] Singletons(Type type)
    {
        if (SelfFields.TryGetValue(type, out FieldInfo[]? fields)) return fields;
        var result = new List<FieldInfo>();
        for (Type? current = type; current != null && current != typeof(object); current = current.BaseType)
            if (!current.ContainsGenericParameters)
                foreach (FieldInfo field in current.GetFields(Static))
                    if (!field.FieldType.IsValueType && field.FieldType.IsAssignableFrom(type) && field.FieldType != typeof(object)) result.Add(field);
        return SelfFields[type] = result.ToArray();
    }

    private static FieldInfo[] Layout(Type type)
    {
        if (Layouts.TryGetValue(type, out FieldInfo[]? fields)) return fields;
        var result = new List<FieldInfo>();
        for (Type? current = type; current != null; current = current.BaseType)
            foreach (FieldInfo field in current.GetFields(Instance))
                if (!Runtime(field.FieldType) && !(field.Name == "_version" && current.Namespace == "System.Collections.Generic"))
                    result.Add(field);
        return Layouts[type] = result.ToArray();
    }

    private void Write(Type type, object? value)
    {
        if (!type.IsValueType) { WriteReference(value); return; }
        if (Nullable.GetUnderlyingType(type) is { } underlying)
        {
            Word(value == null ? 0u : 1u);
            if (value != null) Write(underlying, value);
            return;
        }
        if (type.IsEnum) { Write(Enum.GetUnderlyingType(type), value); return; }
        switch (Type.GetTypeCode(type))
        {
            case TypeCode.Boolean: Word((bool)value! ? 1u : 0u); return;
            case TypeCode.Char: Word((char)value!); return;
            case TypeCode.Byte:
            case TypeCode.UInt16:
            case TypeCode.UInt32:
            case TypeCode.UInt64: Word(Convert.ToUInt64(value)); return;
            case TypeCode.SByte:
            case TypeCode.Int16:
            case TypeCode.Int32:
            case TypeCode.Int64: Word(unchecked((ulong)Convert.ToInt64(value))); return;
            case TypeCode.Single: Word(BitConverter.SingleToUInt32Bits((float)value!)); return;
            case TypeCode.Double: Word(BitConverter.DoubleToUInt64Bits((double)value!)); return;
        }
        if (type == typeof(nint) || type == typeof(nuint)) throw new NotSupportedException("Native pointers cannot be captured.");
        foreach (FieldInfo field in Layout(type)) Write(field.FieldType, value == null ? null : field.GetValue(value));
    }

    private object? Read(Type type, ref int offset)
    {
        if (!type.IsValueType) return _table[checked((int)_buffer[offset++])];
        if (Nullable.GetUnderlyingType(type) is { } underlying)
            return _buffer[offset++] == 0 ? null : Read(underlying, ref offset);
        if (type.IsEnum) return Enum.ToObject(type, Read(Enum.GetUnderlyingType(type), ref offset)!);
        TypeCode code = Type.GetTypeCode(type);
        if (code is >= TypeCode.Boolean and <= TypeCode.Double)
        {
            ulong word = _buffer[offset++];
            return code switch
            {
                TypeCode.Boolean => word != 0,
                TypeCode.Char => (char)word,
                TypeCode.Byte => (byte)word,
                TypeCode.UInt16 => (ushort)word,
                TypeCode.UInt32 => (uint)word,
                TypeCode.UInt64 => word,
                TypeCode.SByte => unchecked((sbyte)word),
                TypeCode.Int16 => unchecked((short)word),
                TypeCode.Int32 => unchecked((int)word),
                TypeCode.Int64 => unchecked((long)word),
                TypeCode.Single => BitConverter.UInt32BitsToSingle((uint)word),
                TypeCode.Double => BitConverter.UInt64BitsToDouble(word),
                _ => throw new InvalidOperationException("Unknown scalar.")
            };
        }
        object instance = RuntimeHelpers.GetUninitializedObject(type);
        foreach (FieldInfo field in Layout(type)) field.SetValue(instance, Read(field.FieldType, ref offset));
        return instance;
    }

    private static Action<Snapshot, object> CompileEncoder(Type type)
    {
        var method = new DynamicMethod("Capture", typeof(void), [typeof(Snapshot), typeof(object)], typeof(Snapshot).Module, true);
        ILGenerator il = method.GetILGenerator();
        if (type.IsArray)
        {
            Type element = type.GetElementType()!;
            LocalBuilder index = il.DeclareLocal(typeof(int));
            Label body = il.DefineLabel(), test = il.DefineLabel();
            il.Emit(OpCodes.Br, test);
            il.MarkLabel(body);
            Encode(il, element, () =>
            {
                Target(il, type, OpCodes.Ldarg_1);
                il.Emit(OpCodes.Ldloc, index);
                il.Emit(OpCodes.Ldelem, element);
            });
            Increment(il, index);
            il.MarkLabel(test);
            il.Emit(OpCodes.Ldloc, index);
            Target(il, type, OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldlen);
            il.Emit(OpCodes.Conv_I4);
            il.Emit(OpCodes.Blt, body);
        }
        else
            foreach (FieldInfo field in Layout(type))
                Encode(il, field.FieldType, () => { Target(il, type, OpCodes.Ldarg_1); il.Emit(OpCodes.Ldfld, field); });
        il.Emit(OpCodes.Ret);
        return method.CreateDelegate<Action<Snapshot, object>>();
    }

    private static void Encode(ILGenerator il, Type type, Action load)
    {
        if (!type.IsValueType)
        {
            il.Emit(OpCodes.Ldarg_0);
            load();
            il.Emit(OpCodes.Call, ReferenceMethod);
            return;
        }
        if (type.IsEnum) { Encode(il, Enum.GetUnderlyingType(type), load); return; }
        if (Nullable.GetUnderlyingType(type) is { } underlying)
        {
            LocalBuilder nullable = il.DeclareLocal(type);
            Label done = il.DefineLabel();
            load();
            il.Emit(OpCodes.Stloc, nullable);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloca, nullable);
            il.Emit(OpCodes.Call, type.GetProperty("HasValue")!.GetMethod!);
            il.Emit(OpCodes.Conv_U8);
            il.Emit(OpCodes.Call, WordMethod);
            il.Emit(OpCodes.Ldloca, nullable);
            il.Emit(OpCodes.Call, type.GetProperty("HasValue")!.GetMethod!);
            il.Emit(OpCodes.Brfalse, done);
            Encode(il, underlying, () => { il.Emit(OpCodes.Ldloca, nullable); il.Emit(OpCodes.Call, type.GetProperty("Value")!.GetMethod!); });
            il.MarkLabel(done);
            return;
        }
        TypeCode code = Type.GetTypeCode(type);
        if (code is >= TypeCode.Boolean and <= TypeCode.Double)
        {
            il.Emit(OpCodes.Ldarg_0);
            load();
            switch (code)
            {
                case TypeCode.Single:
                    il.Emit(OpCodes.Call, typeof(BitConverter).GetMethod(nameof(BitConverter.SingleToUInt32Bits))!);
                    il.Emit(OpCodes.Conv_U8);
                    break;
                case TypeCode.Double:
                    il.Emit(OpCodes.Call, typeof(BitConverter).GetMethod(nameof(BitConverter.DoubleToUInt64Bits))!);
                    break;
                case TypeCode.SByte:
                case TypeCode.Int16:
                case TypeCode.Int32:
                case TypeCode.Int64: il.Emit(OpCodes.Conv_I8); break;
                default: il.Emit(OpCodes.Conv_U8); break;
            }
            il.Emit(OpCodes.Call, WordMethod);
            return;
        }
        if (type == typeof(nint) || type == typeof(nuint)) throw new NotSupportedException("Native pointers cannot be captured.");
        LocalBuilder value = il.DeclareLocal(type);
        load();
        il.Emit(OpCodes.Stloc, value);
        foreach (FieldInfo field in Layout(type))
            Encode(il, field.FieldType, () => { il.Emit(OpCodes.Ldloca, value); il.Emit(OpCodes.Ldfld, field); });
    }

    private static Action<object, ulong[], int, object?[]> CompileDecoder(Type type)
    {
        var method = new DynamicMethod("Restore", typeof(void), [typeof(object), typeof(ulong[]), typeof(int), typeof(object[])], typeof(Snapshot).Module, true);
        ILGenerator il = method.GetILGenerator();
        if (type.IsArray)
        {
            Type element = type.GetElementType()!;
            LocalBuilder index = il.DeclareLocal(typeof(int));
            Label body = il.DefineLabel(), test = il.DefineLabel();
            il.Emit(OpCodes.Br, test);
            il.MarkLabel(body);
            Target(il, type, OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, index);
            Decode(il, element);
            il.Emit(OpCodes.Stelem, element);
            Increment(il, index);
            il.MarkLabel(test);
            il.Emit(OpCodes.Ldloc, index);
            Target(il, type, OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldlen);
            il.Emit(OpCodes.Conv_I4);
            il.Emit(OpCodes.Blt, body);
        }
        else
            foreach (FieldInfo field in Layout(type))
            {
                Target(il, type, OpCodes.Ldarg_0);
                Decode(il, field.FieldType);
                il.Emit(OpCodes.Stfld, field);
            }
        il.Emit(OpCodes.Ret);
        return method.CreateDelegate<Action<object, ulong[], int, object?[]>>();
    }

    private static void Decode(ILGenerator il, Type type)
    {
        if (!type.IsValueType)
        {
            il.Emit(OpCodes.Ldarg_3);
            Next(il);
            il.Emit(OpCodes.Conv_I4);
            il.Emit(OpCodes.Ldelem_Ref);
            il.Emit(OpCodes.Castclass, type);
            return;
        }
        if (type.IsEnum) { Decode(il, Enum.GetUnderlyingType(type)); return; }
        if (Nullable.GetUnderlyingType(type) is { } underlying)
        {
            LocalBuilder nullable = il.DeclareLocal(type);
            Label empty = il.DefineLabel(), done = il.DefineLabel();
            Next(il);
            il.Emit(OpCodes.Brfalse, empty);
            Decode(il, underlying);
            il.Emit(OpCodes.Newobj, type.GetConstructor([underlying])!);
            il.Emit(OpCodes.Br, done);
            il.MarkLabel(empty);
            il.Emit(OpCodes.Ldloca, nullable);
            il.Emit(OpCodes.Initobj, type);
            il.Emit(OpCodes.Ldloc, nullable);
            il.MarkLabel(done);
            return;
        }
        TypeCode code = Type.GetTypeCode(type);
        if (code is >= TypeCode.Boolean and <= TypeCode.Double)
        {
            Next(il);
            switch (code)
            {
                case TypeCode.Boolean:
                case TypeCode.Byte: il.Emit(OpCodes.Conv_U1); break;
                case TypeCode.SByte: il.Emit(OpCodes.Conv_I1); break;
                case TypeCode.Char:
                case TypeCode.UInt16: il.Emit(OpCodes.Conv_U2); break;
                case TypeCode.Int16: il.Emit(OpCodes.Conv_I2); break;
                case TypeCode.UInt32: il.Emit(OpCodes.Conv_U4); break;
                case TypeCode.Int32: il.Emit(OpCodes.Conv_I4); break;
                case TypeCode.Single:
                    il.Emit(OpCodes.Conv_U4);
                    il.Emit(OpCodes.Call, typeof(BitConverter).GetMethod(nameof(BitConverter.UInt32BitsToSingle))!);
                    break;
                case TypeCode.Double:
                    il.Emit(OpCodes.Call, typeof(BitConverter).GetMethod(nameof(BitConverter.UInt64BitsToDouble))!);
                    break;
            }
            return;
        }
        LocalBuilder value = il.DeclareLocal(type);
        foreach (FieldInfo field in Layout(type))
        {
            il.Emit(OpCodes.Ldloca, value);
            Decode(il, field.FieldType);
            il.Emit(OpCodes.Stfld, field);
        }
        il.Emit(OpCodes.Ldloc, value);
    }

    private static void Target(ILGenerator il, Type type, OpCode argument)
    {
        il.Emit(argument);
        il.Emit(type.IsValueType ? OpCodes.Unbox : OpCodes.Castclass, type);
    }

    private static void Next(ILGenerator il)
    {
        il.Emit(OpCodes.Ldarg_1);
        il.Emit(OpCodes.Ldarg_2);
        il.Emit(OpCodes.Ldelem_I8);
        il.Emit(OpCodes.Ldarg_2);
        il.Emit(OpCodes.Ldc_I4_1);
        il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Starg_S, (byte)2);
    }

    private static void Increment(ILGenerator il, LocalBuilder index)
    {
        il.Emit(OpCodes.Ldloc, index);
        il.Emit(OpCodes.Ldc_I4_1);
        il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Stloc, index);
    }
}
