// Entry point of the cracked-whoop five-head sleep stager.
//
// `whoopSleepStages` stages a sleep window from the same positional 1 Hz rows
// and beats that `segmentSleep` receives, and returns one label per second of
// the window ('wake' | 'light' | 'deep' | 'rem'), or null when the window has
// too few beats for the model (the caller then keeps its own stager).
//
// The weights are embedded (whoopsleep_weights.g.dart) so every isolate can
// load them without asset plumbing; they are decoded once per isolate.
//
// A host app can swap in a newer model of the same architecture (e.g. one it
// downloaded) with [useWhoopSleepModel]. The swap is per isolate, so the host
// calls it in every isolate that stages sleep; a model that fails to load or
// fails [whoopSleepSelfTest] is refused and the embedded one stays in use.

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'features.dart';
import 'model.dart';
import 'whoopsleep_weights.g.dart';

export 'model.dart'
    show WhoopSleepModel, WsPrediction, kWsStages, kWhoopSleepBlobFormat;

/// Minimum accepted beats per hour of window before the model is used. The
/// model was only trained on nights with beat intervals throughout; a
/// window with less than ~a quarter of normal beat coverage keeps the
/// rule-based stager.
const int kWhoopSleepMinBeatsPerHour = 800;

WhoopSleepModel? _embedded;
WhoopSleepModel? _custom;
Object? _customKey;

/// The model in use in this isolate: the one installed with
/// [useWhoopSleepModel], else the embedded one (decoded on first use).
WhoopSleepModel whoopSleepModel() =>
    _custom ??
    (_embedded ??= WhoopSleepModel.fromBytes(
        Uint8List.fromList(base64.decode(kWhoopSleepWeightsB64.join()))));

/// Name of the embedded model (e.g. for provenance in `inputs_used`).
const String kWhoopSleepModelName = kWhoopSleepWeightsName;

/// Name of the model [whoopSleepStages] uses in this isolate.
String whoopSleepActiveModelName() => _custom?.name ?? kWhoopSleepModelName;

/// Use the model in [bytes] (a blob written by cracked-whoop's
/// `export_dart_model.py`) for this isolate; null goes back to the embedded
/// model. [key] identifies the bytes: a call with the key already installed
/// is a no-op, so a host can call this before every staging pass. Returns
/// null on success, else why the model was refused (the embedded model is
/// then used).
String? useWhoopSleepModel(Uint8List? bytes, {Object? key}) {
  if (bytes == null) {
    _custom = null;
    _customKey = null;
    return null;
  }
  if (key != null && key == _customKey && _custom != null) return null;
  final (model, error) = _loadChecked(bytes);
  _custom = model;
  _customKey = model == null ? null : key;
  return error;
}

/// Null when [bytes] hold a model this code can run: it parses and stages a
/// short synthetic night to finite probabilities. Else the reason it can't.
String? whoopSleepSelfTest(Uint8List bytes) => _loadChecked(bytes).$2;

(WhoopSleepModel?, String?) _loadChecked(Uint8List bytes) {
  try {
    final model = WhoopSleepModel.fromBytes(bytes);
    final pred = model.predict(_selfTestNight());
    if (pred.epochs != _selfTestSec ~/ kEpochSec) {
      return (null, 'self-test staged ${pred.epochs} epochs');
    }
    for (final v in pred.proba) {
      if (!v.isFinite) return (null, 'self-test produced a non-finite value');
    }
    return (model, null);
  } catch (e) {
    return (null, '$e');
  }
}

const int _selfTestSec = 20 * 60;

/// Twenty minutes of a still wrist and a ~60 bpm heart with breathing-like
/// variation: enough to exercise every head.
WsNight _selfTestNight() {
  const n = _selfTestSec;
  final ts = List<int>.generate(n, (i) => i);
  final beatsTs = <double>[], beatsRr = <double>[];
  var t = 0.0;
  for (var k = 0; t < n * 1000.0; k++) {
    final rr = 1000 + 40 * math.sin(k * 2 * math.pi / 4.5) + (k % 7) * 3.0;
    t += rr;
    beatsTs.add(t);
    beatsRr.add(rr);
  }
  return assembleNight(
    startSec: 0,
    endSec: n,
    tsSec: ts,
    x: List<double>.generate(n, (i) => 0.02 * math.sin(i / 50)),
    y: List<double>.filled(n, 0.1),
    z: List<double>.filled(n, -0.99),
    valid: List<bool>.filled(n, true),
    hr: List<double>.filled(n, 60),
    beatTsMs: beatsTs,
    beatRrMs: beatsRr,
  );
}

/// Per-second stages for [startSec, endSec), or null if the model abstains.
List<String>? whoopSleepStages({
  required int startSec,
  required int endSec,
  required List<int> tsSec,
  required List<double> x,
  required List<double> y,
  required List<double> z,
  required List<bool> valid,
  required List<double> hr,
  required List<double> beatTsMs,
  required List<double> beatRrMs,
}) {
  final len = endSec - startSec;
  if (len < 2 * kEpochSec) return null;
  final night = assembleNight(
    startSec: startSec,
    endSec: endSec,
    tsSec: tsSec,
    x: x,
    y: y,
    z: z,
    valid: valid,
    hr: hr,
    beatTsMs: beatTsMs,
    beatRrMs: beatRrMs,
  );
  if (night.rrMs.length < kWhoopSleepMinBeatsPerHour * len / 3600) return null;
  final stages = whoopSleepModel().predict(night).stages;
  // The trailing partial epoch (< 30 s) takes the last epoch's stage.
  return List<String>.generate(len, (i) {
    final e = i ~/ kEpochSec;
    return kWsStages[stages[e < stages.length ? e : stages.length - 1]];
  }, growable: false);
}
