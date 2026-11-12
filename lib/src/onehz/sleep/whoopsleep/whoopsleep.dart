// Entry point of the cracked-whoop five-head sleep stager.
//
// `whoopSleepStages` stages a sleep window from the same positional 1 Hz rows
// and beats that `segmentSleep` receives, and returns one label per second of
// the window ('wake' | 'light' | 'deep' | 'rem'), or null when the window has
// too few beats for the model (the caller then keeps its own stager).
//
// The weights are embedded (whoopsleep_weights.g.dart) so every isolate can
// load them without asset plumbing; they are decoded once per isolate.

import 'dart:convert';
import 'dart:typed_data';

import 'features.dart';
import 'model.dart';
import 'whoopsleep_weights.g.dart';

export 'model.dart' show WhoopSleepModel, WsPrediction, kWsStages;

/// Minimum accepted beats per hour of window before the model is used. The
/// model was only trained on nights with beat intervals throughout; a
/// window with less than ~a quarter of normal beat coverage keeps the
/// rule-based stager.
const int kWhoopSleepMinBeatsPerHour = 800;

WhoopSleepModel? _model;

/// The embedded model, decoded on first use in this isolate.
WhoopSleepModel whoopSleepModel() => _model ??= WhoopSleepModel.fromBytes(
    Uint8List.fromList(base64.decode(kWhoopSleepWeightsB64.join())));

/// Name of the embedded model (e.g. for provenance in `inputs_used`).
const String kWhoopSleepModelName = kWhoopSleepWeightsName;

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
