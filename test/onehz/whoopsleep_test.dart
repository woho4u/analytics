// The cracked-whoop five-head stager (lib/src/onehz/sleep/whoopsleep/).
//
// 1. Parity: a synthetic night whose expected probabilities were produced by
//    the Python reference pipeline (cracked-whoop
//    `scripts/dart_parity_fixtures.py`, model v8_five_heads). Every stage and
//    every probability must match.
// 2. Wiring: `segmentSleep(whoopSleep: true)` on the anonymised real night of
//    real_night_cardio_stager_test.dart stages the window with the model and
//    yields a plausible night.

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:openstrap_analytics/onehz.dart';
import 'package:openstrap_analytics/src/onehz/sleep/whoopsleep/features.dart';
import 'package:openstrap_analytics/src/onehz/sleep/whoopsleep/whoopsleep.dart';
import 'package:openstrap_analytics/src/onehz/sleep/whoopsleep/whoopsleep_weights.g.dart';
import 'package:test/test.dart';

double _d(Object? v) => v == null ? double.nan : (v as num).toDouble();

void main() {
  test('five-head model reproduces the Python reference on a synthetic night',
      () {
    final raw = File('test/onehz/fixtures/whoopsleep_synthetic.json.gz')
        .readAsBytesSync();
    final doc =
        jsonDecode(utf8.decode(gzip.decode(raw))) as Map<String, dynamic>;
    final inp = doc['inputs'] as Map<String, dynamic>;
    List<double> dl(String k) => [for (final v in inp[k] as List) _d(v)];
    final night = assembleNight(
      startSec: inp['start'] as int,
      endSec: inp['end'] as int,
      tsSec: [for (final v in inp['ts'] as List) v as int],
      x: dl('x'),
      y: dl('y'),
      z: dl('z'),
      valid: [for (final v in inp['valid'] as List) v as bool],
      hr: dl('hr'),
      beatTsMs: dl('beat_ts_ms'),
      beatRrMs: dl('rr_ms'),
    );
    expect(night.rrMs.length,
        (doc['night'] as Map<String, dynamic>)['rr_ms_count'] as int);
    final pred = whoopSleepModel().predict(night);
    final want = (doc['heads'] as Map<String, dynamic>)['final'] as List;
    expect(pred.proba.length, want.length);
    var maxDiff = 0.0;
    for (var i = 0; i < want.length; i++) {
      maxDiff = math.max(maxDiff, (pred.proba[i] - _d(want[i])).abs());
    }
    expect(maxDiff, lessThan(1e-5));
    final stages = pred.stages;
    for (var e = 0; e < pred.epochs; e++) {
      var k = 0;
      for (var c = 1; c < 4; c++) {
        if (_d(want[e * 4 + c]) > _d(want[e * 4 + k])) k = c;
      }
      expect(stages[e], k, reason: 'epoch $e');
    }
  });

  test('segmentSleep(whoopSleep: true) stages a real night with the model', () {
    final onehz = File('test/onehz/fixtures/real_night_2026_07_onehz.csv');
    final rrFile = File('test/onehz/fixtures/real_night_2026_07_rr.csv');
    final hr = <double>[];
    final accel = <AccelSample>[];
    // Rebase onto a fixed absolute time; the fixture starts at 0.
    const t0 = 1780000000;
    for (final l in onehz.readAsLinesSync().skip(1)) {
      if (l.trim().isEmpty) continue;
      final p = l.split(',');
      hr.add(double.parse(p[1]));
      accel.add(AccelSample((t0 + int.parse(p[0])) * 1000.0, double.parse(p[2]),
          double.parse(p[3]), double.parse(p[4])));
    }
    final rrMs = <double>[], rrTs = <double>[];
    for (final l in rrFile.readAsLinesSync().skip(1)) {
      if (l.trim().isEmpty) continue;
      final p = l.split(',');
      rrTs.add(t0 * 1000.0 + double.parse(p[0]));
      rrMs.add(double.parse(p[1]));
    }
    final window = (onsetSec: t0, offsetSec: t0 + hr.length);
    final rule = segmentSleep(accel, hr,
        rrMs: rrMs, rrTsMs: rrTs, forcedWindow: window, tzOffsetSec: 0);
    final model = segmentSleep(accel, hr,
        rrMs: rrMs,
        rrTsMs: rrTs,
        forcedWindow: window,
        tzOffsetSec: 0,
        whoopSleep: true);
    int mins(SleepSegmentation s, String st) =>
        s.stages4.where((x) => x == st).length ~/ 60;
    String show(SleepSegmentation s) => [
          for (final st in ['wake', 'light', 'deep', 'rem', 'unobserved'])
            '$st=${mins(s, st)}'
        ].join(' ');
    print('rule:  ${show(rule)}');
    print('model: ${show(model)}');
    // The model changed the staging, and every stage is present.
    expect(model.stages4, isNot(equals(rule.stages4)));
    for (final st in ['light', 'deep', 'rem']) {
      expect(mins(model, st), greaterThan(10), reason: st);
    }
    // Totals are re-derived from the model's labels.
    expect(model.remSec, model.stages4.where((x) => x == 'rem').length);
    expect(model.deepSec, model.stages4.where((x) => x == 'deep').length);
  });

  test('a swapped-in model is used, and a broken one is refused', () {
    final embedded = whoopSleepModel();
    expect(whoopSleepActiveModelName(), kWhoopSleepModelName);
    // Re-encode the embedded blob under a different name.
    final bytes =
        Uint8List.fromList(base64.decode(kWhoopSleepWeightsB64.join()));
    final hlen = ByteData.sublistView(bytes, 4, 8).getUint32(0, Endian.little);
    final header = utf8.decode(bytes.sublist(8, 8 + hlen));
    final renamed = header.replaceFirst(
        '"model":"${embedded.name}"', '"model":"${embedded.name}_x"');
    expect(renamed, isNot(header), reason: 'header layout changed');
    expect(renamed.length, header.length + 2);
    final hb = utf8.encode(renamed);
    // Keep the 8-byte data alignment: pad the header with spaces.
    var pad = (8 - (8 + hb.length) % 8) % 8;
    final origPad = (8 - (8 + hlen) % 8) % 8;
    final hdr = [...hb, ...List.filled(pad, 0x20)];
    final out = BytesBuilder()
      ..add(bytes.sublist(0, 4))
      ..add((ByteData(4)..setUint32(0, hdr.length, Endian.little))
          .buffer
          .asUint8List())
      ..add(hdr)
      ..add(bytes.sublist(8 + hlen + origPad));
    final custom = out.toBytes();
    expect(whoopSleepSelfTest(custom), isNull);
    expect(useWhoopSleepModel(custom, key: 'a'), isNull);
    expect(whoopSleepActiveModelName(), '${embedded.name}_x');
    expect(identical(whoopSleepModel(), embedded), isFalse);
    // Same key: no reload.
    final again = whoopSleepModel();
    expect(useWhoopSleepModel(custom, key: 'a'), isNull);
    expect(identical(whoopSleepModel(), again), isTrue);
    // A truncated blob is refused and the embedded model is back in use.
    expect(useWhoopSleepModel(Uint8List.sublistView(custom, 0, 5000), key: 'b'),
        isNotNull);
    expect(identical(whoopSleepModel(), embedded), isTrue);
    expect(whoopSleepSelfTest(Uint8List(16)), isNotNull);
    expect(useWhoopSleepModel(null), isNull);
    expect(whoopSleepActiveModelName(), kWhoopSleepModelName);
  });
}
