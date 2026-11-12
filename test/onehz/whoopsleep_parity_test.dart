// Parity of the Dart five-head stager with the cracked-whoop Python reference.
//
// Fixtures come from cracked-whoop `scripts/dart_parity_fixtures.py` (they can
// hold personal data, so they are not committed). Point the test at them:
//   WHOOPSLEEP_FIXTURES=/path/to/parity dart test test/whoopsleep_parity_test.dart
// The embedded model is used unless WHOOPSLEEP_MODEL names a blob file.

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:openstrap_analytics/src/onehz/sleep/whoopsleep/features.dart';
import 'package:openstrap_analytics/src/onehz/sleep/whoopsleep/whoopsleep.dart';
import 'package:test/test.dart';

double _d(Object? v) => v == null ? double.nan : (v as num).toDouble();

/// Max abs difference, counting NaN-vs-number as infinite.
double _maxDiff(List<double> a, List<Object?> b) {
  var m = 0.0;
  for (var i = 0; i < a.length; i++) {
    final x = a[i], y = _d(b[i]);
    if (x.isNaN && y.isNaN) continue;
    if (x.isNaN != y.isNaN) return double.infinity;
    m = math.max(m, (x - y).abs() / math.max(1.0, y.abs()));
  }
  return m;
}

void main() {
  final dir = Platform.environment['WHOOPSLEEP_FIXTURES'];
  final files = dir == null
      ? <File>[]
      : (Directory(dir)
          .listSync()
          .whereType<File>()
          .where((f) => f.path.endsWith('.json'))
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path)));
  final blobPath = Platform.environment['WHOOPSLEEP_MODEL'];
  final model = blobPath != null
      ? WhoopSleepModel.fromBytes(File(blobPath).readAsBytesSync())
      : whoopSleepModel();

  for (final f in files) {
    test('parity ${f.uri.pathSegments.last}', () {
      final doc = jsonDecode(f.readAsStringSync()) as Map<String, dynamic>;
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
      final ref = doc['night'] as Map<String, dynamic>;
      expect(night.rrMs.length, (ref['rr_ms'] as List).length,
          reason: 'beat count');
      expect(_maxDiff(night.rrT, ref['rr_t'] as List), lessThan(1e-9),
          reason: 'beat times');
      final acc = Float64List(night.seconds * 3);
      for (var i = 0; i < night.seconds; i++) {
        acc[3 * i] = night.ax[i];
        acc[3 * i + 1] = night.ay[i];
        acc[3 * i + 2] = night.az[i];
      }
      expect(_maxDiff(acc, ref['accel'] as List), lessThan(1e-12),
          reason: 'accel');

      final cols = model.features(night);
      final want = doc['features'] as Map<String, dynamic>;
      final bad = <String>[];
      for (final name in want.keys) {
        final got = cols[name];
        if (got == null) {
          bad.add('$name missing');
          continue;
        }
        final d = _maxDiff(got, want[name] as List);
        if (d > 1e-6) bad.add('$name ${d.toStringAsExponential(2)}');
      }
      expect(bad, isEmpty, reason: 'features');

      final sw = Stopwatch()..start();
      final dbg = <String, Float64List>{};
      final pred = model.predict(night, cols: cols, debug: dbg);
      final ms = sw.elapsedMilliseconds;
      final heads = doc['heads'] as Map<String, dynamic>;
      for (final k in ['epoch', 'context', 'h2', 'h3', 'h4', 'h5']) {
        print(
            '  $k ${_maxDiff(dbg[k]!, heads[k] as List).toStringAsExponential(2)}');
      }
      final fin = heads['final'] as List;
      final d = _maxDiff(pred.proba, fin);
      var flips = 0;
      final st = pred.stages;
      for (var e = 0; e < pred.epochs; e++) {
        var k = 0;
        for (var c = 1; c < 4; c++) {
          if (_d(fin[e * 4 + c]) > _d(fin[e * 4 + k])) k = c;
        }
        if (k != st[e]) flips++;
      }
      print('${f.uri.pathSegments.last}: ${pred.epochs} epochs, '
          'max prob diff ${d.toStringAsExponential(2)}, $flips stage flips, ${ms} ms');
      expect(d, lessThan(1e-4));
      expect(flips, lessThanOrEqualTo((pred.epochs * 0.002).ceil()));
    });
  }
}
