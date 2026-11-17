// Parity with the cracked-whoop round-12 Python reference
// (scripts/r12_dart_fixtures.py): one emulated WHOOP hour from a young
// (Fantasia) and an older (SLPDB) PSG record.
import 'dart:convert';
import 'dart:io';

import 'package:openstrap_analytics/onehz.dart';
import 'package:test/test.dart';

List<double> _d(dynamic xs) => [for (final x in xs as List) (x as num).toDouble()];

void main() {
  final dir = Directory('test/onehz/fixtures/rsa_preserving');
  final files = dir.listSync().whereType<File>().toList()
    ..sort((a, b) => a.path.compareTo(b.path));
  test('fixtures present', () => expect(files, isNotEmpty));

  for (final f in files) {
    final fx = jsonDecode(f.readAsStringSync()) as Map<String, dynamic>;
    final rr = _d(fx['rr_ms']), ts = _d(fx['rr_ts_ms']);
    final a = fx['start_sec'] as int, b = fx['end_sec'] as int;
    final ex = fx['expect'] as Map<String, dynamic>;

    group(fx['name'], () {
      test('chain adjacency', () {
        final c = chainBeats(rr, ts);
        final adj = c.missing.skip(1).where((m) => m == 0).length / (rr.length - 1);
        expect(adj, closeTo((ex['missing_zero_frac'] as num).toDouble(), 1e-12));
      });

      test('nightly RMSSD', () {
        final m = sleepSessionRmssdRsa(rr, ts, startSec: a, endSec: b);
        expect(m.present, isTrue);
        expect(m.value!.rmssd, closeTo((ex['rmssd_ms'] as num).toDouble(), 1e-6));
      });

      test('per-epoch and nightly breathing rate', () {
        final ep = sleepRespEpochsLs(rr, ts, startSec: a, endSec: b);
        final want = ex['resp_epochs'] as List;
        expect(ep.length, want.length);
        var mismatches = 0;
        for (var i = 0; i < ep.length; i++) {
          final w = want[i];
          if (w == null) {
            if (!ep[i].isNaN) mismatches++;
          } else if (ep[i].isNaN || (ep[i] - (w as num)).abs() > 1e-6) {
            mismatches++;
          }
        }
        expect(mismatches, 0);
        final m = sleepRespRateLs(rr, ts, startSec: a, endSec: b);
        expect(m.value!.brpm, closeTo((ex['resp_brpm'] as num).toDouble(), 1e-6));
      });
    });
  }

  test('no beats -> absent, not zero', () {
    expect(sleepSessionRmssdRsa([], [], startSec: 1, endSec: 2).present, isFalse);
    expect(sleepRespRateLs([], [], startSec: 1, endSec: 2).present, isFalse);
  });
}
