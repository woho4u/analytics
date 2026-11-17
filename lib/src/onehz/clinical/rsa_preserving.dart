// SLEEP HRV + BREATHING RATE THAT KEEP REAL BREATHING SWINGS (cracked-whoop
// round 12, results/r12/REPORT.md in the cracked-whoop repo).
//
// Why: the shipped nightly headline ([sleepSessionRmssdDetail]) runs correctRr
// at alpha 5.2 plus Malik 20 %. On a clean ECG from a young sleeper those
// filters flag the largest genuine respiratory swings as artifacts (RMSSD
// 93 -> 74 -> 67 ms), the +1000 ms seam tolerance lets differences across a
// dropped beat through (inflates older, low-HRV nights), and the ACF1 jitter
// gate refuses about a third of nights. Scored on PSG ECG truth with a WHOOP
// 4.0 strap emulator, locked test split: ln-RMSSD error 0.41 -> 0.20.
//
// What this does instead:
//   * beat adjacency is rebuilt from the record stamps ([chainBeats]): each
//     beat lies in the ~1 s before its record's time, so prev + RR must land
//     there. A successive difference only counts between two beats that truly
//     follow each other;
//   * artifact flags from correctRr at alpha 6.5 (less sensitive);
//   * mean of 5-min window RMSSDs (>= 20 differences each), no jitter refusal.
//     The coverage-ceiling refusal is kept.
//
// Breathing ([sleepRespRateLs]): Lomb-Scargle of clean beats in a 90 s window
// centred on every 30 s epoch, searched from 0.15 Hz (a 0.1 Hz floor locks onto
// Mayer waves) up to 0.95 x the window's beat Nyquist, edge peaks dropped,
// +-5-epoch rolling median, nightly value = median over the sleep session.
// Locked test nightly error 2.89 -> 1.78 br/min.
//
// Python reference: cracked-whoop src/whoopsleep/r12.py (nn_runs,
// window_rmssd, ls_resp_epochs, smooth_epochs) + substrate.chain_beats.

import 'dart:math' as math;

import '../foundations/rr_correction.dart';
import '../respiration/resp_rate.dart';
import '../types.dart';
import '../util.dart';
import 'hrv_time.dart';

/// correctRr sensitivity for the HRV path (the rest of the app keeps 5.2).
const double kRsaPreservingAlpha = 6.5;

/// Beats rebuilt from WHOOP records. [missing] is the number of beats judged
/// dropped just before each beat (0 = directly follows the previous one), or
/// -1 when the chain was re-anchored (unknown gap).
class ChainedBeats {
  final List<double> tSec;
  final List<int> missing;
  const ChainedBeats(this.tSec, this.missing);
}

/// Port of `whoopsleep.substrate.chain_beats`: a beat's true time is the
/// previous beat's time plus 1-4 intervals of its own length, as long as that
/// lands in [recordSec - windowSec - 0.15, recordSec + 0.15]; otherwise the
/// chain re-anchors at the middle of the record's second.
ChainedBeats chainBeats(List<double> rrMs, List<double> recTsMs,
    {double windowSec = 1.0}) {
  const tol = 0.15;
  final n = rrMs.length;
  final t = List<double>.filled(n, 0);
  final miss = List<int>.filled(n, 0);
  var prev = -1e18;
  for (var i = 0; i < n; i++) {
    final rec = recTsMs[i] / 1000.0;
    final rr = rrMs[i] / 1000.0;
    final lo = rec - windowSec - tol, hi = rec + tol;
    var placed = false;
    for (var k = 0; k < 4; k++) {
      final c = prev + rr * (k + 1);
      if (lo <= c && c <= hi) {
        t[i] = c;
        miss[i] = k;
        placed = true;
        break;
      }
    }
    if (!placed) {
      t[i] = rec - windowSec / 2;
      miss[i] = -1;
    }
    prev = t[i];
  }
  return ChainedBeats(t, miss);
}

/// Malik filter as in the round-12 reference: |rr - centred 5-beat median|
/// <= 20 % of it (median over at least 3 beats at the edges), and 300-2000 ms.
List<bool> malikGood(List<double> rrMs) {
  final n = rrMs.length;
  final out = List<bool>.filled(n, false);
  for (var i = 0; i < n; i++) {
    final w = <double>[
      for (var j = math.max(0, i - 2); j <= math.min(n - 1, i + 2); j++) rrMs[j]
    ];
    if (w.length < 3) continue;
    final med = _median(w);
    final rr = rrMs[i];
    out[i] = (rr - med).abs() <= 0.2 * med && rr >= 300 && rr <= 2000;
  }
  return out;
}

double _median(List<double> xs) {
  final s = [...xs]..sort();
  final m = s.length ~/ 2;
  return s.length.isOdd ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

List<bool> _correctRrGood(List<double> rr, List<double> ts, double alpha) {
  final c = correctRr(rr, rrTsMs: ts, alpha: alpha);
  return [for (final k in c.classes) k == BeatClass.normal];
}

/// The beats of [rrMs]/[rrTsMs] whose record stamp is inside [startSec,
/// endSec), in input order.
({List<double> rr, List<double> ts}) _inSession(
    List<double> rrMs, List<double> rrTsMs, int startSec, int endSec) {
  final rr = <double>[], ts = <double>[];
  for (var i = 0; i < rrMs.length; i++) {
    final t = rrTsMs[i];
    if (t >= startSec * 1000.0 && t < endSec * 1000.0) {
      rr.add(rrMs[i]);
      ts.add(t);
    }
  }
  return (rr: rr, ts: ts);
}

/// Nightly sleep HRV (RMSSD, ms) that keeps genuine breathing swings. Same
/// inputs as [sleepSessionRmssdDetail]: raw RR and their record stamps (ms),
/// and the sleep session in epoch seconds.
Metric<SessionRmssd> sleepSessionRmssdRsa(
  List<double> rrMs,
  List<double> rrTsMs, {
  required int startSec,
  required int endSec,
  int windowSec = 300,
  int minDiffsPerWindow = kMinDiffsPerRmssdWindow,
  double alpha = kRsaPreservingAlpha,
}) {
  const inputs = ['rr_sleep_window'];
  if (startSec <= 0 ||
      endSec <= startSec ||
      rrMs.isEmpty ||
      rrMs.length != rrTsMs.length) {
    return const Metric<SessionRmssd>.absent(
      tier: Tier.high,
      inputs_used: inputs,
      note: 'invalid or empty RR session window',
    );
  }
  final s = _inSession(rrMs, rrTsMs, startSec, endSec);
  if (s.rr.length < 3) {
    return const Metric<SessionRmssd>.absent(
      tier: Tier.high,
      inputs_used: inputs,
      note: 'no RR beats inside the session window',
    );
  }
  final cov = rrCoverage(s.rr, s.ts);
  if (cov != null && cov.overCounted) {
    return Metric<SessionRmssd>.absent(
      tier: Tier.high,
      inputs_used: inputs,
      note: 'rr_overcount:coverage=${round6(cov.coverage)} — more beat-time '
          'than elapsed time; the RR stream holds duplicated or interleaved '
          'beats',
    );
  }
  final chain = chainBeats(s.rr, s.ts);
  final good = _correctRrGood(s.rr, s.ts, alpha);

  final nWin = ((endSec - startSec) + windowSec - 1) ~/ windowSec;
  final ssd = List<double>.filled(nWin, 0);
  final nd = List<int>.filled(nWin, 0);
  // Runs of counted differences, for the jitter diagnostic (reported only).
  final runs = <List<double>>[];
  var run = <double>[];
  for (var i = 1; i < s.rr.length; i++) {
    final t = chain.tSec[i];
    if (chain.missing[i] != 0 ||
        !good[i] ||
        !good[i - 1] ||
        t < startSec ||
        t >= endSec) {
      if (run.length >= 2) runs.add(run);
      run = <double>[];
      continue;
    }
    final w = (t - startSec) ~/ windowSec;
    final d = s.rr[i] - s.rr[i - 1];
    ssd[w] += d * d;
    nd[w]++;
    run.add(d);
  }
  if (run.length >= 2) runs.add(run);
  final rmssds = <double>[];
  var thin = 0;
  for (var w = 0; w < nWin; w++) {
    if (nd[w] >= minDiffsPerWindow) {
      rmssds.add(math.sqrt(ssd[w] / nd[w]));
    } else if (nd[w] > 0) {
      thin++;
    }
  }
  if (rmssds.isEmpty) {
    return Metric<SessionRmssd>.absent(
      tier: Tier.high,
      inputs_used: inputs,
      note: 'no valid 5-min windows for sleep-session RMSSD ($thin window(s) '
          'had fewer than $minDiffsPerWindow clean successive differences)',
    );
  }
  return Metric<SessionRmssd>(
    value: SessionRmssd(
      rmssd: mean(rmssds)!,
      windows: rmssds.length,
      thinWindows: thin,
      minDiffsPerWindow: minDiffsPerWindow,
      diffAcf1: nnDiffAcf1(runs),
      rrCoverage: cov?.coverage,
    ),
    confidence: (rmssds.length / 12.0).clamp(0.3, 0.95),
    tier: Tier.high,
    inputs_used: inputs,
    note: 'sleep-session HRV: mean RMSSD over 5-min windows of truly adjacent '
        'beats (correctRr alpha ${alpha.toStringAsFixed(1)}, no jitter gate)',
  );
}

/// Per-30-s-epoch breathing rate (br/min, NaN where none) over the epochs
/// `startSec ~/ 30 .. endSec ~/ 30`, already median-smoothed.
List<double> sleepRespEpochsLs(
  List<double> rrMs,
  List<double> rrTsMs, {
  required int startSec,
  required int endSec,
  double winSec = 90,
  double loHz = 0.15,
  double hiHz = 0.6,
  double nyqFrac = 0.95,
  int minBeats = 30,
  int smoothHalf = 5,
}) {
  const epoch = 30;
  final e0 = startSec ~/ epoch, e1 = endSec ~/ epoch;
  final nE = math.max(0, e1 - e0);
  final raw = List<double>.filled(nE, double.nan);
  final s = _inSession(rrMs, rrTsMs, startSec, endSec);
  if (s.rr.length >= minBeats && nE > 0) {
    final chain = chainBeats(s.rr, s.ts);
    final g1 = _correctRrGood(s.rr, s.ts, 5.2);
    final g2 = malikGood(s.rr);
    final tt = <double>[], yy = <double>[];
    for (var i = 0; i < s.rr.length; i++) {
      if (g1[i] && g2[i]) {
        tt.add(chain.tSec[i]);
        yy.add(s.rr[i]);
      }
    }
    // Re-anchors can step the chain backwards a little; the windows below
    // assume time order (numpy searchsorted on the same array).
    for (var e = 0; e < nE; e++) {
      final c = (e0 + e) * epoch + epoch / 2;
      final i0 = _lowerBound(tt, c - winSec / 2);
      final i1 = _lowerBound(tt, c + winSec / 2);
      if (i1 - i0 < minBeats) continue;
      final st = tt.sublist(i0, i1), sy = yy.sublist(i0, i1);
      if (st.last - st.first < 0.75 * winSec) continue;
      final ny = nyqFrac * 0.5 / (_median(sy) / 1000);
      final top = math.min(hiHz, ny);
      if (top <= loHz) continue;
      final f = freqGrid(loHz, top, 160);
      final ls = lombScargle(st, _detrend(st, sy), f);
      if (ls == null) continue;
      var k = 0;
      for (var j = 1; j < ls.spectrum.length; j++) {
        if (ls.spectrum[j].psd > ls.spectrum[k].psd) k = j;
      }
      if (k == 0 || k == f.length - 1) continue;
      raw[e] = f[k] * 60;
    }
  }
  // Centred rolling median over 2*smoothHalf+1 epochs, at least 3 values.
  final out = List<double>.filled(nE, double.nan);
  for (var e = 0; e < nE; e++) {
    final w = <double>[
      for (var j = math.max(0, e - smoothHalf);
          j <= math.min(nE - 1, e + smoothHalf);
          j++)
        if (!raw[j].isNaN) raw[j]
    ];
    if (w.length >= 3) out[e] = _median(w);
  }
  return out;
}

/// Nightly breathing rate from heartbeat timing: the session median of
/// [sleepRespEpochsLs]. Absent with fewer than 10 epochs carrying a value.
Metric<RespEstimate> sleepRespRateLs(
  List<double> rrMs,
  List<double> rrTsMs, {
  required int startSec,
  required int endSec,
}) {
  const inputs = ['rr_sleep_window'];
  if (startSec <= 0 || endSec <= startSec || rrMs.length != rrTsMs.length) {
    return const Metric<RespEstimate>.absent(
        tier: Tier.estimate,
        inputs_used: inputs,
        note: 'invalid or empty RR session window');
  }
  final ep = sleepRespEpochsLs(rrMs, rrTsMs, startSec: startSec, endSec: endSec);
  final vals = [for (final v in ep) if (!v.isNaN) v];
  if (vals.length < 10) {
    return Metric<RespEstimate>.absent(
        tier: Tier.estimate,
        inputs_used: inputs,
        note: 'only ${vals.length} epoch(s) with a breathing peak; need 10');
  }
  final brpm = _median(vals);
  return Metric<RespEstimate>(
    value: RespEstimate(brpm, brpm / 60, null, 'rsa_ls',
        usableSubwindows: vals.length, subwindows: ep.length),
    confidence: (vals.length / ep.length).clamp(0.2, 0.8),
    tier: Tier.estimate,
    inputs_used: inputs,
    note: 'estimated from heartbeat timing: median of per-epoch Lomb-Scargle '
        'breathing peaks (0.15 Hz floor) over the sleep session',
  );
}

int _lowerBound(List<double> xs, double v) {
  var lo = 0, hi = xs.length;
  while (lo < hi) {
    final m = (lo + hi) >> 1;
    if (xs[m] < v) {
      lo = m + 1;
    } else {
      hi = m;
    }
  }
  return lo;
}

/// y minus its least-squares line in (t - t0).
List<double> _detrend(List<double> t, List<double> y) {
  final n = t.length;
  var sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  for (var i = 0; i < n; i++) {
    final x = t[i] - t[0];
    sx += x;
    sy += y[i];
    sxx += x * x;
    sxy += x * y[i];
  }
  final den = n * sxx - sx * sx;
  final b = den == 0 ? 0.0 : (n * sxy - sx * sy) / den;
  final a = (sy - b * sx) / n;
  return [for (var i = 0; i < n; i++) y[i] - (a + b * (t[i] - t[0]))];
}
