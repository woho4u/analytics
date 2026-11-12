// Per-epoch features from the WHOOP 1 Hz substrate: a line-by-line port of
// cracked-whoop `whoopsleep.substrate` (beat chaining / cleaning),
// `whoopsleep.edge_db.night` (night assembly) and `whoopsleep.features`
// (`epoch_features`). Every epoch is 30 s; epoch 0 starts at the window start.

import 'dart:math' as math;
import 'dart:typed_data';

import 'numeric.dart';

const int kEpochSec = 30;
const double kMinRrMs = 300, kMaxRrMs = 2000;

/// One night on the substrate grid, as the model sees it.
class WsNight {
  /// Seconds in the night (n epochs x 30).
  final int seconds;

  /// Gravity vector per second (g), already gap-filled; length = [seconds].
  final Float64List ax, ay, az;

  /// Device 1 Hz heart rate (NaN = none); only used without beats.
  final Float64List hr;

  /// Beat times (s from the window start) and the interval ending there (ms).
  final Float64List rrT, rrMs;

  WsNight(
      this.seconds, this.ax, this.ay, this.az, this.hr, this.rrT, this.rrMs);

  int get epochs => seconds ~/ kEpochSec;
}

/// Python `chain_beats`: rebuild beat times from record-stamped intervals.
(Float64List, Float64List) chainBeats(List<double> recT, List<double> rrMs,
    [double windowS = 1.0]) {
  final out = Float64List(rrMs.length);
  var prev = -1e18;
  const tol = 0.15;
  for (var i = 0; i < rrMs.length; i++) {
    final T = recT[i], rr = rrMs[i] / 1000.0;
    final lo = T - windowS - tol, hi = T + tol;
    double? t;
    for (var k = 0; k < 4; k++) {
      final c = prev + rr * (k + 1);
      if (lo <= c && c <= hi) {
        t = c;
        break;
      }
    }
    t ??= T - windowS / 2;
    out[i] = t;
    prev = t;
  }
  return (out, Float64List.fromList(rrMs));
}

/// Python `clean_rr_pairs`: plausibility band + 20 % jump vs a 9-beat median.
(Float64List, Float64List) cleanRrPairs(List<double> t, List<double> rr) {
  final med = rolling(rr, 9, 3, Roll.median);
  final ot = <double>[], orr = <double>[];
  for (var i = 0; i < rr.length; i++) {
    final v = rr[i];
    if (v >= kMinRrMs && v <= kMaxRrMs && (v - med[i]).abs() <= 0.2 * med[i]) {
      ot.add(t[i]);
      orr.add(v);
    }
  }
  return (Float64List.fromList(ot), Float64List.fromList(orr));
}

/// Python `edge_db.night`: assemble the night [startSec, startSec + 30 n).
///
/// [tsSec]/[x]/[y]/[z]/[valid]/[hr] are the app's positional 1 Hz rows;
/// [beatTsMs]/[beatRrMs] its beats (stamped as `decoded_rr.beat_ts_ms`).
WsNight assembleNight({
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
  final nE = (endSec - startSec) ~/ kEpochSec;
  final S = nE * kEpochSec;
  final sx = Float64List(S), sy = Float64List(S), sz = Float64List(S);
  final cnt = Int32List(S);
  final hs = Float64List(S), hc = Int32List(S);
  for (var i = 0; i < tsSec.length; i++) {
    final k = tsSec[i] - startSec;
    if (k < 0 || k >= S) continue;
    if (valid[i]) {
      sx[k] += x[i];
      sy[k] += y[i];
      sz[k] += z[i];
      cnt[k]++;
    }
    // decoded_onehz.hr is NULL outside the plausible band; pandas' groupby
    // mean skips it, then 30-220 bpm is enforced on the mean.
    if (hr[i] > 0) {
      hs[k] += hr[i];
      hc[k]++;
    }
  }
  final ax = nanList(S), ay = nanList(S), az = nanList(S), h = nanList(S);
  for (var k = 0; k < S; k++) {
    if (cnt[k] > 0) {
      ax[k] = sx[k] / cnt[k];
      ay[k] = sy[k] / cnt[k];
      az[k] = sz[k] / cnt[k];
    }
    if (hc[k] > 0) {
      final v = hs[k] / hc[k];
      if (v >= 30 && v <= 220) h[k] = v;
    }
  }
  for (final a in [ax, ay, az]) {
    _fillLimited(a, 180);
  }
  // Beats inside the window, time-ordered (stable), plausibility band, chained.
  final idx = <int>[
    for (var i = 0; i < beatTsMs.length; i++)
      if (beatTsMs[i] >= startSec * 1000.0 &&
          beatTsMs[i] < (startSec + S) * 1000.0)
        i
  ];
  idx.sort((a, b) {
    final c = beatTsMs[a].compareTo(beatTsMs[b]);
    return c != 0 ? c : a.compareTo(b);
  });
  final recT = <double>[], rr = <double>[];
  for (final i in idx) {
    final v = beatRrMs[i];
    if (v >= kMinRrMs && v <= kMaxRrMs) {
      recT.add(beatTsMs[i] / 1000.0 - startSec);
      rr.add(v);
    }
  }
  final (ct, cr) = chainBeats(recT, rr);
  final (rt, rm) = cleanRrPairs(ct, cr);
  return WsNight(S, ax, ay, az, h, rt, rm);
}

/// ffill(limit) then bfill(limit) then fill 0 (the night's gravity vectors).
void _fillLimited(Float64List a, int limit) {
  final n = a.length;
  final f = Float64List.fromList(a);
  var last = nan;
  var run = 0;
  for (var i = 0; i < n; i++) {
    if (!a[i].isNaN) {
      last = a[i];
      run = 0;
    } else if (!last.isNaN && run < limit) {
      f[i] = last;
      run++;
    } else {
      run++;
    }
  }
  var next = nan;
  run = 0;
  for (var i = n - 1; i >= 0; i--) {
    if (!f[i].isNaN) {
      next = f[i];
      run = 0;
    } else if (!next.isNaN && run < limit) {
      f[i] = next;
      run++;
    } else {
      run++;
    }
  }
  for (var i = 0; i < n; i++) {
    a[i] = f[i].isNaN ? 0.0 : f[i];
  }
}

/// Python `hr_from_beats`: mean instantaneous rate of the beats ending in each
/// second, gaps bridged up to 5 s.
Float64List hrFromBeats(List<double> rrT, List<double> rrMs, int S) {
  final s = Float64List(S), c = Int32List(S);
  for (var i = 0; i < rrT.length; i++) {
    final sec = rrT[i].floor();
    if (sec < 0 || sec >= S) continue;
    s[sec] += 60000.0 / rrMs[i];
    c[sec]++;
  }
  final hr = nanList(S);
  for (var k = 0; k < S; k++) {
    if (c[k] > 0) hr[k] = s[k] / c[k];
  }
  interpolateInside(hr, 5);
  return hr;
}

typedef Columns = Map<String, Float64List>;

/// HRV over a centred window around each epoch's midpoint (`rr_per_epoch`).
Columns _rrPerEpoch(WsNight n, int E, double hw) {
  final t = n.rrT, rr = n.rrMs;
  final B = rr.length;
  final out = {
    for (final k in ['rmssd', 'sdnn', 'mean_rr', 'pnn50', 'cov']) k: nanList(E)
  };
  final gapOk = List<bool>.filled(B, false);
  final sd = nanList(B);
  for (var i = 1; i < B; i++) {
    gapOk[i] = ((t[i] - t[i - 1]) * 1000 - rr[i]).abs() < 0.25 * rr[i];
    sd[i] = rr[i] - rr[i - 1];
  }
  final d = <double>[];
  final mids = [for (var e = 0; e < E; e++) (e + 0.5) * kEpochSec];
  final los = searchSortedLeft(t, [for (final m in mids) m - hw]);
  final his = searchSortedLeft(t, [for (final m in mids) m + hw]);
  for (var e = 0; e < E; e++) {
    final a = los[e], b = his[e];
    var sum = 0.0;
    for (var i = a; i < b; i++) {
      sum += rr[i];
    }
    out['cov']![e] = (sum / 1000) / (2 * hw);
    if (b - a < 10) continue;
    out['mean_rr']![e] = mean(rr, a, b);
    out['sdnn']![e] = stdPop(rr, a, b);
    d.clear();
    for (var i = a; i < b; i++) {
      if (gapOk[i]) d.add(sd[i]);
    }
    if (d.length >= 8) {
      var s2 = 0.0, big = 0;
      for (final v in d) {
        s2 += v * v;
        if (v.abs() > 50) big++;
      }
      out['rmssd']![e] = math.sqrt(s2 / d.length);
      out['pnn50']![e] = big / d.length;
    }
  }
  return out;
}

double _dfaAlpha1(List<double> rr) {
  if (rr.length < 64) return nan;
  final m = mean(rr);
  final y = Float64List(rr.length);
  var acc = 0.0;
  for (var i = 0; i < rr.length; i++) {
    acc += rr[i] - m;
    y[i] = acc;
  }
  const ns = [4, 6, 8, 11, 16];
  final logN = <double>[], logF = <double>[];
  for (final n in ns) {
    final k = y.length ~/ n;
    // Fit y = a x + b on x = 0..n-1 per segment; closed-form least squares.
    final xm = (n - 1) / 2.0;
    var sxx = 0.0;
    for (var x = 0; x < n; x++) {
      sxx += (x - xm) * (x - xm);
    }
    var res = 0.0;
    for (var s = 0; s < k; s++) {
      final o = s * n;
      var ym = 0.0;
      for (var x = 0; x < n; x++) {
        ym += y[o + x];
      }
      ym /= n;
      var sxy = 0.0;
      for (var x = 0; x < n; x++) {
        sxy += (x - xm) * (y[o + x] - ym);
      }
      final slope = sxy / sxx, icpt = ym - slope * xm;
      for (var x = 0; x < n; x++) {
        final r = y[o + x] - (slope * x + icpt);
        res += r * r;
      }
    }
    logN.add(math.log(n.toDouble()));
    logF.add(math.log(math.sqrt(res / (k * n))));
  }
  return linFit(logN, logF).$1;
}

/// Lomb-Scargle power (SciPy >= 1.17 "power" normalisation, no floating mean).
Float64List _lombScargle(List<double> t, List<double> y, List<double> w) {
  final n = t.length;
  final wt = 1.0 / n;
  final out = Float64List(w.length);
  final eps = 1.1102230246251565e-16; // float64 epsneg
  for (var f = 0; f < w.length; f++) {
    var cc = 0.0, cs = 0.0;
    for (var i = 0; i < n; i++) {
      final a = w[f] * t[i];
      final c = math.cos(a), s = math.sin(a);
      cc += wt * c * c;
      cs += wt * c * s;
    }
    final ss = 1.0 - cc;
    final tau = 0.5 * math.atan2(2.0 * cs, cc - ss);
    var yc = 0.0, ys = 0.0, c2 = 0.0;
    for (var i = 0; i < n; i++) {
      final a = w[f] * t[i] - tau;
      final c = math.cos(a), s = math.sin(a);
      yc += wt * y[i] * c;
      ys += wt * y[i] * s;
      c2 += wt * c * c;
    }
    var s2 = 1.0 - c2;
    if (c2 < eps) c2 = eps;
    if (s2 < eps) s2 = eps;
    final pa = yc / c2, pb = ys / s2;
    out[f] = 2.0 * (pa * yc + pb * ys) * (n / 4.0);
  }
  return out;
}

Columns _spectral(WsNight n, int E, List<double> freqs) {
  const hw = 150.0;
  const keys = ['lf', 'hf', 'lfhf', 'hfnu', 'resp_rate', 'resp_reg', 'dfa_a1'];
  final out = {for (final k in keys) k: nanList(E)};
  final w = [for (final f in freqs) 2 * math.pi * f];
  final lfIdx = <int>[], hfIdx = <int>[];
  for (var i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 0.04 && freqs[i] < 0.15) lfIdx.add(i);
    if (freqs[i] >= 0.15 && freqs[i] <= 0.4) hfIdx.add(i);
  }
  final cache = <int, List<double>>{};
  final mids = [for (var e = 0; e < E; e++) (e + 0.5) * kEpochSec];
  final los = searchSortedLeft(n.rrT, [for (final m in mids) m - hw]);
  final his = searchSortedLeft(n.rrT, [for (final m in mids) m + hw]);
  for (var e = 0; e < E; e++) {
    final a = los[e], b = his[e];
    if (b - a < 120) continue;
    final key = (a ~/ 8) * 1000003 + (b ~/ 8);
    final v = cache.putIfAbsent(key, () {
      final t = n.rrT.sublist(a, b), rr = n.rrMs.sublist(a, b);
      final m = mean(rr);
      final y = [for (final r in rr) r - m];
      final pw = _lombScargle(t, y, w);
      var lf = 0.0, hf = 0.0;
      for (final i in lfIdx) {
        lf += pw[i];
      }
      final hp = [for (final i in hfIdx) pw[i]];
      for (final p in hp) {
        hf += p;
      }
      var k = 0;
      for (var i = 1; i < hp.length; i++) {
        if (hp[i] > hp[k]) k = i;
      }
      var band = 0.0;
      for (var i = math.max(k - 4, 0); i < math.min(k + 5, hp.length); i++) {
        band += hp[i];
      }
      return [
        math.log(lf + 1e-9),
        math.log(hf + 1e-9),
        math.log((lf + 1e-9) / (hf + 1e-9)),
        hf / (lf + hf + 1e-9),
        freqs[hfIdx[k]] * 60,
        band / (hf + 1e-9),
        _dfaAlpha1(rr),
      ];
    });
    for (var j = 0; j < keys.length; j++) {
      out[keys[j]]![e] = v[j];
    }
  }
  return out;
}

/// Python `epoch_features(n)` (hr_source="beats").
Columns epochFeatures(WsNight n, List<double> freqs) {
  final E = n.epochs, S = E * kEpochSec;
  final hasBeats = n.rrMs.isNotEmpty;
  final hr = hasBeats ? hrFromBeats(n.rrT, n.rrMs, n.hr.length) : n.hr;
  final da = Float64List(S), zang = Float64List(S), dz = Float64List(S);
  for (var i = 0; i < S; i++) {
    zang[i] = math.atan2(n.az[i], _hypot(n.ax[i], n.ay[i])) * 180 / math.pi;
    if (i > 0) {
      final dx = n.ax[i] - n.ax[i - 1],
          dy = n.ay[i] - n.ay[i - 1],
          dzz = n.az[i] - n.az[i - 1];
      da[i] = math.sqrt(dx * dx + dy * dy + dzz * dzz);
      dz[i] = (zang[i] - zang[i - 1]).abs();
    }
  }
  final f = <String, Float64List>{};
  Float64List perEpoch(List<double> x, String fn) {
    final out = nanList(E);
    for (var e = 0; e < E; e++) {
      final a = e * kEpochSec, b = a + kEpochSec;
      var cnt = 0, s = 0.0, mx = double.negativeInfinity;
      for (var i = a; i < b; i++) {
        if (x[i].isNaN) continue;
        cnt++;
        s += x[i];
        if (x[i] > mx) mx = x[i];
      }
      if (cnt == 0) continue;
      out[e] = switch (fn) {
        'mean' => s / cnt,
        'max' => mx,
        _ => nanStdSample(x, a, b),
      };
    }
    return out;
  }

  f['mov_mean'] = perEpoch(da, 'mean');
  f['mov_max'] = perEpoch(da, 'max');
  f['zang_sd'] = perEpoch(zang, 'std');
  f['zang_dmax'] = perEpoch(dz, 'max');
  f['hr_mean'] = perEpoch(hr.sublist(0, S), 'mean');
  f['hr_sd'] = perEpoch(hr.sublist(0, S), 'std');
  for (final hw in [60, 150, 300]) {
    _rrPerEpoch(n, E, hw.toDouble()).forEach((k, v) => f['${k}_$hw'] = v);
  }
  f['ln_rmssd_150'] =
      Float64List.fromList([for (final v in f['rmssd_150']!) math.log(v)]);
  if (hasBeats) f.addAll(_spectral(n, E, freqs));

  for (final c in [
    'hr_mean',
    'mean_rr_150',
    'rmssd_150',
    'sdnn_150',
    'ln_rmssd_150',
    'sdnn_300',
    'rmssd_300',
    'lf',
    'hf',
    'lfhf',
    'resp_rate',
    'resp_reg',
    'dfa_a1'
  ]) {
    if (f.containsKey(c)) f['${c}_z'] = robustZ(f[c]!);
  }
  f['mov_log'] =
      Float64List.fromList([for (final v in f['mov_mean']!) log1p(v * 1000)]);

  final base = [
    'mov_log',
    'zang_sd',
    'hr_mean_z',
    'hr_sd',
    'rmssd_150_z',
    'sdnn_150_z',
    'mean_rr_150_z',
    for (final c in ['lfhf_z', 'resp_reg', 'resp_rate_z', 'dfa_a1'])
      if (f.containsKey(c)) c,
  ];
  for (final w in [5, 11, 21, 41]) {
    for (final c in base) {
      final mp = math.max(1, w ~/ 3);
      f['${c}_m$w'] = rolling(f[c]!, w, mp, Roll.mean);
      f['${c}_s$w'] = rolling(f[c]!, w, mp, Roll.std);
    }
  }
  final movLog = f['mov_log']!;
  final p60 = nanPercentile(movLog, 60);
  final still = [for (final v in movLog) v < p60];
  final run = Float64List(E), rev = Float64List(E);
  for (var i = 0; i < E; i++) {
    run[i] = (i > 0 && still[i]) ? run[i - 1] + 1 : (still[i] ? 1.0 : 0.0);
  }
  for (var i = E - 1; i >= 0; i--) {
    rev[i] = (i < E - 1 && still[i]) ? rev[i + 1] + 1 : (still[i] ? 1.0 : 0.0);
  }
  f['still_run'] = run;
  f['still_run_rev'] = rev;
  f['still_total'] =
      Float64List.fromList([for (var i = 0; i < E; i++) run[i] + rev[i]]);
  f['t_hours'] =
      Float64List.fromList([for (var i = 0; i < E; i++) i * kEpochSec / 3600]);
  f['t_frac'] = Float64List.fromList(
      [for (var i = 0; i < E; i++) i / math.max(E - 1, 1)]);
  final hrz = f['hr_mean_z']!;
  final med = rolling(hrz, 121, 40, Roll.median);
  f['hr_detr'] =
      Float64List.fromList([for (var i = 0; i < E; i++) hrz[i] - med[i]]);
  return f;
}

double _hypot(double a, double b) {
  // np.hypot: scaled to avoid overflow; plain form is exact enough for g.
  return math.sqrt(a * a + b * b);
}
