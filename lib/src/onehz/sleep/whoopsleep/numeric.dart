// Numeric helpers reproducing the exact NumPy / pandas semantics the
// cracked-whoop Python reference uses (NaN-skipping statistics, centred
// rolling windows with min_periods, forward-limited linear interpolation).
// Parity with the reference is checked by test/whoopsleep_parity_test.dart.

import 'dart:math' as math;
import 'dart:typed_data';

const double nan = double.nan;

Float64List nanList(int n) => Float64List(n)..fillRange(0, n, nan);

/// Sorted finite values of [x].
List<double> _finiteSorted(List<double> x) => [
      for (final v in x)
        if (!v.isNaN) v
    ]..sort();

/// np.nanpercentile(x, q), linear method. NaN when nothing is finite.
double nanPercentile(List<double> x, double q) =>
    _percentileSorted(_finiteSorted(x), q);

double _percentileSorted(List<double> s, double q) {
  if (s.isEmpty) return nan;
  final idx = q / 100.0 * (s.length - 1);
  final lo = idx.floor();
  final hi = math.min(lo + 1, s.length - 1);
  final g = idx - lo;
  // NumPy's _lerp: a + (b - a) * t, flipped to b - (b - a) * (1 - t) for t >= .5.
  final a = s[lo], b = s[hi];
  final diff = b - a;
  return g >= 0.5 ? b - diff * (1 - g) : a + diff * g;
}

double nanMedian(List<double> x) => _medianSorted(_finiteSorted(x));

/// np.median of sorted values: the mean of the two middle ones.
double _medianSorted(List<double> s) {
  if (s.isEmpty) return nan;
  final m = s.length >> 1;
  return s.length.isOdd ? s[m] : (s[m - 1] + s[m]) / 2;
}

/// (median, IQR) of the finite values — what `_robust_z` needs.
(double, double) medianIqr(List<double> x) {
  final s = _finiteSorted(x);
  return (
    _medianSorted(s),
    _percentileSorted(s, 75) - _percentileSorted(s, 25)
  );
}

/// Python `_robust_z`: (x - median) / IQR, IQR <= 1e-9 -> 1.
Float64List robustZ(List<double> x) {
  final (med, iqr) = medianIqr(x);
  final d = iqr > 1e-9 ? iqr : 1.0;
  return Float64List.fromList([for (final v in x) (v - med) / d]);
}

double mean(List<double> x, [int a = 0, int? b]) {
  b ??= x.length;
  var s = 0.0;
  for (var i = a; i < b; i++) {
    s += x[i];
  }
  return s / (b - a);
}

/// Population (ddof 0) standard deviation, NumPy two-pass.
double stdPop(List<double> x, [int a = 0, int? b]) {
  b ??= x.length;
  final m = mean(x, a, b);
  var s = 0.0;
  for (var i = a; i < b; i++) {
    final d = x[i] - m;
    s += d * d;
  }
  return math.sqrt(s / (b - a));
}

/// Sample (ddof 1) std of the finite values in x[a:b]; NaN below two values.
double nanStdSample(List<double> x, int a, int b) {
  var n = 0;
  var s = 0.0;
  for (var i = a; i < b; i++) {
    if (!x[i].isNaN) {
      n++;
      s += x[i];
    }
  }
  if (n < 2) return nan;
  final m = s / n;
  var v = 0.0;
  for (var i = a; i < b; i++) {
    if (!x[i].isNaN) {
      final d = x[i] - m;
      v += d * d;
    }
  }
  return math.sqrt(v / (n - 1));
}

enum Roll { mean, std, median }

/// pandas `Series.rolling(w, center=True, min_periods=mp).<fn>()`.
///
/// Mean and std reproduce pandas' online algorithms (`roll_mean`: Kahan sums
/// with the GH#42064 same-value guard; `roll_var`: Welford with Kahan
/// compensation and its cancellation re-anchor) operation for operation:
/// tree splits sit right at their ulp-level residues, so a textbook formula
/// is not close enough.
Float64List rolling(List<double> x, int w, int mp, Roll fn) {
  final n = x.length;
  final offset = (w - 1) ~/ 2;
  final start = Int32List(n), end = Int32List(n);
  for (var i = 0; i < n; i++) {
    end[i] = math.min(i + 1 + offset, n);
    start[i] = math.min(math.max(i + 1 + offset - w, 0), n);
  }
  switch (fn) {
    case Roll.mean:
      return _rollMean(x, start, end, mp);
    case Roll.std:
      final v = _rollVar(x, start, end, mp);
      for (var i = 0; i < n; i++) {
        v[i] = v[i] < 0 ? 0.0 : math.sqrt(v[i]);
      }
      return v;
    case Roll.median:
      final out = nanList(n);
      final buf = <double>[];
      for (var i = 0; i < n; i++) {
        buf.clear();
        for (var j = start[i]; j < end[i]; j++) {
          if (!x[j].isNaN) buf.add(x[j]);
        }
        if (buf.length < math.max(mp, 1)) continue;
        buf.sort();
        out[i] = _medianSorted(buf);
      }
      return out;
  }
}

Float64List _rollMean(
    List<double> x, Int32List start, Int32List end, int minp) {
  final n = start.length;
  final out = Float64List(n);
  var compAdd = 0.0, compRem = 0.0, sum = 0.0, prev = 0.0;
  var nobs = 0, neg = 0, same = 0;
  void add(double v) {
    if (v.isNaN) return;
    nobs++;
    final y = v - compAdd;
    final t = sum + y;
    compAdd = t - sum - y;
    sum = t;
    if (v.isNegative) neg++;
    if (v == prev) {
      same++;
    } else {
      same = 1;
    }
    prev = v;
  }

  void remove(double v) {
    if (v.isNaN) return;
    nobs--;
    final y = -v - compRem;
    final t = sum + y;
    compRem = t - sum - y;
    sum = t;
    if (v.isNegative) neg--;
  }

  for (var i = 0; i < n; i++) {
    final s = start[i], e = end[i];
    if (i == 0 || s >= end[i - 1]) {
      compAdd = compRem = sum = 0;
      nobs = neg = 0;
      prev = s < x.length ? x[s] : nan;
      same = 0;
      for (var j = s; j < e; j++) {
        add(x[j]);
      }
    } else {
      for (var j = start[i - 1]; j < s; j++) {
        remove(x[j]);
      }
      for (var j = end[i - 1]; j < e; j++) {
        add(x[j]);
      }
    }
    double r;
    if (nobs >= minp && nobs > 0) {
      r = sum / nobs;
      if (same >= nobs) {
        r = prev;
      } else if (neg == 0 && r < 0) {
        r = 0;
      } else if (neg == nobs && r > 0) {
        r = 0;
      }
    } else {
      r = nan;
    }
    out[i] = r;
  }
  return out;
}

const double _invCondTol = 2.220446049250313e-16 * 1e3;

Float64List _rollVar(List<double> x, Int32List start, Int32List end, int minp,
    [int ddof = 1]) {
  minp = math.max(minp, 1);
  final n = start.length;
  final out = Float64List(n);
  var mean = 0.0, ssq = 0.0, nobs = 0.0, compAdd = 0.0, compRem = 0.0;
  var unstable = false;
  void add(double v) {
    if (v.isNaN) return;
    final prevM2 = ssq;
    nobs += 1;
    final prevMean = mean - compAdd;
    final y = v - compAdd;
    final t = y - mean;
    compAdd = t + mean - y;
    final delta = t;
    mean = nobs != 0 ? mean + delta / nobs : 0;
    ssq = ssq + (v - prevMean) * (v - mean);
    if (prevM2 * _invCondTol > ssq) unstable = true;
  }

  void remove(double v) {
    if (v.isNaN) return;
    final prevM2 = ssq;
    nobs -= 1;
    if (nobs != 0) {
      final prevMean = mean - compRem;
      final y = v - compRem;
      final t = y - mean;
      compRem = t + mean - y;
      final delta = t;
      mean = mean - delta / nobs;
      ssq = ssq - (v - prevMean) * (v - mean);
      if (prevM2 * _invCondTol > ssq) unstable = true;
    } else {
      mean = 0;
      ssq = 0;
      unstable = false;
    }
  }

  for (var i = 0; i < n; i++) {
    final s = start[i], e = end[i];
    final recompute = i == 0 || s >= end[i - 1];
    if (!recompute) {
      for (var j = start[i - 1]; j < s; j++) {
        remove(x[j]);
      }
      for (var j = end[i - 1]; j < e; j++) {
        add(x[j]);
      }
    }
    if (recompute || unstable) {
      mean = ssq = nobs = compAdd = compRem = 0;
      for (var j = s; j < e; j++) {
        add(x[j]);
      }
      unstable = false;
    }
    out[i] = (nobs >= minp && nobs > ddof) ? ssq / (nobs - ddof) : nan;
  }
  return out;
}

/// pandas `Series.interpolate(limit=L, limit_area="inside")` (linear): inside
/// a NaN run bounded by values on both sides, fill only its first L entries.
void interpolateInside(Float64List x, int limit) {
  final n = x.length;
  var i = 0;
  while (i < n && x[i].isNaN) {
    i++;
  }
  while (i < n) {
    // x[i] finite here
    var j = i + 1;
    while (j < n && x[j].isNaN) {
      j++;
    }
    if (j >= n) break; // trailing run: outside
    final gap = j - i;
    if (gap > 1) {
      final a = x[i], b = x[j];
      for (var k = 1; k < gap && k <= limit; k++) {
        x[i + k] = a + (b - a) * k / gap;
      }
    }
    i = j;
  }
}

/// np.searchsorted(arr, keys, side="left"), reproducing NumPy 2.x's
/// branch-free binary search (`npysort/binsearch.cpp`) probe for probe: beat
/// times are not strictly sorted after a re-anchor, and on such arrays the
/// answer depends on exactly which elements the search compares.
Int32List searchSortedLeft(List<double> arr, List<double> keys) {
  final n = arr.length;
  final out = Int32List(keys.length);
  if (n == 0) return out;
  for (var k = 0; k < keys.length; k++) {
    final key = keys[k];
    var len = n;
    var half = len >> 1;
    len -= half;
    var base = arr[half] < key ? half : 0;
    while (len > 1) {
      half = len >> 1;
      len -= half;
      if (arr[base + half] < key) base += half;
    }
    out[k] = arr[base] < key ? base + 1 : base;
  }
  return out;
}

/// Least-squares slope and intercept of y on x (np.polyfit deg 1).
(double, double) linFit(List<double> x, List<double> y) {
  final n = x.length;
  var mx = 0.0, my = 0.0;
  for (var i = 0; i < n; i++) {
    mx += x[i];
    my += y[i];
  }
  mx /= n;
  my /= n;
  var sxy = 0.0, sxx = 0.0;
  for (var i = 0; i < n; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) * (x[i] - mx);
  }
  final slope = sxy / sxx;
  return (slope, my - slope * mx);
}

/// Inverse standard normal CDF (Wichura AS241, ~1e-16 relative accuracy).
double ndtri(double p) {
  final q = p - 0.5;
  if (q.abs() <= 0.425) {
    final r = 0.180625 - q * q;
    return q *
        (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r +
                                                    67265.770927008700853) *
                                                r +
                                            45921.953931549871457) *
                                        r +
                                    13731.693765509461125) *
                                r +
                            1971.5909503065514427) *
                        r +
                    133.14166789178437745) *
                r +
            3.387132872796366608) /
        (((((((r * 5226.495278852545925 + 28729.085735721942674) * r +
                                                    39307.89580009271061) *
                                                r +
                                            21213.794301586595867) *
                                        r +
                                    5394.1960214247511077) *
                                r +
                            687.1870074920579083) *
                        r +
                    42.313330701600911252) *
                r +
            1.0);
  }
  var r = q < 0 ? p : 1 - p;
  r = math.sqrt(-math.log(r));
  double v;
  if (r <= 5.0) {
    r -= 1.6;
    v = (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r +
                                                    0.24178072517745061177) *
                                                r +
                                            1.27045825245236838258) *
                                        r +
                                    3.64784832476320460504) *
                                r +
                            5.7694972214606914055) *
                        r +
                    4.6303378461565452959) *
                r +
            1.42343711074968357734) /
        (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r +
                                                    0.0151986665636164571966) *
                                                r +
                                            0.14810397642748007459) *
                                        r +
                                    0.68976733498510000455) *
                                r +
                            1.6763848301838038494) *
                        r +
                    2.05319162663775882187) *
                r +
            1.0);
  } else {
    r -= 5.0;
    v = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r +
                                                    0.0012426609473880784386) *
                                                r +
                                            0.026532189526576123093) *
                                        r +
                                    0.29656057182850489123) *
                                r +
                            1.7848265399172913358) *
                        r +
                    5.4637849111641143699) *
                r +
            6.6579046435011037772) /
        (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r +
                                                    1.8463183175100546818e-5) *
                                                r +
                                            7.868691311456132591e-4) *
                                        r +
                                    0.0148753612908506148525) *
                                r +
                            0.13692988092273580531) *
                        r +
                    0.59983220655588793769) *
                r +
            1.0);
  }
  return q < 0 ? -v : v;
}

/// log(1 + x), accurate for small x (Goldberg's correction).
double log1p(double x) {
  if (x.isNaN) return nan;
  final u = 1.0 + x;
  if (u == 1.0) return x;
  return math.log(u) * x / (u - 1.0);
}
