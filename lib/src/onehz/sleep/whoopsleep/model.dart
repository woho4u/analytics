// The cracked-whoop five-head sleep stager (v8_five_heads), pure Dart.
//
// Port of `whoopsleep.model.predict` with all five equal votes:
//   1. LightGBM epoch model + LightGBM probability-context stack,
//   2. a TCN over the epoch probabilities (`sequence`),
//   3. a TCN over the per-epoch features (`featseq`),
//   4. the same TCN + whole-night bidirectional GRU (`gruseq`),
//   5. the same TCN + a learned branch over the 2 Hz beat/motion stream
//      (`rawseq`).
// Weights come from cracked-whoop `scripts/export_dart_model.py`.
// Networks run in float64 (the reference runs float32): outputs agree to
// ~1e-5, see test/whoopsleep_parity_test.dart.

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'features.dart';
import 'numeric.dart';

/// Stage indices produced by [WhoopSleepModel.predict].
const List<String> kWsStages = ['wake', 'light', 'deep', 'rem'];

class _Blob {
  final Map<String, dynamic> header;
  final ByteData data;
  final int base;
  _Blob(this.header, this.data, this.base);

  factory _Blob.parse(Uint8List bytes) {
    final bd = ByteData.sublistView(bytes);
    if (utf8.decode(bytes.sublist(0, 4)) != 'WSL8') {
      throw const FormatException('not a whoopsleep model blob');
    }
    final hl = bd.getUint32(4, Endian.little);
    final header = jsonDecode(utf8.decode(bytes.sublist(8, 8 + hl)))
        as Map<String, dynamic>;
    if (header['format'] != kWhoopSleepBlobFormat) {
      throw FormatException('unsupported model format ${header['format']}');
    }
    var base = 8 + hl;
    base += (8 - base % 8) % 8;
    return _Blob(header, bd, base);
  }

  Map<String, dynamic> _t(String name) {
    final t = (header['tensors'] as Map<String, dynamic>)[name];
    if (t == null) throw StateError('missing tensor $name');
    return t as Map<String, dynamic>;
  }

  bool has(String name) =>
      (header['tensors'] as Map<String, dynamic>).containsKey(name);

  List<int> shape(String name) =>
      [for (final v in _t(name)['shape'] as List) v as int];

  int _count(String name) => shape(name).fold(1, (a, b) => a * b);

  Float64List f(String name) {
    final t = _t(name);
    final n = _count(name);
    final off = base + (t['offset'] as int);
    final out = Float64List(n);
    if (t['dtype'] == 'f64') {
      for (var i = 0; i < n; i++) {
        out[i] = data.getFloat64(off + 8 * i, Endian.little);
      }
    } else if (t['dtype'] == 'f32') {
      for (var i = 0; i < n; i++) {
        out[i] = data.getFloat32(off + 4 * i, Endian.little);
      }
    } else {
      throw StateError('$name is not float');
    }
    return out;
  }

  Int32List i(String name) {
    final t = _t(name);
    if (t['dtype'] != 'i32') throw StateError('$name is not i32');
    final n = _count(name);
    final off = base + (t['offset'] as int);
    final out = Int32List(n);
    for (var k = 0; k < n; k++) {
      out[k] = data.getInt32(off + 4 * k, Endian.little);
    }
    return out;
  }
}

// ---------------------------------------------------------------- trees ---

class _Forest {
  final Int32List feat, flags, left, right, nodeStart, leafStart;
  final Float64List thr, leaf;
  final List<String> features;
  _Forest(_Blob b, String p, this.features)
      : feat = b.i('$p.feat'),
        flags = b.i('$p.flags'),
        left = b.i('$p.left'),
        right = b.i('$p.right'),
        nodeStart = b.i('$p.node_start'),
        leafStart = b.i('$p.leaf_start'),
        thr = b.f('$p.thr'),
        leaf = b.f('$p.leaf');

  /// Softmax class probabilities (E x 4, row-major) for the columns [cols].
  Float64List predict(Columns cols, int E) {
    final x = [for (final name in features) cols[name] ?? nanList(E)];
    final raw = Float64List(E * 4);
    final nTrees = nodeStart.length - 1;
    for (var k = 0; k < nTrees; k++) {
      final n0 = nodeStart[k], l0 = leafStart[k], cls = k % 4;
      for (var r = 0; r < E; r++) {
        var i = 0;
        while (true) {
          final j = n0 + i;
          var v = x[feat[j]][r];
          final fl = flags[j];
          bool goLeft;
          if (v.isNaN) {
            if (fl & 2 == 0) {
              v = 0.0;
              goLeft = v <= thr[j];
            } else {
              goLeft = fl & 1 == 1;
            }
          } else {
            goLeft = v <= thr[j];
          }
          final c = goLeft ? left[j] : right[j];
          if (c < 0) {
            raw[r * 4 + cls] += leaf[l0 + ~c];
            break;
          }
          i = c;
        }
      }
    }
    for (var r = 0; r < E; r++) {
      _softmaxRow(raw, r * 4, 4);
    }
    return raw;
  }
}

void _softmaxRow(Float64List v, int o, int n) {
  var m = double.negativeInfinity;
  for (var i = 0; i < n; i++) {
    m = math.max(m, v[o + i]);
  }
  var s = 0.0;
  for (var i = 0; i < n; i++) {
    v[o + i] = math.exp(v[o + i] - m);
    s += v[o + i];
  }
  for (var i = 0; i < n; i++) {
    v[o + i] /= s;
  }
}

// ------------------------------------------------------------- networks ---

/// A [C x L] activation, channel-major.
class _T {
  final int c, l;
  final Float64List v;
  _T(this.c, this.l) : v = Float64List(c * l);
}

class _Conv {
  final int cin, cout, k, stride, pad, dil;
  final Float64List w, b; // w: [cout][cin][k]
  _Conv(_Blob blob, String p, {this.stride = 1, this.pad = 0, this.dil = 1})
      : w = blob.f('$p.weight'),
        b = blob.f('$p.bias'),
        cout = blob.shape('$p.weight')[0],
        cin = blob.shape('$p.weight')[1],
        k = blob.shape('$p.weight')[2];

  _T call(_T x, {bool relu = false}) {
    assert(x.c == cin);
    final lout = (x.l + 2 * pad - dil * (k - 1) - 1) ~/ stride + 1;
    final y = _T(cout, lout);
    final xv = x.v, yv = y.v, L = x.l;
    for (var o = 0; o < cout; o++) {
      final yo = o * lout;
      final bo = b[o];
      for (var t = 0; t < lout; t++) {
        yv[yo + t] = bo;
      }
      for (var ci = 0; ci < cin; ci++) {
        final xo = ci * L;
        for (var kk = 0; kk < k; kk++) {
          final wv = w[(o * cin + ci) * k + kk];
          if (wv == 0) continue;
          final shift = kk * dil - pad;
          // input index = t*stride + shift, must lie in [0, L)
          var t0 = 0;
          if (shift < 0) t0 = (-shift + stride - 1) ~/ stride;
          var t1 = lout;
          final maxT = (L - 1 - shift);
          if (maxT < 0) continue;
          t1 = math.min(t1, maxT ~/ stride + 1);
          var xi = xo + t0 * stride + shift;
          for (var t = t0; t < t1; t++) {
            yv[yo + t] += wv * xv[xi];
            xi += stride;
          }
        }
      }
    }
    if (relu) _reluIn(y);
    return y;
  }
}

void _reluIn(_T x) {
  final v = x.v;
  for (var i = 0; i < v.length; i++) {
    if (v[i] < 0) v[i] = 0;
  }
}

/// Residual dilated blocks: x = relu(x + conv1(relu(convk(x)))).
_T _blocks(List<(_Conv, _Conv)> blocks, _T x) {
  for (final (a, b) in blocks) {
    final y = b(a(x, relu: true));
    for (var i = 0; i < x.v.length; i++) {
      final s = x.v[i] + y.v[i];
      y.v[i] = s > 0 ? s : 0;
    }
    x = y;
  }
  return x;
}

List<(_Conv, _Conv)> _loadBlocks(_Blob b, String p, int k, List<int> dil) => [
      for (var i = 0; i < dil.length; i++)
        (
          _Conv(b, '$p.blocks.$i.0', pad: (k ~/ 2) * dil[i], dil: dil[i]),
          _Conv(b, '$p.blocks.$i.3'),
        )
    ];

/// Channel softmax of [x] (4 x E) to E x 4 row-major.
Float64List _softmaxT(_T x) {
  final out = Float64List(x.l * 4);
  for (var t = 0; t < x.l; t++) {
    for (var c = 0; c < 4; c++) {
      out[t * 4 + c] = x.v[c * x.l + t];
    }
    _softmaxRow(out, t * 4, 4);
  }
  return out;
}

const _dil = [1, 2, 4, 8, 16, 32];

/// `sequence.TCN` over epoch probabilities.
class _SeqTcn {
  final _Conv input, output;
  final List<(_Conv, _Conv)> blocks;
  _SeqTcn(_Blob b, String p)
      : input = _Conv(b, '$p.input'),
        output = _Conv(b, '$p.output'),
        blocks = _loadBlocks(b, p, 5, _dil);
  Float64List call(_T x) =>
      _softmaxT(output(_blocks(blocks, input(x, relu: true))));
}

/// `featseq.Net` trunk (stem + blocks), output optional.
class _FeatTrunk {
  final _Conv s0, s1;
  final List<(_Conv, _Conv)> blocks;
  _FeatTrunk(_Blob b, String p)
      : s0 = _Conv(b, '$p.stem.0'),
        s1 = _Conv(b, '$p.stem.3'),
        blocks = _loadBlocks(b, p, 5, _dil);
  _T call(_T x) => _blocks(blocks, s1(s0(x, relu: true), relu: true));
}

class _RawTrunk {
  final _Conv s0, s1, fuse;
  final List<_Conv> enc;
  final List<(_Conv, _Conv)> blocks;
  _RawTrunk(_Blob b, String p)
      : s0 = _Conv(b, '$p.stem.0'),
        s1 = _Conv(b, '$p.stem.3'),
        fuse = _Conv(b, '$p.fuse.0'),
        enc = [
          _Conv(b, '$p.encoder.0', pad: 4),
          _Conv(b, '$p.encoder.2', stride: 2, pad: 4),
          _Conv(b, '$p.encoder.4', stride: 3, pad: 4),
          _Conv(b, '$p.encoder.7', stride: 2, pad: 3),
          _Conv(b, '$p.encoder.9', stride: 5),
        ],
        blocks = _loadBlocks(b, p, 5, _dil);
  _T call(_T x, _T r) {
    final s = s1(s0(x, relu: true), relu: true);
    var e = r;
    for (final c in enc) {
      e = c(e, relu: true);
    }
    if (e.l != s.l) throw StateError('raw stream length mismatch');
    final cat = _T(s.c + e.c, s.l);
    cat.v.setAll(0, s.v);
    cat.v.setAll(s.v.length, e.v);
    return _blocks(blocks, fuse(cat, relu: true));
  }
}

class _BiGru {
  final List<Float64List> wih, whh, bih, bhh; // [fwd, bwd]
  final int hidden, input;
  _BiGru(_Blob b, String p)
      : wih = [b.f('$p.weight_ih_l0'), b.f('$p.weight_ih_l0_reverse')],
        whh = [b.f('$p.weight_hh_l0'), b.f('$p.weight_hh_l0_reverse')],
        bih = [b.f('$p.bias_ih_l0'), b.f('$p.bias_ih_l0_reverse')],
        bhh = [b.f('$p.bias_hh_l0'), b.f('$p.bias_hh_l0_reverse')],
        hidden = b.shape('$p.weight_hh_l0')[1],
        input = b.shape('$p.weight_ih_l0')[1];

  /// `gruseq._night_context`: pool z (C x E) to 4-epoch means, run both
  /// directions, return (2H x E) repeated back to epochs.
  _T call(_T z) {
    const pool = 4;
    final E = z.l, P = (E + pool - 1) ~/ pool, C = z.c, H = hidden;
    final xs = List.generate(P, (_) => Float64List(C));
    for (var p = 0; p < P; p++) {
      final a = p * pool, bb = math.min(E, a + pool);
      for (var c = 0; c < C; c++) {
        var s = 0.0;
        for (var t = a; t < bb; t++) {
          s += z.v[c * E + t];
        }
        xs[p][c] = s / (bb - a);
      }
    }
    final out = _T(2 * H, E);
    for (var dir = 0; dir < 2; dir++) {
      final h = Float64List(H);
      final gi = Float64List(3 * H), gh = Float64List(3 * H);
      for (var s = 0; s < P; s++) {
        final p = dir == 0 ? s : P - 1 - s;
        final x = xs[p];
        for (var g = 0; g < 3 * H; g++) {
          var a = bih[dir][g];
          final wo = g * C;
          for (var c = 0; c < C; c++) {
            a += wih[dir][wo + c] * x[c];
          }
          gi[g] = a;
          var bsum = bhh[dir][g];
          final ho = g * H;
          for (var k = 0; k < H; k++) {
            bsum += whh[dir][ho + k] * h[k];
          }
          gh[g] = bsum;
        }
        for (var k = 0; k < H; k++) {
          final r = _sigmoid(gi[k] + gh[k]);
          final u = _sigmoid(gi[H + k] + gh[H + k]);
          final n = _tanh(gi[2 * H + k] + r * gh[2 * H + k]);
          h[k] = (1 - u) * n + u * h[k];
        }
        final a = p * pool, bb = math.min(E, a + pool);
        for (var k = 0; k < H; k++) {
          for (var t = a; t < bb; t++) {
            out.v[(dir * H + k) * E + t] = h[k];
          }
        }
      }
    }
    return out;
  }
}

double _sigmoid(double x) => 1 / (1 + math.exp(-x));
double _tanh(double x) {
  if (x > 20) return 1;
  if (x < -20) return -1;
  final e = math.exp(2 * x);
  return (e - 1) / (e + 1);
}

_T _cat(_T a, _T b) {
  final o = _T(a.c + b.c, a.l);
  o.v.setAll(0, a.v);
  o.v.setAll(a.v.length, b.v);
  return o;
}

// ---------------------------------------------------------------- model ---

/// Output of one night: per-epoch probabilities (E x 4, row-major) and the
/// argmax stage index per epoch (0 wake, 1 light, 2 deep, 3 REM).
class WsPrediction {
  final int epochs;
  final Float64List proba;
  WsPrediction(this.epochs, this.proba);
  List<int> get stages => [
        for (var e = 0; e < epochs; e++)
          () {
            var k = 0;
            for (var c = 1; c < 4; c++) {
              if (proba[e * 4 + c] > proba[e * 4 + k]) k = c;
            }
            return k;
          }()
      ];
}

/// The blob layout and architecture this code runs. A model exported for a
/// different architecture carries a different format and is refused.
const String kWhoopSleepBlobFormat = 'whoopsleep-dart-1';

class WhoopSleepModel {
  final String name;
  final _Forest epoch, context;
  final List<int> lags;
  final double blend;
  final Float64List classScale, seqScale, headWeights;
  final _SeqTcn seq;
  final List<String> scalerColumns, flagColumns;
  final Float64List table; // columns x quantiles
  final int quantiles;
  final Map<String, Float64List> headScale;
  final List<(_FeatTrunk, _Conv)> feat;
  final List<(_FeatTrunk, _BiGru, _Conv)> gru;
  final List<(_RawTrunk, _Conv)> raw;
  final Float64List freqs;

  WhoopSleepModel._(
      this.name,
      this.epoch,
      this.context,
      this.lags,
      this.blend,
      this.classScale,
      this.seqScale,
      this.headWeights,
      this.seq,
      this.scalerColumns,
      this.flagColumns,
      this.table,
      this.quantiles,
      this.headScale,
      this.feat,
      this.gru,
      this.raw,
      this.freqs);

  factory WhoopSleepModel.fromBytes(Uint8List bytes) {
    final b = _Blob.parse(bytes);
    final h = b.header;
    List<String> strs(String k) => [for (final v in h[k] as List) v as String];
    Float64List dbl(String k) => Float64List.fromList(
        [for (final v in h[k] as List) (v as num).toDouble()]);
    final seeds = h['seeds'] as int;
    final tableShape = b.shape('scaler.table');
    return WhoopSleepModel._(
      (h['model'] ?? 'whoopsleep').toString(),
      _Forest(b, 'epoch', strs('epoch_features')),
      _Forest(b, 'context', strs('context_features')),
      [for (final v in h['lags'] as List) v as int],
      (h['blend'] as num).toDouble(),
      dbl('class_scale'),
      b.f('sequence.scale'),
      dbl('head_weights'),
      _SeqTcn(b, 'sequence.net'),
      strs('scaler_columns'),
      strs('flags'),
      b.f('scaler.table'),
      tableShape[1],
      {
        for (final k in ['features', 'gru', 'raw']) k: b.f('$k.scale')
      },
      [
        for (var s = 0; s < seeds; s++)
          (_FeatTrunk(b, 'features.net$s'), _Conv(b, 'features.net$s.output'))
      ],
      [
        for (var s = 0; s < seeds; s++)
          (
            _FeatTrunk(b, 'gru.net$s.local'),
            _BiGru(b, 'gru.net$s.gru'),
            _Conv(b, 'gru.net$s.output')
          )
      ],
      [
        for (var s = 0; s < seeds; s++)
          (_RawTrunk(b, 'raw.net$s'), _Conv(b, 'raw.net$s.output'))
      ],
      b.f('spectral.freqs'),
    );
  }

  /// Per-epoch features (exposed for parity tests).
  Columns features(WsNight n) => epochFeatures(n, freqs);

  /// Five-head probabilities for one night. [n] must carry at least one beat.
  WsPrediction predict(WsNight n,
      {Columns? cols, Map<String, Float64List>? debug}) {
    final E = n.epochs;
    cols ??= features(n);
    final w = headWeights;
    // Head 1: epoch model + context stack, class-scaled.
    final p = epoch.predict(cols, E);
    final q = context.predict(_contextColumns(p, cols, E), E);
    debug?['epoch'] = p;
    debug?['context'] = q;
    final out = Float64List(E * 4);
    for (var e = 0; e < E; e++) {
      var s = 0.0;
      for (var c = 0; c < 4; c++) {
        final v =
            ((1 - blend) * p[e * 4 + c] + blend * q[e * 4 + c]) * classScale[c];
        out[e * 4 + c] = v;
        s += v;
      }
      for (var c = 0; c < 4; c++) {
        out[e * 4 + c] = w[0] * out[e * 4 + c] / s;
      }
    }
    // Head 2: TCN over the epoch probabilities.
    final sx = _T(10, E);
    for (var e = 0; e < E; e++) {
      for (var c = 0; c < 4; c++) {
        final v = p[e * 4 + c];
        sx.v[c * E + e] = v;
        sx.v[(4 + c) * E + e] = math.log(v.clamp(1e-5, 1.0)) / 4;
      }
      sx.v[8 * E + e] = cols['t_hours']![e] / 8;
      sx.v[9 * E + e] = cols['t_frac']![e];
    }
    _addHead(out, seq(sx), seqScale, w[1], debug, 'h2');
    // Heads 3-5: feature networks (seed ensembles).
    final x = _transform(cols, E);
    _addHead(out, _mean([for (final (t, o) in feat) _softmaxT(o(t(x)))]),
        headScale['features']!, w[2], debug, 'h3');
    _addHead(
        out,
        _mean([
          for (final (t, g, o) in gru)
            () {
              final z = t(x);
              return _softmaxT(o(_cat(z, g(z))));
            }()
        ]),
        headScale['gru']!,
        w[3],
        debug,
        'h4');
    final r = rawStream(n);
    _addHead(out, _mean([for (final (t, o) in raw) _softmaxT(o(t(x, r)))]),
        headScale['raw']!, w[4], debug, 'h5');
    return WsPrediction(E, out);
  }

  static Float64List _mean(List<Float64List> xs) {
    final o = Float64List(xs.first.length);
    for (final x in xs) {
      for (var i = 0; i < o.length; i++) {
        o[i] += x[i];
      }
    }
    for (var i = 0; i < o.length; i++) {
      o[i] /= xs.length;
    }
    return o;
  }

  static void _addHead(
      Float64List out, Float64List probs, Float64List scale, double weight,
      [Map<String, Float64List>? debug, String? name]) {
    final norm = Float64List(probs.length);
    for (var e = 0; e < out.length ~/ 4; e++) {
      var s = 0.0;
      for (var c = 0; c < 4; c++) {
        s += probs[e * 4 + c] * scale[c];
      }
      for (var c = 0; c < 4; c++) {
        norm[e * 4 + c] = probs[e * 4 + c] * scale[c] / s;
        out[e * 4 + c] += weight * norm[e * 4 + c];
      }
    }
    if (debug != null) debug[name!] = norm;
  }

  Columns _contextColumns(Float64List p, Columns cols, int E) {
    final f = <String, Float64List>{};
    for (var c = 0; c < 4; c++) {
      final s =
          Float64List.fromList([for (var e = 0; e < E; e++) p[e * 4 + c]]);
      f['p$c'] = s;
      for (final lag in lags) {
        final o = nanList(E);
        for (var e = 0; e < E; e++) {
          final j = e - lag;
          if (j >= 0 && j < E) o[e] = s[j];
        }
        f['p${c}_lag$lag'] = o;
      }
      for (final w in [5, 11, 21, 41]) {
        f['p${c}_mean$w'] = rolling(s, w, 1, Roll.mean);
        f['p${c}_std$w'] = rolling(s, w, 1, Roll.std);
      }
    }
    f['t_hours'] = cols['t_hours']!;
    f['t_frac'] = cols['t_frac']!;
    return f;
  }

  /// `featseq.transform`: rank-normalise each column against the training
  /// quantiles, then a missing flag per sensor family.
  _T _transform(Columns cols, int E) {
    final nc = scalerColumns.length, Q = quantiles;
    final out = _T(nc + flagColumns.length, E);
    final xp = Float64List(Q);
    for (var j = 0; j < nc; j++) {
      final v = cols[scalerColumns[j]];
      if (v == null) continue;
      final last = table[j * Q + Q - 1];
      for (var k = 0; k < Q; k++) {
        xp[k] = table[j * Q + k] + k * 1e-9 * (1 + last.abs());
      }
      for (var e = 0; e < E; e++) {
        final x = v[e];
        if (!x.isFinite) continue;
        final u = _interpGrid(x, xp);
        out.v[j * E + e] = ndtri(u.clamp(0.001, 0.999));
      }
    }
    for (var k = 0; k < flagColumns.length; k++) {
      final v = cols[flagColumns[k]];
      for (var e = 0; e < E; e++) {
        out.v[(nc + k) * E + e] = (v == null || v[e].isNaN) ? 1.0 : 0.0;
      }
    }
    return out;
  }

  /// np.interp(x, xp, linspace(0, 1, len(xp))).
  static double _interpGrid(double x, Float64List xp) {
    final n = xp.length;
    if (x <= xp[0]) return 0.0;
    if (x >= xp[n - 1]) return 1.0;
    var lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      final m = (lo + hi) >> 1;
      if (xp[m] <= x) {
        lo = m;
      } else {
        hi = m;
      }
    }
    final g0 = lo / (n - 1), g1 = (lo + 1) / (n - 1);
    final slope = (g1 - g0) / (xp[lo + 1] - xp[lo]);
    return slope * (x - xp[lo]) + g0;
  }

  /// The raw stream values (5 x 60E, channel-major), for parity tests.
  static Float64List rawStreamValues(WsNight n) => rawStream(n).v;

  /// `rawseq.stream`: 5 channels on a 2 Hz grid (60 samples per epoch).
  static _T rawStream(WsNight n) {
    final S = n.epochs * kEpochSec, G = 2 * S;
    final rrT = n.rrT, rrMs = n.rrMs, B = rrT.length;
    final ihr = nanList(G);
    final js =
        searchSortedLeft(rrT, [for (var i = 0; i < G; i++) (i + 0.5) / 2]);
    for (var i = 0; i < G; i++) {
      final g = (i + 0.5) / 2;
      final j = js[i];
      if (j < B && g > rrT[j] - rrMs[j] / 1000) ihr[i] = 60000 / rrMs[j];
    }
    interpolateInside(ihr, 20);
    final out = _T(5, G);
    final (med, iqr) = medianIqr(ihr);
    final d = iqr > 1e-9 ? iqr : 1.0;
    for (var i = 0; i < G; i++) {
      if (ihr[i].isNaN) continue;
      out.v[i] = ((ihr[i] - med) / d).clamp(-6.0, 6.0);
      out.v[G + i] = 1;
    }
    double zang(int i) =>
        math.atan2(n.az[i], math.sqrt(n.ax[i] * n.ax[i] + n.ay[i] * n.ay[i])) *
        180 /
        math.pi;
    var prevZ = 0.0;
    for (var i = 0; i < S; i++) {
      final z = zang(i);
      var da = 0.0, dz = 0.0;
      if (i > 0) {
        final dx = n.ax[i] - n.ax[i - 1],
            dy = n.ay[i] - n.ay[i - 1],
            dzz = n.az[i] - n.az[i - 1];
        da = math.sqrt(dx * dx + dy * dy + dzz * dzz);
        dz = (z - prevZ).abs();
      }
      prevZ = z;
      final ok = n.ax[i].isFinite &&
          n.ay[i].isFinite &&
          n.az[i].isFinite &&
          da.isFinite &&
          dz.isFinite;
      final mov = ok ? log1p(da * 1000) / 4 : 0.0;
      final ang = ok ? log1p(dz) / 3 : 0.0;
      for (var k = 0; k < 2; k++) {
        out.v[2 * G + 2 * i + k] = mov;
        out.v[3 * G + 2 * i + k] = ang;
        out.v[4 * G + 2 * i + k] = ok ? 1.0 : 0.0;
      }
    }
    return out;
  }
}
