//! The console's 64-bit float instructions (`v_*_f64`) as DXIL doubles.
//!
//! The console's float mode keeps 64-bit denormals on input and output and
//! rounds to nearest even (`semantics_tests`). Additions, products, the
//! fused multiply-add, division and the square root's refinement are the
//! host's double arithmetic, which rounds the same way. What a sign, an
//! exponent or a NaN decides is computed on the bits instead: compares,
//! class tests, minimum and maximum, rounding to whole numbers, `frexp` and
//! the saturation of conversions to integers. A host that flushes double
//! denormals in a compare or a select, or orders the zeros its own way,
//! then changes nothing there.
//!
//! Operands: a register pair holds the low dword first. An inline integer
//! constant is that integer sign-extended to 64 bits, an inline float
//! constant the double of that value, and a 32-bit literal the high dword
//! of the double, its low dword zero (LLVM's AMDGPU operand rules). `abs`
//! and `neg` clear and flip the sign bit, as on the hardware.
//!
//! Sources: AMD's RDNA2 ISA reference guide (30 November 2020), the VOP1,
//! VOP3 and VOPC opcode descriptions. The reciprocal, reciprocal square
//! root and square root are approximations there ("precision is (2**29)
//! ULP"); here they are within a few ULPs. A program refines them (the
//! Newton steps of a compiled division or square root) to the same result.
//!
//! Left untranslated (`Unsupported`): `v_div_scale_f64`, `v_div_fmas_f64`,
//! `v_div_fixup_f64` (a compiled division's scaling and fix-up steps) and
//! `v_trig_preop_f64`.

use super::{Lower, Why};
use crate::emit::bitcode::{Attr, Bin, Cast, V, pred};
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Inst, Reg};

const SIGN: u64 = 0x8000_0000_0000_0000;
const MAGNITUDE: u64 = !SIGN;
const INFINITY: u64 = 0x7FF0_0000_0000_0000;
const FRACTION: u64 = 0x000F_FFFF_FFFF_FFFF;
const ONE: u64 = 0x3FF0_0000_0000_0000;
const HALF: u64 = 0x3FE0_0000_0000_0000;
const QUIET: u64 = 0x0008_0000_0000_0000;
/// The largest double below 1.0.
const BELOW_ONE: u64 = 0x3FEF_FFFF_FFFF_FFFF;

/// The 64 bits of an inline constant read as a 64-bit float operand.
pub(crate) fn inline64(code: u8) -> Option<u64> {
    let f = |x: f64| Some(x.to_bits());
    match code {
        128..=192 => Some(u64::from(code) - 128),
        193..=208 => Some((192 - i64::from(code)) as u64),
        240 => f(0.5),
        241 => f(-0.5),
        242 => f(1.0),
        243 => f(-1.0),
        244 => f(2.0),
        245 => f(-2.0),
        246 => f(4.0),
        247 => f(-4.0),
        // 1/(2π) as the hardware holds it (truncated: the rounded double
        // ends in ...c883), as LLVM's AMDGPU inline-constant table has it.
        248 => Some(0x3FC4_5F30_6DC9_C882),
        _ => None,
    }
}

/// How a rounding to a whole number goes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Round {
    Trunc,
    Floor,
    Ceil,
    Even,
}

impl Lower<'_> {
    /// The `v_*_f64` instructions (and the 64-bit conversions); `None` for
    /// any other.
    pub(super) fn valu64(
        &mut self,
        inst: &Inst,
        name: &str,
        guard: Option<V>,
    ) -> Option<Result<(), Why>> {
        let r = match name {
            "v_add_f64" | "v_mul_f64" => self.arith64(inst, name, guard),
            "v_fma_f64" => self.fma64(inst, guard),
            "v_min_f64" | "v_max_f64" => self.min_max64(inst, name == "v_max_f64", guard),
            "v_ldexp_f64" => self.ldexp64(inst, guard),
            "v_trunc_f64" | "v_floor_f64" | "v_ceil_f64" | "v_rndne_f64" => {
                let mode = match name {
                    "v_trunc_f64" => Round::Trunc,
                    "v_floor_f64" => Round::Floor,
                    "v_ceil_f64" => Round::Ceil,
                    _ => Round::Even,
                };
                self.dbits(inst, 0).and_then(|x| {
                    let r = self.round64(x, mode);
                    self.write_bits64(inst, r, guard)
                })
            }
            "v_fract_f64" => self.fract64(inst, guard),
            "v_frexp_mant_f64" => self.frexp_mant64(inst, guard),
            "v_frexp_exp_i32_f64" => self.frexp_exp64(inst, guard),
            "v_rcp_f64" => self.dsrc(inst, 0).and_then(|a| {
                let one = self.s.m.f64(ONE);
                let r = self.s.m.bin(self.cur, Bin::SDiv, one, a);
                self.write_d(inst, r, guard)
            }),
            "v_sqrt_f64" | "v_rsq_f64" => self.dbits(inst, 0).and_then(|x| {
                let r = self.sqrt64(x);
                let r = if name == "v_rsq_f64" {
                    let d = self.double_of(r);
                    let one = self.s.m.f64(ONE);
                    let q = self.s.m.bin(self.cur, Bin::SDiv, one, d);
                    self.bits_of(q)
                } else {
                    r
                };
                self.write_bits64(inst, r, guard)
            }),
            "v_cvt_f64_f32" => self.fsrc(inst, 0).and_then(|f| {
                // The console reads f32 denormals as zeros.
                let f = self.ftz(f);
                let t = self.s.m.double();
                let d = self.s.m.cast(self.cur, Cast::FpExt, f, t);
                self.write_d(inst, d, guard)
            }),
            "v_cvt_f32_f64" => self.dsrc(inst, 0).and_then(|d| {
                // Rounded to nearest even, an f32 denormal result flushed
                // to a zero of its sign.
                let f = self.s.m.cast(self.cur, Cast::FpTrunc, d, self.s.t.f32);
                let f = self.ftz(f);
                self.write_f(inst, f, guard)
            }),
            "v_cvt_f64_i32" | "v_cvt_f64_u32" => self.src(inst, 0).and_then(|x| {
                let op = if name == "v_cvt_f64_i32" {
                    Cast::SiToFp
                } else {
                    Cast::UiToFp
                };
                let t = self.s.m.double();
                let d = self.s.m.cast(self.cur, op, x, t);
                self.write_d(inst, d, guard)
            }),
            "v_cvt_i32_f64" | "v_cvt_u32_f64" => {
                self.convert_to_int64(inst, name == "v_cvt_i32_f64", guard)
            }
            _ => return None,
        };
        Some(r)
    }

    // ---- operands and results ----

    /// Source `k` of a 64-bit float operation as its low and high dwords,
    /// with `abs` and `neg` applied to the sign bit.
    fn dwords64(&mut self, inst: &Inst, k: usize) -> Result<(V, V), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.sdwa.is_some() || v.dpp.is_some() {
            return Err("a 64-bit float operand with SDWA or DPP");
        }
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let read = |l: &mut Self, r: Reg| -> Result<V, Why> {
            l.i32_of(inst.read(r).ok_or("an unread operand")?)
        };
        let (lo, mut hi) = match op {
            Operand::Vector { reg, .. } => (read(self, Reg::V(reg))?, read(self, Reg::V(reg + 1))?),
            Operand::Scalar { code, .. } if code == NULL => (self.s.m.i32(0), self.s.m.i32(0)),
            Operand::Scalar { code, .. } => {
                (read(self, Reg::S(code))?, read(self, Reg::S(code + 1))?)
            }
            Operand::Const(code) => {
                let x = inline64(code).ok_or("a special operand of a 64-bit float operation")?;
                (self.s.m.i32(x as u32), self.s.m.i32((x >> 32) as u32))
            }
            Operand::Literal(x) => (self.s.m.i32(0), self.s.m.i32(x)),
            Operand::Attr { .. } => return Err("an attribute as a 64-bit float operand"),
        };
        let c = self.cur;
        if v.abs >> k & 1 != 0 {
            let m = self.s.m.i32(0x7FFF_FFFF);
            hi = self.s.m.bin(c, Bin::And, hi, m);
        }
        if v.neg >> k & 1 != 0 {
            let m = self.s.m.i32(0x8000_0000);
            hi = self.s.m.bin(c, Bin::Xor, hi, m);
        }
        Ok((lo, hi))
    }

    /// Source `k` as a double.
    fn dsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let (lo, hi) = self.dwords64(inst, k)?;
        Ok(self.make_double(lo, hi))
    }

    /// Source `k` as the double's 64 bits.
    fn dbits(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let (lo, hi) = self.dwords64(inst, k)?;
        Ok(self.join64(lo, hi))
    }

    /// DXIL's MakeDouble: the double of two dwords.
    fn make_double(&mut self, lo: V, hi: V) -> V {
        let t = self.s.m.double();
        self.s.op(
            self.cur,
            "makeDouble.f64",
            Attr::ReadNone,
            t,
            101,
            &[lo, hi],
        )
    }

    /// DXIL's SplitDouble: the low and high dwords of a double.
    fn split_double(&mut self, d: V) -> (V, V) {
        let i32t = self.s.t.i32;
        let st = self
            .s
            .m
            .named_struct("dx.types.splitdouble", vec![i32t, i32t]);
        let r = self
            .s
            .op(self.cur, "splitDouble.f64", Attr::ReadNone, st, 102, &[d]);
        let lo = self.s.m.extract(self.cur, r, 0);
        let hi = self.s.m.extract(self.cur, r, 1);
        (lo, hi)
    }

    /// The double of 64 bits.
    fn double_of(&mut self, x: V) -> V {
        let (lo, hi) = self.split64(x);
        self.make_double(lo, hi)
    }

    /// The 64 bits of a double.
    fn bits_of(&mut self, d: V) -> V {
        let (lo, hi) = self.split_double(d);
        self.join64(lo, hi)
    }

    /// Two dwords as an i64, `lo` the low one.
    fn join64(&mut self, lo: V, hi: V) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let lo = self.s.m.cast(c, Cast::ZExt, lo, i64t);
        let hi = self.s.m.cast(c, Cast::ZExt, hi, i64t);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::Shl, hi, k32);
        self.s.m.bin(c, Bin::Or, lo, hi)
    }

    fn k64(&mut self, x: u64) -> V {
        let i64t = self.s.m.int(64);
        self.s.m.int_const(i64t, x as i64)
    }

    /// A 64-bit float result (its two dwords), after the output modifiers:
    /// `clamp` saturates to [0, 1] (a NaN to 0, as the console's DX10 clamp
    /// mode does); an output multiplier is refused.
    fn write_d(&mut self, inst: &Inst, d: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.omod != 0 {
            return Err("an output multiplier on a 64-bit result");
        }
        let d = if v.clamp {
            let t = self.s.m.double();
            self.s.op(self.cur, "unary.f64", Attr::ReadNone, t, 7, &[d])
        } else {
            d
        };
        let (lo, hi) = self.split_double(d);
        self.write_pair64(inst, lo, hi, guard)
    }

    /// A 64-bit float result given as its bits.
    fn write_bits64(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 {
            let d = self.double_of(x);
            return self.write_d(inst, d, guard);
        }
        let (lo, hi) = self.split64(x);
        self.write_pair64(inst, lo, hi, guard)
    }

    fn write_pair64(&mut self, inst: &Inst, lo: V, hi: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a 64-bit result outside the VGPRs");
        };
        self.write(inst, Reg::V(reg), lo, guard);
        self.write(inst, Reg::V(reg + 1), hi, guard);
        Ok(())
    }

    // ---- arithmetic ----

    fn arith64(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Result<(), Why> {
        let (a, b) = (self.dsrc(inst, 0)?, self.dsrc(inst, 1)?);
        let op = if name == "v_add_f64" {
            Bin::Add
        } else {
            Bin::Mul
        };
        let r = self.s.m.bin(self.cur, op, a, b);
        self.write_d(inst, r, guard)
    }

    /// `a * b + c` rounded once: DXIL's Fma, a double operation.
    fn fma64(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let (a, b, cc) = (
            self.dsrc(inst, 0)?,
            self.dsrc(inst, 1)?,
            self.dsrc(inst, 2)?,
        );
        let t = self.s.m.double();
        let r = self
            .s
            .op(self.cur, "tertiary.f64", Attr::ReadNone, t, 47, &[a, b, cc]);
        self.write_d(inst, r, guard)
    }

    /// `x * 2^n` (n the signed dword of source 1), rounded once: in up to
    /// three exact steps of at most 2^1023 or 2^-969 that keep a normal
    /// value normal, then the one that rounds (musl's `scalbn`).
    fn ldexp64(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let x = self.dsrc(inst, 0)?;
        let n = self.src(inst, 1)?;
        let c = self.cur;
        let k = |l: &mut Self, v: i32| l.s.m.i32(v as u32);
        // Beyond +-2200 every result is 0, infinity or x itself.
        let (lo, hi) = (k(self, -2200), k(self, 2200));
        let below = self.s.m.cmp(c, pred::SLT, n, lo);
        let mut n = self.s.m.select(c, below, lo, n);
        let above = self.s.m.cmp(c, pred::SGT, n, hi);
        n = self.s.m.select(c, above, hi, n);
        let mut y = x;
        let one = self.s.m.f64(ONE);
        for (limit, step, factor) in [
            (1023, 1023, 1023i64),
            (1023, 1023, 1023),
            (-1022, -969, -969),
            (-1022, -969, -969),
        ] {
            let lim = k(self, limit);
            let cond = if limit > 0 {
                self.s.m.cmp(c, pred::SGT, n, lim)
            } else {
                self.s.m.cmp(c, pred::SLT, n, lim)
            };
            let f = self.s.m.f64(((factor + 1023) as u64) << 52);
            let f = self.s.m.select(c, cond, f, one);
            y = self.s.m.bin(c, Bin::Mul, y, f);
            let stepped = k(self, step);
            let stepped = self.s.m.bin(c, Bin::Sub, n, stepped);
            n = self.s.m.select(c, cond, stepped, n);
        }
        // Now -1022 <= n <= 1023 (the steps clamp what they cannot reach).
        let (lo, hi) = (k(self, -1022), k(self, 1023));
        let below = self.s.m.cmp(c, pred::SLT, n, lo);
        n = self.s.m.select(c, below, lo, n);
        let above = self.s.m.cmp(c, pred::SGT, n, hi);
        n = self.s.m.select(c, above, hi, n);
        let bias = k(self, 1023);
        let e = self.s.m.bin(c, Bin::Add, n, bias);
        let k20 = k(self, 20);
        let e = self.s.m.bin(c, Bin::Shl, e, k20);
        let zero = k(self, 0);
        let scale = self.make_double(zero, e);
        let r = self.s.m.bin(c, Bin::Mul, y, scale);
        self.write_d(inst, r, guard)
    }

    // ---- on the bits ----

    /// Whether `x` (bits) is a NaN.
    fn nan64(&mut self, x: V) -> V {
        let c = self.cur;
        let m = self.k64(MAGNITUDE);
        let a = self.s.m.bin(c, Bin::And, x, m);
        let inf = self.k64(INFINITY);
        self.s.m.cmp(c, pred::IUGT, a, inf)
    }

    /// `x` (bits) as an unsigned integer whose order is the doubles' order,
    /// -0 just below +0 (NaNs anywhere).
    fn order_key64(&mut self, x: V) -> V {
        let c = self.cur;
        let zero = self.k64(0);
        let neg = self.s.m.cmp(c, pred::SLT, x, zero);
        let all = self.k64(u64::MAX);
        let flipped = self.s.m.bin(c, Bin::Xor, x, all);
        let sign = self.k64(SIGN);
        let raised = self.s.m.bin(c, Bin::Or, x, sign);
        self.s.m.select(c, neg, flipped, raised)
    }

    /// The hardware's minimum or maximum (IEEE mode off): a NaN operand
    /// gives the other one, -0 is below +0, denormals are values.
    fn min_max64(&mut self, inst: &Inst, max: bool, guard: Option<V>) -> Result<(), Why> {
        let (a, b) = (self.dbits(inst, 0)?, self.dbits(inst, 1)?);
        let c = self.cur;
        let (ka, kb) = (self.order_key64(a), self.order_key64(b));
        let lt = self.s.m.cmp(c, pred::IULT, ka, kb);
        let r = if max {
            self.s.m.select(c, lt, b, a)
        } else {
            self.s.m.select(c, lt, a, b)
        };
        let (na, nb) = (self.nan64(a), self.nan64(b));
        let r = self.s.m.select(c, nb, a, r);
        let r = self.s.m.select(c, na, b, r);
        self.write_bits64(inst, r, guard)
    }

    /// A float compare of two 64-bit sources, on the bits: unordered when
    /// either is a NaN, the zeros equal.
    pub(super) fn compare64(&mut self, inst: &Inst, op: &str) -> Result<V, Why> {
        if op == "f" || op == "tru" {
            return Ok(self.s.m.bool(op == "tru"));
        }
        let (a, b) = (self.dbits(inst, 0)?, self.dbits(inst, 1)?);
        let c = self.cur;
        let (na, nb) = (self.nan64(a), self.nan64(b));
        let uno = self.s.m.bin(c, Bin::Or, na, nb);
        let yes = self.s.m.bool(true);
        let ord = self.s.m.bin(c, Bin::Xor, uno, yes);
        // -0 compares as +0.
        let zero = self.k64(0);
        let mag = self.k64(MAGNITUDE);
        let mut keys = [a, b];
        for x in &mut keys {
            let m = self.s.m.bin(c, Bin::And, *x, mag);
            let z = self.s.m.cmp(c, pred::EQ, m, zero);
            let x0 = self.s.m.select(c, z, zero, *x);
            *x = self.order_key64(x0);
        }
        let [ka, kb] = keys;
        let lt = self.s.m.cmp(c, pred::IULT, ka, kb);
        let gt = self.s.m.cmp(c, pred::IUGT, ka, kb);
        let eq = self.s.m.cmp(c, pred::EQ, ka, kb);
        let not = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Xor, x, yes);
        let and = |l: &mut Self, x: V, y: V| l.s.m.bin(c, Bin::And, x, y);
        let or = |l: &mut Self, x: V, y: V| l.s.m.bin(c, Bin::Or, x, y);
        Ok(match op {
            "lt" => and(self, ord, lt),
            "eq" => and(self, ord, eq),
            "le" => {
                let ngt = not(self, gt);
                and(self, ord, ngt)
            }
            "gt" => and(self, ord, gt),
            "lg" => {
                let ne = not(self, eq);
                and(self, ord, ne)
            }
            "ge" => {
                let nlt = not(self, lt);
                and(self, ord, nlt)
            }
            "o" => ord,
            "u" => uno,
            "nge" => or(self, uno, lt),
            "nlg" => or(self, uno, eq),
            "ngt" => {
                let ngt = not(self, gt);
                or(self, uno, ngt)
            }
            "nle" => or(self, uno, gt),
            "neq" => {
                let ne = not(self, eq);
                or(self, uno, ne)
            }
            "nlt" => {
                let nlt = not(self, lt);
                or(self, uno, nlt)
            }
            _ => return Err(inst.name()),
        })
    }

    /// `v_cmp_class_f64`: bit k of source 1 tests class k (signalling NaN,
    /// quiet NaN, -inf, -normal, -denormal, -0, +0, +denormal, +normal,
    /// +inf), on the high dword and whether the fraction is zero.
    pub(super) fn class64(&mut self, inst: &Inst) -> Result<V, Why> {
        let (lo, hi) = self.dwords64(inst, 0)?;
        let mask = self.src(inst, 1)?;
        let c = self.cur;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k20, k31, emax, high_frac, quiet, zero) = (
            k(self, 20),
            k(self, 31),
            k(self, 0x7FF),
            k(self, 0xF_FFFF),
            k(self, 0x8_0000),
            k(self, 0),
        );
        let e = self.s.m.bin(c, Bin::LShr, hi, k20);
        let e = self.s.m.bin(c, Bin::And, e, emax);
        let f = self.s.m.bin(c, Bin::And, hi, high_frac);
        let f = self.s.m.bin(c, Bin::Or, f, lo);
        let neg = self.s.m.bin(c, Bin::LShr, hi, k31);
        let e0 = self.s.m.cmp(c, pred::EQ, e, zero);
        let einf = self.s.m.cmp(c, pred::EQ, e, emax);
        let f0 = self.s.m.cmp(c, pred::EQ, f, zero);
        let q = self.s.m.bin(c, Bin::And, hi, quiet);
        let q = self.s.m.cmp(c, pred::NE, q, zero);
        let (normal, inf, zero_c, denorm) = (k(self, 8), k(self, 9), k(self, 6), k(self, 7));
        let t = self.s.m.select(c, f0, zero_c, denorm);
        let t = self.s.m.select(c, e0, t, normal);
        let t = self.s.m.select(c, einf, inf, t);
        let eleven = k(self, 11);
        let mirrored = self.s.m.bin(c, Bin::Sub, eleven, t);
        let is_neg = self.s.m.cmp(c, pred::NE, neg, zero);
        let t = self.s.m.select(c, is_neg, mirrored, t);
        let one = k(self, 1);
        let nan_class = self.s.m.select(c, q, one, zero);
        let yes = self.s.m.bool(true);
        let nan = self.s.m.bin(c, Bin::Xor, f0, yes);
        let nan = self.s.m.bin(c, Bin::And, nan, einf);
        let t = self.s.m.select(c, nan, nan_class, t);
        let bit = self.s.m.bin(c, Bin::LShr, mask, t);
        let bit = self.s.m.bin(c, Bin::And, bit, one);
        Ok(self.s.m.cmp(c, pred::NE, bit, zero))
    }

    /// The biased exponent field of `x` (bits) as an i32.
    fn exponent64(&mut self, x: V) -> V {
        let c = self.cur;
        let k52 = self.k64(52);
        let e = self.s.m.bin(c, Bin::LShr, x, k52);
        let e = self.s.m.cast(c, Cast::Trunc, e, self.s.t.i32);
        let m = self.s.m.i32(0x7FF);
        self.s.m.bin(c, Bin::And, e, m)
    }

    /// `x` (bits) rounded to a whole number.
    fn round64(&mut self, x: V, mode: Round) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let k = |l: &mut Self, v: i32| l.s.m.i32(v as u32);
        let e = self.exponent64(x);
        let bias = k(self, 1023);
        let u = self.s.m.bin(c, Bin::Sub, e, bias);
        let zero = self.k64(0);
        let sign_k = self.k64(SIGN);
        let sign = self.s.m.bin(c, Bin::And, x, sign_k);
        let neg = self.s.m.cmp(c, pred::NE, sign, zero);
        let mag_k = self.k64(MAGNITUDE);
        let mag = self.s.m.bin(c, Bin::And, x, mag_k);
        // 0 <= u <= 51: the fraction's low 52 - u bits go.
        let k52 = k(self, 52);
        let fb = self.s.m.bin(c, Bin::Sub, k52, u);
        let (one_i, k52b) = (k(self, 1), k(self, 52));
        let low = self.s.m.cmp(c, pred::SLT, fb, one_i);
        let fb = self.s.m.select(c, low, one_i, fb);
        let high = self.s.m.cmp(c, pred::SGT, fb, k52b);
        let fb = self.s.m.select(c, high, k52b, fb);
        let fb = self.s.m.cast(c, Cast::ZExt, fb, i64t);
        let one = self.k64(1);
        let unit = self.s.m.bin(c, Bin::Shl, one, fb);
        let mask = self.s.m.bin(c, Bin::Sub, unit, one);
        let frac = self.s.m.bin(c, Bin::And, x, mask);
        let all = self.k64(u64::MAX);
        let keep = self.s.m.bin(c, Bin::Xor, mask, all);
        let base = self.s.m.bin(c, Bin::And, x, keep);
        let has_frac = self.s.m.cmp(c, pred::NE, frac, zero);
        let yes = self.s.m.bool(true);
        let pos = self.s.m.bin(c, Bin::Xor, neg, yes);
        let up = match mode {
            Round::Trunc => self.s.m.bool(false),
            Round::Floor => self.s.m.bin(c, Bin::And, neg, has_frac),
            Round::Ceil => self.s.m.bin(c, Bin::And, pos, has_frac),
            Round::Even => {
                let half = self.s.m.bin(c, Bin::LShr, unit, one);
                let over = self.s.m.cmp(c, pred::IUGT, frac, half);
                let tie = self.s.m.cmp(c, pred::EQ, frac, half);
                let odd = self.s.m.bin(c, Bin::And, x, unit);
                let odd = self.s.m.cmp(c, pred::NE, odd, zero);
                let tie_odd = self.s.m.bin(c, Bin::And, tie, odd);
                self.s.m.bin(c, Bin::Or, over, tie_odd)
            }
        };
        let raised = self.s.m.bin(c, Bin::Add, base, unit);
        let middle = self.s.m.select(c, up, raised, base);
        // u < 0: |x| < 1 rounds to a zero or a one of x's sign.
        let nonzero = self.s.m.cmp(c, pred::NE, mag, zero);
        let up_small = match mode {
            Round::Trunc => self.s.m.bool(false),
            Round::Floor => self.s.m.bin(c, Bin::And, neg, nonzero),
            Round::Ceil => self.s.m.bin(c, Bin::And, pos, nonzero),
            Round::Even => {
                let h = self.k64(HALF);
                self.s.m.cmp(c, pred::IUGT, mag, h)
            }
        };
        let one_bits = self.k64(ONE);
        let small = self.s.m.select(c, up_small, one_bits, zero);
        let small = self.s.m.bin(c, Bin::Or, small, sign);
        let zero_i = k(self, 0);
        let is_small = self.s.m.cmp(c, pred::SLT, u, zero_i);
        let k52c = k(self, 52);
        let is_whole = self.s.m.cmp(c, pred::SGE, u, k52c);
        let r = self.s.m.select(c, is_small, small, middle);
        self.s.m.select(c, is_whole, x, r)
    }

    /// `x - floor(x)`, kept below 1.0 (where a tiny negative `x` would
    /// round to it): NaN for an infinity, the NaN for a NaN.
    fn fract64(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let x = self.dbits(inst, 0)?;
        let fl = self.round64(x, Round::Floor);
        let (xd, fd) = (self.double_of(x), self.double_of(fl));
        let c = self.cur;
        let r = self.s.m.bin(c, Bin::Sub, xd, fd);
        let r = self.bits_of(r);
        // On the bits, so no driver folds it into a minimum that loses NaN.
        let mag = self.k64(MAGNITUDE);
        let a = self.s.m.bin(c, Bin::And, r, mag);
        let one = self.k64(ONE);
        let over = self.s.m.cmp(c, pred::EQ, a, one);
        let below = self.k64(BELOW_ONE);
        let r = self.s.m.select(c, over, below, r);
        let nan = self.nan64(x);
        let quiet = self.k64(QUIET);
        let qx = self.s.m.bin(c, Bin::Or, x, quiet);
        let r = self.s.m.select(c, nan, qx, r);
        self.write_bits64(inst, r, guard)
    }

    /// The leading zeros of the 52-bit fraction of `x` (bits), as an i32;
    /// meaningless for a zero fraction.
    fn fraction_zeros64(&mut self, x: V) -> V {
        let c = self.cur;
        let (lo, hi) = self.split64(x);
        let m = self.s.m.i32(0xF_FFFF);
        let top = self.s.m.bin(c, Bin::And, hi, m);
        let i32t = self.s.t.i32;
        // FirstbitHi counts from bit 31.
        let zt = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[top]);
        let zl = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[lo]);
        let k12 = self.s.m.i32(12);
        let k20 = self.s.m.i32(20);
        let from_top = self.s.m.bin(c, Bin::Sub, zt, k12);
        let from_low = self.s.m.bin(c, Bin::Add, zl, k20);
        let zero = self.s.m.i32(0);
        let in_top = self.s.m.cmp(c, pred::NE, top, zero);
        self.s.m.select(c, in_top, from_top, from_low)
    }

    /// `x` (bits) normalized: its exponent as an unbiased i32 (that of a
    /// denormal's leading one) and its fraction without that one, as i64.
    fn normalize64(&mut self, x: V) -> (V, V) {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let e = self.exponent64(x);
        let fk = self.k64(FRACTION);
        let frac = self.s.m.bin(c, Bin::And, x, fk);
        let z = self.fraction_zeros64(x);
        let one = self.s.m.i32(1);
        let shift = self.s.m.bin(c, Bin::Add, z, one);
        let k63 = self.s.m.i32(63);
        let shift = self.s.m.bin(c, Bin::And, shift, k63);
        let shift = self.s.m.cast(c, Cast::ZExt, shift, i64t);
        let shifted = self.s.m.bin(c, Bin::Shl, frac, shift);
        let shifted = self.s.m.bin(c, Bin::And, shifted, fk);
        let zero = self.s.m.i32(0);
        let denormal = self.s.m.cmp(c, pred::EQ, e, zero);
        let bias = self.s.m.i32(1023);
        let normal_e = self.s.m.bin(c, Bin::Sub, e, bias);
        let k = self.s.m.i32((-1023i32) as u32);
        let denormal_e = self.s.m.bin(c, Bin::Sub, k, z);
        let e = self.s.m.select(c, denormal, denormal_e, normal_e);
        let frac = self.s.m.select(c, denormal, shifted, frac);
        (e, frac)
    }

    /// `x` (bits) is a zero, an infinity or a NaN: frexp keeps those.
    fn not_finite_or_zero64(&mut self, x: V) -> V {
        let c = self.cur;
        let mag_k = self.k64(MAGNITUDE);
        let mag = self.s.m.bin(c, Bin::And, x, mag_k);
        let zero = self.k64(0);
        let z = self.s.m.cmp(c, pred::EQ, mag, zero);
        let inf = self.k64(INFINITY);
        let big = self.s.m.cmp(c, pred::IUGE, mag, inf);
        self.s.m.bin(c, Bin::Or, z, big)
    }

    /// The fraction of `frexp`: x's sign, magnitude in [0.5, 1); a zero, an
    /// infinity or a NaN as it is.
    fn frexp_mant64(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let x = self.dbits(inst, 0)?;
        let c = self.cur;
        let (_, frac) = self.normalize64(x);
        let sign_k = self.k64(SIGN);
        let sign = self.s.m.bin(c, Bin::And, x, sign_k);
        let half = self.k64(HALF);
        let r = self.s.m.bin(c, Bin::Or, frac, half);
        let r = self.s.m.bin(c, Bin::Or, r, sign);
        let keep = self.not_finite_or_zero64(x);
        let r = self.s.m.select(c, keep, x, r);
        self.write_bits64(inst, r, guard)
    }

    /// The exponent of `frexp` (x = fraction * 2^exponent, the fraction in
    /// [0.5, 1)); 0 for a zero, an infinity or a NaN.
    fn frexp_exp64(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let x = self.dbits(inst, 0)?;
        let c = self.cur;
        let (e, _) = self.normalize64(x);
        let one = self.s.m.i32(1);
        let r = self.s.m.bin(c, Bin::Add, e, one);
        let keep = self.not_finite_or_zero64(x);
        let zero = self.s.m.i32(0);
        let r = self.s.m.select(c, keep, zero, r);
        self.write_i(inst, r, guard)
    }

    /// The square root of `x` (bits). For a positive finite `x = m * 4^k`
    /// (m in [1, 4)): the f32 square root of m, two Newton steps in double,
    /// times 2^k. A zero stays itself, a negative value is NaN, +inf
    /// itself, a NaN quiet.
    fn sqrt64(&mut self, x: V) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let (e, frac) = self.normalize64(x);
        let one_i = self.s.m.i32(1);
        let k = self.s.m.bin(c, Bin::AShr, e, one_i);
        let two_k = self.s.m.bin(c, Bin::Add, k, k);
        let r = self.s.m.bin(c, Bin::Sub, e, two_k);
        let bias = self.s.m.i32(1023);
        let me = self.s.m.bin(c, Bin::Add, r, bias);
        let me = self.s.m.cast(c, Cast::ZExt, me, i64t);
        let k52 = self.k64(52);
        let me = self.s.m.bin(c, Bin::Shl, me, k52);
        let m_bits = self.s.m.bin(c, Bin::Or, me, frac);
        let m = self.double_of(m_bits);
        let f = self.s.m.cast(c, Cast::FpTrunc, m, self.s.t.f32);
        let g = self.unary(24, f);
        let t = self.s.m.double();
        let mut y = self.s.m.cast(c, Cast::FpExt, g, t);
        let half = self.s.m.f64(0.5f64.to_bits());
        for _ in 0..2 {
            let q = self.s.m.bin(c, Bin::SDiv, m, y);
            let s = self.s.m.bin(c, Bin::Add, y, q);
            y = self.s.m.bin(c, Bin::Mul, half, s);
        }
        let ke = self.s.m.bin(c, Bin::Add, k, bias);
        let k20 = self.s.m.i32(20);
        let ke = self.s.m.bin(c, Bin::Shl, ke, k20);
        let zero_i = self.s.m.i32(0);
        let scale = self.make_double(zero_i, ke);
        let s = self.s.m.bin(c, Bin::Mul, y, scale);
        let s = self.bits_of(s);
        // Special operands, on the bits.
        let zero = self.k64(0);
        let mag_k = self.k64(MAGNITUDE);
        let mag = self.s.m.bin(c, Bin::And, x, mag_k);
        let is_zero = self.s.m.cmp(c, pred::EQ, mag, zero);
        let negative = self.s.m.cmp(c, pred::SLT, x, zero);
        let inf = self.k64(INFINITY);
        let is_inf = self.s.m.cmp(c, pred::EQ, x, inf);
        let nan = self.nan64(x);
        let quiet = self.k64(QUIET);
        let qx = self.s.m.bin(c, Bin::Or, x, quiet);
        let default_nan = self.k64(0xFFF8_0000_0000_0000);
        let r = self.s.m.select(c, negative, default_nan, s);
        let r = self.s.m.select(c, is_inf, x, r);
        let r = self.s.m.select(c, is_zero, x, r);
        self.s.m.select(c, nan, qx, r)
    }

    /// A conversion to a 32-bit integer, toward zero, saturating; NaN 0.
    fn convert_to_int64(&mut self, inst: &Inst, signed: bool, guard: Option<V>) -> Result<(), Why> {
        let d = self.dsrc(inst, 0)?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let (op, lo, hi, lo_v, hi_v) = if signed {
            (
                Cast::FpToSi,
                -2147483648.0f64,
                2147483647.0f64,
                0x8000_0000u32,
                0x7FFF_FFFFu32,
            )
        } else {
            (Cast::FpToUi, 0.0, 4294967295.0, 0, u32::MAX)
        };
        let r = self.s.m.cast(c, op, d, i32t);
        let lo_d = self.s.m.f64(lo.to_bits());
        let hi_d = self.s.m.f64(hi.to_bits());
        let under = self.s.m.cmp(c, pred::OLE, d, lo_d);
        let over = self.s.m.cmp(c, pred::OGE, d, hi_d);
        let nan = self.s.m.cmp(c, pred::UNO, d, d);
        let (lo_k, hi_k, zero) = (self.s.m.i32(lo_v), self.s.m.i32(hi_v), self.s.m.i32(0));
        let r = self.s.m.select(c, under, lo_k, r);
        let r = self.s.m.select(c, over, hi_k, r);
        let r = self.s.m.select(c, nan, zero, r);
        self.write_i(inst, r, guard)
    }
}
