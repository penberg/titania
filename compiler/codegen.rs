//! Turns instructions on virtual registers into Titania instructions:
//! allocates physical registers with a linear scan, and resolves branch
//! targets.

use std::{collections::HashMap, hash::Hash};

use crate::{
    builder::{Item, VInst, Value},
    insn::{Format, Instruction, NUM_PREDS, NUM_REGS},
};

pub fn generate(items: &[Item]) -> Result<Vec<Instruction>, String> {
    // Number the instructions, and find each loop's extent and each label's
    // target.
    let mut insts: Vec<&VInst> = Vec::new();
    let mut loops = Vec::new();
    let mut open = Vec::new();
    let mut labels = HashMap::new();
    for item in items {
        match item {
            Item::Inst(inst) => insts.push(inst),
            Item::Label(label) => {
                labels.insert(*label, insts.len());
            }
            Item::LoopStart => open.push(insts.len()),
            Item::LoopEnd => loops.push((open.pop().expect("unbalanced loop"), insts.len() - 1)),
        }
    }

    let mut values = Intervals::default();
    let mut preds = Intervals::default();
    for (pos, inst) in insts.iter().enumerate() {
        for &value in inst.srcs.iter().chain(&inst.dst) {
            if value != Value::ZERO {
                values.touch(value, pos);
            }
        }
        for &pred in inst.pdst.iter().chain(inst.guard.as_ref().map(|(pred, _)| pred)) {
            preds.touch(pred, pos);
        }
    }
    values.extend_over(&loops);
    preds.extend_over(&loops);
    let regs = values.allocate(1..NUM_REGS).ok_or("the kernel needs too many registers")?;
    let pregs = preds.allocate(0..NUM_PREDS).ok_or("the kernel needs too many predicates")?;

    let reg = |value: &Value| if *value == Value::ZERO { 0 } else { regs[value] };
    let mut out = Vec::with_capacity(insts.len());
    for inst in insts {
        let mut encoded = Instruction::new(inst.insn);
        if let Some((pred, neg)) = inst.guard {
            encoded.guard = pregs[&pred];
            encoded.neg = neg;
        }
        if let Some(dst) = &inst.dst {
            encoded.rd = reg(dst);
        }
        if let Some(pred) = &inst.pdst {
            encoded.rd = pregs[pred];
        }
        let srcs: Vec<u8> = inst.srcs.iter().map(reg).collect();
        match inst.insn.format() {
            // Stores take their address in `ra` and value in `rb`, with no
            // destination; everything else reads its sources in field order.
            Format::Store => (encoded.ra, encoded.rb) = (srcs[0], srcs[1]),
            _ => {
                let fields = [&mut encoded.ra, &mut encoded.rb, &mut encoded.rc];
                for (field, src) in fields.into_iter().zip(srcs) {
                    *field = src;
                }
            }
        }
        encoded.imm = match inst.target {
            Some(label) => Some(labels[&label] as u32),
            None => inst.imm,
        };
        out.push(encoded);
    }
    Ok(out)
}

/// The live range of each register: from the first instruction that
/// mentions it to the last.
struct Intervals<T> {
    ranges: HashMap<T, (usize, usize)>,
}

impl<T> Default for Intervals<T> {
    fn default() -> Self {
        Self { ranges: HashMap::new() }
    }
}

impl<T: Copy + Eq + Hash + Ord> Intervals<T> {
    fn touch(&mut self, reg: T, pos: usize) {
        let range = self.ranges.entry(reg).or_insert((pos, pos));
        range.0 = range.0.min(pos);
        range.1 = range.1.max(pos);
    }

    /// A register live across a loop's boundary is needed on every iteration,
    /// so it must stay allocated for the whole loop.
    fn extend_over(&mut self, loops: &[(usize, usize)]) {
        let mut changed = true;
        while changed {
            changed = false;
            for range in self.ranges.values_mut() {
                for &(start, end) in loops {
                    let overlaps = range.0 <= end && range.1 >= start;
                    let crosses = range.0 < start || range.1 > end;
                    if overlaps && crosses && (range.0 > start || range.1 < end) {
                        *range = (range.0.min(start), range.1.max(end));
                        changed = true;
                    }
                }
            }
        }
    }

    /// Assigns each register a physical one from `pool`, reusing a physical
    /// register once its previous owner's range has ended.
    fn allocate(&self, pool: std::ops::Range<u8>) -> Option<HashMap<T, u8>> {
        let mut order: Vec<(T, (usize, usize))> = self.ranges.iter().map(|(&reg, &range)| (reg, range)).collect();
        order.sort_by_key(|&(reg, (start, _))| (start, reg));
        let mut free: Vec<u8> = pool.rev().collect();
        let mut active: Vec<(usize, u8)> = Vec::new();
        let mut assigned = HashMap::new();
        for (reg, (start, end)) in order {
            active.retain(|&(active_end, phys)| {
                let expired = active_end < start;
                if expired {
                    free.push(phys);
                }
                !expired
            });
            let phys = free.pop()?;
            active.push((end, phys));
            assigned.insert(reg, phys);
        }
        Some(assigned)
    }
}
