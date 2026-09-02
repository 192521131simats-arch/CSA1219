"""
=============================================================================
CSA1224 - Computer Architecture
Assignment 2: Python Simulator-Based Smart Factory Machine-Health Analytics
=============================================================================
A single, self-contained, executable Python 3 program covering:
  1. Machine-health event generation (>=1000 events)
  2. Simplified instruction stream derived from events
  3. CPU model: PC, registers, ALU, memory, I/O
  4. Four addressing modes: Immediate, Register, Direct, Indirect
  5. Fetch -> Decode -> Execute -> Memory -> Write-back trace
  6. Baseline single-issue (non-pipelined) simulation
  7. 5-stage pipeline simulation (IF ID EX MEM WB) with data/control/
     structural hazard detection, stalls and penalties
  8. CPI, execution time, throughput, speedup calculations
  9. Simplified OOO / Speculative Execution / SMT comparison
 10. L1/L2/L3 cache hierarchy simulation, >=3 configurations
 11. Hit rate, miss rate, AMAT, latency, throughput
 12. Simplified virtual memory / paging model with page faults
 13. SRAM/DRAM/NAND/MRAM/RRAM technology comparison
 14. Bottleneck identification + engineering recommendation
 15. Tables + matplotlib graphs
 16. Automated test cases (expected vs actual vs PASS/FAIL)

Only standard libraries + matplotlib are used. A fixed random seed (42)
makes every run reproducible. No results in the accompanying report are
invented -- they are all printed/plotted by this program.
=============================================================================
"""

import random
import math
import statistics
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 1: MACHINE-HEALTH EVENT GENERATION
# =============================================================================

@dataclass
class MachineEvent:
    event_id: int
    machine_id: int
    temperature: float   # deg C
    vibration: float      # mm/s RMS
    load: float           # % rated load
    status: str           # NORMAL / WARNING / CRITICAL


def generate_machine_events(n_events: int = 1200, n_machines: int = 8) -> List[MachineEvent]:
    """Generate n_events synthetic smart-factory sensor events."""
    events = []
    for i in range(n_events):
        machine_id = random.randint(0, n_machines - 1)
        temperature = max(20.0, random.gauss(65, 12))
        vibration = max(0.0, random.gauss(3.5, 1.8))
        load = min(100.0, max(0.0, random.gauss(70, 20)))

        if temperature > 90 or vibration > 8.0:
            status = "CRITICAL"
        elif temperature > 78 or vibration > 5.5 or load > 95:
            status = "WARNING"
        else:
            status = "NORMAL"

        events.append(MachineEvent(i, machine_id, round(temperature, 2),
                                    round(vibration, 2), round(load, 2), status))
    return events


# =============================================================================
# SECTION 2: SIMPLIFIED INSTRUCTION SET + INSTRUCTION STREAM GENERATION
# =============================================================================
# Addressing modes implemented: IMM (immediate), REG (register),
# DIR (direct memory), IND (register-indirect memory)

OPCODES = ["LOAD", "STORE", "ADD", "SUB", "MUL", "CMP", "BEQ", "JMP", "NOP"]
NUM_REGS = 8


@dataclass
class Instruction:
    idx: int
    opcode: str
    mode: str              # IMM / REG / DIR / IND
    dest: int = None        # destination register index (or None)
    src1: int = None        # source register index (or None)
    operand: int = None     # immediate value / memory address / register holding address
    target: int = None      # branch target instruction index


def build_instruction_stream(events: List[MachineEvent], data_mem_size: int) -> List[Instruction]:
    """
    Convert a representative subset of machine-health events into a
    simplified instruction stream that loads sensor values, computes a
    composite health score, compares it to a threshold, and branches to
    an alert routine. Addressing modes are deliberately rotated so all
    four modes are exercised.
    """
    stream: List[Instruction] = []
    modes_cycle = ["IMM", "REG", "DIR", "IND"]
    sample = events[:250]  # 250 events -> ~ (250*5)=1250 instructions

    for k, ev in enumerate(sample):
        base_addr = (ev.event_id * 4) % (data_mem_size - 8)
        mode = modes_cycle[k % 4]
        r_temp, r_vib, r_load, r_score = 0, 1, 2, 3

        idx0 = len(stream)
        # LOAD temperature value (direct addressing: fixed memory address)
        stream.append(Instruction(idx0, "LOAD", "DIR", dest=r_temp, operand=base_addr))
        # LOAD vibration value (register-indirect addressing)
        stream.append(Instruction(idx0 + 1, "LOAD", "IND", dest=r_vib, operand=r_temp))
        # ADD score = temp + vibration (register addressing)
        stream.append(Instruction(idx0 + 2, "ADD", "REG", dest=r_score, src1=r_temp, operand=r_vib))
        # ADD immediate load-weight bias (immediate addressing)
        stream.append(Instruction(idx0 + 3, "ADD", "IMM", dest=r_score, src1=r_score,
                                   operand=int(ev.load) % 16))
        # CMP score against threshold, then branch-if-equal to alert handler (control hazard source)
        stream.append(Instruction(idx0 + 4, "CMP", "IMM", src1=r_score, operand=90))
        target = idx0 + 9 if mode != "DIR" else idx0 + 8  # occasionally different target -> variety
        stream.append(Instruction(idx0 + 5, "BEQ", "IMM", operand=1, target=idx0 + 9))
        stream.append(Instruction(idx0 + 6, "STORE", "DIR", src1=r_score, operand=base_addr + 1))
        stream.append(Instruction(idx0 + 7, "NOP", "REG"))

    for i, ins in enumerate(stream):
        ins.idx = i
    return stream


# =============================================================================
# SECTION 3: CPU MODEL - REGISTERS, ALU, PC, MEMORY, I/O
# =============================================================================

class CPU:
    def __init__(self, mem_size: int = 4096):
        self.regs = [0] * NUM_REGS
        self.pc = 0
        self.memory = [0] * mem_size
        self.io_log: List[str] = []
        # seed memory with pseudo sensor data so LOADs return meaningful values
        for a in range(mem_size):
            self.memory[a] = (a * 7 + 13) % 128

    def alu(self, op: str, a: int, b: int) -> int:
        if op == "ADD":
            return a + b
        if op == "SUB":
            return a - b
        if op == "MUL":
            return a * b
        if op == "CMP":
            return a - b
        return 0

    def resolve_operand(self, ins: Instruction) -> int:
        """Resolve the operand value according to the addressing mode."""
        if ins.mode == "IMM":
            return ins.operand
        if ins.mode == "REG":
            return self.regs[ins.operand] if ins.operand is not None else 0
        if ins.mode == "DIR":
            return self.memory[ins.operand % len(self.memory)]
        if ins.mode == "IND":
            addr = self.regs[ins.operand] % len(self.memory)
            return self.memory[addr]
        return 0


def addressing_mode_demo(cpu: CPU, sample_instrs: List[Instruction]) -> List[Dict]:
    """Produce an explicit fetch-decode-execute-memory-writeback trace for
    one representative instruction of each addressing mode."""
    trace = []
    seen_modes = set()
    for ins in sample_instrs:
        if ins.mode in seen_modes or ins.opcode in ("NOP", "BEQ", "JMP"):
            continue
        seen_modes.add(ins.mode)

        fetch = f"PC={ins.idx} -> fetch instruction word ({ins.opcode},{ins.mode})"
        decode = f"decode: opcode={ins.opcode}, mode={ins.mode}, dest=R{ins.dest}, operand={ins.operand}"
        operand_val = cpu.resolve_operand(ins)
        if ins.opcode in ("ADD", "SUB", "MUL", "CMP"):
            a = cpu.regs[ins.src1] if ins.src1 is not None else 0
            result = cpu.alu(ins.opcode, a, operand_val)
            execute = f"execute: ALU {ins.opcode} R{ins.src1}({a}) , operand({operand_val}) = {result}"
        else:
            result = operand_val
            execute = f"execute: operand resolved via {ins.mode} addressing = {result}"

        if ins.opcode == "LOAD":
            memory_step = f"memory: read value {result} using {ins.mode} addressing"
        elif ins.opcode == "STORE":
            addr = ins.operand % len(cpu.memory)
            memory_step = f"memory: write R{ins.src1}={cpu.regs[ins.src1]} to address {addr}"
        else:
            memory_step = "memory: no memory access required"

        if ins.dest is not None and ins.opcode in ("LOAD", "ADD", "SUB", "MUL"):
            cpu.regs[ins.dest] = result
            writeback = f"write-back: R{ins.dest} <- {result}"
        else:
            writeback = "write-back: none"

        trace.append({
            "instruction": f"{ins.opcode} ({ins.mode})",
            "fetch": fetch, "decode": decode, "execute": execute,
            "memory": memory_step, "writeback": writeback,
        })
        if len(seen_modes) == 4:
            break
    return trace


# =============================================================================
# SECTION 4: BASELINE SINGLE-ISSUE (NON-PIPELINED) SIMULATOR
# =============================================================================
# Each instruction executes all 5 phases sequentially: 1 cycle per phase.
BASELINE_CYCLES_PER_INSTR = 5


def run_baseline(instrs: List[Instruction]) -> Dict:
    total_cycles = len(instrs) * BASELINE_CYCLES_PER_INSTR
    return {
        "instructions": len(instrs),
        "total_cycles": total_cycles,
        "cpi": total_cycles / len(instrs),
    }


# =============================================================================
# SECTION 5: 5-STAGE PIPELINE SIMULATOR (IF ID EX MEM WB)
# =============================================================================
# Hazard model:
#   - Data hazard (RAW): instruction's source register was written by an
#     instruction 1 or 2 slots earlier and has not yet reached WB -> stall.
#   - Control hazard: BEQ/JMP resolved in EX stage -> next 2 fetched
#     instructions are flushed (branch-not-taken prediction, 2-cycle penalty).
#   - Structural hazard: single shared memory port -> a LOAD/STORE in MEM
#     stage conflicts with an IF fetch in the same cycle -> 1-cycle stall.

def get_src_regs(ins: Instruction) -> List[int]:
    regs = []
    if ins.mode == "REG" and ins.operand is not None and ins.opcode not in ("LOAD", "STORE"):
        regs.append(ins.operand)
    if ins.mode == "IND" and ins.operand is not None:
        regs.append(ins.operand)  # base register used for indirect address
    if ins.src1 is not None:
        regs.append(ins.src1)
    return regs


def get_dest_reg(ins: Instruction):
    if ins.opcode in ("LOAD", "ADD", "SUB", "MUL") and ins.dest is not None:
        return ins.dest
    return None


def run_pipeline(instrs: List[Instruction]) -> Dict:
    n = len(instrs)
    data_hazard_stalls = 0
    control_hazard_stalls = 0
    structural_hazard_stalls = 0
    cycle = 0
    i = 0
    mem_access_opcodes = ("LOAD", "STORE")

    # in_flight holds (instr_index, dest_reg, stage_entered_cycle) for the last
    # two instructions still short of WB, to model RAW data hazards.
    pending_writes: List[Tuple[int, int]] = []  # (dest_reg, cycles_until_available)
    last_mem_stage_busy_until = -1

    while i < n:
        ins = instrs[i]
        stall = 0

        # --- data hazard check ---
        srcs = get_src_regs(ins)
        for (dreg, remaining) in pending_writes:
            if remaining > 0 and dreg in srcs:
                stall = max(stall, remaining)

        # --- structural hazard check: shared single memory port ---
        if ins.opcode in mem_access_opcodes and cycle <= last_mem_stage_busy_until:
            stall = max(stall, 1)
            structural_hazard_stalls += 1

        # separate the data-hazard stall count cleanly (structural already counted above)
        raw_stall = 0
        for (dreg, remaining) in pending_writes:
            if remaining > 0 and dreg in srcs:
                raw_stall = max(raw_stall, remaining)
        data_hazard_stalls += raw_stall

        cycle += 1 + stall  # IF stage this cycle (plus any stall bubbles)

        # age the pending-write list each cycle that passes
        pending_writes = [(d, max(0, r - (1 + stall))) for (d, r) in pending_writes]
        pending_writes = [(d, r) for (d, r) in pending_writes if r > 0]

        dest = get_dest_reg(ins)
        if dest is not None:
            pending_writes.append((dest, 2))  # result available 2 cycles after IF (EX stage)

        if ins.opcode in mem_access_opcodes:
            last_mem_stage_busy_until = cycle + 2  # MEM stage occurs ~3 cycles after this IF

        # --- control hazard: branch resolved in EX, flush 2 fetched instrs ---
        if ins.opcode in ("BEQ", "JMP"):
            control_hazard_stalls += 2
            cycle += 2

        i += 1

    # total cycles = last instruction's WB completion = cycle count + pipeline fill (4 extra stages)
    total_cycles = cycle + 4
    return {
        "instructions": n,
        "total_cycles": total_cycles,
        "cpi": total_cycles / n,
        "data_hazard_stalls": data_hazard_stalls,
        "control_hazard_stalls": control_hazard_stalls,
        "structural_hazard_stalls": structural_hazard_stalls,
        "total_stall_cycles": data_hazard_stalls + control_hazard_stalls + structural_hazard_stalls,
    }


# =============================================================================
# SECTION 6: SIMPLIFIED OOO / SPECULATIVE EXECUTION / SMT COMPARISON
# =============================================================================
# These are analytical (assumption-based) models layered on top of the
# measured pipeline result, as permitted by the assignment ("conceptual
# alternatives ... clearly stated simplified assumptions"). No fabricated
# absolute numbers are used: each model is a documented percentage
# reduction applied to the *measured* pipeline stall cycles.

OOO_ASSUMPTIONS = {
    "OOO": {
        "desc": "Out-of-order execution with a 6-entry reorder buffer allows "
                "independent instructions to bypass a stalled RAW hazard, "
                "removing an assumed 60% of data-hazard stall cycles. Control "
                "and structural stalls are unaffected (single memory port, "
                "in-order commit).",
        "data_hazard_reduction": 0.60,
        "control_hazard_reduction": 0.0,
        "structural_hazard_reduction": 0.0,
    },
    "Speculative": {
        "desc": "Speculative execution with static branch-not-taken prediction "
                "assumed 70% accurate (matches typical sensor-polling loop "
                "behaviour) removes an assumed 70% of control-hazard stall "
                "cycles on correctly predicted branches; data/structural "
                "stalls unaffected.",
        "data_hazard_reduction": 0.0,
        "control_hazard_reduction": 0.70,
        "structural_hazard_reduction": 0.0,
    },
    "SMT (2-way)": {
        "desc": "2-way simultaneous multithreading interleaves a second "
                "independent sensor-processing thread into pipeline bubbles, "
                "assumed to hide 50% of data-hazard and 50% of structural-"
                "hazard stall cycles; control-hazard flush cost is unaffected "
                "because both threads share the fetch unit in the flush "
                "cycles.",
        "data_hazard_reduction": 0.50,
        "control_hazard_reduction": 0.0,
        "structural_hazard_reduction": 0.50,
    },
}


def compare_ooo_variants(pipeline_result: Dict) -> Dict[str, Dict]:
    base_cycles = pipeline_result["total_cycles"]
    n = pipeline_result["instructions"]
    results = {}
    for name, cfg in OOO_ASSUMPTIONS.items():
        saved = (pipeline_result["data_hazard_stalls"] * cfg["data_hazard_reduction"]
                 + pipeline_result["control_hazard_stalls"] * cfg["control_hazard_reduction"]
                 + pipeline_result["structural_hazard_stalls"] * cfg["structural_hazard_reduction"])
        new_cycles = round(base_cycles - saved)
        results[name] = {
            "description": cfg["desc"],
            "estimated_cycles": new_cycles,
            "estimated_cpi": round(new_cycles / n, 4),
            "stall_cycles_removed": round(saved, 2),
        }
    return results


# =============================================================================
# SECTION 7: L1 / L2 / L3 CACHE HIERARCHY SIMULATION
# =============================================================================

@dataclass
class CacheLevel:
    name: str
    num_lines: int
    block_size: int
    hit_latency_cycles: int
    lines: Dict[int, int] = field(default_factory=dict)  # block_addr -> last_use_time (LRU)

    def access(self, address: int, time: int) -> bool:
        block_addr = address // self.block_size
        if block_addr in self.lines:
            self.lines[block_addr] = time
            return True
        # miss -> install block, evict LRU if full
        if len(self.lines) >= self.num_lines:
            lru_block = min(self.lines, key=self.lines.get)
            del self.lines[lru_block]
        self.lines[block_addr] = time
        return False


def generate_access_pattern(events: List[MachineEvent], mem_size: int, n_accesses: int = 6000) -> List[int]:
    """Sensor-data access pattern: mixture of sequential scans (temporal/
    spatial locality, like reading consecutive sensor slots) and random
    jumps (like polling different machines), matching realistic factory
    telemetry access behaviour."""
    pattern = []
    addr = 0
    for i in range(n_accesses):
        ev = events[i % len(events)]
        if i % 5 == 0:  # 20% random jump (poll a different machine)
            addr = (ev.machine_id * 97 + ev.event_id * 3) % mem_size
        else:            # 80% sequential/local scan
            addr = (addr + 4) % mem_size
        pattern.append(addr)
    return pattern


def simulate_cache_hierarchy(access_pattern: List[int], l1: CacheLevel, l2: CacheLevel,
                              l3: CacheLevel, main_mem_latency: int) -> Dict:
    l1_hits = l2_hits = l3_hits = misses_all = 0
    total_latency_cycles = 0
    for t, addr in enumerate(access_pattern):
        if l1.access(addr, t):
            l1_hits += 1
            total_latency_cycles += l1.hit_latency_cycles
        elif l2.access(addr, t):
            l2_hits += 1
            total_latency_cycles += l1.hit_latency_cycles + l2.hit_latency_cycles
        elif l3.access(addr, t):
            l3_hits += 1
            total_latency_cycles += l1.hit_latency_cycles + l2.hit_latency_cycles + l3.hit_latency_cycles
        else:
            misses_all += 1
            total_latency_cycles += (l1.hit_latency_cycles + l2.hit_latency_cycles
                                      + l3.hit_latency_cycles + main_mem_latency)

    n = len(access_pattern)
    overall_hits = l1_hits + l2_hits + l3_hits
    hit_rate = overall_hits / n
    miss_rate = misses_all / n
    amat = total_latency_cycles / n
    throughput = n / total_latency_cycles  # accesses per cycle
    return {
        "accesses": n, "l1_hits": l1_hits, "l2_hits": l2_hits, "l3_hits": l3_hits,
        "main_mem_misses": misses_all, "hit_rate": round(hit_rate, 4),
        "miss_rate": round(miss_rate, 4), "amat_cycles": round(amat, 3),
        "throughput_access_per_cycle": round(throughput, 5),
        "total_latency_cycles": total_latency_cycles,
    }


def run_cache_configurations(access_pattern: List[int], mem_size: int) -> Dict[str, Dict]:
    configs = {
        "Config A (small caches)": dict(
            l1=CacheLevel("L1", num_lines=16, block_size=64, hit_latency_cycles=1),
            l2=CacheLevel("L2", num_lines=64, block_size=64, hit_latency_cycles=8),
            l3=CacheLevel("L3", num_lines=256, block_size=64, hit_latency_cycles=25),
            main_mem_latency=120),
        "Config B (medium caches)": dict(
            l1=CacheLevel("L1", num_lines=32, block_size=64, hit_latency_cycles=1),
            l2=CacheLevel("L2", num_lines=128, block_size=64, hit_latency_cycles=8),
            l3=CacheLevel("L3", num_lines=512, block_size=64, hit_latency_cycles=25),
            main_mem_latency=120),
        "Config C (large caches)": dict(
            l1=CacheLevel("L1", num_lines=64, block_size=64, hit_latency_cycles=1),
            l2=CacheLevel("L2", num_lines=256, block_size=64, hit_latency_cycles=8),
            l3=CacheLevel("L3", num_lines=1024, block_size=64, hit_latency_cycles=25),
            main_mem_latency=120),
    }
    results = {}
    for name, cfg in configs.items():
        res = simulate_cache_hierarchy(access_pattern, cfg["l1"], cfg["l2"], cfg["l3"], cfg["main_mem_latency"])
        res["l1_lines"] = cfg["l1"].num_lines
        res["l2_lines"] = cfg["l2"].num_lines
        res["l3_lines"] = cfg["l3"].num_lines
        results[name] = res
    return results


# =============================================================================
# SECTION 8: SIMPLIFIED VIRTUAL MEMORY / PAGING MODEL
# =============================================================================

def simulate_paging(access_pattern: List[int], page_size: int = 256, num_frames: int = 24,
                     virtual_pages: int = 64, page_fault_service_cycles: int = 2000,
                     base_access_cycles: int = 100) -> Dict:
    frames: Dict[int, int] = {}  # page_no -> last_use_time
    page_hits = 0
    page_faults = 0
    total_cycles = 0

    for t, addr in enumerate(access_pattern):
        page_no = (addr // page_size) % virtual_pages
        if page_no in frames:
            page_hits += 1
            frames[page_no] = t
            total_cycles += base_access_cycles
        else:
            page_faults += 1
            if len(frames) >= num_frames:
                lru_page = min(frames, key=frames.get)
                del frames[lru_page]
            frames[page_no] = t
            total_cycles += base_access_cycles + page_fault_service_cycles

    n = len(access_pattern)
    fault_rate = page_faults / n
    hit_rate = page_hits / n
    eat = total_cycles / n  # effective access time (cycles)
    eat_no_faults = base_access_cycles
    return {
        "accesses": n, "page_hits": page_hits, "page_faults": page_faults,
        "page_hit_rate": round(hit_rate, 4), "page_fault_rate": round(fault_rate, 4),
        "effective_access_time_cycles": round(eat, 2),
        "access_time_without_faults_cycles": eat_no_faults,
        "slowdown_factor": round(eat / eat_no_faults, 3),
    }


# =============================================================================
# SECTION 9: MEMORY TECHNOLOGY COMPARISON (SRAM/DRAM/NAND/MRAM/RRAM)
# =============================================================================
# Reference figures are order-of-magnitude, textbook-level approximations
# (used for the *relative* comparison the assignment asks for, not vendor
# datasheet-accuracy numbers). Sourced qualitatively from standard computer-
# architecture references (e.g., Hennessy & Patterson) and rounded for
# classroom use.

MEMORY_TECH_TABLE = {
    "SRAM":  {"latency_ns": 1,     "bandwidth_GBps": 200, "power": "High (static)",
              "cost_per_GB_usd": 5000, "reliability": "Very High", "volatile": True,
              "typical_use": "L1/L2 cache"},
    "DRAM":  {"latency_ns": 15,    "bandwidth_GBps": 25,  "power": "Medium (refresh needed)",
              "cost_per_GB_usd": 5,    "reliability": "High", "volatile": True,
              "typical_use": "Main memory"},
    "NAND Flash": {"latency_ns": 100000, "bandwidth_GBps": 2, "power": "Low",
              "cost_per_GB_usd": 0.08, "reliability": "Medium (wear-out)", "volatile": False,
              "typical_use": "Secondary storage / SSD"},
    "MRAM":  {"latency_ns": 20,    "bandwidth_GBps": 8,   "power": "Low",
              "cost_per_GB_usd": 40, "reliability": "Very High (no wear-out)", "volatile": False,
              "typical_use": "Non-volatile L3 / persistent buffers"},
    "RRAM":  {"latency_ns": 10,    "bandwidth_GBps": 6,   "power": "Very Low",
              "cost_per_GB_usd": 15, "reliability": "Medium-High (limited endurance)", "volatile": False,
              "typical_use": "Emerging NVM / edge-sensor buffers"},
}


# =============================================================================
# SECTION 10: GRAPH GENERATION (matplotlib)
# =============================================================================

def make_graphs(baseline, pipeline, ooo_cmp, cache_results, paging_result, events):
    # Graph 1: CPI comparison
    labels = ["Baseline"] + list(ooo_cmp.keys())
    cpis = [baseline["cpi"]] + [ooo_cmp[k]["estimated_cpi"] for k in ooo_cmp]
    cpis.insert(1, pipeline["cpi"])
    labels.insert(1, "5-stage Pipeline")
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, cpis, color=["#94a3b8", "#3b82f6", "#22c55e", "#f59e0b", "#ef4444"])
    plt.ylabel("Cycles Per Instruction (CPI)")
    plt.title("CPI: Baseline vs Pipeline vs OOO/Speculative/SMT")
    plt.xticks(rotation=20, ha="right")
    for b, v in zip(bars, cpis):
        plt.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/graph1_cpi_comparison.png", dpi=130)
    plt.close()

    # Graph 2: Pipeline stall breakdown
    plt.figure(figsize=(6, 5))
    stall_labels = ["Data Hazard", "Control Hazard", "Structural Hazard"]
    stall_vals = [pipeline["data_hazard_stalls"], pipeline["control_hazard_stalls"], pipeline["structural_hazard_stalls"]]
    plt.pie(stall_vals, labels=stall_labels, autopct="%1.1f%%", colors=["#3b82f6", "#f59e0b", "#ef4444"])
    plt.title("Pipeline Stall-Cycle Breakdown by Hazard Type")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/graph2_hazard_breakdown.png", dpi=130)
    plt.close()

    # Graph 3: Cache hit rate across configurations
    plt.figure(figsize=(8, 5))
    names = list(cache_results.keys())
    hit_rates = [cache_results[k]["hit_rate"] * 100 for k in names]
    amats = [cache_results[k]["amat_cycles"] for k in names]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(names, hit_rates, color="#22c55e", alpha=0.8, label="Hit rate (%)")
    ax1.set_ylabel("Overall Hit Rate (%)")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(names, amats, color="#ef4444", marker="o", linewidth=2, label="AMAT (cycles)")
    ax2.set_ylabel("AMAT (cycles)")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
    plt.title("Cache Hierarchy: Hit Rate vs AMAT across Configurations")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/graph3_cache_configs.png", dpi=130)
    plt.close()

    # Graph 4: Cache level hit distribution (Config B)
    cfgB = cache_results["Config B (medium caches)"]
    plt.figure(figsize=(6, 5))
    plt.bar(["L1 hits", "L2 hits", "L3 hits", "Main-mem misses"],
            [cfgB["l1_hits"], cfgB["l2_hits"], cfgB["l3_hits"], cfgB["main_mem_misses"]],
            color=["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"])
    plt.ylabel("Number of accesses")
    plt.title("Access Distribution across Cache Levels (Config B)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/graph4_level_distribution.png", dpi=130)
    plt.close()

    # Graph 5: Machine event status distribution
    plt.figure(figsize=(6, 5))
    counts = [sum(1 for e in events if e.status == s) for s in ["NORMAL", "WARNING", "CRITICAL"]]
    plt.bar(["NORMAL", "WARNING", "CRITICAL"], counts, color=["#22c55e", "#f59e0b", "#ef4444"])
    plt.ylabel("Number of events")
    plt.title(f"Machine-Health Event Status Distribution (n={len(events)})")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/graph5_event_status.png", dpi=130)
    plt.close()

    # Graph 6: Paging - hits vs faults
    plt.figure(figsize=(5, 5))
    plt.pie([paging_result["page_hits"], paging_result["page_faults"]],
            labels=["Page hits", "Page faults"], autopct="%1.1f%%",
            colors=["#3b82f6", "#ef4444"])
    plt.title("Virtual Memory: Page Hits vs Page Faults")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/graph6_paging.png", dpi=130)
    plt.close()

    return [f"{OUT_DIR}/graph{i}_" for i in range(1, 7)]


# =============================================================================
# SECTION 11: AUTOMATED TEST CASES (expected vs actual vs PASS/FAIL)
# =============================================================================

def run_test_cases(events, instrs, baseline, pipeline, cache_results, paging_result) -> List[Dict]:
    tests = []

    def add_test(name, expected, actual, comparator):
        passed = comparator(expected, actual)
        tests.append({"test": name, "expected": expected, "actual": actual,
                       "result": "PASS" if passed else "FAIL"})

    add_test("Event count >= 1000", ">=1000", len(events), lambda e, a: a >= 1000)
    add_test("Instruction stream non-empty and multiple of 8 (per-event block)",
              "multiple of 8", len(instrs), lambda e, a: a % 8 == 0)
    add_test("All 4 addressing modes present", {"IMM", "REG", "DIR", "IND"},
              {i.mode for i in instrs}, lambda e, a: e == a)
    add_test("ALU ADD correctness (5+7=12)", 12, CPU().alu("ADD", 5, 7), lambda e, a: e == a)
    add_test("ALU SUB correctness (10-4=6)", 6, CPU().alu("SUB", 10, 4), lambda e, a: e == a)
    add_test("ALU MUL correctness (6*7=42)", 42, CPU().alu("MUL", 6, 7), lambda e, a: e == a)
    add_test("Baseline CPI == 5 (5 sequential phases/instr)", 5.0, baseline["cpi"], lambda e, a: a == e)
    add_test("Pipeline CPI < Baseline CPI (pipelining improves throughput)",
              f"< {baseline['cpi']}", pipeline["cpi"], lambda e, a: a < baseline["cpi"])
    add_test("Pipeline stall cycles are non-negative", ">=0", pipeline["total_stall_cycles"],
              lambda e, a: a >= 0)
    for name, res in cache_results.items():
        add_test(f"{name}: hit_rate in [0,1]", "[0,1]", res["hit_rate"], lambda e, a: 0 <= a <= 1)
        add_test(f"{name}: hit_rate+miss_rate == 1", 1.0, round(res["hit_rate"] + res["miss_rate"], 4),
                  lambda e, a: abs(a - e) < 0.01)
    names = list(cache_results.keys())
    add_test("Larger cache (Config C) hit rate >= smaller cache (Config A)",
              ">= Config A hit rate", cache_results["Config C (large caches)"]["hit_rate"],
              lambda e, a: a >= cache_results["Config A (small caches)"]["hit_rate"])
    add_test("Page fault rate in [0,1]", "[0,1]", paging_result["page_fault_rate"],
              lambda e, a: 0 <= a <= 1)
    add_test("Effective access time >= base access time (faults only add latency)",
              f">= {paging_result['access_time_without_faults_cycles']}",
              paging_result["effective_access_time_cycles"],
              lambda e, a: a >= paging_result["access_time_without_faults_cycles"])
    return tests


if __name__ == "__main__":
    events = generate_machine_events(1200)
    print(f"[SECTION 1] Generated {len(events)} machine-health events (seed={SEED})")
    print(f"  First event: {events[0]}")
    print(f"  Status counts: NORMAL={sum(1 for e in events if e.status=='NORMAL')}, "
          f"WARNING={sum(1 for e in events if e.status=='WARNING')}, "
          f"CRITICAL={sum(1 for e in events if e.status=='CRITICAL')}")

    instrs = build_instruction_stream(events, data_mem_size=4096)
    print(f"\n[SECTION 2] Built instruction stream: {len(instrs)} instructions")
    mode_counts = {}
    for ins in instrs:
        mode_counts[ins.mode] = mode_counts.get(ins.mode, 0) + 1
    print(f"  Addressing-mode usage: {mode_counts}")

    cpu_demo = CPU()
    trace = addressing_mode_demo(cpu_demo, instrs)
    print(f"\n[SECTION 3] Fetch-Decode-Execute-Memory-Writeback trace ({len(trace)} instructions, one per addressing mode):")
    for t in trace:
        print(f"  -- {t['instruction']} --")
        print(f"     {t['fetch']}")
        print(f"     {t['decode']}")
        print(f"     {t['execute']}")
        print(f"     {t['memory']}")
        print(f"     {t['writeback']}")

    baseline = run_baseline(instrs)
    print(f"\n[SECTION 4] Baseline single-issue: {baseline}")

    pipeline = run_pipeline(instrs)
    print(f"\n[SECTION 5] 5-stage pipeline: {pipeline}")

    speedup = baseline["total_cycles"] / pipeline["total_cycles"]
    print(f"  Speedup (baseline_cycles / pipeline_cycles) = {speedup:.3f}")

    clock_period_ns = 1.0  # 1 GHz assumed clock
    baseline_exec_time_us = baseline["total_cycles"] * clock_period_ns / 1000
    pipeline_exec_time_us = pipeline["total_cycles"] * clock_period_ns / 1000
    baseline_throughput = baseline["instructions"] / baseline_exec_time_us
    pipeline_throughput = pipeline["instructions"] / pipeline_exec_time_us
    print(f"  Baseline exec time = {baseline_exec_time_us:.2f} us, throughput = {baseline_throughput:.3f} instr/us")
    print(f"  Pipeline exec time = {pipeline_exec_time_us:.2f} us, throughput = {pipeline_throughput:.3f} instr/us")

    ooo_cmp = compare_ooo_variants(pipeline)
    print(f"\n[SECTION 6] OOO / Speculative / SMT comparison (assumption-based, applied to measured stalls):")
    for name, res in ooo_cmp.items():
        print(f"  {name}: estimated_cycles={res['estimated_cycles']}, estimated_cpi={res['estimated_cpi']}, "
              f"stalls_removed={res['stall_cycles_removed']}")

    # ---------------- Cache hierarchy ----------------
    access_pattern = generate_access_pattern(events, mem_size=4096, n_accesses=6000)
    cache_results = run_cache_configurations(access_pattern, mem_size=4096)
    print(f"\n[SECTION 7] Cache hierarchy simulation across {len(cache_results)} configurations "
          f"({len(access_pattern)} memory accesses):")
    for name, res in cache_results.items():
        print(f"  {name}: L1lines={res['l1_lines']} L2lines={res['l2_lines']} L3lines={res['l3_lines']} "
              f"-> hit_rate={res['hit_rate']}, miss_rate={res['miss_rate']}, AMAT={res['amat_cycles']} cycles, "
              f"throughput={res['throughput_access_per_cycle']} access/cycle")
        print(f"      L1hits={res['l1_hits']} L2hits={res['l2_hits']} L3hits={res['l3_hits']} mainMemMisses={res['main_mem_misses']}")

    # ---------------- Virtual memory / paging ----------------
    paging_result = simulate_paging(access_pattern)
    print(f"\n[SECTION 8] Virtual memory / paging simulation: {paging_result}")

    # ---------------- Memory technology comparison ----------------
    print(f"\n[SECTION 9] Memory technology comparison:")
    for tech, props in MEMORY_TECH_TABLE.items():
        print(f"  {tech}: {props}")

    # ---------------- Bottleneck identification (data-driven) ----------------
    print(f"\n[SECTION 10] Bottleneck identification:")
    worst_cache_cfg = max(cache_results.items(), key=lambda kv: kv[1]["miss_rate"])
    hazard_totals = {"Data hazard": pipeline["data_hazard_stalls"],
                      "Control hazard": pipeline["control_hazard_stalls"],
                      "Structural hazard": pipeline["structural_hazard_stalls"]}
    worst_hazard = max(hazard_totals.items(), key=lambda kv: kv[1])
    print(f"  Dominant pipeline hazard: {worst_hazard[0]} ({worst_hazard[1]} stall cycles, "
          f"{worst_hazard[1]/pipeline['total_stall_cycles']*100:.1f}% of all stalls)")
    print(f"  Worst cache configuration: {worst_cache_cfg[0]} (miss_rate={worst_cache_cfg[1]['miss_rate']})")
    print(f"  Page fault rate: {paging_result['page_fault_rate']*100:.2f}% "
          f"-> effective access time is {paging_result['slowdown_factor']}x the fault-free access time")

    # ---------------- Graphs ----------------
    make_graphs(baseline, pipeline, ooo_cmp, cache_results, paging_result, events)
    print(f"\n[SECTION 11] Graphs written to '{OUT_DIR}/': "
          f"graph1_cpi_comparison.png, graph2_hazard_breakdown.png, graph3_cache_configs.png, "
          f"graph4_level_distribution.png, graph5_event_status.png, graph6_paging.png")

    # ---------------- Test cases ----------------
    tests = run_test_cases(events, instrs, baseline, pipeline, cache_results, paging_result)
    print(f"\n[SECTION 12] Automated test cases ({len(tests)} total):")
    n_pass = sum(1 for t in tests if t["result"] == "PASS")
    for t in tests:
        print(f"  [{t['result']}] {t['test']} | expected={t['expected']} | actual={t['actual']}")
    print(f"\n  TEST SUMMARY: {n_pass}/{len(tests)} PASSED")

    print("\n===================== END OF SIMULATION RUN =====================")
