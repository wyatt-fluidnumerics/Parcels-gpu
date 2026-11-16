"""Host-versus-GPU split of a run, from an Nsight Systems trace exported to SQLite.

Record:  nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none -o trace python <script> ...
Export:  nsys export --type sqlite -o trace.sqlite trace.nsys-rep
Analyse: python gpu_benchmarks/nsys_host_gpu_split.py trace.sqlite --last-seconds 4.64

Reports, for the last `--last-seconds` of GPU activity (the timed run):
  GPU busy        : union of the intervals in which at least one kernel was executing
  GPU copies      : union of host<->GPU memory copy intervals, and bytes per direction
  host in CUDA    : time host threads spent inside CUDA runtime calls (launching, copying, synchronizing)
  host in OS calls: time host threads spent inside traced OS runtime calls (file reads, waits), per thread
"""

import argparse
import sqlite3

NS = 1e9


def union_seconds(intervals: list[tuple[int, int]]) -> float:
    total = 0
    current_start, current_end = None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total / NS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sqlite_path")
    parser.add_argument("--last-seconds", type=float, required=True)
    args = parser.parse_args()
    con = sqlite3.connect(args.sqlite_path)

    window_end = con.execute(
        "select max(e) from (select max(end) e from CUPTI_ACTIVITY_KIND_KERNEL union all select max(end) from CUPTI_ACTIVITY_KIND_MEMCPY)"
    ).fetchone()[0]
    window_start = window_end - int(args.last_seconds * NS)
    in_window = f"start >= {window_start} and end <= {window_end}"
    window_s = args.last_seconds
    print(f"window: last {window_s:.2f} s of GPU activity")

    kernels = con.execute(f"select start, end from CUPTI_ACTIVITY_KIND_KERNEL where {in_window}").fetchall()
    gpu_busy_s = union_seconds(kernels)
    print(f"GPU busy (a kernel executing): {gpu_busy_s:6.2f} s = {100 * gpu_busy_s / window_s:4.1f}% of the window, {len(kernels)} kernel launches")

    print("  top kernels by GPU time:")
    rows = con.execute(
        f"""select s.value, count(*), sum(k.end - k.start) from CUPTI_ACTIVITY_KIND_KERNEL k
            join StringIds s on s.id = k.shortName where {in_window.replace('start', 'k.start').replace('end <=', 'k.end <=')}
            group by s.value order by 3 desc limit 8"""
    ).fetchall()
    for name, count, nanoseconds in rows:
        print(f"    {nanoseconds / NS:6.3f} s  {count:6d} launches  {name[:70]}")

    copies = con.execute(f"select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where {in_window}").fetchall()
    print(f"GPU copies: {union_seconds([(s, e) for s, e, _, _ in copies]):6.2f} s, {len(copies)} copies")
    bytes_by_kind: dict[int, int] = {}
    for _, _, nbytes, kind in copies:
        bytes_by_kind[kind] = bytes_by_kind.get(kind, 0) + nbytes
    kind_names = {1: "host to GPU", 2: "GPU to host", 8: "GPU to GPU"}
    for kind, nbytes in sorted(bytes_by_kind.items()):
        print(f"    {kind_names.get(kind, f'kind {kind}')}: {nbytes / 1e9:6.2f} GB")

    thread_names = dict(con.execute("select t.globalTid, s.value from ThreadNames t join StringIds s on s.id = t.nameId").fetchall())

    print("host time inside CUDA runtime calls, per thread and call:")
    rows = con.execute(
        f"""select r.globalTid, s.value, count(*), sum(r.end - r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r
            join StringIds s on s.id = r.nameId where {in_window.replace('start', 'r.start').replace('end <=', 'r.end <=')}
            group by r.globalTid, s.value order by 4 desc limit 8"""
    ).fetchall()
    for tid, name, count, nanoseconds in rows:
        print(f"    {nanoseconds / NS:6.3f} s  {count:6d} calls  {name[:40]:40s} thread {thread_names.get(tid, tid)}")

    print("host time inside traced OS calls, per thread and call:")
    rows = con.execute(
        f"""select o.globalTid, s.value, count(*), sum(o.end - o.start) from OSRT_API o
            join StringIds s on s.id = o.nameId where {in_window.replace('start', 'o.start').replace('end <=', 'o.end <=')}
            and o.nestingLevel = 0 group by o.globalTid, s.value order by 4 desc limit 12"""
    ).fetchall()
    for tid, name, count, nanoseconds in rows:
        print(f"    {nanoseconds / NS:6.3f} s  {count:6d} calls  {name[:28]:28s} thread {thread_names.get(tid, tid)}")
