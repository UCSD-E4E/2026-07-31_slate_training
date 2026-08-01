# Upstream follow-up — `LaserExtrinsics` persistence, and dives 347 / 466

Two items left over from the slate-label panel-offset fix
(api-workflow-worker v1.46.5 / data-worker v2.9.2, PR #462). Neither is urgent
today; both get worse as soon as model-assisted slate labeling ships, because
that increases recalibration volume.

Context: the original defects and their evidence are in
[upstream_bug_report.md](upstream_bug_report.md).

---

## 1. `LaserExtrinsics` inserts NULL `created_at` and appends duplicates

**Severity: dormant, but the blast radius grows with recalibration volume.**

### What's wrong

`services/fishsense-api/src/fishsense_api/models/laser_extrinsics.py:17`:

```python
created_at: datetime | None = Field(sa_type=DateTime(timezone=True), default=None)
```

No `server_default`, so the calibration path inserts **NULL**. The
"latest-wins" read then orders by `created_at DESC`; with Postgres sorting
NULLs first under `DESC`, the lookup fails to resolve a row and returns **404**.

Separately, `put_laser_extrinsics` merges with `id=None`, so each call
**appends a new row** rather than upserting on `dive_id`. Every recalculation
leaves another duplicate behind.

### Why it matters now

It stayed dormant only because recalibration was rare. The slate detector
exists to make labeling cheap, which makes *re*-calibration routine — so both
the duplicate accumulation and the 404 move from theoretical to frequent.

There are also **hand-patched rows** in prod from the PR #462 recovery. They're
correct but not reproducible; the next person to hit this gets a 404 with no
trail explaining why some rows have timestamps and others don't.

### Suggested fix

1. Give `created_at` a server default (`server_default=func.now()` via
   `sa_column_kwargs`), so the database supplies it regardless of caller.
2. Make `put_laser_extrinsics` a genuine upsert keyed on `dive_id`, not a
   merge with `id=None`.
3. Backfill: set `created_at` on existing NULL rows, and de-duplicate the rows
   already appended — keeping the most recent per `dive_id`.
4. Consider a uniqueness constraint on `dive_id` so this cannot silently
   recur. (Check first whether any dive is *intended* to carry a history of
   extrinsics; if so, make the latest-wins read explicit with
   `NULLS LAST` instead.)

---

## 2. Dives 347 and 466 still hold extrinsics from composite-space labels

These two couldn't be recalculated during the PR #462 recovery (2 and 0 valid
laser labels respectively, the rest superseded in the breach recovery), so they
retain their pre-fix extrinsics.

**Those values should not be left as-is.** They were computed from coordinates
now known to be in the wrong space, and the six dives that *could* be
recalculated all showed their old values were **outliers** against the corrected
ones. There is no reason to think these two differ — and stage 14 will keep
producing confidently wrong measurements from them with nothing to signal it.
An explicit gap is recoverable; a silently wrong calibration is not.

### Dive 466 — borrow the calibration, don't delete

Camera assignments across the eight affected dives:

| dive | camera | status |
| ---: | ---: | --- |
| 349 | **6** | recalculated cleanly |
| **466** | **6** | 0 valid laser labels |
| 383 | 3 | recalculated cleanly |
| 471 | 3 | recalculated cleanly |
| 347 | **2** | no other dive uses this camera |

**466 shares camera 6 with 349**, which recalculated cleanly. That is exactly
what `set_calibration_source` is for — point 466 at 349 rather than deleting.

*Domain check required:* this is only valid if the laser/camera mount was not
disturbed between those two dives. If it was, delete instead.

### Dive 347 — delete

Camera 2, used by no other dive, so there is no source to borrow from. It has
exactly 2 valid laser labels — precisely `MIN_LASER_POINTS`, which makes the
fit exactly-determined with **no residual left to validate it against**. The
only available check would be whether the result lands inside the ~10 cm rig
envelope the six good dives established, and an unvalidatable fit feeding fish
measurements is the failure mode this whole exercise was about.

---

## Suggested sequencing

1. `created_at` / upsert fix — it is a prerequisite for any further
   recalculation, including the two below.
2. Resolve 347 / 466.
3. Then wire in the slate `predict_slate` activity, which is what starts
   driving recalculation volume.
