/*
 * ddsketch.h -- Shared-memory DDSketch quantile sketch for Linux
 *
 * Relative-error quantiles: estimates any quantile of a distribution to within a
 * configured relative accuracy alpha, in a fixed amount of memory, using the
 * DDSketch algorithm. Each value falls into a logarithmic bucket key =
 * ceil(log_gamma(|v|)) with gamma = (1+alpha)/(1-alpha), so a bucket's
 * representative value is within alpha of every value it holds; the counts live
 * in a shared mapping so several processes feed one sketch. A write-preferring
 * futex rwlock with reader-slot dead-process recovery guards mutation. Two
 * sketches of the same (alpha, num_buckets) layout merge by an element-wise sum
 * of their bucket counts.
 *
 * Layout: Header -> reader_slots[1024] -> neg[num_buckets] -> pos[num_buckets]
 */

#ifndef DD_H
#define DD_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <pthread.h>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "ddsketch.h: requires little-endian architecture"
#endif


/* ================================================================
 * Constants
 * ================================================================ */

#define DD_MAGIC        0x4B534444  /* DDSketch */
#define DD_VERSION      1
#define DD_ERR_BUFLEN   256
#ifndef DD_READER_SLOTS
#define DD_READER_SLOTS 1024         /* max concurrent reader processes for dead-process recovery */
#endif
#define DD_MIN_BUCKETS  8
#define DD_MAX_BUCKETS  0x1000000U    /* 2^24 buckets per store */
#define DD_MIN_ALPHA    1e-6
#define DD_MAX_ALPHA    0.5

#define DD_ERR(fmt, ...) do { if (errbuf) snprintf(errbuf, DD_ERR_BUFLEN, fmt, ##__VA_ARGS__); } while (0)

/* ================================================================
 * Structs
 * ================================================================ */

/* Per-process slot for dead-process recovery.  Each shared rwlock counter
 * (the main rwlock-reader count, rwlock_waiters, rwlock_writers_waiting)
 * is mirrored here so a wrlock timeout can attribute and reverse a dead
 * process's contribution instead of waiting for the slow per-op timeout
 * drain. */
typedef struct {
    uint32_t pid;            /* 0 = unclaimed */
    uint32_t subcount;       /* in-flight rdlock acquisitions for this process */
    uint32_t waiters_parked; /* contribution to hdr->rwlock_waiters         */
    uint32_t writers_parked; /* contribution to hdr->rwlock_writers_waiting */
} DdReaderSlot;

struct DdHeader {
    uint32_t magic, version;          /* 0,4 */
    uint32_t num_buckets;             /* 8   buckets per store (positive and negative each) */
    uint32_t bias;                    /* 12  slot = key + bias (bias = num_buckets/2) */
    double   gamma;                   /* 16  log base (1+alpha)/(1-alpha) */
    double   inv_ln_gamma;            /* 24  1/ln(gamma), for key = ceil(ln|v| * inv_ln_gamma) */
    double   alpha;                   /* 32  configured relative accuracy */
    uint64_t total_count;             /* 40  total values inserted */
    uint64_t zero_count;              /* 48  count of exact-zero values */
    double   sum;                     /* 56  sum of all values (for mean) */
    double   min_value;               /* 64  smallest value inserted (+inf when empty) */
    double   max_value;               /* 72  largest value inserted (-inf when empty) */
    uint64_t neg_off;                 /* 80  offset of the negative-value store */
    uint64_t pos_off;                 /* 88  offset of the positive-value store */
    uint64_t total_size;              /* 96 */
    uint64_t reader_slots_off;        /* 104 */
    uint32_t rwlock;                  /* 112 */
    uint32_t rwlock_waiters;          /* 116 */
    uint32_t rwlock_writers_waiting;  /* 120 */
    uint32_t slotless_readers;  /* live readers holding the lock with no reader-slot */
    uint64_t stat_ops;                /* 128 */
    uint8_t  _pad[120];               /* 136..255 */
};
typedef struct DdHeader DdHeader;

_Static_assert(sizeof(DdHeader) == 256, "DdHeader must be 256 bytes");

/* ---- Process-local handle ---- */

typedef struct DdHandle {
    DdHeader     *hdr;
    DdReaderSlot *reader_slots;  /* DD_READER_SLOTS entries */
    void         *base;          /* mmap base */
    uint64_t      neg_off;       /* validated store offsets, cached: never re-read from the peer-writable header */
    uint64_t      pos_off;
    uint32_t      num_buckets;   /* cached */
    uint32_t      bias;          /* cached */
    double        gamma;         /* cached */
    double        inv_ln_gamma;  /* cached */
    size_t        mmap_size;
    char         *path;          /* backing file path (strdup'd) */
    int           backing_fd;    /* memfd or reopened-fd to close on destroy, -1 for file/anon */
    uint32_t      my_slot_idx;   /* UINT32_MAX if all slots taken (no recovery for this handle) */
    uint32_t      cached_pid;    /* getpid() cached at last slot claim */
    uint32_t      cached_fork_gen; /* dd_fork_gen value at last slot claim */
    uint32_t slotless_held; /* rwlock read-locks held with no reader-slot */
} DdHandle;

/* ================================================================
 * Futex-based write-preferring read-write lock
 * with reader-slot dead-process recovery
 * ================================================================ */

#define DD_RWLOCK_SPIN_LIMIT 32
#define DD_LOCK_TIMEOUT_SEC  2  /* FUTEX_WAIT timeout for stale lock detection */

static inline void dd_rwlock_spin_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

/* Extract writer PID from rwlock value (lower 31 bits when write-locked). */
#define DD_RWLOCK_WRITER_BIT 0x80000000U
#define DD_RWLOCK_PID_MASK   0x7FFFFFFFU
#define DD_RWLOCK_WR(pid)    (DD_RWLOCK_WRITER_BIT | ((uint32_t)(pid) & DD_RWLOCK_PID_MASK))

/* Check if a PID is alive. Returns 1 if alive or unknown, 0 if definitely dead. */
/* Liveness via kill(pid,0). NOTE: cannot detect PID reuse -- if a dead
 * lock-holder's PID is recycled to an unrelated live process before recovery
 * runs, this reports "alive" and that slot's orphaned contribution is not
 * reclaimed until the recycled process exits. Robust detection would require
 * a per-slot process-start-time epoch (a header-layout/version change).
 * Documented under "Crash Safety" in the POD. */
static inline int dd_pid_alive(uint32_t pid) {
    if (pid == 0) return 1; /* no owner recorded, assume alive */
    return !(kill((pid_t)pid, 0) == -1 && errno == ESRCH);
}

/* Force-recover a stale write lock left by a dead process.
 * CAS to OUR pid to hold the lock while fixing shared state, then release.
 * Using our pid (not a bare WRITER_BIT sentinel) means a subsequent
 * recovering process can detect and re-recover if we crash mid-recovery. */
static inline void dd_recover_stale_lock(DdHandle *h, uint32_t observed_rwlock) {
    DdHeader *hdr = h->hdr;
    uint32_t mypid = DD_RWLOCK_WR((uint32_t)getpid());
    if (!__atomic_compare_exchange_n(&hdr->rwlock, &observed_rwlock,
            mypid, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
        return;
    /* We now hold the write lock as mypid.  No additional shared state needs
     * repair here (this module has no seqlock); just release the lock. */
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static const struct timespec dd_lock_timeout = { DD_LOCK_TIMEOUT_SEC, 0 };

/* Process-global fork-generation counter.  Incremented in the pthread_atfork
 * child callback so every open handle detects a fork transition on the next
 * lock call without paying a getpid() syscall on the hot path. */
static uint32_t dd_fork_gen = 1;
static pthread_once_t dd_atfork_once = PTHREAD_ONCE_INIT;
static void dd_on_fork_child(void) {
    __atomic_add_fetch(&dd_fork_gen, 1, __ATOMIC_RELAXED);
}
static void dd_atfork_init(void) {
    pthread_atfork(NULL, NULL, dd_on_fork_child);
}

/* Ensure this process owns a reader slot.  Called from the lock helpers so
 * that fork()'d children pick up their own slot lazily instead of sharing
 * the parent's.  Hot-path is a single relaxed load + compare; only on a
 * fork-generation mismatch do we touch getpid() and scan slots. */
static inline void dd_claim_reader_slot(DdHandle *h) {
    uint32_t cur_gen = __atomic_load_n(&dd_fork_gen, __ATOMIC_RELAXED);
    if (__builtin_expect(cur_gen == h->cached_fork_gen && h->my_slot_idx != UINT32_MAX, 1))
        return;
    /* Cold path -- register the atfork hook once per process, then claim. */
    pthread_once(&dd_atfork_once, dd_atfork_init);
    /* Re-read after pthread_once: dd_on_fork_child may have bumped it. */
    cur_gen = __atomic_load_n(&dd_fork_gen, __ATOMIC_RELAXED);
    uint32_t now_pid = (uint32_t)getpid();
    h->cached_pid = now_pid;
    if (cur_gen != h->cached_fork_gen) h->slotless_held = 0;  /* fork: child holds none of the parent's slotless read locks */
    h->cached_fork_gen = cur_gen;
    h->my_slot_idx = UINT32_MAX;
    uint32_t start = now_pid % DD_READER_SLOTS;
    for (uint32_t i = 0; i < DD_READER_SLOTS; i++) {
        uint32_t s = (start + i) % DD_READER_SLOTS;
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(&h->reader_slots[s].pid,
                &expected, now_pid, 0,
                __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
            /* Zero all mirror fields, not just subcount: a SIGKILL'd
             * predecessor may have left waiters_parked/writers_parked
             * non-zero, and dd_recover_dead_readers won't drain them
             * once we own the slot (the CAS expects the dead PID). */
            __atomic_store_n(&h->reader_slots[s].subcount, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].waiters_parked, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].writers_parked, 0, __ATOMIC_RELAXED);
            h->my_slot_idx = s;
            return;
        }
    }
    /* Table full -- leave my_slot_idx = UINT32_MAX so we silently skip
     * tracking for this handle (lock still works; just no recovery). */
}

/* Atomically subtract `sub` from a counter, capped at 0 (never underflows). */
static inline void dd_atomic_sub_cap(uint32_t *p, uint32_t sub) {
    if (!sub) return;
    uint32_t cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    for (;;) {
        uint32_t want = (cur > sub) ? cur - sub : 0;
        if (__atomic_compare_exchange_n(p, &cur, want,
                1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return;
    }
}

/* Try to claim a dead slot (CAS pid -> 0) and drain its parked-waiter
 * contributions back to the global counters.  A no-op if the slot was stolen
 * by another recoverer or had no waiter contribution to drain.
 *
 * Note: subcount/waiters_parked/writers_parked are NOT zeroed here.
 * Between our CAS and a follow-up store, a new process could claim the
 * slot and start populating these fields -- our stores would clobber its
 * state.  dd_claim_reader_slot zeros all three on every claim, so
 * leaving stale values is harmless. */
static inline void dd_drain_dead_slot(DdHandle *h, uint32_t i, uint32_t pid) {
    DdHeader *hdr = h->hdr;
    uint32_t expected = pid;
    /* ACQ_REL on success: RELEASE publishes pid=0 to other observers;
     * ACQUIRE syncs us with prior writes from the dead process to
     * waiters_parked/writers_parked.  On weakly-ordered archs (aarch64)
     * a plain RELAXED load before the CAS could miss those writes;
     * loading them after the CAS keeps them inside the acquire window. */
    if (!__atomic_compare_exchange_n(&h->reader_slots[i].pid, &expected, 0,
            0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
        return;
    uint32_t wp    = __atomic_load_n(&h->reader_slots[i].waiters_parked, __ATOMIC_RELAXED);
    uint32_t writp = __atomic_load_n(&h->reader_slots[i].writers_parked, __ATOMIC_RELAXED);
    if (wp)    dd_atomic_sub_cap(&hdr->rwlock_waiters, wp);
    if (writp) dd_atomic_sub_cap(&hdr->rwlock_writers_waiting, writp);
}

/* Scan reader slots for dead-process recovery.
 *
 * For each dead PID with non-zero contributions to the shared rwlock,
 * rwlock_waiters, or rwlock_writers_waiting counters, drain its share back
 * out so live processes don't have to wait for the slow per-op timeout
 * decrement to drain it for them.
 *
 * For the main rwlock counter we use the "no live reader holds -> force-
 * reset to 0" trick (precise) because per-process attribution of the
 * subcount is racy across the inc-counter-then-inc-subcount window. */
static inline void dd_recover_dead_readers(DdHandle *h) {
    if (!h->reader_slots) return;
    DdHeader *hdr = h->hdr;
    int any_live_reader = 0;
    int found_dead_reader = 0;

    /* Pass 1: classify slots.  Slots with dead pid and sc == 0 (no rwlock
     * contribution to lose) are wiped immediately to free the slot for
     * future claimants and drain any orphan parked-waiter counters.  Slots
     * with dead pid and sc > 0 are left intact in this pass: if force-
     * reset cannot fire (because a live reader is concurrently present),
     * wiping the dead slot would lose the only record of its orphan
     * rwlock contribution and strand writers permanently once the live
     * reader releases. */
    for (uint32_t i = 0; i < DD_READER_SLOTS; i++) {
        uint32_t pid = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
        if (pid == 0) continue;
        uint32_t sc = __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED);
        if (dd_pid_alive(pid)) {
            if (sc > 0) any_live_reader = 1;
            continue;
        }
        if (sc > 0) { found_dead_reader = 1; continue; }
        dd_drain_dead_slot(h, i, pid);
    }

    /* Pass 2: only if force-reset will fire.  Issue the rwlock force-
     * reset CAS FIRST, while the window since pass 1's last scan is
     * still narrow (a handful of instructions, as in the original
     * single-pass code).  A new reader that started rdlock between
     * pass 1's scan and the CAS will either:
     *   (a) have already CAS'd rwlock from cur to cur+1 -- our CAS then
     *       fails (cur mismatched), recovery yields and a future
     *       cycle retries; or
     *   (b) be still in the subcount-bump phase -- our CAS sees the
     *       stale cur and resets to 0; the new reader's subsequent CAS
     *       rwlock(0 -> 1) succeeds cleanly.
     * Only after the CAS resolves do we wipe the deferred dead slots,
     * keeping that work outside the race-sensitive window. */
    /* A live reader with no slot (table was full) is invisible to the scan
     * above but still holds a +1 in the lock word; never force-reset under it. */
    if (__atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0)
        any_live_reader = 1;
    if (found_dead_reader && !any_live_reader) {
        /* ACQUIRE: a late reader's subcount++ (before its rwlock CAS) is then visible below. */
        uint32_t cur = __atomic_load_n(&hdr->rwlock, __ATOMIC_ACQUIRE);
        int drain_ok = 1;   /* keep dead slots if the reset doesn't fire */
        if (cur > 0 && cur < DD_RWLOCK_WRITER_BIT) {
            /* Re-scan for a live reader (fail-safe: only suppresses a reset). */
            int live_now = __atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0;
            for (uint32_t i = 0; !live_now && i < DD_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p && dd_pid_alive(p) &&
                    __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED) > 0)
                    live_now = 1;
            }
            if (live_now) {
                drain_ok = 0;
            } else if (__atomic_compare_exchange_n(&hdr->rwlock, &cur, 0,
                    0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
                if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
                    syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
            } else {
                drain_ok = 0;   /* rwlock changed under us -- shares may still be live */
            }
        }
        if (drain_ok) {
            for (uint32_t i = 0; i < DD_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p == 0 || dd_pid_alive(p)) continue;
                dd_drain_dead_slot(h, i, p);
            }
        }
    }
}

/* Inspect the lock word after a futex-wait timeout.  If a dead writer
 * holds it, force-recover the lock.  Otherwise drain dead readers' shares
 * of the rwlock/waiter counters.  Called from rdlock and wrlock ETIMEDOUT
 * branches -- identical recovery logic in both. */
static inline void dd_recover_after_timeout(DdHandle *h) {
    DdHeader *hdr = h->hdr;
    uint32_t val = __atomic_load_n(&hdr->rwlock, __ATOMIC_RELAXED);
    if (val >= DD_RWLOCK_WRITER_BIT) {
        uint32_t pid = val & DD_RWLOCK_PID_MASK;
        if (!dd_pid_alive(pid))
            dd_recover_stale_lock(h, val);
    } else {
        dd_recover_dead_readers(h);
    }
}

/* Park/unpark helpers: bump the global waiter counters together with this
 * process's mirrored slot counters so a wrlock-timeout recovery scan can
 * attribute and reverse a dead PID's contribution.  Kept paired to make
 * accidental drift between global and per-slot counts impossible. */
static inline void dd_park_reader(DdHandle *h) {
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
}
static inline void dd_unpark_reader(DdHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
}
static inline void dd_park_writer(DdHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
}
static inline void dd_unpark_writer(DdHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
}

/* Reader accounting: a reader mirrors its +1 in the lock word so dead-reader
 * recovery can see it. A slotted reader uses its slot subcount; a reader that
 * could not claim a slot (table full) uses the global hdr->slotless_readers,
 * so recovery's force-reset never fires out from under it. leave() peels
 * slotless first so a later slot claim cannot misattribute the decrement. */
static inline void dd_reader_enter(DdHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    } else {
        __atomic_add_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
        h->slotless_held++;
    }
}
static inline void dd_reader_leave(DdHandle *h) {
    if (h->slotless_held > 0) {
        h->slotless_held--;
        __atomic_sub_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
    } else if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    }
}

static inline void dd_rwlock_rdlock(DdHandle *h) {
    dd_claim_reader_slot(h);
    DdHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    uint32_t *writers_waiting = &hdr->rwlock_writers_waiting;
    /* Claim subcount BEFORE bumping the shared rwlock counter.  This way
     * a concurrent writer-side recovery scan that sees our PID alive with
     * subcount > 0 will (correctly) defer force-reset, even while we are
     * still spinning trying to win the rwlock CAS.  Without this, a reader
     * killed between rwlock CAS-success and subcount++ would let recovery
     * force-reset rwlock to 0 underneath us, causing a UINT32_MAX wrap on
     * our eventual rdunlock dec. */
    dd_reader_enter(h);
    for (int spin = 0; ; spin++) {
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Write-preferring: when lock is free (cur==0) and writers are
         * waiting, yield to let the writer acquire. When readers are
         * already active (cur>=1), new readers may join freely. */
        if (cur > 0 && cur < DD_RWLOCK_WRITER_BIT) {
            if (__atomic_compare_exchange_n(lock, &cur, cur + 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        } else if (cur == 0 && !__atomic_load_n(writers_waiting, __ATOMIC_RELAXED)) {
            if (__atomic_compare_exchange_n(lock, &cur, 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        }
        if (__builtin_expect(spin < DD_RWLOCK_SPIN_LIMIT, 1)) {
            dd_rwlock_spin_pause();
            continue;
        }
        dd_park_reader(h);
        cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Sleep when write-locked OR when yielding to waiting writers */
        if (cur >= DD_RWLOCK_WRITER_BIT || cur == 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &dd_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                dd_unpark_reader(h);
                dd_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        dd_unpark_reader(h);
        spin = 0;
    }
}

static inline void dd_rwlock_rdunlock(DdHandle *h) {
    DdHeader *hdr = h->hdr;
    /* Release the shared counter BEFORE dropping our subcount so that
     * "any live PID with subcount > 0" is a reliable in-flight indicator
     * for the writer-side recovery scan.  Inverting these would create a
     * window where we still own a unit of rwlock but our slot subcount is
     * 0, letting recovery force-reset rwlock underneath us. */
    uint32_t after = __atomic_sub_fetch(&hdr->rwlock, 1, __ATOMIC_RELEASE);
    dd_reader_leave(h);
    if (after == 0 && __atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static inline void dd_rwlock_wrlock(DdHandle *h) {
    dd_claim_reader_slot(h);  /* refresh cached_pid across fork */
    DdHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    /* Encode PID in the rwlock word itself (0x80000000 | pid) to eliminate
     * any crash window between acquiring the lock and storing the owner. */
    uint32_t mypid = DD_RWLOCK_WR(h->cached_pid);
    for (int spin = 0; ; spin++) {
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(lock, &expected, mypid,
                1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            return;
        if (__builtin_expect(spin < DD_RWLOCK_SPIN_LIMIT, 1)) {
            dd_rwlock_spin_pause();
            continue;
        }
        dd_park_writer(h);
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        if (cur != 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &dd_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                dd_unpark_writer(h);
                dd_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        dd_unpark_writer(h);
        spin = 0;
    }
}

static inline void dd_rwlock_wrunlock(DdHandle *h) {
    DdHeader *hdr = h->hdr;
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

/* ================================================================
 * Layout math + create / open / destroy
 *
 * Layout: Header -> reader_slots[1024] -> neg[num_buckets] -> pos[num_buckets]
 * ================================================================ */

/* Single source of truth for the mmap region layout offsets.
 * Layout: Header -> reader_slots[1024] -> neg[num_buckets] -> pos[num_buckets] */
typedef struct { uint64_t reader_slots, neg, pos, total; } DdLayout;

static inline DdLayout dd_layout_for(uint32_t num_buckets) {
    DdLayout L;
    L.reader_slots = sizeof(DdHeader);
    L.neg          = L.reader_slots + (uint64_t)DD_READER_SLOTS * sizeof(DdReaderSlot);
    L.neg          = (L.neg + 7) & ~(uint64_t)7;   /* 8-byte align the count arrays */
    L.pos          = L.neg + (uint64_t)num_buckets * sizeof(uint64_t);
    L.total        = L.pos + (uint64_t)num_buckets * sizeof(uint64_t);
    return L;
}

static inline uint64_t dd_total_size(uint32_t num_buckets) {
    return dd_layout_for(num_buckets).total;
}

static inline void dd_init_header(void *base, uint32_t num_buckets, double alpha, uint64_t total) {
    DdLayout L = dd_layout_for(num_buckets);
    DdHeader *hdr = (DdHeader *)base;
    memset(base, 0, (size_t)L.total);   /* zero header + reader slots + both count stores */
    double gamma = (1.0 + alpha) / (1.0 - alpha);
    hdr->magic            = DD_MAGIC;
    hdr->version          = DD_VERSION;
    hdr->num_buckets      = num_buckets;
    hdr->bias             = num_buckets / 2;
    hdr->gamma            = gamma;
    hdr->inv_ln_gamma     = 1.0 / log(gamma);
    hdr->alpha            = alpha;
    hdr->total_count      = 0;
    hdr->zero_count       = 0;
    hdr->sum              = 0.0;
    hdr->min_value        = INFINITY;
    hdr->max_value        = -INFINITY;
    hdr->neg_off          = L.neg;
    hdr->pos_off          = L.pos;
    hdr->total_size       = total;
    hdr->reader_slots_off = L.reader_slots;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static inline uint64_t *dd_neg(DdHandle *h) { return (uint64_t *)((char *)h->base + h->neg_off); }
static inline uint64_t *dd_pos(DdHandle *h) { return (uint64_t *)((char *)h->base + h->pos_off); }

/* Layer B trusted bound: the number of bucket counters guaranteed to lie within
 * the real mapping for the store at `off`.  Derived from the process-local
 * mmap_size (fixed at attach, not peer-writable) and the SAME cached offset the
 * accessors use, so a corrupt hdr->num_buckets can never drive a store access
 * outside the mapping.  Equals num_buckets for a valid sketch. */
static inline uint64_t dd_store_max(DdHandle *h, uint64_t off) {
    if (off >= h->mmap_size) return 0;
    return (h->mmap_size - off) / sizeof(uint64_t);
}

static inline DdHandle *dd_setup(void *base, size_t map_size,
                                 const char *path, int backing_fd) {
    DdHeader *hdr = (DdHeader *)base;
    DdHandle *h = (DdHandle *)calloc(1, sizeof(DdHandle));
    if (!h) {
        munmap(base, map_size);
        if (backing_fd >= 0) close(backing_fd);
        return NULL;
    }
    h->hdr          = hdr;
    h->base         = base;
    h->reader_slots = (DdReaderSlot *)((uint8_t *)base + hdr->reader_slots_off);
    /* single validated read of each geometry field; cached so the bound and the
       pointers/arithmetic it feeds stay consistent even under later peer corruption */
    h->neg_off      = hdr->neg_off;
    h->pos_off      = hdr->pos_off;
    h->num_buckets  = hdr->num_buckets;
    h->bias         = hdr->bias;
    /* Derive gamma / inv_ln_gamma from the validated alpha rather than trusting
       the stored copies: validate_header bounds alpha to [DD_MIN_ALPHA,
       DD_MAX_ALPHA], so these are always finite and in range, whereas a corrupt
       stored inv_ln_gamma (NaN/wild) would make the key computation's double->
       int64 cast undefined behaviour. */
    h->gamma        = (1.0 + hdr->alpha) / (1.0 - hdr->alpha);
    h->inv_ln_gamma = 1.0 / log(h->gamma);
    h->mmap_size    = map_size;
    /* Layer B: if the mapping cannot hold `num_buckets` in each store, clamp the
       cached count to what actually fits (pos is the last, tightest region). */
    {
        uint64_t nfit = dd_store_max(h, h->neg_off);
        uint64_t pfit = dd_store_max(h, h->pos_off);
        uint64_t fit  = nfit < pfit ? nfit : pfit;
        if ((uint64_t)h->num_buckets > fit) h->num_buckets = (uint32_t)fit;
    }
    h->path         = path ? strdup(path) : NULL;
    h->backing_fd   = backing_fd;
    h->my_slot_idx  = UINT32_MAX;
    return h;
}

/* Validate a mapped header (shared by dd_create reopen and dd_open_fd). */
static inline int dd_validate_header(const DdHeader *hdr, uint64_t file_size) {
    if (hdr->magic != DD_MAGIC) return 0;
    if (hdr->version != DD_VERSION) return 0;
    if (hdr->num_buckets < DD_MIN_BUCKETS || hdr->num_buckets > DD_MAX_BUCKETS) return 0;
    if (hdr->bias != hdr->num_buckets / 2) return 0;
    if (!(hdr->alpha >= DD_MIN_ALPHA && hdr->alpha <= DD_MAX_ALPHA)) return 0;  /* same bounds as create; rejects NaN */
    if (!(hdr->gamma > 1.0)) return 0;
    if (hdr->total_size != file_size) return 0;
    if (hdr->total_size != dd_total_size(hdr->num_buckets)) return 0;
    DdLayout L = dd_layout_for(hdr->num_buckets);
    if (hdr->reader_slots_off != L.reader_slots) return 0;
    if (hdr->neg_off != L.neg) return 0;
    if (hdr->pos_off != L.pos) return 0;
    return 1;
}

/* validate the requested relative accuracy + bucket count */
static int dd_validate_args(double alpha, uint64_t num_buckets, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    if (!(alpha >= DD_MIN_ALPHA && alpha <= DD_MAX_ALPHA))
        { DD_ERR("alpha (relative accuracy) must be between 1e-6 and 0.5"); return 0; }
    if (num_buckets < DD_MIN_BUCKETS || num_buckets > DD_MAX_BUCKETS)
        { DD_ERR("num_buckets must be between 8 and 2^24"); return 0; }
    return 1;
}

/* Securely obtain a fd for a path-backed segment: create it exclusively
 * (O_CREAT|O_EXCL|O_NOFOLLOW at `mode`, default 0600 = owner-only), or, if it
 * already exists, attach to it (O_RDWR|O_NOFOLLOW, no O_CREAT). O_EXCL blocks a
 * pre-seeded or hard-linked file and O_NOFOLLOW a symlink swap, so a local
 * attacker can no longer redirect or poison the backing store through the path.
 * Cross-user sharing is opt-in via a wider `mode` (e.g. 0660); the caller still
 * validates the file's contents via dd_validate_header. */
static int dd_secure_open(const char *path, mode_t mode, char *errbuf) {
    for (int attempt = 0; attempt < 100; attempt++) {
        int fd = open(path, O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC, mode);
        if (fd >= 0) { (void)fchmod(fd, mode); return fd; }   /* exact mode: umask narrowed the O_EXCL create */
        if (errno != EEXIST) { DD_ERR("create %s: %s", path, strerror(errno)); return -1; }
        fd = open(path, O_RDWR|O_NOFOLLOW|O_CLOEXEC);
        if (fd >= 0) return fd;
        if (errno == ENOENT) continue;   /* creator unlinked between our two opens; retry */
        DD_ERR("open %s: %s", path, strerror(errno));  /* ELOOP => symlink rejected */
        return -1;
    }
    DD_ERR("open %s: create/attach kept racing", path);
    return -1;
}

static DdHandle *dd_create(const char *path, double alpha, uint64_t num_buckets, mode_t mode, char *errbuf) {
    if (!dd_validate_args(alpha, num_buckets, errbuf)) return NULL;

    uint64_t total = dd_total_size((uint32_t)num_buckets);
    int anonymous = (path == NULL);
    int fd = -1;
    size_t map_size;
    void *base;

    if (anonymous) {
        map_size = (size_t)total;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) { DD_ERR("mmap: %s", strerror(errno)); return NULL; }
    } else {
        fd = dd_secure_open(path, mode, errbuf);
        if (fd < 0) return NULL;
        if (flock(fd, LOCK_EX) < 0) { DD_ERR("flock: %s", strerror(errno)); close(fd); return NULL; }
        struct stat st;
        if (fstat(fd, &st) < 0) { DD_ERR("fstat: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        int is_new = (st.st_size == 0);
        if (!is_new && (uint64_t)st.st_size < sizeof(DdHeader)) {
            DD_ERR("%s: file too small (%lld)", path, (long long)st.st_size);
            flock(fd, LOCK_UN); close(fd); return NULL;
        }
        if (is_new && ftruncate(fd, (off_t)total) < 0) {
            DD_ERR("ftruncate: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL;
        }
        map_size = is_new ? (size_t)total : (size_t)st.st_size;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { DD_ERR("mmap: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        if (!is_new) {
            if (!dd_validate_header((DdHeader *)base, (uint64_t)st.st_size)) {
                DD_ERR("invalid DDSketch file"); munmap(base, map_size); flock(fd, LOCK_UN); close(fd); return NULL;
            }
            flock(fd, LOCK_UN); close(fd);
            return dd_setup(base, map_size, path, -1);
        }
    }
    dd_init_header(base, (uint32_t)num_buckets, alpha, total);
    if (fd >= 0) { flock(fd, LOCK_UN); close(fd); }
    return dd_setup(base, map_size, path, -1);
}

static DdHandle *dd_create_memfd(const char *name, double alpha, uint64_t num_buckets, char *errbuf) {
    if (!dd_validate_args(alpha, num_buckets, errbuf)) return NULL;

    uint64_t total = dd_total_size((uint32_t)num_buckets);
    int fd = memfd_create(name ? name : "ddsketch", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) { DD_ERR("memfd_create: %s", strerror(errno)); return NULL; }
    if (ftruncate(fd, (off_t)total) < 0) {
        DD_ERR("ftruncate: %s", strerror(errno)); close(fd); return NULL;
    }
    (void)fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW);
    void *base = mmap(NULL, (size_t)total, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { DD_ERR("mmap: %s", strerror(errno)); close(fd); return NULL; }
    dd_init_header(base, (uint32_t)num_buckets, alpha, total);
    return dd_setup(base, (size_t)total, NULL, fd);
}

static DdHandle *dd_open_fd(int fd, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    struct stat st;
    if (fstat(fd, &st) < 0) { DD_ERR("fstat: %s", strerror(errno)); return NULL; }
    if ((uint64_t)st.st_size < sizeof(DdHeader)) { DD_ERR("too small"); return NULL; }
    size_t ms = (size_t)st.st_size;
    void *base = mmap(NULL, ms, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { DD_ERR("mmap: %s", strerror(errno)); return NULL; }
    if (!dd_validate_header((DdHeader *)base, (uint64_t)st.st_size)) {
        DD_ERR("invalid DDSketch table"); munmap(base, ms); return NULL;
    }
    int myfd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (myfd < 0) { DD_ERR("fcntl: %s", strerror(errno)); munmap(base, ms); return NULL; }
    return dd_setup(base, ms, NULL, myfd);
}

static void dd_destroy(DdHandle *h) {
    if (!h) return;
    /* Release our reader slot on clean teardown (else short-lived-reader churn
     * exhausts the slot table); skip if a lock is still held (subcount>0). */
    if (h->reader_slots && h->my_slot_idx != UINT32_MAX && h->cached_pid &&
        h->cached_fork_gen == __atomic_load_n(&dd_fork_gen, __ATOMIC_RELAXED) &&
        __atomic_load_n(&h->reader_slots[h->my_slot_idx].subcount, __ATOMIC_ACQUIRE) == 0) {
        uint32_t expected = h->cached_pid;
        __atomic_compare_exchange_n(&h->reader_slots[h->my_slot_idx].pid,
                &expected, 0, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
    }
    if (h->backing_fd >= 0) close(h->backing_fd);
    if (h->base) munmap(h->base, h->mmap_size);
    free(h->path);
    free(h);
}

static inline int dd_msync(DdHandle *h) {
    if (!h || !h->base) return 0;
    return msync(h->base, h->mmap_size, MS_SYNC);
}

/* ================================================================
 * DDSketch operations (callers hold the lock)
 *
 * A value v != 0 maps to a logarithmic bucket key = ceil(log_gamma(|v|)); each
 * bucket [gamma^(k-1), gamma^k) collects a count, so a bucket's representative
 * value is within relative error alpha of every value it holds.  Positive and
 * negative magnitudes use separate stores; exact zeros use a dedicated counter.
 * The bucket key is mapped to a fixed centred window slot = key + bias (bias =
 * num_buckets/2); keys past either end collapse into the extreme bucket, which
 * bounds memory while keeping every sketch of the same (alpha, num_buckets)
 * layout mergeable by a plain element-wise sum.
 * ================================================================ */

/* representative value of positive bucket key k: 2*gamma^k/(gamma+1), the value
 * whose relative distance to every point in the bucket is at most alpha */
static inline double dd_value_of_key(DdHandle *h, int64_t key) {
    return 2.0 * pow(h->gamma, (double)key) / (h->gamma + 1.0);
}

/* map a bucket key to a store slot in the fixed centred window, collapsing keys
 * past either end into the extreme bucket. */
static inline uint64_t dd_slot_of_key(DdHandle *h, int64_t key) {
    int64_t s = key + (int64_t)h->bias;
    if (s < 0) return 0;
    if ((uint64_t)s >= (uint64_t)h->num_buckets) return h->num_buckets ? h->num_buckets - 1 : 0;
    return (uint64_t)s;
}

/* insert one finite value (caller holds the write lock; caller has rejected NaN/Inf) */
static void dd_insert_locked(DdHandle *h, double v, uint64_t count) {
    DdHeader *hdr = h->hdr;
    hdr->total_count += count;
    hdr->sum += v * (double)count;
    if (v < hdr->min_value) hdr->min_value = v;
    if (v > hdr->max_value) hdr->max_value = v;
    if (v == 0.0) { hdr->zero_count += count; return; }
    if (h->num_buckets == 0) return;                         /* Layer B: unusable mapping */
    double mag = v > 0 ? v : -v;
    int64_t key = (int64_t)ceil(log(mag) * h->inv_ln_gamma);
    uint64_t slot = dd_slot_of_key(h, key);
    uint64_t *store = (v > 0) ? dd_pos(h) : dd_neg(h);
    store[slot] += count;
}

/* value at 0-based rank `rank` (typically q*(count-1)), walking buckets from
 * most-negative to most-positive; *found set to 1 when a bucket is returned.
 * (caller holds a lock) */
static double dd_value_at_rank(DdHandle *h, double rank, int *found) {
    uint64_t nb = h->num_buckets;
    uint64_t cum = 0;
    *found = 1;
    /* negative store: larger key = more negative value, so walk high slot -> low */
    uint64_t *neg = dd_neg(h);
    for (uint64_t i = nb; i-- > 0; ) {
        cum += neg[i];
        if ((double)cum > rank) {
            int64_t key = (int64_t)i - (int64_t)h->bias;
            return -dd_value_of_key(h, key);
        }
    }
    /* exact zeros */
    cum += h->hdr->zero_count;
    if ((double)cum > rank) return 0.0;
    /* positive store: low slot -> high */
    uint64_t *pos = dd_pos(h);
    for (uint64_t i = 0; i < nb; i++) {
        cum += pos[i];
        if ((double)cum > rank) {
            int64_t key = (int64_t)i - (int64_t)h->bias;
            return dd_value_of_key(h, key);
        }
    }
    *found = 0;                                              /* rank beyond total (empty) */
    return 0.0;
}

/* merge another sketch's stores into this one (caller guarantees equal geometry).
 * src_neg/src_pos are snapshots of length src_nb. */
static void dd_merge_locked(DdHandle *dst, const uint64_t *src_neg, const uint64_t *src_pos,
                            uint64_t src_nb, uint64_t src_total, uint64_t src_zero,
                            double src_sum, double src_min, double src_max) {
    uint64_t nb = dst->num_buckets;
    if (nb > src_nb) nb = src_nb;                            /* Layer B: clamp to both buffers */
    uint64_t *neg = dd_neg(dst), *pos = dd_pos(dst);
    for (uint64_t i = 0; i < nb; i++) { neg[i] += src_neg[i]; pos[i] += src_pos[i]; }
    dst->hdr->total_count += src_total;
    dst->hdr->zero_count  += src_zero;
    dst->hdr->sum         += src_sum;
    if (src_min < dst->hdr->min_value) dst->hdr->min_value = src_min;
    if (src_max > dst->hdr->max_value) dst->hdr->max_value = src_max;
}

/* reset to an empty sketch (caller holds the write lock) */
static inline void dd_clear_locked(DdHandle *h) {
    DdHeader *hdr = h->hdr;
    uint64_t nb = h->num_buckets;
    uint64_t nfit = dd_store_max(h, h->neg_off);            /* Layer B: clamp memset to the mapping */
    uint64_t pfit = dd_store_max(h, h->pos_off);
    memset(dd_neg(h), 0, (size_t)((nb < nfit ? nb : nfit) * sizeof(uint64_t)));
    memset(dd_pos(h), 0, (size_t)((nb < pfit ? nb : pfit) * sizeof(uint64_t)));
    hdr->total_count = 0;
    hdr->zero_count  = 0;
    hdr->sum         = 0.0;
    hdr->min_value   = INFINITY;
    hdr->max_value   = -INFINITY;
}

#endif /* DD_H */
