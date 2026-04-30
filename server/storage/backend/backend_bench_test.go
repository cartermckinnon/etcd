// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package backend_test

import (
	crand "crypto/rand"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"

	"go.etcd.io/etcd/server/v3/storage/backend"
	betesting "go.etcd.io/etcd/server/v3/storage/backend/testing"
	"go.etcd.io/etcd/server/v3/storage/schema"
)

func BenchmarkBackendPut(b *testing.B) {
	backend, _ := betesting.NewTmpBackend(b, 100*time.Millisecond, 10000)
	defer betesting.Close(b, backend)

	// prepare keys
	keys := make([][]byte, b.N)
	for i := 0; i < b.N; i++ {
		keys[i] = make([]byte, 64)
		_, err := crand.Read(keys[i])
		require.NoError(b, err)
	}
	value := make([]byte, 128)
	_, err := crand.Read(value)
	require.NoError(b, err)

	batchTx := backend.BatchTx()

	batchTx.Lock()
	batchTx.UnsafeCreateBucket(schema.Test)
	batchTx.Unlock()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		batchTx.Lock()
		batchTx.UnsafePut(schema.Test, keys[i], value)
		batchTx.Unlock()
	}
}

// BenchmarkBackendDefrag measures Defrag() latency across a full cross-product
// of db sizes and fragmentation levels. Each iteration sets up a fresh backend,
// loads keys, deletes a fraction of them to induce fragmentation, and times
// only the Defrag() call. db_bytes_before / frag_bytes_before metrics report
// the on-disk size and free-page bytes captured just before Defrag.
//
// Value sizes are sampled log-uniformly in [valueSizeMin, valueSizeMax] per
// key to mimic real K8s workloads (configmaps → pod specs → large CRDs) and
// to exercise bbolt's overflow-page allocator: large values need contiguous
// runs of free pages, so a freelist with many small holes can fail to satisfy
// a put and force file extension even when total free bytes look ample.
//
// The keyCount column targets total file sizes from ~256 MB to ~32 GB
// (assuming ~52 KB/key on-disk, measured empirically). Actual file sizes vary
// — read them from the db_bytes_before metric.
func BenchmarkBackendDefrag(b *testing.B) {
	const (
		keySize      = 128
		valueSizeMin = 512
		valueSizeMax = 256 * 1024
	)
	keyCounts := []int{
		5_000,   // ~256 MB
		10_000,  // ~512 MB
		20_000,  // ~1 GB
		40_000,  // ~2 GB
		80_000,  // ~4 GB
		160_000, // ~8 GB
		320_000, // ~16 GB
		640_000, // ~32 GB
	}
	deleteRatios := []float64{0.0, 0.25, 0.5, 0.75, 0.9}

	for _, keyCount := range keyCounts {
		for _, deleteRatio := range deleteRatios {
			name := fmt.Sprintf("keys=%d/keysz=%d/valsz=%s/frag=%.2f",
				keyCount, keySize, formatValSizeRange(valueSizeMin, valueSizeMax), deleteRatio)
			b.Run(name, func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					b.StopTimer()
					bcknd := newFragmentedBackend(b, keyCount, keySize, valueSizeMin, valueSizeMax, deleteRatio)

					b.ReportMetric(float64(bcknd.Size()), "db_bytes_before")
					b.ReportMetric(float64(bcknd.Size()-bcknd.SizeInUse()), "frag_bytes_before")

					b.StartTimer()
					err := bcknd.Defrag()
					b.StopTimer()

					require.NoError(b, err)
					betesting.Close(b, bcknd)
				}
			})
		}
	}
}

func formatValSizeRange(min, max int) string {
	if min == max {
		return humanBytes(min)
	}
	return humanBytes(min) + "-" + humanBytes(max)
}

func humanBytes(n int) string {
	switch {
	case n >= 1024*1024:
		return fmt.Sprintf("%dMB", n/(1024*1024))
	case n >= 1024:
		return fmt.Sprintf("%dKB", n/1024)
	default:
		return fmt.Sprintf("%dB", n)
	}
}

// newFragmentedBackend creates a temp backend, loads keyCount keys with
// log-uniform value sizes in [valueSizeMin, valueSizeMax], then deletes a
// deleteRatio fraction of them in batched commits to induce fragmentation.
// The caller is responsible for closing the backend.
func newFragmentedBackend(b *testing.B, keyCount, keySize, valueSizeMin, valueSizeMax int, deleteRatio float64) backend.Backend {
	b.Helper()
	require.GreaterOrEqual(b, keySize, 8, "keySize must be >= 8 to hold a uint64 index")
	require.GreaterOrEqual(b, valueSizeMax, valueSizeMin, "valueSizeMax must be >= valueSizeMin")

	// A long BatchInterval prevents the periodic committer from racing with our
	// chunked commits; we drive commits explicitly via ForceCommit.
	bcknd, path := betesting.NewTmpBackend(b, time.Hour, 10_000)

	// One pre-allocated random buffer; per-insert we slice it to a sampled
	// length. bolt copies into its pages so sharing the underlying bytes is
	// safe and keeps memory bounded regardless of keyCount.
	valBuf := make([]byte, valueSizeMax)
	_, err := crand.Read(valBuf)
	require.NoError(b, err)

	// Seeded RNG for reproducible value-size distributions across runs.
	rng := rand.New(rand.NewSource(1))
	logMin := math.Log(float64(valueSizeMin))
	logMax := math.Log(float64(valueSizeMax))
	pickValSize := func() int {
		if valueSizeMin == valueSizeMax {
			return valueSizeMin
		}
		n := int(math.Exp(logMin + rng.Float64()*(logMax-logMin)))
		if n < valueSizeMin {
			n = valueSizeMin
		} else if n > valueSizeMax {
			n = valueSizeMax
		}
		return n
	}

	tx := bcknd.BatchTx()
	tx.Lock()
	tx.UnsafeCreateBucket(schema.Test)
	tx.Unlock()

	const chunk = 10_000
	for start := 0; start < keyCount; start += chunk {
		end := start + chunk
		if end > keyCount {
			end = keyCount
		}
		tx.Lock()
		for j := start; j < end; j++ {
			key := make([]byte, keySize)
			binary.BigEndian.PutUint64(key, uint64(j))
			tx.UnsafePut(schema.Test, key, valBuf[:pickValSize()])
		}
		tx.Unlock()
		bcknd.ForceCommit()
	}

	if deleteRatio > 0 {
		// Distribute deletions evenly across the key space so every leaf page
		// receives roughly the same fraction of deletes. Stride math
		// (every Nth key) breaks down for ratios > 0.5 since stride=1 deletes
		// everything; instead compute target deleteCount and step in float space.
		deleteCount := int(deleteRatio * float64(keyCount))
		if deleteCount > keyCount {
			deleteCount = keyCount
		}
		step := float64(keyCount) / float64(deleteCount)

		// Issue deletes in batches separated by ForceCommit so pages freed by
		// one batch's COW leaf rewrites can be reused by the next batch's
		// writes. Doing all deletions in a single commit forces bbolt to
		// allocate fresh pages for every rewritten leaf, inflating both file
		// size and reported fragmentation. Real etcd compactions delete in
		// bounded batches (typically ~1000 keys per --experimental-compaction-
		// batch-limit), which is what this models.
		const deleteBatchSize = 1000
		for batchStart := 0; batchStart < deleteCount; batchStart += deleteBatchSize {
			batchEnd := batchStart + deleteBatchSize
			if batchEnd > deleteCount {
				batchEnd = deleteCount
			}
			tx.Lock()
			for i := batchStart; i < batchEnd; i++ {
				j := int(float64(i) * step)
				key := make([]byte, keySize)
				binary.BigEndian.PutUint64(key, uint64(j))
				tx.UnsafeDelete(schema.Test, key)
			}
			tx.Unlock()
			bcknd.ForceCommit()
		}
	}

	// Close and reopen so pages bbolt currently has on the pending list
	// migrate to the free list. Bolt persists pending+free as a unified
	// freelist on close; on reopen they're all loaded as free. This better
	// mirrors a long-running etcd whose recently-deleted pages have had
	// time to settle (rather than the rapid create/delete pattern of this
	// setup, which leaves nearly everything pending).
	require.NoError(b, bcknd.Close())
	bcknd = backend.NewDefaultBackend(zaptest.NewLogger(b), path)

	db := backend.DbFromBackendForTest(bcknd)
	stats := db.Stats()
	size := bcknd.Size()
	freeBytes := int64(stats.FreePageN) * int64(db.Info().PageSize)
	b.Logf("setup: size=%d freePages=%d frag=%d (%.1f%%)",
		size, stats.FreePageN, freeBytes,
		100*float64(freeBytes)/float64(size))

	return bcknd
}
